options(warn = 1)
Sys.setenv(JAX_PLATFORMS = Sys.getenv("JAX_PLATFORMS", unset = "cpu"))

repo_root <- normalizePath(file.path(getwd()), winslash = "/", mustWork = TRUE)
out_dir <- file.path(repo_root, "Tmp", "tb_prior_forward_sweep")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

log_file <- file.path(out_dir, "run.log")
log_msg <- function(...) {
  msg <- sprintf(...)
  line <- sprintf("[%s] %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg)
  cat(line, "\n")
  cat(line, "\n", file = log_file, append = TRUE)
}

pkgload::load_all(repo_root, quiet = TRUE)
source(file.path(repo_root, "tests", "testthat", "helper-sim-fit.R"), local = globalenv())

conda_env <- Sys.getenv(
  "NDM_SOFTWARE_CONDA_ENV",
  unset = Sys.getenv("NDM_CONDA_ENV", unset = "jax_cpu")
)
log_msg("Using conda env: %s", conda_env)
backend <- ndm_initialize_backend(conda_env = conda_env, float_type = "32", import_tensorflow = TRUE)
log_msg("JAX default backend: %s", backend$default_backend)

safe_num <- function(x) {
  out <- suppressWarnings(as.numeric(x))
  out[!is.finite(out)] <- NA_real_
  out
}

as_matrix <- function(x, label = "value") {
  ndm_test_as_numeric_matrix(x, label = label)
}

broadcast_like <- function(x, target, label = "value") {
  ndm_test_broadcast_like(x, target, label = label)
}

capture_train_prediction <- function(model, batch, seed = 1L) {
  runtime_env <- model$env
  getpred_saveat_info <- runtime_env$ndm_runtime_normalize_getpred_saveat_info(runtime_env$GetPredSaveAtInfo_default)
  seed_matrix <- runtime_env$jax$random$split(
    runtime_env$jax$random$PRNGKey(as.integer(seed)),
    as.integer(runtime_env$nBatch)
  )
  seed_matrix <- runtime_env$ndm_runtime_data_to_device(seed_matrix)
  runtime_env$GetPred_train_jit(
    runtime_env$ModelList,
    runtime_env$batch2package(batch),
    runtime_env$state,
    runtime_env$PriorList,
    runtime_env$PolicyList,
    getpred_saveat_info,
    seed_matrix
  )[[1L]]
}

collect_time_varying <- function(pred, spec) {
  if (length(spec$time_varying_terms) == 0L) {
    return(list())
  }
  times <- as_matrix(pred$ODEParamsSampList[["diff_eq_sol_ts"]], label = "diff_eq_sol_ts")
  trajectories <- list()
  keys <- names(pred$ODEParamsSampList)
  for (term in spec$time_varying_terms) {
    flat_key <- paste0("diff_eq_sol_ys.", term)
    if (flat_key %in% keys) {
      trajectories[[term]] <- as_matrix(pred$ODEParamsSampList[[flat_key]], label = flat_key)
    } else if (identical(term, "c_t") && identical(spec$preset, "tb_k")) {
      x1 <- broadcast_like(pred$ODEParamsSampList[["x1_samp"]], times, label = "x1_samp")
      x2 <- broadcast_like(pred$ODEParamsSampList[["x2_samp"]], times, label = "x2_samp")
      trajectories[[term]] <- x1 * (pmax(1.0, times) ^ x2)
    } else if (identical(term, "c_t") && identical(spec$preset, "tb_l")) {
      x1 <- broadcast_like(pred$ODEParamsSampList[["x1_samp"]], times, label = "x1_samp")
      x2 <- broadcast_like(pred$ODEParamsSampList[["x2_samp"]], times, label = "x2_samp")
      x3 <- broadcast_like(pred$ODEParamsSampList[["x3_samp"]], times, label = "x3_samp")
      trajectories[[term]] <- x1 * (x2 + exp((-1 * x3) * times))
    }
  }
  trajectories
}

make_spec_cases <- function() {
  list(
    list(label = "tb_a", spec = ndm_model_spec(preset = "tb_a", model_type = "NeuralODE")),
    list(label = "tb_b_n2", spec = ndm_model_spec(preset = "tb_b", model_type = "NeuralODE", family_args = list(n = 2L))),
    list(label = "tb_b_n3", spec = ndm_model_spec(preset = "tb_b", model_type = "NeuralODE", family_args = list(n = 3L))),
    list(label = "tb_b_n5", spec = ndm_model_spec(preset = "tb_b", model_type = "NeuralODE", family_args = list(n = 5L))),
    list(label = "tb_c", spec = ndm_model_spec(preset = "tb_c", model_type = "NeuralODE")),
    list(label = "tb_d", spec = ndm_model_spec(preset = "tb_d", model_type = "NeuralODE")),
    list(label = "tb_e", spec = ndm_model_spec(preset = "tb_e", model_type = "NeuralODE")),
    list(label = "tb_f", spec = ndm_model_spec(preset = "tb_f", model_type = "NeuralODE")),
    list(label = "tb_g", spec = ndm_model_spec(preset = "tb_g", model_type = "NeuralODE")),
    list(label = "tb_h", spec = ndm_model_spec(preset = "tb_h", model_type = "NeuralODE")),
    list(label = "tb_i", spec = ndm_model_spec(preset = "tb_i", model_type = "NeuralODE")),
    list(label = "tb_j_n2", spec = ndm_model_spec(preset = "tb_j", model_type = "NeuralODE", family_args = list(n = 2L))),
    list(label = "tb_j_n3", spec = ndm_model_spec(preset = "tb_j", model_type = "NeuralODE", family_args = list(n = 3L))),
    list(label = "tb_j_n5", spec = ndm_model_spec(preset = "tb_j", model_type = "NeuralODE", family_args = list(n = 5L))),
    list(label = "tb_k", spec = ndm_model_spec(preset = "tb_k", model_type = "NeuralODE")),
    list(label = "tb_l", spec = ndm_model_spec(preset = "tb_l", model_type = "NeuralODE"))
  )
}

build_untrained_model <- function(spec, label, case_seed = 2026L, model_dims = 16L, n_batch = 4L, lookahead = 3L) {
  set.seed(as.integer(case_seed))
  work_dir <- file.path(out_dir, paste0("work_", label))
  dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(work_dir, "SavedModels", "FromSim"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(work_dir, "SavedResults", "Sim"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(work_dir, "results"), recursive = TRUE, showWarnings = FALSE)

  old_wd <- setwd(work_dir)
  on.exit(setwd(old_wd), add = TRUE)

  sim_entry <- list(
    BaseID = 1L,
    betat_init = 0.35,
    invbetat_sd = 0.02,
    i0_a = 5,
    i0_b = 0.2,
    r0_a = 2,
    r0_b = 0.2,
    gamma = 0.25,
    sigma = 0.2,
    xi = 0.02,
    c_endogeneous = 0,
    policy_effectiveness = 0.1,
    policy_responsiveness = 1.0,
    policy_decay = 0.05,
    beta_baseline = 0.35,
    beta_restore_rate = 0.1,
    measurement_noise = 0.05,
    nSamplesTrain = n_batch,
    ResaveThisTFRecord = 1L
  )
  sim_grid <- do.call(
    rbind,
    list(
      as.data.frame(sim_entry, stringsAsFactors = FALSE),
      as.data.frame(utils::modifyList(sim_entry, list(BaseID = 2L, c_endogeneous = 0.1)), stringsAsFactors = FALSE)
    )
  )

  config <- ndm_create_config(
    model_type = "NeuralODE",
    backbone = "transformer",
    float_type = "32",
    force_to_gpu = FALSE,
    resave_tfrecords = TRUE,
    neuralode_optim_dt0 = 1e0,
    neuralode_optim_controller = "constant"
  )
  runtime_env <- ndm_prepare_runtime(
    config = config,
    runtime_env = ndm_new_runtime_env(parent = globalenv()),
    runtime_globals = list(
      SimMode = TRUE,
      GLOBAL_ODE_NPOP = 10000L,
      nBatch = as.integer(n_batch),
      covariateType = "sqrt",
      simCovariates = c(
        XPred_c_sqrt = "inc_case_per_capita_sqrt",
        XPred_h_sqrt = "inc_hosp_per_capita_sqrt",
        XPred_d_sqrt = "inc_death_per_capita_sqrt"
      ),
      rollCompute_window = 52L,
      AnalysisName = paste0("TBPriorForward_", label),
      AnalysisDate = Sys.Date(),
      COMMAND_ARG_INPUT = "tb_prior_forward_sweep",
      TfRecordDir = file.path(work_dir, "tfrecords"),
      nSamples_max = as.integer(n_batch),
      nSamplesTrain = as.integer(n_batch),
      SimEntry = sim_entry,
      SimGrid = sim_grid,
      SEED_ = as.integer(case_seed),
      LEARNING_RATE_MAX = 3e-3,
      nSGD_DefiningLRSeq = 1L,
      nSGD_model = 1L,
      nSGD_pretrain = 0L,
      nSGD_posttrain = 1L,
      nCheckpoints = 0L,
      ModelDims = as.integer(model_dims),
      ModelDepth = 1L,
      nOutcomes = 1L,
      nPlaces = 1L,
      af = 1L,
      HolderFolder = file.path(work_dir, "results"),
      OUTER_ITERATION = 1L,
      endAppend = TRUE,
      EnableKVCaching = TRUE,
      AttentionHeadDim = as.integer(model_dims),
      paddingMethod = "left",
      nBatch_SimGridGen = as.integer(max(4L, n_batch)),
      nMonteEval = 1L,
      SimScalingOuterLoops = 1L,
      SimScalingInnerLoops = 2L
    )
  )

  n_times_past <- 8L
  n_times_total <- n_times_past + as.integer(lookahead)
  n_time_steps_sim <- (n_times_past + as.integer(lookahead)) * 2L
  ndm_set_runtime_globals(
    runtime_env,
    list(
      nTimesPast = n_times_past,
      nTimesLookahead = as.integer(lookahead),
      nTimesTotal = n_times_total,
      VI_TotalTimesInLikelihood = as.integer(lookahead),
      nTimesInLikelihood = as.integer(lookahead),
      NTimeSteps_SIM = n_time_steps_sim,
      nTimesLookValidationInference = as.integer(lookahead),
      MaxSteps = as.integer(1e4),
      VI_SaveAt_ODE_sim = runtime_env$diffrax$SaveAt(
        ts = runtime_env$jnp$array(0L:(n_time_steps_sim - 1L))
      ),
      VI_SaveAt_ODE_optim = runtime_env$diffrax$SaveAt(
        ts = runtime_env$jnp$array(0L:(as.integer(lookahead) - 1L))
      ),
      VI_diff_eq_solver_dgp = runtime_env$diffrax$Tsit5(),
      dt0_init_dgp = 1e-3,
      stepsize_controller_dgp = runtime_env$diffrax$PIDController(
        rtol = 1e-5,
        atol = 1e-6
      ),
      diffraxInterpolator = runtime_env$diffrax$LinearInterpolation
    )
  )

  suppressWarnings(ndm_prepare_data(runtime_env = runtime_env, generator = "sim"))
  snapshot <- ndm_test_copy_env(runtime_env)
  on.exit(ndm_test_restore_env(snapshot), add = TRUE)

  batch <- if (exists("batch_l_cal", envir = runtime_env, inherits = FALSE)) {
    get("batch_l_cal", envir = runtime_env, inherits = FALSE)
  } else if (exists("batch_l", envir = runtime_env, inherits = FALSE)) {
    get("batch_l", envir = runtime_env, inherits = FALSE)
  } else if (exists("TFDataset_train", envir = runtime_env, inherits = FALSE) &&
             exists("TFConst2JAXArray", envir = runtime_env, inherits = FALSE)) {
    runtime_env$TFConst2JAXArray(
      reticulate::iter_next(reticulate::as_iterator(get("TFDataset_train", envir = runtime_env, inherits = FALSE)))
    )
  } else {
    stop("Unable to locate a simulation batch after `ndm_prepare_data()`.", call. = FALSE)
  }
  model <- suppressWarnings(ndm_build_model(
    runtime_env = runtime_env,
    model_type = "NeuralODE",
    model_spec = spec,
    backbone = "transformer"
  ))

  list(model = model, batch = batch, work_dir = work_dir)
}

summarize_prediction <- function(pred, spec, label, pass_name, seed) {
  state_keys <- paste0("diff_eq_sol_ys.", spec$state_terms)
  state_mats <- lapply(
    setNames(state_keys, spec$state_terms),
    function(key) as_matrix(pred$ODEParamsSampList[[key]], label = key)
  )
  state_dim <- dim(state_mats[[1L]])
  state_cube <- array(
    NA_real_,
    dim = c(state_dim[[1L]], state_dim[[2L]], length(state_mats)),
    dimnames = list(NULL, NULL, spec$state_terms)
  )
  for (i in seq_along(state_mats)) {
    state_cube[, , i] <- state_mats[[i]]
  }
  totals <- apply(state_cube, c(1L, 2L), sum)
  initial_totals <- matrix(totals[1L, ], nrow = nrow(totals), ncol = ncol(totals), byrow = TRUE)
  proportions <- sweep(state_cube, c(1L, 2L), pmax(totals, 1e-12), "/")
  init_props <- apply(proportions[1L, , , drop = FALSE], 3L, mean, na.rm = TRUE)
  final_props <- apply(proportions[nrow(proportions), , , drop = FALSE], 3L, mean, na.rm = TRUE)
  first_step_delta <- if (dim(proportions)[[1L]] >= 2L) {
    apply(proportions[2L, , , drop = FALSE] - proportions[1L, , , drop = FALSE], 3L, mean, na.rm = TRUE)
  } else {
    rep(NA_real_, length(spec$state_terms))
  }
  names(first_step_delta) <- names(init_props)

  state_rows <- data.frame(
    label = label,
    pass = pass_name,
    seed = as.integer(seed),
    state = names(init_props),
    init_prop_mean = as.numeric(init_props),
    final_prop_mean = as.numeric(final_props[names(init_props)]),
    first_step_delta_prop_mean = as.numeric(first_step_delta[names(init_props)]),
    stringsAsFactors = FALSE
  )

  param_rows <- do.call(
    rbind,
    lapply(names(spec$execution_spec$parameters), function(param) {
      key <- paste0(param, "_samp")
      value <- if (key %in% names(pred$ODEParamsSampList)) {
        safe_num(as_matrix(pred$ODEParamsSampList[[key]], label = key))
      } else {
        NA_real_
      }
      pdef <- spec$execution_spec$parameters[[param]]
      data.frame(
        label = label,
        pass = pass_name,
        seed = as.integer(seed),
        parameter = param,
        transformation = pdef$transformation,
        prior_mean = safe_num(pdef$prior_mean)[[1L]],
        prior_sd = safe_num(pdef$prior_sd)[[1L]],
        sample_mean = mean(value, na.rm = TRUE),
        sample_min = min(value, na.rm = TRUE),
        sample_max = max(value, na.rm = TRUE),
        stringsAsFactors = FALSE
      )
    })
  )

  pred_mu <- safe_num(as_matrix(pred$y_mu, label = "y_mu"))
  pred_sigma <- safe_num(as_matrix(pred$y_sigma, label = "y_sigma"))
  time_varying <- collect_time_varying(pred, spec)
  tv_rows <- if (length(time_varying) == 0L) {
    data.frame()
  } else {
    do.call(rbind, lapply(names(time_varying), function(term) {
      vals <- safe_num(time_varying[[term]])
      data.frame(
        label = label,
        pass = pass_name,
        seed = as.integer(seed),
        term = term,
        min = min(vals, na.rm = TRUE),
        mean = mean(vals, na.rm = TRUE),
        max = max(vals, na.rm = TRUE),
        stringsAsFactors = FALSE
      )
    }))
  }

  summary_row <- data.frame(
    label = label,
    pass = pass_name,
    seed = as.integer(seed),
    preset = spec$preset,
    family = spec$family,
    family_args = paste(names(spec$family_args), unlist(spec$family_args), sep = "=", collapse = ","),
    n_states = length(spec$state_terms),
    init_state_terms = paste(spec$init_state_terms, collapse = ","),
    parameter_terms = paste(spec$parameter_terms, collapse = ","),
    min_state = min(state_cube, na.rm = TRUE),
    max_state = max(state_cube, na.rm = TRUE),
    max_rel_mass_drift = max(abs(totals - initial_totals) / pmax(abs(initial_totals), 1e-12), na.rm = TRUE),
    i_init_prop = if ("i_l" %in% names(init_props)) init_props[["i_l"]] else NA_real_,
    i_final_prop = if ("i_l" %in% names(final_props)) final_props[["i_l"]] else NA_real_,
    i_first_step_delta = if ("i_l" %in% names(first_step_delta)) first_step_delta[["i_l"]] else NA_real_,
    pred_mu_mean = mean(pred_mu, na.rm = TRUE),
    pred_mu_min = min(pred_mu, na.rm = TRUE),
    pred_mu_max = max(pred_mu, na.rm = TRUE),
    pred_sigma_mean = mean(pred_sigma, na.rm = TRUE),
    stringsAsFactors = FALSE
  )

  list(summary = summary_row, states = state_rows, params = param_rows, time_varying = tv_rows)
}

prior_rows_for_spec <- function(spec, label) {
  params <- do.call(rbind, lapply(names(spec$execution_spec$parameters), function(param) {
    pdef <- spec$execution_spec$parameters[[param]]
    data.frame(
      label = label,
      type = "parameter",
      term = param,
      transformation = pdef$transformation,
      prior_mean = safe_num(pdef$prior_mean)[[1L]],
      prior_sd = safe_num(pdef$prior_sd)[[1L]],
      stringsAsFactors = FALSE
    )
  }))
  states <- do.call(rbind, lapply(names(spec$execution_spec$state_init_priors), function(term) {
    pdef <- spec$execution_spec$state_init_priors[[term]]
    data.frame(
      label = label,
      type = "init_state_tex_prior",
      term = term,
      transformation = "Identity",
      prior_mean = safe_num(pdef$prior_mean)[[1L]],
      prior_sd = safe_num(pdef$prior_sd)[[1L]],
      stringsAsFactors = FALSE
    )
  }))
  rbind(params, states)
}

cases <- make_spec_cases()
all_summary <- list()
all_states <- list()
all_params <- list()
all_tv <- list()
all_priors <- list()
errors <- list()

for (case in cases) {
  label <- case$label
  spec <- case$spec
  log_msg("Starting %s", label)
  all_priors[[label]] <- prior_rows_for_spec(spec, label)
  result <- tryCatch({
    built <- build_untrained_model(spec, label = label, case_seed = 2026L, model_dims = 16L, n_batch = 4L, lookahead = 3L)
    env <- built$model$env

    pass_specs <- list(
      list(name = "deterministic_loc", seed = 123L, test_without_sampling = TRUE),
      list(name = "stochastic_seed_123", seed = 123L, test_without_sampling = FALSE),
      list(name = "stochastic_seed_456", seed = 456L, test_without_sampling = FALSE)
    )
    for (pass in pass_specs) {
      assign("testWithoutSampling", isTRUE(pass$test_without_sampling), envir = env)
      pred <- capture_train_prediction(built$model, built$batch, seed = pass$seed)
      summary <- summarize_prediction(pred, spec, label, pass$name, pass$seed)
      key <- paste(label, pass$name, sep = "__")
      all_summary[[key]] <- summary$summary
      all_states[[key]] <- summary$states
      all_params[[key]] <- summary$params
      all_tv[[key]] <- summary$time_varying
    }
    TRUE
  }, error = function(e) {
    errors[[label]] <<- data.frame(
      label = label,
      error = conditionMessage(e),
      stringsAsFactors = FALSE
    )
    log_msg("ERROR in %s: %s", label, conditionMessage(e))
    FALSE
  })
  if (isTRUE(result)) {
    log_msg("Finished %s", label)
  }
  saveRDS(
    list(
      summary = all_summary,
      states = all_states,
      params = all_params,
      time_varying = all_tv,
      priors = all_priors,
      errors = errors
    ),
    file.path(out_dir, "partial_results.rds")
  )
}

bind_rows_or_empty <- function(x) {
  x <- x[vapply(x, function(z) is.data.frame(z) && nrow(z) > 0L, logical(1))]
  if (length(x) == 0L) {
    return(data.frame())
  }
  do.call(rbind, x)
}

summary_df <- bind_rows_or_empty(all_summary)
states_df <- bind_rows_or_empty(all_states)
params_df <- bind_rows_or_empty(all_params)
tv_df <- bind_rows_or_empty(all_tv)
priors_df <- bind_rows_or_empty(all_priors)
errors_df <- bind_rows_or_empty(errors)

utils::write.csv(summary_df, file.path(out_dir, "tb_forward_summary.csv"), row.names = FALSE)
utils::write.csv(states_df, file.path(out_dir, "tb_state_trajectories_summary.csv"), row.names = FALSE)
utils::write.csv(params_df, file.path(out_dir, "tb_parameter_samples_summary.csv"), row.names = FALSE)
utils::write.csv(tv_df, file.path(out_dir, "tb_time_varying_summary.csv"), row.names = FALSE)
utils::write.csv(priors_df, file.path(out_dir, "tb_prior_definitions.csv"), row.names = FALSE)
utils::write.csv(errors_df, file.path(out_dir, "tb_forward_errors.csv"), row.names = FALSE)

det <- summary_df[summary_df$pass == "deterministic_loc", , drop = FALSE]
state_det <- states_df[states_df$pass == "deterministic_loc", , drop = FALSE]

md <- c(
  "# TB Prior Forward Sweep",
  "",
  sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  sprintf("Repo: `%s`", repo_root),
  sprintf("Cases attempted: %s", length(cases)),
  sprintf("Cases with errors: %s", if (nrow(errors_df) == 0L) 0L else nrow(errors_df)),
  "",
  "## Deterministic initialized pass",
  "",
  paste(capture.output(print(det[, c(
    "label", "n_states", "i_init_prop", "i_final_prop",
    "i_first_step_delta", "min_state", "max_rel_mass_drift",
    "pred_mu_mean"
  )], row.names = FALSE)), collapse = "\n"),
  "",
  "## Deterministic initialized state proportions",
  "",
  paste(capture.output(print(state_det[, c(
    "label", "state", "init_prop_mean", "final_prop_mean",
    "first_step_delta_prop_mean"
  )], row.names = FALSE)), collapse = "\n"),
  "",
  "## Notes",
  "",
  "- `deterministic_loc` sets runtime `testWithoutSampling = TRUE`, so draws use distribution locations.",
  "- `stochastic_seed_123` and `stochastic_seed_456` use the default sampling path from initialized variational distributions.",
  "- State proportions are normalized by total solved disease-state mass per time/batch column.",
  "- Structured TB state-init TeX priors are listed separately in `tb_prior_definitions.csv`."
)
writeLines(md, file.path(out_dir, "tb_forward_report.md"), useBytes = TRUE)
log_msg("Wrote outputs to %s", out_dir)
