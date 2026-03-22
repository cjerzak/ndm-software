ndm_test_copy_env <- function(src, dest = globalenv()) {
  src_names <- ls(src, all.names = TRUE)
  overwritten <- intersect(src_names, ls(dest, all.names = TRUE))
  overwritten_values <- if (length(overwritten) > 0L) {
    mget(overwritten, envir = dest, inherits = FALSE)
  } else {
    list()
  }

  list2env(mget(src_names, envir = src, inherits = FALSE), envir = dest)

  list(
    all_names = src_names,
    overwritten = overwritten,
    overwritten_values = overwritten_values
  )
}

ndm_test_restore_env <- function(snapshot, dest = globalenv()) {
  created <- setdiff(snapshot$all_names, snapshot$overwritten)
  if (length(created) > 0L) {
    rm(list = created, envir = dest)
  }
  if (length(snapshot$overwritten) > 0L) {
    list2env(snapshot$overwritten_values, envir = dest)
  }
  invisible(dest)
}

ndm_test_fit_sim_case <- function(model_type,
                                  endogeneity,
                                  n_sgd = NULL,
                                  n_checkpoints = 0L,
                                  enable_kv_cache = TRUE,
                                  model_dims = 32L,
                                  attention_head_dim = 64L,
                                  attention_kv_heads = NULL,
                                  before_train = NULL,
                                  after_train_define = NULL,
                                  expect_train_error = FALSE,
                                  return_details = FALSE) {
  if (is.null(n_sgd)) {
    n_sgd <- if (identical(model_type, "NeuralODE")) 3L else 2L
  }
  learning_rate_max <- if (identical(model_type, "NeuralODE")) 3e-3 else 2e-3
  case_seed <- 1000L +
    (match(model_type, c("DecoderOnly", "NeuralODE")) - 1L) * 100L +
    as.integer(round(endogeneity * 100))
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  old_seed <- if (had_seed) get(".Random.seed", envir = .GlobalEnv, inherits = FALSE) else NULL
  set.seed(case_seed)
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(list = ".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)

  work_dir <- tempfile("ndm-sim-fit-")
  dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
  old_wd <- setwd(work_dir)
  on.exit(setwd(old_wd), add = TRUE)

  dir.create(file.path(work_dir, "SavedModels", "FromSim"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(work_dir, "SavedResults", "Sim"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(work_dir, "results"), recursive = TRUE, showWarnings = FALSE)

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
    c_endogeneous = endogeneity,
    policy_effectiveness = 0.1,
    policy_responsiveness = 1.0,
    policy_decay = 0.05,
    beta_baseline = 0.35,
    beta_restore_rate = 0.1,
    measurement_noise = 0.05,
    nSamplesTrain = 4L,
    ResaveThisTFRecord = 1L
  )
  sim_grid <- do.call(
    rbind,
    list(
      as.data.frame(sim_entry, stringsAsFactors = FALSE),
      as.data.frame(
        utils::modifyList(
          sim_entry,
          list(BaseID = 2L, c_endogeneous = endogeneity + 0.1)
        ),
        stringsAsFactors = FALSE
      )
    )
  )

  config <- ndm_create_config(
    model_type = model_type,
    backbone = "transformer",
    float_type = "32",
    force_to_gpu = FALSE,
    resave_tfrecords = TRUE
  )

  runtime_env <- ndm_prepare_runtime(
    config = config,
    runtime_env = ndm_new_runtime_env(parent = globalenv()),
    runtime_globals = list(
      SimMode = TRUE,
      GLOBAL_ODE_NPOP = 10000L,
      nBatch = 4L,
      covariateType = "sqrt",
      simCovariates = c(
        XPred_c_sqrt = "inc_case_per_capita_sqrt",
        XPred_h_sqrt = "inc_hosp_per_capita_sqrt",
        XPred_d_sqrt = "inc_death_per_capita_sqrt"
      ),
      rollCompute_window = 52L,
      AnalysisName = sprintf("TestPandemicSim_%s", model_type),
      AnalysisDate = Sys.Date(),
      COMMAND_ARG_INPUT = "test",
      TfRecordDir = file.path(work_dir, "tfrecords"),
      nSamples_max = 4L,
      nSamplesTrain = 4L,
      SimEntry = sim_entry,
      SimGrid = sim_grid,
      SEED_ = 101L,
      LEARNING_RATE_MAX = learning_rate_max,
      nSGD_DefiningLRSeq = n_sgd,
      nSGD_model = n_sgd,
      nSGD_pretrain = 0L,
      nSGD_posttrain = n_sgd,
      nCheckpoints = n_checkpoints,
      ModelDims = as.integer(model_dims),
      ModelDepth = 1L,
      nOutcomes = 1L,
      nPlaces = 1L,
      af = 1L,
      HolderFolder = file.path(work_dir, "results"),
      OUTER_ITERATION = 1L,
      endAppend = TRUE,
      EnableKVCaching = enable_kv_cache,
      AttentionHeadDim = as.integer(attention_head_dim),
      AttentionKVHeads = if (is.null(attention_kv_heads)) NULL else as.integer(attention_kv_heads),
      paddingMethod = "left",
      nBatch_SimGridGen = 8L,
      nMonteEval = 1L,
      SimScalingOuterLoops = 1L,
      SimScalingInnerLoops = 2L
    )
  )

  n_times_lookahead <- 4L
  n_times_past <- 8L
  n_times_total <- n_times_past + n_times_lookahead
  n_time_steps_sim <- (n_times_past + n_times_lookahead) * 2L
  ndm_set_runtime_globals(
    runtime_env,
    list(
      nTimesPast = n_times_past,
      nTimesLookahead = n_times_lookahead,
      nTimesTotal = n_times_total,
      VI_TotalTimesInLikelihood = n_times_lookahead,
      nTimesInLikelihood = n_times_lookahead,
      NTimeSteps_SIM = n_time_steps_sim,
      nTimesLookValidationInference = 4L,
      MaxSteps = as.integer(1e4),
      VI_SaveAt_ODE_sim = runtime_env$diffrax$SaveAt(
        ts = runtime_env$jnp$array(0L:(n_time_steps_sim - 1L))
      ),
      VI_SaveAt_ODE_optim = runtime_env$diffrax$SaveAt(
        ts = runtime_env$jnp$array(0L:(n_times_lookahead - 1L))
      ),
      VI_diff_eq_solver_optim = runtime_env$diffrax$Tsit5(),
      VI_diff_eq_solver_dgp = runtime_env$diffrax$Tsit5(),
      dt0_init_dgp = 1e-3,
      stepsize_controller_dgp = runtime_env$diffrax$PIDController(
        rtol = 1e-5,
        atol = 1e-6
      ),
      dt0_init_optim = 1e0,
      stepsize_controller_optim = runtime_env$diffrax$ConstantStepSize(),
      diffraxInterpolator = runtime_env$diffrax$LinearInterpolation
    )
  )

  suppressWarnings(
    ndm_prepare_data(
      runtime_env = runtime_env,
      generator = "sim"
    )
  )

  snapshot <- ndm_test_copy_env(runtime_env)
  on.exit(ndm_test_restore_env(snapshot), add = TRUE)

  model_spec <- ndm_model_spec(
    preset = "seirs_dynamic_beta",
    model_type = model_type
  )
  model <- suppressWarnings(
    ndm_build_model(
      runtime_env = runtime_env,
      model_type = model_type,
      model_spec = model_spec,
      backbone = "transformer"
    )
  )
  if (is.function(before_train)) {
    before_train(runtime_env = runtime_env, model = model)
  }
  if (is.function(after_train_define)) {
    train_define_result <- suppressWarnings(
      try(
        ndm_train(
          model,
          run_define = TRUE,
          run_loop = FALSE
        ),
        silent = TRUE
      )
    )
    if (inherits(train_define_result, "try-error")) {
      stop(attr(train_define_result, "condition"))
    }
    after_train_define(runtime_env = runtime_env, model = model)
    trained <- suppressWarnings(
      try(
        ndm_train(
          model,
          run_define = FALSE,
          run_loop = TRUE
        ),
        silent = TRUE
      )
    )
  } else {
    trained <- suppressWarnings(
      try(
        ndm_train(
          model,
          run_define = TRUE,
          run_loop = TRUE
        ),
        silent = TRUE
      )
    )
  }
  if (isTRUE(expect_train_error)) {
    if (!inherits(trained, "try-error")) {
      stop("Expected ndm_train() to fail for this simulation test case.")
    }
    if (isTRUE(return_details)) {
      return(list(
        train_error = trained,
        model = model,
        runtime_env = runtime_env,
        work_dir = work_dir,
        holder_folder = file.path(work_dir, "results")
      ))
    }
    return(invisible(trained))
  }
  if (inherits(trained, "try-error")) {
    stop(attr(trained, "condition"))
  }

  losses <- as.numeric(trained$env$in_loss_vec[seq_len(n_sgd)])
  summary <- data.frame(
    model_type = model_type,
    spec_preset = model_spec$preset,
    endogeneity = endogeneity,
    first_loss = losses[[1]],
    last_loss = losses[[length(losses)]],
    loss_delta = losses[[1]] - losses[[length(losses)]],
    stringsAsFactors = FALSE
  )
  if (isTRUE(return_details)) {
    batch <- if (exists("batch_l_cal", envir = runtime_env, inherits = FALSE)) {
      get("batch_l_cal", envir = runtime_env, inherits = FALSE)
    } else if (exists("TFDataset_train", envir = runtime_env, inherits = FALSE) &&
               exists("TFConst2JAXArray", envir = runtime_env, inherits = FALSE)) {
      runtime_env$TFConst2JAXArray(
        reticulate::iter_next(reticulate::as_iterator(get("TFDataset_train", envir = runtime_env, inherits = FALSE)))
      )
    } else {
      NULL
    }
    return(list(
      summary = summary,
      model = model,
      trained = trained,
      runtime_env = runtime_env,
      batch = batch,
      work_dir = work_dir,
      holder_folder = file.path(work_dir, "results")
    ))
  }
  summary
}

ndm_skip_if_no_sim_backend <- function() {
  skip_on_cran()
  ndm_require_backend_test_stack("simulation fit tests")
}

test_that("simulated pandemic fits improve across model families and endogeneity levels", {
  ndm_skip_if_no_sim_backend()

  cases <- data.frame(
    model_type = c("DecoderOnly", "DecoderOnly", "NeuralODE", "NeuralODE"),
    endogeneity = c(0.0, 0.6, 0.0, 0.6),
    stringsAsFactors = FALSE
  )

  results <- do.call(
    rbind,
    lapply(seq_len(nrow(cases)), function(i) {
      ndm_test_fit_sim_case(
        model_type = cases$model_type[[i]],
        endogeneity = cases$endogeneity[[i]]
      )
    })
  )
  results_info <- paste(capture.output(print(results)), collapse = "\n")
  neural_ode_results <- results[results$model_type == "NeuralODE", , drop = FALSE]

  expect_true(all(is.finite(results$first_loss)))
  expect_true(all(is.finite(results$last_loss)))
  expect_true(all(results$spec_preset == "seirs_dynamic_beta"), info = results_info)
  expect_true(all(results$loss_delta >= 0), info = results_info)
  expect_true(mean(results$loss_delta) > 1e-3, info = results_info)
  expect_true(min(neural_ode_results$loss_delta) > 1e-3, info = results_info)
})

test_that("safe log diagnostics plots tolerate non-finite loss histories", {
  pdf_file <- tempfile(fileext = ".pdf")
  grDevices::pdf(pdf_file)
  on.exit({
    try(grDevices::dev.off(), silent = TRUE)
  }, add = TRUE)

  expect_invisible(ndm:::.ndm_plot_log_series_safe(c(Inf, NA_real_, 0, -1), main = "Loss"))
  expect_invisible(ndm:::.ndm_plot_log_series_safe(c(NA_real_, NaN, Inf), main = "Gradients"))
})

test_that("decoder cache and non-cache predictions agree in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    return_details = TRUE
  )

  details$runtime_env$EnableKVCaching <- FALSE
  pred_no_cache <- ndm_predict(
    details$trained,
    batch = details$batch,
    seed = 123L,
    update_state = FALSE
  )
  details$runtime_env$EnableKVCaching <- TRUE
  pred_cache <- ndm_predict(
    details$trained,
    batch = details$batch,
    seed = 123L,
    update_state = FALSE
  )

  pred_no_cache_mu <- details$runtime_env$np$asanyarray(pred_no_cache$y_mu)
  pred_cache_mu <- details$runtime_env$np$asanyarray(pred_cache$y_mu)

  expect_equal(dim(pred_cache_mu), dim(pred_no_cache_mu))
  expect_lt(max(abs(pred_cache_mu - pred_no_cache_mu)), 1e-4)
})

test_that("decoder transformer resolves GQA topology and KV cache shapes in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  # Prime the session with a smaller decoder model so the GQA assertions below
  # verify that helper bindings are scoped to the current runtime.
  invisible(ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    model_dims = 32L,
    return_details = TRUE
  ))

  details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    model_dims = 256L,
    return_details = TRUE
  )

  env <- details$runtime_env
  expect_equal(as.integer(env$TransformerHeads), 4L)
  expect_equal(as.integer(env$TransformerKVHeads), 1L)
  expect_equal(as.integer(env$TransformerHeadDim), 64L)
  expect_equal(as.integer(env$TransformerKVGroupSize), 4L)

  layer <- details$model$env$ModelList$TSList$TSBackbone$d1$Multihead
  wq_shape <- as.integer(unlist(reticulate::py_to_r(layer$W_q$shape)))
  wk_shape <- as.integer(unlist(reticulate::py_to_r(layer$W_k$shape)))
  wv_shape <- as.integer(unlist(reticulate::py_to_r(layer$W_v$shape)))
  wo_shape <- as.integer(unlist(reticulate::py_to_r(layer$W_o$shape)))
  expect_equal(wq_shape, c(256L, 256L))
  expect_equal(wk_shape, c(256L, 64L))
  expect_equal(wv_shape, c(256L, 64L))
  expect_equal(wo_shape, c(256L, 256L))

  cache <- env$kv_cache_allocate(
    max_len = 11L,
    num_layers = 1L,
    num_kv_heads = env$TransformerKVHeads,
    head_dim = env$TransformerHeadDim,
    dtype = details$batch$XPred$dtype
  )
  cache_shape <- as.integer(unlist(reticulate::py_to_r(cache$d1$k$shape)))
  expect_equal(cache_shape, c(11L, 1L, 64L))

  topology_cases <- list(
    "8" = c(1L, 1L, 8L),
    "32" = c(1L, 1L, 32L),
    "64" = c(1L, 1L, 64L),
    "96" = c(2L, 1L, 48L),
    "128" = c(2L, 1L, 64L),
    "192" = c(3L, 1L, 64L),
    "256" = c(4L, 1L, 64L)
  )

  for (case_name in names(topology_cases)) {
    resolved <- env$resolve_transformer_topology(
      model_dims = as.integer(case_name),
      target_head_dim = 64L,
      kv_heads_override = NULL
    )
    expect_equal(
      c(
        as.integer(resolved$num_query_heads),
        as.integer(resolved$num_kv_heads),
        as.integer(resolved$head_dim)
      ),
      topology_cases[[case_name]]
    )
    expect_equal(as.integer(resolved$head_dim) %% 2L, 0L)
    expect_equal(as.integer(resolved$num_query_heads) %% as.integer(resolved$num_kv_heads), 0L)
  }
})

test_that("checkpointed sim runs emit analytics artifacts in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    n_checkpoints = 1L,
    return_details = TRUE
  )

  csv_files <- list.files(details$holder_folder, pattern = "^res.*\\.csv$", full.names = TRUE)
  rdata_files <- list.files(details$holder_folder, pattern = "^res.*\\.Rdata$", full.names = TRUE)

  expect_length(csv_files, 1L)
  expect_length(rdata_files, 1L)

  metrics <- as.data.frame(data.table::fread(csv_files[[1]]))
  metric_names <- names(metrics)

  expect_true("PolicySkill1" %in% metric_names)
  expect_true("RSSBaselineTime1" %in% metric_names)
  expect_true(is.finite(as.numeric(details$trained$env$Skill8SanityCheck)))
})

test_that("checkpointed NeuralODE sim runs emit structural analytics artifacts in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  details <- ndm_test_fit_sim_case(
    model_type = "NeuralODE",
    endogeneity = 0.0,
    n_sgd = 1L,
    n_checkpoints = 1L,
    return_details = TRUE
  )

  csv_files <- list.files(details$holder_folder, pattern = "^res.*\\.csv$", full.names = TRUE)
  rdata_files <- list.files(details$holder_folder, pattern = "^res.*\\.Rdata$", full.names = TRUE)

  expect_length(csv_files, 1L)
  expect_length(rdata_files, 1L)

  metrics <- as.data.frame(data.table::fread(csv_files[[1]]))
  metric_names <- names(metrics)

  expect_true("AbsDiff_init" %in% metric_names)
  expect_true("AbsDiff_gamma" %in% metric_names)
  expect_true("PolicySkill1" %in% metric_names)
  expect_true(is.finite(as.numeric(metrics$AbsDiff_init[[1L]])))
  expect_true(is.finite(as.numeric(metrics$AbsDiff_gamma[[1L]])))
  expect_true(is.finite(as.numeric(details$trained$env$Skill8SanityCheck)))
})

test_that("non-finite sim training fails fast and writes a debug artifact", {
  ndm_skip_if_no_sim_backend()

  details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    after_train_define = function(runtime_env, model) {
      runtime_env$train_step_compiled <- function(ModelList,
                                                  batch_pkg,
                                                  y_true,
                                                  y_mask,
                                                  iteration,
                                                  state,
                                                  PriorList,
                                                  PolicyList,
                                                  GetPredSaveAtInfo,
                                                  seed,
                                                  opt_state) {
        list(
          loss = runtime_env$jnp$array(Inf),
          state = state,
          grad_norm = runtime_env$jnp$array(Inf),
          model = ModelList,
          opt_state = opt_state
        )
      }
    },
    expect_train_error = TRUE,
    return_details = TRUE
  )

  expect_match(
    ndm:::.ndm_condition_message(details$train_error),
    "Non-finite training state"
  )

  artifact_files <- list.files(
    details$holder_folder,
    pattern = "^nonfinite_outer1_i1\\.rds$",
    full.names = TRUE
  )
  expect_length(artifact_files, 1L)

  report <- readRDS(artifact_files[[1L]])
  expect_identical(report$outer_iteration, 1L)
  expect_identical(report$base_id, 1L)
  expect_true(is.infinite(report$loss))
  expect_true("batch_YTrue_out" %in% names(report$tensor_summaries))
  expect_true("prediction_center_param" %in% names(report$tensor_summaries))
})

test_that("GetPred save-at horizon remains a static host integer in sim runtimes", {
  ndm_skip_if_no_sim_backend()

  details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    return_details = TRUE
  )

  saveat_info <- details$runtime_env$GetPredSaveAtInfo_default
  expect_true(is.list(saveat_info))
  expect_false("python.builtin.object" %in% class(saveat_info[[1L]]))
  expect_identical(as.integer(saveat_info[[1L]]), 4L)
})
