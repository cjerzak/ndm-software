# Shared simulation-fit harness for backend-heavy transformer regression tests.

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
                                  case_seed = NULL,
                                  n_checkpoints = 0L,
                                  n_times_lookahead = 4L,
                                  enable_kv_cache = TRUE,
                                  model_dims = 32L,
                                  attention_head_dim = 64L,
                                  attention_kv_heads = NULL,
                                  runtime_globals = NULL,
                                  before_train = NULL,
                                  after_train_define = NULL,
                                  expect_train_error = FALSE,
                                  return_details = FALSE) {
  n_times_lookahead <- as.integer(n_times_lookahead)
  if (is.null(n_sgd)) {
    n_sgd <- if (identical(model_type, "NeuralODE")) 3L else 2L
  }
  learning_rate_max <- if (identical(model_type, "NeuralODE")) 3e-3 else 2e-3
  if (is.null(case_seed)) {
    case_seed <- 1000L +
      (match(model_type, c("DecoderOnly", "NeuralODE")) - 1L) * 100L +
      as.integer(round(endogeneity * 100))
  }
  case_seed <- as.integer(case_seed)
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

  runtime_defaults <- list(
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
  if (is.list(runtime_globals) && length(runtime_globals) > 0L) {
    runtime_defaults <- utils::modifyList(runtime_defaults, runtime_globals)
  }

  runtime_env <- ndm_prepare_runtime(
    config = config,
    runtime_env = ndm_new_runtime_env(parent = globalenv()),
    runtime_globals = runtime_defaults
  )

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
      nTimesLookValidationInference = n_times_lookahead,
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

ndm_test_read_single_sim_metrics <- function(holder_folder) {
  csv_files <- list.files(holder_folder, pattern = "^res.*\\.csv$", full.names = TRUE)
  if (length(csv_files) == 0L) {
    stop(
      sprintf(
        "Expected at least one sim analytics CSV in `%s`, found 0.",
        holder_folder,
        length(csv_files)
      )
    )
  }
  if (length(csv_files) == 1L) {
    return(as.data.frame(data.table::fread(csv_files[[1L]])))
  }

  csv_names <- basename(csv_files)
  match_info <- regexec("^res.*_i([0-9]+)\\.csv$", csv_names)
  match_parts <- regmatches(csv_names, match_info)
  valid_match <- lengths(match_parts) == 2L
  if (!all(valid_match)) {
    stop(
      sprintf(
        paste(
          "Expected sim analytics CSV names to match `res*_i<iter>.csv`",
          "when multiple files are present in `%s`.",
          "Offending files: %s"
        ),
        holder_folder,
        paste(csv_names[!valid_match], collapse = ", ")
      )
    )
  }

  iterations <- as.integer(vapply(match_parts, `[[`, character(1), 2L))
  target_idx <- which(iterations == max(iterations))
  if (length(target_idx) != 1L) {
    stop(
      sprintf(
        paste(
          "Expected a unique final sim analytics CSV in `%s`,",
          "but found %s files at iteration %s: %s"
        ),
        holder_folder,
        length(target_idx),
        max(iterations),
        paste(csv_names[target_idx], collapse = ", ")
      )
    )
  }

  as.data.frame(data.table::fread(csv_files[[target_idx]]))
}

ndm_test_week10_relative_accuracy <- function(metrics, eps = 1e-3, target_week = 10L) {
  pred_col <- paste0("RSSPredTime", target_week)
  baseline_col <- paste0("RSSBaselineTime", target_week)
  skill_col <- paste0("SkillTime", target_week)
  required_cols <- c(pred_col, baseline_col)
  missing_cols <- required_cols[!required_cols %in% names(metrics)]
  if (length(missing_cols) > 0L) {
    stop(
      sprintf(
        "Missing required week-%s analytics columns: %s",
        target_week,
        paste(missing_cols, collapse = ", ")
      )
    )
  }

  rss_pred <- as.numeric(metrics[[pred_col]][[1L]])
  rss_baseline <- as.numeric(metrics[[baseline_col]][[1L]])
  skill <- if (skill_col %in% names(metrics)) {
    as.numeric(metrics[[skill_col]][[1L]])
  } else {
    NA_real_
  }
  list(
    rss_pred = rss_pred,
    rss_baseline = rss_baseline,
    skill = skill,
    relative_accuracy = (eps + sqrt(rss_baseline)) / (eps + sqrt(rss_pred))
  )
}

ndm_test_collect_week10_relative_accuracy_pair <- function(endogeneity = 0.0,
                                                           shared_seed = 1010L,
                                                           n_times_lookahead = 10L,
                                                           n_sgd = 1L,
                                                           model_dims = 32L) {
  shared_seed <- as.integer(shared_seed)
  common_args <- list(
    endogeneity = endogeneity,
    case_seed = shared_seed,
    n_checkpoints = 1L,
    n_times_lookahead = n_times_lookahead,
    n_sgd = as.integer(n_sgd),
    model_dims = as.integer(model_dims),
    runtime_globals = list(SEED_ = shared_seed),
    return_details = TRUE
  )

  decoder_details <- do.call(
    ndm_test_fit_sim_case,
    c(list(model_type = "DecoderOnly"), common_args)
  )
  neuralode_details <- do.call(
    ndm_test_fit_sim_case,
    c(list(model_type = "NeuralODE"), common_args)
  )

  decoder_metrics <- ndm_test_read_single_sim_metrics(decoder_details$holder_folder)
  neuralode_metrics <- ndm_test_read_single_sim_metrics(neuralode_details$holder_folder)
  decoder_week10 <- ndm_test_week10_relative_accuracy(decoder_metrics)
  neuralode_week10 <- ndm_test_week10_relative_accuracy(neuralode_metrics)
  list(
    decoder_relative_accuracy_10 = decoder_week10$relative_accuracy,
    neuralode_relative_accuracy_10 = neuralode_week10$relative_accuracy,
    decoder_skill_10 = decoder_week10$skill,
    neuralode_skill_10 = neuralode_week10$skill,
    info = sprintf(
      paste(
        "week10 decoder rel_acc=%.6f",
        "week10 neuralode rel_acc=%.6f",
        "week10 decoder skill=%.6f",
        "week10 neuralode skill=%.6f",
        "shared_seed=%s",
        "lookahead=%s",
        "model_dims=%s",
        "n_sgd=%s",
        sep = "; "
      ),
      decoder_week10$relative_accuracy,
      neuralode_week10$relative_accuracy,
      decoder_week10$skill,
      neuralode_week10$skill,
      shared_seed,
      as.integer(n_times_lookahead),
      as.integer(model_dims),
      as.integer(n_sgd)
    )
  )
}

ndm_skip_if_no_sim_backend <- function() {
  skip_on_cran()
  ndm_require_backend_test_stack("simulation fit tests")
}
