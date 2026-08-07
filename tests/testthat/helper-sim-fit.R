# Shared simulation-fit harness for backend-heavy transformer regression tests.

.ndm_test_sim_fit_tfrecord_cache <- new.env(parent = emptyenv())

ndm_test_sim_fit_scaling_batches <- function(scaling_outer_loops,
                                             scaling_inner_loops,
                                             n_batch_sim_grid_gen) {
  legacy_counts <- suppressWarnings(as.numeric(c(
    scaling_outer_loops,
    scaling_inner_loops,
    n_batch_sim_grid_gen
  )))
  if (length(legacy_counts) != 3L ||
      any(!is.finite(legacy_counts)) ||
      any(legacy_counts <= 0)) {
    stop("Simulation scaling loop and batch counts must be positive finite numbers.",
         call. = FALSE)
  }

  scaling_examples <- prod(legacy_counts)
  # ndmdatasets generates 16 examples for each canonical scaling batch.
  as.integer(max(1, ceiling(scaling_examples / 16L)))
}

ndm_test_sim_max_time_index <- function(dataset_spec) {
  required_fields <- c(
    "n_time_steps",
    "context_length",
    "lookahead",
    "forward_shift_c"
  )
  missing_fields <- setdiff(required_fields, names(dataset_spec))
  if (length(missing_fields) > 0L) {
    stop(
      "Simulation dataset spec is missing time-index field(s): ",
      paste(missing_fields, collapse = ", "),
      ".",
      call. = FALSE
    )
  }
  values <- suppressWarnings(as.numeric(unlist(
    dataset_spec[required_fields],
    use.names = FALSE
  )))
  if (length(values) != length(required_fields) ||
      any(!is.finite(values)) ||
      any(values < 0) ||
      any(values != floor(values))) {
    stop(
      "Simulation dataset time-index fields must be non-negative integers.",
      call. = FALSE
    )
  }
  max_time_index <- values[[1L]] - sum(values[2L:4L])
  if (max_time_index < 0) {
    stop(
      "Simulation dataset spec leaves no valid initial time index.",
      call. = FALSE
    )
  }
  as.integer(max_time_index)
}

ndm_test_clear_tensorflow_global_seed <- function(runtime_env) {
  if (is.null(runtime_env$tf) ||
      is.null(runtime_env$tf$random) ||
      is.null(runtime_env$tf$random$set_seed)) {
    stop(
      "Simulation fixtures require TensorFlow random-seed control.",
      call. = FALSE
    )
  }
  # Dataset.shuffle() already receives the fixture's explicit operation seed.
  # Clear TensorFlow's process-global seed so an earlier Analysis2 run cannot
  # silently combine a second seed into this test's batch order.
  runtime_env$tf$random$set_seed(NULL)
  invisible(runtime_env)
}

ndm_test_sim_fit_grid <- function(sim_entry,
                                  n_samples_train,
                                  n_times_lookahead,
                                  n_times_past = 8L,
                                  scaling_outer_loops = 1L,
                                  scaling_inner_loops = 2L,
                                  n_batch_sim_grid_gen = 8L) {
  grid <- ndm_test_make_sim_run_grid()[1L, , drop = FALSE]
  sim_entry <- if (is.data.frame(sim_entry)) {
    as.list(sim_entry[1L, , drop = FALSE])
  } else {
    as.list(sim_entry)
  }

  for (field in names(sim_entry)) {
    value <- sim_entry[[field]]
    if (length(value) == 1L && !is.list(value)) {
      grid[[field]] <- value
    }
  }

  grid$BaseID <- 1L
  grid$ContextLength <- as.integer(n_times_past)
  grid$lookahead <- as.integer(n_times_lookahead)
  grid$n_time_steps <- as.integer(2L * (n_times_past + n_times_lookahead))
  grid$n_inference_batches <- 1L
  grid$scaling_batches <- ndm_test_sim_fit_scaling_batches(
    scaling_outer_loops = scaling_outer_loops,
    scaling_inner_loops = scaling_inner_loops,
    n_batch_sim_grid_gen = n_batch_sim_grid_gen
  )
  grid$nSamplesTrain <- as.integer(n_samples_train)
  grid$ModelType <- "DecoderOnly"
  grid$ModelDepth <- 1L
  grid$ModelDims <- 32L
  grid$floatType <- "32"
  grid$ResaveThisTFRecord <- 1L
  rownames(grid) <- NULL
  grid
}

ndm_test_sim_fit_canonical_artifacts <- function(api,
                                                 grid,
                                                 dataset_spec,
                                                 ndmdatasets_pkg = "ndmdatasets") {
  producer <- ndm_test_tfrecord_producer()
  cache_key <- digest::digest(
    list(
      schema = "canonical-sim-v3",
      dataset_spec = dataset_spec,
      n_samples_train = as.integer(grid$nSamplesTrain[[1L]]),
      producer = producer,
      ndmdatasets_version = as.character(utils::packageVersion(ndmdatasets_pkg))
    ),
    algo = "sha256"
  )
  # Keep artifacts available to every fit in this R process. The PID-scoped
  # root lives under R's session tempdir and is removed with that session.
  cache_root <- file.path(
    tempdir(),
    sprintf("ndm-sim-fit-tfrecord-cache-%s", Sys.getpid()),
    cache_key
  )
  dir.create(cache_root, recursive = TRUE, showWarnings = FALSE)
  tfrecord_dir <- file.path(cache_root, "tfrecords")
  paths <- api$analysis2_tfrecord_paths(tfrecord_dir, 1L)
  required_files <- c(
    paths$train_file,
    paths$inference_file,
    paste0(paths$train_file, ".manifest.rds"),
    paste0(paths$inference_file, ".manifest.rds")
  )

  bootstrap <- function(overwrite) {
    suppressMessages(
      ndm_bootstrap_sim_tfrecords(
        project_root = cache_root,
        analysis_name = "SimFitCanonicalCache",
        grid = grid,
        base_ids = 1L,
        tfrecord_dir = tfrecord_dir,
        producer = producer,
        overwrite = isTRUE(overwrite),
        dry_run = FALSE
      )
    )
  }
  if (!all(file.exists(required_files))) {
    bootstrap(overwrite = any(file.exists(required_files)))
  }

  validate <- function() {
    api$analysis2_validate_canonical_tfrecord_pair(
      ndmdatasets_pkg = ndmdatasets_pkg,
      paths = paths,
      schema_kind = "sim",
      dataset_spec = dataset_spec,
      n_train = as.integer(grid$nSamplesTrain[[1L]]),
      n_inference = as.integer(dataset_spec$n_inference_batches) * 128L,
      verify_checksum = TRUE
    )
  }
  validated <- try(validate(), silent = TRUE)
  if (inherits(validated, "try-error")) {
    bootstrap(overwrite = TRUE)
    validated <- validate()
  }
  if (!identical(validated$train$producer, producer) ||
      !identical(validated$inference$producer, producer)) {
    stop("Canonical simulation-fit cache has unexpected producer metadata.", call. = FALSE)
  }

  cache_entry <- get0(
    cache_key,
    envir = .ndm_test_sim_fit_tfrecord_cache,
    inherits = FALSE,
    ifnotfound = NULL
  )
  scaler_sha256 <- validated$train$scaler_sha256
  if (is.null(cache_entry) ||
      !identical(cache_entry$scaler_sha256, scaler_sha256)) {
    scaler <- validated$train$scaler
    if (is.null(scaler)) {
      stop("Canonical simulation-fit manifest did not contain a scaler.", call. = FALSE)
    }
    cache_entry <- list(
      scaler_sha256 = scaler_sha256,
      outcome_sd = api$analysis2_sim_outcome_sd(
        ndmdatasets_pkg = ndmdatasets_pkg,
        dataset_spec = dataset_spec,
        scaler = scaler
      )
    )
    assign(cache_key, cache_entry, envir = .ndm_test_sim_fit_tfrecord_cache)
  }

  list(
    key = cache_key,
    paths = paths,
    tfrecord_dir = tfrecord_dir,
    validated = validated,
    scaler = validated$train$scaler,
    outcome_sd = cache_entry$outcome_sd
  )
}

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

ndm_test_merge_runtime_lists <- function(base, override = NULL) {
  if (!is.list(override) || length(override) == 0L) {
    return(base)
  }
  utils::modifyList(base, override)
}

ndm_test_fit_sim_case <- function(model_type,
                                  endogeneity,
                                  n_sgd = NULL,
                                  case_seed = NULL,
                                  n_checkpoints = 0L,
                                  n_times_lookahead = 4L,
                                  enable_kv_cache = TRUE,
                                  enable_kv_cache_training = TRUE,
                                  model_dims = 32L,
                                  attention_head_dim = 64L,
                                  attention_kv_heads = NULL,
                                  config_overrides = NULL,
                                  runtime_globals = NULL,
                                  runtime_globals_after_setup = NULL,
                                  before_train = NULL,
                                  after_train_define = NULL,
                                  expect_train_error = FALSE,
                                  return_details = FALSE,
                                  model_spec = NULL) {
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

  config_defaults <- list(
    model_type = model_type,
    backbone = "transformer",
    float_type = "32",
    compute_backend = "cpu",
    resave_tfrecords = FALSE,
    enable_kv_cache = enable_kv_cache,
    enable_kv_cache_training = enable_kv_cache_training
  )
  if (is.list(config_overrides) && length(config_overrides) > 0L) {
    config_defaults <- utils::modifyList(config_defaults, config_overrides)
  }
  config <- do.call(ndm_create_config, config_defaults)

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
    EnableKVCachingTraining = enable_kv_cache_training,
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

  n_times_past <- 8L
  n_samples_train <- as.integer(runtime_defaults$nSamplesTrain)
  effective_sim_entry <- utils::modifyList(sim_entry, as.list(runtime_defaults$SimEntry))
  effective_sim_entry$nSamplesTrain <- n_samples_train
  effective_sim_entry$BaseID <- 1L
  effective_sim_entry$ResaveThisTFRecord <- 1L
  artifact_grid <- ndm_test_sim_fit_grid(
    sim_entry = effective_sim_entry,
    n_samples_train = n_samples_train,
    n_times_lookahead = n_times_lookahead,
    n_times_past = n_times_past,
    scaling_outer_loops = runtime_defaults$SimScalingOuterLoops,
    scaling_inner_loops = runtime_defaults$SimScalingInnerLoops,
    n_batch_sim_grid_gen = runtime_defaults$nBatch_SimGridGen
  )
  api <- ndm:::.ndm_legacy_run_env()
  row_values <- api$analysis2_normalize_row_values(
    api$analysis2_row_to_list(artifact_grid[1L, , drop = FALSE])
  )
  dataset_spec <- api$analysis2_sim_dataset_spec("ndmdatasets", row_values)
  if (is.null(runtime_defaults$MaxTimeIndex)) {
    runtime_defaults$MaxTimeIndex <- ndm_test_sim_max_time_index(dataset_spec)
  }
  canonical <- ndm_test_sim_fit_canonical_artifacts(
    api = api,
    grid = artifact_grid,
    dataset_spec = dataset_spec
  )

  sim_grid <- artifact_grid[rep(1L, 2L), , drop = FALSE]
  sim_grid$BaseID <- c(1L, 2L)
  sim_grid$c_endogeneous[[2L]] <- as.numeric(sim_grid$c_endogeneous[[1L]]) + 0.1
  rownames(sim_grid) <- NULL
  runtime_defaults$TfRecordDir <- canonical$tfrecord_dir
  runtime_defaults$SimEntry <- row_values
  runtime_defaults$SimGrid <- sim_grid
  runtime_defaults$nSamplesTrain <- n_samples_train
  runtime_defaults$nSamples_max <- max(
    n_samples_train,
    as.integer(runtime_defaults$nSamples_max)
  )
  runtime_defaults$nObsInference <- as.integer(dataset_spec$n_inference_batches) * 128L

  runtime_env <- ndm_prepare_runtime(
    config = config,
    runtime_env = ndm_new_runtime_env(parent = globalenv()),
    runtime_globals = runtime_defaults
  )
  ndm_test_clear_tensorflow_global_seed(runtime_env)

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
        ts = runtime_env$jnp$arange(
          start = 0L,
          stop = n_time_steps_sim,
          dtype = runtime_env$jnp$int32
        )
      ),
      VI_SaveAt_ODE_optim = runtime_env$diffrax$SaveAt(
        ts = runtime_env$jnp$arange(
          start = 0L,
          stop = n_times_lookahead,
          dtype = runtime_env$jnp$int32
        )
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
  if (is.list(runtime_globals_after_setup) && length(runtime_globals_after_setup) > 0L) {
    ndm_set_runtime_globals(runtime_env, runtime_globals_after_setup)
  }

  training_row_values <- row_values
  training_row_values$ModelType <- model_type
  training_row_values$ModelDepth <- as.integer(runtime_env$ModelDepth)
  training_row_values$ModelDims <- as.integer(runtime_env$ModelDims)
  training_row_values$nSamplesTrain <- as.integer(runtime_env$nSamplesTrain)
  training_spec <- api$analysis2_sim_training_spec(
    "ndmdatasets",
    training_row_values,
    model_type = model_type
  )
  ndm_set_runtime_globals(
    runtime_env,
    list(
      SIM_GLOBAL_OUTCOME_SD = canonical$outcome_sd,
      training_spec = training_spec
    )
  )
  ndm_prepare_data(
    runtime_env = runtime_env,
    generator = "sim"
  )

  snapshot <- ndm_test_copy_env(runtime_env)
  on.exit(ndm_test_restore_env(snapshot), add = TRUE)

  if (is.null(model_spec)) {
    model_spec <- ndm_model_spec(
      preset = "seirs_dynamic_beta",
      model_type = model_type
    )
  }
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
  train_elapsed_seconds <- NA_real_
  if (is.function(after_train_define)) {
    train_elapsed_seconds <- 0
    train_started <- proc.time()[["elapsed"]]
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
    train_elapsed_seconds <- train_elapsed_seconds + (proc.time()[["elapsed"]] - train_started)
    if (inherits(train_define_result, "try-error")) {
      stop(attr(train_define_result, "condition"))
    }
    after_train_define(runtime_env = runtime_env, model = model)
    train_started <- proc.time()[["elapsed"]]
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
    train_elapsed_seconds <- train_elapsed_seconds + (proc.time()[["elapsed"]] - train_started)
  } else {
    train_started <- proc.time()[["elapsed"]]
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
    train_elapsed_seconds <- proc.time()[["elapsed"]] - train_started
  }
  iterations_per_second <- if (is.finite(train_elapsed_seconds) &&
    train_elapsed_seconds > 0 &&
    is.finite(n_sgd) &&
    n_sgd > 0) {
    as.numeric(n_sgd) / train_elapsed_seconds
  } else {
    NA_real_
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
        holder_folder = file.path(work_dir, "results"),
        train_elapsed_seconds = train_elapsed_seconds,
        iterations_per_second = iterations_per_second,
        block_update_log = if (exists("block_update_log", envir = runtime_env, inherits = FALSE)) {
          get("block_update_log", envir = runtime_env, inherits = FALSE)
        } else {
          NULL
        }
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
    train_elapsed_seconds = train_elapsed_seconds,
    iterations_per_second = iterations_per_second,
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
      holder_folder = file.path(work_dir, "results"),
      train_elapsed_seconds = train_elapsed_seconds,
      iterations_per_second = iterations_per_second,
      block_update_log = if (exists("block_update_log", envir = trained$env, inherits = FALSE)) {
        get("block_update_log", envir = trained$env, inherits = FALSE)
      } else {
        NULL
      }
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

ndm_test_sim_evaluation_batch <- function(details) {
  runtime_env <- details$runtime_env
  if (!exists("TFDataset_inference", envir = runtime_env, inherits = FALSE) ||
      !exists("TFConst2JAXArray", envir = runtime_env, inherits = FALSE)) {
    stop("Expected an inference dataset and TF-to-JAX converter for paired evaluation.")
  }

  iterator <- reticulate::as_iterator(runtime_env$TFDataset_inference)
  batch <- reticulate::iter_next(iterator)
  if (is.null(batch) || length(batch) == 0L) {
    stop("The paired simulation inference dataset returned no batch.")
  }
  batch <- runtime_env$TFConst2JAXArray(batch)
  batch_sizes <- vapply(
    batch,
    function(value) as.integer(runtime_env$np$array(value$shape)[[1L]]),
    integer(1)
  )
  if (any(batch_sizes != as.integer(runtime_env$nBatch))) {
    stop(
      "The paired simulation inference batch must contain exactly ",
      as.integer(runtime_env$nBatch),
      " examples."
    )
  }
  batch
}

ndm_test_week_relative_accuracy_on_batch <- function(details,
                                                     batch,
                                                     target_week = 10L,
                                                     prediction_seed = 9001L,
                                                     n_monte = 10L,
                                                     eps = 1e-3) {
  runtime_env <- details$runtime_env
  n_monte <- as.integer(n_monte)
  predictions <- lapply(seq_len(n_monte), function(draw) {
    pred <- ndm_predict(
      details$trained,
      batch = batch,
      seed = as.integer(prediction_seed + draw - 1L),
      update_state = FALSE
    )
    ndm_test_py_numeric_array(runtime_env$np$asanyarray(pred$y_mu))
  })
  pred_mean <- Reduce(`+`, predictions) / n_monte
  truth <- ndm_test_py_numeric_array(runtime_env$np$asanyarray(batch$YTrue_out))
  history <- ndm_test_py_numeric_array(runtime_env$np$asanyarray(batch$YTrue))

  if (length(dim(pred_mean)) == 3L) {
    pred_mean <- pred_mean[, , 1L, drop = TRUE]
  }
  if (length(dim(truth)) == 3L) {
    truth <- truth[, , 1L, drop = TRUE]
  }
  if (length(dim(history)) == 3L) {
    history <- history[, , 1L, drop = TRUE]
  }
  target_week <- as.integer(target_week)
  if (target_week < 1L || target_week > ncol(truth) || target_week > ncol(pred_mean)) {
    stop("Requested paired-evaluation week is outside the prediction horizon.")
  }

  baseline_index <- as.integer(runtime_env$nTimesTotal - runtime_env$nTimesLookahead)
  baseline <- history[, baseline_index]
  squared_pred_error <- (pred_mean[, target_week] - truth[, target_week])^2
  squared_baseline_error <- (baseline - truth[, target_week])^2
  paired <- is.finite(squared_pred_error) &
    is.finite(squared_baseline_error)
  if (!any(paired)) {
    stop("Paired simulation evaluation produced no finite error pairs.")
  }
  rss_pred <- mean(squared_pred_error[paired])
  rss_baseline <- mean(squared_baseline_error[paired])

  list(
    rss_pred = rss_pred,
    rss_baseline = rss_baseline,
    skill = 1 - (eps + sqrt(rss_pred)) / (eps + sqrt(rss_baseline)),
    relative_accuracy = (eps + sqrt(rss_baseline)) / (eps + sqrt(rss_pred)),
    truth = truth[, target_week],
    baseline = baseline
  )
}

ndm_test_sim_parity_shared_runtime_globals <- function(shared_seed,
                                                       n_samples_train = 256L,
                                                       scaling_outer_loops = 2L,
                                                       scaling_inner_loops = 12L) {
  n_samples_train <- as.integer(n_samples_train)
  list(
    SEED_ = as.integer(shared_seed),
    ReshuffleEachIteration = TRUE,
    nSamplesTrain = n_samples_train,
    nSamples_max = n_samples_train,
    SimEntry = list(
      nSamplesTrain = n_samples_train,
      ResaveThisTFRecord = 1L
    ),
    SimScalingOuterLoops = as.integer(scaling_outer_loops),
    SimScalingInnerLoops = as.integer(scaling_inner_loops)
  )
}

ndm_test_sim_parity_neural_config_overrides <- function() {
  list(
    neuralode_optim_dt0 = 1e-2,
    neuralode_optim_controller = "pid",
    # A modest susceptible-state offset is strong enough to encode the SEIRS
    # ordering without pinning the initial simplex near an overconfident corner.
    neuralode_init_state_logit_offset = c(1.0, 0, 0, 0),
    neuralode_init_state_logit_scale_max = 1.0,
    neuralode_variational = TRUE,
    neuralode_kl_weight = 1e-3,
    neuralode_mean_loss_weight = 1.0
  )
}

ndm_test_sim_parity_neural_runtime_globals_after_setup <- function() {
  list(
    TrackBlockUpdateNorms = TRUE
  )
}

ndm_test_py_numeric_array <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  if (isTRUE(tryCatch(reticulate::is_py_object(x), error = function(...) FALSE))) {
    x <- tryCatch(
      reticulate::import("numpy", delay_load = TRUE)$asanyarray(x),
      error = function(...) x
    )
    if (isTRUE(tryCatch(reticulate::is_py_object(x), error = function(...) FALSE))) {
      x <- reticulate::py_to_r(x)
    }
  }
  arr <- as.array(x)
  storage.mode(arr) <- "double"
  arr
}

ndm_test_as_numeric_matrix <- function(x, label = "value") {
  arr <- ndm_test_py_numeric_array(x)
  if (is.null(arr)) {
    stop("Expected `", label, "` to be available.", call. = FALSE)
  }

  dims <- dim(arr)
  if (is.null(dims) || length(dims) == 0L) {
    return(matrix(as.numeric(arr), ncol = 1L))
  }
  if (length(dims) == 1L) {
    return(matrix(as.numeric(arr), nrow = dims[[1L]], ncol = 1L))
  }
  if (length(dims) == 2L) {
    return(matrix(as.numeric(arr), nrow = dims[[1L]], ncol = dims[[2L]]))
  }

  matrix(as.numeric(arr), nrow = dims[[1L]], ncol = prod(dims[-1L]))
}

ndm_test_broadcast_like <- function(x, target, label = "value") {
  value <- ndm_test_as_numeric_matrix(x, label = label)
  target_dim <- dim(target)
  if (is.null(target_dim) || length(target_dim) != 2L) {
    stop("`target` must be a numeric matrix.", call. = FALSE)
  }
  if (identical(dim(value), target_dim)) {
    return(value)
  }
  if (nrow(value) == 1L && ncol(value) == target_dim[[2L]]) {
    return(matrix(
      rep(value[1L, ], each = target_dim[[1L]]),
      nrow = target_dim[[1L]],
      ncol = target_dim[[2L]]
    ))
  }
  if (ncol(value) == 1L && nrow(value) == target_dim[[1L]]) {
    return(matrix(
      rep(value[, 1L], times = target_dim[[2L]]),
      nrow = target_dim[[1L]],
      ncol = target_dim[[2L]]
    ))
  }

  value_vec <- as.numeric(value)
  if (length(value_vec) == 1L) {
    return(matrix(value_vec, nrow = target_dim[[1L]], ncol = target_dim[[2L]]))
  }
  if (length(value_vec) == target_dim[[2L]]) {
    return(matrix(
      rep(value_vec, each = target_dim[[1L]]),
      nrow = target_dim[[1L]],
      ncol = target_dim[[2L]]
    ))
  }
  if (length(value_vec) == target_dim[[1L]]) {
    return(matrix(
      rep(value_vec, times = target_dim[[2L]]),
      nrow = target_dim[[1L]],
      ncol = target_dim[[2L]]
    ))
  }

  stop(
    "Could not broadcast `",
    label,
    "` with dims ",
    paste(dim(value), collapse = "x"),
    " to target dims ",
    paste(target_dim, collapse = "x"),
    ".",
    call. = FALSE
  )
}

ndm_test_capture_train_prediction <- function(details, seed = 1L) {
  if (is.null(details$batch)) {
    stop("Expected `details$batch` to capture a simulation batch for state diagnostics.")
  }

  runtime_env <- details$trained$env
  getpred_saveat_info <- if (exists("ndm_runtime_normalize_getpred_saveat_info", envir = runtime_env, inherits = TRUE)) {
    runtime_env$ndm_runtime_normalize_getpred_saveat_info(runtime_env$GetPredSaveAtInfo_default)
  } else {
    runtime_env$GetPredSaveAtInfo_default
  }
  seed_matrix <- runtime_env$jax$random$split(
    runtime_env$jax$random$PRNGKey(as.integer(seed)),
    as.integer(runtime_env$nBatch)
  )
  if (exists("ndm_runtime_data_to_device", envir = runtime_env, inherits = TRUE)) {
    seed_matrix <- runtime_env$ndm_runtime_data_to_device(seed_matrix)
  }

  runtime_env$GetPred_train_jit(
    runtime_env$ModelList,
    runtime_env$batch2package(details$batch),
    runtime_env$state,
    runtime_env$PriorList,
    runtime_env$PolicyList,
    getpred_saveat_info,
    seed_matrix
  )[[1L]]
}

ndm_test_collect_tb_time_varying_trajectories <- function(pred, spec) {
  if (length(spec$time_varying_terms) == 0L) {
    return(list())
  }

  times <- ndm_test_as_numeric_matrix(
    pred$ODEParamsSampList[["diff_eq_sol_ts"]],
    label = "diff_eq_sol_ts"
  )
  trajectories <- list()
  keys <- names(pred$ODEParamsSampList)

  for (term in spec$time_varying_terms) {
    flat_key <- paste0("diff_eq_sol_ys.", term)
    if (flat_key %in% keys) {
      trajectories[[term]] <- ndm_test_as_numeric_matrix(
        pred$ODEParamsSampList[[flat_key]],
        label = flat_key
      )
      next
    }

    if (identical(term, "c_t") && identical(spec$preset, "tb_k")) {
      x1 <- ndm_test_broadcast_like(pred$ODEParamsSampList[["x1_samp"]], times, label = "x1_samp")
      x2 <- ndm_test_broadcast_like(pred$ODEParamsSampList[["x2_samp"]], times, label = "x2_samp")
      trajectories[[term]] <- x1 * (pmax(1.0, times) ^ x2)
      next
    }

    if (identical(term, "c_t") && identical(spec$preset, "tb_l")) {
      x1 <- ndm_test_broadcast_like(pred$ODEParamsSampList[["x1_samp"]], times, label = "x1_samp")
      x2 <- ndm_test_broadcast_like(pred$ODEParamsSampList[["x2_samp"]], times, label = "x2_samp")
      x3 <- ndm_test_broadcast_like(pred$ODEParamsSampList[["x3_samp"]], times, label = "x3_samp")
      trajectories[[term]] <- x1 * (x2 + exp((-1 * x3) * times))
      next
    }

    stop(
      "Missing trajectory data for time-varying term `",
      term,
      "` in preset `",
      spec$preset,
      "`.",
      call. = FALSE
    )
  }

  trajectories
}

ndm_test_capture_tb_structure_diagnostics <- function(details,
                                                      spec,
                                                      label,
                                                      seed = 123L) {
  runtime_env <- details$trained$env
  public_pred <- details$smoke_pred
  if (is.null(public_pred)) {
    public_pred <- ndm_predict(
      details$trained,
      batch = details$batch,
      seed = seed,
      update_state = FALSE
    )
  }
  raw_pred <- ndm_test_capture_train_prediction(details, seed = seed)

  keys <- names(raw_pred$ODEParamsSampList)
  state_keys <- paste0("diff_eq_sol_ys.", spec$state_terms)
  missing_state_keys <- state_keys[!state_keys %in% keys]
  if (length(missing_state_keys) > 0L) {
    stop(
      "Missing solved-state keys for `",
      label,
      "`: ",
      paste(missing_state_keys, collapse = ", "),
      ".",
      call. = FALSE
    )
  }

  state_mats <- lapply(
    setNames(state_keys, spec$state_terms),
    function(key) {
      ndm_test_as_numeric_matrix(raw_pred$ODEParamsSampList[[key]], label = key)
    }
  )
  state_dim <- dim(state_mats[[1L]])
  bad_dims <- names(state_mats)[!vapply(state_mats, function(mat) identical(dim(mat), state_dim), logical(1))]
  if (length(bad_dims) > 0L) {
    stop(
      "State trajectory dims differ for `",
      label,
      "`: ",
      paste(bad_dims, collapse = ", "),
      ".",
      call. = FALSE
    )
  }

  state_cube <- array(
    NA_real_,
    dim = c(state_dim[[1L]], state_dim[[2L]], length(state_mats)),
    dimnames = list(NULL, NULL, spec$state_terms)
  )
  for (i in seq_along(state_mats)) {
    state_cube[, , i] <- state_mats[[i]]
  }

  state_totals <- apply(state_cube, c(1L, 2L), sum)
  initial_totals <- matrix(
    state_totals[1L, ],
    nrow = nrow(state_totals),
    ncol = ncol(state_totals),
    byrow = TRUE
  )
  state_scale <- suppressWarnings(max(abs(initial_totals), na.rm = TRUE))
  if (!is.finite(state_scale) || state_scale <= 0) {
    state_scale <- 1.0
  }
  state_epsilon <- 1e-6 * max(1.0, state_scale)
  time_varying <- ndm_test_collect_tb_time_varying_trajectories(raw_pred, spec)
  time_varying_all_finite <- if (length(time_varying) == 0L) {
    TRUE
  } else {
    all(vapply(time_varying, function(mat) all(is.finite(mat)), logical(1)))
  }
  time_varying_min <- if (length(time_varying) == 0L) {
    NA_real_
  } else {
    min(unlist(time_varying, use.names = FALSE), na.rm = TRUE)
  }

  list(
    label = label,
    public_pred = public_pred,
    raw_pred = raw_pred,
    pred_mu = runtime_env$np$asanyarray(public_pred$y_mu),
    pred_sigma = runtime_env$np$asanyarray(public_pred$y_sigma),
    state_cube = state_cube,
    state_totals = state_totals,
    state_epsilon = state_epsilon,
    min_state = min(state_cube, na.rm = TRUE),
    max_rel_mass_drift = max(abs(state_totals - initial_totals) / pmax(abs(initial_totals), 1e-12), na.rm = TRUE),
    time_varying = time_varying,
    time_varying_all_finite = time_varying_all_finite,
    time_varying_min = time_varying_min
  )
}

ndm_test_capture_initial_state_metrics <- function(details, seed = 1L) {
  pred <- ndm_test_capture_train_prediction(details, seed = seed)
  runtime_env <- details$trained$env

  state_components <- cbind(
    s = as.numeric(reticulate::py_to_r(runtime_env$np$asanyarray(pred$ODEParamsSampList$s_l_samp))),
    e = as.numeric(reticulate::py_to_r(runtime_env$np$asanyarray(pred$ODEParamsSampList$e_l_samp))),
    i = as.numeric(reticulate::py_to_r(runtime_env$np$asanyarray(pred$ODEParamsSampList$i_l_samp))),
    r = as.numeric(reticulate::py_to_r(runtime_env$np$asanyarray(pred$ODEParamsSampList$r_l_samp)))
  )
  state_components[!is.finite(state_components)] <- NA_real_
  row_totals <- rowSums(state_components, na.rm = TRUE)
  proportions <- sweep(state_components, 1L, pmax(row_totals, 1e-12), "/")
  entropies <- -rowSums(proportions * log(pmax(proportions, 1e-12)), na.rm = TRUE)

  list(
    mean_entropy = mean(entropies, na.rm = TRUE),
    max_component = max(proportions, na.rm = TRUE),
    mean_max_component = mean(apply(proportions, 1L, max, na.rm = TRUE), na.rm = TRUE),
    mean_components = colMeans(proportions, na.rm = TRUE),
    proportions = proportions
  )
}

ndm_test_collect_week10_relative_accuracy_pair <- function(endogeneity = 0.0,
                                                           shared_seed = 1010L,
                                                           n_times_lookahead = 10L,
                                                           n_sgd = 1L,
                                                           model_dims = 32L,
                                                           shared_config_overrides = NULL,
                                                           shared_runtime_globals = NULL,
                                                           shared_runtime_globals_after_setup = NULL,
                                                           decoder_config_overrides = NULL,
                                                           decoder_runtime_globals = NULL,
                                                           decoder_runtime_globals_after_setup = NULL,
                                                           neuralode_config_overrides = NULL,
                                                           neuralode_runtime_globals = NULL,
                                                           neuralode_runtime_globals_after_setup = NULL,
                                                           decoder_before_train = NULL,
                                                           neuralode_before_train = NULL,
                                                           return_details = FALSE) {
  shared_seed <- as.integer(shared_seed)
  common_args <- list(
    endogeneity = endogeneity,
    case_seed = shared_seed,
    n_checkpoints = 1L,
    n_times_lookahead = n_times_lookahead,
    n_sgd = as.integer(n_sgd),
    model_dims = as.integer(model_dims),
    return_details = TRUE
  )
  shared_runtime_globals <- ndm_test_merge_runtime_lists(
    list(SEED_ = shared_seed),
    shared_runtime_globals
  )
  shared_runtime_globals_after_setup <- ndm_test_merge_runtime_lists(
    list(),
    shared_runtime_globals_after_setup
  )
  shared_config_overrides <- ndm_test_merge_runtime_lists(
    list(),
    shared_config_overrides
  )

  decoder_details <- do.call(
    ndm_test_fit_sim_case,
    c(
      list(
        model_type = "DecoderOnly",
        config_overrides = ndm_test_merge_runtime_lists(
          shared_config_overrides,
          decoder_config_overrides
        ),
        runtime_globals = ndm_test_merge_runtime_lists(shared_runtime_globals, decoder_runtime_globals),
        runtime_globals_after_setup = ndm_test_merge_runtime_lists(
          shared_runtime_globals_after_setup,
          decoder_runtime_globals_after_setup
        ),
        before_train = decoder_before_train
      ),
      common_args
    )
  )
  neuralode_details <- do.call(
    ndm_test_fit_sim_case,
    c(
      list(
        model_type = "NeuralODE",
        config_overrides = ndm_test_merge_runtime_lists(
          shared_config_overrides,
          neuralode_config_overrides
        ),
        runtime_globals = ndm_test_merge_runtime_lists(shared_runtime_globals, neuralode_runtime_globals),
        runtime_globals_after_setup = ndm_test_merge_runtime_lists(
          shared_runtime_globals_after_setup,
          neuralode_runtime_globals_after_setup
        ),
        before_train = neuralode_before_train
      ),
      common_args
    )
  )

  evaluation_batch <- ndm_test_sim_evaluation_batch(decoder_details)
  evaluation_seed <- as.integer(shared_seed + 9001L)
  decoder_week10 <- ndm_test_week_relative_accuracy_on_batch(
    decoder_details,
    evaluation_batch,
    target_week = n_times_lookahead,
    prediction_seed = evaluation_seed
  )
  neuralode_week10 <- ndm_test_week_relative_accuracy_on_batch(
    neuralode_details,
    evaluation_batch,
    target_week = n_times_lookahead,
    prediction_seed = evaluation_seed
  )
  if (!isTRUE(all.equal(
    decoder_week10$rss_baseline,
    neuralode_week10$rss_baseline,
    tolerance = 1e-12
  ))) {
    stop("Paired decoder and NeuralODE evaluations produced different baseline RSS values.")
  }
  result <- list(
    decoder_relative_accuracy_10 = decoder_week10$relative_accuracy,
    neuralode_relative_accuracy_10 = neuralode_week10$relative_accuracy,
    decoder_skill_10 = decoder_week10$skill,
    neuralode_skill_10 = neuralode_week10$skill,
    decoder_train_elapsed_seconds = decoder_details$train_elapsed_seconds,
    neuralode_train_elapsed_seconds = neuralode_details$train_elapsed_seconds,
    decoder_iterations_per_second = decoder_details$iterations_per_second,
    neuralode_iterations_per_second = neuralode_details$iterations_per_second,
    info = sprintf(
      paste(
        "week10 decoder rel_acc=%.6f",
        "week10 neuralode rel_acc=%.6f",
        "week10 decoder skill=%.6f",
        "week10 neuralode skill=%.6f",
        "decoder train_s=%.6f",
        "neuralode train_s=%.6f",
        "decoder iter_per_sec=%.6f",
        "neuralode iter_per_sec=%.6f",
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
      decoder_details$train_elapsed_seconds,
      neuralode_details$train_elapsed_seconds,
      decoder_details$iterations_per_second,
      neuralode_details$iterations_per_second,
      shared_seed,
      as.integer(n_times_lookahead),
      as.integer(model_dims),
      as.integer(n_sgd)
    )
  )
  if (isTRUE(return_details)) {
    result$decoder_details <- decoder_details
    result$neuralode_details <- neuralode_details
    result$evaluation_batch <- evaluation_batch
    result$decoder_week10 <- decoder_week10
    result$neuralode_week10 <- neuralode_week10
  }
  result
}

ndm_skip_if_no_sim_backend <- function() {
  skip_on_cran()
  ndm_require_backend_test_stack("simulation fit tests")
}
