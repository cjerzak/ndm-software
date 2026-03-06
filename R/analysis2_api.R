# Generated from the Analysis2 rewrite so ndm remains the single
# executable source of truth for Analysis2 orchestration.

analysis2_log <- function(text) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), text))
  invisible(text)
}

analysis2_f2n <- function(x) {
  as.numeric(as.character(x))
}

analysis2_parse_bool <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  if (is.logical(x) && length(x) == 1L) {
    return(x)
  }

  value <- tolower(trimws(as.character(x)))
  if (value %in% c("true", "t", "1", "yes", "y")) {
    return(TRUE)
  }
  if (value %in% c("false", "f", "0", "no", "n")) {
    return(FALSE)
  }

  stop("Could not parse logical value: ", x, call. = FALSE)
}

analysis2_parse_args <- function(args = commandArgs(TRUE)) {
  out <- list(positional = character())

  for (arg in args) {
    if (!startsWith(arg, "--")) {
      out$positional <- c(out$positional, arg)
      next
    }

    arg <- sub("^--", "", arg)
    pieces <- strsplit(arg, "=", fixed = TRUE)[[1]]
    key <- pieces[[1]]
    value <- if (length(pieces) > 1L) {
      paste(pieces[-1], collapse = "=")
    } else {
      "TRUE"
    }
    out[[gsub("-", "_", key)]] <- value
  }

  for (name in c("dry_run", "run_figures", "respect_grid_model_type", "resave_tfrecords")) {
    if (!is.null(out[[name]])) {
      out[[name]] <- analysis2_parse_bool(out[[name]])
    }
  }

  out
}

analysis2_script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) == 0L) {
    return(NULL)
  }
  normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = TRUE)
}

analysis2_paths <- function(project_root = NULL, script_file = analysis2_script_path()) {
  if (is.null(project_root)) {
    if (is.null(script_file)) {
      stop("Could not infer project root. Supply `--project_root=...` when running the script.", call. = FALSE)
    }
    analysis2_root <- dirname(script_file)
    project_root <- dirname(analysis2_root)
  } else {
    project_root <- normalizePath(project_root, winslash = "/", mustWork = TRUE)
    analysis2_root <- file.path(project_root, "Analysis2")
  }

  list(
    project_root = normalizePath(project_root, winslash = "/", mustWork = TRUE),
    analysis_root = normalizePath(file.path(project_root, "Analysis"), winslash = "/", mustWork = TRUE),
    analysis2_root = normalizePath(analysis2_root, winslash = "/", mustWork = TRUE)
  )
}

analysis2_resolve_outer_iterations <- function(opts, default) {
  raw <- opts$outer %||% if (length(opts$positional) > 0L) opts$positional[[1]] else NULL
  if (is.null(raw) || !nzchar(raw)) {
    return(as.integer(default))
  }

  parts <- strsplit(as.character(raw), ",", fixed = TRUE)[[1]]
  as.integer(trimws(parts))
}

analysis2_row_to_list <- function(row) {
  stopifnot(nrow(row) == 1L)
  out <- vector("list", length = ncol(row))
  names(out) <- names(row)

  for (name in names(row)) {
    value <- row[[name]][[1]]
    numeric_value <- suppressWarnings(analysis2_f2n(value))
    out[[name]] <- if (!is.na(numeric_value)) numeric_value else as.character(value)
  }

  out
}

analysis2_resolve_model_spec <- function(model_tex_loc,
                                         model_type = "DecoderOnly",
                                         project_root) {
  if (is.null(model_tex_loc) || !nzchar(model_tex_loc) || is.na(model_tex_loc)) {
    return(NULL)
  }

  packaged_tex_path <- system.file(
    "extdata",
    "model_specs",
    basename(model_tex_loc),
    package = "ndm"
  )
  if (nzchar(packaged_tex_path) && file.exists(packaged_tex_path)) {
    return(ndm_model_spec_from_tex(
      path = packaged_tex_path,
      model_type = model_type
    ))
  }

  tex_path <- if (startsWith(model_tex_loc, "./")) {
    file.path(project_root, sub("^\\./", "", model_tex_loc))
  } else {
    model_tex_loc
  }

  ndm_model_spec_from_tex(
    path = normalizePath(tex_path, winslash = "/", mustWork = TRUE),
    model_type = model_type
  )
}

analysis2_model_type <- function(opts,
                                 grid_model_type = NULL,
                                 default = "DecoderOnly") {
  if (!is.null(opts$model_type) && nzchar(opts$model_type)) {
    return(opts$model_type)
  }

  if (isTRUE(opts$respect_grid_model_type) &&
      !is.null(grid_model_type) &&
      nzchar(as.character(grid_model_type))) {
    return(as.character(grid_model_type))
  }

  default
}

analysis2_export_bindings <- function(env = parent.frame(), exclude = character()) {
  names_to_skip <- unique(c(
    "args", "opts", "paths", "config", "spec", "model", "trained", "runtime_env",
    "outer_iterations", "outer_iteration", "grid", "row_values", "helper_path",
    exclude
  ))

  export_names <- setdiff(ls(env, all.names = TRUE), names_to_skip)
  mget(export_names, envir = env, inherits = FALSE)
}

analysis2_sync_runtime_env <- function(runtime_env,
                                      env = parent.frame(),
                                      exclude = character()) {
  ndm_set_runtime_globals(
    runtime_env,
    analysis2_export_bindings(env = env, exclude = exclude)
  )
  invisible(runtime_env)
}

analysis2_assert_no_pretraining <- function(nSGD_pretrain) {
  if (!identical(as.integer(nSGD_pretrain), 0L)) {
    stop(
      "The Analysis2 rewrite currently supports the active no-pretraining production path only. ",
      "Set `nSGD_pretrain <- 0L` or extend the migrated pipeline before using pretraining.",
      call. = FALSE
    )
  }
}

analysis2_dir_create <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
  invisible(path)
}

analysis2_run_real <- function(args = commandArgs(TRUE)) {
  options(error = NULL)
  analysis2_log("Starting Analysis2 SuperLModel_MasterReal.R")

  opts <- analysis2_parse_args(args)
  paths <- analysis2_paths(project_root = opts$project_root)
  setwd(paths$project_root)
  options(ndm.analysis_root = paths$analysis_root)

  AnalysisName <- opts$analysis_name %||% "RealApril15"
  AnalysisDate <- Sys.Date()

  ReSaveTfRecords <- isTRUE(opts$resave_tfrecords %||% FALSE)
  if (ReSaveTfRecords) {
    stop("Analysis2 does not regenerate TFRecords in this phase. Use existing TFRecords only.", call. = FALSE)
  }
  force2GPU <- TRUE
  nRealGridSeed <- 128L
  nExamplesPerCell <- 10L
  nRealGrid <- 704L
  GPU_MEM_FRAC <- NULL
  UseShortOutcomes <- TRUE
  nPlaces <- 170L

  nTimesLookahead <- 4L * 3L
  nTimesLookValidationInference <- nTimesLookahead
  OverDoDataFrac <- 0.90
  DecoderInNeuralODE <- FALSE
  endAppend <- FALSE
  analysis2_log(if (isTRUE(endAppend)) {
    "Appending special tokens to END"
  } else {
    "Appending special tokens to START"
  })

  LEARNING_RATE_MAX_model <- 0.0002
  LEARNING_RATE_MAX_pretrain <- 1e-5
  useLSTM <- FALSE
  doGrid <- TRUE
  SimMode <- FALSE
  nBatch <- as.integer(32L)
  nSGD_pretrain <- 0L
  nCheckpoints <- 10L
  nEpochesMax <- 9L
  nSamples_max <- 20000L
  nSGD_DefiningLRSeq <- nSGD_model <- as.integer(round(nEpochesMax * (nSamples_max / nBatch)))
  PreTrain <- ifelse(nSGD_pretrain == 0L, FALSE, TRUE)
  nTotalDiseases <- (nSynthDiseases <- 3L) + 1L
  nSGD_posttrain <- nSGD_model
  if (ReSaveTfRecords) {
    nSGD_pretrain <- nSGD_DefiningLRSeq <- nSGD_model <- 0L
  }
  specificOptState <- TRUE
  SharedListNames <- c("TS")

  outer_iterations <- analysis2_resolve_outer_iterations(opts, default = 3L)
  set.seed(theInitialSeed <- 12L)

  HolderFolder <- file.path(paths$project_root, "SavedResults", "Real", sprintf("Results_%s", AnalysisName))
  analysis2_dir_create(HolderFolder)

  local_data_env <- new.env(parent = baseenv())
  load(file.path(paths$project_root, "Data", "COVID_LEARNER_REALDAT_RESTART.rdata"), envir = local_data_env)
  list2env(as.list(local_data_env), environment())
  outcome_metric <- get("outcome_metric", envir = environment(), inherits = FALSE)
  removable_objects <- intersect(
    c("cumsum_totals", "predicted_df", "truth_df_MASTER_add", "context_df_red"),
    ls(envir = environment(), all.names = TRUE)
  )
  if (length(removable_objects) > 0L) {
    rm(list = removable_objects, envir = environment())
  }
  truth_df$time_id <- truth_df$week_id
  truth_df$targetTime_id <- truth_df$targetWeek_id

  data_subset <- "high_income"
  dataInputs_pool_orig <- dataInputs_pool <- c(
    "ihme_true_value_inc_death_per_capita_sqrt",
    "ihme_true_value_inc_case_per_capita_sqrt",
    "ihme_true_value_inc_hosp_per_capita_sqrt",
    "NPI_DAT_bsg_stringency_index"
  )
  if (any(!dataInputs_pool_orig %in% colnames(truth_df))) {
    missing_cols <- dataInputs_pool_orig[!dataInputs_pool_orig %in% colnames(truth_df)]
    stop("Missing required columns in `truth_df`: ", paste(missing_cols, collapse = ", "), call. = FALSE)
  }

  true_value_names <- "ihme_true_value_per_capita"
  all_true_value_names <- c(
    "ihme_true_value_per_capita",
    "ihme_true_value_inc_case_per_capita",
    "ihme_true_value_inc_hosp_per_capita",
    "ihme_true_value_inc_death_per_capita"
  )
  nOutcomes <- length(true_value_names)

  dataInputs_pool <- dataInputs_pool[dataInputs_pool != outcome_metric]
  dataInputs_pool_DropNone <- dataInputs_pool[dataInputs_pool != ""]
  dataInputs_pool_use <- c()
  nMonte_DataInterSpace <- 3L
  for (comb_ in 3:length(dataInputs_pool_DropNone)) {
    comb_grid <- utils::combn(length(dataInputs_pool_DropNone), comb_)
    comb_grid <- as.matrix(comb_grid[, sample(seq_len(ncol(comb_grid)), min(ncol(comb_grid), nMonte_DataInterSpace), replace = FALSE), drop = FALSE])
    dataInputs_pool_use <- unique(c(
      dataInputs_pool_use,
      apply(comb_grid, 2, function(indices) paste(dataInputs_pool_DropNone[indices], collapse = "__"))
    ))
  }
  dataInputs_pool <- grep("__", dataInputs_pool_use, value = TRUE)
  evaluation_seq <- c(1L, 2L, 3L, 4L)

  RealGrid <- as.data.frame(data.table::fread(
    file.path(paths$project_root, "Data", "RunGrids", "RealGrids", sprintf("RealGrid_%s.csv", AnalysisName))
  ))

  if (!ReSaveTfRecords) {
    RealGrid <- RealGrid[order(RealGrid$BaseID), ]
    if (RealGrid$BaseID[outer_iterations[[1]]] %% 2 == 0) {
      even_rows <- RealGrid$BaseID %% 2 == 0
      RealGrid[even_rows, ] <- RealGrid[even_rows, ][
        order(apply(RealGrid[even_rows, ], 1, function(zr) rlang::hash(paste(zr, collapse = "_")))),
      ]
    }
  }

  if (isTRUE(opts$dry_run)) {
    analysis2_log(sprintf("Dry run. AnalysisName=%s; outer iterations=%s", AnalysisName, paste(outer_iterations, collapse = ",")))
    return(invisible(list(paths = paths, analysis_name = AnalysisName, outer_iterations = outer_iterations)))
  }

  for (outer_iteration in outer_iterations) {
    analysis2_log(sprintf("Starting outer iteration %s", outer_iteration))
    set.seed(SEED_ <- as.integer(outer_iteration))

    row_values <- analysis2_row_to_list(RealGrid[outer_iteration, , drop = FALSE])
    list2env(row_values, environment())
    BaseID <- row_values$BaseID
    model_tex_loc <- row_values$model_tex_loc
    floatType <- row_values$floatType
    ContextLength <- row_values$ContextLength
    evaluationTime <- row_values$evaluationTime

    modelingStrategyNameKey <- paste(c(
      "RealMode",
      paste(names(row_values), unlist(row_values), sep = "_")
    ), collapse = "__")

    TfRecordDir <- file.path(paths$project_root, "Data", "RunTFRecords", "RealTFRecords", AnalysisName)
    if (!ReSaveTfRecords) {
      stopifnot(file.exists(file.path(TfRecordDir, sprintf("train_%s.tfrecord", BaseID))))
    }

    ModelType <- analysis2_model_type(
      opts = opts,
      grid_model_type = row_values$ModelType,
      default = "DecoderOnly"
    )
    spec <- analysis2_resolve_model_spec(
      model_tex_loc = model_tex_loc,
      model_type = ModelType,
      project_root = paths$project_root
    )

    config <- ndm_create_config(
      model_type = ModelType,
      backbone = "transformer",
      analysis_root = paths$analysis_root,
      float_type = as.character(floatType),
      force_to_gpu = force2GPU,
      resave_tfrecords = ReSaveTfRecords,
      gpu_mem_frac = GPU_MEM_FRAC
    )
    runtime_env <- ndm_prepare_runtime(config = config)
    jnp <- runtime_env$jnp
    diffrax <- runtime_env$diffrax

    maxTimesPast <- ContextLength
    nTimesTotal <- nTimesLookahead + maxTimesPast
    minAnchoringTimeID <- 4L
    MIN_NA_ACCEPT_FRAC <- 4 / maxTimesPast

    NTimeSteps_SIM <- (nTimesLookahead + abs(maxTimesPast)) * 2L
    MaxSteps <- 100000L
    dt0_init <- 1e-1
    VI_TotalTimesInLikelihood <- nTimesLookahead
    VI_SaveAt_ODE <- diffrax$SaveAt(ts = jnp$array(1:VI_TotalTimesInLikelihood))
    diff_eq_solver <- VI_diff_eq_solver <- diffrax$Dopri8()
    stepsize_controller <- diffrax$PIDController(rtol = 1e-7, atol = 1e-9)
    diffraxInterpolator <- diffrax$LinearInterpolation

    MaxSteps <- as.integer(10^6)
    VI_SaveAt_ODE_sim <- diffrax$SaveAt(ts = jnp$array(0L:(NTimeSteps_SIM - 1L)))
    VI_SaveAt_ODE_optim <- diffrax$SaveAt(ts = jnp$array(0L:(VI_TotalTimesInLikelihood - 1L)))
    VI_diff_eq_solver_optim <- VI_diff_eq_solver_dgp <- diffrax$Tsit5()
    dt0_init_dgp <- 1e-3
    stepsize_controller_dgp <- diffrax$PIDController(rtol = 1e-6, atol = 1e-7)
    if (!DecoderInNeuralODE) {
      dt0_init_optim <- 1e-3
      stepsize_controller_optim <- diffrax$PIDController(rtol = 1e-5, atol = 1e-7)
    } else {
      dt0_init_optim <- 1e-3
      stepsize_controller_optim <- diffrax$ConstantStepSize()
    }

    dataInputs <- dataInputs[dataInputs != ""]
    dataInputs_colnames <- unique(unlist(strsplit(dataInputs, split = "__", fixed = TRUE)))
    dataInputs_colnames_past <- unique(dataInputs_pool_orig)
    dataInputs_colnames_future <- unique(dataInputs_colnames[grepl("FUTURE", dataInputs_colnames)])
    perspective_pool <- c(
      if (length(dataInputs_colnames_past) > 0L) "past" else NULL,
      if (length(dataInputs_colnames_future) > 0L) "future" else NULL
    )

    truth_df_red <- truth_df[!is.na(truth_df$ihme_true_value_per_capita), ]
    if (identical(data_subset, "all")) {
      keep_place_ids <- unique(truth_df_red$location_id)
    } else {
      keep_place_ids <- unique(truth_df_red[
        truth_df_red$LOC2_region_name %in% c(
          "Western Europe",
          "High-income North America",
          "Central Europe",
          "High-income Asia Pacific"
        ),
      ]$location_id)
    }
    truth_df_red <- truth_df_red[truth_df_red$location_id %in% keep_place_ids, ]
    truth_df_red$Pop <- truth_df_red$POP_population
    truth_df_red <- truth_df_red[!is.na(truth_df_red$location_name), ]
    truth_df_red$location_id_numeric <- as.integer(as.factor(truth_df_red$location_id)) - 1L

    coordinates_mat <- utils::read.csv(file.path(paths$project_root, "Data", "coordinates_mat.csv"))

    in_out_cutpoint <- round(stats::quantile(
      sort(unique(truth_df$time_id)),
      prob = evaluationTime / (max(evaluation_seq) + 1)
    ))
    times_out <- in_out_cutpoint + 1L
    times_out <- times_out[times_out <= max(truth_df$time_id)]
    times_in <- 0L:in_out_cutpoint
    AVERAGE_TRUTH <- mean(truth_df_red$ihme_true_value_per_capita, na.rm = TRUE)

    analysis2_sync_runtime_env(runtime_env, env = environment())
    ndm_prepare_data(
      runtime_env = runtime_env,
      analysis_root = paths$analysis_root,
      generator = "real"
    )

    if (!ReSaveTfRecords) {
      tf_exists <- sapply(unique(RealGrid$BaseID), function(base_id) {
        file.exists(file.path(TfRecordDir, sprintf("train_%s.tfrecord", base_id)))
      })
      if (any(!tf_exists)) {
        warning(
          sprintf(
            "Some TFRecords are missing under %s. Expected %s unique records.",
            TfRecordDir,
            length(unique(RealGrid$BaseID))
          ),
          call. = FALSE
        )
      }
    }

    if (nSGD_model > 0L) {
      analysis2_assert_no_pretraining(nSGD_pretrain)
      analysis2_log("Building core ML model...")
      model <- ndm_build_model(
        runtime_env = runtime_env,
        analysis_root = paths$analysis_root,
        model_type = ModelType,
        model_spec = spec,
        backbone = "transformer"
      )

      analysis2_log("Running calibration...")
      ndm_source_runtime_calibration(
        analysis_root = paths$analysis_root,
        env = runtime_env
      )

      LEARNING_RATE_MAX <- LEARNING_RATE_MAX_model
      nSGD_DefiningLRSeq <- nSGD_model
      analysis2_sync_runtime_env(runtime_env, env = environment())

      analysis2_log("Starting final training run...")
      trained <- ndm_train(
        x = model,
        analysis_root = paths$analysis_root,
        run_define = TRUE,
        run_loop = TRUE
      )

      analysis2_log("Generating final analytics...")
      ndm_source_runtime_results_get(
        analysis_root = paths$analysis_root,
        env = trained$env
      )
    }

    analysis2_log(sprintf("Done with outer iteration %s", outer_iteration))
  }

  invisible(TRUE)
}


analysis2_run_sim <- function(args = commandArgs(TRUE)) {
  options(error = NULL)
  analysis2_log("Starting Analysis2 SuperLModel_MasterSim.R")

  opts <- analysis2_parse_args(args)
  paths <- analysis2_paths(project_root = opts$project_root)
  setwd(paths$project_root)
  options(ndm.analysis_root = paths$analysis_root)

  ReSaveTfRecords <- isTRUE(opts$resave_tfrecords %||% FALSE)
  if (ReSaveTfRecords) {
    stop("Analysis2 does not regenerate TFRecords in this phase. Use existing TFRecords only.", call. = FALSE)
  }

  force2GPU <- TRUE
  nSimGridSeed <- 128L
  nExamplesPerCell <- 1L
  GPU_MEM_FRAC <- NULL
  endAppend <- TRUE
  analysis2_log(if (isTRUE(endAppend)) "Appending special tokens to end" else "Appending special tokens to start")

  outer_iterations <- analysis2_resolve_outer_iterations(opts, default = 1L)
  set.seed(theInitialSeed <- 453L)

  LEARNING_RATE_MAX <- 0.0002
  covariateType <- "sqrt"
  useLSTM <- FALSE
  quantizeX <- FALSE
  SimMode <- TRUE
  GLOBAL_ODE_NPOP <- 10000
  tryReuseJit <- FALSE
  rollCompute_window <- 52L
  nPolicies <- 1L
  nOutcomes <- 1L
  AppendTimeEmbeds <- FALSE
  AppendPlaceEmbeds <- FALSE

  nBatch <- as.integer(32L)
  nEpochesMax <- 6L
  nSamples_max <- 100000L
  nCheckpoints <- 10L
  FocalDisease <- 1L
  nPlaces <- 170L
  nDiseasesPretrain <- nTotalDiseases_pretrain <- 10L
  nSGD_DefiningLRSeq <- nSGD_model <- as.integer(round(nEpochesMax * (nSamples_max / nBatch)))
  AnalysisName <- opts$analysis_name %||% "BigSimsLatest"
  nSGD_pretrain <- 0L
  nSGD_posttrain <- nSGD_model
  AnalysisDate <- Sys.Date()
  if (ReSaveTfRecords) {
    nSGD_model <- nSGD_pretrain <- nSGD_posttrain <- 0L
  }

  HolderFolder <- file.path(paths$project_root, "SavedResults", "Sim", sprintf("Results_%s", AnalysisName))
  analysis2_dir_create(HolderFolder)

  simCovariates_sqrt <- c(
    "XPred_c_sqrt" = "inc_case_per_capita_sqrt",
    "XPred_h_sqrt" = "inc_hosp_per_capita_sqrt",
    "XPred_d_sqrt" = "inc_death_per_capita_sqrt"
  )
  simCovariates_log <- c(
    "XPred_c_log" = "inc_case_per_capita_log",
    "XPred_h_log" = "inc_hosp_per_capita_log",
    "XPred_d_log" = "inc_death_per_capita_log"
  )
  simCovariates_raw <- c(
    "XPred_c" = "inc_case_per_capita",
    "XPred_h" = "inc_hosp_per_capita",
    "XPred_d" = "inc_death_per_capita"
  )
  simCovariates <- c()
  if (grepl("log", covariateType, fixed = TRUE)) {
    simCovariates <- c(simCovariates, simCovariates_log)
  }
  if (grepl("sqrt", covariateType, fixed = TRUE)) {
    simCovariates <- c(simCovariates, simCovariates_sqrt)
  }
  if (grepl("raw", covariateType, fixed = TRUE)) {
    simCovariates <- c(simCovariates, simCovariates_raw)
  }

  SimGrid <- as.data.frame(data.table::fread(
    file.path(paths$project_root, "Data", "RunGrids", "SimGrids", sprintf("SimGrid_%s.csv", AnalysisName))
  ))
  nSimGrid <- nrow(SimGrid)

  SimGrid <- SimGrid[order(SimGrid$BaseID), ]
  if (SimGrid$BaseID[outer_iterations[[1]]] %% 2 == 0) {
    even_rows <- SimGrid$BaseID %% 2 == 0
    SimGrid[even_rows, ] <- SimGrid[even_rows, ][
      order(apply(SimGrid[even_rows, ], 1, function(zr) rlang::hash(paste(zr, collapse = "_")))),
    ]
  }

  if (isTRUE(opts$dry_run)) {
    analysis2_log(sprintf("Dry run. AnalysisName=%s; outer iterations=%s", AnalysisName, paste(outer_iterations, collapse = ",")))
    return(invisible(list(paths = paths, analysis_name = AnalysisName, outer_iterations = outer_iterations)))
  }

  for (outer_iteration in outer_iterations) {
    analysis2_log(sprintf("Starting outer iteration %s", outer_iteration))
    set.seed(SEED_ <- 100L + as.integer(outer_iteration))

    row_values <- analysis2_row_to_list(SimGrid[outer_iteration, , drop = FALSE])
    list2env(row_values, environment())
    BaseID <- row_values$BaseID
    model_tex_loc <- row_values$model_tex_loc
    floatType <- row_values$floatType
    ContextLength <- row_values$ContextLength

    modelingStrategyNameKey <- paste(c(
      "SimMode",
      paste(
        names(row_values),
        vapply(row_values, function(x) {
          numeric_value <- suppressWarnings(as.numeric(x))
          if (!is.na(numeric_value)) {
            as.character(round(numeric_value, 4))
          } else {
            as.character(x)
          }
        }, character(1)),
        sep = "_"
      )
    ), collapse = "__")

    TfRecordDir <- file.path(paths$project_root, "Data", "RunTFRecords", "SimTFRecords", AnalysisName)
    alt1_tfrecord_dir <- file.path("/mnt/share/local/AISystem/Data/RunTFRecords/SimTFRecords", AnalysisName)
    if (dir.exists(alt1_tfrecord_dir) && length(list.files(alt1_tfrecord_dir)) > 0L) {
      TfRecordDir <- alt1_tfrecord_dir
    }
    stopifnot(file.exists(file.path(TfRecordDir, sprintf("train_%s.tfrecord", BaseID))))

    ModelType <- analysis2_model_type(
      opts = opts,
      grid_model_type = row_values$ModelType,
      default = "DecoderOnly"
    )
    spec <- analysis2_resolve_model_spec(
      model_tex_loc = model_tex_loc,
      model_type = ModelType,
      project_root = paths$project_root
    )

    config <- ndm_create_config(
      model_type = ModelType,
      backbone = "transformer",
      analysis_root = paths$analysis_root,
      float_type = as.character(floatType),
      force_to_gpu = force2GPU,
      resave_tfrecords = FALSE,
      gpu_mem_frac = GPU_MEM_FRAC
    )
    runtime_env <- ndm_prepare_runtime(config = config)
    jnp <- runtime_env$jnp
    diffrax <- runtime_env$diffrax

    nTimesTotal <- (nTimesPast <- as.integer(ContextLength)) + (nTimesLookahead <- 4L * 3L)
    nTimesThres <- 10L
    VI_TotalTimesInLikelihood <- nTimesInLikelihood <- nTimesLookahead
    NTimeSteps_SIM <- (nTimesPast + nTimesLookahead) * 4L
    nTimesLookValidationInference <- 50L
    MaxSteps <- as.integer(10^6)
    VI_SaveAt_ODE_sim <- diffrax$SaveAt(ts = jnp$array(0L:(NTimeSteps_SIM - 1L)))
    VI_SaveAt_ODE_optim <- diffrax$SaveAt(ts = jnp$array(0L:(nTimesInLikelihood - 1L)))
    VI_diff_eq_solver_optim <- VI_diff_eq_solver_dgp <- diffrax$Tsit5()
    dt0_init_dgp <- 1e-3
    stepsize_controller_dgp <- diffrax$PIDController(rtol = 1e-6, atol = 1e-7)
    dt0_init_optim <- 1e0
    stepsize_controller_optim <- diffrax$ConstantStepSize()
    diffraxInterpolator <- diffrax$LinearInterpolation

    analysis2_sync_runtime_env(runtime_env, env = environment())
    ndm_prepare_data(
      runtime_env = runtime_env,
      analysis_root = paths$analysis_root,
      generator = "sim"
    )

    analysis2_assert_no_pretraining(nSGD_pretrain)
    if (nSGD_posttrain > 0L) {
      analysis2_log("Building core ML model...")
      model <- ndm_build_model(
        runtime_env = runtime_env,
        analysis_root = paths$analysis_root,
        model_type = ModelType,
        model_spec = spec,
        backbone = "transformer"
      )

      analysis2_log("Starting final training run...")
      nSGD_model <- nSGD_DefiningLRSeq <- nSGD_posttrain
      analysis2_sync_runtime_env(runtime_env, env = environment())
      trained <- ndm_train(
        x = model,
        analysis_root = paths$analysis_root,
        run_define = TRUE,
        run_loop = TRUE
      )

      analysis2_log("Generating final analytics...")
      ndm_source_runtime_results_get(
        analysis_root = paths$analysis_root,
        env = trained$env
      )
    }

    analysis2_log(sprintf("Done with outer iteration %s", outer_iteration))
  }

  invisible(TRUE)
}


#' Run the packaged Analysis2 orchestration scripts
#'
#' These wrappers expose the migrated Analysis2 entry points from within the R
#' package. They are primarily intended for script-style execution.
#'
#' @param args Character vector of command line style arguments, usually of the
#'   form `"--key=value"`.
#'
#' @returns The invisible result returned by the underlying Analysis2 driver.
#'
#' @examples
#' \dontrun{
#' ndm_run_analysis2_real(c("--dry_run=TRUE"))
#' ndm_run_analysis2_sim(c("--dry_run=TRUE"))
#' }
#'
#' @export
ndm_run_analysis2_real <- function(args = commandArgs(TRUE)) {
  analysis2_run_real(args = args)
}

#' @rdname ndm_run_analysis2_real
#' @export
ndm_run_analysis2_sim <- function(args = commandArgs(TRUE)) {
  analysis2_run_sim(args = args)
}
