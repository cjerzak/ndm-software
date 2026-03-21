#' Create package-native real-data run configurations
#'
#' These helpers capture the maintained package-only orchestration surface for
#' real, simulation, and multidisease runs. They replace the old expectation
#' that callers execute an external runtime checkout.
#'
#' @details
#' `dry_run = TRUE` resolves paths and selected rows without executing the
#' underlying workflow. Dry runs can use lightweight in-memory grids such as the
#' `BaseID` / `ModelType` preview shown in `README.md`.
#'
#' Non-dry execution expects Analysis2-compatible grids. The tested simulation
#' grids in this package are produced by `ndmdatasets::ndm_sim_build_grid()` and
#' then augmented with fields such as `paddingMethod`, `lookahead`,
#' `n_time_steps`, `n_inference_batches`, `scaling_batches`, and either
#' `model_spec_name` or `model_tex_loc`. The tested real and multidisease grids
#' include `BaseID`, `ContextLength`, `evaluationTime`, `initialTransform`,
#' `initialNormType`, `paddingMethod`, `OSSType`, `dataInputs`, `ModelType`,
#' `ModelDepth`, `ModelDims`, `nSamplesTrain`, `nObsInference`, `floatType`,
#' and either `model_spec_name` or `model_tex_loc`.
#'
#' Non-dry runner workflows also require the full execution stack: the
#' `ndmdatasets` package, the helper packages used by the runtime, and a backend
#' environment aligned with `NDM_SOFTWARE_CONDA_ENV` (or the platform-specific
#' fallback used by the runner layer).
#'
#' Distributed simulation TFRecord bootstrap is handled separately via
#' [ndm_bootstrap_sim_tfrecords()]. Unlike `ndm_run_sim(..., outer = ...)`, the
#' bootstrap entrypoint schedules work by canonical `BaseID` in raw grid order
#' and is the supported surface for parallel TFRecord materialization.
#'
#' @param project_root Working project directory used for grids, TFRecords, and
#'   outputs.
#' @param analysis_name Analysis label threaded through output paths.
#' @param grid Optional in-memory grid data frame. When supplied, `grid_file`
#'   must be `NULL`.
#' @param grid_file Optional CSV path for the run grid.
#' @param outer Integer vector of outer-iteration rows to execute.
#' @param model_type Optional model family override.
#' @param respect_grid_model_type Logical scalar indicating whether grid rows
#'   should be allowed to override the requested model type.
#' @param resave_tfrecords Logical scalar controlling TFRecord regeneration.
#'   Supported for real and simulation workflows. Multidisease workflows reject
#'   `TRUE` because the legacy regeneration path has been retired. When
#'   selected rows share a `BaseID`, regeneration writes one canonical TFRecord
#'   pair per `BaseID` using the largest `nSamplesTrain` among those rows.
#' @param tfrecord_dir Optional TFRecord output directory.
#' @param raw_data_dir Real-data input directory.
#' @param outcome_metric Outcome metric name for real-data or multidisease runs.
#' @param data_subset Optional real-data subset name.
#' @param disease_names Character vector of multidisease names.
#' @param data_format Multidisease data format.
#' @param dry_run Logical scalar indicating whether the run should stop after
#'   resolving paths and grid rows.
#'
#' @returns `ndm_create_*_run_config()` returns a classed list describing the
#'   requested workflow. `ndm_run_real()` and `ndm_run_sim()` return the
#'   underlying workflow result; with `resave_tfrecords = TRUE` they return a
#'   list containing the written `BaseID` values, TFRecord directory, and a
#'   canonical write-plan summary.
#'
#' @examples
#' \dontrun{
#' preview_grid <- data.frame(
#'   BaseID = c(1L, 2L),
#'   ModelType = c("DecoderOnly", "NeuralODE"),
#'   stringsAsFactors = FALSE
#' )
#'
#' cfg <- ndm_create_sim_run_config(
#'   project_root = tempdir(),
#'   grid = preview_grid,
#'   outer = 1:2,
#'   dry_run = TRUE
#' )
#'
#' ndm_run_sim(cfg)
#' }
#'
#' @export
ndm_create_real_run_config <- function(project_root = getwd(),
                                       analysis_name = "RealApril15",
                                       grid = NULL,
                                       grid_file = file.path("Data", "RunGrids", "RealGrids", sprintf("RealGrid_%s.csv", analysis_name)),
                                       outer = 3L,
                                       model_type = NULL,
                                       respect_grid_model_type = FALSE,
                                       resave_tfrecords = FALSE,
                                       tfrecord_dir = file.path("Data", "RunTFRecords", "RealTFRecords", analysis_name),
                                       raw_data_dir = file.path("Data", "MainData"),
                                       outcome_metric = "inc_death",
                                       data_subset = "high_income",
                                       dry_run = FALSE) {
  if (!is.null(grid)) {
    grid_file <- NULL
  }
  .ndm_make_run_config(
    mode = "real",
    project_root = project_root,
    analysis_name = analysis_name,
    grid = grid,
    grid_file = grid_file,
    outer = outer,
    model_type = model_type,
    respect_grid_model_type = respect_grid_model_type,
    resave_tfrecords = resave_tfrecords,
    tfrecord_dir = tfrecord_dir,
    raw_data_dir = raw_data_dir,
    outcome_metric = outcome_metric,
    data_subset = data_subset,
    dry_run = dry_run
  )
}

#' @rdname ndm_create_real_run_config
#' @export
ndm_create_sim_run_config <- function(project_root = getwd(),
                                      analysis_name = "BigSimsLatest",
                                      grid = NULL,
                                      grid_file = file.path("Data", "RunGrids", "SimGrids", sprintf("SimGrid_%s.csv", analysis_name)),
                                      outer = 1L,
                                      model_type = NULL,
                                      respect_grid_model_type = FALSE,
                                      resave_tfrecords = TRUE,
                                      tfrecord_dir = file.path("Data", "RunTFRecords", "SimTFRecords", analysis_name),
                                      dry_run = FALSE) {
  if (!is.null(grid)) {
    grid_file <- NULL
  }
  .ndm_make_run_config(
    mode = "sim",
    project_root = project_root,
    analysis_name = analysis_name,
    grid = grid,
    grid_file = grid_file,
    outer = outer,
    model_type = model_type,
    respect_grid_model_type = respect_grid_model_type,
    resave_tfrecords = resave_tfrecords,
    tfrecord_dir = tfrecord_dir,
    dry_run = dry_run
  )
}

#' @rdname ndm_create_real_run_config
#' @export
ndm_create_multidisease_run_config <- function(project_root = getwd(),
                                               analysis_name = "RealLatest",
                                               grid = NULL,
                                               grid_file = file.path("Data", "RunGrids", "RealGrids", sprintf("RealGrid_%s.csv", analysis_name)),
                                               outer = 1L,
                                               model_type = NULL,
                                               respect_grid_model_type = FALSE,
                                               resave_tfrecords = FALSE,
                                               tfrecord_dir = file.path("Data", "RunTFRecords", "RealTFRecords", analysis_name),
                                               outcome_metric = "CountValue",
                                               data_subset = "all",
                                               disease_names = c("Covid", "Flu"),
                                               data_format = "IHME",
                                               dry_run = FALSE) {
  if (!is.null(grid)) {
    grid_file <- NULL
  }
  .ndm_make_run_config(
    mode = "multidisease",
    project_root = project_root,
    analysis_name = analysis_name,
    grid = grid,
    grid_file = grid_file,
    outer = outer,
    model_type = model_type,
    respect_grid_model_type = respect_grid_model_type,
    resave_tfrecords = resave_tfrecords,
    tfrecord_dir = tfrecord_dir,
    outcome_metric = outcome_metric,
    data_subset = data_subset,
    disease_names = disease_names,
    data_format = data_format,
    dry_run = dry_run
  )
}

#' Bootstrap canonical simulation TFRecords by BaseID
#'
#' This helper materializes canonical simulation TFRecords without going through
#' the legacy `outer` scheduling path. It reads the simulation grid in raw CSV
#' or data-frame order, builds one canonical write plan row per `BaseID`, and
#' optionally writes those TFRecords with a per-`BaseID` cross-process lock.
#'
#' @param project_root Working project directory used to resolve relative grid
#'   and TFRecord paths.
#' @param analysis_name Analysis label used to derive default grid and TFRecord
#'   paths.
#' @param grid Optional in-memory simulation grid. When supplied, `grid_file`
#'   must be `NULL`.
#' @param grid_file Optional CSV path for the simulation grid.
#' @param base_ids Optional integer vector of canonical `BaseID` values to
#'   materialize. When `NULL`, bootstrap all `BaseID`s in raw grid order of
#'   first appearance.
#' @param tfrecord_dir Output directory for canonical TFRecords.
#' @param overwrite Logical scalar controlling whether existing canonical
#'   TFRecords should be regenerated.
#' @param dry_run Logical scalar indicating whether to return the canonical
#'   write plan without writing TFRecords.
#'
#' @returns A data frame with one row per canonical `BaseID` and columns
#'   `BaseID`, `selected_rows`, `canonical_row`,
#'   `artifact_n_samples_train`, `train_file`, `inference_file`, and `status`.
#'
#' @export
ndm_bootstrap_sim_tfrecords <- function(project_root = getwd(),
                                        analysis_name = "BigSimsLatest",
                                        grid = NULL,
                                        grid_file = file.path("Data", "RunGrids", "SimGrids", sprintf("SimGrid_%s.csv", analysis_name)),
                                        base_ids = NULL,
                                        tfrecord_dir = file.path("Data", "RunTFRecords", "SimTFRecords", analysis_name),
                                        overwrite = FALSE,
                                        dry_run = FALSE) {
  project_root <- .ndm_normalize_path(project_root, must_work = TRUE)

  if (!is.null(grid)) {
    grid_file <- NULL
  }

  if (!is.null(grid) && !is.null(grid_file)) {
    stop("Supply either `grid` or `grid_file`, not both.", call. = FALSE)
  }
  if (!is.null(grid) && !is.data.frame(grid)) {
    stop("`grid` must be a data.frame when supplied.", call. = FALSE)
  }

  if (!is.null(grid)) {
    grid_file <- NULL
    sim_grid <- as.data.frame(grid, stringsAsFactors = FALSE)
  } else {
    if (!startsWith(grid_file, "/")) {
      grid_file <- file.path(project_root, grid_file)
    }
    grid_file <- .ndm_normalize_path(grid_file, must_work = TRUE)
    sim_grid <- as.data.frame(data.table::fread(grid_file), stringsAsFactors = FALSE)
  }

  if (!startsWith(tfrecord_dir, "/")) {
    tfrecord_dir <- file.path(project_root, tfrecord_dir)
  }
  tfrecord_dir <- .ndm_normalize_path(tfrecord_dir, must_work = FALSE)

  api_env <- .ndm_legacy_run_env()
  bootstrap_fun <- get("analysis2_bootstrap_sim_tfrecords", envir = api_env, inherits = FALSE)
  bootstrap_fun(
    project_root = project_root,
    analysis_name = as.character(analysis_name),
    grid = sim_grid,
    base_ids = base_ids,
    tfrecord_dir = tfrecord_dir,
    overwrite = isTRUE(overwrite),
    dry_run = isTRUE(dry_run)
  )
}

.ndm_validate_resave_tfrecords <- function(mode, resave_tfrecords) {
  if (identical(mode, "multidisease") && isTRUE(resave_tfrecords)) {
    stop(
      "`resave_tfrecords = TRUE` is no longer supported for multidisease workflows.",
      call. = FALSE
    )
  }

  invisible(TRUE)
}

.ndm_run_env_get <- function(env, name) {
  get(name, envir = env, inherits = FALSE)
}

.ndm_make_run_config <- function(mode,
                                 project_root,
                                 analysis_name,
                                 grid = NULL,
                                 grid_file = NULL,
                                 outer,
                                 model_type = NULL,
                                 respect_grid_model_type = FALSE,
                                 resave_tfrecords = FALSE,
                                 tfrecord_dir = NULL,
                                 raw_data_dir = NULL,
                                 outcome_metric = NULL,
                                 data_subset = NULL,
                                 disease_names = NULL,
                                 data_format = NULL,
                                 dry_run = FALSE) {
  project_root <- .ndm_normalize_path(project_root, must_work = TRUE)

  if (!is.null(grid) && !is.null(grid_file)) {
    stop("Supply either `grid` or `grid_file`, not both.", call. = FALSE)
  }
  if (!is.null(grid) && !is.data.frame(grid)) {
    stop("`grid` must be a data.frame when supplied.", call. = FALSE)
  }

  outer <- as.integer(outer)
  if (length(outer) == 0L || anyNA(outer)) {
    stop("`outer` must contain at least one integer row index.", call. = FALSE)
  }
  .ndm_validate_resave_tfrecords(mode, resave_tfrecords)

  class_name <- switch(
    mode,
    real = "ndm_real_run_config",
    sim = "ndm_sim_run_config",
    multidisease = "ndm_multidisease_run_config",
    stop("Unsupported run mode: ", mode, call. = FALSE)
  )

  .ndm_make_classed_list(
    list(
      mode = mode,
      project_root = project_root,
      analysis_name = as.character(analysis_name),
      grid = grid,
      grid_file = grid_file,
      outer = outer,
      model_type = model_type,
      respect_grid_model_type = isTRUE(respect_grid_model_type),
      resave_tfrecords = isTRUE(resave_tfrecords),
      tfrecord_dir = tfrecord_dir,
      raw_data_dir = raw_data_dir,
      outcome_metric = outcome_metric,
      data_subset = data_subset,
      disease_names = disease_names,
      data_format = data_format,
      dry_run = isTRUE(dry_run)
    ),
    class_name
  )
}

.ndm_run_mode_fun_name <- function(mode) {
  switch(
    mode,
    real = "analysis2_run_real",
    sim = "analysis2_run_sim",
    multidisease = "analysis2_run_real_multidisease",
    stop("Unsupported run mode: ", mode, call. = FALSE)
  )
}

.ndm_run_config_to_args <- function(config) {
  stopifnot(inherits(config, "list"))

  grid_file <- config$grid_file
  if (!is.null(config$grid)) {
    grid_file <- tempfile(sprintf("ndm-%s-grid-", config$mode), fileext = ".csv")
    utils::write.csv(config$grid, file = grid_file, row.names = FALSE)
  }

  args <- c(
    sprintf("--project_root=%s", config$project_root),
    sprintf("--analysis_name=%s", config$analysis_name),
    sprintf("--outer=%s", paste(config$outer, collapse = ",")),
    sprintf("--respect_grid_model_type=%s", toupper(as.character(isTRUE(config$respect_grid_model_type)))),
    sprintf("--resave_tfrecords=%s", toupper(as.character(isTRUE(config$resave_tfrecords)))),
    "--run_figures=FALSE",
    sprintf("--dry_run=%s", toupper(as.character(isTRUE(config$dry_run))))
  )

  if (!is.null(grid_file)) {
    if (!startsWith(grid_file, "/")) {
      grid_file <- file.path(config$project_root, grid_file)
    }
    args <- c(args, sprintf("--grid_file=%s", grid_file))
  }
  if (!is.null(config$model_type) && nzchar(config$model_type)) {
    args <- c(args, sprintf("--model_type=%s", config$model_type))
  }
  if (!is.null(config$tfrecord_dir) && nzchar(config$tfrecord_dir)) {
    tfrecord_dir <- config$tfrecord_dir
    if (!startsWith(tfrecord_dir, "/")) {
      tfrecord_dir <- file.path(config$project_root, tfrecord_dir)
    }
    args <- c(args, sprintf("--tfrecord_dir=%s", tfrecord_dir))
  }
  if (!is.null(config$raw_data_dir) && nzchar(config$raw_data_dir)) {
    raw_data_dir <- config$raw_data_dir
    if (!startsWith(raw_data_dir, "/")) {
      raw_data_dir <- file.path(config$project_root, raw_data_dir)
    }
    args <- c(args, sprintf("--raw_data_dir=%s", raw_data_dir))
  }
  if (!is.null(config$outcome_metric) && nzchar(config$outcome_metric)) {
    args <- c(args, sprintf("--outcome_metric=%s", config$outcome_metric))
  }
  if (!is.null(config$data_subset) && nzchar(config$data_subset)) {
    args <- c(args, sprintf("--data_subset=%s", config$data_subset))
  }
  if (!is.null(config$data_format) && nzchar(config$data_format)) {
    args <- c(args, sprintf("--data_format=%s", config$data_format))
  }
  if (!is.null(config$disease_names) && length(config$disease_names) > 0L) {
    args <- c(args, sprintf("--disease_names=%s", paste(config$disease_names, collapse = ",")))
  }

  args
}

.ndm_disable_legacy_run_manifests <- function(env) {
  stopifnot(is.environment(env))

  # The maintained `ndm_run_*()` entrypoints consume structured config objects
  # directly and no longer require the legacy YAML manifest layer.
  require_yaml <- function() invisible("yaml")
  assign("analysis2_require_yaml", require_yaml, envir = env)

  resolve_config_file <- function(mode, opts, analysis_root, project_root) {
    normalize_string <- .ndm_run_env_get(env, "analysis2_normalize_string")
    path_from_project <- .ndm_run_env_get(env, "analysis2_path_from_project")
    explicit_path <- normalize_string(opts$config)
    if (is.null(explicit_path)) {
      return(NULL)
    }
    path_from_project(explicit_path, project_root = project_root, must_work = FALSE)
  }
  assign("analysis2_resolve_config_file", resolve_config_file, envir = env)

  invisible(env)
}

.ndm_filter_formal_args <- function(fun, args) {
  formal_names <- names(formals(fun))
  if (is.null(formal_names) || "..." %in% formal_names) {
    return(args)
  }

  arg_names <- names(args)
  if (is.null(arg_names)) {
    arg_names <- rep("", length(args))
  }

  keep <- !nzchar(arg_names) | arg_names %in% formal_names
  args[keep]
}

.ndm_override_legacy_model_tex_loc <- function(env) {
  resolve_model_tex_loc <- function(row_values) {
    normalize_string <- .ndm_run_env_get(env, "analysis2_normalize_string")
    model_spec_name <- .ndm_run_env_get(env, "analysis2_model_spec_name")

    legacy_path <- normalize_string(row_values$model_tex_loc)
    if (!is.null(legacy_path)) {
      return(legacy_path)
    }

    spec_name <- model_spec_name(
      model_spec_name = row_values$model_spec_name,
      model_tex_loc = row_values$model_tex_loc
    )
    if (is.null(spec_name)) {
      return(NULL)
    }

    packaged_tex_path <- system.file(
      "extdata",
      "model_specs",
      sprintf("%s.tex", spec_name),
      package = "ndm"
    )
    if (nzchar(packaged_tex_path) && file.exists(packaged_tex_path)) {
      return(packaged_tex_path)
    }

    file.path("./Analysis2/ModelStructureTex", sprintf("%s.tex", spec_name))
  }
  assign("analysis2_resolve_model_tex_loc", resolve_model_tex_loc, envir = env)

  invisible(env)
}

.ndm_override_legacy_multidisease_runner <- function(env) {
  run_real_multidisease <- function(args = commandArgs(TRUE)) {
    analysis2_log <- .ndm_run_env_get(env, "analysis2_log")
    analysis2_build_run_spec <- .ndm_run_env_get(env, "analysis2_build_run_spec")
    analysis2_print_usage <- .ndm_run_env_get(env, "analysis2_print_usage")
    analysis2_order_grid <- .ndm_run_env_get(env, "analysis2_order_grid")
    analysis2_validate_outer_iterations <- .ndm_run_env_get(env, "analysis2_validate_outer_iterations")
    analysis2_dry_run_result <- .ndm_run_env_get(env, "analysis2_dry_run_result")
    analysis2_prepare_output_roots <- .ndm_run_env_get(env, "analysis2_prepare_output_roots")
    analysis2_dir_create <- .ndm_run_env_get(env, "analysis2_dir_create")
    analysis2_as_int <- .ndm_run_env_get(env, "analysis2_as_int")
    analysis2_small_run_n_checkpoints <- .ndm_run_env_get(env, "analysis2_small_run_n_checkpoints")
    analysis2_small_run_n_obs_inference <- .ndm_run_env_get(env, "analysis2_small_run_n_obs_inference")
    analysis2_model_type <- .ndm_run_env_get(env, "analysis2_model_type")

    options(error = NULL)
    analysis2_log("Starting package-native multidisease runner")
    spec <- analysis2_build_run_spec("multidisease", args)
    if (isTRUE(spec$help)) {
      return(analysis2_print_usage("multidisease", paths = spec$paths))
    }

    paths <- spec$paths
    grid_file <- normalizePath(spec$grid_file, winslash = "/", mustWork = TRUE)
    real_grid_raw <- as.data.frame(data.table::fread(grid_file), stringsAsFactors = FALSE)
    nsgd_resolver <- utils::getFromNamespace(".ndm_resolve_nsgd_calibration", "ndm")
    nsgd_formatter <- utils::getFromNamespace(".ndm_nsgd_calibration_message", "ndm")
    nsgd_calibration <- tryCatch(
      nsgd_resolver(
        mode = "multidisease",
        project_root = spec$project_root,
        analysis_name = spec$analysis_name,
        n_epoches_max = 9L,
        grid = real_grid_raw,
        grid_file = grid_file
      ),
      error = function(e) {
        missing_anchor_msg <- "Unable to resolve an nSGD calibration anchor because no candidate grid with `nSamplesTrain` was found"
        if (isTRUE(spec$dry_run) && grepl(missing_anchor_msg, conditionMessage(e), fixed = TRUE)) {
          return(NULL)
        }
        stop(e)
      }
    )
    real_grid <- analysis2_order_grid(real_grid_raw, spec$outer)
    analysis2_validate_outer_iterations(real_grid, spec$outer, grid_file)

    if (isTRUE(spec$dry_run)) {
      return(analysis2_dry_run_result(spec, real_grid, nsgd_calibration = nsgd_calibration))
    }

    setwd(paths$project_root)
    analysis2_prepare_output_roots(paths$project_root, sim_mode = FALSE)
    analysis2_log(nsgd_formatter("multidisease", nsgd_calibration))
    holder_folder <- file.path(paths$project_root, "SavedResults", "Real", sprintf("Results_%s", spec$analysis_name))
    analysis2_dir_create(holder_folder)

    driver_env <- new.env(parent = globalenv())
    driver_env$analysis2_as_int <- analysis2_as_int
    driver_env$analysis2_small_run_n_checkpoints <- analysis2_small_run_n_checkpoints
    driver_env$analysis2_small_run_n_obs_inference <- analysis2_small_run_n_obs_inference
    driver_env$analysis2_model_type <- analysis2_model_type
    driver_env$analysis2_multidisease_spec <- spec
    driver_env$analysis2_multidisease_grid <- real_grid
    driver_env$analysis2_nsgd_calibration <- nsgd_calibration
    source(
      file.path(paths$analysis_root, "SetupEnv", "Analysis2_legacy_multidisease_driver.R"),
      local = driver_env,
      chdir = FALSE
    )

    invisible(get0("analysis2_multidisease_result", envir = driver_env, inherits = FALSE, ifnotfound = TRUE))
  }
  assign("analysis2_run_real_multidisease", run_real_multidisease, envir = env)

  invisible(env)
}

.ndm_configure_legacy_run_env <- function(env) {
  stopifnot(is.environment(env))

  assign("analysis2_internal_analysis_root", function() .ndm_internal_analysis_root(), envir = env)
  assign(
    "analysis2_call",
    function(pkg, name, ...) {
      fun <- getExportedValue(pkg, name)
      args <- list(...)
      do.call(fun, .ndm_filter_formal_args(fun, args))
    },
    envir = env
  )

  .ndm_disable_legacy_run_manifests(env)
  .ndm_override_legacy_model_tex_loc(env)
  .ndm_override_legacy_multidisease_runner(env)
  env
}

.ndm_legacy_run_env <- local({
  cache <- NULL

  function(refresh = FALSE) {
    if (isTRUE(refresh) || is.null(cache)) {
      cache <<- .ndm_configure_legacy_run_env(.ndm_new_run_impl_env())
    }
    cache
  }
})

.ndm_call_analysis2_runner <- function(mode, config) {
  api_env <- .ndm_legacy_run_env()
  run_fun <- get(.ndm_run_mode_fun_name(mode), envir = api_env, inherits = FALSE)
  run_fun(.ndm_run_config_to_args(config))
}

.ndm_invoke_legacy_analysis2_runner <- function(mode, args = commandArgs(TRUE)) {
  api_env <- .ndm_legacy_run_env()
  run_fun <- get(.ndm_run_mode_fun_name(mode), envir = api_env, inherits = FALSE)
  run_fun(args)
}

#' @param config A package-native run configuration created by the matching
#'   `ndm_create_*_run_config()` helper.
#'
#' @rdname ndm_create_real_run_config
#' @returns The underlying workflow result, or a dry-run preview when
#'   `config$dry_run` is `TRUE`.
#' @export
ndm_run_real <- function(config = ndm_create_real_run_config()) {
  if (!inherits(config, "ndm_real_run_config")) {
    stop("`config` must inherit from class 'ndm_real_run_config'.", call. = FALSE)
  }
  .ndm_call_analysis2_runner("real", config)
}

#' @rdname ndm_create_real_run_config
#' @export
ndm_run_sim <- function(config = ndm_create_sim_run_config()) {
  if (!inherits(config, "ndm_sim_run_config")) {
    stop("`config` must inherit from class 'ndm_sim_run_config'.", call. = FALSE)
  }
  .ndm_call_analysis2_runner("sim", config)
}

#' @rdname ndm_create_real_run_config
#' @export
ndm_run_multidisease <- function(config = ndm_create_multidisease_run_config()) {
  if (!inherits(config, "ndm_multidisease_run_config")) {
    stop("`config` must inherit from class 'ndm_multidisease_run_config'.", call. = FALSE)
  }
  .ndm_validate_resave_tfrecords("multidisease", config$resave_tfrecords)
  .ndm_call_analysis2_runner("multidisease", config)
}
