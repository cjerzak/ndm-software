# Package-owned Analysis2 orchestration used by `ndm`'s maintained package
# workflows.

`%||%` <- function(x, y) {
  if (is.null(x)) {
    return(y)
  }
  x
}

analysis2_internal_analysis_root <- function() {
  analysis_root <- get0("NDM_PACKAGE_ANALYSIS_ROOT", inherits = TRUE, ifnotfound = NULL)
  if (is.null(analysis_root) || !nzchar(analysis_root)) {
    stop("`NDM_PACKAGE_ANALYSIS_ROOT` must be defined before sourcing Analysis2_api.R.", call. = FALSE)
  }
  normalizePath(analysis_root, winslash = "/", mustWork = TRUE)
}

analysis2_log <- function(text) {
  message(sprintf("[%s] %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), text))
  invisible(text)
}

analysis2_f2n <- function(x) {
  as.numeric(as.character(x))
}

analysis2_as_int <- function(x) {
  as.integer(round(analysis2_f2n(x)))
}

analysis2_small_run_n_checkpoints <- function(n_samples_train, n_sgd, default = 10L) {
  n_samples <- max(1L, analysis2_as_int(n_samples_train))
  max_sgd <- max(1L, analysis2_as_int(n_sgd))
  if (n_samples < 32L) {
    return(max(1L, min(3L, max_sgd)))
  }
  max(1L, min(analysis2_as_int(default), max_sgd))
}

analysis2_small_run_n_obs_inference <- function(n_samples_train,
                                                n_batch,
                                                configured = NULL,
                                                default = 1024L) {
  if (!is.null(configured) && !is.na(configured) && configured > 0L) {
    return(analysis2_as_int(configured))
  }
  n_samples <- max(1L, analysis2_as_int(n_samples_train))
  if (n_samples < 32L) {
    return(max(32L, as.integer(8L * max(1L, analysis2_as_int(n_batch)))))
  }
  analysis2_as_int(default)
}

analysis2_resolve_nsgd_calibration <- function(mode,
                                               spec,
                                               grid,
                                               n_epoches_max,
                                               fallback_n_samples_train = NULL) {
  resolver <- utils::getFromNamespace(".ndm_resolve_nsgd_calibration", "ndm")
  tryCatch(
    resolver(
      mode = mode,
      project_root = spec$project_root,
      analysis_name = spec$analysis_name,
      n_epoches_max = n_epoches_max,
      grid = grid,
      grid_file = spec$grid_file,
      fallback_n_samples_train = fallback_n_samples_train
    ),
    error = function(e) {
      missing_anchor_msg <- "Unable to resolve an nSGD calibration anchor because no candidate grid with `nSamplesTrain` was found"
      if (isTRUE(spec$dry_run) && grepl(missing_anchor_msg, conditionMessage(e), fixed = TRUE)) {
        return(NULL)
      }
      stop(e)
    }
  )
}

analysis2_nsgd_calibration_globals <- function(calibration) {
  globals_fun <- utils::getFromNamespace(".ndm_nsgd_calibration_globals", "ndm")
  globals_fun(calibration)
}

analysis2_log_nsgd_calibration <- function(mode,
                                           calibration) {
  formatter <- utils::getFromNamespace(".ndm_nsgd_calibration_message", "ndm")
  analysis2_log(formatter(mode, calibration))
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
    key <- gsub("-", "_", pieces[[1]])
    value <- if (length(pieces) > 1L) {
      paste(pieces[-1], collapse = "=")
    } else {
      "TRUE"
    }
    out[[key]] <- value
  }

  for (name in c("dry_run", "run_figures", "respect_grid_model_type", "resave_tfrecords", "help")) {
    if (!is.null(out[[name]])) {
      out[[name]] <- analysis2_parse_bool(out[[name]])
    }
  }

  out
}

analysis2_normalize_string <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  value <- trimws(as.character(x[[1]]))
  if (!nzchar(value) || tolower(value) %in% c("null", "na")) {
    return(NULL)
  }
  value
}

analysis2_parse_csv <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  if (is.list(x)) {
    x <- unlist(x, recursive = TRUE, use.names = FALSE)
  }
  if (length(x) > 1L) {
    out <- trimws(as.character(x))
  } else {
    out <- trimws(strsplit(as.character(x), ",", fixed = TRUE)[[1]])
  }
  out[nzchar(out)]
}

analysis2_as_flag <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  analysis2_parse_bool(x)
}

analysis2_supported_flags <- function(mode) {
  base <- c(
    "config",
    "project_root",
    "analysis_name",
    "grid_file",
    "outer",
    "run_seed",
    "model_type",
    "respect_grid_model_type",
    "resave_tfrecords",
    "run_figures",
    "tfrecord_dir",
    "dry_run",
    "help"
  )

  extras <- switch(
    mode,
    real = c("raw_data_dir", "outcome_metric", "data_subset"),
    sim = character(),
    multidisease = c("outcome_metric", "data_subset", "disease_names", "data_format"),
    stop("Unsupported Analysis2 mode: ", mode, call. = FALSE)
  )

  unique(c(base, extras))
}

analysis2_validate_args <- function(opts, mode) {
  known <- analysis2_supported_flags(mode)
  present <- setdiff(names(opts), "positional")
  unknown <- setdiff(present, known)
  if (length(unknown) == 0L) {
    return(invisible(opts))
  }

  stop(
    "Unknown option(s) for Analysis2 ", mode, " runs: ",
    paste(sprintf("--%s", gsub("_", "-", unknown)), collapse = ", "),
    ". Use `--help` to inspect the supported interface.",
    call. = FALSE
  )
}

analysis2_validate_manifest <- function(manifest, mode) {
  if (length(manifest) == 0L) {
    return(invisible(manifest))
  }

  known <- names(analysis2_mode_defaults(mode))
  unknown <- setdiff(names(manifest), c(known, "config_file", "paths"))
  if (length(unknown) == 0L) {
    return(invisible(manifest))
  }

  stop(
    "Unknown config field(s) in the Analysis2 ", mode, " manifest: ",
    paste(unknown, collapse = ", "),
    ".",
    call. = FALSE
  )
}

analysis2_default_config_name <- function(mode) {
  switch(
    mode,
    real = "real.yaml",
    sim = "sim.yaml",
    multidisease = "real_multidisease.yaml",
    stop("Unsupported Analysis2 mode: ", mode, call. = FALSE)
  )
}

analysis2_require_yaml <- function() {
  if (!requireNamespace("yaml", quietly = TRUE)) {
    stop("The `yaml` package is required for Analysis2 config manifests.", call. = FALSE)
  }
  "yaml"
}

analysis2_load_yaml_config <- function(path) {
  if (is.null(path) || !file.exists(path)) {
    return(list())
  }
  if (!requireNamespace("yaml", quietly = TRUE)) {
    stop(
      "The `yaml` package is required when `config` points to a YAML manifest.",
      call. = FALSE
    )
  }
  config <- yaml::read_yaml(path)
  if (is.null(config)) {
    return(list())
  }
  as.list(config)
}

analysis2_mode_defaults <- function(mode) {
  switch(
    mode,
    real = list(
      mode = "real",
      analysis_name = "RealApril15",
      grid_file = NULL,
      outer = 3L,
      run_seed = NULL,
      model_type = NULL,
      respect_grid_model_type = FALSE,
      resave_tfrecords = FALSE,
      run_figures = FALSE,
      project_root = NULL,
      raw_data_dir = "Data/MainData",
      tfrecord_dir = NULL,
      outcome_metric = "inc_death",
      data_subset = "high_income",
      disease_names = NULL,
      data_format = NULL,
      dry_run = FALSE,
      help = FALSE
    ),
    sim = list(
      mode = "sim",
      analysis_name = "BigSimsLatest",
      grid_file = NULL,
      outer = 1L,
      run_seed = NULL,
      model_type = NULL,
      respect_grid_model_type = FALSE,
      resave_tfrecords = TRUE,
      run_figures = FALSE,
      project_root = NULL,
      raw_data_dir = NULL,
      tfrecord_dir = NULL,
      outcome_metric = NULL,
      data_subset = NULL,
      disease_names = NULL,
      data_format = NULL,
      dry_run = FALSE,
      help = FALSE
    ),
    multidisease = list(
      mode = "multidisease",
      analysis_name = "RealLatest",
      grid_file = NULL,
      outer = 1L,
      run_seed = NULL,
      model_type = NULL,
      respect_grid_model_type = FALSE,
      resave_tfrecords = FALSE,
      run_figures = FALSE,
      project_root = NULL,
      raw_data_dir = NULL,
      tfrecord_dir = NULL,
      outcome_metric = "CountValue",
      data_subset = "all",
      disease_names = c("Covid", "Flu"),
      data_format = "IHME",
      dry_run = FALSE,
      help = FALSE
    ),
    stop("Unsupported Analysis2 mode: ", mode, call. = FALSE)
  )
}

analysis2_mode_default_grid_file <- function(mode, project_root, analysis_name) {
  path <- switch(
    mode,
    real = file.path(project_root, "Data", "RunGrids", "RealGrids", sprintf("RealGrid_%s.csv", analysis_name)),
    sim = file.path(project_root, "Data", "RunGrids", "SimGrids", sprintf("SimGrid_%s.csv", analysis_name)),
    multidisease = file.path(project_root, "Data", "RunGrids", "RealGrids", sprintf("RealGrid_%s.csv", analysis_name)),
    stop("Unsupported Analysis2 mode: ", mode, call. = FALSE)
  )
  normalizePath(path, winslash = "/", mustWork = FALSE)
}

analysis2_mode_default_tfrecord_dir <- function(mode, project_root, analysis_name) {
  path <- switch(
    mode,
    real = file.path(project_root, "Data", "RunTFRecords", "RealTFRecords", analysis_name),
    sim = file.path(project_root, "Data", "RunTFRecords", "SimTFRecords", analysis_name),
    multidisease = file.path(project_root, "Data", "RunTFRecords", "RealTFRecords", analysis_name),
    stop("Unsupported Analysis2 mode: ", mode, call. = FALSE)
  )
  normalizePath(path, winslash = "/", mustWork = FALSE)
}

analysis2_path_from_project <- function(path, project_root, must_work = FALSE) {
  value <- analysis2_normalize_string(path)
  if (is.null(value)) {
    return(NULL)
  }

  if (startsWith(value, "~")) {
    return(normalizePath(path.expand(value), winslash = "/", mustWork = must_work))
  }
  is_absolute <- grepl("^(/|[A-Za-z]:[/\\\\]|[/\\\\]{2})", value)
  if (!is_absolute) {
    value <- file.path(project_root, value)
  }
  normalizePath(value, winslash = "/", mustWork = must_work)
}

analysis2_resolve_config_file <- function(mode, opts, analysis_root, project_root) {
  explicit_path <- analysis2_normalize_string(opts$config)

  if (is.null(explicit_path)) {
    return(NULL)
  }

  resolved <- analysis2_path_from_project(explicit_path, project_root = project_root, must_work = FALSE)
  if (!file.exists(resolved)) {
    stop("Config file not found: ", resolved, call. = FALSE)
  }
  resolved
}

analysis2_cli_overrides <- function(opts, mode) {
  overrides <- list()

  scalar_fields <- intersect(
    c(
      "project_root",
      "analysis_name",
      "grid_file",
      "model_type",
      "raw_data_dir",
      "tfrecord_dir",
      "outcome_metric",
      "data_subset",
      "data_format"
    ),
    names(opts)
  )

  for (field in scalar_fields) {
    overrides[[field]] <- analysis2_normalize_string(opts[[field]])
  }

  if ("run_seed" %in% names(opts)) {
    overrides$run_seed <- opts$run_seed
  }

  if (!is.null(opts$outer) || length(opts$positional) > 0L) {
    overrides$outer <- analysis2_resolve_outer_iterations(opts, default = integer())
  }

  if ("disease_names" %in% names(opts)) {
    overrides$disease_names <- analysis2_parse_csv(opts$disease_names)
  }

  for (field in c("respect_grid_model_type", "resave_tfrecords", "run_figures", "dry_run", "help")) {
    if (!is.null(opts[[field]])) {
      overrides[[field]] <- analysis2_as_flag(opts[[field]])
    }
  }

  overrides$mode <- mode
  overrides
}

analysis2_normalize_outer <- function(value, default) {
  if (is.null(value)) {
    return(as.integer(default))
  }
  if (is.list(value)) {
    value <- unlist(value, recursive = TRUE, use.names = FALSE)
  }
  if (length(value) == 0L) {
    return(as.integer(default))
  }
  if (length(value) == 1L && is.character(value)) {
    value <- strsplit(value, ",", fixed = TRUE)[[1]]
  }
  out <- as.integer(value)
  if (length(out) == 0L || anyNA(out)) {
    stop("`outer` must be an integer or comma-separated integer list.", call. = FALSE)
  }
  out
}

analysis2_normalize_run_spec <- function(spec, mode, paths) {
  defaults <- analysis2_mode_defaults(mode)

  spec$mode <- mode
  spec$project_root <- analysis2_path_from_project(spec$project_root %||% paths$project_root, paths$project_root, must_work = TRUE)
  spec$analysis_name <- analysis2_normalize_string(spec$analysis_name %||% defaults$analysis_name)
  spec$model_type <- analysis2_normalize_string(spec$model_type)
  spec$respect_grid_model_type <- isTRUE(spec$respect_grid_model_type %||% FALSE)
  spec$resave_tfrecords <- isTRUE(spec$resave_tfrecords %||% defaults$resave_tfrecords)
  spec$run_figures <- isTRUE(spec$run_figures %||% FALSE)
  spec$dry_run <- isTRUE(spec$dry_run %||% FALSE)
  spec$help <- isTRUE(spec$help %||% FALSE)
  spec$outcome_metric <- analysis2_normalize_string(spec$outcome_metric)
  spec$data_subset <- analysis2_normalize_string(spec$data_subset)
  spec$data_format <- analysis2_normalize_string(spec$data_format)
  spec$disease_names <- analysis2_parse_csv(spec$disease_names %||% defaults$disease_names)
  spec$outer <- analysis2_normalize_outer(spec$outer, default = defaults$outer)
  if (!is.null(spec$run_seed)) {
    run_seed_numeric <- suppressWarnings(as.numeric(spec$run_seed))
    if (length(run_seed_numeric) != 1L || !is.finite(run_seed_numeric) ||
        run_seed_numeric < 0 || run_seed_numeric > .Machine$integer.max ||
        run_seed_numeric != floor(run_seed_numeric)) {
      stop("`run_seed` must be NULL or one non-negative integer.", call. = FALSE)
    }
    spec$run_seed <- as.integer(run_seed_numeric)
  }

  spec$grid_file <- analysis2_path_from_project(
    spec$grid_file %||% analysis2_mode_default_grid_file(mode, spec$project_root, spec$analysis_name),
    project_root = spec$project_root,
    must_work = FALSE
  )
  spec$tfrecord_dir <- analysis2_path_from_project(
    spec$tfrecord_dir %||% analysis2_mode_default_tfrecord_dir(mode, spec$project_root, spec$analysis_name),
    project_root = spec$project_root,
    must_work = FALSE
  )
  spec$raw_data_dir <- analysis2_path_from_project(spec$raw_data_dir, project_root = spec$project_root, must_work = FALSE)

  if (identical(mode, "real") && is.null(spec$raw_data_dir)) {
    spec$raw_data_dir <- analysis2_path_from_project(defaults$raw_data_dir, project_root = spec$project_root, must_work = FALSE)
  }

  if (identical(mode, "real") && is.null(spec$data_subset)) {
    spec$data_subset <- defaults$data_subset
  }
  if (identical(mode, "multidisease")) {
    spec$data_subset <- spec$data_subset %||% defaults$data_subset
    spec$outcome_metric <- spec$outcome_metric %||% defaults$outcome_metric
    spec$data_format <- spec$data_format %||% defaults$data_format
    if (length(spec$disease_names) == 0L) {
      spec$disease_names <- defaults$disease_names
    }
  }

  spec
}

analysis2_build_run_spec <- function(mode, args = commandArgs(TRUE)) {
  opts <- analysis2_parse_args(args)
  analysis2_validate_args(opts, mode)

  initial_paths <- analysis2_paths(project_root = opts$project_root)
  config_file <- analysis2_resolve_config_file(
    mode = mode,
    opts = opts,
    analysis_root = initial_paths$analysis_root,
    project_root = initial_paths$project_root
  )

  defaults <- analysis2_mode_defaults(mode)
  manifest <- analysis2_load_yaml_config(config_file)
  analysis2_validate_manifest(manifest, mode)
  cli <- analysis2_cli_overrides(opts, mode)

  spec <- utils::modifyList(defaults, manifest)
  spec <- utils::modifyList(spec, cli)
  spec$config_file <- config_file
  spec <- analysis2_normalize_run_spec(spec, mode = mode, paths = initial_paths)
  spec$paths <- analysis2_paths(project_root = spec$project_root)
  spec
}

analysis2_usage <- function(mode, paths = analysis2_paths()) {
  config_default <- file.path(paths$analysis_root, "config", analysis2_default_config_name(mode))
  mode_title <- switch(
    mode,
    real = "real",
    sim = "sim",
    multidisease = "multidisease real",
    mode
  )
  default_example <- switch(
    mode,
    real = "Rscript Analysis2/SuperLModel_MasterReal.R --config=Analysis2/config/real.yaml --outer=3",
    sim = "Rscript Analysis2/SuperLModel_MasterSim.R --config=Analysis2/config/sim.yaml --outer=1 --respect_grid_model_type=TRUE",
    multidisease = "Rscript Analysis2/SuperLModel_MasterReal_MultiDisease.R --config=Analysis2/config/real_multidisease.yaml --outer=1"
  )
  extra_flags <- switch(
    mode,
    real = c("  --raw_data_dir=PATH", "  --outcome_metric=NAME", "  --data_subset=NAME"),
    sim = character(),
    multidisease = c("  --data_subset=NAME", "  --disease_names=a,b", "  --data_format=IHME|WHO|Tycho", "  --outcome_metric=NAME")
  )

  paste(
    c(
      sprintf("Analysis2 %s runner", mode_title),
      "",
      "Supported precedence: CLI > config manifest > grid row > code defaults",
      sprintf("Default config: %s", config_default),
      "",
      "Core flags:",
      "  --config=PATH",
      "  --project_root=PATH",
      "  --analysis_name=NAME",
      "  --grid_file=PATH",
      "  --outer=1,2,3",
      "  --run_seed=INTEGER",
      "  --model_type=DecoderOnly|NeuralODE",
      "  --respect_grid_model_type=TRUE|FALSE",
      "  --resave_tfrecords=TRUE|FALSE",
      "  --run_figures=TRUE|FALSE",
      "  --tfrecord_dir=PATH",
      "  --dry_run=TRUE",
      "  --help",
      extra_flags,
      "",
      "Examples:",
      paste0("  ", default_example)
    ),
    collapse = "\n"
  )
}

analysis2_print_usage <- function(mode, paths = analysis2_paths()) {
  cat(analysis2_usage(mode, paths = paths), "\n")
  invisible(TRUE)
}

analysis2_script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) == 0L) {
    return(NULL)
  }
  normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = TRUE)
}

analysis2_paths <- function(project_root = NULL, script_file = analysis2_script_path()) {
  analysis2_root <- analysis2_internal_analysis_root()
  if (is.null(project_root)) {
    project_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
  } else {
    project_root <- normalizePath(project_root, winslash = "/", mustWork = TRUE)
  }

  list(
    project_root = normalizePath(project_root, winslash = "/", mustWork = TRUE),
    analysis_root = analysis2_root,
    analysis2_root = analysis2_root
  )
}

analysis2_dir_create <- function(path) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  invisible(path)
}

analysis2_prepare_output_roots <- function(project_root, sim_mode = FALSE) {
  analysis2_dir_create(file.path(project_root, "SavedModels"))
  analysis2_dir_create(file.path(project_root, "SavedResults"))
  analysis2_dir_create(file.path(project_root, "SavedResults", if (isTRUE(sim_mode)) "Sim" else "Real"))
  analysis2_dir_create(file.path(project_root, "SavedModels", if (isTRUE(sim_mode)) "FromSim" else "FromReal"))
  invisible(project_root)
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

analysis2_scalar_df <- function(values) {
  as.data.frame(values, stringsAsFactors = FALSE, check.names = FALSE)
}

analysis2_order_grid <- function(grid, outer_iterations) {
  grid <- grid[order(grid$BaseID), , drop = FALSE]
  if (nrow(grid) == 0L || length(outer_iterations) == 0L) {
    return(grid)
  }
  first_outer <- outer_iterations[[1]]
  if (is.na(first_outer) || first_outer < 1L || first_outer > nrow(grid)) {
    return(grid)
  }
  if (!isTRUE(grid$BaseID[[first_outer]] %% 2 == 0)) {
    return(grid)
  }

  even_rows <- grid$BaseID %% 2 == 0
  if (!any(even_rows)) {
    return(grid)
  }

  grid[even_rows, ] <- grid[even_rows, ][
    order(apply(grid[even_rows, , drop = FALSE], 1, function(zr) rlang::hash(paste(zr, collapse = "_")))),
    ,
    drop = FALSE
  ]
  grid
}

analysis2_validate_outer_iterations <- function(grid, outer_iterations, grid_file) {
  if (length(outer_iterations) == 0L) {
    stop("No outer iterations were requested.", call. = FALSE)
  }

  bad <- outer_iterations[is.na(outer_iterations) | outer_iterations < 1L | outer_iterations > nrow(grid)]
  if (length(bad) == 0L) {
    return(invisible(outer_iterations))
  }

  stop(
    "Requested outer iteration(s) outside the grid bounds for ", grid_file, ": ",
    paste(bad, collapse = ", "),
    ". Valid rows are between 1 and ", nrow(grid), ".",
    call. = FALSE
  )
}

analysis2_require_ndm <- function() {
  pkg <- tryCatch(utils::packageName(), error = function(e) NULL)
  if (!is.null(pkg) && nzchar(pkg)) {
    return(pkg)
  }
  "ndm"
}

analysis2_require_ndmdatasets <- function() {
  if (requireNamespace("ndmdatasets", quietly = TRUE)) {
    return("ndmdatasets")
  }

  stop(
    "The `ndmdatasets` package is required. Install it before running Analysis2.",
    call. = FALSE
  )
}

analysis2_backend_conda_env <- function() {
  env_name <- Sys.getenv("NDM_SOFTWARE_CONDA_ENV", unset = "")
  if (nzchar(env_name)) {
    return(env_name)
  }
  env_name <- Sys.getenv("NDM_CONDA_ENV", unset = "")
  if (nzchar(env_name)) {
    return(env_name)
  }
  if (grepl(version$os, pattern = "darwin")) {
    return("jax_cpu")
  }
  "ndm_software_env"
}

analysis2_call <- function(pkg, name, ...) {
  do.call(getExportedValue(pkg, name), list(...))
}

analysis2_import_tensorflow <- function(ndm_pkg = analysis2_require_ndm(),
                                        float_type = "32") {
  backend <- analysis2_call(
    ndm_pkg,
    "ndm_initialize_backend",
    conda_env = analysis2_backend_conda_env(),
    float_type = as.character(float_type),
    import_tensorflow = TRUE
  )
  tf <- backend$tf

  if (is.null(tf)) {
    stop(
      "TensorFlow is not available in the configured ndm backend environment (`",
      analysis2_backend_conda_env(),
      "`).",
      call. = FALSE
    )
  }

  tf
}

analysis2_prepare_runtime <- function(ndm_pkg, config) {
  runtime_env <- analysis2_call(ndm_pkg, "ndm_new_runtime_env", parent = globalenv())
  analysis2_call(ndm_pkg, "ndm_prepare_runtime", config = config, runtime_env = runtime_env)
}

analysis2_runtime_global_exposure_enabled <- function() {
  default <- getOption("ndm.analysis2.expose_runtime_env", TRUE)
  flag <- get0(
    "analysis2_expose_runtime_env_enabled",
    envir = globalenv(),
    inherits = FALSE,
    ifnotfound = default
  )
  isTRUE(flag)
}

analysis2_expose_runtime_env <- function(runtime_env, target_env = globalenv(), force = FALSE) {
  if (!isTRUE(force) && !analysis2_runtime_global_exposure_enabled()) {
    return(invisible(runtime_env))
  }
  names_all <- ls(runtime_env, all.names = TRUE)
  if (length(names_all) == 0L) {
    return(invisible(runtime_env))
  }
  list2env(mget(names_all, envir = runtime_env, inherits = FALSE), envir = target_env)
  invisible(runtime_env)
}

analysis2_model_spec_name <- function(model_spec_name = NULL,
                                      model_tex_loc = NULL) {
  explicit <- analysis2_normalize_string(model_spec_name)
  if (!is.null(explicit)) {
    return(sub("\\.tex$", "", basename(explicit)))
  }

  legacy_path <- analysis2_normalize_string(model_tex_loc)
  if (is.null(legacy_path)) {
    return(NULL)
  }
  sub("\\.tex$", "", basename(legacy_path))
}

analysis2_resolve_model_tex_loc <- function(row_values) {
  legacy_path <- analysis2_normalize_string(row_values$model_tex_loc)
  if (!is.null(legacy_path)) {
    return(legacy_path)
  }

  spec_name <- analysis2_model_spec_name(
    model_spec_name = row_values$model_spec_name,
    model_tex_loc = row_values$model_tex_loc
  )
  if (is.null(spec_name)) {
    return(NULL)
  }

  file.path("./Analysis2/ModelStructureTex", sprintf("%s.tex", spec_name))
}

analysis2_normalize_row_values <- function(row_values) {
  row_values$model_spec_name <- analysis2_model_spec_name(
    model_spec_name = row_values$model_spec_name,
    model_tex_loc = row_values$model_tex_loc
  )
  row_values$model_tex_loc <- analysis2_resolve_model_tex_loc(row_values)
  row_values
}

analysis2_unique_preserve_order <- function(x) {
  x[!duplicated(x)]
}

analysis2_canonical_variation_fields <- function(mode) {
  switch(
    mode,
    real = c(
      "ModelType",
      "ModelDepth",
      "ModelDims",
      "nSamplesTrain",
      "nObsInference",
      "floatType",
      "model_spec_name",
      "model_tex_loc",
      "ResaveThisTFRecord"
    ),
    sim = c(
      "ModelType",
      "ModelDepth",
      "ModelDims",
      "nSamplesTrain",
      "floatType",
      "model_spec_name",
      "model_tex_loc",
      "ResaveThisTFRecord"
    ),
    stop("Unsupported Analysis2 mode for canonical TFRecord validation: ", mode, call. = FALSE)
  )
}

analysis2_scalar_equal <- function(x, y) {
  if (is.null(x) || is.null(y)) {
    return(is.null(x) && is.null(y))
  }
  if (length(x) == 1L && length(y) == 1L && is.na(x) && is.na(y)) {
    return(TRUE)
  }
  identical(as.character(x), as.character(y))
}

analysis2_row_flag_true <- function(row_values, field = "ResaveThisTFRecord") {
  if (!field %in% names(row_values)) {
    return(FALSE)
  }
  value <- suppressWarnings(analysis2_as_int(row_values[[field]]))
  !is.na(value) && value != 0L
}

analysis2_validate_canonical_group <- function(mode,
                                               base_id,
                                               row_indices,
                                               row_group) {
  varying_fields <- analysis2_canonical_variation_fields(mode)
  fields_to_compare <- setdiff(names(row_group[[1L]]), c("BaseID", varying_fields))
  conflicting_fields <- character()
  reference_row <- row_group[[1L]]

  for (field in fields_to_compare) {
    same_field <- vapply(
      row_group[-1L],
      function(row_values) analysis2_scalar_equal(reference_row[[field]], row_values[[field]]),
      logical(1)
    )
    if (!all(same_field)) {
      conflicting_fields <- c(conflicting_fields, field)
    }
  }

  if (length(conflicting_fields) > 0L) {
    stop(
      "Selected rows for BaseID ",
      base_id,
      " disagree on dataset-defining fields for ",
      mode,
      " TFRecord reuse/regeneration: ",
      paste(conflicting_fields, collapse = ", "),
      ". Rows: ",
      paste(row_indices, collapse = ", "),
      ".",
      call. = FALSE
    )
  }

  invisible(TRUE)
}

analysis2_build_canonical_tfrecord_plan <- function(mode,
                                                    grid,
                                                    outer_iterations) {
  outer_rows <- analysis2_unique_preserve_order(as.integer(outer_iterations))
  if (length(outer_rows) == 0L) {
    return(data.frame(
      BaseID = integer(),
      selected_rows = character(),
      canonical_row = integer(),
      artifact_n_samples_train = integer(),
      stringsAsFactors = FALSE
    ))
  }

  selected_rows <- lapply(
    outer_rows,
    function(row_idx) {
      analysis2_normalize_row_values(
        analysis2_row_to_list(grid[row_idx, , drop = FALSE])
      )
    }
  )
  base_ids <- vapply(selected_rows, function(row_values) analysis2_as_int(row_values$BaseID), integer(1))
  ordered_base_ids <- analysis2_unique_preserve_order(base_ids)

  plan_rows <- lapply(
    ordered_base_ids,
    function(base_id) {
      group_positions <- which(base_ids == base_id)
      row_indices <- outer_rows[group_positions]
      row_group <- selected_rows[group_positions]

      analysis2_validate_canonical_group(
        mode = mode,
        base_id = base_id,
        row_indices = row_indices,
        row_group = row_group
      )

      n_samples_train <- vapply(
        row_group,
        function(row_values) analysis2_as_int(row_values$nSamplesTrain),
        integer(1)
      )
      if (anyNA(n_samples_train)) {
        stop(
          "Selected rows for BaseID ",
          base_id,
          " contain missing `nSamplesTrain` values.",
          call. = FALSE
        )
      }

      max_n_samples_train <- max(n_samples_train)
      max_rows <- row_indices[n_samples_train == max_n_samples_train]
      flagged_rows <- row_indices[vapply(row_group, analysis2_row_flag_true, logical(1))]

      if (length(flagged_rows) > 1L) {
        stop(
          "Selected rows for BaseID ",
          base_id,
          " contain multiple `ResaveThisTFRecord = 1` rows: ",
          paste(flagged_rows, collapse = ", "),
          ".",
          call. = FALSE
        )
      }

      canonical_row <- min(max_rows)
      if (length(flagged_rows) == 1L) {
        canonical_row <- flagged_rows[[1L]]
        if (!(canonical_row %in% max_rows)) {
          stop(
            "Selected rows for BaseID ",
            base_id,
            " mark row ",
            canonical_row,
            " with `ResaveThisTFRecord = 1`, but canonical TFRecord regeneration ",
            "requires the flagged row to use the largest `nSamplesTrain`. ",
            "Rows with max `nSamplesTrain`: ",
            paste(max_rows, collapse = ", "),
            ".",
            call. = FALSE
          )
        }
      }

      data.frame(
        BaseID = as.integer(base_id),
        selected_rows = paste(row_indices, collapse = ","),
        canonical_row = as.integer(canonical_row),
        artifact_n_samples_train = as.integer(max_n_samples_train),
        stringsAsFactors = FALSE
      )
    }
  )

  do.call(rbind, plan_rows)
}

analysis2_bootstrap_base_id_rows <- function(grid,
                                             base_ids = NULL) {
  if (nrow(grid) == 0L) {
    return(integer())
  }

  grid_base_ids <- suppressWarnings(as.integer(grid$BaseID))
  if (length(grid_base_ids) != nrow(grid) || anyNA(grid_base_ids)) {
    stop(
      "Simulation TFRecord bootstrap requires every grid row to contain an integer `BaseID`.",
      call. = FALSE
    )
  }

  if (is.null(base_ids)) {
    return(seq_len(nrow(grid)))
  }

  requested_base_ids <- analysis2_unique_preserve_order(as.integer(base_ids))
  if (length(requested_base_ids) == 0L || anyNA(requested_base_ids)) {
    stop(
      "`base_ids` must contain at least one integer BaseID.",
      call. = FALSE
    )
  }

  missing_base_ids <- setdiff(requested_base_ids, analysis2_unique_preserve_order(grid_base_ids))
  if (length(missing_base_ids) > 0L) {
    stop(
      "Requested BaseID(s) are missing from the simulation grid: ",
      paste(missing_base_ids, collapse = ", "),
      ".",
      call. = FALSE
    )
  }

  unlist(
    lapply(
      requested_base_ids,
      function(base_id) which(grid_base_ids == base_id)
    ),
    use.names = FALSE
  )
}

analysis2_canonical_tfrecord_artifact_paths <- function(paths) {
  c(
    paths$train_file,
    paths$inference_file,
    analysis2_manifest_path(paths$train_file),
    analysis2_manifest_path(paths$inference_file)
  )
}

analysis2_has_any_tfrecord_artifacts <- function(paths) {
  any(file.exists(analysis2_canonical_tfrecord_artifact_paths(paths)))
}

analysis2_has_incomplete_tfrecord_artifacts <- function(paths) {
  analysis2_has_any_tfrecord_artifacts(paths) && !analysis2_has_canonical_tfrecords(paths)
}

analysis2_bootstrap_lock_dir <- function(output_dir, base_id) {
  file.path(output_dir, sprintf(".bootstrap_lock_%s", base_id))
}

analysis2_wait_for_canonical_tfrecords <- function(paths,
                                                   timeout_seconds = 30,
                                                   poll_seconds = 0.2) {
  deadline <- Sys.time() + as.difftime(timeout_seconds, units = "secs")
  repeat {
    if (analysis2_has_canonical_tfrecords(paths)) {
      return(TRUE)
    }
    if (Sys.time() >= deadline) {
      return(FALSE)
    }
    Sys.sleep(poll_seconds)
  }
}

analysis2_build_base_id_tfrecord_plan <- function(mode,
                                                  grid,
                                                  base_ids = NULL,
                                                  tfrecord_dir = NULL) {
  selected_rows <- analysis2_bootstrap_base_id_rows(
    grid = grid,
    base_ids = base_ids
  )

  plan <- analysis2_build_canonical_tfrecord_plan(
    mode = mode,
    grid = grid,
    outer_iterations = selected_rows
  )

  if (is.null(tfrecord_dir)) {
    return(plan)
  }

  if (nrow(plan) == 0L) {
    plan$train_file <- character()
    plan$inference_file <- character()
    return(plan)
  }

  train_files <- character(nrow(plan))
  inference_files <- character(nrow(plan))
  for (plan_idx in seq_len(nrow(plan))) {
    paths <- analysis2_tfrecord_paths(tfrecord_dir, plan$BaseID[[plan_idx]])
    train_files[[plan_idx]] <- paths$train_file
    inference_files[[plan_idx]] <- paths$inference_file
  }

  plan$train_file <- train_files
  plan$inference_file <- inference_files
  plan
}

analysis2_bootstrap_dry_run_status <- function(paths,
                                               base_id,
                                               output_dir,
                                               overwrite = FALSE) {
  if (!isTRUE(overwrite) && analysis2_has_incomplete_tfrecord_artifacts(paths)) {
    stop(
      "Found incomplete TFRecord artifacts for BaseID ",
      base_id,
      " under ",
      output_dir,
      ". Remove them or rerun with `overwrite = TRUE` before bootstrapping canonical TFRecords.",
      call. = FALSE
    )
  }

  if (!isTRUE(overwrite) && analysis2_has_canonical_tfrecords(paths)) {
    return("skipped_existing")
  }

  "planned"
}

analysis2_bootstrap_write_sim_tfrecord <- function(base_id,
                                                   canonical_row,
                                                   artifact_n_samples_train,
                                                   row_values,
                                                   tfrecord_dir,
                                                   resolve_backend,
                                                   overwrite = FALSE) {
  paths <- analysis2_tfrecord_paths(tfrecord_dir, base_id)
  if (!isTRUE(overwrite) && analysis2_has_incomplete_tfrecord_artifacts(paths)) {
    stop(
      "Found incomplete TFRecord artifacts for BaseID ",
      base_id,
      " under ",
      tfrecord_dir,
      ". Remove them or rerun with `overwrite = TRUE` before bootstrapping canonical TFRecords.",
      call. = FALSE
    )
  }

  if (!isTRUE(overwrite) && analysis2_has_canonical_tfrecords(paths)) {
    analysis2_log(
      sprintf(
        "Skipping canonical sim TFRecords for BaseID %s because canonical artifacts already exist",
        base_id
      )
    )
    return("skipped_existing")
  }

  lock_dir <- analysis2_bootstrap_lock_dir(tfrecord_dir, base_id)
  acquired_lock <- dir.create(lock_dir, recursive = FALSE, showWarnings = FALSE)
  if (!isTRUE(acquired_lock)) {
    if (!isTRUE(overwrite) && analysis2_wait_for_canonical_tfrecords(paths)) {
      analysis2_log(
        sprintf(
          "Skipping canonical sim TFRecords for BaseID %s because another worker materialized them first",
          base_id
        )
      )
      return("skipped_locked_existing")
    }

    stop(
      "Could not acquire the canonical TFRecord bootstrap lock for BaseID ",
      base_id,
      " under ",
      tfrecord_dir,
      ". Another worker may still be writing this BaseID or a stale lock remains at ",
      lock_dir,
      ".",
      call. = FALSE
    )
  }

  on.exit(unlink(lock_dir, recursive = TRUE, force = TRUE), add = TRUE)

  if (!isTRUE(overwrite) && analysis2_has_incomplete_tfrecord_artifacts(paths)) {
    stop(
      "Found incomplete TFRecord artifacts for BaseID ",
      base_id,
      " under ",
      tfrecord_dir,
      " after acquiring the bootstrap lock. Remove them or rerun with `overwrite = TRUE`.",
      call. = FALSE
    )
  }

  if (!isTRUE(overwrite) && analysis2_has_canonical_tfrecords(paths)) {
    analysis2_log(
      sprintf(
        "Skipping canonical sim TFRecords for BaseID %s because canonical artifacts were written before lock acquisition completed",
        base_id
      )
    )
    return("skipped_existing")
  }

  backend <- resolve_backend()
  model_type <- analysis2_model_type(
    opts = list(model_type = NULL, respect_grid_model_type = TRUE),
    grid_model_type = row_values$ModelType,
    default = "DecoderOnly"
  )
  dataset_spec <- analysis2_sim_dataset_spec(backend$ndmdatasets_pkg, row_values)
  training_spec <- analysis2_sim_training_spec(backend$ndmdatasets_pkg, row_values, model_type = model_type)
  training_spec$n_samples_train <- as.integer(artifact_n_samples_train)

  analysis2_log(
    sprintf(
      "Regenerating canonical sim TFRecords for BaseID %s from row %s with nSamplesTrain %s",
      base_id,
      canonical_row,
      artifact_n_samples_train
    )
  )
  analysis2_write_sim_tfrecords(
    ndmdatasets_pkg = backend$ndmdatasets_pkg,
    dataset_spec = dataset_spec,
    training_spec = training_spec,
    output_dir = tfrecord_dir,
    tensorflow = backend$tensorflow
  )

  "written"
}

analysis2_bootstrap_sim_tfrecords <- function(project_root,
                                              analysis_name = "BigSimsLatest",
                                              grid,
                                              base_ids = NULL,
                                              tfrecord_dir,
                                              overwrite = FALSE,
                                              dry_run = FALSE) {
  stopifnot(is.data.frame(grid))

  project_root <- normalizePath(project_root, winslash = "/", mustWork = TRUE)
  paths <- analysis2_paths(project_root = project_root)
  tfrecord_dir <- normalizePath(tfrecord_dir, winslash = "/", mustWork = FALSE)
  write_plan <- analysis2_build_base_id_tfrecord_plan(
    mode = "sim",
    grid = grid,
    base_ids = base_ids,
    tfrecord_dir = tfrecord_dir
  )

  if (nrow(write_plan) == 0L) {
    write_plan$status <- character()
    return(write_plan)
  }

  if (isTRUE(dry_run)) {
    write_plan$status <- vapply(
      write_plan$BaseID,
      function(base_id) {
        analysis2_bootstrap_dry_run_status(
          paths = analysis2_tfrecord_paths(tfrecord_dir, base_id),
          base_id = base_id,
          output_dir = tfrecord_dir,
          overwrite = overwrite
        )
      },
      character(1)
    )
    return(write_plan)
  }

  old_wd <- tryCatch(getwd(), error = function(e) NULL)
  if (!is.null(old_wd) && nzchar(old_wd)) {
    on.exit(setwd(old_wd), add = TRUE)
  }
  setwd(paths$project_root)
  analysis2_prepare_output_roots(paths$project_root, sim_mode = TRUE)
  analysis2_dir_create(tfrecord_dir)

  ndmdatasets_pkg <- NULL
  ndm_pkg <- NULL
  tensorflow <- NULL
  resolve_backend <- function() {
    if (is.null(ndmdatasets_pkg)) {
      ndmdatasets_pkg <<- analysis2_require_ndmdatasets()
    }
    if (is.null(ndm_pkg)) {
      ndm_pkg <<- analysis2_require_ndm()
    }
    if (is.null(tensorflow)) {
      tensorflow <<- analysis2_import_tensorflow(ndm_pkg = ndm_pkg)
    }
    list(ndmdatasets_pkg = ndmdatasets_pkg, tensorflow = tensorflow)
  }

  statuses <- character(nrow(write_plan))
  for (plan_idx in seq_len(nrow(write_plan))) {
    canonical_row <- write_plan$canonical_row[[plan_idx]]
    row_values <- analysis2_normalize_row_values(
      analysis2_row_to_list(grid[canonical_row, , drop = FALSE])
    )

    statuses[[plan_idx]] <- analysis2_bootstrap_write_sim_tfrecord(
      base_id = analysis2_as_int(write_plan$BaseID[[plan_idx]]),
      canonical_row = canonical_row,
      artifact_n_samples_train = write_plan$artifact_n_samples_train[[plan_idx]],
      row_values = row_values,
      tfrecord_dir = tfrecord_dir,
      resolve_backend = resolve_backend,
      overwrite = overwrite
    )
  }

  write_plan$status <- statuses
  write_plan
}

analysis2_resolve_model_spec <- function(model_tex_loc = NULL,
                                         model_spec_name = NULL,
                                         model_type = "DecoderOnly",
                                         project_root,
                                         ndm_pkg = "ndm") {
  model_spec_name <- analysis2_model_spec_name(model_spec_name = model_spec_name, model_tex_loc = model_tex_loc)
  model_tex_loc <- analysis2_normalize_string(model_tex_loc)
  if (is.null(model_tex_loc) && is.null(model_spec_name)) {
    return(NULL)
  }

  packaged_name <- if (is.null(model_spec_name)) basename(model_tex_loc) else sprintf("%s.tex", model_spec_name)
  packaged_tex_path <- system.file("extdata", "model_specs", packaged_name, package = ndm_pkg)
  if (nzchar(packaged_tex_path) && file.exists(packaged_tex_path)) {
    return(analysis2_call(ndm_pkg, "ndm_model_spec_from_tex", path = packaged_tex_path, model_type = model_type))
  }

  if (is.null(model_tex_loc)) {
    stop(
      "Could not resolve model spec `", packaged_name,
      "` from the packaged ndm model specs and no repo-relative `model_tex_loc` was provided.",
      call. = FALSE
    )
  }

  tex_path <- if (startsWith(model_tex_loc, "./")) {
    file.path(project_root, sub("^\\./", "", model_tex_loc))
  } else {
    model_tex_loc
  }

  analysis2_call(
    ndm_pkg,
    "ndm_model_spec_from_tex",
    path = normalizePath(tex_path, winslash = "/", mustWork = TRUE),
    model_type = model_type
  )
}

analysis2_model_type <- function(opts,
                                 grid_model_type = NULL,
                                 default = "DecoderOnly") {
  if (!is.null(opts$model_type) && nzchar(as.character(opts$model_type))) {
    return(opts$model_type)
  }

  if (isTRUE(opts$respect_grid_model_type) &&
      !is.null(grid_model_type) &&
      nzchar(as.character(grid_model_type))) {
    return(as.character(grid_model_type))
  }

  default
}

analysis2_real_dataset_spec <- function(ndmdatasets_pkg,
                                        row_values,
                                        outcome_metric = "inc_death",
                                        data_subset = "high_income") {
  analysis2_call(
    ndmdatasets_pkg,
    "ndm_datasets_dataset_spec",
    kind = "real",
    disease = "Covid",
    context_length = analysis2_as_int(row_values$ContextLength),
    lookahead = 12L,
    evaluation_time = analysis2_as_int(row_values$evaluationTime),
    evaluation_sequence = 1L:4L,
    initial_transform = as.character(row_values$initialTransform),
    initial_norm_type = as.character(row_values$initialNormType),
    padding_method = as.character(row_values$paddingMethod),
    split_type = as.character(row_values$OSSType),
    data_subset = as.character(data_subset),
    data_inputs = as.character(row_values$dataInputs),
    outcomes = "ihme_true_value_per_capita",
    per_capita_scaling_factor = 10000,
    roll_window = 26L,
    min_anchoring_time = 4L,
    train_location_fraction = 0.8,
    n_inference_samples = 1024L,
    base_id = analysis2_as_int(row_values$BaseID),
    outcome_metric = outcome_metric
  )
}

analysis2_real_training_spec <- function(ndmdatasets_pkg, row_values, model_type) {
  analysis2_call(
    ndmdatasets_pkg,
    "ndm_datasets_training_spec",
    model_type = model_type,
    model_depth = analysis2_as_int(row_values$ModelDepth),
    model_dims = analysis2_as_int(row_values$ModelDims),
    n_samples_train = analysis2_as_int(row_values$nSamplesTrain),
    float_type = as.character(row_values$floatType),
    model_tex_loc = analysis2_resolve_model_tex_loc(row_values)
  )
}

analysis2_sim_dataset_spec <- function(ndmdatasets_pkg, row_values) {
  lookahead <- if ("lookahead" %in% names(row_values)) {
    analysis2_as_int(row_values$lookahead)
  } else {
    12L
  }
  n_time_steps <- if ("n_time_steps" %in% names(row_values)) {
    analysis2_as_int(row_values$n_time_steps)
  } else {
    NULL
  }
  analysis2_call(
    ndmdatasets_pkg,
    "ndm_datasets_dataset_spec",
    kind = "sim",
    context_length = analysis2_as_int(row_values$ContextLength),
    lookahead = lookahead,
    n_time_steps = n_time_steps,
    global_population = 10000,
    gamma = analysis2_f2n(row_values$gamma),
    sigma = analysis2_f2n(row_values$sigma),
    xi = analysis2_f2n(row_values$xi),
    i0_a = analysis2_f2n(row_values$i0_a),
    i0_b = analysis2_f2n(row_values$i0_b),
    r0_a = analysis2_f2n(row_values$r0_a),
    r0_b = analysis2_f2n(row_values$r0_b),
    betat_init = analysis2_f2n(row_values$betat_init),
    invbetat_sd = analysis2_f2n(row_values$invbetat_sd),
    c_endogeneous = analysis2_f2n(row_values$c_endogeneous),
    policy_responsiveness = analysis2_f2n(row_values$policy_responsiveness),
    policy_effectiveness = analysis2_f2n(row_values$policy_effectiveness),
    policy_decay = analysis2_f2n(row_values$policy_decay),
    covariate_type = "sqrt",
    roll_window = 52L,
    measurement_noise = analysis2_f2n(row_values$measurement_noise),
    hosp_rate = 0.1,
    death_rate = 0.01,
    forward_shift_h = 4L,
    forward_shift_c = 7L,
    n_inference_batches = if ("n_inference_batches" %in% names(row_values)) {
      analysis2_as_int(row_values$n_inference_batches)
    } else {
      8L
    },
    scaling_batches = if ("scaling_batches" %in% names(row_values)) {
      analysis2_as_int(row_values$scaling_batches)
    } else {
      8L
    },
    base_id = analysis2_as_int(row_values$BaseID)
  )
}

analysis2_sim_training_spec <- function(ndmdatasets_pkg, row_values, model_type) {
  analysis2_call(
    ndmdatasets_pkg,
    "ndm_datasets_training_spec",
    model_type = model_type,
    model_depth = analysis2_as_int(row_values$ModelDepth),
    model_dims = analysis2_as_int(row_values$ModelDims),
    n_samples_train = analysis2_as_int(row_values$nSamplesTrain),
    float_type = as.character(row_values$floatType),
    model_tex_loc = analysis2_resolve_model_tex_loc(row_values)
  )
}

analysis2_resolve_real_inputs <- function(ndmdatasets_pkg, bundle, dataset_spec) {
  analysis2_call(ndmdatasets_pkg, "ndm_real_resolve_inputs", bundle, dataset_spec)
}

analysis2_prepare_real_state <- function(ndmdatasets_pkg,
                                         bundle,
                                         dataset_spec,
                                         project_root) {
  prepared <- analysis2_call(
    ndmdatasets_pkg,
    "ndm_real_prepare_tables",
    table_bundle = bundle,
    dataset_spec = dataset_spec
  )

  truth_df_red <- as.data.frame(prepared$truth_df, stringsAsFactors = FALSE)
  truth_df_red$Pop <- truth_df_red$POP_population

  input_df_red_full <- as.data.frame(prepared$normalized_df, stringsAsFactors = FALSE)
  input_df_red_in <- as.data.frame(prepared$train_df, stringsAsFactors = FALSE)
  input_df_red_out <- as.data.frame(prepared$out_df, stringsAsFactors = FALSE)
  context_df_red <- truth_df_red[, c("location_id", "time_id", "targetTime_id"), drop = FALSE]

  coordinates_file <- file.path(project_root, "Data", "coordinates_mat.csv")
  coordinates_mat <- if (file.exists(coordinates_file)) {
    utils::read.csv(coordinates_file)
  } else {
    NULL
  }

  list(
    prepared = prepared,
    truth_df_red = truth_df_red,
    input_df_red_full = input_df_red_full,
    input_df_red_in = input_df_red_in,
    input_df_red_out = input_df_red_out,
    context_df_red = context_df_red,
    coordinates_mat = coordinates_mat,
    data_inputs = analysis2_resolve_real_inputs(ndmdatasets_pkg, bundle, dataset_spec)
  )
}

analysis2_batch_to_runtime_jax <- function(x, runtime_env) {
  if (is.list(x)) {
    return(lapply(x, analysis2_batch_to_runtime_jax, runtime_env = runtime_env))
  }
  if (inherits(x, "python.builtin.object")) {
    x <- reticulate::py_to_r(x)
  }
  runtime_env$jnp$array(x)
}

analysis2_real_get_batch_factory <- function(ndmdatasets_pkg,
                                             prepared_state,
                                             dataset_spec,
                                             runtime_env) {
  function(nBatch = NULL,
           training = TRUE,
           INPUT_REF_DAT = NULL,
           finalGenLocID = NULL,
           finalGenTimeID = NULL,
           nTimesLook = NULL,
           openBrowser = FALSE) {
    if (isTRUE(openBrowser)) {
      browser()
    }
    if (is.null(nBatch) || is.na(nBatch) || nBatch <= 0L) {
      stop("`nBatch` must be a positive integer.", call. = FALSE)
    }

    split <- if (isTRUE(training)) "train" else "inference"
    examples <- vector("list", length = nBatch)
    loc_ids <- if (is.null(finalGenLocID)) rep(list(NULL), nBatch) else as.list(rep(finalGenLocID, length.out = nBatch))
    time_ids <- if (is.null(finalGenTimeID)) rep(list(NULL), nBatch) else as.list(rep(finalGenTimeID, length.out = nBatch))

    for (i in seq_len(nBatch)) {
      attempts <- 0L
      repeat {
        attempts <- attempts + 1L
        example <- analysis2_call(
          ndmdatasets_pkg,
          "ndm_real_make_example",
          prepared = prepared_state$prepared,
          dataset_spec = dataset_spec,
          split = split,
          seed = 1000L + i + attempts,
          loc_id = loc_ids[[i]],
          anchor_time = time_ids[[i]]
        )
        if (!is.null(example)) {
          examples[[i]] <- example
          break
        }
        if (attempts > 500L) {
          stop("Could not draw a valid real-data batch example.", call. = FALSE)
        }
      }
    }

    stacked <- analysis2_call(ndmdatasets_pkg, "ndm_datasets_stack_batches", examples)
    analysis2_batch_to_runtime_jax(stacked, runtime_env = runtime_env)
  }
}

analysis2_sim_softplus <- function(x) log1p(exp(x))
analysis2_sim_sigmoid <- function(x) 1 / (1 + exp(-x))
analysis2_sim_inv_softplus <- function(x) log(exp(x) - 1)
analysis2_sim_transition_fraction <- function(rate) 1 - exp(-max(as.numeric(rate), 0))

analysis2_simulate_one <- function(dataset_spec,
                                   seed,
                                   compute_scenario = FALSE,
                                   policy_scenario = NULL) {
  set.seed(seed)

  past <- as.integer(dataset_spec$context_length)
  lookahead <- as.integer(dataset_spec$lookahead)
  n_times <- as.integer(dataset_spec$n_time_steps %||% ((past + lookahead) * 4L))
  n_total <- past + lookahead
  n_pop <- as.numeric(dataset_spec$global_population)

  invbeta_init <- analysis2_sim_inv_softplus(as.numeric(dataset_spec$betat_init))
  beta_noise <- stats::rnorm(n_times, mean = 0, sd = as.numeric(dataset_spec$invbetat_sd))

  i0 <- exp(stats::runif(
    1,
    min = dataset_spec$i0_a - abs(dataset_spec$i0_a * dataset_spec$i0_b),
    max = dataset_spec$i0_a + abs(dataset_spec$i0_a * dataset_spec$i0_b)
  ))
  r0 <- exp(stats::runif(
    1,
    min = dataset_spec$r0_a - abs(dataset_spec$r0_a * dataset_spec$r0_b),
    max = dataset_spec$r0_a + abs(dataset_spec$r0_a * dataset_spec$r0_b)
  ))
  e0 <- max(10, i0 * 10)
  s0 <- max(n_pop - e0 - i0 - r0, 1)
  init_true <- c(s0, e0, i0, r0)

  forward_shift_h <- as.integer(dataset_spec$forward_shift_h)
  forward_shift_c <- as.integer(dataset_spec$forward_shift_c)
  max_shift <- n_times - (past + lookahead + forward_shift_c)
  if (max_shift < 0L) {
    stop("Simulation configuration leaves no valid initial shift.", call. = FALSE)
  }
  initial_shift <- sample.int(max_shift + 1L, 1L) - 1L

  policy_scenario_indicator <- numeric(n_times)
  if (isTRUE(compute_scenario)) {
    policy_scenario_indicator[-seq_len(min(n_times, initial_shift + past + 1L))] <- 1
  }

  if (is.null(policy_scenario)) {
    policy_scenario <- numeric(n_times)
  } else {
    policy_scenario <- rep(as.numeric(policy_scenario), length.out = n_times)
  }

  raw_beta <- raw_policy <- rep(NA_real_, n_times)
  S <- E <- I <- R <- rep(NA_real_, n_times)
  raw_beta[[1]] <- invbeta_init
  raw_policy[[1]] <- -1
  S[[1]] <- s0
  E[[1]] <- e0
  I[[1]] <- i0
  R[[1]] <- r0

  for (t in seq_len(n_times - 1L)) {
    beta <- analysis2_sim_softplus(raw_beta[[t]])
    policy <- analysis2_sim_sigmoid(raw_policy[[t]])
    prevalence <- (10 / n_pop) * I[[t]]

    infection_force <- beta * I[[t]] / n_pop
    infection_force <- infection_force * (1 - as.numeric(dataset_spec$policy_effectiveness) * policy)
    infection_force <- max(infection_force, 0)
    new_inf <- S[[t]] * analysis2_sim_transition_fraction(infection_force)
    move_ei <- E[[t]] * analysis2_sim_transition_fraction(dataset_spec$sigma)
    move_ir <- I[[t]] * analysis2_sim_transition_fraction(dataset_spec$gamma)
    move_rs <- R[[t]] * analysis2_sim_transition_fraction(dataset_spec$xi)

    S[[t + 1L]] <- max(S[[t]] - new_inf + move_rs, 0)
    E[[t + 1L]] <- max(E[[t]] + new_inf - move_ei, 0)
    I[[t + 1L]] <- max(I[[t]] + move_ei - move_ir, 0)
    R[[t + 1L]] <- max(R[[t]] + move_ir - move_rs, 0)

    raw_beta[[t + 1L]] <- raw_beta[[t]] +
      as.numeric(dataset_spec$beta_restore_rate) * (invbeta_init - raw_beta[[t]]) -
      as.numeric(dataset_spec$c_endogeneous) * prevalence * beta -
      policy * as.numeric(dataset_spec$policy_effectiveness) * beta +
      beta_noise[[t]]

    raw_policy[[t + 1L]] <- raw_policy[[t]] +
      (1 - policy_scenario_indicator[[t]]) * (
        0.1 * prevalence * as.numeric(dataset_spec$policy_responsiveness) -
          policy * as.numeric(dataset_spec$policy_decay)
      ) +
      policy_scenario_indicator[[t]] * (
        as.numeric(dataset_spec$policy_responsiveness) * (policy_scenario[[t]] - raw_policy[[t]])
      )
  }

  obs_policy <- analysis2_sim_sigmoid(raw_policy)
  obs_c_pure <- I
  obs_h_pure <- I * as.numeric(dataset_spec$hosp_rate)
  obs_d_pure <- I * as.numeric(dataset_spec$death_rate)

  noise_fraction <- as.numeric(dataset_spec$measurement_noise)
  noise_fun <- function(x) x + x * stats::runif(length(x), min = -noise_fraction, max = noise_fraction)
  obs_c <- pmax(noise_fun(obs_c_pure), 0)
  obs_h <- pmax(noise_fun(obs_h_pure), 0)
  obs_d <- pmax(noise_fun(obs_d_pure), 0)

  idx_past <- initial_shift + seq_len(past)
  idx_h <- idx_past + forward_shift_h
  idx_c <- idx_past + forward_shift_c
  idx_out <- max(idx_past) + seq_len(lookahead)
  idx_all <- min(idx_past):max(idx_out)
  policy_out_idx <- max(idx_c) + seq_len(lookahead)
  policy_all_idx <- c(idx_c, policy_out_idx)

  xpred_d <- pmax(obs_d[idx_past], 0)
  xpred_h <- pmax(obs_h[idx_h], 0)
  xpred_c <- pmax(obs_c[idx_c], 0)
  ytrue_d <- pmax(obs_d[idx_all], 0)
  ytrue_d_out <- pmax(obs_d[idx_out], 0)
  policy_seen <- obs_policy[idx_c]
  policy_all <- obs_policy[policy_all_idx]
  policy_out <- obs_policy[policy_out_idx]

  features <- list()
  if (grepl("sqrt", dataset_spec$covariate_type, fixed = TRUE)) {
    features$XPred_c_sqrt <- sqrt(xpred_c)
    features$XPred_h_sqrt <- sqrt(xpred_h)
    features$XPred_d_sqrt <- sqrt(xpred_d)
  }
  if (grepl("log", dataset_spec$covariate_type, fixed = TRUE)) {
    log_const <- 1 / n_pop
    features$XPred_c_log <- log(log_const + xpred_c)
    features$XPred_h_log <- log(log_const + xpred_h)
    features$XPred_d_log <- log(log_const + xpred_d)
  }
  if (grepl("raw", dataset_spec$covariate_type, fixed = TRUE)) {
    features$XPred_c <- xpred_c
    features$XPred_h <- xpred_h
    features$XPred_d <- xpred_d
  }

  xpred_pure <- do.call(cbind, unname(features))
  xpred_unscaled <- cbind(xpred_pure, PolicyDat_seen = policy_seen)

  list(
    XPred_unscaled = xpred_unscaled,
    XPred_pure = xpred_pure,
    YTrue_pure = matrix(ytrue_d, ncol = 1L),
    YTrue_out_pure = matrix(ytrue_d_out, ncol = 1L),
    PolicyDat_all = matrix(policy_all, ncol = 1L),
    inv_beta_true = matrix(raw_beta[idx_all], ncol = 1L),
    gamma_true = matrix(as.numeric(dataset_spec$gamma), nrow = 1L, ncol = 1L),
    sigma_true = matrix(as.numeric(dataset_spec$sigma), nrow = 1L, ncol = 1L),
    xi_true = matrix(as.numeric(dataset_spec$xi), nrow = 1L, ncol = 1L),
    init_true = matrix(init_true, nrow = 1L),
    location_id_numeric = matrix(0L, nrow = 1L, ncol = 1L),
    time_id_numeric = matrix(initial_shift, nrow = 1L, ncol = 1L),
    initial_shift = matrix(initial_shift, nrow = 1L, ncol = 1L),
    ytrue = cbind(ytrue_d, policy_all),
    ytrue_out = cbind(ytrue_d_out, policy_out)
  )
}

analysis2_stack_examples <- function(values) {
  first <- values[[1]]
  dims <- dim(first)

  if (is.null(dims)) {
    vec_len <- length(first)
    out <- array(unlist(values, use.names = FALSE), dim = c(length(values), vec_len))
    if (vec_len == 1L) {
      return(matrix(out, ncol = 1L))
    }
    return(out)
  }

  out <- array(
    vector(mode = typeof(first), length = length(values) * prod(dims)),
    dim = c(length(values), dims)
  )

  for (i in seq_along(values)) {
    if (length(dims) == 1L) {
      out[i, ] <- values[[i]]
    } else if (length(dims) == 2L) {
      out[i, , ] <- values[[i]]
    } else if (length(dims) == 3L) {
      out[i, , , ] <- values[[i]]
    } else {
      stop("Batch stacking currently supports up to 3 dimensions per field.", call. = FALSE)
    }
  }

  out
}

analysis2_stack_batches <- function(examples) {
  fields <- names(examples[[1]])
  out <- stats::setNames(vector("list", length(fields)), fields)
  for (field in fields) {
    out[[field]] <- analysis2_stack_examples(lapply(examples, `[[`, field))
  }
  out
}

analysis2_make_sim_example <- function(dataset_spec,
                                       scaler,
                                       seed,
                                       compute_scenario = FALSE,
                                       policy_scenario = NULL) {
  sim <- analysis2_simulate_one(
    dataset_spec = dataset_spec,
    seed = seed,
    compute_scenario = compute_scenario,
    policy_scenario = policy_scenario
  )
  xpred <- sweep(sim$XPred_unscaled, 2, scaler$mean, "-")
  xpred <- sweep(xpred, 2, 0.001 + scaler$sd, "/")

  ytrue <- array(sim$ytrue, dim = c(nrow(sim$ytrue), 2L))
  ytrue_out <- array(sim$ytrue_out, dim = c(nrow(sim$ytrue_out), 2L))
  y_mask <- matrix(TRUE, nrow = nrow(ytrue), ncol = ncol(ytrue))
  y_out_mask <- matrix(TRUE, nrow = nrow(ytrue_out), ncol = ncol(ytrue_out))

  list(
    XPred = xpred,
    XPred_mask = matrix(TRUE, nrow = nrow(xpred), ncol = 1L),
    Context = matrix(0, nrow = 1L, ncol = 1L),
    Context_mask = matrix(FALSE, nrow = 1L, ncol = 1L),
    location_id_numeric = sim$location_id_numeric,
    time_id_numeric = sim$time_id_numeric,
    YTrue = ytrue,
    YTrue_mask = y_mask,
    YTrue_out = ytrue_out,
    YTrue_out_mask = y_out_mask,
    inv_beta_true = sim$inv_beta_true,
    gamma_true = sim$gamma_true,
    sigma_true = sim$sigma_true,
    xi_true = sim$xi_true,
    init_true = sim$init_true,
    XPred_pure = sim$XPred_pure,
    YTrue_pure = sim$YTrue_pure,
    YTrue_out_pure = sim$YTrue_out_pure,
    PolicyDat_all = sim$PolicyDat_all,
    initial_shift = sim$initial_shift
  )
}

analysis2_sim_scaler <- function(ndmdatasets_pkg, dataset_spec, seed = 1L) {
  analysis2_call(
    ndmdatasets_pkg,
    "ndm_sim_compute_scaler",
    dataset_spec = dataset_spec,
    seed = seed
  )
}

analysis2_sim_outcome_sd <- function(ndmdatasets_pkg,
                                     dataset_spec,
                                     scaler,
                                     n_batch = 32L,
                                     n_outer = 4L) {
  values <- c()
  counter <- 0L
  for (i in seq_len(n_outer)) {
    for (j in seq_len(n_batch)) {
      counter <- counter + 1L
      example <- analysis2_call(
        ndmdatasets_pkg,
        "ndm_sim_make_example",
        dataset_spec = dataset_spec,
        scaler = scaler,
        seed = 6000L + counter
      )
      values <- c(values, as.numeric(example$YTrue[, 1L]))
    }
  }
  stats::sd(values, na.rm = TRUE)
}

analysis2_sim_get_batch_factory <- function(ndmdatasets_pkg,
                                            dataset_spec,
                                            scaler,
                                            runtime_env) {
  function(nBatch = NULL,
           training = TRUE,
           openBrowser = FALSE,
           INPUT_REF_DAT = NULL,
           finalGenLocID = NULL,
           finalGenTimeID = NULL,
           nTimes = dataset_spec$n_time_steps %||% ((dataset_spec$context_length + dataset_spec$lookahead) * 4L),
           nTimesLook = dataset_spec$lookahead,
           PolicyList = list(ComputeScenario = FALSE, Scenario = NULL)) {
    if (isTRUE(openBrowser)) {
      browser()
    }
    if (is.null(nBatch) || is.na(nBatch) || nBatch <= 0L) {
      stop("`nBatch` must be a positive integer.", call. = FALSE)
    }

    local_spec <- dataset_spec
    local_spec$n_time_steps <- as.integer(nTimes)
    local_spec$lookahead <- as.integer(nTimesLook)

    compute_scenario <- isTRUE(PolicyList$ComputeScenario)
    policy_scenario <- PolicyList$Scenario %||% numeric(local_spec$n_time_steps)

    examples <- lapply(seq_len(nBatch), function(i) {
      analysis2_call(
        ndmdatasets_pkg,
        "ndm_sim_make_example",
        dataset_spec = local_spec,
        scaler = scaler,
        seed = 5000L + i,
        compute_scenario = compute_scenario,
        policy_scenario = policy_scenario
      )
    })

    stacked <- analysis2_call(ndmdatasets_pkg, "ndm_datasets_stack_batches", examples)
    analysis2_batch_to_runtime_jax(stacked, runtime_env = runtime_env)
  }
}

analysis2_tfrecord_paths <- function(output_dir, base_id) {
  list(
    train_file = file.path(output_dir, sprintf("train_%s.tfrecord", base_id)),
    inference_file = file.path(output_dir, sprintf("inference_%s.tfrecord", base_id))
  )
}

analysis2_manifest_path <- function(file) {
  paste0(file, ".manifest.rds")
}

analysis2_has_canonical_tfrecords <- function(paths) {
  all(file.exists(c(
    paths$train_file,
    paths$inference_file,
    analysis2_manifest_path(paths$train_file),
    analysis2_manifest_path(paths$inference_file)
  )))
}

analysis2_assert_canonical_tfrecords <- function(paths, base_id, output_dir) {
  if (analysis2_has_canonical_tfrecords(paths)) {
    return(invisible(paths))
  }

  legacy_files <- file.exists(c(paths$train_file, paths$inference_file))
  if (any(legacy_files)) {
    stop(
      "Found TFRecord files for BaseID ", base_id, " under ", output_dir,
      " but they are missing canonical `.manifest.rds` sidecars. ",
      "Regenerate them with `--resave_tfrecords=TRUE`.",
      call. = FALSE
    )
  }

  stop(
    "Missing canonical TFRecords for BaseID ", base_id, " under ", output_dir,
    ". Regenerate them with `--resave_tfrecords=TRUE`.",
    call. = FALSE
  )
}

analysis2_preflight_canonical_tfrecords <- function(base_ids,
                                                    tfrecord_dir) {
  missing_base_ids <- integer()
  legacy_base_ids <- integer()

  for (base_id in analysis2_unique_preserve_order(as.integer(base_ids))) {
    paths <- analysis2_tfrecord_paths(tfrecord_dir, base_id)

    if (analysis2_has_canonical_tfrecords(paths)) {
      next
    }

    if (any(file.exists(c(paths$train_file, paths$inference_file)))) {
      legacy_base_ids <- c(legacy_base_ids, base_id)
    } else {
      missing_base_ids <- c(missing_base_ids, base_id)
    }
  }

  if (length(legacy_base_ids) > 0L) {
    stop(
      "Found TFRecord files missing canonical `.manifest.rds` sidecars for BaseID(s) ",
      paste(sort(unique(legacy_base_ids)), collapse = ", "),
      " under ",
      tfrecord_dir,
      ". Regenerate them with `--resave_tfrecords=TRUE`.",
      call. = FALSE
    )
  }

  if (length(missing_base_ids) > 0L) {
    stop(
      "Missing canonical TFRecords for BaseID(s) ",
      paste(sort(unique(missing_base_ids)), collapse = ", "),
      " under ",
      tfrecord_dir,
      ". Regenerate them with `--resave_tfrecords=TRUE`.",
      call. = FALSE
    )
  }

  invisible(TRUE)
}

analysis2_attach_canonical_tfrecords <- function(ndmdatasets_pkg,
                                                 runtime_env,
                                                 train_file,
                                                 inference_file,
                                                 schema_kind,
                                                 batch_size,
                                                 shuffle_train = FALSE) {
  tf <- runtime_env$tf %||% analysis2_import_tensorflow()
  max_train_examples <- get0("nSamplesTrain", envir = runtime_env, inherits = TRUE, ifnotfound = NULL)
  max_inference_examples <- get0("nObsInference", envir = runtime_env, inherits = TRUE, ifnotfound = NULL)
  train_dataset <- analysis2_call(
    ndmdatasets_pkg,
    "ndm_datasets_read_tfrecord_dataset",
    file = train_file,
    batch_size = batch_size,
    schema = schema_kind,
    max_examples = max_train_examples,
    shuffle = FALSE,
    tensorflow = tf
  )

  if (isTRUE(shuffle_train)) {
    train_dataset <- train_dataset$shuffle(
      buffer_size = tf$constant(as.integer(10L * batch_size), dtype = tf$int64),
      reshuffle_each_iteration = TRUE
    )
  }

  inference_dataset <- analysis2_call(
    ndmdatasets_pkg,
    "ndm_datasets_read_tfrecord_dataset",
    file = inference_file,
    batch_size = batch_size,
    schema = schema_kind,
    max_examples = max_inference_examples,
    shuffle = FALSE,
    tensorflow = tf
  )

  assign("TFDataset_train", train_dataset, envir = runtime_env)
  assign("TFDatasetIterator_train", reticulate::as_iterator(train_dataset), envir = runtime_env)
  assign("TFDataset_inference", inference_dataset, envir = runtime_env)
  assign("TFDatasetIterator_inference", reticulate::as_iterator(inference_dataset), envir = runtime_env)

  invisible(runtime_env)
}

analysis2_seed_runtime_batch <- function(runtime_env, batch_l) {
  assign("batch_l", batch_l, envir = runtime_env)
  assign("batch_l_cal", batch_l, envir = runtime_env)
  assign("jax_batchx", batch_l$XPred, envir = runtime_env)
  assign("jax_batchy", batch_l$YTrue, envir = runtime_env)
  if ("XPred" %in% names(batch_l)) {
    assign("nCovars_real", batch_l$XPred$shape[[3]], envir = runtime_env)
  }
  invisible(runtime_env)
}

analysis2_write_real_tfrecords <- function(ndmdatasets_pkg,
                                           bundle,
                                           dataset_spec,
                                           training_spec,
                                           output_dir,
                                           tensorflow) {
  analysis2_dir_create(output_dir)
  analysis2_call(
    ndmdatasets_pkg,
    "ndm_real_write_tfrecords",
    table_bundle = bundle,
    dataset_spec = dataset_spec,
    output_dir = output_dir,
    training_spec = training_spec,
    batch_size = 64L,
    tensorflow = tensorflow,
    quiet = TRUE
  )
}

analysis2_write_sim_tfrecords <- function(ndmdatasets_pkg,
                                          dataset_spec,
                                          training_spec,
                                          output_dir,
                                          tensorflow) {
  analysis2_dir_create(output_dir)
  analysis2_call(
    ndmdatasets_pkg,
    "ndm_sim_write_tfrecords",
    dataset_spec = dataset_spec,
    output_dir = output_dir,
    training_spec = training_spec,
    batch_size = 128L,
    tensorflow = tensorflow,
    quiet = TRUE
  )
}

analysis2_real_runtime_globals <- function(row_values,
                                           dataset_spec,
                                           training_spec,
                                           state,
                                           runtime_env,
                                           analysis_name,
                                           analysis_date,
                                           outer_iteration,
                                           holder_folder,
                                           tfrecord_dir,
                                           nsgd_calibration) {
  n_samples_train <- max(1L, analysis2_as_int(row_values$nSamplesTrain))
  n_batch <- min(32L, n_samples_train)
  n_sgd <- analysis2_as_int(nsgd_calibration$resolved_n_sgd)
  max_times_past <- analysis2_as_int(row_values$ContextLength)
  n_times_lookahead <- analysis2_as_int(dataset_spec$lookahead)
  jnp <- runtime_env$jnp
  diffrax <- runtime_env$diffrax
  vi_total_times <- n_times_lookahead
  n_time_steps_sim <- (n_times_lookahead + abs(max_times_past)) * 2L
  max_time_index <- if ("time_id" %in% names(state$truth_df_red)) {
    analysis2_as_int(max(state$truth_df_red$time_id, na.rm = TRUE))
  } else if ("time_id" %in% names(state$input_df_red_full)) {
    analysis2_as_int(max(state$input_df_red_full$time_id, na.rm = TRUE))
  } else {
    max(0L, max_times_past + n_times_lookahead - 1L)
  }

  n_checkpoints <- analysis2_small_run_n_checkpoints(
    nsgd_calibration$anchor_max_n_samples_train,
    n_sgd
  )
  n_obs_inference <- analysis2_small_run_n_obs_inference(
    n_samples_train = n_samples_train,
    n_batch = n_batch,
    configured = row_values$nObsInference %||% NULL
  )

  globals <- list(
    AnalysisName = analysis_name,
    AnalysisDate = analysis_date,
    BaseID = analysis2_as_int(row_values$BaseID),
    RealEntry = analysis2_scalar_df(row_values),
    modelingStrategyNameKey = paste(
      c("RealMode", paste(names(row_values), unlist(row_values, use.names = FALSE), sep = "_")),
      collapse = "__"
    ),
    ContextLength = max_times_past,
    evaluationMethod = as.character(row_values$evaluationMethod),
    evaluationTime = analysis2_as_int(row_values$evaluationTime),
    initialTransform = as.character(row_values$initialTransform),
    initialNormType = as.character(row_values$initialNormType),
    paddingMethod = as.character(row_values$paddingMethod),
    floatType = as.character(row_values$floatType),
    dataInputs = as.character(row_values$dataInputs),
    OSSType = as.character(row_values$OSSType),
    nSamplesTrain = n_samples_train,
    ModelDepth = analysis2_as_int(row_values$ModelDepth),
    ModelDims = analysis2_as_int(row_values$ModelDims),
    nBatch = n_batch,
    nCheckpoints = n_checkpoints,
    nEpochesMax = 9L,
    nSGD_DefiningLRSeq = n_sgd,
    nSGD_model = n_sgd,
    nSGD_pretrain = 0L,
    nSGD_posttrain = n_sgd,
    LEARNING_RATE_MAX = 0.0002,
    LEARNING_RATE_MAX_model = 0.0002,
    LEARNING_RATE_MAX_pretrain = 1e-5,
    UseShortOutcomes = TRUE,
    SimMode = FALSE,
    PreTrain = FALSE,
    IsPretraining = FALSE,
    OUTER_ITERATION = analysis2_as_int(outer_iteration),
    SEED_ = analysis2_as_int(outer_iteration),
    COMMAND_ARG_INPUT = "test",
    HolderFolder = holder_folder,
    TfRecordDir = tfrecord_dir,
    maxTimesPast = max_times_past,
    nTimesLookahead = n_times_lookahead,
    nTimesLookValidationInference = n_times_lookahead,
    nObsInference = n_obs_inference,
    nTimesTotal = max_times_past + n_times_lookahead,
    VI_TotalTimesInLikelihood = vi_total_times,
    minAnchoringTimeID = analysis2_as_int(dataset_spec$min_anchoring_time),
    MIN_NA_ACCEPT_FRAC = 4 / max_times_past,
    NTimeSteps_SIM = n_time_steps_sim,
    MaxSteps = as.integer(10^6),
    DecoderInNeuralODE = FALSE,
    endAppend = TRUE,
    OverDoDataFrac = 0.90,
    specificOptState = TRUE,
    SharedListNames = c("TS"),
    nOutcomes = 1L,
    AppendTimeEmbeds = TRUE,
    AppendPlaceEmbeds = TRUE,
    AttentionHeadDim = 64L,
    AttentionKVHeads = NULL,
    nPlaces = length(unique(state$truth_df_red$location_id)),
    MaxTimeIndex = max_time_index,
    useLSTM = FALSE,
    doGrid = TRUE,
    nRealGridSeed = 128L,
    nExamplesPerCell = 10L,
    nRealGrid = nrow(state$truth_df_red),
    GPU_MEM_FRAC = NULL,
    AVERAGE_TRUTH = mean(state$truth_df_red$ihme_true_value_per_capita, na.rm = TRUE),
    VI_SaveAt_ODE = diffrax$SaveAt(ts = jnp$array(1L:vi_total_times)),
    diff_eq_solver = diffrax$Dopri8(),
    VI_diff_eq_solver = diffrax$Dopri8(),
    stepsize_controller = diffrax$PIDController(rtol = 1e-7, atol = 1e-9),
    diffraxInterpolator = diffrax$LinearInterpolation,
    VI_SaveAt_ODE_sim = diffrax$SaveAt(ts = jnp$array(0L:(n_time_steps_sim - 1L))),
    VI_SaveAt_ODE_optim = diffrax$SaveAt(ts = jnp$array(0L:(vi_total_times - 1L))),
    VI_diff_eq_solver_optim = diffrax$Tsit5(),
    VI_diff_eq_solver_dgp = diffrax$Tsit5(),
    dt0_init = 1e-1,
    dt0_init_dgp = 1e-3,
    dt0_init_optim = 1e-3,
    stepsize_controller_dgp = diffrax$PIDController(rtol = 1e-6, atol = 1e-7),
    stepsize_controller_optim = diffrax$PIDController(rtol = 1e-5, atol = 1e-7),
    truth_df_red = state$truth_df_red,
    input_df_red_full = state$input_df_red_full,
    input_df_red_in = state$input_df_red_in,
    input_df_red_out = state$input_df_red_out,
    context_df_red = state$context_df_red,
    times_in = state$prepared$times_in,
    times_out = state$prepared$times_out,
    dataInputs_colnames = unique(unlist(strsplit(as.character(row_values$dataInputs), split = "__", fixed = TRUE))),
    dataInputs_colnames_past = bundle_default_data_inputs <- state$data_inputs,
    dataInputs_colnames_future = character(),
    perspective_pool = "past",
    all_true_value_names = c(
      "ihme_true_value_per_capita",
      "ihme_true_value_inc_case_per_capita",
      "ihme_true_value_inc_hosp_per_capita",
      "ihme_true_value_inc_death_per_capita"
    ),
    true_value_names = "ihme_true_value_per_capita",
    outcome_metric = dataset_spec$outcome_metric %||% "inc_death",
    training_spec = training_spec
  )

  globals <- c(globals, analysis2_nsgd_calibration_globals(nsgd_calibration))

  if (!is.null(state$coordinates_mat)) {
    globals$coordinates_mat <- state$coordinates_mat
  }

  globals
}

analysis2_sim_runtime_globals <- function(row_values,
                                          dataset_spec,
                                          training_spec,
                                          runtime_env,
                                          analysis_name,
                                          analysis_date,
                                          outer_iteration,
                                          holder_folder,
                                          tfrecord_dir,
                                          sim_scaler,
                                          sim_outcome_sd,
                                          sim_covariates,
                                          get_batch,
                                          nsgd_calibration) {
  n_samples_train <- max(1L, analysis2_as_int(row_values$nSamplesTrain))
  n_batch <- min(32L, n_samples_train)
  n_sgd <- analysis2_as_int(nsgd_calibration$resolved_n_sgd)
  n_times_past <- analysis2_as_int(dataset_spec$context_length)
  n_times_lookahead <- analysis2_as_int(dataset_spec$lookahead)
  n_times_total <- n_times_past + n_times_lookahead
  n_time_steps_sim <- as.integer((n_times_past + n_times_lookahead) * 2L)
  jnp <- runtime_env$jnp
  diffrax <- runtime_env$diffrax
  max_time_index <- max(0L, n_time_steps_sim - 1L)

  n_checkpoints <- analysis2_small_run_n_checkpoints(
    nsgd_calibration$anchor_max_n_samples_train,
    n_sgd
  )

  c(
    list(
    AnalysisName = analysis_name,
    AnalysisDate = analysis_date,
    BaseID = analysis2_as_int(row_values$BaseID),
    SimEntry = row_values,
    ContextLength = analysis2_as_int(row_values$ContextLength),
    nSamplesTrain = n_samples_train,
    nSamples_max = analysis2_as_int(nsgd_calibration$anchor_max_n_samples_train),
    floatType = as.character(row_values$floatType),
    paddingMethod = as.character(row_values$paddingMethod %||% "left"),
    ModelDepth = analysis2_as_int(row_values$ModelDepth),
    ModelDims = analysis2_as_int(row_values$ModelDims),
    nBatch = n_batch,
    nCheckpoints = n_checkpoints,
    nSGD_DefiningLRSeq = n_sgd,
    nSGD_model = n_sgd,
    nSGD_pretrain = 0L,
    nSGD_posttrain = n_sgd,
    LEARNING_RATE_MAX = 0.0002,
    covariateType = "sqrt",
    simCovariates = sim_covariates,
    useLSTM = FALSE,
    quantizeX = FALSE,
    paddingMethod = if ("paddingMethod" %in% names(row_values)) {
      as.character(row_values$paddingMethod)
    } else {
      "left"
    },
    SimMode = TRUE,
    IsPretraining = FALSE,
    OUTER_ITERATION = analysis2_as_int(outer_iteration),
    SEED_ = 100L + analysis2_as_int(outer_iteration),
    COMMAND_ARG_INPUT = "test",
    HolderFolder = holder_folder,
    TfRecordDir = tfrecord_dir,
    GLOBAL_ODE_NPOP = 10000,
    rollCompute_window = 52L,
    nPolicies = 1L,
    nOutcomes = 1L,
    af = 1L,
    AppendTimeEmbeds = FALSE,
    AppendPlaceEmbeds = FALSE,
    AttentionHeadDim = 64L,
    AttentionKVHeads = NULL,
    endAppend = TRUE,
    EnableKVCaching = TRUE,
    MaxTimeIndex = max_time_index,
    nPlaces = 1L,
    nMonteEval = 1L,
    nBatch_SimGridGen = 8L,
    SimScalingOuterLoops = 1L,
    SimScalingInnerLoops = 2L,
    nTimesPast = n_times_past,
    nTimesLookahead = n_times_lookahead,
    nTimesTotal = n_times_total,
    nTimesThres = 10L,
    VI_TotalTimesInLikelihood = n_times_lookahead,
    nTimesInLikelihood = n_times_lookahead,
    NTimeSteps_SIM = n_time_steps_sim,
    nTimesLookValidationInference = n_times_lookahead,
    MaxSteps = as.integer(10^4),
    VI_SaveAt_ODE_sim = diffrax$SaveAt(ts = jnp$array(0L:(n_time_steps_sim - 1L))),
    VI_SaveAt_ODE_optim = diffrax$SaveAt(ts = jnp$array(0L:(n_times_lookahead - 1L))),
    VI_diff_eq_solver_optim = diffrax$Tsit5(),
    VI_diff_eq_solver_dgp = diffrax$Tsit5(),
    dt0_init_dgp = 1e-3,
    stepsize_controller_dgp = diffrax$PIDController(rtol = 1e-6, atol = 1e-7),
    dt0_init_optim = 1e-3,
    stepsize_controller_optim = diffrax$PIDController(rtol = 1e-5, atol = 1e-7),
    diffraxInterpolator = diffrax$LinearInterpolation,
    analysis2_sim_get_batch = get_batch,
    GetBatch = get_batch,
    GetBatch_sim = get_batch,
    GetBatch_base = get_batch,
    input_df_red_in = data.frame(dummy = 0),
    input_df_red_out = data.frame(dummy = 0),
    input_df_red_full = data.frame(dummy = 0),
    SIM_GLOBAL_SCALE_MEAN = sim_scaler$mean,
    SIM_GLOBAL_SCALE_SD = sim_scaler$sd,
    SIM_GLOBAL_OUTCOME_SD = sim_outcome_sd,
    training_spec = training_spec
    ),
    analysis2_nsgd_calibration_globals(nsgd_calibration)
  )
}

analysis2_dry_run_result <- function(spec,
                                     grid,
                                     nsgd_calibration = NULL) {
  preview_rows <- spec$outer[seq_len(min(3L, length(spec$outer)))]
  preview <- grid[preview_rows, , drop = FALSE]
  invisible(list(
    run_spec = spec[setdiff(names(spec), "paths")],
    grid_rows = nrow(grid),
    outer_rows = spec$outer,
    grid_preview = preview,
    training_schedule = nsgd_calibration
  ))
}

analysis2_run_real <- function(args = commandArgs(TRUE)) {
  old_error <- getOption("error")
  old_wd <- getwd()
  on.exit(options(error = old_error), add = TRUE)
  on.exit(setwd(old_wd), add = TRUE)
  options(error = NULL)
  analysis2_log("Starting Analysis2 SuperLModel_MasterReal.R")
  spec <- analysis2_build_run_spec("real", args)
  if (isTRUE(spec$help)) {
    return(analysis2_print_usage("real", paths = spec$paths))
  }

  paths <- spec$paths
  analysis_name <- spec$analysis_name
  analysis_date <- Sys.Date()
  outer_iterations <- spec$outer
  resave_tfrecords <- isTRUE(spec$resave_tfrecords)
  outcome_metric <- spec$outcome_metric
  raw_data_dir <- normalizePath(
    spec$raw_data_dir,
    winslash = "/",
    mustWork = !isTRUE(spec$dry_run)
  )
  tfrecord_dir <- normalizePath(spec$tfrecord_dir, winslash = "/", mustWork = FALSE)
  grid_file <- normalizePath(spec$grid_file, winslash = "/", mustWork = TRUE)

  real_grid_raw <- as.data.frame(data.table::fread(grid_file), stringsAsFactors = FALSE)
  nsgd_calibration <- analysis2_resolve_nsgd_calibration(
    mode = "real",
    spec = spec,
    grid = real_grid_raw,
    n_epoches_max = 9L
  )
  real_grid <- analysis2_order_grid(real_grid_raw, outer_iterations)
  analysis2_validate_outer_iterations(real_grid, outer_iterations, grid_file)
  if (isTRUE(spec$dry_run)) {
    return(analysis2_dry_run_result(spec, real_grid, nsgd_calibration = nsgd_calibration))
  }
  write_plan <- analysis2_build_canonical_tfrecord_plan(
    mode = "real",
    grid = real_grid,
    outer_iterations = outer_iterations
  )
  holder_folder <- file.path(paths$project_root, "SavedResults", "Real", sprintf("Results_%s", analysis_name))
  analysis2_dir_create(holder_folder)

  setwd(paths$project_root)
  analysis2_prepare_output_roots(paths$project_root, sim_mode = FALSE)
  analysis2_log_nsgd_calibration("real", nsgd_calibration)
  if (!isTRUE(resave_tfrecords)) {
    analysis2_preflight_canonical_tfrecords(
      base_ids = write_plan$BaseID,
      tfrecord_dir = tfrecord_dir
    )
  }

  ndmdatasets_pkg <- analysis2_require_ndmdatasets()
  bundle <- analysis2_call(
    ndmdatasets_pkg,
    "ndm_real_build_tables",
    raw_data_dir = raw_data_dir,
    outcome_metric = outcome_metric,
    quiet = TRUE
  )

  if (isTRUE(resave_tfrecords)) {
    ndm_pkg <- analysis2_require_ndm()
    tensorflow <- analysis2_import_tensorflow(ndm_pkg = ndm_pkg)
    written_base_ids <- integer()

    for (plan_idx in seq_len(nrow(write_plan))) {
      canonical_row <- write_plan$canonical_row[[plan_idx]]
      row_values <- analysis2_normalize_row_values(
        analysis2_row_to_list(real_grid[canonical_row, , drop = FALSE])
      )
      model_type <- analysis2_model_type(spec, row_values$ModelType, default = "DecoderOnly")
      dataset_spec <- analysis2_real_dataset_spec(
        ndmdatasets_pkg = ndmdatasets_pkg,
        row_values = row_values,
        outcome_metric = outcome_metric,
        data_subset = spec$data_subset
      )
      training_spec <- analysis2_real_training_spec(ndmdatasets_pkg, row_values, model_type = model_type)
      training_spec$n_samples_train <- as.integer(write_plan$artifact_n_samples_train[[plan_idx]])

      analysis2_log(
        sprintf(
          "Regenerating canonical real TFRecords for BaseID %s from row %s with nSamplesTrain %s",
          row_values$BaseID,
          canonical_row,
          write_plan$artifact_n_samples_train[[plan_idx]]
        )
      )
      analysis2_write_real_tfrecords(
        ndmdatasets_pkg = ndmdatasets_pkg,
        bundle = bundle,
        dataset_spec = dataset_spec,
        training_spec = training_spec,
        output_dir = tfrecord_dir,
        tensorflow = tensorflow
      )
      written_base_ids <- c(written_base_ids, analysis2_as_int(row_values$BaseID))
    }

    return(invisible(list(
      written_base_ids = sort(unique(written_base_ids)),
      tfrecord_dir = tfrecord_dir,
      write_plan = write_plan
    )))
  }

  ndm_pkg <- analysis2_require_ndm()
  for (outer_iteration in outer_iterations) {
    analysis2_log(sprintf("Starting real outer iteration %s", outer_iteration))
    row_values <- analysis2_normalize_row_values(
      analysis2_row_to_list(real_grid[outer_iteration, , drop = FALSE])
    )
    model_type <- analysis2_model_type(spec, row_values$ModelType, default = "DecoderOnly")
    dataset_spec <- analysis2_real_dataset_spec(
      ndmdatasets_pkg = ndmdatasets_pkg,
      row_values = row_values,
      outcome_metric = outcome_metric,
      data_subset = spec$data_subset
    )
    training_spec <- analysis2_real_training_spec(ndmdatasets_pkg, row_values, model_type = model_type)
    tfrecord_paths <- analysis2_tfrecord_paths(tfrecord_dir, analysis2_as_int(row_values$BaseID))

    config <- analysis2_call(
      ndm_pkg,
      "ndm_create_config",
      model_type = model_type,
      backbone = "transformer",
      analysis_root = paths$analysis_root,
      float_type = as.character(row_values$floatType),
      force_to_gpu = TRUE,
      resave_tfrecords = FALSE,
      gpu_mem_frac = NULL
    )
    runtime_env <- analysis2_prepare_runtime(ndm_pkg, config)

    state <- analysis2_prepare_real_state(
      ndmdatasets_pkg = ndmdatasets_pkg,
      bundle = bundle,
      dataset_spec = dataset_spec,
      project_root = paths$project_root
    )
    get_batch <- analysis2_real_get_batch_factory(
      ndmdatasets_pkg = ndmdatasets_pkg,
      prepared_state = state,
      dataset_spec = dataset_spec,
      runtime_env = runtime_env
    )

    analysis2_call(
      ndm_pkg,
      "ndm_set_runtime_globals",
      runtime_env,
      c(
        analysis2_real_runtime_globals(
          row_values = row_values,
          dataset_spec = dataset_spec,
          training_spec = training_spec,
          state = state,
          runtime_env = runtime_env,
          analysis_name = analysis_name,
          analysis_date = analysis_date,
          outer_iteration = outer_iteration,
          holder_folder = holder_folder,
          tfrecord_dir = tfrecord_dir,
          nsgd_calibration = nsgd_calibration
        ),
        list(
          RealGrid = real_grid,
          GetBatch = get_batch,
          GetBatch_base = get_batch,
          GetBatch_real = get_batch,
          analysis2_run_spec = spec,
          analysis2_nsgd_calibration = nsgd_calibration
        )
      )
    )

    preview_batch <- get_batch(
      nBatch = get("nBatch", envir = runtime_env),
      training = TRUE,
      INPUT_REF_DAT = state$input_df_red_in
    )
    analysis2_seed_runtime_batch(runtime_env, preview_batch)
    analysis2_attach_canonical_tfrecords(
      ndmdatasets_pkg = ndmdatasets_pkg,
      runtime_env = runtime_env,
      train_file = tfrecord_paths$train_file,
      inference_file = tfrecord_paths$inference_file,
      schema_kind = "real",
      batch_size = get("nBatch", envir = runtime_env),
      shuffle_train = TRUE
    )
    analysis2_expose_runtime_env(runtime_env)

    model_spec <- analysis2_resolve_model_spec(
      model_tex_loc = row_values$model_tex_loc,
      model_spec_name = row_values$model_spec_name,
      model_type = model_type,
      project_root = paths$project_root,
      ndm_pkg = ndm_pkg
    )

    model <- analysis2_call(
      ndm_pkg,
      "ndm_build_model",
      runtime_env = runtime_env,
      model_type = model_type,
      model_spec = model_spec,
      backbone = "transformer"
    )
    analysis2_expose_runtime_env(runtime_env)
    analysis2_call(ndm_pkg, "ndm_source_runtime_calibration", analysis_root = paths$analysis_root, env = runtime_env)
    analysis2_expose_runtime_env(runtime_env)
    trained <- analysis2_call(
      ndm_pkg,
      "ndm_train",
      x = model,
      run_define = TRUE,
      run_loop = TRUE
    )
    analysis2_expose_runtime_env(trained$env)
    analysis2_call(ndm_pkg, "ndm_source_runtime_results_get", analysis_root = paths$analysis_root, env = trained$env)

    if (isTRUE(spec$run_figures)) {
      analysis2_expose_runtime_env(trained$env)
      analysis2_call(ndm_pkg, "ndm_source_runtime_results_analyze", analysis_root = paths$analysis_root, env = trained$env)
    }

    analysis2_log(sprintf("Done with real outer iteration %s", outer_iteration))
  }

  invisible(TRUE)
}

analysis2_run_sim <- function(args = commandArgs(TRUE)) {
  old_error <- getOption("error")
  old_wd <- getwd()
  on.exit(options(error = old_error), add = TRUE)
  on.exit(setwd(old_wd), add = TRUE)
  options(error = NULL)
  analysis2_log("Starting Analysis2 SuperLModel_MasterSim.R")
  spec <- analysis2_build_run_spec("sim", args)
  if (isTRUE(spec$help)) {
    return(analysis2_print_usage("sim", paths = spec$paths))
  }

  paths <- spec$paths
  analysis_name <- spec$analysis_name
  analysis_date <- Sys.Date()
  outer_iterations <- spec$outer
  resave_tfrecords <- isTRUE(spec$resave_tfrecords)
  tfrecord_dir <- normalizePath(spec$tfrecord_dir, winslash = "/", mustWork = FALSE)
  grid_file <- normalizePath(spec$grid_file, winslash = "/", mustWork = TRUE)

  sim_grid_raw <- as.data.frame(data.table::fread(grid_file), stringsAsFactors = FALSE)
  nsgd_calibration <- analysis2_resolve_nsgd_calibration(
    mode = "sim",
    spec = spec,
    grid = sim_grid_raw,
    n_epoches_max = 6L
  )
  sim_grid <- analysis2_order_grid(sim_grid_raw, outer_iterations)
  analysis2_validate_outer_iterations(sim_grid, outer_iterations, grid_file)
  if (isTRUE(spec$dry_run)) {
    return(analysis2_dry_run_result(spec, sim_grid, nsgd_calibration = nsgd_calibration))
  }
  write_plan <- analysis2_build_canonical_tfrecord_plan(
    mode = "sim",
    grid = sim_grid,
    outer_iterations = outer_iterations
  )
  holder_folder <- file.path(paths$project_root, "SavedResults", "Sim", sprintf("Results_%s", analysis_name))
  analysis2_dir_create(holder_folder)

  setwd(paths$project_root)
  analysis2_prepare_output_roots(paths$project_root, sim_mode = TRUE)
  analysis2_log_nsgd_calibration("sim", nsgd_calibration)
  if (!isTRUE(resave_tfrecords)) {
    analysis2_preflight_canonical_tfrecords(
      base_ids = write_plan$BaseID,
      tfrecord_dir = tfrecord_dir
    )
  }

  ndmdatasets_pkg <- analysis2_require_ndmdatasets()
  sim_covariates <- c(
    XPred_c_sqrt = "inc_case_per_capita_sqrt",
    XPred_h_sqrt = "inc_hosp_per_capita_sqrt",
    XPred_d_sqrt = "inc_death_per_capita_sqrt"
  )

  if (isTRUE(resave_tfrecords)) {
    ndm_pkg <- analysis2_require_ndm()
    tensorflow <- analysis2_import_tensorflow(ndm_pkg = ndm_pkg)
    written_base_ids <- integer()

    for (plan_idx in seq_len(nrow(write_plan))) {
      canonical_row <- write_plan$canonical_row[[plan_idx]]
      row_values <- analysis2_normalize_row_values(
        analysis2_row_to_list(sim_grid[canonical_row, , drop = FALSE])
      )
      model_type <- analysis2_model_type(spec, row_values$ModelType, default = "DecoderOnly")
      dataset_spec <- analysis2_sim_dataset_spec(ndmdatasets_pkg, row_values)
      training_spec <- analysis2_sim_training_spec(ndmdatasets_pkg, row_values, model_type = model_type)
      training_spec$n_samples_train <- as.integer(write_plan$artifact_n_samples_train[[plan_idx]])

      analysis2_log(
        sprintf(
          "Regenerating canonical sim TFRecords for BaseID %s from row %s with nSamplesTrain %s",
          row_values$BaseID,
          canonical_row,
          write_plan$artifact_n_samples_train[[plan_idx]]
        )
      )
      analysis2_write_sim_tfrecords(
        ndmdatasets_pkg = ndmdatasets_pkg,
        dataset_spec = dataset_spec,
        training_spec = training_spec,
        output_dir = tfrecord_dir,
        tensorflow = tensorflow
      )
      written_base_ids <- c(written_base_ids, analysis2_as_int(row_values$BaseID))
    }

    return(invisible(list(
      written_base_ids = sort(unique(written_base_ids)),
      tfrecord_dir = tfrecord_dir,
      write_plan = write_plan
    )))
  }

  ndm_pkg <- analysis2_require_ndm()
  for (outer_iteration in outer_iterations) {
    analysis2_log(sprintf("Starting sim outer iteration %s", outer_iteration))
    row_values <- analysis2_normalize_row_values(
      analysis2_row_to_list(sim_grid[outer_iteration, , drop = FALSE])
    )
    model_type <- analysis2_model_type(spec, row_values$ModelType, default = "DecoderOnly")
    dataset_spec <- analysis2_sim_dataset_spec(ndmdatasets_pkg, row_values)
    training_spec <- analysis2_sim_training_spec(ndmdatasets_pkg, row_values, model_type = model_type)
    tfrecord_paths <- analysis2_tfrecord_paths(tfrecord_dir, analysis2_as_int(row_values$BaseID))

    config <- analysis2_call(
      ndm_pkg,
      "ndm_create_config",
      model_type = model_type,
      backbone = "transformer",
      analysis_root = paths$analysis_root,
      float_type = as.character(row_values$floatType),
      force_to_gpu = TRUE,
      resave_tfrecords = FALSE,
      gpu_mem_frac = NULL
    )
    runtime_env <- analysis2_prepare_runtime(ndm_pkg, config)

    sim_scaler <- analysis2_sim_scaler(ndmdatasets_pkg, dataset_spec, seed = 1L)
    sim_outcome_sd <- analysis2_sim_outcome_sd(ndmdatasets_pkg, dataset_spec, sim_scaler)
    get_batch <- analysis2_sim_get_batch_factory(
      ndmdatasets_pkg = ndmdatasets_pkg,
      dataset_spec = dataset_spec,
      scaler = sim_scaler,
      runtime_env = runtime_env
    )

    analysis2_call(
      ndm_pkg,
      "ndm_set_runtime_globals",
      runtime_env,
      c(
        analysis2_sim_runtime_globals(
          row_values = row_values,
          dataset_spec = dataset_spec,
          training_spec = training_spec,
          runtime_env = runtime_env,
          analysis_name = analysis_name,
          analysis_date = analysis_date,
          outer_iteration = outer_iteration,
          holder_folder = holder_folder,
          tfrecord_dir = tfrecord_dir,
          sim_scaler = sim_scaler,
          sim_outcome_sd = sim_outcome_sd,
          sim_covariates = sim_covariates,
          get_batch = get_batch,
          nsgd_calibration = nsgd_calibration
        ),
        list(
          SimGrid = sim_grid,
          analysis2_run_spec = spec,
          analysis2_nsgd_calibration = nsgd_calibration
        )
      )
    )
    if (!exists("paddingMethod", envir = runtime_env, inherits = FALSE) ||
        is.null(get("paddingMethod", envir = runtime_env, inherits = FALSE))) {
      assign("paddingMethod", "left", envir = runtime_env)
    }
    if (!exists("endAppend", envir = runtime_env, inherits = FALSE) ||
        is.null(get("endAppend", envir = runtime_env, inherits = FALSE))) {
      assign("endAppend", TRUE, envir = runtime_env)
    }

    preview_batch <- get_batch(nBatch = get("nBatch", envir = runtime_env))
    analysis2_seed_runtime_batch(runtime_env, preview_batch)
    analysis2_attach_canonical_tfrecords(
      ndmdatasets_pkg = ndmdatasets_pkg,
      runtime_env = runtime_env,
      train_file = tfrecord_paths$train_file,
      inference_file = tfrecord_paths$inference_file,
      schema_kind = "sim",
      batch_size = get("nBatch", envir = runtime_env),
      shuffle_train = TRUE
    )
    analysis2_expose_runtime_env(runtime_env)

    model_spec <- analysis2_resolve_model_spec(
      model_tex_loc = row_values$model_tex_loc,
      model_spec_name = row_values$model_spec_name,
      model_type = model_type,
      project_root = paths$project_root,
      ndm_pkg = ndm_pkg
    )

    model <- analysis2_call(
      ndm_pkg,
      "ndm_build_model",
      runtime_env = runtime_env,
      model_type = model_type,
      model_spec = model_spec,
      backbone = "transformer"
    )
    analysis2_expose_runtime_env(runtime_env)
    analysis2_call(ndm_pkg, "ndm_source_runtime_calibration", analysis_root = paths$analysis_root, env = runtime_env)
    analysis2_expose_runtime_env(runtime_env)
    trained <- analysis2_call(
      ndm_pkg,
      "ndm_train",
      x = model,
      run_define = TRUE,
      run_loop = TRUE
    )
    analysis2_expose_runtime_env(trained$env)
    analysis2_call(ndm_pkg, "ndm_source_runtime_results_get", analysis_root = paths$analysis_root, env = trained$env)

    if (isTRUE(spec$run_figures)) {
      analysis2_expose_runtime_env(trained$env)
      analysis2_call(ndm_pkg, "ndm_source_runtime_results_analyze", analysis_root = paths$analysis_root, env = trained$env)
    }

    analysis2_log(sprintf("Done with sim outer iteration %s", outer_iteration))
  }

  invisible(TRUE)
}

analysis2_run_real_multidisease <- function(args = commandArgs(TRUE)) {
  old_error <- getOption("error")
  old_wd <- getwd()
  on.exit(options(error = old_error), add = TRUE)
  on.exit(setwd(old_wd), add = TRUE)
  options(error = NULL)
  analysis2_log("Starting Analysis2 SuperLModel_MasterReal_MultiDisease.R")
  spec <- analysis2_build_run_spec("multidisease", args)
  if (isTRUE(spec$help)) {
    return(analysis2_print_usage("multidisease", paths = spec$paths))
  }

  paths <- spec$paths
  grid_file <- normalizePath(spec$grid_file, winslash = "/", mustWork = TRUE)
  real_grid <- analysis2_order_grid(as.data.frame(data.table::fread(grid_file)), spec$outer)
  analysis2_validate_outer_iterations(real_grid, spec$outer, grid_file)

  if (isTRUE(spec$dry_run)) {
    return(analysis2_dry_run_result(spec, real_grid))
  }

  setwd(paths$project_root)
  analysis2_prepare_output_roots(paths$project_root, sim_mode = FALSE)
  holder_folder <- file.path(paths$project_root, "SavedResults", "Real", sprintf("Results_%s", spec$analysis_name))
  analysis2_dir_create(holder_folder)

  driver_env <- new.env(parent = globalenv())
  driver_env$analysis2_as_int <- analysis2_as_int
  driver_env$analysis2_small_run_n_checkpoints <- analysis2_small_run_n_checkpoints
  driver_env$analysis2_small_run_n_obs_inference <- analysis2_small_run_n_obs_inference
  driver_env$analysis2_model_type <- analysis2_model_type
  driver_env$analysis2_multidisease_spec <- spec
  driver_env$analysis2_multidisease_grid <- real_grid
  source(
    file.path(paths$analysis_root, "SetupEnv", "Analysis2_legacy_multidisease_driver.R"),
    local = driver_env,
    chdir = FALSE
  )

  if (!exists("analysis2_multidisease_result", envir = driver_env, inherits = FALSE)) {
    stop("Legacy multidisease driver completed without assigning `analysis2_multidisease_result`.", call. = FALSE)
  }
  result <- get("analysis2_multidisease_result", envir = driver_env, inherits = FALSE)
  if (!isTRUE(result)) {
    stop("Legacy multidisease driver did not report successful completion.", call. = FALSE)
  }

  invisible(result)
}
