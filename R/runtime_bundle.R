#' Inspect a package-managed runtime bundle
#'
#' `ndm` now keeps its maintained runtime sources in package code and the
#' development `tools/runtime_source` tree rather than shipping an executable
#' snapshot from `inst/extdata`.
#'
#' @param analysis_root Optional runtime root override. When `NULL`, `ndm`
#'   resolves the package-managed runtime bundle.
#'
#' @returns `ndm_runtime_paths()` returns a named list of normalized file paths
#'   for the requested local runtime bundle.
#'
#' @examples
#' \dontrun{
#' paths <- ndm_runtime_paths()
#' names(paths)
#' }
#'
#' @noRd
.ndm_runtime_manifest_sources <- list(
  "config/real.yaml" = paste(
    c(
      "# Default manifest for package-backed real-data runs.",
      "mode: real",
      "analysis_name: RealApril15",
      "grid_file: Data/RunGrids/RealGrids/RealGrid_RealApril15.csv",
      "outer:",
      "  - 3",
      "model_type: null",
      "respect_grid_model_type: true",
      "run_seed: null",
      "force_to_gpu: true",
      "gpu_mem_frac: null",
      "neuralode_kl_weight: 1.0",
      "neuralode_mean_loss_weight: 0.0",
      "resave_tfrecords: false",
      "run_figures: false",
      "raw_data_dir: Data/MainData",
      "tfrecord_dir: Data/RunTFRecords/RealTFRecords/RealApril15",
      "outcome_metric: inc_death",
      "data_subset: high_income",
      "dry_run: false"
    ),
    collapse = "\n"
  ),
  "config/sim.yaml" = paste(
    c(
      "# Default manifest for package-backed simulation runs.",
      "mode: sim",
      "analysis_name: BigSimsLatest",
      "grid_file: Data/RunGrids/SimGrids/SimGrid_BigSimsLatest.csv",
      "outer:",
      "  - 1",
      "model_type: null",
      "respect_grid_model_type: true",
      "run_seed: null",
      "force_to_gpu: true",
      "gpu_mem_frac: null",
      "neuralode_kl_weight: 1.0",
      "neuralode_mean_loss_weight: 0.0",
      "resave_tfrecords: false",
      "run_figures: false",
      "tfrecord_dir: Data/RunTFRecords/SimTFRecords/BigSimsLatest",
      "dry_run: false"
    ),
    collapse = "\n"
  ),
  "config/real_multidisease.yaml" = paste(
    c(
      "# Default manifest for the package-backed multidisease real-data driver.",
      "mode: multidisease",
      "analysis_name: RealLatest",
      "grid_file: Data/RunGrids/RealGrids/RealGrid_RealLatest.csv",
      "outer:",
      "  - 1",
      "model_type: null",
      "respect_grid_model_type: true",
      "run_seed: null",
      "force_to_gpu: true",
      "gpu_mem_frac: null",
      "neuralode_kl_weight: 1.0",
      "neuralode_mean_loss_weight: 0.0",
      "resave_tfrecords: false",
      "run_figures: false",
      "tfrecord_dir: Data/RunTFRecords/RealTFRecords/RealLatest",
      "outcome_metric: CountValue",
      "data_subset: all",
      "data_format: IHME",
      "disease_names:",
      "  - Covid",
      "  - Flu",
      "dry_run: false"
    ),
    collapse = "\n"
  )
)

.ndm_runtime_source_root <- function() {
  candidate <- file.path(.ndm_package_root(), "tools", "runtime_source", "ndm_runtime")
  if (!dir.exists(candidate)) {
    return(NULL)
  }
  .ndm_normalize_path(candidate, must_work = TRUE)
}

.ndm_runtime_special_exprs <- function() {
  list(
    "SetupEnv/Analysis2_api.R" = .ndm_run_impl_expr,
    "SetupEnv/Analysis2_legacy_multidisease_driver.R" = .ndm_multidisease_driver_expr
  )
}

.ndm_runtime_expr_name <- function(relative_path) {
  sprintf(
    ".ndm_stage_expr_%s",
    gsub("[^A-Za-z0-9]+", "_", sub("\\.R$", "", relative_path))
  )
}

.ndm_runtime_relative_catalog <- function() {
  c(names(.ndm_runtime_stage_registry), names(.ndm_runtime_special_exprs()))
}

.ndm_embedded_runtime_key <- function(path) {
  .ndm_embedded_match_key(
    path = .ndm_normalize_runtime_relative(path),
    keys = .ndm_runtime_relative_catalog(),
    label = "runtime"
  )
}

.ndm_runtime_expression <- function(path) {
  key <- .ndm_embedded_runtime_key(path)
  special <- .ndm_runtime_special_exprs()
  if (key %in% names(special)) {
    return(special[[key]])
  }
  get(.ndm_runtime_expr_name(key), envir = asNamespace("ndm"), inherits = FALSE)
}

.ndm_runtime_deparse_lines <- function(expr) {
  unlist(
    lapply(as.list(expr), function(node) {
      deparse(node, width.cutoff = 500L)
    }),
    use.names = FALSE
  )
}

.ndm_embedded_runtime_source <- function(path) {
  paste(.ndm_embedded_runtime_lines(path), collapse = "\n")
}

.ndm_embedded_runtime_lines <- function(path) {
  key <- .ndm_embedded_runtime_key(path)
  source_root <- .ndm_runtime_source_root()
  if (!is.null(source_root)) {
    candidate <- file.path(source_root, key)
    if (file.exists(candidate)) {
      return(readLines(candidate, warn = FALSE))
    }
  }
  .ndm_runtime_deparse_lines(.ndm_runtime_expression(key))
}

.ndm_runtime_tree_write_file <- function(root, relative_path, lines) {
  target <- file.path(root, relative_path)
  dir.create(dirname(target), recursive = TRUE, showWarnings = FALSE)
  writeLines(lines, target, useBytes = TRUE)
  invisible(target)
}

.ndm_materialize_runtime_tree <- function(root) {
  stopifnot(is.character(root), length(root) == 1L)

  dir.create(root, recursive = TRUE, showWarnings = FALSE)

  for (relative_path in .ndm_runtime_relative_catalog()) {
    .ndm_runtime_tree_write_file(
      root,
      relative_path,
      .ndm_embedded_runtime_lines(relative_path)
    )
  }

  for (relative_path in names(.ndm_runtime_manifest_sources)) {
    .ndm_runtime_tree_write_file(
      root,
      relative_path,
      strsplit(.ndm_runtime_manifest_sources[[relative_path]], "\n", fixed = TRUE)[[1L]]
    )
  }

  for (relative_path in names(.ndm_embedded_model_spec_sources)) {
    .ndm_runtime_tree_write_file(
      root,
      file.path("ModelStructureTex", basename(relative_path)),
      strsplit(.ndm_embedded_model_spec_sources[[relative_path]], "\n", fixed = TRUE)[[1L]]
    )
  }

  invisible(.ndm_normalize_path(root, must_work = TRUE))
}

.ndm_internal_analysis_root <- function(refresh = FALSE) {
  source_root <- .ndm_runtime_source_root()
  if (!isTRUE(refresh) && !is.null(source_root)) {
    return(source_root)
  }

  cached <- ndm_env$embedded_analysis_root %||% NULL
  if (!isTRUE(refresh) && !is.null(cached) && dir.exists(cached)) {
    return(.ndm_normalize_path(cached, must_work = TRUE))
  }

  root <- file.path(tempdir(), "ndm_package_runtime", "ndm_runtime")
  .ndm_materialize_runtime_tree(root)
  ndm_env$embedded_analysis_root <- root
  .ndm_normalize_path(root, must_work = TRUE)
}

.ndm_embedded_analysis_root <- function(refresh = FALSE) {
  .ndm_internal_analysis_root(refresh = refresh)
}

.ndm_eval_embedded_runtime <- function(path, env) {
  stopifnot(is.environment(env))

  key <- .ndm_embedded_runtime_key(path)
  source_root <- .ndm_runtime_source_root()
  if (!is.null(source_root)) {
    candidate <- file.path(source_root, key)
    if (file.exists(candidate)) {
      sys.source(candidate, envir = env, keep.source = FALSE)
      return(invisible(env))
    }
  }

  .ndm_eval_runtime_stage_expr(.ndm_runtime_expression(key), env)
}

.ndm_normalize_runtime_relative <- function(path) {
  path <- gsub("\\\\", "/", as.character(path))
  path <- sub("^\\./", "", path)
  path <- sub("^.*/Analysis2/", "", path)
  path <- sub("^.*/ndm_runtime/", "", path)
  path
}

.ndm_has_embedded_runtime_relative <- function(relative_path) {
  rel <- .ndm_normalize_runtime_relative(relative_path)
  !inherits(try(.ndm_embedded_runtime_key(rel), silent = TRUE), "try-error")
}

.ndm_eval_embedded_runtime_relative <- function(relative_path, env) {
  .ndm_eval_embedded_runtime(.ndm_normalize_runtime_relative(relative_path), env)
}

.ndm_eval_runtime_path <- function(path, env, analysis_root = NULL) {
  rel <- .ndm_normalize_runtime_relative(path)
  if (.ndm_has_embedded_runtime_relative(rel)) {
    return(.ndm_eval_embedded_runtime_relative(rel, env))
  }

  if (!is.null(analysis_root) && nzchar(analysis_root)) {
    candidate <- file.path(analysis_root, rel)
    if (file.exists(candidate)) {
      sys.source(candidate, envir = env, keep.source = FALSE)
      return(invisible(env))
    }
  }

  if (file.exists(path)) {
    sys.source(path, envir = env, keep.source = FALSE)
    return(invisible(env))
  }

  stop(
    "Runtime component could not be resolved from the package-owned registry or filesystem: ",
    path,
    call. = FALSE
  )
}

ndm_runtime_paths <- function(analysis_root = .ndm_default_analysis_root()) {
  analysis_root <- .ndm_resolve_analysis_root(analysis_root, must_work = FALSE)
  if (is.null(analysis_root)) {
    analysis_root <- .ndm_internal_analysis_root()
  }
  project_root <- dirname(analysis_root)

  paths <- list(
    analysis_root = analysis_root,
    project_root = project_root,
    helper_fxns = file.path(analysis_root, "SetupEnv", "SuperLModel_helperFxns.R"),
    master_imports = file.path(analysis_root, "SetupEnv", "SuperLModel_MasterImports.R"),
    calibrate_ml = file.path(analysis_root, "SetupData", "SuperLModel_CalibrateML.R"),
    data_sim = file.path(analysis_root, "SetupData", "SuperLModel_DataGenerator_Sim.R"),
    data_real = file.path(analysis_root, "SetupData", "SuperLModel_DataGenerator_Real.R"),
    build_model = file.path(analysis_root, "ModelDefiners", "SuperLModel_BuildML.R"),
    train_define = file.path(analysis_root, "ModelTrainers", "SuperLModel_TrainDefine.R"),
    train_do = file.path(analysis_root, "ModelTrainers", "SuperLModel_TrainDo.R"),
    results_get = file.path(analysis_root, "ResultsGet", "SuperLModel_GetAnalytics.R"),
    results_analyze = file.path(analysis_root, "ResultsAnalyze", "SuperLModel_GenFigs.R"),
    model_structure_dir = file.path(analysis_root, "ModelStructureTex")
  )

  check_names <- setdiff(names(paths), c("project_root", "analysis_root"))
  runtime_relative <- c(
    helper_fxns = "SetupEnv/SuperLModel_helperFxns.R",
    master_imports = "SetupEnv/SuperLModel_MasterImports.R",
    calibrate_ml = "SetupData/SuperLModel_CalibrateML.R",
    data_sim = "SetupData/SuperLModel_DataGenerator_Sim.R",
    data_real = "SetupData/SuperLModel_DataGenerator_Real.R",
    build_model = "ModelDefiners/SuperLModel_BuildML.R",
    train_define = "ModelTrainers/SuperLModel_TrainDefine.R",
    train_do = "ModelTrainers/SuperLModel_TrainDo.R",
    results_get = "ResultsGet/SuperLModel_GetAnalytics.R",
    results_analyze = "ResultsAnalyze/SuperLModel_GenFigs.R"
  )
  missing <- check_names[!vapply(check_names, function(name) {
    file.exists(paths[[name]]) ||
      (.ndm_has_embedded_runtime_relative(runtime_relative[[name]]))
  }, logical(1))]
  if (length(missing) > 0L) {
    stop(
      "Missing runtime files under `analysis_root`: ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }

  paths
}

.ndm_install_runtime_helpers <- function(env, analysis_root = NULL) {
  stopifnot(is.environment(env))

  if (!is.null(analysis_root) && nzchar(analysis_root)) {
    analysis_root <- .ndm_resolve_analysis_root(analysis_root, must_work = TRUE)
  } else {
    analysis_root <- .ndm_internal_analysis_root()
  }

  assign("NDM_INTERNAL_ANALYSIS_DIR", analysis_root, envir = env)
  assign(
    "ndm_student_t_masked_nll",
    .ndm_student_t_masked_nll,
    envir = env
  )
  assign(
    "ndm_source_extracted",
    function(relative_path, env_target = NULL, ...) {
      if (is.null(env_target)) {
        env_target <- parent.frame()
        if (identical(env_target, globalenv())) {
          env_target <- env
        }
      }

      root <- get("NDM_INTERNAL_ANALYSIS_DIR", envir = env, inherits = FALSE)
      if ((is.null(root) || !nzchar(root)) &&
          !.ndm_has_embedded_runtime_relative(relative_path)) {
        stop(
          "`NDM_INTERNAL_ANALYSIS_DIR` is not set and the requested runtime component is not available: ",
          relative_path,
          call. = FALSE
        )
      }

      .ndm_eval_runtime_path(relative_path, env = env_target, analysis_root = root)
      invisible(env_target)
    },
    envir = env
  )

  invisible(env)
}

.ndm_namespace_available <- function(package) {
  requireNamespace(package, quietly = TRUE)
}

.ndm_require_namespaces <- function(packages,
                                    context,
                                    install_hint = "Install them before running this workflow.") {
  packages <- unique(as.character(packages))
  packages <- packages[nzchar(packages)]
  if (length(packages) == 0L) {
    return(invisible(TRUE))
  }

  missing <- packages[!vapply(packages, .ndm_namespace_available, logical(1))]
  if (length(missing) == 0L) {
    return(invisible(TRUE))
  }

  stop(
    "The following R packages are required for ",
    context,
    ": ",
    paste(missing, collapse = ", "),
    ". ",
    install_hint,
    call. = FALSE
  )
}

#' Create and populate runtime environments
#'
#' These helpers manage isolated environments used to load package-managed
#' runtime code and seed it with R values.
#'
#' @param parent Parent environment for a new runtime environment. Defaults to
#'   `globalenv()` so package-managed runtime code can resolve attached-package
#'   functions consistently.
#' @param env Runtime environment that should receive new bindings.
#' @param values Named list of values to assign into `env`.
#' @param overwrite Logical scalar indicating whether existing bindings in `env`
#'   should be overwritten.
#'
#' @returns `ndm_new_runtime_env()` returns an environment of class
#'   `ndm_runtime_env`. `ndm_set_runtime_globals()` invisibly returns `env`.
#'
#' @examples
#' env <- ndm_new_runtime_env()
#' ndm_set_runtime_globals(env, list(example_value = 1L))
#' env$example_value
#'
#' @export
ndm_new_runtime_env <- function(parent = globalenv()) {
  env <- new.env(parent = parent)
  class(env) <- c("ndm_runtime_env", class(env))
  .ndm_install_runtime_helpers(env, analysis_root = .ndm_internal_analysis_root())
  env
}

#' @rdname ndm_new_runtime_env
#' @export
ndm_set_runtime_globals <- function(env, values, overwrite = TRUE) {
  stopifnot(is.environment(env))
  if (length(values) == 0L) {
    return(invisible(env))
  }

  if (is.null(names(values)) || any(names(values) == "")) {
    stop("`values` must be a named list.", call. = FALSE)
  }

  for (name in names(values)) {
    if (!overwrite && exists(name, envir = env, inherits = FALSE)) {
      next
    }
    assign(name, values[[name]], envir = env)
  }

  invisible(env)
}

.ndm_source_runtime_file <- function(path, env) {
  analysis_root <- get0("NDM_INTERNAL_ANALYSIS_DIR", envir = env, inherits = FALSE, ifnotfound = NULL)
  .ndm_eval_runtime_path(path, env = env, analysis_root = analysis_root)
  invisible(env)
}

#' Source local runtime components
#'
#' These helpers load package-managed runtime stages into a dedicated
#' environment. They are lower-level building blocks used by
#' `ndm_prepare_runtime()` and `ndm_fit()`.
#'
#' @inheritParams ndm_runtime_paths
#' @param env Runtime environment that should receive the sourced objects.
#' @param float_type Floating point precision passed into the runtime bootstrap
#'   code. Use `"32"` or `"64"`.
#' @param force_to_gpu Logical scalar indicating whether the runtime should try
#'   to place arrays on a GPU device when available.
#' @param gpu_mem_frac Optional GPU memory fraction forwarded to the runtime
#'   bootstrap code.
#' @param resave_tfrecords Logical scalar preserved for compatibility with
#'   existing run workflows.
#' @param generator Which data generator script to source.
#'
#' @returns Each function on this page invisibly returns `env` after sourcing the
#'   requested runtime component.
#'
#' @examples
#' \dontrun{
#' env <- ndm_new_runtime_env()
#' ndm_load_runtime(env = env)
#' }
#'
#' @noRd
ndm_source_runtime_helper_fxns <- function(analysis_root = .ndm_default_analysis_root(),
                                           env = ndm_new_runtime_env()) {
  .ndm_require_namespaces("fastmatch", context = "ndm_prepare_runtime()")
  .ndm_install_runtime_helpers(env, analysis_root = analysis_root)
  paths <- ndm_runtime_paths(analysis_root)
  .ndm_source_runtime_file(paths$helper_fxns, env)
}

#' @rdname ndm_source_runtime_helper_fxns
#' @noRd
ndm_source_runtime_backend <- function(analysis_root = .ndm_default_analysis_root(),
                                       env = ndm_new_runtime_env(),
                                       float_type = "32",
                                       force_to_gpu = TRUE,
                                       gpu_mem_frac = NULL,
                                       resave_tfrecords = FALSE) {
  .ndm_install_runtime_helpers(env, analysis_root = analysis_root)
  paths <- ndm_runtime_paths(analysis_root)
  ndm_set_runtime_globals(
    env,
    list(
      floatType = float_type,
      force2GPU = isTRUE(force_to_gpu),
      GPU_MEM_FRAC = gpu_mem_frac,
      ReSaveTfRecords = isTRUE(resave_tfrecords)
    )
  )
  .ndm_source_runtime_file(paths$master_imports, env)
}

#' @rdname ndm_source_runtime_helper_fxns
#' @noRd
ndm_load_runtime <- function(analysis_root = .ndm_default_analysis_root(),
                             env = ndm_new_runtime_env(),
                             float_type = "32",
                             force_to_gpu = TRUE,
                             gpu_mem_frac = NULL,
                             resave_tfrecords = FALSE) {
  ndm_source_runtime_helper_fxns(analysis_root = analysis_root, env = env)
  ndm_source_runtime_backend(
    analysis_root = analysis_root,
    env = env,
    float_type = float_type,
    force_to_gpu = force_to_gpu,
    gpu_mem_frac = gpu_mem_frac,
    resave_tfrecords = resave_tfrecords
  )
}

#' @rdname ndm_source_runtime_helper_fxns
#' @noRd
ndm_source_runtime_data <- function(analysis_root = .ndm_default_analysis_root(),
                                    env = ndm_new_runtime_env(),
                                    generator = c("sim", "real")) {
  .ndm_install_runtime_helpers(env, analysis_root = analysis_root)
  generator <- match.arg(generator)
  paths <- ndm_runtime_paths(analysis_root)
  source_path <- if (identical(generator, "sim")) paths$data_sim else paths$data_real
  .ndm_source_runtime_file(source_path, env)
}

#' Source Analysis2 compatibility runtime helpers
#'
#' These helpers preserve access to legacy Analysis2 runtime scripts that are
#' still referenced by package-native orchestration code. They are exported for
#' backward compatibility; prefer `ndm_prepare_runtime()` and higher-level run
#' helpers for new code.
#'
#' @param analysis_root Analysis root directory containing the legacy runtime
#'   scripts to source.
#' @param env Runtime environment that should receive the sourced objects.
#'
#' @returns Invisibly returns `env` after sourcing the requested runtime
#'   component.
#'
#' @seealso [ndm_prepare_runtime()], [ndm_fit()]
#' @keywords internal
#' @name ndm_source_runtime_compatibility
NULL

#' @rdname ndm_source_runtime_compatibility
#' @export
ndm_source_runtime_calibration <- function(analysis_root = .ndm_default_analysis_root(),
                                           env = ndm_new_runtime_env()) {
  .ndm_install_runtime_helpers(env, analysis_root = analysis_root)
  paths <- ndm_runtime_paths(analysis_root)
  .ndm_source_runtime_file(paths$calibrate_ml, env)
}

#' @rdname ndm_source_runtime_compatibility
#' @export
ndm_source_runtime_results_get <- function(analysis_root = .ndm_default_analysis_root(),
                                           env = ndm_new_runtime_env()) {
  .ndm_install_runtime_helpers(env, analysis_root = analysis_root)
  paths <- ndm_runtime_paths(analysis_root)
  .ndm_source_runtime_file(paths$results_get, env)
}

#' @rdname ndm_source_runtime_compatibility
#' @export
ndm_source_runtime_results_analyze <- function(analysis_root = .ndm_default_analysis_root(),
                                               env = ndm_new_runtime_env()) {
  .ndm_install_runtime_helpers(env, analysis_root = analysis_root)
  paths <- ndm_runtime_paths(analysis_root)
  .ndm_source_runtime_file(paths$results_analyze, env)
}
