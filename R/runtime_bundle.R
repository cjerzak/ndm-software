#' Inspect the vendored runtime bundle
#'
#' ndm vendors the active Phase 1 runtime files under `inst/extracted`
#' so the package can load the historical analysis code without depending on an
#' external checkout.
#'
#' @param analysis_root Retained for API compatibility. The current
#'   implementation always resolves paths from the vendored runtime bundle.
#'
#' @returns `ndm_runtime_paths()` returns a named list of normalized file paths
#'   for the extracted runtime bundle.
#'
#' @examples
#' paths <- ndm_runtime_paths()
#' names(paths)
#'
#' @export
ndm_runtime_paths <- function(analysis_root = .ndm_default_analysis_root()) {
  analysis_root <- .ndm_extracted_analysis_dir()
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
  missing <- check_names[!file.exists(unlist(paths[check_names], use.names = FALSE))]
  if (length(missing) > 0L) {
    stop("Missing legacy analysis files: ", paste(missing, collapse = ", "), call. = FALSE)
  }

  paths
}

.ndm_install_runtime_helpers <- function(env) {
  stopifnot(is.environment(env))

  assign("NDM_INTERNAL_ANALYSIS_DIR", .ndm_extracted_analysis_dir(), envir = env)
  assign(
    "ndm_source_extracted",
    function(relative_path, env_target = NULL, ...) {
      if (is.null(env_target)) {
        env_target <- parent.frame()
        if (identical(env_target, globalenv())) {
          env_target <- env
        }
      }
      sys.source(
        file.path(get("NDM_INTERNAL_ANALYSIS_DIR", envir = env, inherits = FALSE), relative_path),
        envir = env_target,
        keep.source = FALSE
      )
      invisible(env_target)
    },
    envir = env
  )

  invisible(env)
}

#' Create and populate runtime environments
#'
#' These helpers manage isolated environments used to source the vendored model
#' runtime and seed it with R values.
#'
#' @param parent Parent environment for a new runtime environment.
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
ndm_new_runtime_env <- function(parent = baseenv()) {
  env <- new.env(parent = parent)
  class(env) <- c("ndm_runtime_env", class(env))
  .ndm_install_runtime_helpers(env)
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
  sys.source(path, envir = env, keep.source = FALSE)
  invisible(env)
}

#' Source vendored runtime components
#'
#' These helpers source the extracted legacy runtime files into a dedicated
#' environment. They are lower-level building blocks used by
#' `ndm_prepare_runtime()` and `ndm_fit()`.
#'
#' @inheritParams ndm_runtime_paths
#' @param env Runtime environment that should receive the sourced objects.
#' @param float_type Floating point precision passed into the vendored backend
#'   bootstrap code. Use `"32"` or `"64"`.
#' @param force_to_gpu Logical scalar indicating whether the runtime should try
#'   to place arrays on a GPU device when available.
#' @param gpu_mem_frac Optional GPU memory fraction forwarded to the vendored
#'   backend bootstrap code.
#' @param resave_tfrecords Logical scalar preserved for compatibility with the
#'   legacy runtime setup.
#' @param generator Which extracted data generator script to source.
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
#' @export
ndm_source_runtime_helper_fxns <- function(analysis_root = .ndm_default_analysis_root(),
                                          env = ndm_new_runtime_env()) {
  .ndm_install_runtime_helpers(env)
  paths <- ndm_runtime_paths(analysis_root)
  .ndm_source_runtime_file(paths$helper_fxns, env)
}

#' @rdname ndm_source_runtime_helper_fxns
#' @export
ndm_source_runtime_backend <- function(analysis_root = .ndm_default_analysis_root(),
                                      env = ndm_new_runtime_env(),
                                      float_type = "32",
                                      force_to_gpu = TRUE,
                                      gpu_mem_frac = NULL,
                                      resave_tfrecords = FALSE) {
  .ndm_install_runtime_helpers(env)
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
#' @export
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
#' @export
ndm_source_runtime_data <- function(analysis_root = .ndm_default_analysis_root(),
                                   env = ndm_new_runtime_env(),
                                   generator = c("sim", "real")) {
  .ndm_install_runtime_helpers(env)
  generator <- match.arg(generator)
  paths <- ndm_runtime_paths(analysis_root)
  source_path <- if (identical(generator, "sim")) paths$data_sim else paths$data_real
  .ndm_source_runtime_file(source_path, env)
}

#' @rdname ndm_source_runtime_helper_fxns
#' @export
ndm_source_runtime_calibration <- function(analysis_root = .ndm_default_analysis_root(),
                                          env = ndm_new_runtime_env()) {
  .ndm_install_runtime_helpers(env)
  paths <- ndm_runtime_paths(analysis_root)
  .ndm_source_runtime_file(paths$calibrate_ml, env)
}

#' @rdname ndm_source_runtime_helper_fxns
#' @export
ndm_source_runtime_results_get <- function(analysis_root = .ndm_default_analysis_root(),
                                          env = ndm_new_runtime_env()) {
  .ndm_install_runtime_helpers(env)
  paths <- ndm_runtime_paths(analysis_root)
  .ndm_source_runtime_file(paths$results_get, env)
}

#' @rdname ndm_source_runtime_helper_fxns
#' @export
ndm_source_runtime_results_analyze <- function(analysis_root = .ndm_default_analysis_root(),
                                              env = ndm_new_runtime_env()) {
  .ndm_install_runtime_helpers(env)
  paths <- ndm_runtime_paths(analysis_root)
  .ndm_source_runtime_file(paths$results_analyze, env)
}
