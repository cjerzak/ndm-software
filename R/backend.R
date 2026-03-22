.ndm_backend_sys_info <- function() {
  Sys.info()
}

.ndm_backend_system <- function(...) {
  system(...)
}

.ndm_backend_conda_create <- function(...) {
  reticulate::conda_create(...)
}

.ndm_backend_py_install <- function(...) {
  reticulate::py_install(...)
}

.ndm_backend_conda_list <- function(...) {
  reticulate::conda_list(...)
}

.ndm_backend_use_condaenv <- function(...) {
  reticulate::use_condaenv(...)
}

.ndm_backend_py_available <- function(...) {
  reticulate::py_available(...)
}

.ndm_backend_py_module_available <- function(...) {
  reticulate::py_module_available(...)
}

.ndm_backend_import <- function(...) {
  reticulate::import(...)
}

.ndm_backend_jax_install_plan <- function(os_name, machine, driver_major = NA_integer_) {
  if (identical(os_name, "Darwin") && identical(machine, "arm64")) {
    return(list(
      message = "Apple Silicon detected: installing CPU-only JAX. Current official JAX docs mark Apple GPU support as experimental rather than the default install path.",
      packages = "jax",
      fallback_packages = NULL
    ))
  }

  if (identical(os_name, "Linux")) {
    if (!is.na(driver_major) && driver_major >= 580) {
      return(list(
        message = sprintf("Driver %s detected (>=580): installing JAX CUDA 13 wheels.", driver_major),
        packages = "jax[cuda13]",
        fallback_packages = "jax[cuda12]"
      ))
    }

    if (!is.na(driver_major) && driver_major >= 525) {
      return(list(
        message = sprintf("Driver %s detected (>=525,<580): installing JAX CUDA 12 wheels.", driver_major),
        packages = "jax[cuda12]",
        fallback_packages = NULL
      ))
    }
  }

  list(
    message = "Installing CPU-only JAX.",
    packages = "jax",
    fallback_packages = NULL
  )
}

#' Provision and initialize the Python backend
#'
#' These helpers manage the reticulate-backed Python environment used by
#' ndm. Use `ndm_build_backend()` to provision dependencies,
#' `ndm_check_backend()` to verify availability, and
#' `ndm_initialize_backend()` to import the active JAX stack into R.
#'
#' @details
#' When you want backend helpers and the package-native `ndm_run_*()` runners to
#' target the same conda environment, set `NDM_SOFTWARE_CONDA_ENV` (or
#' `NDM_CONDA_ENV`) once and pass that value into these helpers. The runner
#' layer currently falls back to `jax_cpu` on macOS and `ndm_software_env`
#' elsewhere when neither environment variable is set.
#'
#' @param conda_env Name of the conda environment that should contain the JAX
#'   runtime.
#' @param conda Conda executable to use. `"auto"` delegates discovery to
#'   reticulate.
#' @param modules Character vector of Python module names that must be importable
#'   for the backend to be considered healthy.
#'
#' @returns `ndm_check_backend()` returns `TRUE` invisibly when the requested
#'   environment and modules are available. It returns `NULL` after printing a
#'   diagnostic message when the backend is not ready.
#'
#' @examples
#' ndm_check_backend()
#'
#' @export
ndm_check_backend <- function(conda_env = "ndm_software_env",
                              conda = "auto",
                              modules = c("jax", "numpy", "optax", "equinox", "diffrax")) {
  try_condaenv <- try(
    .ndm_backend_use_condaenv(conda_env, required = TRUE, conda = conda),
    silent = TRUE
  )

  if (inherits(try_condaenv, "try-error")) {
    message(
      "Conda environment is not available. Build it with ",
      "ndm::ndm_build_backend(conda_env = '", conda_env, "', conda = '", conda, "')."
    )
    return(NULL)
  }

  if (!.ndm_backend_py_available(initialize = TRUE)) {
    message(
      "Python is not available. Build the backend with ",
      "ndm::ndm_build_backend(conda_env = '", conda_env, "', conda = '", conda, "')."
    )
    return(NULL)
  }

  missing_modules <- modules[!vapply(modules, .ndm_backend_py_module_available, logical(1))]
  if (length(missing_modules) > 0L) {
    message(
      "Missing Python modules: ", paste(missing_modules, collapse = ", "),
      ". Build the backend with ndm::ndm_build_backend()."
    )
    return(NULL)
  }

  invisible(TRUE)
}

#' @rdname ndm_check_backend
#'
#' @param python_version Python version requested when creating the conda
#'   environment.
#' @param include_tensorflow Logical scalar indicating whether TensorFlow should
#'   be installed alongside JAX.
#' @param extra_packages Additional Python packages to install with `pip`.
#'
#' @returns `ndm_build_backend()` invisibly returns the conda environment name
#'   after attempting to provision the requested packages.
#'
#' @examples
#' \dontrun{
#' ndm_build_backend()
#' }
#'
#' @export
ndm_build_backend <- function(conda_env = "ndm_software_env",
                              conda = "auto",
                              python_version = "3.13",
                              include_tensorflow = TRUE,
                              extra_packages = c("optax", "equinox", "diffrax")) {
  .ndm_backend_conda_create(
    envname = conda_env,
    conda = conda,
    python_version = python_version
  )

  sys_info <- .ndm_backend_sys_info()
  os_name <- sys_info[["sysname"]]
  machine <- sys_info[["machine"]]
  msg <- function(...) message(sprintf(...))
  pip_install <- function(pkgs) {
    .ndm_backend_py_install(
      packages = pkgs,
      envname = conda_env,
      conda = conda,
      pip = TRUE
    )
    invisible(TRUE)
  }

  pip_install("numpy")

  install_jax <- function() {
    drv_major <- NA_integer_
    if (identical(os_name, "Linux")) {
      drv <- try(
        suppressWarnings(
          .ndm_backend_system(
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1",
            intern = TRUE
          )
        ),
        silent = TRUE
      )
      drv_major <- suppressWarnings(as.integer(sub("^([0-9]+).*", "\\1", drv[1])))
    }
    install_plan <- .ndm_backend_jax_install_plan(
      os_name = os_name,
      machine = machine,
      driver_major = drv_major
    )
    msg("%s", install_plan$message)
    tryCatch(
      pip_install(install_plan$packages),
      error = function(e) {
        if (is.null(install_plan$fallback_packages)) {
          stop(e)
        }
        msg(
          "Primary JAX install failed (%s); falling back to %s.",
          e$message,
          install_plan$fallback_packages
        )
        pip_install(install_plan$fallback_packages)
      }
    )
  }

  install_jax()

  packages <- extra_packages
  if (isTRUE(include_tensorflow)) {
    packages <- c(packages, "tensorflow")
  }
  packages <- unique(packages)
  if (length(packages) > 0L) {
    pip_install(packages)
  }

  if (identical(os_name, "Linux")) {
    try({
      conda_envs <- .ndm_backend_conda_list(conda = conda)
      env_python <- conda_envs$python[conda_envs$name == conda_env]
      if (length(env_python) > 0L) {
        env_path <- dirname(dirname(env_python[1]))
        actdir <- file.path(env_path, "etc", "conda", "activate.d")
        dir.create(actdir, recursive = TRUE, showWarnings = FALSE)
        writeLines("unset LD_LIBRARY_PATH", file.path(actdir, "00-unset-ld.sh"))
      }
    }, silent = TRUE)
  }

  msg("Environment '%s' is ready.", conda_env)
  invisible(conda_env)
}

#' @rdname ndm_check_backend
#'
#' @param conda_env_required Passed through to `reticulate::use_condaenv()` as
#'   the `required` flag.
#' @param float_type Floating point precision used when configuring JAX. Either
#'   `"32"` or `"64"`.
#' @param import_tensorflow Logical scalar indicating whether TensorFlow should
#'   be imported when it is available in the selected environment.
#'
#' @returns `ndm_initialize_backend()` returns an object of class
#'   `ndm_backend`. The returned list contains imported Python modules, the
#'   configured float type, and a small compatibility shim used by the legacy
#'   runtime.
#'
#' @examples
#' \dontrun{
#' backend <- ndm_initialize_backend()
#' backend$default_backend
#' }
#'
#' @export
ndm_initialize_backend <- function(conda_env = "ndm_software_env",
                                   conda = "auto",
                                   conda_env_required = TRUE,
                                   float_type = c("32", "64"),
                                   import_tensorflow = TRUE) {
  float_type <- match.arg(float_type)

  if (!nzchar(Sys.getenv("JAX_PLATFORMS", unset = "")) &&
      grepl("cpu", conda_env, ignore.case = TRUE)) {
    Sys.setenv(JAX_PLATFORMS = "cpu")
  }

  .ndm_backend_use_condaenv(
    condaenv = conda_env,
    required = conda_env_required,
    conda = conda
  )

  jax <- .ndm_backend_import("jax")
  jnp <- .ndm_backend_import("jax.numpy")
  np <- .ndm_backend_import("numpy")
  optax <- .ndm_backend_import("optax")
  eq <- .ndm_backend_import("equinox")
  diffrax <- .ndm_backend_import("diffrax")
  flash_mha <- try(.ndm_backend_import("flash_attn_jax")$flash_mha, silent = TRUE)
  py_gc <- try(.ndm_backend_import("gc"), silent = TRUE)
  tf <- if (isTRUE(import_tensorflow) && .ndm_backend_py_module_available("tensorflow")) {
    .ndm_backend_import("tensorflow")
  } else {
    NULL
  }

  if (identical(float_type, "64")) {
    jax$config$update("jax_enable_x64", TRUE)
    jax_float_type <- jnp$float64
  } else {
    jax$config$update("jax_enable_x64", FALSE)
    jax_float_type <- jnp$float32
  }

  default_backend <- tolower(jax$default_backend())
  send2cpu <- function(x) x
  send2gpu <- function(x) x

  if (!identical(default_backend, "cpu")) {
    send2cpu <- function(x) jax$device_put(x, jax$devices("cpu")[[1]])
    send2gpu <- function(x) {
      target_platform <- if (identical(Sys.info()[["machine"]], "arm64")) "METAL" else "gpu"
      jax$device_put(x, jax$devices(target_platform)[[1]])
    }
  }

  backend <- list(
    jax = jax,
    jnp = jnp,
    np = np,
    optax = optax,
    eq = eq,
    diffrax = diffrax,
    flash_mha = if (inherits(flash_mha, "try-error")) NULL else flash_mha,
    py_gc = if (inherits(py_gc, "try-error")) NULL else py_gc,
    tf = tf,
    jaxFloatType = jax_float_type,
    float_type = float_type,
    default_backend = default_backend,
    send2cpu = send2cpu,
    send2gpu = send2gpu
  )

  backend$JaxKey <- function(int_) {
    backend$jax$random$PRNGKey(as.integer(int_))
  }
  backend$SoftPlus <- backend$jax$nn$softplus
  backend$Sigmoid <- backend$jax$nn$sigmoid
  backend$InvSoftPlus <- function(z) {
    backend$jnp$log(backend$jnp$subtract(backend$jnp$exp(z), backend$jnp$array(1)))
  }
  backend$TFConst2JAXArray <- function(x) {
    lapply(x, function(value) backend$jnp$array(value))
  }
  backend$batch2package <- function(batch) {
    ndm_batch_to_model_inputs(batch)
  }
  backend$oryx <- ndm_make_oryx_shim(
    jax = backend$jax,
    jnp = backend$jnp,
    np = backend$np
  )

  class(backend) <- c("ndm_backend", "list")
  ndm_env$backend <- backend
  invisible(backend)
}

#' @rdname ndm_check_backend
#'
#' @returns `ndm_backend_modules()` returns the currently initialized
#'   `ndm_backend` object.
#'
#' @examples
#' \dontrun{
#' ndm_initialize_backend()
#' ndm_backend_modules()
#' }
#'
#' @export
ndm_backend_modules <- function() {
  if (is.null(ndm_env$backend)) {
    stop("Backend not initialized. Run ndm_initialize_backend() first.", call. = FALSE)
  }
  ndm_env$backend
}
