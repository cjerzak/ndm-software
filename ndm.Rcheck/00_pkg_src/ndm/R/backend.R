#' Provision and initialize the Python backend
#'
#' These helpers manage the reticulate-backed Python environment used by
#' ndm. Use `ndm_build_backend()` to provision dependencies,
#' `ndm_check_backend()` to verify availability, and
#' `ndm_initialize_backend()` to import the active JAX stack into R.
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
    reticulate::use_condaenv(conda_env, required = TRUE, conda = conda),
    silent = TRUE
  )

  if (inherits(try_condaenv, "try-error")) {
    message(
      "Conda environment is not available. Build it with ",
      "ndm::ndm_build_backend(conda_env = '", conda_env, "', conda = '", conda, "')."
    )
    return(NULL)
  }

  if (!reticulate::py_available(initialize = TRUE)) {
    message(
      "Python is not available. Build the backend with ",
      "ndm::ndm_build_backend(conda_env = '", conda_env, "', conda = '", conda, "')."
    )
    return(NULL)
  }

  missing_modules <- modules[!vapply(modules, reticulate::py_module_available, logical(1))]
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
  reticulate::conda_create(
    envname = conda_env,
    conda = conda,
    python_version = python_version
  )

  os_name <- Sys.info()[["sysname"]]
  machine <- Sys.info()[["machine"]]
  msg <- function(...) message(sprintf(...))
  pip_install <- function(pkgs) {
    reticulate::py_install(
      packages = pkgs,
      envname = conda_env,
      conda = conda,
      pip = TRUE
    )
    invisible(TRUE)
  }

  pip_install("numpy")

  install_jax <- function() {
    if (identical(os_name, "Darwin") && identical(machine, "arm64")) {
      msg("Apple Silicon detected: installing JAX 0.5.0 with Metal support.")
      pip_install(c("jax==0.5.0", "jaxlib==0.5.0", "jax-metal==0.1.1"))
      return(invisible(TRUE))
    }

    if (identical(os_name, "Linux")) {
      drv <- try(
        suppressWarnings(
          system(
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1",
            intern = TRUE
          )
        ),
        silent = TRUE
      )
      drv_major <- suppressWarnings(as.integer(sub("^([0-9]+).*", "\\1", drv[1])))

      if (!is.na(drv_major) && drv_major >= 580) {
        msg("Driver %s detected (>=580): installing JAX CUDA 13 wheels.", drv[1])
        tryCatch(
          pip_install("jax[cuda13]"),
          error = function(e) {
            msg("CUDA 13 wheels failed (%s); falling back to CUDA 12.", e$message)
            pip_install("jax[cuda12]")
          }
        )
        return(invisible(TRUE))
      }

      if (!is.na(drv_major) && drv_major >= 525) {
        msg("Driver %s detected (>=525,<580): installing JAX CUDA 12 wheels.", drv[1])
        pip_install("jax[cuda12]")
        return(invisible(TRUE))
      }
    }

    msg("Installing CPU-only JAX.")
    pip_install("jax")
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
      conda_envs <- reticulate::conda_list(conda = conda)
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

  reticulate::use_condaenv(
    condaenv = conda_env,
    required = conda_env_required,
    conda = conda
  )

  jax <- reticulate::import("jax")
  jnp <- reticulate::import("jax.numpy")
  np <- reticulate::import("numpy")
  optax <- reticulate::import("optax")
  eq <- reticulate::import("equinox")
  diffrax <- reticulate::import("diffrax")
  flash_mha <- try(reticulate::import("flash_attn_jax")$flash_mha, silent = TRUE)
  py_gc <- try(reticulate::import("gc"), silent = TRUE)
  tf <- if (isTRUE(import_tensorflow) && reticulate::py_module_available("tensorflow")) {
    reticulate::import("tensorflow")
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
