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

.ndm_backend_conda_python <- function(...) {
  reticulate::conda_python(...)
}

.ndm_backend_system2 <- function(...) {
  system2(...)
}

.ndm_backend_python_path <- function(conda_env, conda = "auto") {
  python <- tryCatch(
    suppressWarnings(.ndm_backend_conda_python(conda_env, conda = conda)),
    error = function(e) ""
  )
  if (length(python) < 1L || !nzchar(python[[1L]])) {
    return("")
  }
  normalizePath(python[[1L]], winslash = "/", mustWork = FALSE)
}

.ndm_backend_jax_preloaded <- function() {
  if (!isTRUE(.ndm_backend_py_available(initialize = FALSE))) {
    return(FALSE)
  }
  sys <- .ndm_backend_import("sys", convert = FALSE)
  isTRUE(reticulate::py_to_r(sys$modules$`__contains__`("jax")))
}

.ndm_validate_preloaded_jax <- function(conda_env,
                                        conda,
                                        python_path,
                                        compute_backend,
                                        gpu_mem_frac) {
  if (!.ndm_backend_jax_preloaded()) {
    return(invisible(FALSE))
  }
  contract <- ndm_env$backend_process_contract
  if (is.null(contract) || !isTRUE(ndm_env$jax_initialized_by_ndm)) {
    stop(
      "JAX was imported before ndm could apply the requested compute policy. ",
      "Start a fresh R process and call `ndm_initialize_backend()` before any ",
      "direct Python/JAX imports.",
      call. = FALSE
    )
  }
  compatible_backend <- identical(compute_backend, "auto") ||
    identical(compute_backend, contract$compute_backend_resolved)
  current_platforms <- Sys.getenv("JAX_PLATFORMS", unset = "")
  current_mem_frac <- Sys.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION", unset = "")
  requested_mem_frac <- if (is.null(gpu_mem_frac)) "" else format(
    gpu_mem_frac,
    scientific = FALSE,
    trim = TRUE
  )
  contract_conda <- contract[["conda", exact = TRUE]] %||% conda
  contract_python_path <- contract[["python_path", exact = TRUE]] %||% python_path
  if (!compatible_backend ||
      !identical(as.character(conda_env), contract$conda_env) ||
      !identical(as.character(conda), contract_conda) ||
      (nzchar(python_path) && nzchar(contract_python_path) &&
       !identical(python_path, contract_python_path)) ||
      !identical(current_platforms, contract$jax_platforms) ||
      !identical(current_mem_frac, contract$gpu_mem_frac_env) ||
      !identical(requested_mem_frac, contract$gpu_mem_frac_env)) {
    stop(
      "The active JAX process does not match the requested ndm backend policy. ",
      "Start a fresh R process.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

.ndm_backend_probe_modules <- function(conda_env,
                                       conda = "auto",
                                       modules) {
  if (!is.character(modules) || length(modules) < 1L || anyNA(modules) ||
      any(!grepl("^[A-Za-z_][A-Za-z0-9_.]*$", modules))) {
    stop("`modules` must contain valid non-empty Python module names.", call. = FALSE)
  }
  python <- tryCatch(
    .ndm_backend_conda_python(conda_env, conda = conda),
    error = function(e) ""
  )
  if (length(python) < 1L || !nzchar(python[[1L]])) {
    return(list(
      ok = FALSE,
      missing = character(),
      detail = "could not resolve the conda environment's Python executable"
    ))
  }

  python_modules <- paste(vapply(modules, shQuote, character(1)), collapse = ",")
  script <- paste(
    "import importlib",
    paste0("modules=[", python_modules, "]"),
    "failed=[]",
    "for module in modules:",
    "    try:",
    "        importlib.import_module(module)",
    "    except Exception as exc:",
    "        failed.append((module, repr(exc)))",
    "if failed:",
    "    print('NDM_MODULES_MISSING:' + ','.join(item[0] for item in failed))",
    "    [print('NDM_MODULE_ERROR:' + item[0] + ':' + item[1]) for item in failed]",
    "else:",
    "    print('NDM_MODULES_OK')",
    "raise SystemExit(3 if failed else 0)",
    sep = "\n"
  )
  output <- tryCatch(
    suppressWarnings(
      .ndm_backend_system2(
        python[[1L]],
        args = c("-c", shQuote(script)),
        stdout = TRUE,
        stderr = TRUE
      )
    ),
    error = function(e) structure(e$message, status = 1L)
  )
  missing_line <- grep(
    "^NDM_MODULES_MISSING:",
    as.character(output),
    value = TRUE
  )
  missing <- if (length(missing_line) > 0L) {
    value <- sub("^NDM_MODULES_MISSING:", "", missing_line[[1L]])
    if (nzchar(value)) strsplit(value, ",", fixed = TRUE)[[1L]] else character()
  } else {
    character()
  }
  status <- attr(output, "status", exact = TRUE)
  list(
    ok = (is.null(status) || identical(as.integer(status), 0L)) &&
      any(as.character(output) == "NDM_MODULES_OK"),
    missing = missing,
    detail = paste(as.character(output), collapse = "\n")
  )
}

.ndm_validate_compute_backend <- function(compute_backend) {
  choices <- c("auto", "cpu", "gpu")
  if (!is.character(compute_backend) || length(compute_backend) != 1L ||
      is.na(compute_backend) || !nzchar(compute_backend)) {
    stop(
      "`compute_backend` must be one of 'auto', 'cpu', or 'gpu'.",
      call. = FALSE
    )
  }
  compute_backend <- tolower(compute_backend)
  if (!compute_backend %in% choices) {
    stop(
      "`compute_backend` must be one of 'auto', 'cpu', or 'gpu'.",
      call. = FALSE
    )
  }
  compute_backend
}

.ndm_warn_force_to_gpu_deprecated <- function() {
  if (!isTRUE(ndm_env$warned_force_to_gpu)) {
    warning(
      "`force_to_gpu` is deprecated; use `compute_backend = 'gpu'` or ",
      "`compute_backend = 'cpu'` instead.",
      call. = FALSE
    )
    ndm_env$warned_force_to_gpu <- TRUE
  }
  invisible(NULL)
}

.ndm_resolve_compute_backend <- function(compute_backend = "auto",
                                         force_to_gpu = NULL,
                                         compute_backend_supplied = TRUE,
                                         warn_deprecated = TRUE) {
  if (!isTRUE(compute_backend_supplied) &&
      identical(compute_backend, c("auto", "cpu", "gpu"))) {
    compute_backend <- "auto"
  }
  compute_backend <- .ndm_validate_compute_backend(compute_backend)
  if (is.null(force_to_gpu)) {
    return(compute_backend)
  }
  if (!is.logical(force_to_gpu) || length(force_to_gpu) != 1L ||
      is.na(force_to_gpu)) {
    stop("`force_to_gpu` must be one non-missing logical value or NULL.", call. = FALSE)
  }

  legacy_backend <- if (isTRUE(force_to_gpu)) "gpu" else "cpu"
  if (isTRUE(compute_backend_supplied) &&
      !identical(compute_backend, legacy_backend)) {
    stop(
      "Conflicting backend controls: `force_to_gpu = ",
      toupper(as.character(force_to_gpu)),
      "` maps to `compute_backend = '", legacy_backend,
      "'`, not `compute_backend = '", compute_backend, "'`.",
      call. = FALSE
    )
  }
  if (isTRUE(warn_deprecated)) {
    .ndm_warn_force_to_gpu_deprecated()
  }
  legacy_backend
}

.ndm_validate_gpu_mem_frac <- function(gpu_mem_frac,
                                       compute_backend = NULL) {
  if (is.null(gpu_mem_frac)) {
    return(NULL)
  }
  gpu_mem_frac <- suppressWarnings(as.numeric(gpu_mem_frac))
  if (length(gpu_mem_frac) != 1L || !is.finite(gpu_mem_frac) ||
      gpu_mem_frac <= 0 || gpu_mem_frac > 1) {
    stop("`gpu_mem_frac` must be NULL or one finite value in (0, 1].", call. = FALSE)
  }
  if (!is.null(compute_backend) && identical(compute_backend, "cpu")) {
    stop(
      "`gpu_mem_frac` cannot be used when the resolved compute backend is CPU.",
      call. = FALSE
    )
  }
  gpu_mem_frac[[1L]]
}

.ndm_cached_backend <- function(conda_env,
                                conda,
                                python_path,
                                float_type,
                                compute_backend,
                                import_tensorflow,
                                gpu_mem_frac) {
  cached <- ndm_env$backend
  if (is.null(cached)) {
    return(NULL)
  }
  cached_resolved <- cached$compute_backend_resolved %||%
    cached$compute_backend %||% cached$default_backend
  compatible_backend <- identical(compute_backend, "auto") ||
    identical(compute_backend, cached_resolved)
  if (!compatible_backend) {
    stop(
      "The current R process already initialized JAX on '", cached_resolved,
      "'; it cannot be reinitialized for `compute_backend = '", compute_backend,
      "'`. Start a fresh R process.",
      call. = FALSE
    )
  }
  if (!identical(as.character(float_type), as.character(cached$float_type))) {
    stop(
      "The current R process already initialized JAX with float_type = '",
      cached$float_type, "'. Start a fresh R process to change precision.",
      call. = FALSE
    )
  }
  cached_conda_env <- cached$conda_env %||% conda_env
  if (!identical(as.character(conda_env), as.character(cached_conda_env))) {
    stop(
      "The current R process already initialized JAX from conda environment '",
      cached_conda_env, "'. Start a fresh R process to use '", conda_env, "'.",
      call. = FALSE
    )
  }
  cached_conda <- cached[["conda", exact = TRUE]] %||% conda
  cached_python_path <- cached[["python_path", exact = TRUE]] %||% python_path
  if (!identical(as.character(conda), as.character(cached_conda)) ||
      (nzchar(python_path) && nzchar(cached_python_path) &&
       !identical(python_path, cached_python_path))) {
    stop(
      "The current R process already initialized JAX from a different Conda ",
      "installation or Python executable. Start a fresh R process.",
      call. = FALSE
    )
  }
  cached_import_tensorflow <- cached$import_tensorflow_requested %||%
    !is.null(cached$tf)
  if (isTRUE(import_tensorflow) && !isTRUE(cached_import_tensorflow)) {
    stop(
      "The cached backend was initialized without TensorFlow. Start a fresh R ",
      "process with `import_tensorflow = TRUE`.",
      call. = FALSE
    )
  }
  .ndm_validate_gpu_mem_frac(gpu_mem_frac, cached_resolved)
  cached_mem_frac <- cached$gpu_mem_frac %||% NULL
  if (!is.null(gpu_mem_frac) && !identical(gpu_mem_frac, cached_mem_frac)) {
    stop(
      "The current R process already initialized JAX with a different ",
      "`gpu_mem_frac`. Start a fresh R process.",
      call. = FALSE
    )
  }
  cached
}

.ndm_backend_probe_platform <- function(conda_env,
                                        conda = "auto",
                                        platform = c("cuda", "rocm", "cpu")) {
  platform <- match.arg(platform)
  python <- tryCatch(
    .ndm_backend_conda_python(conda_env, conda = conda),
    error = function(e) ""
  )
  if (length(python) < 1L || !nzchar(python[[1L]])) {
    return(list(
      ok = FALSE,
      platform = platform,
      detail = "could not resolve the conda environment's Python executable"
    ))
  }

  script <- paste(
    "import jax",
    sprintf("devices = jax.devices(%s)", shQuote(platform)),
    "assert len(devices) > 0",
    sprintf("print(%s)", shQuote(paste0("NDM_BACKEND_PROBE_OK:", platform))),
    sep = "; "
  )
  output <- tryCatch(
    suppressWarnings(
      .ndm_backend_system2(
        python[[1L]],
        args = c("-c", shQuote(script)),
        stdout = TRUE,
        stderr = TRUE,
        env = sprintf("JAX_PLATFORMS=%s,cpu", platform)
      )
    ),
    error = function(e) structure(e$message, status = 1L)
  )
  status <- attr(output, "status", exact = TRUE)
  ok <- (is.null(status) || identical(as.integer(status), 0L)) &&
    any(grepl(paste0("NDM_BACKEND_PROBE_OK:", platform), output, fixed = TRUE))
  list(
    ok = isTRUE(ok),
    platform = platform,
    detail = paste(as.character(output), collapse = "\n")
  )
}

.ndm_select_compute_backend <- function(compute_backend,
                                        conda_env,
                                        conda = "auto",
                                        sys_info = .ndm_backend_sys_info()) {
  requested <- .ndm_validate_compute_backend(compute_backend)
  os_name <- as.character(sys_info[["sysname"]] %||% "")

  if (identical(os_name, "Darwin")) {
    if (identical(requested, "gpu")) {
      stop(
        "`compute_backend = 'gpu'` is not supported on macOS. ",
        "Use `compute_backend = 'auto'` or 'cpu'; experimental JAX Metal ",
        "devices are not part of the supported backend contract.",
        call. = FALSE
      )
    }
    return(list(
      requested = requested,
      resolved = "cpu",
      accelerator_runtime = "cpu",
      probe = NULL
    ))
  }

  if (identical(requested, "cpu")) {
    return(list(
      requested = requested,
      resolved = "cpu",
      accelerator_runtime = "cpu",
      probe = NULL
    ))
  }

  probe_attempts <- lapply(c("cuda", "rocm"), function(runtime) {
    .ndm_backend_probe_platform(
      conda_env = conda_env,
      conda = conda,
      platform = runtime
    )
  })
  names(probe_attempts) <- c("cuda", "rocm")
  successful <- which(vapply(probe_attempts, function(x) isTRUE(x$ok), logical(1)))
  gpu_probe <- list(
    ok = length(successful) > 0L,
    attempts = probe_attempts,
    detail = paste(
      vapply(
        names(probe_attempts),
        function(runtime) {
          paste0(runtime, ": ", probe_attempts[[runtime]]$detail %||% "no detail")
        },
        character(1)
      ),
      collapse = "\n"
    )
  )
  if (length(successful) > 0L) {
    accelerator_runtime <- names(probe_attempts)[[successful[[1L]]]]
    return(list(
      requested = requested,
      resolved = "gpu",
      accelerator_runtime = accelerator_runtime,
      probe = gpu_probe
    ))
  }
  if (identical(requested, "gpu")) {
    stop(
      "`compute_backend = 'gpu'` requires a working JAX GPU device in conda ",
      "environment '", conda_env, "'. Isolated probe failed: ",
      gpu_probe$detail %||% "unknown probe failure",
      call. = FALSE
    )
  }
  list(
    requested = requested,
    resolved = "cpu",
    accelerator_runtime = "cpu",
    probe = gpu_probe
  )
}

.ndm_backend_device_value <- function(device, field, default = NULL) {
  tryCatch({
    value <- if (inherits(device, "python.builtin.object")) {
      reticulate::py_get_attr(device, field, silent = TRUE)
    } else if (is.environment(device)) {
      get0(field, envir = device, inherits = FALSE, ifnotfound = NULL)
    } else {
      device[[field]]
    }
    if (is.null(value)) default else value
  }, error = function(e) default)
}

.ndm_backend_device_provenance <- function(device,
                                           platform,
                                           device_count = 1L,
                                           accelerator_runtime = platform) {
  kind <- as.character(.ndm_backend_device_value(device, "device_kind", ""))
  platform_value <- tolower(as.character(
    .ndm_backend_device_value(device, "platform", platform)
  ))
  kind_lower <- tolower(kind)
  vendor <- if (identical(platform_value, "cpu")) {
    "cpu"
  } else if (identical(accelerator_runtime, "cuda")) {
    "nvidia"
  } else if (identical(accelerator_runtime, "rocm")) {
    "amd"
  } else if (grepl("nvidia|cuda", kind_lower)) {
    "nvidia"
  } else if (grepl("amd|rocm|radeon", kind_lower)) {
    "amd"
  } else {
    "unknown"
  }
  list(
    platform = platform_value,
    device_kind = kind,
    vendor = vendor,
    accelerator_runtime = accelerator_runtime,
    device_id = as.character(.ndm_backend_device_value(device, "id", "")),
    device_count = as.integer(device_count),
    is_cuda = identical(accelerator_runtime, "cuda") ||
      (identical(platform_value, "gpu") && identical(vendor, "nvidia")),
    os = as.character(.ndm_backend_sys_info()[["sysname"]] %||% ""),
    architecture = as.character(.ndm_backend_sys_info()[["machine"]] %||% "")
  )
}

.ndm_hide_tensorflow_gpus <- function(tf) {
  if (is.null(tf)) {
    return(invisible(TRUE))
  }
  tryCatch(
    tf$config$set_visible_devices(list(), "GPU"),
    error = function(e) {
      stop(
        "TensorFlow GPU isolation failed before TFRecord use: ",
        conditionMessage(e),
        call. = FALSE
      )
    }
  )
  visible <- tryCatch(
    tf$config$get_visible_devices("GPU"),
    error = function(e) {
      stop(
        "TensorFlow GPU isolation could not be verified: ",
        conditionMessage(e),
        call. = FALSE
      )
    }
  )
  if (length(visible) > 0L) {
    stop(
      "TensorFlow GPU isolation failed: TensorFlow still reports visible GPU devices.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

.ndm_backend_jax_install_plan <- function(os_name,
                                          machine,
                                          driver_major = NA_integer_,
                                          compute_capability = NA_real_) {
  if (identical(os_name, "Darwin") && identical(machine, "arm64")) {
    return(list(
      message = "Apple Silicon detected: installing CPU-only JAX. Current official JAX docs mark Apple GPU support as experimental rather than the default install path.",
      packages = "jax",
      fallback_packages = NULL
    ))
  }

  if (identical(os_name, "Linux")) {
    if (!is.na(driver_major) && driver_major >= 580 &&
        !is.na(compute_capability) && compute_capability >= 7.5) {
      return(list(
        message = sprintf(
          "Driver %s and compute capability %.1f detected: installing JAX CUDA 13 wheels.",
          driver_major,
          compute_capability
        ),
        packages = "jax[cuda13]",
        fallback_packages = "jax[cuda12]"
      ))
    }

    if (!is.na(driver_major) && driver_major >= 525) {
      return(list(
        message = sprintf(
          "Driver %s detected with compute capability %s: installing JAX CUDA 12 wheels.",
          driver_major,
          if (is.na(compute_capability)) "unknown" else format(compute_capability, nsmall = 1L)
        ),
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
  probe <- .ndm_backend_probe_modules(
    conda_env = conda_env,
    conda = conda,
    modules = modules
  )
  if (!isTRUE(probe$ok) && length(probe$missing) == 0L) {
    message(
      "Conda environment is not available. Build it with ",
      "ndm::ndm_build_backend(conda_env = '", conda_env, "', conda = '", conda,
      "'). Probe detail: ", probe$detail
    )
    return(NULL)
  }
  if (length(probe$missing) > 0L) {
    message(
      "Missing Python modules: ", paste(probe$missing, collapse = ", "),
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
    compute_capability <- NA_real_
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
      compute_cap <- try(
        suppressWarnings(
          .ndm_backend_system(
            "nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1",
            intern = TRUE
          )
        ),
        silent = TRUE
      )
      compute_capability <- suppressWarnings(as.numeric(
        sub("^([0-9]+\\.?[0-9]*).*", "\\1", compute_cap[1])
      ))
    }
    install_plan <- .ndm_backend_jax_install_plan(
      os_name = os_name,
      machine = machine,
      driver_major = drv_major,
      compute_capability = compute_capability
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
#' @param compute_backend Compute device policy. `"auto"` selects a supported
#'   JAX GPU when an isolated pre-import probe succeeds and otherwise uses CPU;
#'   `"cpu"` and `"gpu"` require that backend explicitly. macOS supports CPU
#'   execution only under this contract.
#' @param force_to_gpu Deprecated logical alias for `compute_backend`. `TRUE`
#'   maps to `"gpu"` and `FALSE` maps to `"cpu"`. Supplying conflicting old
#'   and new controls is an error.
#' @param gpu_mem_frac Optional finite GPU memory fraction in `(0, 1]`. It is
#'   rejected whenever backend selection resolves to CPU.
#'
#' @returns `ndm_initialize_backend()` returns an object of class
#'   `ndm_backend`. The returned list contains imported Python modules, the
#'   configured float type, requested and resolved compute backends,
#'   selected-device provenance, and compatibility shims used by the legacy
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
                                   import_tensorflow = TRUE,
                                   compute_backend = c("auto", "cpu", "gpu"),
                                   force_to_gpu = NULL,
                                   gpu_mem_frac = NULL) {
  compute_backend_supplied <- !missing(compute_backend)
  float_type <- match.arg(float_type)
  compute_backend <- .ndm_resolve_compute_backend(
    compute_backend = compute_backend,
    force_to_gpu = force_to_gpu,
    compute_backend_supplied = compute_backend_supplied
  )
  gpu_mem_frac <- .ndm_validate_gpu_mem_frac(gpu_mem_frac)
  python_path <- .ndm_backend_python_path(conda_env, conda = conda)

  cached_backend <- .ndm_cached_backend(
    conda_env = conda_env,
    conda = conda,
    python_path = python_path,
    float_type = float_type,
    compute_backend = compute_backend,
    import_tensorflow = import_tensorflow,
    gpu_mem_frac = gpu_mem_frac
  )
  if (!is.null(cached_backend)) {
    return(invisible(cached_backend))
  }
  backend_selection <- .ndm_select_compute_backend(
    compute_backend = compute_backend,
    conda_env = conda_env,
    conda = conda
  )
  gpu_mem_frac <- .ndm_validate_gpu_mem_frac(
    gpu_mem_frac,
    backend_selection$resolved
  )
  .ndm_validate_preloaded_jax(
    conda_env = conda_env,
    conda = conda,
    python_path = python_path,
    compute_backend = compute_backend,
    gpu_mem_frac = gpu_mem_frac
  )

  .ndm_backend_use_condaenv(
    condaenv = conda_env,
    required = conda_env_required,
    conda = conda
  )

  old_jax_platforms <- Sys.getenv("JAX_PLATFORMS", unset = NA_character_)
  old_gpu_mem_frac <- Sys.getenv(
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    unset = NA_character_
  )
  backend_initialized <- FALSE
  on.exit({
    if (!backend_initialized) {
      if (is.na(old_jax_platforms)) {
        Sys.unsetenv("JAX_PLATFORMS")
      } else {
        Sys.setenv(JAX_PLATFORMS = old_jax_platforms)
      }
      if (is.na(old_gpu_mem_frac)) {
        Sys.unsetenv("XLA_PYTHON_CLIENT_MEM_FRACTION")
      } else {
        Sys.setenv(XLA_PYTHON_CLIENT_MEM_FRACTION = old_gpu_mem_frac)
      }
    }
  }, add = TRUE)
  jax_platforms <- if (identical(backend_selection$resolved, "gpu")) {
    paste0(backend_selection$accelerator_runtime, ",cpu")
  } else {
    "cpu"
  }
  Sys.setenv(JAX_PLATFORMS = jax_platforms)
  if (!is.null(gpu_mem_frac)) {
    Sys.setenv(
      XLA_PYTHON_CLIENT_MEM_FRACTION = format(
        gpu_mem_frac,
        scientific = FALSE,
        trim = TRUE
      )
    )
  } else {
    Sys.unsetenv("XLA_PYTHON_CLIENT_MEM_FRACTION")
  }

  jax <- .ndm_backend_import("jax")
  jnp <- .ndm_backend_import("jax.numpy")
  np <- .ndm_backend_import("numpy")
  optax <- .ndm_backend_import("optax")
  eq <- .ndm_backend_import("equinox")
  diffrax <- .ndm_backend_import("diffrax")
  py_gc <- try(.ndm_backend_import("gc"), silent = TRUE)
  tf <- if (isTRUE(import_tensorflow) && .ndm_backend_py_module_available("tensorflow")) {
    tensorflow <- .ndm_backend_import("tensorflow")
    .ndm_hide_tensorflow_gpus(tensorflow)
    tensorflow
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

  jax_default_backend <- tolower(as.character(jax$default_backend()))
  default_backend <- if (jax_default_backend %in% c("gpu", "cuda", "rocm")) {
    "gpu"
  } else {
    jax_default_backend
  }
  resolved_backend <- backend_selection$resolved
  if (!identical(default_backend, resolved_backend)) {
    stop(
      "JAX initialized on '", jax_default_backend,
      "' after ndm selected `compute_backend = '", resolved_backend,
      "'`. Start a fresh R process so ndm can select the backend before JAX import.",
      call. = FALSE
    )
  }
  device_query_backend <- if (identical(resolved_backend, "gpu")) {
    backend_selection$accelerator_runtime
  } else {
    resolved_backend
  }
  selected_devices <- tryCatch(
    jax$devices(device_query_backend),
    error = function(e) list()
  )
  if (length(selected_devices) < 1L) {
    stop(
      "No JAX ", resolved_backend, " device is visible after backend initialization.",
      call. = FALSE
    )
  }
  selected_device <- selected_devices[[1L]]
  device_provenance <- .ndm_backend_device_provenance(
    device = selected_device,
    platform = resolved_backend,
    device_count = length(selected_devices),
    accelerator_runtime = backend_selection$accelerator_runtime
  )
  flash_mha <- if (isTRUE(device_provenance$is_cuda)) {
    try(.ndm_backend_import("flash_attn_jax")$flash_mha, silent = TRUE)
  } else {
    NULL
  }
  send2device <- function(x) jax$device_put(x, selected_device)
  send2cpu <- function(x) jax$device_put(x, jax$devices("cpu")[[1L]])

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
    import_tensorflow_requested = isTRUE(import_tensorflow),
    jaxFloatType = jax_float_type,
    float_type = float_type,
    conda_env = as.character(conda_env),
    conda = as.character(conda),
    python_path = python_path,
    gpu_mem_frac = gpu_mem_frac,
    default_backend = default_backend,
    jax_default_backend = jax_default_backend,
    compute_backend_requested = backend_selection$requested,
    compute_backend = resolved_backend,
    compute_backend_resolved = resolved_backend,
    accelerator_runtime = backend_selection$accelerator_runtime,
    selected_device = selected_device,
    device_count = as.integer(length(selected_devices)),
    device_platform = device_provenance$platform,
    device_kind = device_provenance$device_kind,
    device_vendor = device_provenance$vendor,
    is_cuda = device_provenance$is_cuda,
    device_provenance = device_provenance,
    backend_probe = backend_selection$probe,
    send2device = send2device,
    send2cpu = send2cpu,
    send2gpu = send2device
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
  ndm_env$jax_initialized_by_ndm <- TRUE
  ndm_env$backend_process_contract <- list(
    conda_env = as.character(conda_env),
    conda = as.character(conda),
    python_path = python_path,
    compute_backend_resolved = resolved_backend,
    accelerator_runtime = backend_selection$accelerator_runtime,
    jax_platforms = Sys.getenv("JAX_PLATFORMS", unset = ""),
    gpu_mem_frac_env = Sys.getenv(
      "XLA_PYTHON_CLIENT_MEM_FRACTION",
      unset = ""
    )
  )
  backend_initialized <- TRUE
  invisible(backend)
}

.ndm_assert_gpu_available <- function(jax, force_to_gpu = TRUE) {
  if (!isTRUE(force_to_gpu)) {
    return(invisible(TRUE))
  }
  devices <- tryCatch(jax$devices(), error = function(e) list())
  platforms <- vapply(
    devices,
    function(device) {
      value <- tryCatch(device$platform, error = function(e) NULL)
      if (is.null(value)) "" else tolower(as.character(value))
    },
    character(1)
  )
  if (!any(platforms %in% c("gpu", "cuda", "rocm"))) {
    visible <- if (length(platforms) == 0L) "none" else paste(unique(platforms), collapse = ", ")
    stop(
      "`force_to_gpu = TRUE` requires a JAX GPU device; visible platform(s): ",
      visible,
      ". Prefer `compute_backend = 'auto'` for GPU-with-CPU-fallback execution.",
      call. = FALSE
    )
  }
  invisible(TRUE)
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
