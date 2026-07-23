ndm_backend_test_is_strict <- function() {
  strict_flags <- c(Sys.getenv("CI", unset = ""), Sys.getenv("NDM_STRICT_BACKEND_TESTS", unset = ""))
  any(tolower(strict_flags) %in% c("1", "true", "yes"))
}

ndm_backend_test_conda_env <- function() {
  Sys.getenv(
    "NDM_SOFTWARE_CONDA_ENV",
    unset = Sys.getenv("NDM_CONDA_ENV", unset = "jax_cpu")
  )
}

ndm_require_backend_test_stack <- function(context,
                                           modules = c("jax", "numpy", "optax", "equinox", "diffrax", "tensorflow"),
                                           packages = c("reticulate", "fastmatch", "rrapply", "yaml", "zip", "zoo")) {
  strict <- ndm_backend_test_is_strict()
  if (!nzchar(Sys.getenv("JAX_PLATFORMS", unset = ""))) {
    Sys.setenv(JAX_PLATFORMS = "cpu")
  }
  missing_packages <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_packages) > 0L) {
    message <- paste0(
      "Missing R packages for ",
      context,
      ": ",
      paste(missing_packages, collapse = ", "),
      "."
    )
    if (strict) {
      stop(message, call. = FALSE)
    }
    skip(message)
  }

  conda_env <- ndm_backend_test_conda_env()
  backend_ready <- ndm_check_backend(conda_env = conda_env, modules = modules)
  if (is.null(backend_ready)) {
    message <- paste0(
      context,
      " requires backend environment '",
      conda_env,
      "' with modules: ",
      paste(modules, collapse = ", "),
      "."
    )
    if (strict) {
      stop(message, call. = FALSE)
    }
    skip(message)
  }

  invisible(conda_env)
}
