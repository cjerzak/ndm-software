ndm_test_fake_backend_modules <- function(default_backend = "cpu") {
  config_updates <- list()
  jax <- list(
    config = list(
      update = function(name, value) {
        config_updates[[name]] <<- value
        invisible(NULL)
      }
    ),
    default_backend = function() default_backend,
    devices = function(platform = NULL) {
      list(paste0(platform %||% default_backend, "-device"))
    },
    device_put = function(x, device) {
      list(value = x, device = device)
    },
    random = list(
      PRNGKey = function(int_) {
        paste("key", as.integer(int_))
      }
    ),
    nn = list(
      softplus = function(x) x + 1,
      sigmoid = function(x) x / (1 + abs(x))
    )
  )
  jnp <- list(
    float32 = "float32",
    float64 = "float64",
    array = function(x) x,
    log = function(x) log(x),
    subtract = function(x, y) x - y,
    exp = function(x) exp(x)
  )

  list(jax = jax, jnp = jnp, np = list(), config_updates = config_updates)
}

ndm_test_fake_conda_env <- function(prefix = "ndm-fake-conda-") {
  root <- tempfile(prefix)
  python <- file.path(root, "bin", "python")
  dir.create(dirname(python), recursive = TRUE, showWarnings = FALSE)
  file.create(python)

  list(root = root, python = python)
}

test_that("ndm_check_backend reports unavailable environments and modules", {
  local_mocked_bindings(
    .ndm_backend_use_condaenv = function(...) stop("missing env", call. = FALSE),
    .package = "ndm"
  )

  expect_message(
    expect_null(ndm_check_backend(conda_env = "missing")),
    "Conda environment is not available"
  )

  local_mocked_bindings(
    .ndm_backend_use_condaenv = function(...) invisible(TRUE),
    .ndm_backend_py_available = function(...) TRUE,
    .ndm_backend_py_module_available = function(module) module != "diffrax",
    .package = "ndm"
  )

  expect_message(
    expect_null(ndm_check_backend(conda_env = "jax_cpu")),
    "Missing Python modules: diffrax"
  )
})

test_that("ndm_build_backend selects Apple Silicon Metal wheels", {
  installs <- list()
  created <- list()

  local_mocked_bindings(
    .ndm_backend_conda_create = function(...) {
      created <<- list(...)
      invisible(NULL)
    },
    .ndm_backend_py_install = function(packages, ...) {
      installs <<- c(installs, as.list(packages))
      invisible(TRUE)
    },
    .ndm_backend_sys_info = function() c(sysname = "Darwin", machine = "arm64"),
    .package = "ndm"
  )

  expect_message(
    ndm_build_backend(conda_env = "jax_cpu", include_tensorflow = FALSE),
    "Apple Silicon detected"
  )

  flat_installs <- unlist(installs, use.names = FALSE)
  expect_equal(created$envname, "jax_cpu")
  expect_true(all(c("numpy", "jax==0.5.0", "jaxlib==0.5.0", "jax-metal==0.1.1") %in% flat_installs))
  expect_false("tensorflow" %in% flat_installs)
})

test_that("ndm_build_backend falls back from CUDA 13 to CUDA 12", {
  installs <- list()
  fake_env <- ndm_test_fake_conda_env()
  on.exit(unlink(fake_env$root, recursive = TRUE, force = TRUE), add = TRUE)

  local_mocked_bindings(
    .ndm_backend_conda_create = function(...) invisible(NULL),
    .ndm_backend_py_install = function(packages, ...) {
      installs <<- c(installs, as.list(packages))
      if (identical(packages, "jax[cuda13]")) {
        stop("wheel failure", call. = FALSE)
      }
      invisible(TRUE)
    },
    .ndm_backend_sys_info = function() c(sysname = "Linux", machine = "x86_64"),
    .ndm_backend_system = function(...) "590.12",
    .ndm_backend_conda_list = function(...) data.frame(name = "jax_cpu", python = fake_env$python),
    .package = "ndm"
  )

  expect_message(
    ndm_build_backend(conda_env = "jax_cpu", include_tensorflow = FALSE),
    "falling back to CUDA 12"
  )

  flat_installs <- unlist(installs, use.names = FALSE)
  expect_true("jax[cuda13]" %in% flat_installs)
  expect_true("jax[cuda12]" %in% flat_installs)
})

test_that("ndm_build_backend uses CPU-only JAX when no supported GPU driver is detected", {
  installs <- list()
  fake_env <- ndm_test_fake_conda_env()
  on.exit(unlink(fake_env$root, recursive = TRUE, force = TRUE), add = TRUE)

  local_mocked_bindings(
    .ndm_backend_conda_create = function(...) invisible(NULL),
    .ndm_backend_py_install = function(packages, ...) {
      installs <<- c(installs, as.list(packages))
      invisible(TRUE)
    },
    .ndm_backend_sys_info = function() c(sysname = "Linux", machine = "x86_64"),
    .ndm_backend_system = function(...) character(),
    .ndm_backend_conda_list = function(...) data.frame(name = "jax_cpu", python = fake_env$python),
    .package = "ndm"
  )

  expect_message(
    ndm_build_backend(conda_env = "jax_cpu", include_tensorflow = FALSE),
    "Installing CPU-only JAX"
  )

  flat_installs <- unlist(installs, use.names = FALSE)
  expect_true("jax" %in% flat_installs)
  expect_false(any(grepl("cuda", flat_installs, fixed = TRUE)))
})

test_that("ndm_initialize_backend configures float types and caches the backend", {
  fake_modules <- ndm_test_fake_backend_modules(default_backend = "cpu")
  old_backend <- ndm:::ndm_env$backend
  on.exit(assign("backend", old_backend, envir = ndm:::ndm_env), add = TRUE)
  old_jax_platforms <- Sys.getenv("JAX_PLATFORMS", unset = NA_character_)
  on.exit(
    if (is.na(old_jax_platforms)) {
      Sys.unsetenv("JAX_PLATFORMS")
    } else {
      Sys.setenv(JAX_PLATFORMS = old_jax_platforms)
    },
    add = TRUE
  )
  Sys.unsetenv("JAX_PLATFORMS")

  local_mocked_bindings(
    .ndm_backend_use_condaenv = function(...) invisible(TRUE),
    .ndm_backend_import = function(module) {
      switch(
        module,
        "jax" = fake_modules$jax,
        "jax.numpy" = fake_modules$jnp,
        "numpy" = list(),
        "optax" = list(name = "optax"),
        "equinox" = list(name = "equinox"),
        "diffrax" = list(name = "diffrax"),
        "gc" = list(name = "gc"),
        "tensorflow" = list(name = "tensorflow"),
        stop("Unexpected module import: ", module, call. = FALSE)
      )
    },
    .ndm_backend_py_module_available = function(module) identical(module, "tensorflow"),
    ndm_make_oryx_shim = function(...) "oryx-shim",
    .package = "ndm"
  )

  backend <- ndm_initialize_backend(conda_env = "jax_cpu", float_type = "64", import_tensorflow = TRUE)

  expect_s3_class(backend, "ndm_backend")
  expect_equal(backend$float_type, "64")
  expect_equal(backend$jaxFloatType, "float64")
  expect_equal(backend$tf$name, "tensorflow")
  expect_equal(backend$oryx, "oryx-shim")
  expect_true(is.function(backend$JaxKey))
  expect_identical(ndm_backend_modules(), backend)
  expect_identical(Sys.getenv("JAX_PLATFORMS", unset = ""), "cpu")
})

test_that("ndm_backend_modules errors before initialization", {
  old_backend <- ndm:::ndm_env$backend
  on.exit(assign("backend", old_backend, envir = ndm:::ndm_env), add = TRUE)
  assign("backend", NULL, envir = ndm:::ndm_env)

  expect_error(
    ndm_backend_modules(),
    "Backend not initialized"
  )
})

test_that("oryx shim supports sampling, triangular fill, and KL divergence in jax_cpu", {
  conda_env <- ndm_require_backend_test_stack("oryx shim tests", packages = c("reticulate"))
  old_backend <- ndm:::ndm_env$backend
  on.exit(assign("backend", old_backend, envir = ndm:::ndm_env), add = TRUE)

  backend <- ndm_initialize_backend(
    conda_env = conda_env,
    float_type = "32",
    import_tensorflow = FALSE
  )
  oryx <- backend$oryx

  normal <- oryx$Normal(
    loc = backend$jnp$array(c(0, 1)),
    scale = backend$jnp$array(c(1, 2))
  )
  sample_from_int <- reticulate::py_to_r(backend$np$asanyarray(normal$sample(seed = 1L)))
  sample_from_key <- reticulate::py_to_r(
    backend$np$asanyarray(normal$sample(seed = backend$jax$random$PRNGKey(1L)))
  )

  expect_equal(sample_from_int, sample_from_key, tolerance = 1e-6)

  tri <- reticulate::py_to_r(
    backend$np$asanyarray(
      oryx$math$fill_triangular(backend$jnp$array(c(1, 2, 3)))
    )
  )
  expect_equal(tri, matrix(c(1, 0, 2, 3), nrow = 2, byrow = TRUE))

  kl <- reticulate::py_to_r(
    backend$np$asanyarray(
      oryx$kl_divergence(
        oryx$Normal(backend$jnp$array(0), backend$jnp$array(1)),
        oryx$Normal(backend$jnp$array(0), backend$jnp$array(2))
      )
    )
  )
  expect_true(is.finite(as.numeric(kl)))
  expect_gte(as.numeric(kl), 0)
})
