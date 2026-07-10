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

test_that("ndm_build_backend selects the current Apple Silicon default install path", {
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
  expect_true(all(c("numpy", "jax") %in% flat_installs))
  expect_false(any(grepl("jax-metal|jaxlib==", flat_installs)))
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

  expect_invisible(
    ndm_build_backend(conda_env = "jax_cpu", include_tensorflow = FALSE)
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

test_that("force_to_gpu requires an actual CUDA JAX platform", {
  cpu_jax <- list(devices = function() list(list(platform = "cpu")))
  gpu_jax <- list(devices = function() list(list(platform = "gpu")))

  expect_error(
    ndm:::.ndm_assert_gpu_available(cpu_jax, TRUE),
    "CUDA-capable JAX GPU"
  )
  expect_invisible(ndm:::.ndm_assert_gpu_available(cpu_jax, FALSE))
  expect_invisible(ndm:::.ndm_assert_gpu_available(gpu_jax, TRUE))
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

test_that("oryx shim computes analytic Gaussian KLs including batched mixed factors", {
  conda_env <- ndm_require_backend_test_stack("Gaussian KL tests", packages = c("reticulate"))
  old_backend <- ndm:::ndm_env$backend
  on.exit(assign("backend", old_backend, envir = ndm:::ndm_env), add = TRUE)
  backend <- ndm_initialize_backend(
    conda_env = conda_env,
    float_type = "64",
    import_tensorflow = FALSE
  )
  oryx <- backend$oryx
  as_r <- function(x) reticulate::py_to_r(backend$np$asanyarray(x))

  normal_kl <- as.numeric(as_r(oryx$kl_divergence(
    oryx$Normal(1.5, 0.75),
    oryx$Normal(-0.5, 2.0)
  )))
  normal_expected <- log(2 / 0.75) + (0.75^2 + (1.5 + 0.5)^2) / (2 * 2^2) - 0.5
  expect_equal(normal_kl, normal_expected, tolerance = 1e-10)

  diag_q <- oryx$MultivariateNormalDiag(
    backend$jnp$array(c(1, -1)),
    backend$jnp$array(c(0.5, 2))
  )
  diag_p <- oryx$MultivariateNormalDiag(
    backend$jnp$array(c(0, 0.25)),
    backend$jnp$array(c(1.5, 0.75))
  )
  diag_kl <- as.numeric(as_r(oryx$kl_divergence(diag_q, diag_p)))
  diag_expected <- 0.5 * sum(
    (c(0.5, 2)^2 + (c(1, -1) - c(0, 0.25))^2) / c(1.5, 0.75)^2 -
      1 + 2 * log(c(1.5, 0.75) / c(0.5, 2))
  )
  expect_equal(diag_kl, diag_expected, tolerance = 1e-10)

  q_locations <- rbind(c(0, 0), c(1, -1))
  q_scales <- rbind(c(1, 2), c(0.5, 1.5))
  p_location <- c(0.2, -0.3)
  p_factor <- matrix(c(1.2, 0, 0.4, 0.8), nrow = 2, byrow = TRUE)
  mixed_kl <- as.numeric(as_r(oryx$kl_divergence(
    oryx$MultivariateNormalDiag(
      backend$jnp$array(q_locations),
      backend$jnp$array(q_scales)
    ),
    oryx$MultivariateNormalTriL(
      backend$jnp$array(p_location),
      backend$jnp$array(p_factor)
    )
  )))
  reference_kl <- function(mu_q, l_q, mu_p, l_p) {
    covariance_q <- l_q %*% t(l_q)
    covariance_p <- l_p %*% t(l_p)
    difference <- mu_p - mu_q
    0.5 * (
      sum(diag(solve(covariance_p, covariance_q))) +
        drop(t(difference) %*% solve(covariance_p, difference)) -
        length(mu_q) +
        as.numeric(determinant(covariance_p, logarithm = TRUE)$modulus) -
        as.numeric(determinant(covariance_q, logarithm = TRUE)$modulus)
    )
  }
  mixed_expected <- vapply(seq_len(nrow(q_locations)), function(i) {
    reference_kl(
      q_locations[i, ],
      diag(q_scales[i, ]),
      p_location,
      p_factor
    )
  }, numeric(1))
  expect_equal(mixed_kl, mixed_expected, tolerance = 1e-9)

  q_covariance <- matrix(c(1.4, 0.25, 0.25, 0.7), nrow = 2)
  full_vs_tril <- as.numeric(as_r(oryx$kl_divergence(
    oryx$MultivariateNormalFullCovariance(
      backend$jnp$array(c(-0.2, 0.6)),
      backend$jnp$array(q_covariance)
    ),
    oryx$MultivariateNormalTriL(
      backend$jnp$array(p_location),
      backend$jnp$array(p_factor)
    )
  )))
  full_expected <- reference_kl(
    c(-0.2, 0.6),
    chol(q_covariance + diag(1e-6, 2)) |> t(),
    p_location,
    p_factor
  )
  expect_equal(full_vs_tril, full_expected, tolerance = 1e-9)

  zero_scale_kl <- as.numeric(as_r(oryx$kl_divergence(
    oryx$Normal(0, 0),
    oryx$Normal(0, 0)
  )))
  expect_true(is.finite(zero_scale_kl))
  expect_equal(zero_scale_kl, 0, tolerance = 1e-10)
  expect_error(
    oryx$kl_divergence(oryx$Uniform(), oryx$Normal(0, 1)),
    "not implemented"
  )
})
