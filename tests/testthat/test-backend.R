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
      selected_platform <- platform %||% default_backend
      list(list(
        platform = selected_platform,
        device_kind = if (identical(selected_platform, "gpu")) "NVIDIA test GPU" else "test CPU",
        id = 0L
      ))
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
    .ndm_backend_probe_modules = function(...) {
      list(ok = FALSE, missing = character(), detail = "missing env")
    },
    .package = "ndm"
  )

  expect_message(
    expect_null(ndm_check_backend(conda_env = "missing")),
    "Conda environment is not available"
  )

  local_mocked_bindings(
    .ndm_backend_probe_modules = function(...) {
      list(ok = FALSE, missing = "diffrax", detail = "missing diffrax")
    },
    .package = "ndm"
  )

  expect_message(
    expect_null(ndm_check_backend(conda_env = "jax_cpu")),
    "Missing Python modules: diffrax"
  )
})

test_that("ndm_check_backend probes in a subprocess without importing Python", {
  local_mocked_bindings(
    .ndm_backend_conda_python = function(...) "/fake/python",
    .ndm_backend_system2 = function(command, args, ...) {
      expect_identical(command, "/fake/python")
      expect_true("-c" %in% args)
      "NDM_MODULES_OK"
    },
    .ndm_backend_py_available = function(...) {
      stop("main-process Python availability must not be queried", call. = FALSE)
    },
    .ndm_backend_py_module_available = function(...) {
      stop("main-process module import must not be queried", call. = FALSE)
    },
    .package = "ndm"
  )
  expect_true(ndm_check_backend(conda_env = "isolated"))
  expect_error(
    ndm_check_backend(modules = "bad-module-name!"),
    "valid non-empty Python module names"
  )
})

test_that("isolated module probe rejects import-time failures", {
  python <- Sys.which("python3")
  skip_if(!nzchar(python), "python3 is unavailable")
  module_root <- tempfile("ndm-broken-module-")
  dir.create(module_root, recursive = TRUE)
  on.exit(unlink(module_root, recursive = TRUE, force = TRUE), add = TRUE)
  writeLines(
    "raise RuntimeError('intentional import failure')",
    file.path(module_root, "ndm_broken_probe_module.py")
  )
  old_pythonpath <- Sys.getenv("PYTHONPATH", unset = NA_character_)
  on.exit(
    if (is.na(old_pythonpath)) Sys.unsetenv("PYTHONPATH") else
      Sys.setenv(PYTHONPATH = old_pythonpath),
    add = TRUE
  )
  Sys.setenv(PYTHONPATH = module_root)
  local_mocked_bindings(
    .ndm_backend_conda_python = function(...) python,
    .package = "ndm"
  )

  probe <- ndm:::.ndm_backend_probe_modules(
    conda_env = "unused",
    modules = "ndm_broken_probe_module"
  )
  expect_false(probe$ok)
  expect_identical(probe$missing, "ndm_broken_probe_module")
  expect_match(probe$detail, "intentional import failure")
})

test_that("backend check followed by CPU initialization works in a fresh process", {
  conda_env <- ndm_backend_test_conda_env()
  ready <- ndm_check_backend(conda_env = conda_env)
  skip_if(is.null(ready), "configured backend environment is unavailable")

  package_path <- normalizePath(find.package("ndm"), winslash = "/")
  installed_package <- file.exists(file.path(package_path, "Meta", "package.rds"))
  if (!installed_package) {
    skip_if_not_installed("pkgload")
  }
  r_quote <- function(x) encodeString(x, quote = "\"")
  load_package <- if (installed_package) {
    sprintf(
      ".libPaths(c(%s, .libPaths())); library(ndm)",
      r_quote(dirname(package_path))
    )
  } else {
    sprintf("pkgload::load_all(%s, quiet = TRUE)", r_quote(package_path))
  }
  script <- paste(
    load_package,
    "stopifnot(!reticulate::py_available(initialize = FALSE))",
    sprintf(
      "stopifnot(isTRUE(ndm_check_backend(conda_env = %s)))",
      r_quote(conda_env)
    ),
    "stopifnot(!reticulate::py_available(initialize = FALSE))",
    sprintf(
      paste0(
        "backend <- ndm_initialize_backend(conda_env = %s, ",
        "compute_backend = 'cpu', import_tensorflow = FALSE)"
      ),
      r_quote(conda_env)
    ),
    "stopifnot(identical(backend$compute_backend_resolved, 'cpu'))",
    "cat('NDM_CHECK_THEN_INIT_OK\\n')",
    sep = "; "
  )
  output <- suppressWarnings(system2(
    file.path(R.home("bin"), "Rscript"),
    args = c("-e", shQuote(script)),
    stdout = TRUE,
    stderr = TRUE
  ))
  expect_null(attr(output, "status", exact = TRUE), info = paste(output, collapse = "\n"))
  expect_true(any(output == "NDM_CHECK_THEN_INIT_OK"))
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

test_that("CUDA provisioning accounts for GPU compute capability", {
  modern <- ndm:::.ndm_backend_jax_install_plan(
    os_name = "Linux",
    machine = "x86_64",
    driver_major = 590L,
    compute_capability = 8.0
  )
  older_gpu <- ndm:::.ndm_backend_jax_install_plan(
    os_name = "Linux",
    machine = "x86_64",
    driver_major = 590L,
    compute_capability = 7.0
  )
  unknown_gpu <- ndm:::.ndm_backend_jax_install_plan(
    os_name = "Linux",
    machine = "x86_64",
    driver_major = 590L,
    compute_capability = NA_real_
  )
  expect_identical(modern$packages, "jax[cuda13]")
  expect_identical(older_gpu$packages, "jax[cuda12]")
  expect_identical(unknown_gpu$packages, "jax[cuda12]")
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
  tf_visibility_updates <- list()
  fake_tf <- list(
    name = "tensorflow",
    config = list(
      set_visible_devices = function(devices, device_type) {
        tf_visibility_updates[[length(tf_visibility_updates) + 1L]] <<-
          list(devices = devices, device_type = device_type)
        invisible(NULL)
      },
      get_visible_devices = function(device_type) list()
    )
  )
  old_backend <- ndm:::ndm_env$backend
  old_initialized <- ndm:::ndm_env$jax_initialized_by_ndm
  old_contract <- ndm:::ndm_env$backend_process_contract
  on.exit(assign("backend", old_backend, envir = ndm:::ndm_env), add = TRUE)
  on.exit(
    assign("jax_initialized_by_ndm", old_initialized, envir = ndm:::ndm_env),
    add = TRUE
  )
  on.exit(
    assign("backend_process_contract", old_contract, envir = ndm:::ndm_env),
    add = TRUE
  )
  assign("backend", NULL, envir = ndm:::ndm_env)
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
    .ndm_validate_preloaded_jax = function(...) invisible(FALSE),
    .ndm_backend_import = function(module, ...) {
      switch(
        module,
        "jax" = fake_modules$jax,
        "jax.numpy" = fake_modules$jnp,
        "numpy" = list(),
        "optax" = list(name = "optax"),
        "equinox" = list(name = "equinox"),
        "diffrax" = list(name = "diffrax"),
        "gc" = list(name = "gc"),
        "tensorflow" = fake_tf,
        stop("Unexpected module import: ", module, call. = FALSE)
      )
    },
    .ndm_backend_py_module_available = function(module) identical(module, "tensorflow"),
    ndm_make_oryx_shim = function(...) "oryx-shim",
    .package = "ndm"
  )

  backend <- ndm_initialize_backend(
    conda_env = "jax_cpu",
    float_type = "64",
    import_tensorflow = TRUE,
    compute_backend = "cpu"
  )

  expect_s3_class(backend, "ndm_backend")
  expect_equal(backend$float_type, "64")
  expect_equal(backend$jaxFloatType, "float64")
  expect_equal(backend$tf$name, "tensorflow")
  expect_length(tf_visibility_updates, 1L)
  expect_identical(tf_visibility_updates[[1L]]$device_type, "GPU")
  expect_length(tf_visibility_updates[[1L]]$devices, 0L)
  expect_equal(backend$oryx, "oryx-shim")
  expect_identical(backend$compute_backend_requested, "cpu")
  expect_identical(backend$compute_backend, "cpu")
  expect_identical(backend$compute_backend_resolved, "cpu")
  expect_identical(backend$accelerator_runtime, "cpu")
  expect_identical(backend$device_count, 1L)
  expect_identical(backend$device_platform, "cpu")
  expect_identical(backend$device_vendor, "cpu")
  expect_false(backend$is_cuda)
  expect_true(is.function(backend$send2device))
  expect_identical(backend$send2cpu(1)$device$platform, "cpu")
  expect_identical(backend$send2gpu(1)$device$platform, "cpu")
  expect_identical(backend$send2gpu, backend$send2device)
  expect_true(is.function(backend$JaxKey))
  expect_identical(ndm_backend_modules(), backend)
  expect_identical(Sys.getenv("JAX_PLATFORMS", unset = ""), "cpu")
})

test_that("TensorFlow GPU isolation fails closed", {
  setter_failure <- list(
    config = list(
      set_visible_devices = function(...) stop("already initialized", call. = FALSE),
      get_visible_devices = function(...) list()
    )
  )
  expect_error(
    ndm:::.ndm_hide_tensorflow_gpus(setter_failure),
    "TensorFlow GPU isolation failed.*already initialized"
  )

  visible_gpu <- list(
    config = list(
      set_visible_devices = function(...) invisible(NULL),
      get_visible_devices = function(...) list("gpu:0")
    )
  )
  expect_error(
    ndm:::.ndm_hide_tensorflow_gpus(visible_gpu),
    "still reports visible GPU"
  )
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

test_that("force_to_gpu requires an actual JAX GPU platform", {
  cpu_jax <- list(devices = function() list(list(platform = "cpu")))
  gpu_jax <- list(devices = function() list(list(platform = "gpu")))

  expect_error(
    ndm:::.ndm_assert_gpu_available(cpu_jax, TRUE),
    "requires a JAX GPU"
  )
  expect_invisible(ndm:::.ndm_assert_gpu_available(cpu_jax, FALSE))
  expect_invisible(ndm:::.ndm_assert_gpu_available(gpu_jax, TRUE))
})

test_that("compute backend controls validate aliases and conflicts", {
  expect_identical(
    ndm:::.ndm_resolve_compute_backend("auto", NULL, FALSE, FALSE),
    "auto"
  )
  expect_identical(
    ndm:::.ndm_resolve_compute_backend("auto", TRUE, FALSE, FALSE),
    "gpu"
  )
  expect_identical(
    ndm:::.ndm_resolve_compute_backend("auto", FALSE, FALSE, FALSE),
    "cpu"
  )
  expect_identical(
    ndm:::.ndm_resolve_compute_backend("gpu", TRUE, TRUE, FALSE),
    "gpu"
  )
  expect_error(
    ndm:::.ndm_resolve_compute_backend("cpu", TRUE, TRUE, FALSE),
    "Conflicting backend controls"
  )
  expect_error(
    ndm:::.ndm_resolve_compute_backend("metal", NULL, TRUE, FALSE),
    "auto.*cpu.*gpu"
  )
  expect_error(
    ndm:::.ndm_resolve_compute_backend(c("cpu", "gpu"), NULL, TRUE, FALSE),
    "auto.*cpu.*gpu"
  )
  expect_error(
    ndm:::.ndm_resolve_compute_backend("", NULL, TRUE, FALSE),
    "auto.*cpu.*gpu"
  )
  expect_error(
    ndm:::.ndm_resolve_compute_backend("g", NULL, TRUE, FALSE),
    "auto.*cpu.*gpu"
  )
  expect_error(
    ndm:::.ndm_resolve_compute_backend("auto", NA, FALSE, FALSE),
    "one non-missing logical value or NULL"
  )
})

test_that("backend selection is portable and auto falls back to CPU", {
  mac_auto <- ndm:::.ndm_select_compute_backend(
    "auto",
    conda_env = "unused",
    sys_info = c(sysname = "Darwin", machine = "arm64")
  )
  expect_identical(mac_auto$resolved, "cpu")
  expect_error(
    ndm:::.ndm_select_compute_backend(
      "gpu",
      conda_env = "unused",
      sys_info = c(sysname = "Darwin", machine = "arm64")
    ),
    "not supported on macOS"
  )

  local_mocked_bindings(
    .ndm_backend_probe_platform = function(...) {
      list(ok = FALSE, platform = "gpu", detail = "no accelerator")
    },
    .package = "ndm"
  )
  linux_auto <- ndm:::.ndm_select_compute_backend(
    "auto",
    conda_env = "test",
    sys_info = c(sysname = "Linux", machine = "x86_64")
  )
  expect_identical(linux_auto$requested, "auto")
  expect_identical(linux_auto$resolved, "cpu")
  expect_match(linux_auto$probe$detail, "no accelerator")
  expect_error(
    ndm:::.ndm_select_compute_backend(
      "gpu",
      conda_env = "test",
      sys_info = c(sysname = "Linux", machine = "x86_64")
    ),
    "Isolated probe failed"
  )
})

test_that("isolated accelerator probes use one concrete runtime plus CPU", {
  observed_env <- character()
  local_mocked_bindings(
    .ndm_backend_conda_python = function(...) "/fake/python",
    .ndm_backend_system2 = function(command, args, env, ...) {
      observed_env <<- c(observed_env, env)
      runtime <- sub("^JAX_PLATFORMS=([^,]+),cpu$", "\\1", env)
      paste0("NDM_BACKEND_PROBE_OK:", runtime)
    },
    .package = "ndm"
  )
  expect_true(ndm:::.ndm_backend_probe_platform("test", platform = "cuda")$ok)
  expect_true(ndm:::.ndm_backend_probe_platform("test", platform = "rocm")$ok)
  expect_identical(
    observed_env,
    c("JAX_PLATFORMS=cuda,cpu", "JAX_PLATFORMS=rocm,cpu")
  )
})

test_that("generic JAX GPUs resolve without requiring NVIDIA", {
  local_mocked_bindings(
    .ndm_backend_probe_platform = function(platform, ...) {
      list(
        ok = identical(platform, "rocm"),
        platform = platform,
        detail = if (identical(platform, "rocm")) "ok" else "not installed"
      )
    },
    .package = "ndm"
  )
  selected <- ndm:::.ndm_select_compute_backend(
    "gpu",
    conda_env = "rocm",
    sys_info = c(sysname = "Linux", machine = "x86_64")
  )
  expect_identical(selected$resolved, "gpu")
  expect_identical(selected$accelerator_runtime, "rocm")

  provenance <- ndm:::.ndm_backend_device_provenance(
    list(platform = "gpu", device_kind = "AMD Radeon test", id = 2L),
    platform = "gpu",
    device_count = 2L,
    accelerator_runtime = "rocm"
  )
  expect_identical(provenance$vendor, "amd")
  expect_false(provenance$is_cuda)
  expect_identical(provenance$device_count, 2L)
})

test_that("concrete CUDA and ROCm selection keeps the CPU backend available", {
  fake_modules <- ndm_test_fake_backend_modules(default_backend = "gpu")
  accelerator_runtime <- "cuda"
  old_backend <- ndm:::ndm_env$backend
  old_initialized <- ndm:::ndm_env$jax_initialized_by_ndm
  old_contract <- ndm:::ndm_env$backend_process_contract
  old_jax_platforms <- Sys.getenv("JAX_PLATFORMS", unset = NA_character_)
  old_mem_frac <- Sys.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION", unset = NA_character_)
  on.exit(assign("backend", old_backend, envir = ndm:::ndm_env), add = TRUE)
  on.exit(
    assign("jax_initialized_by_ndm", old_initialized, envir = ndm:::ndm_env),
    add = TRUE
  )
  on.exit(
    assign("backend_process_contract", old_contract, envir = ndm:::ndm_env),
    add = TRUE
  )
  on.exit(
    if (is.na(old_jax_platforms)) Sys.unsetenv("JAX_PLATFORMS") else
      Sys.setenv(JAX_PLATFORMS = old_jax_platforms),
    add = TRUE
  )
  on.exit(
    if (is.na(old_mem_frac)) Sys.unsetenv("XLA_PYTHON_CLIENT_MEM_FRACTION") else
      Sys.setenv(XLA_PYTHON_CLIENT_MEM_FRACTION = old_mem_frac),
    add = TRUE
  )
  assign("backend", NULL, envir = ndm:::ndm_env)

  local_mocked_bindings(
    .ndm_backend_use_condaenv = function(...) invisible(TRUE),
    .ndm_validate_preloaded_jax = function(...) invisible(FALSE),
    .ndm_select_compute_backend = function(...) {
      list(
        requested = "gpu",
        resolved = "gpu",
        accelerator_runtime = accelerator_runtime,
        probe = list(ok = TRUE)
      )
    },
    .ndm_backend_import = function(module, ...) {
      switch(
        module,
        "jax" = fake_modules$jax,
        "jax.numpy" = fake_modules$jnp,
        "numpy" = list(),
        "optax" = list(),
        "equinox" = list(),
        "diffrax" = list(),
        "gc" = list(),
        stop("optional module unavailable", call. = FALSE)
      )
    },
    .ndm_backend_py_module_available = function(...) FALSE,
    ndm_make_oryx_shim = function(...) "oryx-shim",
    .package = "ndm"
  )
  backend <- ndm_initialize_backend(
    conda_env = "cuda-env",
    compute_backend = "gpu",
    import_tensorflow = FALSE,
    gpu_mem_frac = 0.4
  )
  expect_identical(Sys.getenv("JAX_PLATFORMS"), "cuda,cpu")
  expect_identical(Sys.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION"), "0.4")
  expect_identical(backend$default_backend, "gpu")
  expect_identical(backend$compute_backend_resolved, "gpu")
  expect_identical(backend$accelerator_runtime, "cuda")
  expect_true(backend$is_cuda)
  expect_identical(backend$send2cpu(1)$device$platform, "cpu")

  assign("backend", NULL, envir = ndm:::ndm_env)
  Sys.setenv(XLA_PYTHON_CLIENT_MEM_FRACTION = "0.25")
  accelerator_runtime <- "rocm"
  rocm_backend <- ndm_initialize_backend(
    conda_env = "rocm-env",
    compute_backend = "gpu",
    import_tensorflow = FALSE
  )
  expect_identical(Sys.getenv("JAX_PLATFORMS"), "rocm,cpu")
  expect_identical(Sys.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION", unset = ""), "")
  expect_identical(rocm_backend$compute_backend_resolved, "gpu")
  expect_identical(rocm_backend$accelerator_runtime, "rocm")
  expect_identical(rocm_backend$device_vendor, "amd")
  expect_false(rocm_backend$is_cuda)
  expect_identical(rocm_backend$send2cpu(1)$device$platform, "cpu")
})

test_that("cached backend prevents incompatible process-global reinitialization", {
  old_backend <- ndm:::ndm_env$backend
  old_jax_platforms <- Sys.getenv("JAX_PLATFORMS", unset = NA_character_)
  on.exit(assign("backend", old_backend, envir = ndm:::ndm_env), add = TRUE)
  on.exit(
    if (is.na(old_jax_platforms)) Sys.unsetenv("JAX_PLATFORMS") else
      Sys.setenv(JAX_PLATFORMS = old_jax_platforms),
    add = TRUE
  )
  cached <- structure(
    list(
      compute_backend = "cpu",
      compute_backend_resolved = "cpu",
      default_backend = "cpu",
      float_type = "32",
      conda_env = "cached-env",
      tf = list(name = "tensorflow"),
      gpu_mem_frac = NULL
    ),
    class = c("ndm_backend", "list")
  )
  assign("backend", cached, envir = ndm:::ndm_env)
  Sys.setenv(JAX_PLATFORMS = "sentinel")

  expect_identical(
    ndm_initialize_backend(
      conda_env = "cached-env",
      float_type = "32",
      compute_backend = "auto"
    ),
    cached
  )
  expect_identical(Sys.getenv("JAX_PLATFORMS"), "sentinel")
  expect_error(
    ndm_initialize_backend(
      conda_env = "cached-env",
      float_type = "32",
      compute_backend = "gpu"
    ),
    "already initialized JAX on 'cpu'"
  )
  expect_error(
    ndm_initialize_backend(
      conda_env = "other-env",
      float_type = "32",
      compute_backend = "cpu"
    ),
    "already initialized JAX from conda environment"
  )
  cached$conda <- "/conda/A"
  cached$python_path <- ""
  assign("backend", cached, envir = ndm:::ndm_env)
  expect_error(
    ndm_initialize_backend(
      conda_env = "cached-env",
      conda = "/conda/B",
      float_type = "32",
      compute_backend = "cpu"
    ),
    "different Conda installation or Python executable"
  )
})

test_that("failed backend initialization restores process environment", {
  old_backend <- ndm:::ndm_env$backend
  old_jax_platforms <- Sys.getenv("JAX_PLATFORMS", unset = NA_character_)
  old_mem_frac <- Sys.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION", unset = NA_character_)
  on.exit(assign("backend", old_backend, envir = ndm:::ndm_env), add = TRUE)
  on.exit(
    if (is.na(old_jax_platforms)) Sys.unsetenv("JAX_PLATFORMS") else
      Sys.setenv(JAX_PLATFORMS = old_jax_platforms),
    add = TRUE
  )
  on.exit(
    if (is.na(old_mem_frac)) Sys.unsetenv("XLA_PYTHON_CLIENT_MEM_FRACTION") else
      Sys.setenv(XLA_PYTHON_CLIENT_MEM_FRACTION = old_mem_frac),
    add = TRUE
  )
  assign("backend", NULL, envir = ndm:::ndm_env)
  Sys.setenv(
    JAX_PLATFORMS = "original-platform",
    XLA_PYTHON_CLIENT_MEM_FRACTION = "0.25"
  )

  local_mocked_bindings(
    .ndm_backend_use_condaenv = function(...) invisible(TRUE),
    .ndm_select_compute_backend = function(...) {
      list(
        requested = "gpu",
        resolved = "gpu",
        accelerator_runtime = "cuda",
        probe = list(ok = TRUE)
      )
    },
    .ndm_backend_import = function(...) stop("import failed", call. = FALSE),
    .package = "ndm"
  )
  expect_error(
    ndm_initialize_backend(
      conda_env = "test",
      compute_backend = "gpu",
      gpu_mem_frac = 0.5
    ),
    "import failed"
  )
  expect_identical(Sys.getenv("JAX_PLATFORMS"), "original-platform")
  expect_identical(Sys.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION"), "0.25")
})

test_that("externally preloaded JAX is rejected even on the requested platform", {
  old_backend <- ndm:::ndm_env$backend
  old_initialized <- ndm:::ndm_env$jax_initialized_by_ndm
  old_contract <- ndm:::ndm_env$backend_process_contract
  old_jax_platforms <- Sys.getenv("JAX_PLATFORMS", unset = NA_character_)
  on.exit(assign("backend", old_backend, envir = ndm:::ndm_env), add = TRUE)
  on.exit(
    assign("jax_initialized_by_ndm", old_initialized, envir = ndm:::ndm_env),
    add = TRUE
  )
  on.exit(
    assign("backend_process_contract", old_contract, envir = ndm:::ndm_env),
    add = TRUE
  )
  on.exit(
    if (is.na(old_jax_platforms)) Sys.unsetenv("JAX_PLATFORMS") else
      Sys.setenv(JAX_PLATFORMS = old_jax_platforms),
    add = TRUE
  )
  assign("backend", NULL, envir = ndm:::ndm_env)
  assign("jax_initialized_by_ndm", FALSE, envir = ndm:::ndm_env)
  assign("backend_process_contract", NULL, envir = ndm:::ndm_env)
  Sys.setenv(JAX_PLATFORMS = "cpu")
  local_mocked_bindings(
    .ndm_backend_jax_preloaded = function() TRUE,
    .ndm_backend_use_condaenv = function(...) {
      stop("backend selection must not continue", call. = FALSE)
    },
    .package = "ndm"
  )
  expect_error(
    ndm_initialize_backend(
      conda_env = "test",
      compute_backend = "cpu",
      import_tensorflow = FALSE
    ),
    "JAX was imported before ndm"
  )
  expect_identical(Sys.getenv("JAX_PLATFORMS"), "cpu")
})

test_that("GPU memory fractions are rejected whenever selection resolves CPU", {
  expect_error(
    ndm:::.ndm_validate_gpu_mem_frac(0.5, "cpu"),
    "resolved compute backend is CPU"
  )
  local_mocked_bindings(
    .ndm_backend_use_condaenv = function(...) invisible(TRUE),
    .ndm_select_compute_backend = function(...) {
      list(
        requested = "auto",
        resolved = "cpu",
        accelerator_runtime = "cpu",
        probe = list(ok = FALSE)
      )
    },
    .package = "ndm"
  )
  old_backend <- ndm:::ndm_env$backend
  on.exit(assign("backend", old_backend, envir = ndm:::ndm_env), add = TRUE)
  assign("backend", NULL, envir = ndm:::ndm_env)
  expect_error(
    ndm_initialize_backend(
      conda_env = "test",
      compute_backend = "auto",
      gpu_mem_frac = 0.5
    ),
    "resolved compute backend is CPU"
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

test_that("oryx shim computes analytic Gaussian KLs including batched mixed factors", {
  conda_env <- ndm_require_backend_test_stack("Gaussian KL tests", packages = c("reticulate"))
  old_backend <- ndm:::ndm_env$backend
  on.exit(assign("backend", old_backend, envir = ndm:::ndm_env), add = TRUE)
  backend <- ndm_initialize_backend(
    conda_env = conda_env,
    float_type = "32",
    import_tensorflow = FALSE
  )
  oryx <- backend$oryx
  as_r <- function(x) reticulate::py_to_r(backend$np$asanyarray(x))

  normal_kl <- as.numeric(as_r(oryx$kl_divergence(
    oryx$Normal(1.5, 0.75),
    oryx$Normal(-0.5, 2.0)
  )))
  normal_expected <- log(2 / 0.75) + (0.75^2 + (1.5 + 0.5)^2) / (2 * 2^2) - 0.5
  expect_equal(normal_kl, normal_expected, tolerance = 1e-5)

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
  expect_equal(diag_kl, diag_expected, tolerance = 1e-5)

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
  expect_equal(mixed_kl, mixed_expected, tolerance = 1e-5)

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
  expect_equal(full_vs_tril, full_expected, tolerance = 1e-5)

  zero_scale_kl <- as.numeric(as_r(oryx$kl_divergence(
    oryx$Normal(0, 0),
    oryx$Normal(0, 0)
  )))
  expect_true(is.finite(zero_scale_kl))
  expect_equal(zero_scale_kl, 0, tolerance = 1e-6)
  expect_error(
    oryx$kl_divergence(oryx$Uniform(), oryx$Normal(0, 1)),
    "not implemented"
  )
})
