test_that("the package no longer ships the legacy runtime snapshot from inst/extdata", {
  expect_false(dir.exists(testthat::test_path("..", "..", "inst", "extdata", "analysis_runtime", "Analysis2")))

  installed_path <- system.file("extdata", "analysis_runtime", "Analysis2", package = "ndm")
  expect_false(nzchar(installed_path) && dir.exists(installed_path))
})

test_that("sysdata keeps model specs and package-owned runtime sources", {
  ns <- asNamespace("ndm")

  expect_true(exists(".ndm_embedded_model_spec_sources", envir = ns, inherits = FALSE))
  expect_true(exists(".ndm_embedded_runtime_sources", envir = ns, inherits = FALSE))
  expect_true(is.list(get(".ndm_embedded_runtime_sources", envir = ns, inherits = FALSE)))

  embedded_specs <- get(
    ".ndm_embedded_model_spec_sources",
    envir = ns,
    inherits = FALSE
  )
  for (name in names(embedded_specs)) {
    packaged_path <- system.file(
      "extdata",
      "model_specs",
      name,
      package = "ndm"
    )
    expect_true(nzchar(packaged_path), info = name)
    expect_identical(
      embedded_specs[[name]],
      paste(readLines(packaged_path, warn = FALSE), collapse = "\n"),
      info = name
    )
  }
})

test_that("runtime source root resolves pkgload and installed layouts explicitly", {
  checkout_root <- tempfile("ndm-pkgload-checkout-")
  source_root <- file.path(
    checkout_root,
    "tools",
    "runtime_source",
    "ndm_runtime"
  )
  dir.create(source_root, recursive = TRUE)
  writeLines("Package: ndm", file.path(checkout_root, "DESCRIPTION"))
  on.exit(unlink(checkout_root, recursive = TRUE), add = TRUE)
  source_root <- normalizePath(source_root, winslash = "/", mustWork = TRUE)

  local_mocked_bindings(
    .ndm_package_root = function() file.path(checkout_root, "inst"),
    .package = "ndm"
  )
  expect_identical(ndm:::.ndm_runtime_source_root(), source_root)

  installed_root <- tempfile("ndm-installed-package-")
  dir.create(installed_root)
  on.exit(unlink(installed_root, recursive = TRUE), add = TRUE)
  local_mocked_bindings(
    .ndm_package_root = function() installed_root,
    .package = "ndm"
  )
  expect_null(ndm:::.ndm_runtime_source_root())
})

test_that("materialized runtime manifests come from generated authoritative sources", {
  ns <- asNamespace("ndm")
  embedded_sources <- get(".ndm_embedded_runtime_sources", envir = ns, inherits = FALSE)
  expect_false(exists(".ndm_runtime_manifest_sources", envir = ns, inherits = FALSE))
  manifest_paths <- grep("^config/[^/]+[.]yaml$", names(embedded_sources), value = TRUE)
  expect_setequal(
    manifest_paths,
    c("config/real.yaml", "config/sim.yaml", "config/real_multidisease.yaml")
  )

  materialized_root <- tempfile("ndm-materialized-runtime-")
  on.exit(unlink(materialized_root, recursive = TRUE), add = TRUE)
  local_mocked_bindings(
    .ndm_runtime_source_root = function() NULL,
    .package = "ndm"
  )
  ndm:::.ndm_materialize_runtime_tree(materialized_root)
  expect_setequal(
    list.files(materialized_root, recursive = TRUE),
    names(embedded_sources)
  )

  for (relative_path in manifest_paths) {
    expect_identical(
      ndm:::.ndm_embedded_runtime_source(relative_path),
      embedded_sources[[relative_path]],
      info = relative_path
    )
    materialized <- paste(
      readLines(file.path(materialized_root, relative_path), warn = FALSE),
      collapse = "\n"
    )
    expect_identical(
      materialized,
      embedded_sources[[relative_path]],
      info = relative_path
    )
    expect_match(materialized, "enable_kv_cache:", fixed = TRUE)
    expect_match(materialized, "inference_mc_draws:", fixed = TRUE)
    expect_match(materialized, "observation_scale_floor:", fixed = TRUE)
    expect_match(materialized, "initial_observation_scale:", fixed = TRUE)
    expect_match(materialized, "neuralode_variational:", fixed = TRUE)
  }
})

test_that("ndm_prepare_runtime loads package-managed helpers without inst/extdata", {
  env <- ndm_new_runtime_env()

  local_mocked_bindings(
    .ndm_require_namespaces = function(...) invisible(TRUE),
    ndm_source_runtime_backend = function(...) invisible(env),
    .package = "ndm"
  )

  out <- ndm_prepare_runtime(
    config = ndm_create_config(force_to_gpu = FALSE),
    runtime_env = env
  )

  expect_identical(out, env)
  expect_true(exists("GlobalPartition", envir = env, inherits = FALSE))
  expect_true(exists("ndm_source_extracted", envir = env, inherits = FALSE))
  expect_match(env$NDM_INTERNAL_ANALYSIS_DIR, "ndm_runtime")
  expect_false(grepl("inst/extdata/analysis_runtime/Analysis2$", env$NDM_INTERNAL_ANALYSIS_DIR))
  expect_false(env$EnableKVCaching)
  expect_identical(env$InferenceMCDraws, 5L)
  expect_equal(env$ObservationScaleFloor, 1e-5)
  expect_equal(env$InitialObservationScale, 1)
  expect_true(env$neuralode_variational)
})

test_that("ndm_prepare_runtime exposes validated inference globals", {
  env <- ndm_new_runtime_env()
  config <- ndm_create_config(
    model_type = "NeuralODE",
    force_to_gpu = FALSE,
    enable_kv_cache = TRUE,
    inference_mc_draws = 7L,
    observation_scale_floor = 2e-5,
    initial_observation_scale = 0.02,
    neuralode_variational = FALSE
  )

  local_mocked_bindings(
    .ndm_require_namespaces = function(...) invisible(TRUE),
    ndm_source_runtime_backend = function(...) invisible(env),
    .package = "ndm"
  )
  ndm_prepare_runtime(config = config, runtime_env = env)

  expect_true(env$EnableKVCaching)
  expect_identical(env$InferenceMCDraws, 7L)
  expect_equal(env$ObservationScaleFloor, 2e-5)
  expect_equal(env$InitialObservationScale, 0.02)
  expect_false(env$neuralode_variational)
})

test_that("ndm_prepare_runtime rejects every public regeneration override", {
  config <- ndm_create_config()
  config$resave_tfrecords <- TRUE
  env <- ndm_new_runtime_env()

  expect_error(
    ndm_prepare_runtime(config = config, runtime_env = env),
    "ndm_bootstrap_real_tfrecords"
  )
  expect_false(exists("ndm_config", envir = env, inherits = FALSE))

  for (flag in c("resave_tfrecords", "ReSaveTfRecords")) {
    existing_env <- ndm_new_runtime_env()
    existing_env[[flag]] <- TRUE
    expect_error(
      ndm_prepare_runtime(config = ndm_create_config(), runtime_env = existing_env),
      "ndm_bootstrap_real_tfrecords",
      info = paste("existing", flag)
    )

    incoming_env <- ndm_new_runtime_env()
    incoming <- stats::setNames(list(TRUE), flag)
    expect_error(
      ndm_prepare_runtime(
        config = ndm_create_config(),
        runtime_env = incoming_env,
        runtime_globals = incoming
      ),
      "ndm_bootstrap_real_tfrecords",
      info = paste("incoming", flag)
    )
    expect_false(exists(flag, envir = incoming_env, inherits = FALSE))
  }
})

test_that("ndm_prepare_data rejects real and sim regeneration flags before sourcing", {
  guidance <- c(
    real = "ndm_bootstrap_real_tfrecords",
    sim = "ndm_bootstrap_sim_tfrecords"
  )

  for (generator in names(guidance)) {
    for (flag in c("resave_tfrecords", "ReSaveTfRecords")) {
      existing_env <- ndm_new_runtime_env()
      existing_env[[flag]] <- TRUE
      expect_error(
        ndm_prepare_data(existing_env, generator = generator),
        guidance[[generator]],
        info = paste(generator, "existing", flag)
      )
      expect_false(exists("ndm_data_generator", envir = existing_env, inherits = FALSE))

      incoming_env <- ndm_new_runtime_env()
      incoming <- stats::setNames(list(TRUE), flag)
      expect_error(
        ndm_prepare_data(
          incoming_env,
          generator = generator,
          runtime_globals = incoming
        ),
        guidance[[generator]],
        info = paste(generator, "incoming", flag)
      )
      expect_false(exists(flag, envir = incoming_env, inherits = FALSE))
    }
  }
})

test_that("public data preparation rejects inherited regeneration flags", {
  parent <- new.env(parent = emptyenv())
  parent$ReSaveTfRecords <- TRUE
  runtime_env <- ndm_new_runtime_env(parent = parent)

  expect_error(
    ndm_prepare_data(runtime_env, generator = "real"),
    "ndm_bootstrap_real_tfrecords"
  )

  runtime_env$ReSaveTfRecords <- FALSE
  expect_false(ndm:::.ndm_requests_tfrecord_regeneration(runtime_env))
})

test_that("public data preparation rejects non-logical regeneration flags", {
  for (generator in c("real", "sim")) {
    runtime_env <- ndm_new_runtime_env()
    runtime_env$ReSaveTfRecords <- 1L
    expect_error(
      ndm_prepare_data(runtime_env, generator = generator),
      "no longer supported",
      info = generator
    )
  }
})

test_that("runtime helper sourcing fails early when required runtime packages are missing", {
  local_mocked_bindings(
    .ndm_namespace_available = function(package) !identical(package, "fastmatch"),
    .package = "ndm"
  )

  expect_error(
    ndm:::ndm_source_runtime_helper_fxns(),
    "fastmatch"
  )
})
