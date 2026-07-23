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
