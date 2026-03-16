test_that("the package no longer ships the legacy runtime snapshot from inst/extdata", {
  expect_false(dir.exists(testthat::test_path("..", "..", "inst", "extdata", "analysis_runtime", "Analysis2")))

  installed_path <- system.file("extdata", "analysis_runtime", "Analysis2", package = "ndm")
  expect_false(nzchar(installed_path) && dir.exists(installed_path))
})

test_that("sysdata keeps model specs but not the retired embedded runtime payload", {
  ns <- asNamespace("ndm")

  expect_true(exists(".ndm_embedded_model_spec_sources", envir = ns, inherits = FALSE))
  expect_false(exists(".ndm_embedded_runtime_sources", envir = ns, inherits = FALSE))
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
