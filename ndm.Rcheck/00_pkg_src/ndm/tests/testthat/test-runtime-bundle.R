test_that("runtime bundle helpers source embedded files into fresh environments", {
  env <- ndm_new_runtime_env()

  expect_true(is.function(env$ndm_source_extracted))
  env$ndm_source_extracted("SetupEnv/SuperLModel_helperFxns.R", env_target = env)

  expect_true(exists("ai", envir = env, inherits = FALSE))
  expect_true(exists("GlobalPartition", envir = env, inherits = FALSE))
})

test_that("runtime path evaluation falls back to filesystem sources", {
  env <- new.env(parent = baseenv())
  script <- tempfile(fileext = ".R")
  on.exit(unlink(script), add = TRUE)
  writeLines("test_value <- 42L", con = script, useBytes = TRUE)

  ndm:::.ndm_eval_runtime_path(script, env = env, analysis_root = NULL)

  expect_equal(env$test_value, 42L)
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
