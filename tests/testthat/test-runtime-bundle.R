test_that("runtime paths resolve against the vendored package analysis code", {
  paths <- ndm_runtime_paths()

  expect_true(dir.exists(paths$analysis_root))
  expect_false(grepl("/Dropbox/CovidSuperlearner/Analysis/?$", paths$analysis_root))
  expect_true(file.exists(paths$helper_fxns))
  expect_true(file.exists(paths$calibrate_ml))
  expect_true(file.exists(paths$build_model))
  expect_true(file.exists(paths$results_get))
  expect_true(dir.exists(paths$model_structure_dir))
})

test_that("runtime environments accept explicit globals", {
  env <- ndm_new_runtime_env()
  ndm_set_runtime_globals(env, list(ModelType = "DecoderOnly", BackboneType = "transformer"))

  expect_true(is.environment(env))
  expect_equal(get("ModelType", envir = env), "DecoderOnly")
  expect_equal(get("BackboneType", envir = env), "transformer")
})
