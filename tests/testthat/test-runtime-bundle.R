test_that("runtime helpers honor caller-supplied analysis_root end-to-end", {
  fixture <- make_runtime_fixture()
  on.exit(unlink(fixture$root, recursive = TRUE), add = TRUE)

  cfg <- ndm_create_config(
    model_type = "DecoderOnly",
    backbone = "transformer",
    analysis_root = fixture$analysis_root,
    float_type = "32",
    force_to_gpu = FALSE,
    resave_tfrecords = TRUE
  )

  paths <- ndm_runtime_paths(cfg$analysis_root)
  expect_equal(paths$analysis_root, fixture$analysis_root)
  expect_true(file.exists(paths$helper_fxns))
  expect_true(file.exists(paths$calibrate_ml))
  expect_true(file.exists(paths$build_model))
  expect_true(file.exists(paths$results_get))
  expect_true(dir.exists(paths$model_structure_dir))

  runtime_env <- ndm_new_runtime_env()
  ndm_prepare_runtime(cfg, runtime_env = runtime_env)
  expect_true(isTRUE(runtime_env$fixture_helper_loaded))
  expect_true(isTRUE(runtime_env$fixture_backend_loaded))
  expect_equal(runtime_env$fixture_backend_settings$floatType, "32")
  expect_false(isTRUE(runtime_env$fixture_backend_settings$force2GPU))
  expect_true(isTRUE(runtime_env$fixture_backend_settings$ReSaveTfRecords))

  ndm_prepare_data(
    runtime_env = runtime_env,
    analysis_root = cfg$analysis_root,
    generator = "sim"
  )
  expect_equal(runtime_env$fixture_data_generator, "sim")
  expect_true(isTRUE(runtime_env$fixture_relative_loaded))
  expect_equal(runtime_env$NDM_INTERNAL_ANALYSIS_DIR, fixture$analysis_root)

  model <- ndm_build_model(
    runtime_env = runtime_env,
    model_type = "DecoderOnly"
  )
  expect_equal(model$analysis_root, fixture$analysis_root)
  expect_equal(runtime_env$fixture_build_model_analysis_root, fixture$analysis_root)

  ndm_source_runtime_calibration(analysis_root = cfg$analysis_root, env = runtime_env)
  ndm_source_runtime_results_get(analysis_root = cfg$analysis_root, env = runtime_env)
  ndm_source_runtime_results_analyze(analysis_root = cfg$analysis_root, env = runtime_env)
  expect_true(isTRUE(runtime_env$fixture_calibration_loaded))
  expect_true(isTRUE(runtime_env$fixture_results_get_loaded))
  expect_true(isTRUE(runtime_env$fixture_results_analyze_loaded))

  trained <- ndm_train(runtime_env)
  expect_equal(trained$analysis_root, fixture$analysis_root)
  expect_true(isTRUE(runtime_env$fixture_train_define_loaded))
  expect_true(isTRUE(runtime_env$fixture_train_do_loaded))
  expect_equal(runtime_env$state$stage, "trained")
  expect_equal(runtime_env$opt_state$stage, "trained")
})

test_that("build fails fast when runtime outputs are incomplete", {
  fixture <- make_runtime_fixture()
  on.exit(unlink(fixture$root, recursive = TRUE), add = TRUE)

  writeLines(
    c(
      "ModelList <- list(source = NDM_INTERNAL_ANALYSIS_DIR)",
      "state <- list(stage = \"built\")"
    ),
    file.path(fixture$analysis_root, "ModelDefiners", "SuperLModel_BuildML.R")
  )

  cfg <- ndm_create_config(
    model_type = "DecoderOnly",
    backbone = "transformer",
    analysis_root = fixture$analysis_root,
    force_to_gpu = FALSE
  )

  runtime_env <- ndm_prepare_runtime(cfg, runtime_env = ndm_new_runtime_env())
  ndm_prepare_data(runtime_env = runtime_env, analysis_root = cfg$analysis_root, generator = "sim")

  expect_error(
    ndm_build_model(runtime_env = runtime_env, model_type = "DecoderOnly"),
    "PriorList"
  )
})

test_that("runtime environments accept explicit globals", {
  env <- ndm_new_runtime_env()
  ndm_set_runtime_globals(env, list(ModelType = "DecoderOnly", BackboneType = "transformer"))

  expect_true(is.environment(env))
  expect_equal(get("ModelType", envir = env), "DecoderOnly")
  expect_equal(get("BackboneType", envir = env), "transformer")
})
