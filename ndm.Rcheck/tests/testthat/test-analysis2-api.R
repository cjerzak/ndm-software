test_that("package-native sim runner regenerates canonical TFRecords in non-dry mode", {
  ndm_require_runner_test_stack("package-native sim runner tests")

  project_root <- ndm_test_runner_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)

  analysis_name <- "SimNonDry"
  grid <- ndm_test_make_sim_run_grid()
  grid_path <- file.path(
    project_root,
    "Data",
    "RunGrids",
    "SimGrids",
    sprintf("SimGrid_%s.csv", analysis_name)
  )
  tfrecord_dir <- file.path(project_root, "Data", "RunTFRecords", "SimTFRecords", analysis_name)

  ndm_test_write_grid(grid, grid_path)

  resave <- ndm_run_sim(
    ndm_create_sim_run_config(
      project_root = project_root,
      analysis_name = analysis_name,
      grid = grid,
      outer = 1L,
      model_type = "DecoderOnly",
      dry_run = FALSE,
      resave_tfrecords = TRUE
    )
  )

  expect_equal(resave$written_base_ids, 1L)
  ndm_test_assert_canonical_tfrecords(tfrecord_dir, base_id = 1L)

  from_file <- ndm_run_sim(
    ndm_create_sim_run_config(
      project_root = project_root,
      analysis_name = analysis_name,
      grid_file = grid_path,
      outer = 1L,
      model_type = "DecoderOnly",
      dry_run = FALSE,
      resave_tfrecords = TRUE
    )
  )

  expect_equal(from_file$written_base_ids, 1L)
  ndm_test_assert_canonical_tfrecords(tfrecord_dir, base_id = 1L)
})

test_that("package-native real runner regenerates canonical TFRecords and completes non-dry runs", {
  ndm_require_runner_test_stack("package-native real runner tests")

  project_root <- ndm_test_runner_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)

  analysis_name <- "RealNonDry"
  raw_data_dir <- ndm_test_copy_raw_covid_fixture(project_root)
  grid <- ndm_test_make_real_run_grid()
  grid_path <- file.path(
    project_root,
    "Data",
    "RunGrids",
    "RealGrids",
    sprintf("RealGrid_%s.csv", analysis_name)
  )
  tfrecord_dir <- file.path(project_root, "Data", "RunTFRecords", "RealTFRecords", analysis_name)
  holder_folder <- file.path(project_root, "SavedResults", "Real", sprintf("Results_%s", analysis_name))
  saved_model_root <- file.path(project_root, "SavedModels", "FromReal")

  ndm_test_write_grid(grid, grid_path)

  resave_config <- ndm_create_real_run_config(
    project_root = project_root,
    analysis_name = analysis_name,
    grid = grid,
    outer = 1L,
    model_type = "NeuralODE",
    resave_tfrecords = TRUE,
    raw_data_dir = raw_data_dir,
    outcome_metric = "inc_death",
    data_subset = "all",
    dry_run = FALSE
  )
  expect_false("config_file" %in% names(resave_config))

  resave <- ndm_run_real(resave_config)

  expect_equal(resave$written_base_ids, 1L)
  ndm_test_assert_canonical_tfrecords(tfrecord_dir, base_id = 1L)

  result <- ndm_run_real(
    ndm_create_real_run_config(
      project_root = project_root,
      analysis_name = analysis_name,
      grid_file = grid_path,
      outer = 1L,
      model_type = "NeuralODE",
      resave_tfrecords = FALSE,
      raw_data_dir = raw_data_dir,
      outcome_metric = "inc_death",
      data_subset = "all",
      dry_run = FALSE
    )
  )

  expect_true(isTRUE(result))
  expect_true(dir.exists(holder_folder))
  expect_gt(length(list.files(holder_folder, full.names = TRUE)), 0L)
  expect_true(dir.exists(saved_model_root))
  expect_gte(
    length(list.files(saved_model_root, pattern = paste0("^Model_", analysis_name), full.names = TRUE)),
    1L
  )
})

test_that("package-native multidisease runner completes a non-dry run", {
  ndm_require_runner_test_stack("package-native multidisease runner tests")

  project_root <- ndm_test_multidisease_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)

  analysis_name <- "MultidiseaseNonDry"
  grid <- ndm_test_make_multidisease_run_grid()
  holder_folder <- file.path(project_root, "SavedResults", "Real", sprintf("Results_%s", analysis_name))

  ndm_test_write_ihme_fixture(project_root)

  result <- ndm_run_multidisease(
    ndm_create_multidisease_run_config(
      project_root = project_root,
      analysis_name = analysis_name,
      grid = grid,
      outer = 1L,
      model_type = "NeuralODE",
      data_format = "IHME",
      disease_names = "hiv",
      data_subset = "all",
      outcome_metric = "CountValue",
      dry_run = FALSE
    )
  )

  expect_true(isTRUE(result))
  expect_true(dir.exists(holder_folder))
  expect_gt(length(list.files(holder_folder, full.names = TRUE)), 0L)
})

test_that("legacy Analysis2-branded entrypoints are not exported", {
  exports <- getNamespaceExports("ndm")

  expect_false("ndm_run_analysis2_real" %in% exports)
  expect_false("ndm_run_analysis2_sim" %in% exports)
  expect_false("ndm_run_analysis2_multidisease" %in% exports)
})

test_that("Analysis2 compatibility hooks remain callable via getExportedValue", {
  helper_names <- c(
    "ndm_source_runtime_calibration",
    "ndm_source_runtime_results_get",
    "ndm_source_runtime_results_analyze"
  )
  helper_targets <- c(
    "SuperLModel_CalibrateML.R",
    "SuperLModel_GetAnalytics.R",
    "SuperLModel_GenFigs.R"
  )
  env <- ndm_new_runtime_env()
  sourced <- character()

  local_mocked_bindings(
    .ndm_install_runtime_helpers = function(env, analysis_root = .ndm_default_analysis_root()) {
      invisible(env)
    },
    ndm_runtime_paths = function(analysis_root = .ndm_default_analysis_root()) {
      list(
        calibrate_ml = file.path(analysis_root, helper_targets[[1]]),
        results_get = file.path(analysis_root, helper_targets[[2]]),
        results_analyze = file.path(analysis_root, helper_targets[[3]])
      )
    },
    .ndm_source_runtime_file = function(path, env) {
      sourced <<- c(sourced, basename(path))
      invisible(env)
    },
    .package = "ndm"
  )

  funcs <- lapply(helper_names, function(name) getExportedValue("ndm", name))

  expect_true(all(vapply(funcs, is.function, logical(1))))
  for (fn in funcs) {
    fn(analysis_root = tempdir(), env = env)
  }
  expect_equal(sourced, helper_targets)
})

test_that("run config helpers validate mutually exclusive grid inputs and outer rows", {
  project_root <- tempdir()

  expect_error(
    ndm:::.ndm_make_run_config(
      mode = "real",
      project_root = project_root,
      analysis_name = "Demo",
      grid = data.frame(BaseID = 1L),
      grid_file = "grid.csv",
      outer = 1L
    ),
    "Supply either `grid` or `grid_file`, not both"
  )

  expect_error(
    ndm:::.ndm_make_run_config(
      mode = "real",
      project_root = project_root,
      analysis_name = "Demo",
      outer = integer()
    ),
    "`outer` must contain at least one integer row index"
  )
})

test_that("multidisease run configs reject retired TFRecord regeneration", {
  expect_error(
    ndm_create_multidisease_run_config(
      project_root = tempdir(),
      resave_tfrecords = TRUE
    ),
    "no longer supported for multidisease workflows"
  )
})
