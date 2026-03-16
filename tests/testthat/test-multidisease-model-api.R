ndm_test_multidisease_project_root <- function() {
  root <- tempfile("ndm-multidisease-")
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  root
}

ndm_test_write_ihme_fixture <- function(project_root) {
  data_dir <- file.path(
    project_root,
    "Data",
    "MultiDiseaseRuns",
    "DiseasePool",
    "IHMEData",
    "IHME-GBD_2021_DATA-ea2ad67b-1"
  )
  dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

  prevalence_rows <- do.call(
    rbind,
    lapply(2000:2007, function(year_) {
      year_offset <- year_ - 2000L
      data.frame(
        sex_name = c("Both", "Both"),
        age_name = c("All ages", "All ages"),
        cause_name = c(
          "HIV/AIDS and sexually transmitted infections",
          "HIV/AIDS and sexually transmitted infections"
        ),
        measure_name = c("Prevalence", "Prevalence"),
        metric_name = c("Rate", "Rate"),
        val = c(100 + 5 * year_offset, 120 + 5 * year_offset),
        location_id = c(1L, 2L),
        location_name = c("Loc1", "Loc2"),
        year = c(year_, year_),
        stringsAsFactors = FALSE
      )
    })
  )

  fixture <- rbind(
    prevalence_rows,
    data.frame(
      sex_name = "Both",
      age_name = "All ages",
      cause_name = "HIV/AIDS and sexually transmitted infections",
      measure_name = "Prevalence",
      metric_name = "Percent",
      val = 2,
      location_id = 1L,
      location_name = "Loc1",
      year = 2000L,
      stringsAsFactors = FALSE
    ),
    data.frame(
      sex_name = "Both",
      age_name = "All ages",
      cause_name = "HIV/AIDS and sexually transmitted infections",
      measure_name = "Incidence",
      metric_name = "Rate",
      val = 50,
      location_id = 1L,
      location_name = "Loc1",
      year = 2000L,
      stringsAsFactors = FALSE
    ),
    data.frame(
      sex_name = "Male",
      age_name = "All ages",
      cause_name = "HIV/AIDS and sexually transmitted infections",
      measure_name = "Prevalence",
      metric_name = "Rate",
      val = 999,
      location_id = 1L,
      location_name = "Loc1",
      year = 2000L,
      stringsAsFactors = FALSE
    )
  )

  utils::write.csv(
    fixture,
    file.path(data_dir, "IHME-GBD_2021_DATA-ea2ad67b-1.csv"),
    row.names = FALSE
  )
}

ndm_test_write_who_fixture <- function(project_root) {
  data_dir <- file.path(project_root, "Data", "MultiDiseaseRuns", "DiseasePool")
  dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

  writeLines(
    c(
      "",
      "Entity,Code,Year,Estimated incidence of all forms of tuberculosis",
      "Country A,CTA,2000,15",
      "Country A,CTA,2001,20",
      "Country B,CTB,2000,30"
    ),
    con = file.path(data_dir, "incidence-of-tuberculosis-sdgs.csv")
  )
}

ndm_test_write_tycho_fixture <- function(project_root) {
  data_dir <- file.path(project_root, "Data", "MultiDiseaseRuns", "TB-USA-2")
  dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

  tycho_fixture <- data.frame(
    PeriodStartDate = c("2020-01-05", "2020-01-12"),
    Admin1ISO = c("US-AL", "US-AL"),
    Admin1Name = c("Alabama", "Alabama"),
    CountValue = c(10, 20),
    stringsAsFactors = FALSE
  )
  utils::write.csv(
    tycho_fixture,
    file.path(data_dir, "tycho_fixture.csv"),
    row.names = FALSE
  )

  pop_dir <- file.path(project_root, "Data", "MultiDiseaseRuns")
  dir.create(pop_dir, recursive = TRUE, showWarnings = FALSE)
  writeLines(
    c("AL,2020,4903185"),
    con = file.path(pop_dir, "population_state_year.csv")
  )
}

ndm_skip_if_no_multidisease_backend <- function() {
  skip_on_cran()
  ndm_require_backend_test_stack("multidisease tests")
}

test_that("multidisease preparation validates required globals early", {
  env <- ndm_new_runtime_env()

  expect_error(
    ndm_prepare_data(env, generator = "multidisease"),
    paste(
      "Missing required multidisease globals:",
      "ContextLength, evaluationTime, dataInputs, initialTransform,",
      "initialNormType, paddingMethod, OSSType, nSamplesTrain"
    )
  )
})

test_that("IHME bundle loading resolves aliases and rejects unsupported diseases", {
  project_root <- ndm_test_multidisease_project_root()
  ndm_test_write_ihme_fixture(project_root)

  bundle <- ndm:::.ndm_load_multidisease_bundle(
    project_root = project_root,
    data_format = "IHME",
    disease_names = "hiv"
  )

  expect_equal(
    bundle$resolved_diseases,
    "HIV/AIDS and sexually transmitted infections"
  )
  expect_equal(bundle$dataInputs_colnames_past, c("CountValue", "Covariate1"))
  expect_true(all(c("CountValue", "Covariate1") %in% names(bundle$truth_df_red)))
  expect_true(all(bundle$truth_df_red$CountValue > 0))

  expect_error(
    ndm:::.ndm_load_multidisease_bundle(
      project_root = project_root,
      data_format = "IHME",
      disease_names = "influenza"
    ),
    "not available for IHME"
  )
})

test_that("WHO bundle loading supports renamed outcomes and TB-only validation", {
  project_root <- ndm_test_multidisease_project_root()
  ndm_test_write_who_fixture(project_root)

  bundle <- ndm:::.ndm_load_multidisease_bundle(
    project_root = project_root,
    data_format = "WHO",
    disease_names = "TB",
    outcome_metric = "CaseRate"
  )

  expect_equal(bundle$resolved_diseases, "TB")
  expect_equal(bundle$true_value_names, "CaseRate")
  expect_true(all(c("CaseRate", "Covariate1") %in% names(bundle$truth_df_red)))
  expect_true(all(bundle$truth_df_red$CaseRate > 0))

  expect_error(
    ndm:::.ndm_multidisease_resolve_diseases("HIV", data_format = "WHO"),
    "TB only"
  )
})

test_that("Tycho bundle loading normalizes counts by population", {
  project_root <- ndm_test_multidisease_project_root()
  ndm_test_write_tycho_fixture(project_root)

  bundle <- ndm:::.ndm_load_multidisease_bundle(
    project_root = project_root,
    data_format = "Tycho",
    disease_names = "tuberculosis"
  )

  expect_equal(bundle$resolved_diseases, "TB")
  expect_true("POP_population" %in% names(bundle$truth_df_red))
  expect_true(all(bundle$truth_df_red$CountValue > 0))
  expect_true(all(bundle$truth_df_red$CountValue < 1))
})

test_that("IHME bundle loading honors desired_measure overrides", {
  project_root <- ndm_test_multidisease_project_root()
  ndm_test_write_ihme_fixture(project_root)

  bundle <- ndm:::.ndm_load_multidisease_bundle(
    project_root = project_root,
    data_format = "IHME",
    disease_names = "hiv",
    desired_measure = "Incidence"
  )

  expect_equal(bundle$resolved_diseases, "HIV/AIDS and sexually transmitted infections")
  expect_equal(nrow(bundle$truth_df_red), 1L)
  expect_equal(bundle$truth_df_red$CountValue, 50 / 1e5)
})

test_that("multidisease training calibrates and runs under project_root", {
  project_root <- ndm_test_multidisease_project_root()
  env <- ndm_new_runtime_env()
  env$ModelList <- list(model = TRUE)
  env$ndm_data_generator <- "multidisease"
  env$project_root <- project_root

  calibration_calls <- 0L
  cwd_seen <- character()

  local_mocked_bindings(
    ndm_source_runtime_calibration = function(analysis_root = .ndm_default_analysis_root(),
                                              env = ndm_new_runtime_env()) {
      calibration_calls <<- calibration_calls + 1L
      cwd_seen <<- c(cwd_seen, paste0("cal:", getwd()))
      assign("batch_l_cal", list(ok = TRUE), envir = env)
      invisible(env)
    },
    .ndm_source_runtime_file = function(path, env) {
      cwd_seen <<- c(cwd_seen, paste0(basename(path), ":", getwd()))
      if (grepl("TrainDefine", path, fixed = TRUE)) {
        assign("state", list(stage = "defined"), envir = env)
        assign("PriorList", list(prior = TRUE), envir = env)
        assign("PolicyList", list(policy = TRUE), envir = env)
        assign("GetPredSaveAtInfo_default", list(info = TRUE), envir = env)
      }
      if (grepl("TrainDo", path, fixed = TRUE)) {
        assign("state", list(stage = "trained"), envir = env)
        assign("PriorList", list(prior = TRUE), envir = env)
        assign("PolicyList", list(policy = TRUE), envir = env)
        assign("GetPredSaveAtInfo_default", list(info = TRUE), envir = env)
        assign("gradLoss_jax", TRUE, envir = env)
        assign("opt_state", list(opt = TRUE), envir = env)
      }
      invisible(env)
    }
  )

  trained <- ndm_train(env, run_define = TRUE, run_loop = TRUE)
  seen_dirs <- normalizePath(sub("^[^:]*:", "", cwd_seen), winslash = "/", mustWork = FALSE)
  expected_dir <- normalizePath(project_root, winslash = "/", mustWork = TRUE)

  expect_equal(calibration_calls, 1L)
  expect_true(exists("batch_l_cal", envir = env, inherits = FALSE))
  expect_true(all(seen_dirs == expected_dir))
  expect_equal(trained$state$stage, "trained")
})

test_that("multidisease preparation validates requested inputs and high_income fallback", {
  project_root <- ndm_test_multidisease_project_root()
  ndm_test_write_ihme_fixture(project_root)
  ndm_skip_if_no_multidisease_backend()
  old_backend <- ndm:::ndm_env$backend
  on.exit(assign("backend", old_backend, envir = ndm:::ndm_env), add = TRUE)
  ndm_initialize_backend(
    conda_env = ndm_backend_test_conda_env(),
    float_type = "32",
    import_tensorflow = TRUE
  )
  env <- ndm_prepare_runtime(
    config = ndm_create_config(
      model_type = "DecoderOnly",
      backbone = "transformer",
      float_type = "32",
      force_to_gpu = FALSE
    ),
    runtime_env = ndm_new_runtime_env(parent = globalenv()),
    runtime_globals = list(
      project_root = project_root,
      AnalysisName = "FixturePrep",
      AnalysisDate = Sys.Date(),
      COMMAND_ARG_INPUT = "test"
    )
  )
  old_warning <- getOption("warn")
  on.exit(options(warn = old_warning), add = TRUE)
  options(warn = 1)

  ndm_set_runtime_globals(
    env,
    list(
      project_root = project_root,
      ContextLength = 8L,
      evaluationTime = 1L,
      dataInputs = "all",
      initialTransform = "none",
      initialNormType = "all",
      paddingMethod = "left",
      OSSType = "OutOfTime",
      nSamplesTrain = 2L,
      disease_names = "hiv",
      data_format = "IHME",
      data_subset = "high_income",
      nBatch = 2L,
      nCheckpoints = 0L,
      nEpochesMax = 1L,
      ModelDepth = 1L,
      ModelDims = 8L,
      HolderFolder = tempfile("ndm-md-results-")
    )
  )
  dir.create(env$HolderFolder, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(env$HolderFolder, recursive = TRUE, force = TRUE), add = TRUE)

  local_mocked_bindings(
    ndm_source_runtime_data = function(...) invisible(env),
    .package = "ndm"
  )
  expect_warning(
    ndm_prepare_data(env, generator = "multidisease"),
    "high_income subset requested but LOC2_region_name not available"
  )
  expect_equal(sort(unique(env$truth_df_red$location_id)), c("1", "2"))

  env_bad <- ndm_prepare_runtime(
    config = ndm_create_config(
      model_type = "DecoderOnly",
      backbone = "transformer",
      float_type = "32",
      force_to_gpu = FALSE
    ),
    runtime_env = ndm_new_runtime_env(parent = globalenv()),
    runtime_globals = list(
      project_root = project_root,
      AnalysisName = "FixturePrepBad",
      AnalysisDate = Sys.Date(),
      COMMAND_ARG_INPUT = "test"
    )
  )
  ndm_set_runtime_globals(
    env_bad,
    as.list(env)
  )
  ndm_set_runtime_globals(env_bad, list(dataInputs = "MissingInput"))

  bundle <- ndm:::.ndm_load_multidisease_bundle(
    project_root = project_root,
    data_format = "IHME",
    disease_names = "hiv"
  )
  expect_error(
    ndm:::.ndm_multidisease_set_default_globals(env_bad, bundle = bundle),
    "Requested multidisease `dataInputs` are not present"
  )
})

test_that("multidisease preparation rejects retired TFRecord regeneration", {
  env <- ndm_new_runtime_env()
  env$resave_tfrecords <- TRUE

  expect_error(
    ndm_prepare_data(env, generator = "multidisease"),
    "no longer supported for multidisease workflows"
  )
})
