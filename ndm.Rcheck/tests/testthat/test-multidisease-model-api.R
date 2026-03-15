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

  fixture <- data.frame(
    sex_name = c("Both", "Both", "Both", "Both", "Male"),
    age_name = c("All ages", "All ages", "All ages", "All ages", "All ages"),
    cause_name = c(
      "HIV/AIDS and sexually transmitted infections",
      "HIV/AIDS and sexually transmitted infections",
      "HIV/AIDS and sexually transmitted infections",
      "All causes",
      "HIV/AIDS and sexually transmitted infections"
    ),
    measure_name = c("Prevalence", "Prevalence", "Incidence", "Prevalence", "Prevalence"),
    metric_name = c("Rate", "Percent", "Rate", "Rate", "Rate"),
    val = c(100, 2, 50, 200, 999),
    location_id = c(1L, 2L, 1L, 1L, 1L),
    location_name = c("Loc1", "Loc2", "Loc1", "Loc1", "Loc1"),
    year = c(2000L, 2001L, 2000L, 2000L, 2000L),
    stringsAsFactors = FALSE
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
  skip_if_not_installed("reticulate")
  skip_if_not_installed("fastmatch")
  skip_if_not_installed("rrapply")
  skip_if_not_installed("progress")
  skip_if_not_installed("zip")
  skip_if_not_installed("zoo")

  backend_ready <- ndm_check_backend(
    conda_env = "jax_cpu",
    modules = c("jax", "numpy", "optax", "equinox", "diffrax", "tensorflow")
  )
  skip_if(is.null(backend_ready), "jax_cpu with JAX/TensorFlow is required for the multidisease fit test.")
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

test_that("opt-in multidisease smoke fit reaches the package-native path", {
  project_root <- Sys.getenv("NDM_MULTIDISEASE_PROJECT_ROOT", unset = "")
  skip_if(!nzchar(project_root), "Set NDM_MULTIDISEASE_PROJECT_ROOT to run the multidisease smoke test.")
  skip_if(!dir.exists(project_root), "NDM_MULTIDISEASE_PROJECT_ROOT does not exist.")

  ihme_paths <- c(
    file.path(project_root, "Data", "MultiDiseaseRuns", "IHMEData", "IHME-GBD_2021_DATA-ea2ad67b-1", "IHME-GBD_2021_DATA-ea2ad67b-1.csv"),
    file.path(project_root, "Data", "MultiDiseaseRuns", "DiseasePool", "IHMEData", "IHME-GBD_2021_DATA-ea2ad67b-1", "IHME-GBD_2021_DATA-ea2ad67b-1.csv")
  )
  skip_if(!any(file.exists(ihme_paths)), "IHME multidisease data was not found under the supplied project root.")
  ndm_skip_if_no_multidisease_backend()

  analysis_name <- sprintf("TestMultidisease_%s", as.integer(Sys.time()))
  analysis_date <- Sys.Date()
  saved_model_dir <- file.path(
    project_root,
    "SavedModels",
    "FromReal",
    sprintf("Model_%s_%s", analysis_name, analysis_date)
  )
  holder_folder <- tempfile("ndm-multidisease-results-")
  dir.create(holder_folder, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(holder_folder, recursive = TRUE, force = TRUE), add = TRUE)
  on.exit({
    parent_dir <- dirname(saved_model_dir)
    if (dir.exists(parent_dir)) {
      model_dirs <- list.files(
        parent_dir,
        pattern = paste0("^", basename(saved_model_dir)),
        full.names = TRUE
      )
      unlink(model_dirs, recursive = TRUE, force = TRUE)
    }
  }, add = TRUE)

  config <- ndm_create_config(
    model_type = "DecoderOnly",
    backbone = "transformer",
    float_type = "32",
    force_to_gpu = FALSE,
    resave_tfrecords = TRUE
  )
  spec <- ndm_model_spec(
    preset = "seirs_dynamic_beta",
    model_type = "DecoderOnly"
  )

  trained <- ndm_fit(
    config = config,
    model_spec = spec,
    data_generator = "multidisease",
    runtime_globals = list(
      project_root = project_root,
      AnalysisName = analysis_name,
      AnalysisDate = analysis_date,
      COMMAND_ARG_INPUT = "test",
      nBatch = 2L,
      nCheckpoints = 0L,
      nEpochesMax = 1L,
      nSamples_max = 2L,
      LEARNING_RATE_MAX = 1e-3,
      LEARNING_RATE_MAX_model = 1e-3,
      LEARNING_RATE_MAX_pretrain = 1e-5,
      HolderFolder = holder_folder,
      ModelDepth = 1L,
      ModelDims = 16L
    ),
    data_globals = list(
      ContextLength = 8L,
      evaluationTime = 1L,
      dataInputs = "all",
      OSSType = "OutOfTime",
      initialTransform = "none",
      initialNormType = "all",
      paddingMethod = "left",
      nSamplesTrain = 2L,
      disease_names = "all",
      data_format = "IHME",
      data_subset = "all"
    ),
    build_globals = list(
      ModelDepth = 1L,
      ModelDims = 16L
    ),
    run_define = TRUE,
    run_loop = TRUE
  )

  expect_s3_class(trained, "ndm_trained_model")
  expect_true(exists("batch_l_cal", envir = trained$env, inherits = FALSE))
  expect_equal(
    get("ndm_data_generator", envir = trained$env, inherits = FALSE),
    "multidisease"
  )
})
