test_that("canonical sim bootstrap coordinates concurrent writers per BaseID", {
  ndm_require_runner_test_stack("package-native sim bootstrap concurrency tests")
  skip_if(.Platform$OS.type == "windows")

  project_root <- ndm_test_runner_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)

  analysis_name <- "SimBootstrapConcurrency"
  grid <- ndm_test_make_sim_duplicate_base_grid(
    n_samples_train = c(1024L, 1024L),
    resave_flags = c(0L, 1L)
  )
  tfrecord_dir <- file.path(project_root, "Data", "RunTFRecords", "SimTFRecords", analysis_name)
  producer <- ndm_test_tfrecord_producer()

  workers <- list(
    parallel::mcparallel(
      ndm_bootstrap_sim_tfrecords(
        project_root = project_root,
        analysis_name = analysis_name,
        grid = grid,
        base_ids = 1L,
        producer = producer,
        overwrite = FALSE,
        dry_run = FALSE
      )
    ),
    parallel::mcparallel(
      ndm_bootstrap_sim_tfrecords(
        project_root = project_root,
        analysis_name = analysis_name,
        grid = grid,
        base_ids = 1L,
        producer = producer,
        overwrite = FALSE,
        dry_run = FALSE
      )
    )
  )
  results <- parallel::mccollect(workers, wait = TRUE)

  expect_false(any(vapply(results, inherits, logical(1), what = "try-error")))

  statuses <- unname(vapply(results, function(res) res$status[[1L]], character(1)))
  manifest <- ndm_test_read_canonical_manifest(tfrecord_dir, base_id = 1L, split = "train")

  expect_equal(sum(statuses == "written"), 1L)
  expect_equal(sum(statuses %in% c("skipped_existing", "skipped_locked_existing")), 1L)
  expect_equal(manifest$metadata$n_examples, 1024L)
  expect_identical(manifest$producer, producer)
})

test_that("canonical sim bootstrap dry run plans one row per BaseID in raw order", {
  ndm_require_runner_test_stack("package-native sim bootstrap dry-run tests")

  project_root <- ndm_test_runner_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)

  analysis_name <- "SimBootstrapDryRun"
  grid <- ndm_test_make_sim_bootstrap_grid(
    n_base_ids = 12L,
    rows_per_base_id = 4L,
    canonical_n_samples_train = 8L,
    other_n_samples_train = 4L
  )

  plan <- ndm_bootstrap_sim_tfrecords(
    project_root = project_root,
    analysis_name = analysis_name,
    grid = grid,
    dry_run = TRUE
  )

  expect_true(is.data.frame(plan))
  expect_equal(nrow(plan), 12L)
  expect_equal(plan$BaseID, 1:12)
  expect_equal(plan$canonical_row, 1:12)
  expect_true(all(plan$artifact_n_samples_train == 8L))
  expect_true(all(plan$status == "planned"))
})

test_that("canonical real bootstrap exposes an explicit BaseID-scoped dry-run plan", {
  ndm_require_runner_test_stack("package-native real bootstrap dry-run tests")
  project_root <- ndm_test_runner_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)
  grid <- ndm_test_make_real_duplicate_base_grid(
    n_samples_train = c(4L, 8L),
    resave_flags = c(0L, 1L)
  )

  plan <- ndm_bootstrap_real_tfrecords(
    project_root = project_root,
    grid = grid,
    base_ids = 1L,
    dry_run = TRUE
  )

  expect_equal(plan$BaseID, 1L)
  expect_equal(plan$canonical_row, 2L)
  expect_equal(plan$artifact_n_samples_train, 8L)
  expect_equal(plan$status, "planned")
  expect_false(dir.exists(file.path(project_root, "Data", "RunTFRecords")))
})

test_that("canonical publication requires producer metadata", {
  expect_error(
    ndm_bootstrap_sim_tfrecords(project_root = tempdir(), grid = data.frame()),
    "`producer` is required"
  )
  expect_error(
    ndm_bootstrap_real_tfrecords(project_root = tempdir(), grid = data.frame()),
    "`producer` is required"
  )
})

test_that("canonical sim bootstrap writes the first 10 canonical BaseIDs once each", {
  ndm_require_runner_test_stack("package-native sim bootstrap integration tests")

  project_root <- ndm_test_runner_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)

  analysis_name <- "SimBootstrapIntegration"
  grid <- ndm_test_make_sim_bootstrap_grid(
    n_base_ids = 12L,
    rows_per_base_id = 4L,
    canonical_n_samples_train = 8L,
    other_n_samples_train = 4L
  )
  tfrecord_dir <- file.path(project_root, "Data", "RunTFRecords", "SimTFRecords", analysis_name)

  written <- ndm_bootstrap_sim_tfrecords(
    project_root = project_root,
    analysis_name = analysis_name,
    grid = grid,
    base_ids = 1:10,
    producer = ndm_test_tfrecord_producer(),
    overwrite = FALSE,
    dry_run = FALSE
  )

  expect_equal(written$BaseID, 1:10)
  expect_true(all(written$status == "written"))
  expect_equal(length(unique(written$train_file)), 10L)
  expect_equal(length(unique(written$inference_file)), 10L)
  for (base_id in 1:10) {
    ndm_test_assert_canonical_tfrecords(tfrecord_dir, base_id = base_id)
  }
})

test_that("Analysis2 training TFRecord shuffle is reproducible from run_seed", {
  ndm_require_runner_test_stack("package-native seeded TFRecord shuffle tests")
  project_root <- ndm_test_runner_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)
  grid <- ndm_test_make_sim_run_grid_with_samples(16L)
  tfrecord_dir <- file.path(project_root, "Data", "RunTFRecords", "SimTFRecords", "SeededShuffle")
  ndm_bootstrap_sim_tfrecords(
    project_root = project_root,
    analysis_name = "SeededShuffle",
    grid = grid,
    tfrecord_dir = tfrecord_dir,
    producer = ndm_test_tfrecord_producer(),
    overwrite = TRUE,
    dry_run = FALSE
  )

  backend <- ndm_backend_modules()
  api <- ndm:::.ndm_legacy_run_env(refresh = TRUE)
  first_batch <- function(seed) {
    runtime_env <- ndm_new_runtime_env()
    runtime_env$tf <- backend$tf
    runtime_env$nSamplesTrain <- 16L
    runtime_env$nObsInference <- 8L
    api$analysis2_attach_canonical_tfrecords(
      ndmdatasets_pkg = "ndmdatasets",
      runtime_env = runtime_env,
      train_file = file.path(tfrecord_dir, "train_1.tfrecord"),
      inference_file = file.path(tfrecord_dir, "inference_1.tfrecord"),
      schema_kind = "sim",
      batch_size = 8L,
      shuffle_train = TRUE,
      run_seed = seed
    )
    batch <- reticulate::iter_next(runtime_env$TFDatasetIterator_train)
    lapply(batch, function(value) as.array(backend$np$asarray(value)))
  }

  seed_11_a <- first_batch(11L)
  seed_11_b <- first_batch(11L)
  seed_12 <- first_batch(12L)
  expect_equal(seed_11_a, seed_11_b, tolerance = 0)
  expect_false(identical(seed_11_a, seed_12))
})

test_that("canonical sim bootstrap skips existing artifacts on rerun", {
  ndm_require_runner_test_stack("package-native sim bootstrap rerun tests")

  project_root <- ndm_test_runner_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)

  analysis_name <- "SimBootstrapRerun"
  grid <- ndm_test_make_sim_bootstrap_grid(
    n_base_ids = 4L,
    rows_per_base_id = 3L,
    canonical_n_samples_train = 8L,
    other_n_samples_train = 4L
  )
  tfrecord_dir <- file.path(project_root, "Data", "RunTFRecords", "SimTFRecords", analysis_name)

  first <- ndm_bootstrap_sim_tfrecords(
    project_root = project_root,
    analysis_name = analysis_name,
    grid = grid,
    producer = ndm_test_tfrecord_producer(),
    parallel_workers = 2L,
    overwrite = FALSE,
    dry_run = FALSE
  )
  second <- ndm_bootstrap_sim_tfrecords(
    project_root = project_root,
    analysis_name = analysis_name,
    grid = grid,
    producer = ndm_test_tfrecord_producer(),
    overwrite = FALSE,
    dry_run = FALSE
  )

  manifest <- ndm_test_read_canonical_manifest(tfrecord_dir, base_id = 1L, split = "train")

  expect_true(all(first$status == "written"))
  expect_true(all(second$status == "skipped_existing"))
  expect_equal(manifest$metadata$n_examples, 8L)
})

test_that("canonical sim bootstrap rejects non-max canonical flags within a BaseID", {
  ndm_require_runner_test_stack("package-native sim bootstrap flag validation tests")

  project_root <- ndm_test_runner_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)

  grid <- ndm_test_make_sim_duplicate_base_grid(
    n_samples_train = c(4L, 8L),
    resave_flags = c(1L, 0L)
  )

  expect_error(
    ndm_bootstrap_sim_tfrecords(
      project_root = project_root,
      analysis_name = "SimBootstrapBadFlag",
      grid = grid,
      dry_run = TRUE
    ),
    "requires the flagged row to use the largest `nSamplesTrain`"
  )
})

test_that("canonical sim bootstrap rejects conflicting dataset fields within a BaseID", {
  ndm_require_runner_test_stack("package-native sim bootstrap duplicate field validation tests")

  project_root <- ndm_test_runner_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)

  grid <- ndm_test_make_sim_duplicate_base_grid(
    n_samples_train = c(4L, 8L),
    resave_flags = c(0L, 1L)
  )
  grid$gamma[[2L]] <- 0.4

  expect_error(
    ndm_bootstrap_sim_tfrecords(
      project_root = project_root,
      analysis_name = "SimBootstrapConflict",
      grid = grid,
      dry_run = TRUE
    ),
    "disagree on dataset-defining fields.*gamma"
  )
})

test_that("canonical bootstraps treat production row and pair identities as run metadata", {
  ndm_require_runner_test_stack("production-shaped canonical bootstrap identity tests")

  project_root <- ndm_test_runner_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)

  sim_grid <- ndm_test_make_sim_duplicate_base_grid(
    n_samples_train = c(4L, 8L),
    resave_flags = c(0L, 1L)
  )
  sim_grid$row_id <- c("sim-row-a", "sim-row-b")
  sim_grid$pair_id <- c("sim-pair-a", "sim-pair-b")
  sim_plan <- ndm_bootstrap_sim_tfrecords(
    project_root = project_root,
    analysis_name = "SimBootstrapProductionIdentity",
    grid = sim_grid,
    dry_run = TRUE
  )

  real_grid <- ndm_test_make_real_duplicate_base_grid(
    n_samples_train = c(4L, 8L),
    resave_flags = c(0L, 1L)
  )
  real_grid$row_id <- c("real-row-a", "real-row-b")
  real_grid$pair_id <- c("real-pair-a", "real-pair-b")
  real_plan <- ndm_bootstrap_real_tfrecords(
    project_root = project_root,
    analysis_name = "RealBootstrapProductionIdentity",
    grid = real_grid,
    dry_run = TRUE
  )

  expect_equal(sim_plan$BaseID, 1L)
  expect_equal(real_plan$BaseID, 1L)
})

test_that("package-native sim runner consumes a requested canonical prefix", {
  ndm_require_runner_test_stack("package-native sim canonical prefix tests")

  project_root <- ndm_test_runner_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)

  analysis_name <- "SimDuplicateBaseID"
  grid <- ndm_test_make_sim_duplicate_base_grid(
    n_samples_train = c(4L, 8L),
    resave_flags = c(0L, 1L)
  )
  tfrecord_dir <- file.path(project_root, "Data", "RunTFRecords", "SimTFRecords", analysis_name)

  ndm_bootstrap_sim_tfrecords(
    project_root = project_root,
    analysis_name = analysis_name,
    grid = grid,
    producer = ndm_test_tfrecord_producer(),
    overwrite = FALSE
  )

  manifest <- ndm_test_read_canonical_manifest(tfrecord_dir, base_id = 1L, split = "train")
  expect_equal(manifest$metadata$n_examples, 8L)

  result <- ndm_run_sim(
    ndm_create_sim_run_config(
      project_root = project_root,
      analysis_name = analysis_name,
      grid = grid,
      outer = 1L,
      model_type = "DecoderOnly",
      force_to_gpu = FALSE,
      max_sgd_steps = 1L,
      resave_tfrecords = FALSE,
      dry_run = FALSE
    )
  )

  expect_true(isTRUE(result))
  expect_equal(
    ndm_test_read_canonical_manifest(tfrecord_dir, base_id = 1L, split = "train")$n_examples,
    8L
  )
})

test_that("package-native real runner consumes a requested canonical prefix", {
  ndm_require_runner_test_stack("package-native real canonical prefix tests")

  project_root <- ndm_test_runner_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)

  analysis_name <- "RealDuplicateBaseID"
  raw_data_dir <- ndm_test_copy_raw_covid_fixture(project_root)
  grid <- ndm_test_make_real_duplicate_base_grid(
    n_samples_train = c(4L, 8L),
    resave_flags = c(0L, 1L)
  )
  tfrecord_dir <- file.path(project_root, "Data", "RunTFRecords", "RealTFRecords", analysis_name)

  ndm_bootstrap_real_tfrecords(
    project_root = project_root,
    analysis_name = analysis_name,
    grid = grid,
    raw_data_dir = raw_data_dir,
    outcome_metric = "inc_death",
    data_subset = "all",
    producer = ndm_test_tfrecord_producer(),
    overwrite = FALSE
  )

  manifest <- ndm_test_read_canonical_manifest(tfrecord_dir, base_id = 1L, split = "train")
  expect_equal(manifest$metadata$n_examples, 8L)
  expect_identical(manifest$producer, ndm_test_tfrecord_producer())

  result <- ndm_run_real(
    ndm_create_real_run_config(
      project_root = project_root,
      analysis_name = analysis_name,
      grid = grid,
      outer = 1L,
      model_type = "NeuralODE",
      force_to_gpu = FALSE,
      max_sgd_steps = 1L,
      resave_tfrecords = FALSE,
      raw_data_dir = raw_data_dir,
      outcome_metric = "inc_death",
      data_subset = "all",
      dry_run = FALSE
    )
  )

  expect_true(isTRUE(result))
  expect_equal(
    ndm_test_read_canonical_manifest(tfrecord_dir, base_id = 1L, split = "train")$n_examples,
    8L
  )
})

test_that("training run configs reject inline TFRecord regeneration", {
  constructors <- list(
    real = ndm_create_real_run_config,
    sim = ndm_create_sim_run_config,
    multidisease = ndm_create_multidisease_run_config
  )
  guidance <- c(
    real = "ndm_bootstrap_real_tfrecords",
    sim = "ndm_bootstrap_sim_tfrecords",
    multidisease = "ndm_bootstrap_multidisease_tfrecords"
  )

  for (mode in names(constructors)) {
    expect_error(
      constructors[[mode]](project_root = tempdir(), resave_tfrecords = TRUE),
      guidance[[mode]],
      info = mode
    )
  }

  runtime <- ndm:::.ndm_new_run_impl_env()
  for (mode in names(constructors)) {
    for (invalid_value in list(TRUE, "TRUE")) {
      spec <- runtime$analysis2_mode_defaults(mode)
      spec$project_root <- tempdir()
      spec$resave_tfrecords <- invalid_value
      expect_error(
        runtime$analysis2_normalize_run_spec(
          spec,
          mode = mode,
          paths = list(project_root = tempdir())
        ),
        guidance[[mode]],
        info = paste("runtime", mode, deparse(invalid_value))
      )
    }

    spec <- runtime$analysis2_mode_defaults(mode)
    spec$project_root <- tempdir()
    spec$resave_tfrecords <- 1L
    expect_error(
      runtime$analysis2_normalize_run_spec(
        spec,
        mode = mode,
        paths = list(project_root = tempdir())
      ),
      "`resave_tfrecords` must be one non-missing logical value",
      fixed = TRUE,
      info = paste("runtime", mode, "numeric boolean")
    )
  }
})

test_that("package-native multidisease runner fails closed when canonical artifacts are missing", {
  ndm_require_runner_test_stack("package-native multidisease runner tests")

  project_root <- ndm_test_multidisease_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)

  analysis_name <- "MultidiseaseNonDry"
  grid <- ndm_test_make_multidisease_run_grid()
  tfrecord_dir <- file.path(project_root, "Data", "RunTFRecords", "RealTFRecords", analysis_name)

  ndm_test_write_ihme_fixture(project_root)

  expect_error(
    ndm_run_multidisease(
      ndm_create_multidisease_run_config(
        project_root = project_root,
        analysis_name = analysis_name,
        grid = grid,
        outer = 1L,
        model_type = "NeuralODE",
        force_to_gpu = FALSE,
        data_format = "IHME",
        disease_names = "hiv",
        data_subset = "all",
        outcome_metric = "CountValue",
        dry_run = FALSE
      )
    ),
    "ndm_bootstrap_multidisease_tfrecords"
  )
  expect_false(dir.exists(tfrecord_dir))
})

test_that("multidisease compatibility driver cannot enable inline TFRecord writing", {
  driver_source <- ndm:::.ndm_embedded_runtime_sources[[
    "SetupEnv/Analysis2_legacy_multidisease_driver.R"
  ]]

  expect_match(
    driver_source,
    "Canonical multidisease TFRecords are missing",
    fixed = TRUE
  )
  expect_match(
    driver_source,
    "ndm_bootstrap_multidisease_tfrecords",
    fixed = TRUE
  )
  expect_match(
    driver_source,
    ".ndm_preflight_multidisease_tfrecords",
    fixed = TRUE
  )
  preflight_position <- regexpr(
    ".ndm_preflight_multidisease_tfrecords",
    driver_source,
    fixed = TRUE
  )[[1L]]
  attach_source_position <- regexpr(
    'ndm_source_extracted("SetupData/SuperLModel_DataGenerator_Real.R")',
    driver_source,
    fixed = TRUE
  )[[1L]]
  expect_gt(preflight_position, 0L)
  expect_gt(attach_source_position, preflight_position)
  expect_false(grepl("ndm_real_bootstrap_tfrecords", driver_source, fixed = TRUE))
  expect_false(grepl("ReSaveTfRecords", driver_source, fixed = TRUE))
  expect_false(grepl(
    "generating them on demand",
    driver_source,
    fixed = TRUE
  ))
})

test_that("package-native sim dry runs anchor nSGD to the largest sibling sim grid", {
  project_root <- ndm_test_runner_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)

  ndm_test_write_grid(
    ndm_test_make_sim_run_grid_with_samples(100L),
    ndm_test_grid_path(project_root, mode = "sim", analysis_name = "SimAnchorPrimary")
  )
  ndm_test_write_grid(
    ndm_test_make_sim_run_grid_with_samples(1000L),
    ndm_test_grid_path(project_root, mode = "sim", analysis_name = "SimAnchorSibling")
  )

  primary <- ndm_run_sim(
    ndm_create_sim_run_config(
      project_root = project_root,
      analysis_name = "SimAnchorPrimary",
      outer = 1L,
      dry_run = TRUE
    )
  )
  sibling <- ndm_run_sim(
    ndm_create_sim_run_config(
      project_root = project_root,
      analysis_name = "SimAnchorSibling",
      outer = 1L,
      dry_run = TRUE
    )
  )

  expect_equal(primary$training_schedule$anchor_scope, "mode_folder_sim")
  expect_equal(primary$training_schedule$anchor_max_n_samples_train, 1000L)
  expect_equal(primary$training_schedule$resolved_n_sgd, as.integer(round(6 * (1000 / 32))))
  expect_equal(primary$training_schedule$resolved_n_sgd, sibling$training_schedule$resolved_n_sgd)
})

test_that("package-native real dry runs anchor nSGD to non-multidisease real grid maxima", {
  project_root <- ndm_test_runner_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)

  ndm_test_write_grid(
    ndm_test_make_real_run_grid_with_samples(100L),
    ndm_test_grid_path(project_root, mode = "real", analysis_name = "RealAnchorPrimary")
  )
  ndm_test_write_grid(
    ndm_test_make_real_run_grid_with_samples(640L),
    ndm_test_grid_path(project_root, mode = "real", analysis_name = "RealAnchorSibling")
  )
  ndm_test_write_grid(
    ndm_test_make_multidisease_run_grid_with_samples(4096L),
    ndm_test_grid_path(project_root, mode = "multidisease", analysis_name = "RealAnchor_MULTIDISEASE")
  )

  primary <- ndm_run_real(
    ndm_create_real_run_config(
      project_root = project_root,
      analysis_name = "RealAnchorPrimary",
      outer = 1L,
      dry_run = TRUE
    )
  )
  sibling <- ndm_run_real(
    ndm_create_real_run_config(
      project_root = project_root,
      analysis_name = "RealAnchorSibling",
      outer = 1L,
      dry_run = TRUE
    )
  )

  expect_equal(primary$training_schedule$anchor_scope, "mode_folder_real")
  expect_equal(primary$training_schedule$anchor_max_n_samples_train, 640L)
  expect_equal(primary$training_schedule$resolved_n_sgd, as.integer(round(9 * (640 / 32))))
  expect_equal(primary$training_schedule$resolved_n_sgd, sibling$training_schedule$resolved_n_sgd)
})

test_that("package-native multidisease dry runs anchor nSGD to sibling multidisease grids only", {
  project_root <- ndm_test_runner_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)

  active_path <- ndm_test_grid_path(
    project_root,
    mode = "multidisease",
    analysis_name = "RealAnchorPrimary_MULTIDISEASE"
  )
  ndm_test_write_grid(
    ndm_test_make_multidisease_run_grid_with_samples(100L),
    active_path
  )
  ndm_test_write_grid(
    ndm_test_make_multidisease_run_grid_with_samples(800L),
    ndm_test_grid_path(project_root, mode = "multidisease", analysis_name = "RealAnchorSibling_MULTIDISEASE")
  )
  ndm_test_write_grid(
    ndm_test_make_real_run_grid_with_samples(4096L),
    ndm_test_grid_path(project_root, mode = "real", analysis_name = "RealAnchorSibling")
  )

  dry_run <- ndm_run_multidisease(
    ndm_create_multidisease_run_config(
      project_root = project_root,
      analysis_name = "RealAnchorPrimary_MULTIDISEASE",
      grid_file = active_path,
      outer = 1L,
      disease_names = "hiv",
      data_format = "IHME",
      max_sgd_steps = 1L,
      dry_run = TRUE
    )
  )

  expect_equal(dry_run$training_schedule$anchor_scope, "mode_folder_multidisease")
  expect_equal(dry_run$training_schedule$anchor_max_n_samples_train, 800L)
  expect_equal(dry_run$training_schedule$unbounded_resolved_n_sgd, as.integer(round(9 * (800 / 32))))
  expect_equal(dry_run$training_schedule$resolved_n_sgd, 1L)
  expect_match(dry_run$training_schedule$policy, "\\+pilot_cap$")
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

test_that("run configs expose production checkpoint and preregistered sensitivity controls", {
  config <- ndm_create_sim_run_config(
    project_root = tempdir(),
    grid = data.frame(BaseID = 1L, ModelType = "NeuralODE"),
    n_checkpoints = 10L,
    max_sgd_steps = 100L,
    prior_sd_multiplier = 2,
    solver_profile = "tight",
    enable_kv_cache = TRUE,
    inference_mc_draws = 7L,
    observation_scale_floor = 2e-5,
    initial_observation_scale = 0.02,
    neuralode_variational = FALSE,
    neuralode_mean_loss_weight = 0.3,
    dry_run = TRUE
  )

  expect_equal(config$n_checkpoints, 10L)
  expect_equal(config$max_sgd_steps, 100L)
  expect_equal(config$prior_sd_multiplier, 2)
  expect_equal(config$solver_profile, "tight")
  expect_true(config$enable_kv_cache)
  expect_identical(config$inference_mc_draws, 7L)
  expect_equal(config$observation_scale_floor, 2e-5)
  expect_equal(config$initial_observation_scale, 0.02)
  expect_false(config$neuralode_variational)
  expect_equal(config$neuralode_mean_loss_weight, 0.3)
  expect_false(config$resave_tfrecords)
  expect_error(
    ndm_create_sim_run_config(project_root = tempdir(), n_checkpoints = 0L),
    "positive integer"
  )
  expect_error(
    ndm_create_sim_run_config(project_root = tempdir(), prior_sd_multiplier = 0),
    "finite positive"
  )
  expect_error(
    ndm_create_sim_run_config(project_root = tempdir(), solver_profile = "unknown"),
    "default, loose, tight, or alternative"
  )
  expect_error(
    ndm_create_sim_run_config(project_root = tempdir(), neuralode_mean_loss_weight = -1),
    "non-negative"
  )
  expect_error(
    ndm_create_sim_run_config(project_root = tempdir(), enable_kv_cache = NA),
    "non-missing logical"
  )
  expect_error(
    ndm_create_sim_run_config(project_root = tempdir(), neuralode_variational = NA),
    "non-missing logical"
  )
  expect_error(
    ndm_create_sim_run_config(project_root = tempdir(), inference_mc_draws = 0L),
    "positive integer"
  )
  expect_error(
    ndm_create_sim_run_config(project_root = tempdir(), observation_scale_floor = 0),
    "finite positive"
  )
  expect_error(
    ndm_create_sim_run_config(project_root = tempdir(), initial_observation_scale = Inf),
    "finite positive"
  )
  expect_error(
    ndm_create_sim_run_config(
      project_root = tempdir(),
      observation_scale_floor = 0.01,
      initial_observation_scale = 0.01
    ),
    "greater than `observation_scale_floor`"
  )
  expect_error(
    ndm_create_sim_run_config(project_root = tempdir(), respect_grid_model_type = 1L),
    "`respect_grid_model_type` must be one non-missing logical"
  )
  expect_error(
    ndm_create_sim_run_config(project_root = tempdir(), dry_run = 1L),
    "`dry_run` must be one non-missing logical"
  )

  mutated <- ndm_create_sim_run_config(project_root = tempdir(), dry_run = TRUE)
  mutated$force_to_gpu <- 1L
  expect_error(
    ndm:::.ndm_run_config_to_args(mutated),
    "`force_to_gpu` must be one non-missing logical"
  )

  runtime_env <- ndm:::.ndm_new_run_impl_env()
  expect_equal(runtime_env$analysis2_small_run_n_checkpoints(8L, 100L, default = 1L), 1L)
  expect_equal(runtime_env$analysis2_small_run_n_checkpoints(8L, 100L, default = 10L), 10L)
})

test_that("Analysis2 YAML flags reject numeric scalars instead of silently disabling", {
  api <- ndm:::.ndm_new_run_impl_env()
  numeric_manifest <- tempfile(fileext = ".yaml")
  canonical_manifest <- tempfile(fileext = ".yaml")
  on.exit(unlink(c(numeric_manifest, canonical_manifest)), add = TRUE)
  writeLines("force_to_gpu: 1", numeric_manifest)
  writeLines(c("force_to_gpu: yes", "dry_run: \"false\""), canonical_manifest)

  numeric_config <- api$analysis2_load_yaml_config(numeric_manifest)
  expect_identical(numeric_config$force_to_gpu, 1L)
  expect_error(
    api$analysis2_validate_manifest(numeric_config, "real"),
    "`force_to_gpu` must be one non-missing logical value"
  )

  canonical_config <- api$analysis2_load_yaml_config(canonical_manifest)
  expect_silent(api$analysis2_validate_manifest(canonical_config, "real"))
  spec <- utils::modifyList(api$analysis2_mode_defaults("real"), canonical_config)
  spec$project_root <- tempdir()
  normalized <- api$analysis2_normalize_run_spec(
    spec,
    mode = "real",
    paths = list(project_root = tempdir())
  )
  expect_true(normalized$force_to_gpu)
  expect_false(normalized$dry_run)

  for (field in c(
    "respect_grid_model_type",
    "run_figures",
    "force_to_gpu",
    "dry_run",
    "help"
  )) {
    invalid <- api$analysis2_mode_defaults("real")
    invalid$project_root <- tempdir()
    invalid[[field]] <- 1L
    expect_error(
      api$analysis2_normalize_run_spec(
        invalid,
        mode = "real",
        paths = list(project_root = tempdir())
      ),
      paste0("`", field, "` must be one non-missing logical value"),
      info = field
    )
  }
})

test_that("multidisease driver consumes all normalized sensitivity controls", {
  runtime_env <- ndm:::.ndm_new_run_impl_env()
  grid_globals <- list2env(
    list(
      nCheckpointsDefault = 99L,
      nCheckpoints = 99L,
      PriorSDMultiplier = 0.25,
      force2GPU = FALSE,
      GPU_MEM_FRAC = 0.9,
      nSGD_DefiningLRSeq = 999L,
      nSGD_model = 999L,
      nSGD_posttrain = 999L
    ),
    parent = emptyenv()
  )
  structured_globals <- runtime_env$analysis2_multidisease_structured_control_globals(
    spec = list(
      n_checkpoints = 7L,
      prior_sd_multiplier = 2.5,
      force_to_gpu = TRUE,
      gpu_mem_frac = 0.4
    ),
    n_samples_train = 64L,
    n_sgd = 5L
  )
  list2env(structured_globals, envir = grid_globals)

  expect_identical(grid_globals$nCheckpointsDefault, 7L)
  expect_identical(grid_globals$nCheckpoints, 5L)
  expect_equal(grid_globals$PriorSDMultiplier, 2.5)
  expect_true(grid_globals$force2GPU)
  expect_equal(grid_globals$GPU_MEM_FRAC, 0.4)
  expect_identical(grid_globals$nSGD_DefiningLRSeq, 5L)
  expect_identical(grid_globals$nSGD_model, 5L)
  expect_identical(grid_globals$nSGD_posttrain, 5L)

  solver_runtime <- list(
    diffrax = list(
      Tsit5 = function() structure("tsit5", class = "ndm_test_solver"),
      Dopri8 = function() structure("dopri8", class = "ndm_test_solver"),
      PIDController = function(rtol, atol) {
        list(rtol = rtol, atol = atol)
      }
    )
  )
  tight_solver <- runtime_env$analysis2_solver_profile(
    solver_runtime,
    "tight"
  )
  alternative_solver <- runtime_env$analysis2_solver_profile(
    solver_runtime,
    "alternative"
  )

  expect_identical(tight_solver$name, "tight")
  expect_identical(unclass(tight_solver$solver), "tsit5")
  expect_equal(tight_solver$rtol, 1e-6)
  expect_equal(tight_solver$atol, 1e-8)
  expect_equal(tight_solver$controller$rtol, tight_solver$rtol)
  expect_equal(tight_solver$controller$atol, tight_solver$atol)
  expect_identical(alternative_solver$name, "alternative")
  expect_identical(unclass(alternative_solver$solver), "dopri8")

  list2env(
    list(
      SolverProfile = tight_solver$name,
      SolverRtol = tight_solver$rtol,
      SolverAtol = tight_solver$atol,
      VI_diff_eq_solver_optim = tight_solver$solver,
      dt0_init_optim = tight_solver$dt0,
      stepsize_controller_optim = tight_solver$controller
    ),
    envir = grid_globals
  )
  expect_identical(grid_globals$SolverProfile, "tight")
  expect_equal(grid_globals$SolverRtol, 1e-6)
  expect_equal(grid_globals$SolverAtol, 1e-8)
  expect_identical(unclass(grid_globals$VI_diff_eq_solver_optim), "tsit5")
  expect_equal(grid_globals$dt0_init_optim, 1e-3)
  expect_equal(grid_globals$stepsize_controller_optim$rtol, 1e-6)
  expect_equal(grid_globals$stepsize_controller_optim$atol, 1e-8)

  driver_source <- ndm_test_runtime_source_text(
    "SetupEnv/Analysis2_legacy_multidisease_driver.R"
  )

  expect_match(
    driver_source,
    "nCheckpointsDefault <- analysis2_as_int(analysis2_multidisease_spec$n_checkpoints)",
    fixed = TRUE
  )
  expect_match(
    driver_source,
    "analysis2_multidisease_structured_control_globals(",
    fixed = TRUE
  )
  expect_match(
    driver_source,
    "analysis2_multidisease_spec$solver_profile",
    fixed = TRUE
  )
  expect_match(
    driver_source,
    "nsgd_calibration <- analysis2_resolve_nsgd_calibration(",
    fixed = TRUE
  )
})

test_that("Analysis2 normalizes corrected inference controls", {
  api <- ndm:::.ndm_new_run_impl_env()
  spec <- api$analysis2_mode_defaults("real")

  expect_false(spec$enable_kv_cache)
  expect_identical(spec$inference_mc_draws, 5L)
  expect_equal(spec$observation_scale_floor, 1e-5)
  expect_equal(spec$initial_observation_scale, 1)
  expect_true(spec$neuralode_variational)

  opts <- api$analysis2_parse_args(c(
    "--enable_kv_cache=TRUE",
    "--inference_mc_draws=7",
    "--observation_scale_floor=2e-5",
    "--initial_observation_scale=0.02",
    "--neuralode_variational=FALSE"
  ))
  overrides <- api$analysis2_cli_overrides(opts, "real")
  expect_true(overrides$enable_kv_cache)
  expect_identical(overrides$inference_mc_draws, "7")
  expect_false(overrides$neuralode_variational)

  spec <- utils::modifyList(spec, overrides)
  spec$project_root <- tempdir()
  normalized <- api$analysis2_normalize_run_spec(
    spec,
    mode = "real",
    paths = list(project_root = tempdir())
  )
  expect_true(normalized$enable_kv_cache)
  expect_identical(normalized$inference_mc_draws, 7L)
  expect_equal(normalized$observation_scale_floor, 2e-5)
  expect_equal(normalized$initial_observation_scale, 0.02)
  expect_false(normalized$neuralode_variational)

  spec$inference_mc_draws <- 0L
  expect_error(
    api$analysis2_normalize_run_spec(
      spec,
      mode = "real",
      paths = list(project_root = tempdir())
    ),
    "positive integer"
  )

  spec <- api$analysis2_mode_defaults("real")
  spec$project_root <- tempdir()
  spec$observation_scale_floor <- 0.01
  spec$initial_observation_scale <- 0.01
  expect_error(
    api$analysis2_normalize_run_spec(
      spec,
      mode = "real",
      paths = list(project_root = tempdir())
    ),
    "greater than `observation_scale_floor`"
  )
})

test_that("Analysis2 reconstructs the real training target horizon from the grid", {
  api <- ndm:::.ndm_new_run_impl_env()
  captured <- NULL
  api$analysis2_call <- function(pkg, name, ...) {
    captured <<- list(...)
    captured
  }
  row <- list(
    ContextLength = 8L,
    evaluationTime = 4L,
    initialTransform = "none",
    initialNormType = "all",
    paddingMethod = "left",
    OSSType = "OutOfTime",
    dataInputs = "all",
    BaseID = 2014L,
    training_target_horizon = 4L
  )

  spec <- api$analysis2_real_dataset_spec("fake", row)

  expect_identical(spec$training_target_horizon, 4L)

  row$training_target_horizon <- NA_integer_
  spec <- api$analysis2_real_dataset_spec("fake", row)
  expect_null(spec$training_target_horizon)
})

test_that("path helpers recognize POSIX, Windows, UNC, and relative paths", {
  expect_true(ndm:::.ndm_is_absolute_path("/tmp/grid.csv"))
  expect_true(ndm:::.ndm_is_absolute_path("C:\\tmp\\grid.csv"))
  expect_true(ndm:::.ndm_is_absolute_path("//server/share/grid.csv"))
  expect_false(ndm:::.ndm_is_absolute_path("Data/RunGrids/grid.csv"))

  expect_equal(
    ndm:::.ndm_path_join_if_relative("/project", "Data/RunGrids/grid.csv"),
    file.path("/project", "Data/RunGrids/grid.csv")
  )
  expect_equal(
    ndm:::.ndm_path_join_if_relative("/project", "C:\\tmp\\grid.csv"),
    "C:\\tmp\\grid.csv"
  )
})

ndm_test_fake_multidisease_env <- function(driver_text, project_root = tempfile("ndm-runner-project-")) {
  dir.create(project_root, recursive = TRUE, showWarnings = FALSE)
  analysis_root <- file.path(project_root, "Analysis")
  driver_dir <- file.path(analysis_root, "SetupEnv")
  dir.create(driver_dir, recursive = TRUE, showWarnings = FALSE)
  writeLines(driver_text, file.path(driver_dir, "Analysis2_legacy_multidisease_driver.R"))

  grid_file <- file.path(project_root, "grid.csv")
  utils::write.csv(data.frame(BaseID = 1L, nSamplesTrain = 32L), grid_file, row.names = FALSE)

  env <- new.env(parent = emptyenv())
  env$analysis2_log <- function(...) invisible(NULL)
  env$analysis2_build_run_spec <- function(mode, args) {
    list(
      help = FALSE,
      dry_run = FALSE,
      paths = list(project_root = project_root, analysis_root = analysis_root),
      grid_file = grid_file,
      project_root = project_root,
      analysis_name = "FakeMultidisease",
      outer = 1L
    )
  }
  env$analysis2_print_usage <- function(...) "usage"
  env$analysis2_order_grid <- function(grid, outer) grid
  env$analysis2_validate_outer_iterations <- function(...) invisible(TRUE)
  env$analysis2_dry_run_result <- function(...) list(dry_run = TRUE)
  env$analysis2_prepare_output_roots <- function(...) invisible(TRUE)
  env$analysis2_dir_create <- function(path) dir.create(path, recursive = TRUE, showWarnings = FALSE)
  env$analysis2_as_int <- as.integer
  env$analysis2_small_run_n_checkpoints <- function(...) 1L
  env$analysis2_multidisease_structured_control_globals <- function(...) list()
  env$analysis2_small_run_n_obs_inference <- function(...) 1L
  env$analysis2_resolve_nsgd_calibration <- function(...) {
    list(
      resolved_n_sgd = 1L,
      anchor_max_n_samples_train = 32L,
      anchor_n_batch = 32L,
      anchor_scope = "test",
      policy = "test"
    )
  }
  env$analysis2_log_nsgd_calibration <- function(...) invisible(NULL)
  env$analysis2_solver_profile <- function(...) list()
  env$analysis2_model_type <- function(...) "NeuralODE"
  ndm:::.ndm_override_legacy_multidisease_runner(env)
  env
}

test_that("multidisease runner restores cwd and error option", {
  old_wd <- getwd()
  old_error <- getOption("error")
  sentinel_error <- quote(stop("sentinel"))
  on.exit(setwd(old_wd), add = TRUE)
  on.exit(options(error = old_error), add = TRUE)
  options(error = sentinel_error)

  project_root <- tempfile("ndm-runner-project-")
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)
  env <- ndm_test_fake_multidisease_env("analysis2_multidisease_result <- TRUE", project_root)
  run_fun <- get("analysis2_run_real_multidisease", envir = env, inherits = FALSE)

  result <- run_fun(character())

  expect_true(isTRUE(result))
  expect_identical(getwd(), old_wd)
  expect_identical(getOption("error"), sentinel_error)
})

test_that("multidisease runner errors when legacy driver omits result", {
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)

  project_root <- tempfile("ndm-runner-project-")
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)
  env <- ndm_test_fake_multidisease_env("invisible(TRUE)", project_root)
  run_fun <- get("analysis2_run_real_multidisease", envir = env, inherits = FALSE)

  expect_error(
    run_fun(character()),
    "without assigning `analysis2_multidisease_result`"
  )
  expect_identical(getwd(), old_wd)
})
