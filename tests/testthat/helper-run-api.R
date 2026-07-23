ndm_require_runner_test_stack <- function(context) {
  skip_on_cran()
  ndm_require_backend_test_stack(
    context,
    packages = c(
      "reticulate",
      "fastmatch",
      "rrapply",
      "yaml",
      "zip",
      "zoo",
      "ndmdatasets"
    )
  )
}

ndm_test_runner_project_root <- function(prefix = "ndm-runner-") {
  root <- tempfile(prefix)
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  root
}

ndm_test_tfrecord_producer <- function() {
  list(contract = "ndm-test-producer-v1")
}

ndm_test_write_grid <- function(grid, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(grid, path, row.names = FALSE)
  invisible(path)
}

ndm_test_grid_path <- function(project_root,
                               mode = c("sim", "real", "multidisease"),
                               analysis_name) {
  mode <- match.arg(mode)
  file.path(
    project_root,
    "Data",
    "RunGrids",
    switch(
      mode,
      sim = "SimGrids",
      real = "RealGrids",
      multidisease = "RealGrids"
    ),
    sprintf(
      "%s_%s.csv",
      switch(
        mode,
        sim = "SimGrid",
        real = "RealGrid",
        multidisease = "RealGrid"
      ),
      analysis_name
    )
  )
}

ndm_test_assert_canonical_tfrecords <- function(output_dir, base_id = 1L) {
  expected <- c(
    file.path(output_dir, sprintf("train_%s.tfrecord", base_id)),
    file.path(output_dir, sprintf("inference_%s.tfrecord", base_id))
  )
  expected <- c(expected, paste0(expected, ".manifest.rds"))
  missing <- expected[!file.exists(expected)]
  expect_true(
    length(missing) == 0L,
    info = paste("Missing canonical TFRecord artifacts:", paste(missing, collapse = ", "))
  )
}

ndm_test_read_canonical_manifest <- function(output_dir,
                                             base_id = 1L,
                                             split = c("train", "inference")) {
  split <- match.arg(split)
  readRDS(file.path(output_dir, sprintf("%s_%s.tfrecord.manifest.rds", split, base_id)))
}

ndm_test_copy_raw_covid_fixture <- function(project_root) {
  source_dir <- system.file("extdata", "examples", "raw_covid", package = "ndmdatasets")
  if (!nzchar(source_dir) || !dir.exists(source_dir)) {
    stop("Unable to locate ndmdatasets raw_covid fixture.", call. = FALSE)
  }

  target_dir <- file.path(project_root, "Data", "MainData")
  dir.create(target_dir, recursive = TRUE, showWarnings = FALSE)

  source_entries <- list.files(source_dir, full.names = TRUE, all.files = TRUE, no.. = TRUE)
  copied <- file.copy(source_entries, target_dir, recursive = TRUE)
  if (!all(copied)) {
    stop("Failed to copy raw_covid fixture into test project root.", call. = FALSE)
  }

  target_dir
}

ndm_test_model_spec_path <- function(name = "bayes_ode_SEIRS_DynamicBeta_FixedGlobal") {
  root <- ndm:::.ndm_package_root()
  candidates <- c(
    file.path(root, "extdata", "model_specs", sprintf("%s.tex", name)),
    file.path(root, "inst", "extdata", "model_specs", sprintf("%s.tex", name))
  )
  existing <- candidates[file.exists(candidates)]
  if (length(existing) == 0L) {
    stop("Unable to resolve test model spec path.", call. = FALSE)
  }
  normalizePath(existing[[1]], winslash = "/", mustWork = TRUE)
}

ndm_test_make_sim_run_grid <- function() {
  dataset_spec <- ndmdatasets::ndm_datasets_dataset_spec(
    kind = "sim",
    context_length = 8L,
    gamma = 0.25,
    sigma = 0.2,
    xi = 0.02,
    i0_a = 5,
    i0_b = 0.2,
    r0_a = 2,
    r0_b = 0.2,
    betat_init = 0.35,
    invbetat_sd = 0.02,
    c_endogeneous = 0.1,
    policy_responsiveness = 1,
    policy_effectiveness = 0.1,
    policy_decay = 0.05,
    measurement_noise = 0.05
  )
  training_spec <- ndmdatasets::ndm_datasets_training_spec(
    model_type = "DecoderOnly",
    model_depth = 1L,
    model_dims = 32L,
    n_samples_train = 4L,
    float_type = "32"
  )
  grid <- as.data.frame(
    ndmdatasets::ndm_sim_build_grid(
      dataset_spec = dataset_spec,
      training_spec = training_spec
    ),
    stringsAsFactors = FALSE
  )

  names(grid)[match("context_length", names(grid))] <- "ContextLength"
  names(grid)[match("model_type", names(grid))] <- "ModelType"
  names(grid)[match("model_depth", names(grid))] <- "ModelDepth"
  names(grid)[match("model_dims", names(grid))] <- "ModelDims"
  names(grid)[match("n_samples_train", names(grid))] <- "nSamplesTrain"
  names(grid)[match("float_type", names(grid))] <- "floatType"
  grid$ModelType <- as.character(grid$ModelType)
  grid$paddingMethod <- "left"
  grid$lookahead <- 4L
  grid$n_time_steps <- as.integer((grid$ContextLength + grid$lookahead) * 2L)
  grid$n_inference_batches <- 1L
  grid$scaling_batches <- 4L
  grid$model_spec_name <- "bayes_ode_SEIRS_DynamicBeta_FixedGlobal"
  grid$model_tex_loc <- ndm_test_model_spec_path()
  grid$ResaveThisTFRecord <- 1L
  grid
}

ndm_test_make_sim_run_grid_with_samples <- function(n_samples_train) {
  grid <- ndm_test_make_sim_run_grid()
  grid$nSamplesTrain <- as.integer(n_samples_train)
  grid$ResaveThisTFRecord <- 1L
  grid
}

ndm_test_make_sim_duplicate_base_grid <- function(n_samples_train = c(4L, 8L),
                                                  resave_flags = c(0L, 1L)) {
  stopifnot(length(n_samples_train) == length(resave_flags))

  grid <- ndm_test_make_sim_run_grid()[rep(1L, length(n_samples_train)), , drop = FALSE]
  grid$BaseID <- 1L
  grid$nSamplesTrain <- as.integer(n_samples_train)
  grid$ResaveThisTFRecord <- as.integer(resave_flags)
  rownames(grid) <- NULL
  grid
}

ndm_test_make_sim_bootstrap_grid <- function(n_base_ids = 10L,
                                             rows_per_base_id = 4L,
                                             canonical_n_samples_train = 8L,
                                             other_n_samples_train = 4L) {
  stopifnot(n_base_ids >= 1L)
  stopifnot(rows_per_base_id >= 1L)

  template <- ndm_test_make_sim_run_grid()[1L, , drop = FALSE]
  rows <- vector("list", length = n_base_ids * rows_per_base_id)
  row_idx <- 1L

  for (pass_idx in seq_len(rows_per_base_id)) {
    for (base_id in seq_len(n_base_ids)) {
      row <- template
      row$BaseID <- as.integer(base_id)
      row$gamma <- as.numeric(template$gamma) + (base_id / 1000)
      row$nSamplesTrain <- if (pass_idx == 1L) {
        as.integer(canonical_n_samples_train)
      } else {
        as.integer(other_n_samples_train)
      }
      row$ResaveThisTFRecord <- if (pass_idx == 1L) 1L else 0L
      rows[[row_idx]] <- row
      row_idx <- row_idx + 1L
    }
  }

  grid <- do.call(rbind, rows)
  rownames(grid) <- NULL
  grid
}

ndm_test_make_real_run_grid <- function() {
  data.frame(
    BaseID = 1L,
    ContextLength = 4L,
    evaluationTime = 2L,
    initialTransform = "none",
    initialNormType = "at_t",
    paddingMethod = "left",
    OSSType = "OutOfTime",
    dataInputs = "all",
    ModelType = "NeuralODE",
    ModelDepth = 1L,
    ModelDims = 32L,
    nSamplesTrain = 4L,
    nObsInference = 32L,
    floatType = "32",
    ResaveThisTFRecord = 1L,
    model_spec_name = "bayes_ode_SEIRS_DynamicBeta_FixedGlobal",
    model_tex_loc = ndm_test_model_spec_path(),
    stringsAsFactors = FALSE
  )
}

ndm_test_make_real_run_grid_with_samples <- function(n_samples_train) {
  grid <- ndm_test_make_real_run_grid()
  grid$nSamplesTrain <- as.integer(n_samples_train)
  grid$ResaveThisTFRecord <- 1L
  grid
}

ndm_test_make_real_duplicate_base_grid <- function(n_samples_train = c(4L, 8L),
                                                   resave_flags = c(0L, 1L)) {
  stopifnot(length(n_samples_train) == length(resave_flags))

  grid <- ndm_test_make_real_run_grid()[rep(1L, length(n_samples_train)), , drop = FALSE]
  grid$BaseID <- 1L
  grid$nSamplesTrain <- as.integer(n_samples_train)
  grid$ResaveThisTFRecord <- as.integer(resave_flags)
  rownames(grid) <- NULL
  grid
}

ndm_test_make_multidisease_run_grid <- function() {
  data.frame(
    BaseID = 1L,
    ContextLength = 8L,
    evaluationTime = 1L,
    initialTransform = "none",
    initialNormType = "all",
    paddingMethod = "left",
    OSSType = "OutOfTime",
    dataInputs = "all",
    ModelType = "NeuralODE",
    ModelDepth = 1L,
    ModelDims = 32L,
    nSamplesTrain = 2L,
    nObsInference = 32L,
    floatType = "32",
    ResaveThisTFRecord = 1L,
    model_spec_name = "bayes_ode_SEIRS_DynamicBeta_FixedGlobal",
    model_tex_loc = ndm_test_model_spec_path(),
    stringsAsFactors = FALSE
  )
}

ndm_test_make_multidisease_run_grid_with_samples <- function(n_samples_train) {
  grid <- ndm_test_make_multidisease_run_grid()
  grid$nSamplesTrain <- as.integer(n_samples_train)
  grid$ResaveThisTFRecord <- 1L
  grid
}

ndm_test_multidisease_project_root <- function() {
  ndm_test_runner_project_root("ndm-multidisease-")
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
    lapply(2000:2024, function(year_) {
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
  ndm_require_runner_test_stack("multidisease tests")
}
