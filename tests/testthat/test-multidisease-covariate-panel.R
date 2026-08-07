ndm_test_write_covariate_panel <- function(project_root,
                                           panel = NULL,
                                           manifest = NULL,
                                           stem = "covariates") {
  if (is.null(panel)) {
    panel <- data.frame(
      location_id = c("CTA", "CTA", "CTB", "ZZZ"),
      year = c(2000L, 2001L, 2000L, 2000L),
      food_score = c(10, 11, NA, 99),
      hiv_rate = c(1, 2, 3, 99),
      stringsAsFactors = FALSE
    )
  }
  if (is.null(manifest)) {
    manifest <- data.frame(
      feature_name = c("hiv_rate", "food_score"),
      source = c("fixture-hiv", "fixture-food"),
      unit = c("per_1000", "index"),
      stringsAsFactors = FALSE
    )
  }
  panel_file <- file.path(project_root, paste0(stem, "-panel.csv"))
  manifest_file <- file.path(project_root, paste0(stem, "-manifest.csv"))
  utils::write.csv(panel, panel_file, row.names = FALSE, na = "")
  utils::write.csv(manifest, manifest_file, row.names = FALSE, na = "")
  list(panel = panel_file, manifest = manifest_file)
}

test_that("WHO covariate panels left join in manifest order and bind provenance", {
  project_root <- ndm_test_multidisease_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)
  ndm_test_write_who_fixture(project_root)
  files <- ndm_test_write_covariate_panel(project_root)

  baseline <- ndm:::.ndm_prepare_multidisease_bundle(
    project_root = project_root,
    data_format = "WHO",
    disease_names = "TB",
    outcome_metric = "CountValue",
    data_subset = "all"
  )
  bundle <- ndm:::.ndm_prepare_multidisease_bundle(
    project_root = project_root,
    data_format = "WHO",
    disease_names = "TB",
    outcome_metric = "CountValue",
    data_subset = "all",
    covariate_panel_file = files$panel,
    covariate_manifest_file = files$manifest
  )

  expect_identical(nrow(bundle$truth_df_red), nrow(baseline$truth_df_red))
  expect_identical(
    bundle$truth_df_red[c("location_id", "year", "CountValue", "Covariate1")],
    baseline$truth_df_red[c("location_id", "year", "CountValue", "Covariate1")]
  )
  expect_identical(
    bundle$dataInputs_colnames_past,
    c("CountValue", "Covariate1", "hiv_rate", "food_score")
  )
  expect_identical(bundle$dataInputs_colnames_future, character())
  expect_identical(bundle$covariate_feature_names, c("hiv_rate", "food_score"))
  expect_equal(bundle$truth_df_red$hiv_rate, c(1, 2, 3))
  expect_equal(bundle$truth_df_red$food_score, c(10, 11, NA))
  expect_identical(
    bundle$truth_df_red[c("hiv_rate", "food_score")],
    bundle$input_df_red[c("hiv_rate", "food_score")]
  )
  expect_match(bundle$covariate_panel_file_sha256, "^[0-9a-f]{64}$")
  expect_match(bundle$covariate_manifest_file_sha256, "^[0-9a-f]{64}$")
  expect_match(bundle$covariate_schema_sha256, "^[0-9a-f]{64}$")

  row <- ndm_test_make_multidisease_run_grid()
  row$dataInputs <- "Covariate1__hiv_rate__food_score"
  row$inferenceSupportInputs <- "Covariate1"
  captured <- NULL
  dataset_call <- function(name, ...) {
    expect_identical(name, "ndm_datasets_dataset_spec")
    captured <<- list(...)
    captured
  }
  spec <- ndm:::.ndm_multidisease_dataset_spec(
    row_values = row,
    bundle = bundle,
    dataset_call = dataset_call
  )
  expect_identical(spec$inference_support_inputs, "Covariate1")
  expect_identical(
    spec$data_inputs,
    c("Covariate1", "hiv_rate", "food_score")
  )
  expect_identical(
    spec$covariate_panel_file_sha256,
    bundle$covariate_panel_file_sha256
  )
  expect_identical(
    spec$covariate_manifest_file_sha256,
    bundle$covariate_manifest_file_sha256
  )
  expect_identical(spec$covariate_schema_sha256, bundle$covariate_schema_sha256)
  expect_identical(spec$covariate_manifest$source, c("fixture-hiv", "fixture-food"))
})

test_that("covariate panel values and manifest order change the pinned contract", {
  project_root <- ndm_test_multidisease_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)
  ndm_test_write_who_fixture(project_root)
  files_a <- ndm_test_write_covariate_panel(project_root, stem = "a")
  panel_b <- data.frame(
    location_id = c("CTA", "CTA", "CTB"),
    year = c(2000L, 2001L, 2000L),
    food_score = c(10, 11, NA),
    hiv_rate = c(101, 2, 3),
    stringsAsFactors = FALSE
  )
  manifest_b <- data.frame(
    feature_name = c("food_score", "hiv_rate"),
    source = c("fixture-food", "fixture-hiv"),
    stringsAsFactors = FALSE
  )
  files_b <- ndm_test_write_covariate_panel(
    project_root,
    panel = panel_b,
    manifest = manifest_b,
    stem = "b"
  )
  prepare <- function(files) {
    ndm:::.ndm_prepare_multidisease_bundle(
      project_root = project_root,
      data_format = "WHO",
      disease_names = "TB",
      outcome_metric = "CountValue",
      data_subset = "all",
      covariate_panel_file = files$panel,
      covariate_manifest_file = files$manifest
    )
  }
  bundle_a <- prepare(files_a)
  bundle_b <- prepare(files_b)

  expect_false(identical(
    bundle_a$covariate_panel_file_sha256,
    bundle_b$covariate_panel_file_sha256
  ))
  expect_false(identical(
    bundle_a$covariate_manifest_file_sha256,
    bundle_b$covariate_manifest_file_sha256
  ))
  expect_false(identical(
    bundle_a$covariate_schema_sha256,
    bundle_b$covariate_schema_sha256
  ))
  expect_identical(
    tail(bundle_b$dataInputs_colnames_past, 2L),
    c("food_score", "hiv_rate")
  )
  source_hash <- function(bundle) {
    ndm:::.ndm_multidisease_table_bundle(
      bundle,
      data_inputs = paste(bundle$covariate_feature_names, collapse = "__")
    )$source_sha256
  }
  expect_false(identical(source_hash(bundle_a), source_hash(bundle_b)))
})

test_that("covariate panel validation rejects ambiguous or unsafe schemas", {
  project_root <- ndm_test_multidisease_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)
  ndm_test_write_who_fixture(project_root)
  valid <- ndm_test_write_covariate_panel(project_root, stem = "valid")
  prepare <- function(files, data_format = "WHO") {
    ndm:::.ndm_prepare_multidisease_bundle(
      project_root = project_root,
      data_format = data_format,
      disease_names = if (identical(data_format, "WHO")) "TB" else "hiv",
      outcome_metric = "CountValue",
      data_subset = "all",
      covariate_panel_file = files$panel,
      covariate_manifest_file = files$manifest
    )
  }

  expect_error(
    ndm:::.ndm_prepare_multidisease_bundle(
      project_root = project_root,
      data_format = "WHO",
      disease_names = "TB",
      outcome_metric = "CountValue",
      data_subset = "all",
      covariate_panel_file = valid$panel
    ),
    "must be supplied together"
  )

  duplicate_panel <- data.frame(
    location_id = c("CTA", "CTA"),
    year = c(2000L, 2000L),
    x = c(1, 2)
  )
  duplicate <- ndm_test_write_covariate_panel(
    project_root,
    panel = duplicate_panel,
    manifest = data.frame(feature_name = "x"),
    stem = "duplicate"
  )
  expect_error(prepare(duplicate), "unique.*location_id, year")

  nonnumeric <- ndm_test_write_covariate_panel(
    project_root,
    panel = data.frame(location_id = "CTA", year = 2000L, x = "bad"),
    manifest = data.frame(feature_name = "x"),
    stem = "nonnumeric"
  )
  expect_error(prepare(nonnumeric), "must be numeric")

  nonfinite <- ndm_test_write_covariate_panel(
    project_root,
    panel = data.frame(location_id = "CTA", year = 2000L, x = Inf),
    manifest = data.frame(feature_name = "x"),
    stem = "nonfinite"
  )
  expect_error(prepare(nonfinite), "not NaN or infinite")

  unsafe <- ndm_test_write_covariate_panel(
    project_root,
    panel = data.frame(location_id = "CTA", year = 2000L, check.names = FALSE,
                       "bad__name" = 1),
    manifest = data.frame(feature_name = "bad__name"),
    stem = "unsafe"
  )
  expect_error(prepare(unsafe), "must not contain `__`")

  undeclared <- ndm_test_write_covariate_panel(
    project_root,
    panel = data.frame(location_id = "CTA", year = 2000L, x = 1, y = 2),
    manifest = data.frame(feature_name = "x"),
    stem = "undeclared"
  )
  expect_error(prepare(undeclared), "undeclared panel feature")

  collision <- ndm_test_write_covariate_panel(
    project_root,
    panel = data.frame(location_id = "CTA", year = 2000L, Covariate1 = 1),
    manifest = data.frame(feature_name = "Covariate1"),
    stem = "collision"
  )
  expect_error(prepare(collision), "collide with existing bundle columns")

  no_overlap <- ndm_test_write_covariate_panel(
    project_root,
    panel = data.frame(location_id = "ZZZ", year = 1900L, x = 1),
    manifest = data.frame(feature_name = "x"),
    stem = "no-overlap"
  )
  expect_error(prepare(no_overlap), "no.*keys matching the WHO bundle")

  empty_feature <- ndm_test_write_covariate_panel(
    project_root,
    panel = data.frame(
      location_id = c("CTA", "CTB"),
      year = c(2000L, 2000L),
      x = c(1, 2),
      empty = c(NA_real_, NA_real_)
    ),
    manifest = data.frame(feature_name = c("x", "empty")),
    stem = "empty-feature"
  )
  expect_error(prepare(empty_feature), "at least one finite value.*empty")

  ndm_test_write_ihme_fixture(project_root)
  expect_error(prepare(valid, data_format = "IHME"), "supported only for WHO")
})

test_that("inference support inputs are dataset-defining and selected-input bounded", {
  bundle <- list(
    resolved_diseases = "TB",
    dataInputs_colnames_past = c("CountValue", "Covariate1", "hiv_rate"),
    outcome_metric = "CountValue",
    true_value_names = "CountValue",
    truth_df_red = data.frame(
      location_id = "AAA",
      time_id = 0L,
      CountValue = 1,
      Covariate1 = 1,
      hiv_rate = 2
    )
  )
  row <- ndm_test_make_multidisease_run_grid()
  row$dataInputs <- "Covariate1__hiv_rate"
  row$inferenceSupportInputs <- "Covariate1"
  dataset_call <- function(name, ...) list(...)
  spec <- ndm:::.ndm_multidisease_dataset_spec(
    row,
    bundle,
    dataset_call = dataset_call
  )
  expect_identical(spec$inference_support_inputs, "Covariate1")

  row$inferenceSupportInputs <- "all"
  spec_all <- ndm:::.ndm_multidisease_dataset_spec(
    row,
    bundle,
    dataset_call = dataset_call
  )
  expect_identical(
    spec_all$inference_support_inputs,
    c("Covariate1", "hiv_rate")
  )

  row$inferenceSupportInputs <- "CountValue"
  expect_error(
    ndm:::.ndm_multidisease_dataset_spec(
      row,
      bundle,
      dataset_call = dataset_call
    ),
    "subset of `dataInputs`"
  )

  grid <- ndm_test_make_multidisease_run_grid()[rep(1L, 2L), , drop = FALSE]
  grid$inferenceSupportInputs <- c("Covariate1", "all")
  expect_error(
    ndm:::.ndm_multidisease_plan(
      grid,
      tfrecord_dir = file.path(tempdir(), "ndm-support-plan")
    ),
    "dataset-defining field.*inferenceSupportInputs"
  )
})

test_that("trajectory support preserves locations with masked external inputs", {
  time_ids <- 0:11
  truth <- do.call(rbind, lapply(seq_along(c("AAA", "BBB")), function(i) {
    location <- c("AAA", "BBB")[[i]]
    values <- (time_ids + i) / 100
    data.frame(
      location_id = location,
      location_id_numeric = i - 1L,
      location_name = location,
      time_id = time_ids,
      targetTime_id = time_ids,
      CountValue = values,
      Covariate1 = values,
      external = if (location == "BBB") NA_real_ else time_ids + 1,
      stringsAsFactors = FALSE
    )
  }))
  bundle <- ndm:::.ndm_multidisease_make_bundle(
    data_format = "WHO",
    resolved_diseases = "TB",
    truth_df_red = truth,
    input_df_red = truth,
    data_inputs_past = c("CountValue", "Covariate1", "external"),
    true_value_names = "CountValue",
    outcome_metric = "CountValue"
  )
  row <- ndm_test_make_multidisease_run_grid()
  row$BaseID <- 1L
  row$ContextLength <- 4L
  row$evaluationOriginTimeID <- 6L
  row$evaluationHorizon <- 4L
  row$inferenceSampling <- "complete_locations"
  row$nObsInference <- 2L
  row$dataInputs <- "Covariate1__external"
  row$inferenceSupportInputs <- "Covariate1"

  supported <- ndm:::.ndm_multidisease_artifact_contract(
    row_values = row,
    bundle = bundle,
    lookahead = 4L,
    min_anchoring_time = 0L
  )
  prepared_supported <- ndmdatasets::ndm_real_prepare_tables(
    supported$table_bundle,
    supported$dataset_spec
  )
  expect_identical(prepared_supported$inference_support_inputs, "Covariate1")
  expect_true(all(prepared_supported$inference_support$eligible))

  row$inferenceSupportInputs <- NULL
  row$nObsInference <- 1L
  default_support <- ndm:::.ndm_multidisease_artifact_contract(
    row_values = row,
    bundle = bundle,
    lookahead = 4L,
    min_anchoring_time = 0L
  )
  prepared_default <- ndmdatasets::ndm_real_prepare_tables(
    default_support$table_bundle,
    default_support$dataset_spec
  )
  expect_identical(
    prepared_default$inference_support_inputs,
    c("Covariate1", "external")
  )
  expect_identical(prepared_default$inference_support$eligible, c(TRUE, FALSE))
})

test_that("public multidisease config and runtime specs carry panel paths", {
  project_root <- ndm_test_multidisease_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)
  files <- ndm_test_write_covariate_panel(project_root)
  grid <- ndm_test_make_multidisease_run_grid()
  config <- ndm_create_multidisease_run_config(
    project_root = project_root,
    grid = grid,
    data_format = "WHO",
    covariate_panel_file = basename(files$panel),
    covariate_manifest_file = basename(files$manifest),
    dry_run = TRUE
  )
  expect_identical(
    config$covariate_panel_file,
    normalizePath(files$panel, winslash = "/", mustWork = TRUE)
  )
  expect_identical(
    config$covariate_manifest_file,
    normalizePath(files$manifest, winslash = "/", mustWork = TRUE)
  )
  args <- ndm:::.ndm_run_config_to_args(config)
  expect_true(paste0("--covariate_panel_file=", config$covariate_panel_file) %in% args)
  expect_true(paste0("--covariate_manifest_file=", config$covariate_manifest_file) %in% args)

  default_config <- ndm_create_multidisease_run_config(
    project_root = project_root,
    grid = grid,
    dry_run = TRUE
  )
  default_args <- ndm:::.ndm_run_config_to_args(default_config)
  expect_null(default_config$covariate_panel_file)
  expect_null(default_config$covariate_manifest_file)
  expect_false(any(grepl("--covariate_", default_args, fixed = TRUE)))

  expect_error(
    ndm_create_multidisease_run_config(
      project_root = project_root,
      grid = grid,
      covariate_panel_file = files$panel,
      dry_run = TRUE
    ),
    "must be supplied together"
  )

  api <- ndm:::.ndm_new_run_impl_env()
  defaults <- api$analysis2_mode_defaults("multidisease")
  defaults$project_root <- project_root
  defaults$data_format <- "WHO"
  defaults$covariate_panel_file <- basename(files$panel)
  defaults$covariate_manifest_file <- basename(files$manifest)
  normalized <- api$analysis2_normalize_run_spec(
    defaults,
    mode = "multidisease",
    paths = list(project_root = project_root)
  )
  expect_identical(normalized$covariate_panel_file, config$covariate_panel_file)
  expect_identical(
    normalized$covariate_manifest_file,
    config$covariate_manifest_file
  )
  expect_match(
    api$analysis2_usage(
      "multidisease",
      paths = list(analysis_root = project_root)
    ),
    "--covariate_panel_file=PATH"
  )
})

test_that("multidisease bootstrap validates paired panel arguments in dry runs", {
  project_root <- ndm_test_multidisease_project_root()
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)
  files <- ndm_test_write_covariate_panel(project_root)
  grid <- ndm_test_make_multidisease_run_grid()

  plan <- ndm_bootstrap_multidisease_tfrecords(
    project_root = project_root,
    grid = grid,
    disease_names = "TB",
    data_format = "WHO",
    covariate_panel_file = files$panel,
    covariate_manifest_file = files$manifest,
    dry_run = TRUE
  )
  expect_identical(plan$status, "planned")
  expect_error(
    ndm_bootstrap_multidisease_tfrecords(
      project_root = project_root,
      grid = grid,
      disease_names = "TB",
      data_format = "WHO",
      covariate_panel_file = files$panel,
      dry_run = TRUE
    ),
    "must be supplied together"
  )
})
