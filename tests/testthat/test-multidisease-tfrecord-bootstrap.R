ndm_test_multidisease_bootstrap_grid <- function() {
  origins <- c(2007L, 2009L, 2013L, 2014L, 2015L, 2017L, 2019L)
  rows <- expand.grid(
    BaseID = origins,
    trainingSeed = c(101L, 202L, 303L),
    model_arm = seq_len(13L),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  rows <- rows[order(match(rows$BaseID, origins)), , drop = FALSE]
  rows$ContextLength <- 8L
  rows$evaluationTime <- match(rows$BaseID, origins)
  rows$evaluationOriginTimeID <- rows$BaseID - 2000L
  rows$evaluationHorizon <- 4L
  rows$inferenceSampling <- "complete_locations"
  rows$dataSeed <- 20260709L
  rows$initialTransform <- "none"
  rows$initialNormType <- "all"
  rows$paddingMethod <- "left"
  rows$OSSType <- "OutOfTime"
  rows$dataInputs <- "all"
  rows$ModelType <- ifelse(rows$model_arm == 1L, "DecoderOnly", "NeuralODE")
  rows$nSamplesTrain <- 20000L
  rows$nObsInference <- 2L
  rows$ResaveThisTFRecord <- 1L
  rows
}

test_that("multidisease dry run plans the 273-fit battery by seven BaseIDs", {
  project_root <- tempfile("ndm-multidisease-plan-")
  dir.create(project_root)
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)
  grid <- ndm_test_multidisease_bootstrap_grid()

  plan <- ndm_bootstrap_multidisease_tfrecords(
    project_root = project_root,
    grid = grid,
    disease_names = "TB",
    dry_run = TRUE
  )

  expect_equal(nrow(grid), 273L)
  expect_equal(plan$BaseID, c(2007L, 2009L, 2013L, 2014L, 2015L, 2017L, 2019L))
  expect_equal(plan$canonical_row, seq.int(1L, 235L, by = 39L))
  expect_true(all(plan$artifact_n_samples_train == 20000L))
  expect_true(all(plan$status == "planned"))
  expect_false(dir.exists(file.path(project_root, "Data", "RunTFRecords")))

  grid$dataSeed[[2L]] <- 7L
  expect_error(
    ndm_bootstrap_multidisease_tfrecords(
      project_root = project_root,
      grid = grid,
      disease_names = "TB",
      dry_run = TRUE
    ),
    "dataset-defining field.*dataSeed"
  )
})

test_that("multidisease row mapping separates tensor and scoring horizons", {
  bundle <- list(
    resolved_diseases = "TB",
    dataInputs_colnames_past = c("CountValue", "Covariate1"),
    outcome_metric = "CountValue",
    true_value_names = "CountValue",
    truth_df_red = data.frame(
      location_id = "AAA",
      time_id = 0L,
      CountValue = 1,
      Covariate1 = 1
    )
  )
  row <- ndm_test_multidisease_bootstrap_grid()[1L, , drop = FALSE]
  row$training_target_horizon <- 4L
  captured <- NULL
  dataset_call <- function(name, ...) {
    expect_identical(name, "ndm_datasets_dataset_spec")
    captured <<- list(...)
    captured
  }

  spec <- ndm:::.ndm_multidisease_dataset_spec(
    row_values = row,
    bundle = bundle,
    data_subset = "all",
    lookahead = 12L,
    min_anchoring_time = 4L,
    dataset_call = dataset_call
  )

  expect_identical(spec$lookahead, 12L)
  expect_identical(spec$training_target_horizon, 4L)
  expect_identical(spec$evaluation_horizon, 4L)
  expect_identical(spec$evaluation_sequence, 1L:4L)
  expect_identical(spec$evaluation_origin_time_id, 7L)
  expect_identical(spec$inference_sampling, "complete_locations")
  expect_identical(spec$n_inference_samples, 2L)
  expect_identical(spec$data_inputs, "Covariate1")
  expect_identical(ndm:::.ndm_multidisease_data_seed(row), 20260709L)
  expect_false("trainingSeed" %in% names(spec))

  row$evaluationHorizon <- 3L
  spec <- ndm:::.ndm_multidisease_dataset_spec(
    row_values = row,
    bundle = bundle,
    dataset_call = dataset_call
  )
  expect_identical(spec$evaluation_horizon, 3L)
  expect_identical(spec$evaluation_sequence, 1L:3L)

  row$evaluationHorizon <- 0L
  expect_error(
    ndm:::.ndm_multidisease_dataset_spec(
      row_values = row,
      bundle = bundle,
      dataset_call = dataset_call
    ),
    "evaluationHorizon.*positive"
  )
})

test_that("canonical multidisease preparation binds the explicit rolling origin", {
  time_ids <- 0:29
  truth_df <- do.call(
    rbind,
    lapply(c("AAA", "BBB"), function(location) {
      data.frame(
        location_id = location,
        location_id_numeric = match(location, c("AAA", "BBB")) - 1L,
        location_name = location,
        time_id = time_ids,
        targetTime_id = time_ids,
        CountValue = (time_ids + if (location == "AAA") 1 else 2) / 100,
        Covariate1 = (time_ids + if (location == "AAA") 2 else 4) / 100,
        stringsAsFactors = FALSE
      )
    })
  )
  bundle <- ndm:::.ndm_multidisease_make_bundle(
    data_format = "IHME",
    resolved_diseases = "Disease A",
    truth_df_red = truth_df,
    input_df_red = truth_df,
    data_inputs_past = c("CountValue", "Covariate1"),
    true_value_names = "CountValue",
    outcome_metric = "CountValue"
  )
  row <- ndm_test_multidisease_bootstrap_grid()[1L, , drop = FALSE]
  row$evaluationTime <- 1L

  prepare_at <- function(origin) {
    row$evaluationOriginTimeID <- as.integer(origin)
    contract <- ndm:::.ndm_multidisease_artifact_contract(
      row_values = row,
      bundle = bundle
    )
    prepared <- ndmdatasets::ndm_real_prepare_tables(
      contract$table_bundle,
      contract$dataset_spec
    )
    list(contract = contract, prepared = prepared)
  }
  origin_10 <- prepare_at(10L)
  origin_20 <- prepare_at(20L)

  expect_identical(origin_10$prepared$times_in, 0:9)
  expect_identical(origin_10$prepared$times_out, 10:29)
  expect_identical(origin_20$prepared$times_in, 0:19)
  expect_identical(origin_20$prepared$times_out, 20:29)
  expect_true(all(origin_10$prepared$train_df$time_id < 10L))
  expect_true(all(origin_10$prepared$out_df$time_id >= 10L))
  expect_true(all(origin_20$prepared$train_df$time_id < 20L))
  expect_true(all(origin_20$prepared$out_df$time_id >= 20L))
  expect_identical(
    origin_10$contract$dataset_spec$ndm_requested_evaluation_time,
    1L
  )
  expect_identical(
    origin_20$contract$dataset_spec$ndm_requested_evaluation_time,
    1L
  )
  expect_false(identical(
    origin_10$contract$dataset_spec$evaluation_time,
    origin_20$contract$dataset_spec$evaluation_time
  ))
  expect_identical(
    origin_10$contract$dataset_spec$ndm_origin_compatibility,
    "explicit-origin-v1"
  )
})

test_that("multidisease Yeo-Johnson fitting excludes producer holdout rows", {
  time_ids <- 0:19
  locations <- c("AAA", "BBB", "CCC", "DDD")
  truth_df <- do.call(
    rbind,
    lapply(seq_along(locations), function(location_index) {
      location <- locations[[location_index]]
      values <- exp((time_ids + 1) / 7) + location_index / 10
      data.frame(
        location_id = location,
        location_id_numeric = location_index - 1L,
        location_name = location,
        time_id = time_ids,
        targetTime_id = time_ids,
        CountValue = values,
        Covariate1 = values * 1.5,
        stringsAsFactors = FALSE
      )
    })
  )
  bundle <- ndm:::.ndm_multidisease_make_bundle(
    data_format = "IHME",
    resolved_diseases = "Disease A",
    truth_df_red = truth_df,
    input_df_red = truth_df,
    data_inputs_past = c("CountValue", "Covariate1"),
    true_value_names = "CountValue",
    outcome_metric = "CountValue"
  )

  for (split_type in c("OutOfTime", "OutOfPlace", "OutOfPlacetime")) {
    row <- ndm_test_multidisease_bootstrap_grid()[1L, , drop = FALSE]
    row$BaseID <- 1L
    row$ContextLength <- 2L
    row$evaluationTime <- 1L
    row$evaluationHorizon <- 1L
    row$nObsInference <- 2L
    row$OSSType <- split_type
    row$initialTransform <- "none"
    if (identical(split_type, "OutOfPlace")) {
      row$evaluationOriginTimeID <- NULL
    } else {
      row$evaluationOriginTimeID <- 10L
    }

    raw_contract <- ndm:::.ndm_multidisease_artifact_contract(
      row_values = row,
      bundle = bundle,
      lookahead = 2L,
      min_anchoring_time = 0L
    )
    raw_prepared <- ndmdatasets::ndm_real_prepare_tables(
      raw_contract$table_bundle,
      raw_contract$dataset_spec
    )
    training_keys <- paste(
      raw_prepared$train_df$location_id,
      raw_prepared$train_df$time_id,
      sep = "::"
    )
    source_keys <- paste(
      bundle$truth_df_red$location_id,
      bundle$truth_df_red$time_id,
      sep = "::"
    )
    training_source_rows <- source_keys %in% training_keys
    holdout_rows <- if (identical(split_type, "OutOfPlace")) {
      !training_source_rows
    } else {
      bundle$truth_df_red$time_id >= 10L
    }
    expect_true(any(holdout_rows), info = split_type)
    expect_false(any(training_source_rows & holdout_rows), info = split_type)

    perturbed_bundle <- bundle
    perturbed_bundle$truth_df_red$Covariate1[holdout_rows] <-
      perturbed_bundle$truth_df_red$Covariate1[holdout_rows] + 1e5
    perturbed_bundle$input_df_red <- perturbed_bundle$truth_df_red
    row$initialTransform <- "yeoJohnson"

    prepare <- function(candidate_bundle) {
      contract <- ndm:::.ndm_multidisease_artifact_contract(
        row_values = row,
        bundle = candidate_bundle,
        lookahead = 2L,
        min_anchoring_time = 0L
      )
      prepared <- ndmdatasets::ndm_real_prepare_tables(
        contract$table_bundle,
        contract$dataset_spec
      )
      ordering <- order(prepared$train_df$location_id, prepared$train_df$time_id)
      list(
        contract = contract,
        training = prepared$train_df[ordering, , drop = FALSE],
        transformed_training = contract$table_bundle$truth_df$Covariate1[
          training_source_rows
        ]
      )
    }
    original <- prepare(bundle)
    perturbed <- prepare(perturbed_bundle)
    metadata <- original$contract$dataset_spec$ndm_preapplied_initial_transform

    expect_identical(
      original$contract$dataset_spec$initial_transform,
      "none",
      info = split_type
    )
    expect_identical(metadata$method, "yeoJohnson", info = split_type)
    expect_identical(
      metadata$fit_partition,
      "producer_train_rows",
      info = split_type
    )
    expect_identical(
      metadata$n_fit_rows,
      nrow(raw_prepared$train_df),
      info = split_type
    )
    expect_equal(
      metadata$parameters$Covariate1$lambda,
      perturbed$contract$dataset_spec$
        ndm_preapplied_initial_transform$parameters$Covariate1$lambda,
      tolerance = 1e-12,
      info = split_type
    )
    expect_equal(
      original$transformed_training,
      perturbed$transformed_training,
      tolerance = 1e-12,
      info = split_type
    )
    if (!identical(split_type, "OutOfPlace")) {
      expect_equal(
        original$training$Covariate1,
        perturbed$training$Covariate1,
        tolerance = 1e-12,
        info = split_type
      )
    }
    expect_false(
      identical(
        original$contract$source_sha256,
        perturbed$contract$source_sha256
      ),
      info = split_type
    )
  }
})

test_that("multidisease bootstrap delegates serial publication with dataSeed", {
  project_root <- tempfile("ndm-multidisease-write-")
  dir.create(project_root)
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)
  grid <- ndm_test_multidisease_bootstrap_grid()[1:2, , drop = FALSE]
  grid$nSamplesTrain <- c(10L, 20L)
  publication <- NULL
  contract_args <- NULL
  publication_count <- 0L

  local_mocked_bindings(
    .ndm_prepare_multidisease_bundle = function(...) list(fake = TRUE),
    .ndm_multidisease_artifact_contract = function(row_values, ...) {
      contract_args <<- list(...)
      list(
        table_bundle = structure(list(), class = "fake_bundle"),
        dataset_spec = list(
          base_id = as.integer(row_values$BaseID[[1L]]),
          n_inference_samples = 2L
        ),
        data_seed = as.integer(row_values$dataSeed[[1L]])
      )
    },
    .ndm_resolve_tensorflow = function(...) "tensorflow",
    .ndm_canonical_dataset_call = function(name, ...) {
      args <- list(...)
      if (identical(name, "ndm_datasets_training_spec")) {
        return(args)
      }
      expect_identical(name, "ndm_real_bootstrap_tfrecords")
      publication <<- args
      publication_count <<- publication_count + 1L
      list(status = c("written", "skipped_existing")[[publication_count]])
    },
    .package = "ndm"
  )

  result <- ndm_bootstrap_multidisease_tfrecords(
    project_root = project_root,
    grid = grid,
    disease_names = "TB",
    training_target_horizon = 4L,
    producer = list(contract = "test-v1")
  )

  expect_identical(result$canonical_row, 2L)
  expect_identical(result$artifact_n_samples_train, 20L)
  expect_identical(publication$training_spec$n_samples_train, 20L)
  expect_identical(publication$seed, 20260709L)
  expect_identical(contract_args$training_target_horizon, 4L)
  expect_true(publication$verify_readable)
  expect_false(publication$overwrite)
  expect_identical(result$status, "written")

  reused <- ndm_bootstrap_multidisease_tfrecords(
    project_root = project_root,
    grid = grid,
    disease_names = "TB",
    training_target_horizon = 4L,
    producer = list(contract = "test-v1")
  )
  expect_identical(reused$status, "skipped_existing")
})

test_that("multidisease bootstrap validates the training target horizon", {
  project_root <- tempfile("ndm-multidisease-horizon-")
  dir.create(project_root)
  on.exit(unlink(project_root, recursive = TRUE, force = TRUE), add = TRUE)
  grid <- ndm_test_multidisease_bootstrap_grid()[1L, , drop = FALSE]

  expect_error(
    ndm_bootstrap_multidisease_tfrecords(
      project_root = project_root,
      grid = grid,
      disease_names = "TB",
      lookahead = 4L,
      training_target_horizon = 0L,
      dry_run = TRUE
    ),
    "positive"
  )
  expect_error(
    ndm_bootstrap_multidisease_tfrecords(
      project_root = project_root,
      grid = grid,
      disease_names = "TB",
      lookahead = 4L,
      training_target_horizon = 5L,
      dry_run = TRUE
    ),
    "no greater than"
  )
})

test_that("multidisease training preflight binds the complete row contract", {
  variable <- "NDM_TFRECORD_PRODUCER_CONTRACT"
  old_value <- Sys.getenv(variable, unset = NA_character_)
  on.exit({
    if (is.na(old_value)) {
      Sys.unsetenv(variable)
    } else {
      do.call(Sys.setenv, stats::setNames(list(old_value), variable))
    }
  }, add = TRUE)
  Sys.setenv(NDM_TFRECORD_PRODUCER_CONTRACT = "test-v1")

  runtime_env <- new.env(parent = emptyenv())
  support <- data.frame(location_id = "AAA", eligible = TRUE)
  contract <- list(
    table_bundle = list(kind = "real"),
    dataset_spec = list(base_id = 2007L),
    source_sha256 = "source",
    data_seed = 20260709L,
    n_inference = 2L,
    inference_support = support
  )
  seen <- NULL

  local_mocked_bindings(
    .ndm_multidisease_artifact_contract = function(...) contract,
    .ndm_preflight_canonical_real_runtime = function(...) {
      seen <<- list(...)
      invisible(list())
    },
    .package = "ndm"
  )

  result <- ndm:::.ndm_preflight_multidisease_tfrecords(
    runtime_env = runtime_env,
    row_values = data.frame(
      BaseID = 2007L,
      nSamplesTrain = 20000L
    ),
    bundle = list(),
    paths = list(train_file = "train", inference_file = "inference"),
    verify_checksum = FALSE
  )

  expect_identical(result, contract)
  expect_identical(seen$dataset_spec, contract$dataset_spec)
  expect_identical(seen$base_id, 2007L)
  expect_identical(seen$n_train, 20000L)
  expect_identical(seen$n_inference, 2L)
  expect_identical(seen$source_sha256, "source")
  expect_identical(seen$expected_seed, 20260709L)
  expect_identical(seen$expected_producer, list(contract = "test-v1"))
  expect_identical(seen$expected_inference_support, support)
  expect_false(seen$verify_checksum)
  expect_identical(runtime_env$ndm_real_table_bundle, contract$table_bundle)
  expect_false(runtime_env$ndm_canonical_verify_checksum)
})

test_that("multidisease expected producer has an explicit cross-process contract", {
  variable <- "NDM_TFRECORD_PRODUCER_CONTRACT"
  old_value <- Sys.getenv(variable, unset = NA_character_)
  on.exit({
    if (is.na(old_value)) {
      Sys.unsetenv(variable)
    } else {
      do.call(Sys.setenv, stats::setNames(list(old_value), variable))
    }
  }, add = TRUE)

  runtime_env <- new.env(parent = emptyenv())
  Sys.setenv(NDM_TFRECORD_PRODUCER_CONTRACT = "tb-rolling-origin-v1")
  expect_identical(
    ndm:::.ndm_multidisease_expected_producer(runtime_env),
    list(contract = "tb-rolling-origin-v1")
  )

  runtime_env$ndm_tfrecord_producer <- list(contract = "runtime-v1")
  expect_identical(
    ndm:::.ndm_multidisease_expected_producer(runtime_env),
    list(contract = "runtime-v1")
  )

  rm("ndm_tfrecord_producer", envir = runtime_env)
  Sys.unsetenv(variable)
  expect_error(
    ndm:::.ndm_multidisease_expected_producer(runtime_env),
    "requires expected producer metadata"
  )
})

test_that("canonical pair validation binds seed, producer, shapes, and support", {
  support <- data.frame(
    location_id = c("AAA", "BBB"),
    location_id_numeric = 0:1,
    eligible = c(TRUE, TRUE),
    origin_time_id = c(7L, 7L),
    evaluation_horizon = c(4L, 4L),
    context_length = c(8L, 8L),
    stringsAsFactors = FALSE
  )
  normalized_support <- lapply(support, identity)
  manifest <- function(split) {
    list(
      split = split,
      base_id = 2007L,
      n_examples = if (identical(split, "train")) 20L else 2L,
      dataset_spec = list(kind = "real", base_id = 2007L),
      scaler_sha256 = "scaler",
      producer_sha256 = "producer",
      schema = list(example_shape_map = list(XPred = c(8L, 1L))),
      metadata = list(
        source_sha256 = "source",
        inference_support = normalized_support,
        inference_support_sha256 = "support"
      )
    )
  }
  manifests <- list(train = manifest("train"), inference = manifest("inference"))
  seen <- list()
  dataset_call <- function(name, ...) {
    args <- list(...)
    seen[[length(seen) + 1L]] <<- args
    if (!identical(args$expected_seed, 20260709L)) {
      stop("different seed metadata")
    }
    manifests[[args$expected_split]]
  }
  paths <- list(train_file = "train", inference_file = "inference")

  pair <- ndm:::.ndm_validate_canonical_tfrecord_pair(
    paths = paths,
    schema_kind = "real",
    dataset_spec = manifests$train$dataset_spec,
    n_train = 20L,
    n_inference = 2L,
    source_sha256 = "source",
    expected_seed = 20260709L,
    expected_producer = list(contract = "test-v1"),
    expected_inference_support = support,
    dataset_call = dataset_call
  )
  expect_identical(pair$inference_support, normalized_support)
  expect_identical(seen[[1L]]$expected_seed, 20260709L)
  expect_identical(seen[[2L]]$expected_seed, 20260709L)
  expect_identical(seen[[1L]]$expected_producer, list(contract = "test-v1"))

  expect_error(
    ndm:::.ndm_validate_canonical_tfrecord_pair(
      paths = paths,
      schema_kind = "real",
      dataset_spec = manifests$train$dataset_spec,
      expected_seed = 0L,
      dataset_call = dataset_call
    ),
    "different seed"
  )

  manifests$inference$producer_sha256 <- "other"
  expect_error(
    ndm:::.ndm_validate_canonical_tfrecord_pair(
      paths = paths,
      schema_kind = "real",
      dataset_spec = manifests$train$dataset_spec,
      expected_seed = 20260709L,
      dataset_call = dataset_call
    ),
    "different producers"
  )
})
