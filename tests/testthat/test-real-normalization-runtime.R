real_normalization_runtime_helper <- function() {
  helper_env <- new.env(parent = baseenv())
  expressions <- ndm_test_runtime_source_expressions(
    "SetupData/SuperLModel_DataGenerator_Real.R"
  )
  eval(expressions[[1L]], envir = helper_env)
  helper_env$.ndm_real_normalize_column
}

test_that("real Yeo-Johnson normalization never fits on future observations", {
  skip_if_not_installed("bestNormalize")
  normalize_column <- real_normalization_runtime_helper()

  time_id <- rep(0:15, each = 4L)
  in_sample_indices <- which(time_id <= 9L)
  location_offset <- rep(c(-0.3, -0.05, 0.1, 0.4), times = 16L)
  training_values <- exp(seq(-1.2, 2.2, length.out = length(in_sample_indices))) +
    location_offset[in_sample_indices]

  future_a <- seq(1, 4, length.out = length(time_id) - length(in_sample_indices))
  future_b <- c(
    seq(-250, -25, length.out = 12L),
    seq(50, 5000, length.out = 12L)
  )
  values_a <- c(training_values, future_a)
  values_b <- c(training_values, future_b)

  normalized <- lapply(c("all", "at_t"), function(norm_type) {
    list(
      a = normalize_column(
        values_a,
        in_sample_indices,
        time_id,
        initial_transform = "yeoJohnson",
        initial_norm_type = norm_type
      ),
      b = normalize_column(
        values_b,
        in_sample_indices,
        time_id,
        initial_transform = "yeoJohnson",
        initial_norm_type = norm_type
      )
    )
  })

  for (result in normalized) {
    expect_equal(
      result$a[in_sample_indices],
      result$b[in_sample_indices],
      tolerance = 1e-12
    )
  }

  # Keep the fixture sensitive to the original defect: a full-column fit must
  # produce a measurably different normalized training slice.
  legacy_normalize <- function(values) {
    transformed <- as.numeric(stats::predict(bestNormalize::yeojohnson(values)))
    train <- transformed[in_sample_indices]
    (transformed - mean(train)) / (0.001 + stats::sd(train))
  }
  expect_gt(
    max(abs(
      legacy_normalize(values_a)[in_sample_indices] -
        legacy_normalize(values_b)[in_sample_indices]
    )),
    1e-3
  )
})

real_normalization_analysis2_api <- function() {
  ndm:::.ndm_new_run_impl_env()
}

real_normalization_test_bundle <- function(future_values) {
  locations <- sprintf("L%d", 1:4)
  truth_df <- data.table::CJ(
    location_id = locations,
    week_id = 0:15
  )
  truth_df[, location_name := location_id]
  truth_df[, LOC2_region_name := "Western Europe"]
  truth_df[, targetWeek_id := week_id]
  truth_df[, POP_population := 1e6]
  truth_df[, x := exp((week_id + match(location_id, locations) / 5) / 5)]
  truth_df[week_id > 9L, x := rep(future_values, length.out = .N)]
  truth_df[, y := 0.01 * (week_id + 1)]

  structure(
    list(
      disease = "Covid",
      outcome_metric = "inc_death",
      per_capita_scaling_factor = 10000,
      roll_window = 26L,
      truth_df = truth_df,
      predicted_df = NULL,
      default_data_inputs = "x",
      true_value_names = "y",
      all_true_value_names = "y"
    ),
    class = c("ndm_datasets_table_bundle", "list")
  )
}

real_normalization_test_spec <- function() {
  ndmdatasets::ndm_datasets_dataset_spec(
    kind = "real",
    context_length = 4L,
    lookahead = 3L,
    evaluation_time = 3L,
    evaluation_sequence = 1:4,
    initial_transform = "yeoJohnson",
    initial_norm_type = "all",
    padding_method = "left",
    split_type = "OutOfTime",
    data_subset = "all",
    data_inputs = "x",
    outcomes = "y",
    min_anchoring_time = 1L,
    n_inference_samples = 4L,
    base_id = 1L
  )
}

test_that("Analysis2 canonical real preparation preapplies a train-only transform", {
  skip_if_not_installed("bestNormalize")
  skip_if_not_installed("data.table")
  skip_if_not_installed("ndmdatasets")
  api <- real_normalization_analysis2_api()
  dataset_spec <- real_normalization_test_spec()
  bundle_a <- real_normalization_test_bundle(seq(1, 4, length.out = 24L))
  bundle_b <- real_normalization_test_bundle(c(
    seq(-200, -10, length.out = 12L),
    seq(10, 5000, length.out = 12L)
  ))

  transformed_a <- api$analysis2_preapply_real_initial_transform(
    "ndmdatasets",
    bundle_a,
    dataset_spec
  )
  transformed_b <- api$analysis2_preapply_real_initial_transform(
    "ndmdatasets",
    bundle_b,
    dataset_spec
  )
  training_rows <- which(bundle_a$truth_df$week_id <= 9L)

  expect_identical(dataset_spec$initial_transform, "yeoJohnson")
  expect_identical(transformed_a$dataset_spec$initial_transform, "none")
  expect_identical(
    transformed_a$dataset_spec$analysis2_preapplied_initial_transform$method,
    "yeoJohnson"
  )
  expect_identical(
    transformed_a$dataset_spec$
      analysis2_preapplied_initial_transform$fit_partition,
    "train"
  )
  expect_equal(
    transformed_a$dataset_spec$
      analysis2_preapplied_initial_transform$parameters$x$lambda,
    transformed_b$dataset_spec$
      analysis2_preapplied_initial_transform$parameters$x$lambda,
    tolerance = 0
  )
  expect_equal(
    transformed_a$table_bundle$truth_df$x[training_rows],
    transformed_b$table_bundle$truth_df$x[training_rows],
    tolerance = 0
  )

  state_a <- api$analysis2_prepare_real_state(
    "ndmdatasets",
    bundle_a,
    dataset_spec,
    tempdir()
  )
  state_b <- api$analysis2_prepare_real_state(
    "ndmdatasets",
    bundle_b,
    dataset_spec,
    tempdir()
  )
  expect_identical(state_a$dataset_spec$initial_transform, "none")
  expect_identical(state_a$requested_dataset_spec$initial_transform, "yeoJohnson")
  expect_equal(state_a$input_df_red_in$x, state_b$input_df_red_in$x, tolerance = 0)
})

test_that("Analysis2 real bootstrap passes only the pretransformed canonical contract", {
  skip_if_not_installed("bestNormalize")
  skip_if_not_installed("data.table")
  skip_if_not_installed("ndmdatasets")
  dataset_spec <- real_normalization_test_spec()

  capture_bootstrap <- function(bundle) {
    api <- real_normalization_analysis2_api()
    captured <- NULL
    api$analysis2_model_type <- function(...) "DecoderOnly"
    api$analysis2_real_dataset_spec <- function(...) dataset_spec
    api$analysis2_real_training_spec <- function(...) {
      list(n_samples_train = 1L)
    }
    api$analysis2_log <- function(...) invisible(NULL)
    api$analysis2_call <- function(pkg, name, ...) {
      args <- list(...)
      if (identical(name, "ndm_real_resolve_inputs")) {
        return("x")
      }
      if (identical(name, "ndm_real_bootstrap_tfrecords")) {
        captured <<- args
        return(list(status = "written"))
      }
      stop("Unexpected mocked ndmdatasets call: ", name)
    }

    status <- api$analysis2_bootstrap_write_real_tfrecord(
      base_id = 1L,
      canonical_row = 1L,
      artifact_n_samples_train = 4L,
      row_values = list(ModelType = "DecoderOnly"),
      tfrecord_dir = tempdir(),
      outcome_metric = "inc_death",
      data_subset = "all",
      producer = list(contract = "test"),
      ndmdatasets_pkg = "ndmdatasets",
      resolve_backend = function() {
        list(
          ndmdatasets_pkg = "ndmdatasets",
          bundle = bundle,
          tensorflow = NULL
        )
      }
    )
    expect_identical(status, "written")
    captured
  }

  captured_a <- capture_bootstrap(
    real_normalization_test_bundle(seq(1, 4, length.out = 24L))
  )
  captured_b <- capture_bootstrap(real_normalization_test_bundle(c(
    seq(-200, -10, length.out = 12L),
    seq(10, 5000, length.out = 12L)
  )))
  training_rows <- which(captured_a$table_bundle$truth_df$week_id <= 9L)

  expect_identical(captured_a$dataset_spec$initial_transform, "none")
  expect_identical(
    captured_a$dataset_spec$analysis2_preapplied_initial_transform$method,
    "yeoJohnson"
  )
  expect_equal(
    captured_a$table_bundle$truth_df$x[training_rows],
    captured_b$table_bundle$truth_df$x[training_rows],
    tolerance = 0
  )
})

test_that("Analysis2 real preflight validates the pretransformed manifest contract", {
  skip_if_not_installed("bestNormalize")
  skip_if_not_installed("data.table")
  skip_if_not_installed("ndmdatasets")
  api <- real_normalization_analysis2_api()
  dataset_spec <- real_normalization_test_spec()
  bundle <- real_normalization_test_bundle(seq(1, 4, length.out = 24L))
  source_bundle <- NULL
  validated <- NULL

  api$analysis2_real_dataset_spec <- function(...) dataset_spec
  api$analysis2_trusted_artifact_index <- function(...) NULL
  api$analysis2_trusted_index_covers_pair <- function(...) FALSE
  api$analysis2_call <- function(pkg, name, ...) {
    args <- list(...)
    if (identical(name, "ndm_real_resolve_inputs")) {
      return("x")
    }
    if (identical(name, "ndm_real_source_sha256")) {
      source_bundle <<- args[[1L]]
      return("pretransformed-source")
    }
    stop("Unexpected mocked ndmdatasets call: ", name)
  }
  api$analysis2_validate_canonical_tfrecord_pair <- function(...) {
    validated <<- list(...)
    list(train = list(), inference = list())
  }

  result <- api$analysis2_preflight_expected_tfrecords(
    mode = "real",
    write_plan = data.frame(
      BaseID = 1L,
      canonical_row = 1L,
      artifact_n_samples_train = 4L
    ),
    grid = data.frame(BaseID = 1L),
    tfrecord_dir = tempdir(),
    ndmdatasets_pkg = "ndmdatasets",
    data_subset = "all",
    real_bundle = bundle
  )

  expect_named(result, "1")
  expect_identical(validated$dataset_spec$initial_transform, "none")
  expect_identical(
    validated$dataset_spec$analysis2_preapplied_initial_transform$method,
    "yeoJohnson"
  )
  expect_identical(validated$source_sha256, "pretransformed-source")
  expect_false(identical(source_bundle$truth_df$x, bundle$truth_df$x))
})
