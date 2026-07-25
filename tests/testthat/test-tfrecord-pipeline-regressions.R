ndm_test_write_initial_shift_tfrecord <- function(file, value, tensorflow) {
  tf <- tensorflow
  writer <- tf$io$TFRecordWriter(file)
  serialized <- tf$io$serialize_tensor(
    tf$constant(as.integer(value), dtype = tf$int32)
  )$numpy()
  feature <- reticulate::dict(
    initial_shift = tf$train$Feature(
      bytes_list = tf$train$BytesList(value = list(serialized))
    )
  )
  example <- tf$train$Example(
    features = tf$train$Features(feature = feature)
  )
  writer$write(example$SerializeToString())
  writer$close()
  invisible(file)
}

test_that("simulation TFRecord shifts use the canonical integer dtype", {
  dtype_map <- ndm_tfrecord_dtype_map("sim")

  expect_identical(unname(dtype_map[["initial_shift"]]), "int32")
  expect_identical(unname(dtype_map[["location_id_numeric"]]), "int32")
  expect_identical(unname(dtype_map[["time_id_numeric"]]), "int32")

  if (requireNamespace("ndmdatasets", quietly = TRUE)) {
    canonical_map <- ndmdatasets::ndm_datasets_tfrecord_schema("sim")$dtype_map
    expect_identical(dtype_map[["initial_shift"]], canonical_map[["initial_shift"]])
  }

  inferred <- ndm:::.ndm_normalize_dtype_map(
    c("XPred", "XPred_mask", "location_id_numeric", "initial_shift"),
    dtype_map = NULL
  )
  expect_identical(
    unname(inferred),
    c("float32", "bool", "int32", "int32")
  )
})

test_that("TFRecord iteration distinguishes clean completion from parser errors", {
  local_mocked_bindings(
    iter_next = function(iterator, completed = NULL) completed,
    .package = "reticulate"
  )
  completed <- ndm:::.ndm_tfrecord_iter_next(structure(list(), class = "mock_iterator"))
  expect_true(completed$completed)
  expect_null(completed$batch)

  local_mocked_bindings(
    iter_next = function(iterator, completed = NULL) {
      stop("synthetic TFRecord parse failure", call. = FALSE)
    },
    .package = "reticulate"
  )
  expect_error(
    ndm:::.ndm_tfrecord_iter_next(structure(list(), class = "mock_iterator")),
    "synthetic TFRecord parse failure"
  )
})

test_that("TFRecord collectors do not return partial data after parser errors", {
  dataset <- structure(list(), class = "mock_dataset")
  iterator_calls <- 0L
  local_mocked_bindings(
    ndm_read_tfrecord_dataset = function(...) dataset,
    .package = "ndm"
  )
  local_mocked_bindings(
    as_iterator = function(dataset) structure(list(), class = "mock_iterator"),
    iter_next = function(iterator, completed = NULL) {
      iterator_calls <<- iterator_calls + 1L
      if (iterator_calls == 1L) {
        return(list(XPred = 1))
      }
      stop("second-batch parse failure", call. = FALSE)
    },
    .package = "reticulate"
  )

  expect_error(
    ndm_collect_tfrecord_batches(
      file = "unused.tfrecord",
      batch_size = 1L,
      field_names = "XPred"
    ),
    "second-batch parse failure"
  )
})

test_that("TensorFlow dtype and simulation issue-time regressions stay fixed", {
  conda_env <- ndm_require_backend_test_stack(
    "TFRecord dtype mismatch regression test",
    modules = "tensorflow",
    packages = "reticulate"
  )
  old_backend <- ndm:::ndm_env$backend
  on.exit(assign("backend", old_backend, envir = ndm:::ndm_env), add = TRUE)
  backend <- ndm_initialize_backend(
    conda_env = conda_env,
    float_type = "32",
    import_tensorflow = TRUE
  )

  record_file <- tempfile(fileext = ".tfrecord")
  on.exit(unlink(record_file), add = TRUE)
  ndm_test_write_initial_shift_tfrecord(
    record_file,
    value = 7L,
    tensorflow = backend$tf
  )

  good_batches <- ndm_collect_tfrecord_batches(
    file = record_file,
    batch_size = 1L,
    field_names = "initial_shift",
    dtype_map = ndm_tfrecord_dtype_map("sim")["initial_shift"],
    tensorflow = backend$tf
  )
  expect_length(good_batches, 1L)
  expect_identical(
    as.integer(reticulate::py_to_r(good_batches[[1]]$initial_shift$numpy())[[1L]]),
    7L
  )

  expect_error(
    ndm_collect_tfrecord_batches(
      file = record_file,
      batch_size = 1L,
      field_names = "initial_shift",
      dtype_map = c(initial_shift = "float32"),
      tensorflow = backend$tf
    ),
    "Type mismatch|int32.*float|float.*int32"
  )

  dataset_spec <- ndmdatasets::ndm_datasets_dataset_spec(
    kind = "sim",
    context_length = 8L,
    lookahead = 12L,
    n_time_steps = 40L,
    covariate_type = "sqrt",
    measurement_noise = 0,
    hosp_rate = 0.1,
    death_rate = 0.01,
    forward_shift_h = 0L,
    forward_shift_c = 0L,
    scaling_batches = 1L
  )
  identity_scaler <- list(mean = rep(0, 4L), sd = rep(0.999, 4L))
  example <- ndmdatasets::ndm_sim_make_example(
    dataset_spec = dataset_spec,
    scaler = identity_scaler,
    seed = 20260725L,
    backend = backend
  )
  initial_shift <- as.integer(example$initial_shift[[1L]])
  idx_past <- initial_shift + seq_len(dataset_spec$context_length)
  idx_cases <- idx_past + dataset_spec$forward_shift_c
  idx_hospitalizations <- idx_past + dataset_spec$forward_shift_h
  idx_targets <- max(idx_past) + seq_len(dataset_spec$lookahead)
  expect_length(intersect(idx_cases, idx_targets), 0L)
  expect_length(intersect(idx_hospitalizations, idx_targets), 0L)
  expect_lte(max(idx_cases), max(idx_past))
  expect_lte(max(idx_hospitalizations), max(idx_past))

  case_sqrt <- as.numeric(example$XPred[, 1L])
  hospitalization_sqrt <- as.numeric(example$XPred[, 2L])
  past_deaths <- as.numeric(
    example$YTrue[seq_len(dataset_spec$context_length), 1L]
  )
  expect_equal(
    dataset_spec$death_rate * case_sqrt^2,
    past_deaths,
    tolerance = 1e-5
  )
  expect_equal(
    (dataset_spec$death_rate / dataset_spec$hosp_rate) *
      hospitalization_sqrt^2,
    past_deaths,
    tolerance = 1e-5
  )
})

test_that("simulation dataset defaults keep context covariates before targets", {
  captured_spec <- NULL
  runtime_env <- new.env(parent = emptyenv())
  runtime_env$SimEntry <- list(
    BaseID = 1L,
    forward_shift_h = 4L,
    forward_shift_c = 7L
  )
  dataset_call <- function(name, ...) {
    expect_identical(name, "ndm_datasets_dataset_spec")
    captured_spec <<- list(...)
    captured_spec
  }

  ndm:::.ndm_sim_dataset_spec_from_runtime(
    runtime_env = runtime_env,
    dataset_call = dataset_call
  )
  expect_identical(captured_spec$forward_shift_h, 0L)
  expect_identical(captured_spec$forward_shift_c, 0L)

  idx_past <- seq_len(captured_spec$context_length)
  idx_cases <- idx_past + captured_spec$forward_shift_c
  idx_targets <- max(idx_past) + seq_len(captured_spec$lookahead)
  expect_length(intersect(idx_cases, idx_targets), 0L)
  expect_lte(max(idx_cases), max(idx_past))

  expect_error(
    ndm:::.ndm_assert_sim_covariates_available_at_issue(list(
      forward_shift_h = 4L,
      forward_shift_c = 7L
    )),
    "non-zero.*causal lag"
  )
})

test_that("runtime TFRecord consumers retain fail-fast and issue-time contracts", {
  analytics <- ndm_test_runtime_source_text(
    "ResultsGet/SuperLModel_GetAnalytics_Real.R"
  )
  sim_generator <- ndm_test_runtime_source_text(
    "SetupData/SuperLModel_DataGenerator_Sim.R"
  )

  expect_match(analytics, ".ndm_tfrecord_iter_next", fixed = TRUE)
  expect_false(grepl("try(reticulate::iter_next", analytics, fixed = TRUE))
  expect_match(analytics, "Malformed empty inference batch", fixed = TRUE)
  expect_match(sim_generator, "forward_shift_h <- 0L", fixed = TRUE)
  expect_match(sim_generator, "forward_shift_c <- 0L", fixed = TRUE)
  expect_match(
    sim_generator,
    "\"initial_shift\" = initial_shift$astype(jnp$int32)",
    fixed = TRUE
  )
})

test_that("Analysis2 simulation examples use issue-time-available covariates", {
  skip_if_not_installed("ndmdatasets")
  api <- ndm:::.ndm_new_run_impl_env()

  row_values <- list(
    ContextLength = 8L,
    lookahead = 12L,
    n_time_steps = 40L,
    gamma = 0.2,
    sigma = 0.3,
    xi = 0.01,
    i0_a = log(10),
    i0_b = 0.2,
    r0_a = log(100),
    r0_b = 0.2,
    betat_init = 0.4,
    invbetat_sd = 0.01,
    beta_restore_rate = 0.25,
    c_endogeneous = 0.05,
    policy_responsiveness = 0,
    policy_effectiveness = 0,
    policy_decay = 0,
    measurement_noise = 0,
    n_inference_batches = 1L,
    scaling_batches = 1L,
    BaseID = 1L
  )
  dataset_spec <- api$analysis2_sim_dataset_spec("ndmdatasets", row_values)

  expect_identical(dataset_spec$forward_shift_h, 0L)
  expect_identical(dataset_spec$forward_shift_c, 0L)

  leaky_spec <- dataset_spec
  leaky_spec$forward_shift_c <- 7L
  expect_error(
    api$analysis2_simulate_one(leaky_spec, seed = 20260725L),
    "non-zero forward shifts"
  )

  idx_past <- seq_len(dataset_spec$context_length)
  idx_cases <- idx_past + dataset_spec$forward_shift_c
  idx_targets <- max(idx_past) + seq_len(dataset_spec$lookahead)
  expect_length(intersect(idx_cases, idx_targets), 0L)

  example <- api$analysis2_simulate_one(dataset_spec, seed = 20260725L)
  case_sqrt <- example$XPred_unscaled[, 1L]
  past_deaths <- example$YTrue_pure[
    seq_len(dataset_spec$context_length),
    1L
  ]
  expect_equal(
    as.numeric(dataset_spec$death_rate) * case_sqrt^2,
    as.numeric(past_deaths),
    tolerance = 1e-12
  )
})
