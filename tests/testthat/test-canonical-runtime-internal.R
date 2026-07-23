test_that("canonical pair validation adopts the train manifest specification", {
  calls <- new.env(parent = emptyenv())
  calls$values <- list()
  dataset_spec <- list(kind = "sim", context_length = 8L, base_id = 1L)
  dataset_call <- function(name, ...) {
    args <- list(...)
    calls$values[[length(calls$values) + 1L]] <- c(list(name = name), args)
    list(
      n_examples = if (identical(args$file, "train")) 4L else 128L,
      dataset_spec = dataset_spec,
      scaler = list(mean = c(1, 2), sd = c(3, 4)),
      scaler_sha256 = "shared",
      metadata = list()
    )
  }

  pair <- ndm:::.ndm_validate_canonical_tfrecord_pair(
    paths = list(train_file = "train", inference_file = "inference"),
    schema_kind = "sim",
    dataset_spec = NULL,
    n_train = 4L,
    n_inference = 128L,
    dataset_call = dataset_call
  )

  expect_identical(pair$dataset_spec, dataset_spec)
  expect_false("expected_dataset_spec" %in% names(calls$values[[1L]]))
  expect_identical(calls$values[[2L]]$expected_dataset_spec, dataset_spec)
  expect_identical(calls$values[[2L]]$expected_n_examples, 128L)
})

test_that("canonical attachment uses ndmdatasets readers and seeds stable handles", {
  calls <- new.env(parent = emptyenv())
  calls$values <- list()
  dataset_call <- function(name, ...) {
    args <- list(...)
    calls$values[[length(calls$values) + 1L]] <- c(list(name = name), args)
    structure(list(file = args$file), class = "fake_dataset")
  }
  runtime_env <- new.env(parent = emptyenv())
  runtime_env$tf <- list(fake = TRUE)
  runtime_env$nSamplesTrain <- 5L
  runtime_env$nObsInference <- 7L

  ndm:::.ndm_attach_canonical_tfrecords(
    runtime_env = runtime_env,
    train_file = "train",
    inference_file = "inference",
    schema_kind = "real",
    batch_size = 2L,
    shuffle_train = TRUE,
    run_seed = 11L,
    dataset_call = dataset_call,
    iterator_factory = function(dataset) list(iterator_for = dataset$file)
  )

  expect_equal(length(calls$values), 2L)
  expect_identical(calls$values[[1L]]$max_examples, 5L)
  expect_identical(calls$values[[1L]]$shuffle_seed, 11L)
  expect_true(calls$values[[1L]]$reshuffle_each_iteration)
  expect_identical(calls$values[[2L]]$max_examples, 7L)
  expect_false(calls$values[[2L]]$shuffle)
  expect_identical(runtime_env$TFDatasetIterator_train$iterator_for, "train")
  expect_identical(runtime_env$ndm_canonical_tfrecord_schema, "real")
})

test_that("runtime batch seeding preserves canonical shapes", {
  runtime_env <- new.env(parent = emptyenv())
  batch <- list(
    XPred = array(seq_len(24L), dim = c(2L, 4L, 3L)),
    YTrue = array(seq_len(8L), dim = c(2L, 4L, 1L))
  )

  ndm:::.ndm_seed_runtime_batch(runtime_env, batch)

  expect_identical(runtime_env$batch_l, batch)
  expect_identical(runtime_env$batch_l_cal, batch)
  expect_identical(runtime_env$jax_batchx, batch$XPred)
  expect_identical(runtime_env$jax_batchy, batch$YTrue)
  expect_identical(runtime_env$nCovars_real, 3L)
})

test_that("model outcome selection preserves rank and pairs values with masks", {
  y <- array(seq_len(24L), dim = c(2L, 3L, 4L))
  y_mask <- array(rep(c(TRUE, FALSE), 12L), dim = c(2L, 3L, 4L))

  selected <- ndm:::.ndm_select_model_targets(
    y = y,
    y_mask = y_mask,
    n_outcomes = 1L
  )

  expect_identical(dim(selected$y), c(2L, 3L, 1L))
  expect_identical(dim(selected$y_mask), c(2L, 3L, 1L))
  expect_identical(selected$y, y[, , 1L, drop = FALSE])
  expect_identical(selected$y_mask, y_mask[, , 1L, drop = FALSE])
  expect_identical(dim(y), c(2L, 3L, 4L))

  selected_two <- ndm:::.ndm_select_model_targets(
    y = y,
    y_mask = y_mask,
    n_outcomes = 2L
  )
  expect_identical(dim(selected_two$y), c(2L, 3L, 2L))
  expect_identical(selected_two$y, y[, , 1:2, drop = FALSE])
})

test_that("model outcome selection validates tensor shape and channel availability", {
  expect_error(
    ndm:::.ndm_select_model_targets(1:3, 1:3, 1L),
    "`y` must have at least one dimension",
    fixed = TRUE
  )
  expect_error(
    ndm:::.ndm_select_model_targets(
      array(1, dim = c(2L, 3L, 1L)),
      array(TRUE, dim = c(2L, 3L)),
      1L
    ),
    "same rank"
  )
  expect_error(
    ndm:::.ndm_select_model_targets(
      array(1, dim = c(2L, 3L, 1L)),
      array(TRUE, dim = c(2L, 4L, 1L)),
      1L
    ),
    "non-channel dimensions"
  )
  expect_error(
    ndm:::.ndm_select_model_targets(
      array(1, dim = c(2L, 3L, 1L)),
      array(TRUE, dim = c(2L, 3L, 2L)),
      2L
    ),
    "`y` has 1 outcome channel"
  )
  expect_error(
    ndm:::.ndm_select_model_targets(
      array(1, dim = c(2L, 3L, 2L)),
      array(TRUE, dim = c(2L, 3L, 1L)),
      2L
    ),
    "`y_mask` has 1 outcome channel"
  )
  expect_error(
    ndm:::.ndm_select_model_targets(
      array(1, dim = c(2L, 3L, 2L)),
      array(TRUE, dim = c(2L, 3L, 2L)),
      0L
    ),
    "positive integer"
  )
})

test_that("model outcome selection uses trace-safe backend take operations", {
  calls <- new.env(parent = emptyenv())
  calls$arange <- list()
  calls$take <- list()
  backend <- list(
    arange = function(stop) {
      calls$arange[[length(calls$arange) + 1L]] <- stop
      seq.int(0L, stop - 1L)
    },
    take = function(x, indices, axis) {
      calls$take[[length(calls$take) + 1L]] <- list(
        indices = indices,
        axis = axis
      )
      selectors <- rep(list(TRUE), length(dim(x)))
      selectors[[length(selectors)]] <- indices + 1L
      do.call(`[`, c(list(x), selectors, list(drop = FALSE)))
    }
  )
  y <- array(seq_len(24L), dim = c(2L, 3L, 4L))
  y_mask <- array(TRUE, dim = c(2L, 3L, 4L))

  selected <- ndm:::.ndm_select_model_targets(
    y,
    y_mask,
    n_outcomes = 1L,
    jnp = backend
  )

  expect_identical(dim(selected$y), c(2L, 3L, 1L))
  expect_identical(dim(selected$y_mask), c(2L, 3L, 1L))
  expect_identical(calls$arange, list(1L))
  expect_length(calls$take, 2L)
  expect_identical(calls$take[[1L]], list(indices = 0L, axis = -1L))
  expect_error(
    ndm:::.ndm_select_model_targets(
      y,
      y_mask,
      n_outcomes = 1L,
      jnp = list(take = backend$take)
    ),
    "provide `arange()` and `take()`",
    fixed = TRUE
  )
})

test_that("canonical simulation setup validates the active runtime specification", {
  dataset_spec <- NULL
  scaler <- list(mean = c(10, 20), sd = c(2, 4))
  calls <- new.env(parent = emptyenv())
  calls$validation <- list()
  dataset_call <- function(name, ...) {
    args <- list(...)
    if (identical(name, "ndm_datasets_dataset_spec")) {
      dataset_spec <<- c(list(kind = args$kind), args[setdiff(names(args), "kind")])
      return(dataset_spec)
    }
    if (identical(name, "ndm_datasets_validate_tfrecord_artifact")) {
      calls$validation[[length(calls$validation) + 1L]] <- args
      return(list(
        n_examples = if (grepl("train", args$file, fixed = TRUE)) 2L else 128L,
        dataset_spec = dataset_spec,
        scaler = scaler,
        scaler_sha256 = "shared",
        metadata = list()
      ))
    }
    if (identical(name, "ndm_sim_make_example")) {
      return(list(
        XPred = matrix(seq_len(6L), nrow = 3L, ncol = 2L),
        YTrue = matrix(seq_len(3L), nrow = 3L, ncol = 1L)
      ))
    }
    if (identical(name, "ndm_datasets_stack_batches")) {
      examples <- args$examples
      return(list(
        XPred = array(
          unlist(lapply(examples, `[[`, "XPred"), use.names = FALSE),
          dim = c(length(examples), 3L, 2L)
        ),
        YTrue = array(
          unlist(lapply(examples, `[[`, "YTrue"), use.names = FALSE),
          dim = c(length(examples), 3L, 1L)
        )
      ))
    }
    if (identical(name, "ndm_datasets_read_tfrecord_dataset")) {
      return(structure(list(file = args$file), class = "fake_dataset"))
    }
    stop("Unexpected dataset call: ", name)
  }
  runtime_env <- new.env(parent = emptyenv())
  runtime_env$tf <- list(fake = TRUE)
  runtime_env$jnp <- list(array = function(x) x)
  runtime_env$TfRecordDir <- "records"
  runtime_env$SimEntry <- list(
    BaseID = 1L,
    nSamplesTrain = 2L,
    n_inference_batches = 1L
  )
  runtime_env$nSamplesTrain <- 2L
  runtime_env$nBatch <- 2L
  runtime_env$nTimesPast <- 2L
  runtime_env$nTimesLookahead <- 1L
  runtime_env$NTimeSteps_SIM <- 6L

  result <- ndm:::.ndm_prepare_canonical_sim_runtime(
    runtime_env = runtime_env,
    sim_outcome_sd = 0.5,
    dataset_call = dataset_call,
    iterator_factory = function(dataset) list(iterator_for = dataset$file)
  )

  expect_false(result$skipped)
  expect_identical(result$dataset_spec, dataset_spec)
  expect_identical(calls$validation[[1L]]$expected_dataset_spec, dataset_spec)
  expect_identical(calls$validation[[2L]]$expected_dataset_spec, dataset_spec)
  expect_identical(runtime_env$SIM_GLOBAL_SCALE_MEAN, scaler$mean)
  expect_identical(runtime_env$SIM_GLOBAL_SCALE_SD, scaler$sd)
  expect_identical(runtime_env$SIM_GLOBAL_OUTCOME_SD, 0.5)
  expect_true(is.function(runtime_env$GetBatch))
  expect_identical(runtime_env$nCovars_real, 2L)
  expect_identical(runtime_env$nBatch_SimGridGen, 256L)
  expect_identical(runtime_env$nMonteEval, 4L)
  expect_identical(runtime_env$nTimesLookValidation, 1L)
  expect_identical(runtime_env$TFDatasetIterator_train$iterator_for, "records/train_1.tfrecord")
})

test_that("canonical simulation preflight rejects a stale DGP specification", {
  runtime_env <- new.env(parent = emptyenv())
  runtime_env$TfRecordDir <- "records"
  runtime_env$nBatch <- 2L
  runtime_env$nSamplesTrain <- 2L
  runtime_env$nTimesPast <- 8L
  runtime_env$nTimesLookahead <- 4L
  runtime_env$NTimeSteps_SIM <- 24L
  runtime_env$SimEntry <- list(
    BaseID = 1L,
    gamma = 0.99,
    sigma = 0.2,
    xi = 0.02,
    c_endogeneous = 9,
    scaling_batches = 12L,
    n_inference_batches = 1L
  )
  observed_spec <- NULL
  artifact_spec <- NULL
  dataset_call <- function(name, ...) {
    args <- list(...)
    if (identical(name, "ndm_datasets_dataset_spec")) {
      observed_spec <<- c(list(kind = args$kind), args[setdiff(names(args), "kind")])
      artifact_spec <<- observed_spec
      artifact_spec$gamma <<- 0.1
      return(observed_spec)
    }
    if (identical(name, "ndm_datasets_validate_tfrecord_artifact")) {
      if (!identical(args$expected_dataset_spec, artifact_spec)) {
        stop("dataset specification mismatch")
      }
    }
    stop("unexpected call")
  }

  expect_error(
    ndm:::.ndm_prepare_canonical_sim_runtime(
      runtime_env,
      dataset_call = dataset_call
    ),
    class = "ndm_tfrecord_preflight_error",
    regexp = "dataset specification mismatch"
  )
  expect_identical(observed_spec$gamma, 0.99)
  expect_identical(observed_spec$c_endogeneous, 9)
  expect_identical(observed_spec$scaling_batches, 12L)
})

test_that("canonical simulation setup keeps SkipTfRecords as a no-I/O path", {
  runtime_env <- new.env(parent = emptyenv())
  runtime_env$SkipTfRecords <- TRUE
  called <- FALSE

  result <- ndm:::.ndm_prepare_canonical_sim_runtime(
    runtime_env,
    dataset_call = function(...) {
      called <<- TRUE
      stop("must not be called")
    }
  )

  expect_true(result$skipped)
  expect_false(called)
  expect_false(exists("TFDataset_train", envir = runtime_env, inherits = FALSE))
})

test_that("canonical simulation setup gives bootstrap guidance for missing artifacts", {
  runtime_env <- new.env(parent = emptyenv())
  runtime_env$tf <- list(fake = TRUE)
  runtime_env$TfRecordDir <- "missing"
  runtime_env$SimEntry <- list(BaseID = 3L, nSamplesTrain = 2L)
  runtime_env$nBatch <- 2L

  expect_error(
    ndm:::.ndm_prepare_canonical_sim_runtime(
      runtime_env,
      dataset_call = function(...) stop("record does not exist")
    ),
    class = "ndm_tfrecord_preflight_error",
    regexp = "ndm_bootstrap_sim_tfrecords"
  )
})

test_that("canonical real preflight validates the active specification and complete pair", {
  dataset_spec <- NULL
  calls <- list()
  dataset_call <- function(name, ...) {
    args <- list(...)
    calls[[length(calls) + 1L]] <<- c(list(name = name), args)
    if (identical(name, "ndm_datasets_dataset_spec")) {
      dataset_spec <<- c(list(kind = args$kind), args[setdiff(names(args), "kind")])
      return(dataset_spec)
    }
    list(
      base_id = 7L,
      n_examples = if (grepl("train", args$file, fixed = TRUE)) 4L else 32L,
      dataset_spec = dataset_spec,
      scaler = list(type = "not_applicable"),
      scaler_sha256 = "shared",
      metadata = list()
    )
  }
  runtime_env <- new.env(parent = emptyenv())
  runtime_env$TfRecordDir <- "records"
  runtime_env$RealEntry <- list(
    BaseID = 7L,
    nSamplesTrain = 4L,
    n_inference_samples = 32L
  )

  result <- ndm:::.ndm_preflight_canonical_real_runtime(
    runtime_env,
    dataset_call = dataset_call
  )

  expect_false(result$skipped)
  expect_length(calls, 3L)
  expect_identical(calls[[2L]]$expected_split, "train")
  expect_identical(calls[[2L]]$expected_dataset_spec, dataset_spec)
  expect_identical(calls[[3L]]$expected_split, "inference")
  expect_identical(calls[[3L]]$expected_dataset_spec, dataset_spec)
  expect_identical(calls[[3L]]$expected_n_examples, 32L)
  expect_identical(runtime_env$ndm_canonical_dataset_spec, dataset_spec)
})

test_that("canonical pair validation rejects train and inference source drift", {
  call_index <- 0L
  dataset_call <- function(name, ...) {
    call_index <<- call_index + 1L
    list(
      n_examples = if (call_index == 1L) 4L else 8L,
      dataset_spec = list(kind = "real", base_id = 1L),
      scaler_sha256 = "not_applicable",
      metadata = list(source_sha256 = if (call_index == 1L) "train-source" else "inference-source")
    )
  }

  expect_error(
    ndm:::.ndm_validate_canonical_tfrecord_pair(
      paths = list(train_file = "train", inference_file = "inference"),
      schema_kind = "real",
      dataset_spec = list(kind = "real", base_id = 1L),
      n_train = 4L,
      n_inference = 8L,
      dataset_call = dataset_call
    ),
    "different source tables"
  )
})

test_that("canonical real preflight reports bootstrap guidance", {
  runtime_env <- new.env(parent = emptyenv())
  runtime_env$TfRecordDir <- "missing"
  runtime_env$RealEntry <- list(BaseID = 9L, nSamplesTrain = 4L)

  expect_error(
    ndm:::.ndm_preflight_canonical_real_runtime(
      runtime_env,
      dataset_call = function(...) stop("record does not exist")
    ),
    class = "ndm_tfrecord_preflight_error",
    regexp = "ndm_bootstrap_real_tfrecords"
  )
})
