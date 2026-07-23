ndm_test_train_batch_functions <- function() {
  source_path <- testthat::test_path(
    "..", "..", "tools", "runtime_source", "ndm_runtime",
    "ModelTrainers", "SuperLModel_TrainDo.R"
  )
  if (!file.exists(source_path)) {
    testthat::skip("runtime source tree is not available in this installed-package context")
  }

  expressions <- parse(file = source_path, keep.source = FALSE)
  assignment_name <- function(expr) {
    if (!is.call(expr) || !identical(expr[[1L]], as.name("<-"))) {
      return(NA_character_)
    }
    as.character(expr[[2L]])
  }
  expression_names <- vapply(expressions, assignment_name, character(1L))
  required <- c(
    "reset_train_iterator",
    "batch_has_expected_shape",
    "next_train_batch"
  )
  if (!all(required %in% expression_names)) {
    stop("Could not locate the training batch helper definitions.", call. = FALSE)
  }

  env <- new.env(parent = globalenv())
  for (name in required) {
    eval(expressions[[match(name, expression_names)]], envir = env)
  }
  env
}

ndm_test_runtime_batch <- function(batch_size) {
  list(
    list(shape = c(as.integer(batch_size), 3L)),
    list(shape = c(as.integer(batch_size), 1L))
  )
}

test_that("TrainDo propagates iterator exceptions without resetting", {
  env <- ndm_test_train_batch_functions()
  sentinel <- structure(
    list(message = "iterator sentinel", call = NULL, token = "iterator-token"),
    class = c("ndm_test_iterator_error", "error", "condition")
  )
  reset_count <- 0L
  env$TFDatasetIterator_train <- structure(list(), class = "ndm_test_iterator")
  env$TFDataset_train <- structure(list(), class = "ndm_test_dataset")
  env$nBatch <- 2L
  env$np <- list(array = identity)
  env$TFConst2JAXArray <- identity
  env$print2 <- function(...) invisible(NULL)

  local_mocked_bindings(
    iter_next = function(it, completed = NULL) stop(sentinel),
    as_iterator = function(iterable) {
      reset_count <<- reset_count + 1L
      iterable
    },
    .package = "reticulate"
  )

  caught <- tryCatch(env$next_train_batch(max_attempts = 3L), error = identity)

  expect_identical(caught, sentinel)
  expect_s3_class(caught, "ndm_test_iterator_error")
  expect_identical(reset_count, 0L)
})

test_that("TrainDo retries incomplete batches and accepts a full batch", {
  env <- ndm_test_train_batch_functions()
  batches <- list(
    NULL,
    list(list()),
    ndm_test_runtime_batch(1L),
    ndm_test_runtime_batch(2L)
  )
  next_index <- 0L
  reset_count <- 0L
  retry_messages <- character()
  env$TFDatasetIterator_train <- structure(list(), class = "ndm_test_iterator")
  env$TFDataset_train <- structure(list(), class = "ndm_test_dataset")
  env$nBatch <- 2L
  env$np <- list(array = identity)
  env$TFConst2JAXArray <- function(batch) structure(batch, converted = TRUE)
  env$print2 <- function(message) {
    retry_messages <<- c(retry_messages, message)
    invisible(NULL)
  }

  local_mocked_bindings(
    iter_next = function(it, completed = NULL) {
      next_index <<- next_index + 1L
      batches[[next_index]]
    },
    as_iterator = function(iterable) {
      reset_count <<- reset_count + 1L
      iterable
    },
    .package = "reticulate"
  )

  batch <- env$next_train_batch(max_attempts = 4L)

  expect_true(isTRUE(attr(batch, "converted")))
  expect_identical(next_index, 4L)
  expect_identical(reset_count, 3L)
  expect_length(retry_messages, 3L)
})

test_that("TrainDo reports full-batch exhaustion accurately", {
  env <- ndm_test_train_batch_functions()
  reset_count <- 0L
  env$TFDatasetIterator_train <- structure(list(), class = "ndm_test_iterator")
  env$TFDataset_train <- structure(list(), class = "ndm_test_dataset")
  env$nBatch <- 7L
  env$np <- list(array = identity)
  env$TFConst2JAXArray <- identity
  env$print2 <- function(...) invisible(NULL)

  local_mocked_bindings(
    iter_next = function(it, completed = NULL) NULL,
    as_iterator = function(iterable) {
      reset_count <<- reset_count + 1L
      iterable
    },
    .package = "reticulate"
  )

  expect_error(
    env$next_train_batch(max_attempts = 3L),
    "Could not obtain a full training batch of 7 examples after 3 attempts \\(2 retries\\) in TrainDo\\.R\\."
  )
  expect_identical(reset_count, 2L)
})
