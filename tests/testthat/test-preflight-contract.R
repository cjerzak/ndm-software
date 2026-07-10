test_that("training preflight consumes published scaler metadata instead of recomputing it", {
  env <- ndm:::.ndm_new_run_impl_env()
  captured <- NULL
  env$analysis2_sim_dataset_spec <- function(...) list(n_inference_batches = 1L)
  env$analysis2_sim_scaler <- function(...) stop("preflight must not recompute a device-specific scaler")
  env$analysis2_tfrecord_paths <- function(...) list(train_file = "train", inference_file = "inference")
  env$analysis2_validate_canonical_tfrecord_pair <- function(...) {
    captured <<- list(...)
    invisible(list(train = list(scaler = list(mean = 1, sd = 2))))
  }

  validated <- env$analysis2_preflight_expected_tfrecords(
    mode = "sim",
    write_plan = data.frame(BaseID = 1L, canonical_row = 1L, artifact_n_samples_train = 8L),
    grid = data.frame(BaseID = 1L),
    tfrecord_dir = tempdir(),
    ndmdatasets_pkg = "ndmdatasets"
  )
  expect_equal(captured$n_train, 8L)
  expect_null(captured$source_sha256)
  expect_false("scaler" %in% names(captured))
  expect_equal(validated[["1"]]$train$scaler, list(mean = 1, sd = 2))
})

test_that("canonical pair preflight enforces pair scaler and real source identity", {
  env <- ndm:::.ndm_new_run_impl_env()
  paths <- list(train_file = "train", inference_file = "inference")
  manifest <- function(scaler = "shared", source = "current") {
    list(scaler_sha256 = scaler, metadata = list(source_sha256 = source))
  }

  env$analysis2_call <- function(pkg, name, file, ...) manifest()
  expect_silent(env$analysis2_validate_canonical_tfrecord_pair(
    "ndmdatasets", paths, "real", list(), 1L, 1L, source_sha256 = "current"
  ))

  env$analysis2_call <- function(pkg, name, file, ...) {
    if (identical(file, "train")) manifest() else manifest(scaler = "different")
  }
  expect_error(
    env$analysis2_validate_canonical_tfrecord_pair(
      "ndmdatasets", paths, "sim", list(), 1L, 1L
    ),
    "different scalers"
  )

  env$analysis2_call <- function(pkg, name, file, ...) manifest(source = "stale")
  expect_error(
    env$analysis2_validate_canonical_tfrecord_pair(
      "ndmdatasets", paths, "real", list(), 1L, 1L, source_sha256 = "current"
    ),
    "different source tables"
  )
})
