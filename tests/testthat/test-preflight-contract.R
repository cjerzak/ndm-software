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
  manifest <- function(scaler = "shared", source = "current", n_examples = 8L) {
    list(
      n_examples = n_examples,
      scaler_sha256 = scaler,
      metadata = list(source_sha256 = source)
    )
  }

  calls <- list()
  env$analysis2_call <- function(pkg, name, file, ...) {
    calls[[file]] <<- list(...)
    manifest(n_examples = if (identical(file, "inference")) 1L else 8L)
  }
  expect_silent(env$analysis2_validate_canonical_tfrecord_pair(
    "ndmdatasets", paths, "real", list(), 4L, 1L, source_sha256 = "current"
  ))
  expect_false("expected_n_examples" %in% names(calls$train))
  expect_equal(calls$inference$expected_n_examples, 1L)
  expect_true(calls$train$verify_checksum)
  expect_true(calls$inference$verify_checksum)

  env$analysis2_call <- function(pkg, name, file, ...) {
    if (identical(file, "train")) manifest(n_examples = 3L) else manifest()
  }
  expect_error(
    env$analysis2_validate_canonical_tfrecord_pair(
      "ndmdatasets", paths, "sim", list(), 4L, 1L
    ),
    "requires at least 4"
  )

  env$analysis2_call <- function(pkg, name, file, ...) {
    if (identical(file, "train")) {
      manifest()
    } else {
      manifest(scaler = "different", n_examples = 1L)
    }
  }
  expect_error(
    env$analysis2_validate_canonical_tfrecord_pair(
      "ndmdatasets", paths, "sim", list(), 1L, 1L
    ),
    "different scalers"
  )

  env$analysis2_call <- function(pkg, name, file, ...) {
    manifest(
      source = "stale",
      n_examples = if (identical(file, "inference")) 1L else 8L
    )
  }
  expect_error(
    env$analysis2_validate_canonical_tfrecord_pair(
      "ndmdatasets", paths, "real", list(), 1L, 1L, source_sha256 = "current"
    ),
    "different source tables"
  )
})

test_that("trusted controller indexes skip duplicate hashing only for the exact bundle", {
  env <- ndm:::.ndm_new_run_impl_env()
  tfrecord_dir <- tempfile("trusted-tfrecords-")
  dir.create(tfrecord_dir)
  on.exit(unlink(tfrecord_dir, recursive = TRUE, force = TRUE), add = TRUE)
  paths <- env$analysis2_tfrecord_paths(tfrecord_dir, 1L)
  artifact_files <- c(
    paths$train_file,
    env$analysis2_manifest_path(paths$train_file),
    paths$inference_file,
    env$analysis2_manifest_path(paths$inference_file)
  )
  writeBin(charToRaw("train-record"), paths$train_file)
  writeBin(charToRaw("inference-record"), paths$inference_file)
  for (record_file in c(paths$train_file, paths$inference_file)) {
    saveRDS(
      list(record = list(
        file = basename(record_file),
        bytes = as.numeric(file.info(record_file)$size),
        sha256 = digest::digest(
          file = record_file,
          algo = "sha256",
          serialize = FALSE
        )
      )),
      env$analysis2_manifest_path(record_file)
    )
  }

  index <- data.frame(
    BaseID = rep(1L, 4L),
    artifact_type = c("train", "train", "inference", "inference"),
    sidecar = c(FALSE, TRUE, FALSE, TRUE),
    relative_path = basename(artifact_files),
    bytes = as.numeric(file.info(artifact_files)$size),
    modified_utc = rep("2026-07-22T00:00:00Z", 4L),
    sha256 = vapply(
      artifact_files,
      function(file) digest::digest(
        file = file,
        algo = "sha256",
        serialize = FALSE
      ),
      character(1),
      USE.NAMES = FALSE
    ),
    stringsAsFactors = FALSE
  )
  index_file <- file.path(tfrecord_dir, "artifact_checksums.tsv")
  utils::write.table(
    index,
    index_file,
    sep = "\t",
    row.names = FALSE,
    quote = TRUE
  )

  env_names <- c(
    "ANALYSIS2_TRUSTED_TFRECORD_DIR",
    "ANALYSIS2_TRUSTED_ARTIFACT_INDEX_SHA256"
  )
  old_env <- Sys.getenv(env_names, unset = NA_character_)
  on.exit({
    missing <- is.na(old_env)
    if (any(missing)) {
      Sys.unsetenv(env_names[missing])
    }
    if (any(!missing)) {
      do.call(Sys.setenv, as.list(stats::setNames(old_env[!missing], env_names[!missing])))
    }
  }, add = TRUE)
  trusted_index_sha256 <- digest::digest(
    file = index_file,
    algo = "sha256",
    serialize = FALSE
  )
  Sys.setenv(
    ANALYSIS2_TRUSTED_TFRECORD_DIR = tfrecord_dir,
    ANALYSIS2_TRUSTED_ARTIFACT_INDEX_SHA256 = trusted_index_sha256
  )

  captured <- NULL
  env$analysis2_sim_dataset_spec <- function(...) list(n_inference_batches = 1L)
  env$analysis2_validate_canonical_tfrecord_pair <- function(...) {
    captured <<- list(...)
    list(train = list(scaler = list(mean = 1, sd = 2)))
  }
  preflight <- function() {
    env$analysis2_preflight_expected_tfrecords(
      mode = "sim",
      write_plan = data.frame(
        BaseID = 1L,
        canonical_row = 1L,
        artifact_n_samples_train = 8L
      ),
      grid = data.frame(BaseID = 1L),
      tfrecord_dir = tfrecord_dir,
      ndmdatasets_pkg = "ndmdatasets"
    )
  }

  expect_silent(preflight())
  expect_false(captured$verify_checksum)

  Sys.setenv(ANALYSIS2_TRUSTED_ARTIFACT_INDEX_SHA256 = strrep("b", 64L))
  expect_error(preflight(), "index SHA-256 does not match")
  Sys.setenv(ANALYSIS2_TRUSTED_ARTIFACT_INDEX_SHA256 = trusted_index_sha256)

  sidecar <- env$analysis2_manifest_path(paths$train_file)
  sidecar_raw <- readBin(sidecar, what = "raw", n = file.info(sidecar)$size)
  writeBin(as.raw(rep(0L, length(sidecar_raw))), sidecar)
  expect_error(preflight(), "sidecar SHA-256 disagrees")
  writeBin(sidecar_raw, sidecar)

  Sys.setenv(ANALYSIS2_TRUSTED_TFRECORD_DIR = tempdir())
  expect_error(preflight(), "does not match the trusted directory")

  Sys.setenv(ANALYSIS2_TRUSTED_TFRECORD_DIR = "")
  Sys.unsetenv("ANALYSIS2_TRUSTED_ARTIFACT_INDEX_SHA256")
  expect_error(preflight(), "missing or malformed environment markers")

  Sys.unsetenv(env_names)
  expect_silent(preflight())
  expect_true(captured$verify_checksum)
})
