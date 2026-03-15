ndm_test_real_tfrecord_examples <- function() {
  list(
    list(
      XPred = array(c(0.1, 0.2, 0.3, 0.4), dim = c(2, 2)),
      XPred_mask = array(c(TRUE, TRUE, FALSE, TRUE), dim = c(2, 2)),
      Context = array(c(1, 2, 3), dim = c(1, 3)),
      Context_mask = array(c(TRUE, TRUE, TRUE), dim = c(1, 3)),
      location_id_numeric = array(1L, dim = c(1)),
      time_id_numeric = array(5L, dim = c(1)),
      YTrue = array(c(0.5, 0.6), dim = c(2, 1)),
      YTrue_mask = array(c(TRUE, TRUE), dim = c(2, 1)),
      YTrue_out = array(c(0.7, 0.8), dim = c(2, 1)),
      YTrue_out_mask = array(c(TRUE, FALSE), dim = c(2, 1))
    ),
    list(
      XPred = array(c(1.1, 1.2, 1.3, 1.4), dim = c(2, 2)),
      XPred_mask = array(c(TRUE, FALSE, TRUE, TRUE), dim = c(2, 2)),
      Context = array(c(4, 5, 6), dim = c(1, 3)),
      Context_mask = array(c(TRUE, FALSE, TRUE), dim = c(1, 3)),
      location_id_numeric = array(2L, dim = c(1)),
      time_id_numeric = array(7L, dim = c(1)),
      YTrue = array(c(1.5, 1.6), dim = c(2, 1)),
      YTrue_mask = array(c(TRUE, TRUE), dim = c(2, 1)),
      YTrue_out = array(c(1.7, 1.8), dim = c(2, 1)),
      YTrue_out_mask = array(c(TRUE, TRUE), dim = c(2, 1))
    )
  )
}

ndm_test_write_real_tfrecord <- function(file, examples, tensorflow) {
  tf <- tensorflow
  writer <- tf$io$TFRecordWriter(file)
  on.exit(writer$close(), add = TRUE)

  tf_dtype <- function(x) {
    if (is.logical(x)) {
      return(tf$bool)
    }
    if (is.integer(x)) {
      return(tf$int32)
    }
    tf$float32
  }

  for (example in examples) {
    feature <- list()
    for (name in names(example)) {
      value <- example[[name]]
      tensor <- tf$constant(value, dtype = tf_dtype(value))
      feature[[name]] <- tf$train$Feature(
        bytes_list = tf$train$BytesList(
          value = list(tf$io$serialize_tensor(tensor)$numpy())
        )
      )
      feature[[paste0(name, "_shape")]] <- tf$train$Feature(
        bytes_list = tf$train$BytesList(
          value = list(
            tf$io$serialize_tensor(
              tf$constant(as.integer(dim(value)), dtype = tf$int32)
            )$numpy()
          )
        )
      )
    }

    record <- tf$train$Example(
      features = tf$train$Features(feature = feature)
    )
    writer$write(record$SerializeToString())
  }

  invisible(file)
}

test_that("dtype maps follow the legacy TFRecord conventions", {
  real_map <- ndm_tfrecord_dtype_map("real")
  sim_map <- ndm_tfrecord_dtype_map("sim")

  expect_setequal(names(real_map), ndm_tfrecord_fields_real())
  expect_setequal(names(sim_map), ndm_tfrecord_fields_sim())
  expect_equal(unname(real_map[c("XPred_mask", "Context_mask", "YTrue_mask", "YTrue_out_mask")]),
               rep("bool", 4))
  expect_equal(real_map[["location_id_numeric"]], "int32")
  expect_equal(real_map[["time_id_numeric"]], "int32")
})

test_that("parser creation is lazy with respect to TensorFlow imports", {
  parser <- ndm_create_tfrecord_parser(
    ndm_tfrecord_fields_real(),
    ndm_tfrecord_dtype_map("real")
  )

  expect_true(is.function(parser))
})

test_that("dataset reads fail with an actionable TensorFlow backend error", {
  tfrecord_file <- tempfile(fileext = ".tfrecord")
  file.create(tfrecord_file)
  on.exit(unlink(tfrecord_file), add = TRUE)

  local_mocked_bindings(
    .ndm_resolve_tensorflow = function(tensorflow = NULL) {
      stop("Build and initialize the backend with ndm_build_backend().", call. = FALSE)
    },
    .package = "ndm"
  )

  expect_error(
    ndm_read_tfrecord_dataset(
      file = tfrecord_file,
      batch_size = 1L,
      field_names = ndm_tfrecord_fields_real()
    ),
    "ndm_build_backend"
  )
})

test_that("batch packaging preserves the model input contract", {
  batch <- list(
    XPred = array(0, dim = c(2, 3, 4)),
    XPred_mask = array(TRUE, dim = c(2, 3, 4)),
    Context = array(1, dim = c(2, 5)),
    Context_mask = array(TRUE, dim = c(2, 5)),
    location_id_numeric = array(1L, dim = c(2, 1)),
    time_id_numeric = array(5L, dim = c(2, 1))
  )

  packaged <- ndm_batch_to_model_inputs(batch)

  expect_length(packaged, 4L)
  expect_equal(packaged[[1]][[1]], batch$XPred)
  expect_equal(packaged[[2]][[2]], batch$Context_mask)
  expect_equal(packaged[[3]][[1]], batch$location_id_numeric)
  expect_equal(packaged[[4]][[1]], batch$time_id_numeric)
})

test_that("TFRecord readers and bundle helpers round-trip synthetic batches", {
  conda_env <- ndm_require_backend_test_stack("TFRecord round-trip tests", packages = c("reticulate"))
  old_backend <- ndm:::ndm_env$backend
  on.exit(assign("backend", old_backend, envir = ndm:::ndm_env), add = TRUE)

  backend <- ndm_initialize_backend(
    conda_env = conda_env,
    float_type = "32",
    import_tensorflow = TRUE
  )
  tf <- backend$tf
  train_file <- tempfile(fileext = ".tfrecord")
  inference_file <- tempfile(fileext = ".tfrecord")
  on.exit(unlink(c(train_file, inference_file)), add = TRUE)

  examples <- ndm_test_real_tfrecord_examples()
  ndm_test_write_real_tfrecord(train_file, examples, tensorflow = tf)
  ndm_test_write_real_tfrecord(inference_file, rev(examples), tensorflow = tf)

  dataset <- ndm_read_tfrecord_dataset(
    file = train_file,
    batch_size = 2L,
    field_names = ndm_tfrecord_fields_real(),
    dtype_map = ndm_tfrecord_dtype_map("real"),
    tensorflow = tf
  )
  expect_false(is.null(dataset))

  batches <- ndm_collect_tfrecord_batches(
    file = train_file,
    batch_size = 2L,
    field_names = ndm_tfrecord_fields_real(),
    dtype_map = ndm_tfrecord_dtype_map("real"),
    max_batches = 1L,
    tensorflow = tf
  )
  expect_length(batches, 1L)

  batch_r <- ndm_tf_batch_to_r(batches[[1]])
  expect_equal(dim(batch_r$XPred), c(2L, 2L, 2L))
  expect_equal(as.integer(batch_r$location_id_numeric[, 1]), c(1L, 2L))

  batch_jax <- ndm_tf_batch_to_jax(batches[[1]], backend = backend)
  expect_equal(as.integer(unlist(reticulate::py_to_r(batch_jax$XPred$shape))), c(2L, 2L, 2L))

  bundle <- ndm_load_tfrecord_bundle(
    train_file = train_file,
    inference_file = inference_file,
    batch_size = 2L,
    kind = "real",
    tensorflow = tf
  )
  expect_s3_class(bundle, "ndm_tfrecord_bundle")

  env <- ndm_new_runtime_env()
  ndm_attach_tfrecord_bundle(
    env = env,
    bundle = bundle,
    backend = backend,
    calibration_source = "inference"
  )

  expect_true(exists("TFDataset_train", envir = env, inherits = FALSE))
  expect_true(exists("TFDataset_inference", envir = env, inherits = FALSE))
  expect_true(exists("batch_l_cal", envir = env, inherits = FALSE))
  expect_equal(get("nCovars_real", envir = env, inherits = FALSE), 2L)
  expect_equal(as.integer(unlist(reticulate::py_to_r(env$batch_l_cal$XPred$shape))), c(2L, 2L, 2L))
})
