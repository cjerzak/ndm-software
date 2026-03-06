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
    }
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
