#' Inspect TFRecord field contracts
#'
#' These helpers describe the field names and default TensorFlow dtypes expected
#' by the packaged TFRecord readers.
#'
#' @returns `ndm_tfrecord_fields_real()` and `ndm_tfrecord_fields_sim()` return
#'   character vectors of field names. `ndm_tfrecord_dtype_map()` returns a
#'   named character vector mapping each field to its default TensorFlow dtype.
#'
#' @examples
#' ndm_tfrecord_fields_real()
#' ndm_tfrecord_dtype_map("sim")
#'
#' @export
ndm_tfrecord_fields_real <- function() {
  c(
    "XPred", "XPred_mask", "Context", "Context_mask",
    "location_id_numeric", "time_id_numeric",
    "YTrue", "YTrue_mask", "YTrue_out", "YTrue_out_mask"
  )
}

#' @rdname ndm_tfrecord_fields_real
#' @export
ndm_tfrecord_fields_sim <- function() {
  c(
    "XPred", "XPred_mask", "Context", "Context_mask",
    "location_id_numeric", "time_id_numeric",
    "YTrue", "YTrue_mask", "YTrue_out", "YTrue_out_mask",
    "inv_beta_true", "gamma_true", "sigma_true", "xi_true",
    "init_true", "XPred_pure", "YTrue_pure", "YTrue_out_pure",
    "PolicyDat_all", "initial_shift"
  )
}

#' @rdname ndm_tfrecord_fields_real
#'
#' @param kind TFRecord schema to inspect. Use `"real"` for observed-data
#'   records and `"sim"` for simulation records.
#'
#' @export
ndm_tfrecord_dtype_map <- function(kind = c("real", "sim")) {
  kind <- match.arg(kind)
  field_names <- switch(
    kind,
    real = ndm_tfrecord_fields_real(),
    sim = ndm_tfrecord_fields_sim()
  )

  dtype_map <- stats::setNames(rep("float32", length(field_names)), field_names)
  mask_fields <- grep("mask$", field_names, value = TRUE)
  dtype_map[mask_fields] <- "bool"
  dtype_map[c("location_id_numeric", "time_id_numeric")] <- "int32"
  dtype_map
}

.ndm_resolve_tf_dtype <- function(dtype, tf) {
  if ("python.builtin.object" %in% class(dtype)) {
    return(dtype)
  }

  if (!is.character(dtype) || length(dtype) != 1L) {
    stop("dtype values must be TensorFlow dtype objects or scalar character names.", call. = FALSE)
  }

  switch(
    dtype,
    float16 = tf$float16,
    float32 = tf$float32,
    float64 = tf$float64,
    int32 = tf$int32,
    int64 = tf$int64,
    bool = tf$bool,
    string = tf$string,
    stop("Unsupported TensorFlow dtype string '", dtype, "'.", call. = FALSE)
  )
}

.ndm_normalize_dtype_map <- function(field_names, dtype_map) {
  if (is.null(dtype_map)) {
    dtype_map <- stats::setNames(rep("float32", length(field_names)), field_names)
  }

  if (is.null(names(dtype_map)) || !all(field_names %in% names(dtype_map))) {
    stop("`dtype_map` must be a named vector or list covering every field name.", call. = FALSE)
  }

  dtype_map[field_names]
}

.ndm_resolve_tensorflow <- function(tensorflow = NULL) {
  if (!is.null(tensorflow)) {
    return(tensorflow)
  }

  if (!is.null(ndm_env$backend) && !is.null(ndm_env$backend$tf)) {
    return(ndm_env$backend$tf)
  }

  tf_available <- tryCatch(
    reticulate::py_module_available("tensorflow"),
    error = function(e) FALSE
  )
  if (isTRUE(tf_available)) {
    return(reticulate::import("tensorflow"))
  }

  stop(
    paste(
      "TensorFlow is required for TFRecord parsing but is not available in the active reticulate Python environment.",
      "Build and initialize the backend with ndm_build_backend(include_tensorflow = TRUE) and ndm_initialize_backend(import_tensorflow = TRUE),",
      "or point reticulate at a Python environment where the `tensorflow` module is installed."
    ),
    call. = FALSE
  )
}

#' Create and read TensorFlow TFRecord datasets
#'
#' These wrappers construct parsers for the ndm TFRecord schema and use
#' them to build TensorFlow datasets or eagerly collect batches into R.
#'
#' @details
#' TFRecord parsing requires TensorFlow to be importable from the active
#' reticulate environment. In practice this usually means provisioning the
#' backend with `ndm_build_backend(include_tensorflow = TRUE)` and then calling
#' `ndm_initialize_backend(import_tensorflow = TRUE)` for the same conda
#' environment before reading datasets.
#'
#' @param field_names Character vector of TFRecord field names that should be
#'   parsed from each example.
#' @param dtype_map Optional named vector or list mapping `field_names` to
#'   TensorFlow dtype names or TensorFlow dtype objects.
#' @param tensorflow Optional TensorFlow module. When omitted, ndm uses
#'   the active backend's TensorFlow module when available and otherwise
#'   imports TensorFlow from the active reticulate environment at call time.
#' @param file Path to the `.tfrecord` file to read.
#' @param batch_size Batch size used when constructing the TensorFlow dataset.
#' @param max_examples Optional maximum number of examples to consume from the
#'   file before batching.
#' @param shuffle Logical scalar indicating whether the dataset should be
#'   shuffled.
#' @param buffer_scaler Integer multiplier used to derive the TensorFlow shuffle
#'   buffer size from `batch_size`.
#' @param max_batches Optional maximum number of batches to collect when using
#'   `ndm_collect_tfrecord_batches()`.
#'
#' @returns `ndm_create_tfrecord_parser()` returns a parser function compatible
#'   with `tf$data$Dataset$map()`. `ndm_read_tfrecord_dataset()` returns a
#'   TensorFlow dataset object. `ndm_collect_tfrecord_batches()` returns a list
#'   of parsed batches.
#'
#' @examples
#' fields <- ndm_tfrecord_fields_real()
#' parser <- ndm_create_tfrecord_parser(fields, ndm_tfrecord_dtype_map("real"))
#' is.function(parser)
#'
#' @export
ndm_create_tfrecord_parser <- function(field_names,
                                       dtype_map = NULL,
                                       tensorflow = NULL) {
  if (length(field_names) == 0L) {
    stop("`field_names` must contain at least one entry.", call. = FALSE)
  }

  dtype_map <- .ndm_normalize_dtype_map(field_names, dtype_map)

  function(example_proto) {
    tf <- .ndm_resolve_tensorflow(tensorflow)
    feature_description <- list()
    for (name in field_names) {
      feature_description[[name]] <- tf$io$FixedLenFeature(list(), tf$string)
      feature_description[[paste0(name, "_shape")]] <- tf$io$FixedLenFeature(list(), tf$string)
    }

    example <- tf$io$parse_single_example(example_proto, feature_description)
    out <- list()

    for (name in field_names) {
      tensor <- tf$io$parse_tensor(
        example[[name]],
        out_type = .ndm_resolve_tf_dtype(dtype_map[[name]], tf)
      )
      shape <- tf$io$parse_tensor(
        example[[paste0(name, "_shape")]],
        out_type = tf$int32
      )
      out[[name]] <- tf$reshape(tensor, shape)
    }

    out
  }
}

#' @rdname ndm_create_tfrecord_parser
#' @export
ndm_read_tfrecord_dataset <- function(file,
                                      batch_size,
                                      field_names,
                                      dtype_map = NULL,
                                      max_examples = NULL,
                                      shuffle = FALSE,
                                      buffer_scaler = 10L,
                                      tensorflow = NULL) {
  tf <- .ndm_resolve_tensorflow(tensorflow)
  parser <- ndm_create_tfrecord_parser(
    field_names = field_names,
    dtype_map = dtype_map,
    tensorflow = tf
  )

  file <- .ndm_normalize_path(file, must_work = TRUE)
  dataset <- tf$data$TFRecordDataset(file)
  dataset <- dataset$map(parser)

  if (!is.null(max_examples)) {
    dataset <- dataset$take(as.integer(max_examples))
  }

  if (isTRUE(shuffle)) {
    dataset <- dataset$shuffle(
      buffer_size = tf$constant(as.integer(buffer_scaler * batch_size), dtype = tf$int64),
      reshuffle_each_iteration = FALSE
    )
  }

  dataset$batch(as.integer(batch_size))
}

#' @rdname ndm_create_tfrecord_parser
#' @export
ndm_collect_tfrecord_batches <- function(file,
                                         batch_size,
                                         field_names,
                                         dtype_map = NULL,
                                         max_batches = NULL,
                                         max_examples = NULL,
                                         shuffle = FALSE,
                                         buffer_scaler = 10L,
                                         tensorflow = NULL) {
  if (!is.null(max_batches) && is.null(max_examples)) {
    max_examples <- as.integer(max_batches * batch_size)
  }

  dataset <- ndm_read_tfrecord_dataset(
    file = file,
    batch_size = batch_size,
    field_names = field_names,
    dtype_map = dtype_map,
    max_examples = max_examples,
    shuffle = shuffle,
    buffer_scaler = buffer_scaler,
    tensorflow = tensorflow
  )

  iterator <- reticulate::as_iterator(dataset)
  batches <- list()
  batch_idx <- 0L

  repeat {
    batch <- tryCatch(
      reticulate::iter_next(iterator),
      error = function(e) NULL
    )

    if (is.null(batch)) {
      break
    }

    batch_idx <- batch_idx + 1L
    batches[[batch_idx]] <- batch

    if (!is.null(max_batches) && batch_idx >= max_batches) {
      break
    }
  }

  batches
}

.ndm_tf_value_to_r <- function(x) {
  if ("python.builtin.object" %in% class(x) &&
      isTRUE(reticulate::py_has_attr(x, "numpy"))) {
    return(reticulate::py_to_r(x$numpy()))
  }
  reticulate::py_to_r(x)
}

#' Convert and bundle TFRecord batches
#'
#' These helpers convert TensorFlow batches to native R or JAX objects, package
#' them into the shape expected by the package-managed runtime, and attach
#' dataset handles to a runtime environment.
#'
#' @param batch A parsed TFRecord batch or a nested list containing TensorFlow
#'   tensors.
#' @param backend Optional `ndm_backend` object. When omitted,
#'   `ndm_tf_batch_to_jax()` and `ndm_attach_tfrecord_bundle()` use the active
#'   backend from `ndm_backend_modules()` when available.
#' @param train_file Path to the training TFRecord file.
#' @param inference_file Optional path to an inference TFRecord file.
#' @param batch_size Batch size used when constructing the training and
#'   inference datasets.
#' @param kind TFRecord schema to use when reading `train_file` and
#'   `inference_file`.
#' @param dtype_map Optional named dtype mapping passed through to the TFRecord
#'   readers.
#' @param tensorflow Optional TensorFlow module. When omitted, ndm uses the
#'   active backend's TensorFlow module when available and otherwise imports
#'   TensorFlow from the active reticulate environment at call time.
#' @param max_train_examples Optional cap on the number of training examples to
#'   read.
#' @param max_inference_examples Optional cap on the number of inference
#'   examples to read.
#' @param shuffle_train Logical scalar indicating whether the training dataset
#'   should be shuffled.
#' @param shuffle_inference Logical scalar indicating whether the inference
#'   dataset should be shuffled.
#' @param env Runtime environment that should receive dataset handles and the
#'   preview calibration batch.
#' @param bundle An object of class `ndm_tfrecord_bundle` returned by
#'   `ndm_load_tfrecord_bundle()`.
#' @param calibration_source Which dataset should supply the preview batch used
#'   to seed runtime globals such as `batch_l_cal`.
#'
#' @returns `ndm_tf_batch_to_r()` recursively converts a TensorFlow batch to R
#'   arrays and lists. `ndm_tf_batch_to_jax()` recursively converts tensors into
#'   JAX arrays. `ndm_batch_to_model_inputs()` returns the packaged four-element
#'   list expected by the package-managed model stages. `ndm_load_tfrecord_bundle()`
#'   returns an `ndm_tfrecord_bundle` object. `ndm_attach_tfrecord_bundle()`
#'   invisibly returns `env`.
#'
#' @examples
#' batch <- list(
#'   XPred = array(0, c(2, 3, 4)),
#'   XPred_mask = array(TRUE, c(2, 3, 4)),
#'   Context = array(0, c(2, 1, 4)),
#'   Context_mask = array(TRUE, c(2, 1, 4)),
#'   location_id_numeric = matrix(1L, nrow = 2),
#'   time_id_numeric = matrix(1L, nrow = 2)
#' )
#' packaged <- ndm_batch_to_model_inputs(batch)
#' length(packaged)
#'
#' @rdname ndm_tf_batch_to_r
#' @export
ndm_tf_batch_to_r <- function(batch) {
  if (is.list(batch)) {
    return(lapply(batch, ndm_tf_batch_to_r))
  }
  .ndm_tf_value_to_r(batch)
}

#' @rdname ndm_tf_batch_to_r
#' @export
ndm_tf_batch_to_jax <- function(batch, backend = NULL) {
  backend <- backend %||% ndm_backend_modules()
  if (is.list(batch)) {
    return(lapply(batch, ndm_tf_batch_to_jax, backend = backend))
  }
  value <- .ndm_tf_value_to_r(batch)
  if (identical(backend$default_backend, "cpu")) {
    return(backend$jnp$array(value))
  }
  backend$send2gpu(value)
}

#' @rdname ndm_tf_batch_to_r
#' @export
ndm_batch_to_model_inputs <- function(batch) {
  required <- c("XPred", "XPred_mask", "Context", "Context_mask", "location_id_numeric", "time_id_numeric")
  missing <- setdiff(required, names(batch))
  if (length(missing) > 0L) {
    stop("Missing required batch fields: ", paste(missing, collapse = ", "), call. = FALSE)
  }

  list(
    list(batch$XPred, batch$XPred_mask),
    list(batch$Context, batch$Context_mask),
    list(batch$location_id_numeric),
    list(batch$time_id_numeric)
  )
}

#' @rdname ndm_tf_batch_to_r
#' @export
ndm_load_tfrecord_bundle <- function(train_file,
                                     inference_file = NULL,
                                     batch_size,
                                     kind = c("real", "sim"),
                                     dtype_map = NULL,
                                     tensorflow = NULL,
                                     max_train_examples = NULL,
                                     max_inference_examples = NULL,
                                     shuffle_train = TRUE,
                                     shuffle_inference = FALSE) {
  kind <- match.arg(kind)
  field_names <- switch(
    kind,
    real = ndm_tfrecord_fields_real(),
    sim = ndm_tfrecord_fields_sim()
  )
  dtype_map <- dtype_map %||% ndm_tfrecord_dtype_map(kind)

  train_dataset <- ndm_read_tfrecord_dataset(
    file = train_file,
    batch_size = batch_size,
    field_names = field_names,
    dtype_map = dtype_map,
    max_examples = max_train_examples,
    shuffle = shuffle_train,
    tensorflow = tensorflow
  )

  inference_dataset <- NULL
  if (!is.null(inference_file)) {
    inference_dataset <- ndm_read_tfrecord_dataset(
      file = inference_file,
      batch_size = batch_size,
      field_names = field_names,
      dtype_map = dtype_map,
      max_examples = max_inference_examples,
      shuffle = shuffle_inference,
      tensorflow = tensorflow
    )
  }

  structure(
    list(
      kind = kind,
      batch_size = as.integer(batch_size),
      field_names = field_names,
      dtype_map = dtype_map,
      train_file = .ndm_normalize_path(train_file, must_work = TRUE),
      inference_file = if (is.null(inference_file)) NULL else .ndm_normalize_path(inference_file, must_work = TRUE),
      train_dataset = train_dataset,
      train_iterator = reticulate::as_iterator(train_dataset),
      inference_dataset = inference_dataset,
      inference_iterator = if (is.null(inference_dataset)) NULL else reticulate::as_iterator(inference_dataset)
    ),
    class = c("ndm_tfrecord_bundle", "list")
  )
}

#' @rdname ndm_tf_batch_to_r
#' @export
ndm_attach_tfrecord_bundle <- function(env,
                                       bundle,
                                       backend = NULL,
                                       calibration_source = c("inference", "train")) {
  if (!is.environment(env)) {
    stop("`env` must be an environment.", call. = FALSE)
  }
  if (!inherits(bundle, "ndm_tfrecord_bundle")) {
    stop("`bundle` must inherit from class 'ndm_tfrecord_bundle'.", call. = FALSE)
  }

  calibration_source <- match.arg(calibration_source)
  backend <- backend %||% tryCatch(ndm_backend_modules(), error = function(e) NULL)

  assign("TFDataset_train", bundle$train_dataset, envir = env)
  assign("TFDatasetIterator_train", reticulate::as_iterator(bundle$train_dataset), envir = env)

  if (!is.null(bundle$inference_dataset)) {
    assign("TFDataset_inference", bundle$inference_dataset, envir = env)
    assign("TFDatasetIterator_inference", reticulate::as_iterator(bundle$inference_dataset), envir = env)
  }
  assign("ndm_tfrecord_bundle_ref", .ndm_bundle_ref_from_bundle(bundle), envir = env)

  preview_dataset <- if (identical(calibration_source, "inference") && !is.null(bundle$inference_dataset)) {
    bundle$inference_dataset
  } else {
    bundle$train_dataset
  }
  preview_batch <- reticulate::iter_next(reticulate::as_iterator(preview_dataset))

  if (!is.null(backend)) {
    preview_batch <- ndm_tf_batch_to_jax(preview_batch, backend = backend)
  }

  assign("batch_l_cal", preview_batch, envir = env)
  if ("XPred" %in% names(preview_batch)) {
    assign("jax_batchx", preview_batch$XPred, envir = env)
  }
  if ("YTrue" %in% names(preview_batch)) {
    assign("jax_batchy", preview_batch$YTrue, envir = env)
  }
  if ("XPred" %in% names(preview_batch)) {
    preview_shape <- try(reticulate::py_to_r(preview_batch$XPred$shape), silent = TRUE)
    if (!inherits(preview_shape, "try-error") && length(preview_shape) >= 3L) {
      assign("nCovars_real", as.integer(preview_shape[[3]]), envir = env)
    }
  }

  invisible(env)
}
