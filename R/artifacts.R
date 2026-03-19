#' Save and restore model artifacts
#'
#' These helpers persist the public `ndm_model` / `ndm_trained_model` objects to
#' a versioned artifact directory, restore them into a fresh runtime
#' environment, and resume training from the restored optimizer state.
#'
#' @details
#' These helpers assume `ndm_initialize_backend()` has already run for the
#' active conda environment. `ndm_save_model()` only works for runtime-backed
#' `ndm_model` / `ndm_trained_model` objects whose environments still contain
#' the built model state, including `ModelList` and `state`. Exact resume also
#' requires saved optimizer state. When `bundle = NULL` in
#' `ndm_resume_training()`, the artifact metadata must already contain a
#' recorded TFRecord bundle reference. `ndm_load_model()` and training-restore
#' paths require the `rrapply` package.
#'
#' @param x An object of class `ndm_model` or `ndm_trained_model`.
#' @param path Artifact directory path.
#' @param bundle Optional `ndm_tfrecord_bundle` used to record TFRecord
#'   references for later resume workflows.
#' @param overwrite Logical scalar indicating whether an existing artifact
#'   directory should be replaced.
#' @param calibration_source Which dataset should supply the preview batch used
#'   to seed runtime globals such as `batch_l_cal` when `bundle` is attached.
#' @param train_globals Optional named list of runtime globals applied
#'   immediately before `ndm_resume_training()` continues the training loop.
#'
#' @returns `ndm_save_model()` returns the normalized artifact directory path,
#'   invisibly. `ndm_load_model()` returns an `ndm_model` or
#'   `ndm_trained_model` matching the saved artifact. `ndm_resume_training()`
#'   returns an `ndm_trained_model`.
#'
#' @examples
#' \dontrun{
#' # After ndm_initialize_backend() and ndm_train() / ndm_fit():
#' artifact_dir <- ndm_save_model(trained_model, tempfile("ndm-artifact-"))
#' restored <- ndm_load_model(artifact_dir, bundle = tf_bundle)
#' resumed <- ndm_resume_training(artifact_dir, bundle = tf_bundle)
#' }
#'
#' @export
ndm_save_model <- function(x,
                           path,
                           bundle = NULL,
                           overwrite = FALSE) {
  if (!inherits(x, "ndm_model") && !inherits(x, "ndm_trained_model")) {
    stop("`x` must inherit from class 'ndm_model' or 'ndm_trained_model'.", call. = FALSE)
  }

  runtime_env <- .ndm_runtime_env_from_object(x)
  .ndm_require_runtime_bindings(runtime_env, c("ModelList", "state"), context = "ndm_save_model()")
  .ndm_require_backend_for_artifacts("ndm_save_model()")

  files <- .ndm_prepare_artifact_dir(path, overwrite = overwrite)
  bundle_ref <- .ndm_resolve_artifact_bundle_ref(runtime_env, bundle = bundle)
  training_state <- if (inherits(x, "ndm_trained_model")) {
    .ndm_collect_training_state(runtime_env)
  } else {
    list()
  }
  metadata <- .ndm_build_artifact_metadata(
    x = x,
    runtime_env = runtime_env,
    bundle_ref = bundle_ref,
    has_training_state = length(training_state) > 0L
  )

  saveRDS(metadata, file = files$metadata)
  saveRDS(.ndm_collect_runtime_snapshot(runtime_env), file = files$runtime)
  if (length(training_state) > 0L) {
    saveRDS(training_state, file = files$training)
  }
  .ndm_write_artifact_checkpoint(runtime_env, metadata = metadata, files = files)
  invisible(files$root)
}

#' @rdname ndm_save_model
#' @export
ndm_load_model <- function(path,
                           bundle = NULL,
                           calibration_source = c("inference", "train")) {
  calibration_source <- match.arg(calibration_source)
  .ndm_require_backend_for_artifacts("ndm_load_model()")

  files <- .ndm_artifact_files(path)
  metadata <- .ndm_read_artifact_metadata(files)
  runtime_env <- ndm_prepare_runtime(
    config = metadata$config,
    runtime_env = ndm_new_runtime_env()
  )

  ndm_set_runtime_globals(runtime_env, .ndm_collect_artifact_globals(files$runtime))
  ndm_build_model(
    runtime_env = runtime_env,
    model_type = metadata$model_type,
    model_spec = metadata$model_spec,
    backbone = metadata$backbone
  )

  if (isTRUE(metadata$has_opt_state)) {
    .ndm_prepare_training_restore_skeleton(runtime_env)
  }
  .ndm_restore_artifact_checkpoint(runtime_env, metadata = metadata, files = files)

  if (isTRUE(metadata$has_training_state) && file.exists(files$training)) {
    .ndm_restore_training_state(
      runtime_env,
      training_state = readRDS(files$training),
      artifact_root = files$root
    )
  }

  if (!is.null(bundle)) {
    ndm_attach_tfrecord_bundle(
      env = runtime_env,
      bundle = bundle,
      calibration_source = calibration_source
    )
  }

  .ndm_artifact_object_from_runtime(runtime_env, metadata = metadata, artifact_root = files$root)
}

#' @rdname ndm_save_model
#' @export
ndm_resume_training <- function(path,
                                bundle = NULL,
                                calibration_source = c("train", "inference"),
                                train_globals = list()) {
  calibration_source <- match.arg(calibration_source)
  files <- .ndm_artifact_files(path)
  metadata <- .ndm_read_artifact_metadata(files)

  if (is.null(bundle)) {
    bundle <- .ndm_bundle_from_artifact_ref(metadata$bundle_ref)
  }

  restored <- ndm_load_model(
    path = files$root,
    bundle = bundle,
    calibration_source = calibration_source
  )

  if (!inherits(restored, "ndm_trained_model") || is.null(restored$opt_state)) {
    stop(
      "This artifact does not contain optimizer state and cannot be resumed exactly.",
      call. = FALSE
    )
  }

  if (length(train_globals) > 0L) {
    ndm_set_runtime_globals(restored$env, train_globals)
  }

  ndm_train(
    x = restored,
    run_define = FALSE,
    run_loop = TRUE
  )
}

.ndm_artifact_version <- function() {
  1L
}

.ndm_artifact_files <- function(path) {
  root <- normalizePath(path, winslash = "/", mustWork = FALSE)
  list(
    root = root,
    metadata = file.path(root, "metadata.rds"),
    runtime = file.path(root, "runtime_globals.rds"),
    training = file.path(root, "training_state.rds"),
    model_state = file.path(root, "model_state.eqx"),
    aux_state = file.path(root, "aux_state.rds")
  )
}

.ndm_prepare_artifact_dir <- function(path, overwrite = FALSE) {
  files <- .ndm_artifact_files(path)
  if (file.exists(files$root)) {
    if (!isTRUE(overwrite)) {
      stop("Artifact path already exists: ", files$root, call. = FALSE)
    }
    unlink(files$root, recursive = TRUE, force = TRUE)
  }

  dir.create(files$root, recursive = TRUE, showWarnings = FALSE)
  files
}

.ndm_artifact_runtime_exclusions <- function() {
  c(
    "backend", "jax", "jnp", "np", "optax", "eq", "diffrax", "flash_mha",
    "py_gc", "tf", "oryx", "send2cpu", "send2gpu", "SoftPlus", "Sigmoid",
    "InvSoftPlus", "switch_filter_jit", "JaxKey", "jaxFloatType",
    "DefaultDtypeTf", "force2GPU", "ReSaveTfRecords", "NDM_INTERNAL_ANALYSIS_DIR",
    "ModelList", "state", "opt_state", "PriorList", "PolicyList",
    "GetPredSaveAtInfo_default", "GetPred_inference", "GetPred_train_jit",
    "getLoss_train", "gradLoss_jax", "TFDataset_train", "TFDatasetIterator_train",
    "TFDataset_inference", "TFDatasetIterator_inference", "batch_l_cal",
    "jax_batchx", "jax_batchy", "ndm_tfrecord_bundle_ref", "SavedModelDir"
  )
}

.ndm_artifact_training_state_names <- function() {
  c(
    "saveCheckpointCounter", "outSampCounter", "grad_norm_vec", "out_loss_vec",
    "in_loss_vec", "i_", "grad_norm_mat", "LastOutCor", "Skill8SanityCheck",
    "te_total", "te_grads", "st0", "plottingSeq_counter",
    "ExecuteUpdateCounter", "crossIterCor_vec", "values3_past",
    "tmpSamp_minus1", "tmpSamp_t0", "sharedXSamp"
  )
}

.ndm_collect_runtime_snapshot <- function(runtime_env) {
  names_all <- ls(runtime_env, all.names = TRUE)
  keep <- setdiff(
    names_all,
    unique(c(.ndm_artifact_runtime_exclusions(), .ndm_artifact_training_state_names()))
  )
  if (length(keep) == 0L) {
    return(list())
  }

  values <- mget(keep, envir = runtime_env, inherits = FALSE)
  values[!vapply(values, is.function, logical(1))]
}

.ndm_artifact_last_completed_iteration <- function(runtime_env) {
  if (exists("in_loss_vec", envir = runtime_env, inherits = FALSE)) {
    observed <- suppressWarnings(as.numeric(get("in_loss_vec", envir = runtime_env, inherits = FALSE)))
    finished <- which(is.finite(observed))
    if (length(finished) > 0L) {
      return(as.integer(max(finished)))
    }
  }

  current_i <- .ndm_runtime_get0(runtime_env, "i_", ifnotfound = 1L)
  as.integer(max(0L, suppressWarnings(as.integer(current_i)) - 1L))
}

.ndm_collect_training_state <- function(runtime_env) {
  present <- intersect(
    .ndm_artifact_training_state_names(),
    ls(runtime_env, all.names = TRUE)
  )
  state <- if (length(present) > 0L) {
    mget(present, envir = runtime_env, inherits = FALSE)
  } else {
    list()
  }

  last_completed <- .ndm_artifact_last_completed_iteration(runtime_env)
  state$last_completed_iteration <- as.integer(last_completed)
  state$resume_iteration <- as.integer(last_completed + 1L)
  state
}

.ndm_build_artifact_metadata <- function(x,
                                         runtime_env,
                                         bundle_ref,
                                         has_training_state) {
  has_opt_state <- exists("opt_state", envir = runtime_env, inherits = FALSE) &&
    !is.null(get("opt_state", envir = runtime_env, inherits = FALSE))
  has_scalers <- all(c("SIM_GLOBAL_SCALE_MEAN", "SIM_GLOBAL_SCALE_SD") %in%
                       ls(runtime_env, all.names = TRUE))
  has_aux_state <- any(c("ModelList_notshared_set", "sim_dat_norm_list") %in%
                         ls(runtime_env, all.names = TRUE))

  list(
    version = .ndm_artifact_version(),
    saved_class = if (inherits(x, "ndm_trained_model")) "ndm_trained_model" else "ndm_model",
    config = .ndm_runtime_get0(runtime_env, "ndm_config", ifnotfound = x$config %||% NULL),
    model_spec = .ndm_runtime_get0(runtime_env, "ndm_model_spec", ifnotfound = x$model_spec %||% NULL),
    model_type = .ndm_runtime_get0(runtime_env, "ndm_model_type", ifnotfound = x$model_type %||% NULL),
    backbone = .ndm_runtime_get0(runtime_env, "ndm_backbone", ifnotfound = x$backbone %||% NULL),
    data_generator = .ndm_runtime_get0(runtime_env, "ndm_data_generator", ifnotfound = x$data_generator %||% NULL),
    bundle_ref = bundle_ref,
    has_opt_state = isTRUE(has_opt_state),
    has_scalers = isTRUE(has_scalers),
    has_aux_state = isTRUE(has_aux_state),
    has_training_state = isTRUE(has_training_state)
  )
}

.ndm_collect_artifact_globals <- function(path) {
  if (!file.exists(path)) {
    return(list())
  }
  readRDS(path)
}

.ndm_write_artifact_checkpoint <- function(runtime_env, metadata, files) {
  eq <- .ndm_runtime_get0(runtime_env, "eq", ifnotfound = ndm_backend_modules()$eq)
  jnp <- .ndm_runtime_get0(runtime_env, "jnp", ifnotfound = ndm_backend_modules()$jnp)

  payload <- list(
    get("ModelList", envir = runtime_env, inherits = FALSE),
    get("state", envir = runtime_env, inherits = FALSE)
  )
  if (isTRUE(metadata$has_opt_state)) {
    payload[[length(payload) + 1L]] <- get("opt_state", envir = runtime_env, inherits = FALSE)
  }
  if (isTRUE(metadata$has_scalers)) {
    payload[[length(payload) + 1L]] <- jnp$array(list(
      get("SIM_GLOBAL_SCALE_MEAN", envir = runtime_env, inherits = FALSE),
      get("SIM_GLOBAL_SCALE_SD", envir = runtime_env, inherits = FALSE)
    ))
  }
  eq$tree_serialise_leaves(files$model_state, payload)

  if (isTRUE(metadata$has_aux_state)) {
    aux_names <- intersect(
      c("ModelList_notshared_set", "sim_dat_norm_list"),
      ls(runtime_env, all.names = TRUE)
    )
    if (length(aux_names) > 0L) {
      saveRDS(
        mget(aux_names, envir = runtime_env, inherits = FALSE),
        file = files$aux_state
      )
    }
  }

  invisible(files$model_state)
}

.ndm_require_backend_for_artifacts <- function(context) {
  tryCatch(
    ndm_backend_modules(),
    error = function(e) {
      stop(
        context,
        " requires an initialized backend. Run ndm_initialize_backend() first.",
        call. = FALSE
      )
    }
  )
}

.ndm_read_artifact_metadata <- function(files) {
  if (!file.exists(files$metadata)) {
    stop("Artifact metadata file not found: ", files$metadata, call. = FALSE)
  }

  metadata <- readRDS(files$metadata)
  required <- c("version", "saved_class", "config", "model_type", "backbone")
  missing <- setdiff(required, names(metadata))
  if (length(missing) > 0L) {
    stop(
      "Artifact metadata is missing required fields: ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }
  if (!identical(metadata$version, .ndm_artifact_version())) {
    stop(
      "Unsupported artifact version: ",
      metadata$version,
      call. = FALSE
    )
  }
  if (!inherits(metadata$config, "ndm_config")) {
    stop("Artifact metadata does not contain a valid `ndm_config`.", call. = FALSE)
  }

  metadata
}

.ndm_prepare_training_restore_skeleton <- function(runtime_env) {
  .ndm_require_namespaces("rrapply", context = "ndm_load_model()")

  plot_file <- tempfile(fileext = ".pdf")
  grDevices::pdf(plot_file)
  on.exit({
    try(grDevices::dev.off(), silent = TRUE)
    unlink(plot_file)
  }, add = TRUE)

  .ndm_with_working_directory(tempdir(), function() {
    .ndm_source_runtime_file(ndm_runtime_paths()$train_define, runtime_env)
  })

  invisible(runtime_env)
}

.ndm_restore_artifact_checkpoint <- function(runtime_env, metadata, files) {
  if (!file.exists(files$model_state)) {
    stop("Artifact checkpoint file not found: ", files$model_state, call. = FALSE)
  }

  eq <- .ndm_runtime_get0(runtime_env, "eq", ifnotfound = ndm_backend_modules()$eq)
  jnp <- .ndm_runtime_get0(runtime_env, "jnp", ifnotfound = ndm_backend_modules()$jnp)
  like <- list(
    get("ModelList", envir = runtime_env, inherits = FALSE),
    get("state", envir = runtime_env, inherits = FALSE)
  )
  if (isTRUE(metadata$has_opt_state)) {
    like[[length(like) + 1L]] <- get("opt_state", envir = runtime_env, inherits = FALSE)
  }
  if (isTRUE(metadata$has_scalers)) {
    mean_like <- .ndm_runtime_get0(runtime_env, "SIM_GLOBAL_SCALE_MEAN", ifnotfound = 0)
    sd_like <- .ndm_runtime_get0(runtime_env, "SIM_GLOBAL_SCALE_SD", ifnotfound = 1)
    like[[length(like) + 1L]] <- jnp$array(list(mean_like, sd_like))
  }

  restored <- eq$tree_deserialise_leaves(files$model_state, like)
  assign("ModelList", restored[[1L]], envir = runtime_env)
  assign("state", restored[[2L]], envir = runtime_env)

  next_index <- 3L
  if (isTRUE(metadata$has_opt_state)) {
    assign("opt_state", restored[[next_index]], envir = runtime_env)
    next_index <- next_index + 1L
  }
  if (isTRUE(metadata$has_scalers)) {
    scalers <- restored[[next_index]]
    assign("SIM_GLOBAL_SCALE_MEAN", jnp$take(scalers, 0L, axis = 0L)$tolist(), envir = runtime_env)
    assign("SIM_GLOBAL_SCALE_SD", jnp$take(scalers, 1L, axis = 0L)$tolist(), envir = runtime_env)
  }

  if (isTRUE(metadata$has_aux_state) && file.exists(files$aux_state)) {
    ndm_set_runtime_globals(runtime_env, readRDS(files$aux_state))
  }

  invisible(runtime_env)
}

.ndm_restore_training_state <- function(runtime_env,
                                        training_state,
                                        artifact_root = NULL) {
  if (!is.list(training_state) || length(training_state) == 0L) {
    return(invisible(runtime_env))
  }

  values <- training_state[setdiff(
    names(training_state),
    c("last_completed_iteration", "resume_iteration")
  )]
  if (!is.null(artifact_root)) {
    checkpoint_dir <- file.path(artifact_root, "runtime_checkpoints")
    dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)
    values$SavedModelDir <- checkpoint_dir
  }
  ndm_set_runtime_globals(runtime_env, values)

  if ("resume_iteration" %in% names(training_state)) {
    assign("i_", as.integer(training_state$resume_iteration), envir = runtime_env)
  }

  invisible(runtime_env)
}

.ndm_resolve_artifact_bundle_ref <- function(runtime_env, bundle = NULL) {
  if (!is.null(bundle)) {
    return(.ndm_bundle_ref_from_bundle(bundle))
  }

  .ndm_runtime_get0(runtime_env, "ndm_tfrecord_bundle_ref", ifnotfound = NULL)
}

.ndm_bundle_ref_from_bundle <- function(bundle) {
  if (is.null(bundle)) {
    return(NULL)
  }
  if (!inherits(bundle, "ndm_tfrecord_bundle")) {
    stop("`bundle` must inherit from class 'ndm_tfrecord_bundle'.", call. = FALSE)
  }

  list(
    train_file = .ndm_normalize_path(bundle$train_file, must_work = TRUE),
    inference_file = if (is.null(bundle$inference_file)) {
      NULL
    } else {
      .ndm_normalize_path(bundle$inference_file, must_work = TRUE)
    },
    batch_size = bundle$batch_size,
    kind = bundle$kind,
    dtype_map = bundle$dtype_map
  )
}

.ndm_bundle_from_artifact_ref <- function(bundle_ref) {
  if (is.null(bundle_ref)) {
    stop(
      "No TFRecord bundle was supplied and the artifact does not contain bundle references.",
      call. = FALSE
    )
  }

  ndm_load_tfrecord_bundle(
    train_file = bundle_ref$train_file,
    inference_file = bundle_ref$inference_file,
    batch_size = bundle_ref$batch_size,
    kind = bundle_ref$kind,
    dtype_map = bundle_ref$dtype_map
  )
}

.ndm_artifact_object_from_runtime <- function(runtime_env,
                                              metadata,
                                              artifact_root) {
  common <- list(
    env = runtime_env,
    config = metadata$config,
    model_spec = metadata$model_spec,
    model_type = metadata$model_type,
    backbone = metadata$backbone,
    data_generator = metadata$data_generator,
    artifact_path = artifact_root,
    model = .ndm_runtime_get0(runtime_env, "ModelList", ifnotfound = NULL),
    state = .ndm_runtime_get0(runtime_env, "state", ifnotfound = NULL)
  )

  if (identical(metadata$saved_class, "ndm_trained_model")) {
    common$opt_state <- .ndm_runtime_get0(runtime_env, "opt_state", ifnotfound = NULL)
    return(structure(common, class = "ndm_trained_model"))
  }

  structure(common, class = "ndm_model")
}
