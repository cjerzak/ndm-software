ndm_test_artifact_backend <- function() {
  list(
    eq = list(
      tree_serialise_leaves = function(path, object) {
        saveRDS(object, file = path)
        invisible(path)
      },
      tree_deserialise_leaves = function(path, like) {
        readRDS(path)
      }
    ),
    jnp = list(
      array = function(x) {
        structure(list(value = x), class = "ndm_fake_jax_array")
      },
      take = function(x, idx, axis = 0L) {
        list(
          tolist = function() {
            x$value[[as.integer(idx) + 1L]]
          }
        )
      }
    )
  )
}

ndm_test_artifact_batch <- function() {
  list(
    XPred = array(0, dim = c(2, 3, 4)),
    XPred_mask = array(TRUE, dim = c(2, 3, 4)),
    Context = array(1, dim = c(2, 5)),
    Context_mask = array(TRUE, dim = c(2, 5)),
    location_id_numeric = array(1L, dim = c(2, 1)),
    time_id_numeric = array(5L, dim = c(2, 1)),
    YTrue_out = array(2, dim = c(2, 3, 1)),
    YTrue_out_mask = array(TRUE, dim = c(2, 3, 1))
  )
}

ndm_test_artifact_env <- function(include_opt_state = TRUE) {
  backend <- ndm_test_artifact_backend()
  env <- ndm_new_runtime_env()
  env$eq <- backend$eq
  env$jnp <- backend$jnp
  env$ModelList <- list(model = TRUE)
  env$state <- list(stage = "trained")
  if (isTRUE(include_opt_state)) {
    env$opt_state <- list(opt = TRUE)
  }
  env$ndm_config <- ndm_create_config(
    model_type = "DecoderOnly",
    force_to_gpu = FALSE
  )
  env$ndm_model_spec <- ndm_model_spec(preset = "seirs_dynamic_beta")
  env$ndm_model_type <- "DecoderOnly"
  env$ndm_backbone <- "transformer"
  env$ndm_data_generator <- "sim"
  env$nTimesPast <- 8L
  env$nTimesLookahead <- 4L
  env$nBatch <- 2L
  env$SIM_GLOBAL_SCALE_MEAN <- 0
  env$SIM_GLOBAL_SCALE_SD <- 1
  env$in_loss_vec <- c(1, 0.5, NA_real_)
  env$grad_norm_vec <- c(0.8, 0.4, NA_real_)
  env$i_ <- 2L
  env$batch_l_cal <- ndm_test_artifact_batch()
  env$TFDataset_train <- "dataset"
  env$GetPred_inference <- function(...) NULL
  env$getLoss_train <- function(...) NULL
  env
}

test_that("prepare helpers stamp provenance on runtime environments", {
  env <- ndm_new_runtime_env()
  cfg <- ndm_create_config(
    model_type = "DecoderOnly",
    force_to_gpu = FALSE
  )

  local_mocked_bindings(
    ndm_load_runtime = function(...) invisible(env),
    ndm_source_runtime_data = function(...) invisible(env),
    .ndm_require_namespaces = function(...) invisible(TRUE),
    .package = "ndm"
  )

  ndm_prepare_runtime(
    config = cfg,
    runtime_env = env
  )
  ndm_prepare_data(
    runtime_env = env,
    generator = "sim"
  )

  expect_identical(env$ndm_config, cfg)
  expect_equal(env$ndm_data_generator, "sim")
})

test_that("ndm_save_model writes artifact metadata and filtered runtime globals", {
  env <- ndm_test_artifact_env(include_opt_state = TRUE)
  trained <- structure(
    list(
      env = env,
      config = env$ndm_config,
      model_spec = env$ndm_model_spec,
      model_type = env$ndm_model_type,
      backbone = env$ndm_backbone,
      data_generator = env$ndm_data_generator
    ),
    class = "ndm_trained_model"
  )
  train_file <- tempfile(fileext = ".tfrecord")
  inference_file <- tempfile(fileext = ".tfrecord")
  file.create(train_file, inference_file)
  bundle <- structure(
    list(
      train_file = train_file,
      inference_file = inference_file,
      batch_size = 2L,
      kind = "sim",
      dtype_map = ndm_tfrecord_dtype_map("sim")
    ),
    class = c("ndm_tfrecord_bundle", "list")
  )
  artifact_path <- tempfile("ndm-artifact-")

  local_mocked_bindings(
    ndm_backend_modules = function() {
      list(eq = env$eq, jnp = env$jnp)
    },
    .package = "ndm"
  )

  ndm_save_model(
    x = trained,
    path = artifact_path,
    bundle = bundle
  )

  metadata <- readRDS(file.path(artifact_path, "metadata.rds"))
  runtime_globals <- readRDS(file.path(artifact_path, "runtime_globals.rds"))
  training_state <- readRDS(file.path(artifact_path, "training_state.rds"))

  expect_equal(metadata$saved_class, "ndm_trained_model")
  expect_true(inherits(metadata$config, "ndm_config"))
  expect_equal(metadata$data_generator, "sim")
  expect_equal(metadata$bundle_ref$train_file, normalizePath(train_file, winslash = "/", mustWork = TRUE))
  expect_false("ModelList" %in% names(runtime_globals))
  expect_false("state" %in% names(runtime_globals))
  expect_false("opt_state" %in% names(runtime_globals))
  expect_false("batch_l_cal" %in% names(runtime_globals))
  expect_false("TFDataset_train" %in% names(runtime_globals))
  expect_equal(training_state$resume_iteration, 3L)
})

test_that("ndm_load_model rebuilds the runtime and restores checkpoint payloads", {
  save_env <- ndm_test_artifact_env(include_opt_state = FALSE)
  model <- structure(
    list(
      env = save_env,
      config = save_env$ndm_config,
      model_spec = save_env$ndm_model_spec,
      model_type = save_env$ndm_model_type,
      backbone = save_env$ndm_backbone,
      data_generator = save_env$ndm_data_generator
    ),
    class = "ndm_model"
  )
  artifact_path <- tempfile("ndm-load-artifact-")
  built <- new.env(parent = emptyenv())

  local_mocked_bindings(
    ndm_backend_modules = function() {
      list(eq = save_env$eq, jnp = save_env$jnp)
    },
    .package = "ndm"
  )
  ndm_save_model(model, artifact_path)

  local_mocked_bindings(
    ndm_backend_modules = function() {
      list(eq = save_env$eq, jnp = save_env$jnp)
    },
    ndm_prepare_runtime = function(config,
                                   runtime_env = ndm_new_runtime_env(),
                                   runtime_globals = list()) {
      built$config <- config
      runtime_env$eq <- save_env$eq
      runtime_env$jnp <- save_env$jnp
      runtime_env$ndm_config <- config
      runtime_env
    },
    ndm_build_model = function(runtime_env,
                               model_type = c("DecoderOnly", "NeuralODE"),
                               model_spec = NULL,
                               backbone = "transformer",
                               runtime_globals = list()) {
      built$model_type <- model_type
      built$model_spec <- model_spec
      built$backbone <- backbone
      runtime_env$ModelList <- list(model = "skeleton")
      runtime_env$state <- list(stage = "skeleton")
      runtime_env$ndm_model_spec <- model_spec
      runtime_env$ndm_model_type <- model_type
      runtime_env$ndm_backbone <- backbone
      structure(list(env = runtime_env), class = "ndm_model")
    },
    .package = "ndm"
  )

  restored <- ndm_load_model(artifact_path)

  expect_s3_class(restored, "ndm_model")
  expect_identical(built$config, save_env$ndm_config)
  expect_identical(built$model_spec$preset, save_env$ndm_model_spec$preset)
  expect_equal(built$model_type, "DecoderOnly")
  expect_equal(restored$state$stage, "trained")
  expect_equal(restored$env$nTimesPast, 8L)
})

test_that("ndm_resume_training auto-loads bundle refs and continues with run_define disabled", {
  artifact_path <- tempfile("ndm-resume-artifact-")
  dir.create(artifact_path, recursive = TRUE, showWarnings = FALSE)
  metadata <- list(
    version = 1L,
    saved_class = "ndm_trained_model",
    config = ndm_create_config(force_to_gpu = FALSE),
    model_type = "DecoderOnly",
    backbone = "transformer",
    bundle_ref = list(
      train_file = "/tmp/train.tfrecord",
      inference_file = "/tmp/inference.tfrecord",
      batch_size = 4L,
      kind = "sim",
      dtype_map = ndm_tfrecord_dtype_map("sim")
    ),
    has_opt_state = TRUE
  )
  saveRDS(metadata, file = file.path(artifact_path, "metadata.rds"))
  seen <- new.env(parent = emptyenv())

  local_mocked_bindings(
    ndm_load_tfrecord_bundle = function(...) {
      seen$bundle_args <- list(...)
      structure(
        list(train_file = "/tmp/train.tfrecord", batch_size = 4L, kind = "sim"),
        class = c("ndm_tfrecord_bundle", "list")
      )
    },
    ndm_load_model = function(path, bundle = NULL, calibration_source = c("inference", "train")) {
      seen$bundle <- bundle
      seen$calibration_source <- calibration_source
      structure(list(env = ndm_new_runtime_env(), opt_state = list(opt = TRUE)), class = "ndm_trained_model")
    },
    ndm_train = function(x, run_define = TRUE, run_loop = TRUE) {
      seen$run_define <- run_define
      seen$run_loop <- run_loop
      x
    },
    .package = "ndm"
  )

  resumed <- ndm_resume_training(
    path = artifact_path,
    train_globals = list(extra_flag = TRUE)
  )

  expect_s3_class(resumed, "ndm_trained_model")
  expect_equal(seen$bundle_args$train_file, "/tmp/train.tfrecord")
  expect_equal(seen$run_define, FALSE)
  expect_true(seen$run_loop)
})
