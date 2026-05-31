ndm_test_runtime_env <- function() {
  env <- ndm_new_runtime_env()
  env$ModelList <- list(model = TRUE)
  env$state <- list(stage = "built")
  env$PriorList <- list(prior = TRUE)
  env$PolicyList <- list(policy = TRUE)
  env$GetPredSaveAtInfo_default <- list(1L)
  env$jax <- list(
    random = list(
      split = function(key, n) {
        matrix(seq_len(as.integer(n)), ncol = 1L)
      }
    )
  )
  env$JaxKey <- function(seed) as.integer(seed)
  env$jnp <- list(
    array = function(x) x
  )
  env$GetPred_inference <- function(ModelList, x, state, PriorList, PolicyList, GetPredSaveAtInfo, seed) {
    list(list(kind = "inference", batch = x), list(stage = "predicted"))
  }
  env$GetPred_train_jit <- function(ModelList, x, state, PriorList, PolicyList, GetPredSaveAtInfo, seed) {
    list(list(kind = "train", batch = x), list(stage = "train-predicted"))
  }
  env$getLoss_train <- function(ModelList, x, y, y_mask, iteration, state, PriorList, PolicyList, GetPredSaveAtInfo, seed) {
    list(
      list(kind = "loss", iteration = iteration, y = y, y_mask = y_mask),
      list(stage = "loss-updated")
    )
  }
  env
}

ndm_test_named_batch <- function() {
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

test_that("predict and loss accept trained models", {
  env <- ndm_test_runtime_env()
  trained <- structure(list(env = env), class = "ndm_trained_model")
  batch <- ndm_test_named_batch()

  pred <- ndm_predict(trained, batch = batch)
  expect_equal(pred$kind, "inference")
  expect_equal(env$state$stage, "predicted")

  loss <- ndm_loss(trained, batch = batch)
  expect_equal(loss$kind, "loss")
  expect_equal(loss$iteration, 1)
  expect_equal(env$state$stage, "loss-updated")
})

test_that("predict rejects malformed packaged batches early", {
  env <- ndm_test_runtime_env()
  malformed <- list(
    list(array(0, dim = c(2, 3, 4))),
    list(array(TRUE, dim = c(2, 3, 4)), array(TRUE, dim = c(2, 3, 4))),
    list(array(1L, dim = c(2, 1))),
    list(array(5L, dim = c(2, 1)))
  )

  expect_error(
    ndm_predict(env, batch = malformed),
    "ndm_batch_to_model_inputs"
  )
})

test_that("predict uses batch_l_cal and supports the training prediction path", {
  env <- ndm_test_runtime_env()
  env$batch_l_cal <- ndm_test_named_batch()

  pred <- ndm_predict(
    env,
    batch = NULL,
    inference = FALSE,
    update_state = FALSE
  )

  expect_equal(pred$kind, "train")
  expect_equal(env$state$stage, "built")
})

test_that("loss accepts explicit targets and leaves state unchanged when requested", {
  env <- ndm_test_runtime_env()
  batch <- ndm_test_named_batch()
  explicit_y <- array(9, dim = c(2, 3, 1))
  explicit_mask <- array(FALSE, dim = c(2, 3, 1))

  loss <- ndm_loss(
    env,
    batch = batch,
    y = explicit_y,
    y_mask = explicit_mask,
    update_state = FALSE
  )

  expect_equal(loss$y, explicit_y)
  expect_equal(loss$y_mask, explicit_mask)
  expect_equal(env$state$stage, "built")
})

test_that("temporal latent extraction exposes decoder head inputs", {
  env <- ndm_test_runtime_env()
  env$ModelType <- "DecoderOnly"
  env$ModelDims <- 3L
  env$nOutcomes <- 1L
  decoder_head_input <- array(seq_len(12), dim = c(2, 2, 3))
  env$GetPred_inference <- function(ModelList, x, state, PriorList, PolicyList, GetPredSaveAtInfo, seed) {
    list(
      list(
        y_mu = array(1, dim = c(2, 2, 1)),
        y_sigma = array(1, dim = c(2, 2, 1)),
        TemporalLatents = list(decoder_head_input = decoder_head_input),
        ODEParamsSampList = list(
          center_param = array(1, dim = c(2, 2, 1)),
          scale_param = array(1, dim = c(2, 2, 1))
        )
      ),
      list(stage = "predicted")
    )
  }
  trained <- structure(list(env = env, model_type = "DecoderOnly"), class = "ndm_trained_model")

  latents <- ndm_extract_temporal_latents(
    trained,
    batch = ndm_test_named_batch(),
    update_state = FALSE
  )

  expect_s3_class(latents, "ndm_temporal_latents")
  expect_equal(dim(latents$decoder$head_input), c(2L, 2L, 3L))
  expect_equal(latents$time, 0:1)
  expect_equal(nrow(latents$feature_map$ode$local), 0L)
  decoder_matrix <- ndm_latent_matrix(latents, "decoder_head_input")
  expect_equal(dim(decoder_matrix), c(4L, 3L))
  expect_equal(colnames(decoder_matrix), paste0("decoder_head_input_", 1:3))
})

test_that("temporal latent extraction maps NeuralODE hidden and interpretable features", {
  env <- ndm_test_runtime_env()
  env$ModelType <- "NeuralODE"
  env$ModelDims <- 4L
  env$nOutcomes <- 1L
  env$LocalNeuralEmbedDim <- 2L
  env$GlobalNeuralEmbedDim <- 2L
  env$uq_encneural_vec <- c("beta_l", "p_l")
  env$uq_globalneural_vec <- "delta"
  env$ndm_model_spec <- ndm_model_spec(
    preset = "seirs_dynamic_beta_dynamic_global",
    model_type = "NeuralODE"
  )

  local_state <- array(seq(-3, 26), dim = c(2, 3, 5))
  global_state <- array(seq(-2, 15), dim = c(2, 3, 3))
  s_state <- array(seq_len(6), dim = c(2, 3))
  env$GetPred_inference <- function(ModelList, x, state, PriorList, PolicyList, GetPredSaveAtInfo, seed) {
    list(
      list(
        y_mu = array(1, dim = c(2, 3, 1)),
        y_sigma = array(1, dim = c(2, 3, 1)),
        ODEParamsSampList = list(
          "diff_eq_sol_ts" = 0:2,
          "diff_eq_sol_ys.Neural1" = local_state,
          "diff_eq_sol_ys.Neural2" = global_state,
          "diff_eq_sol_ys.s_l" = s_state,
          center_param = array(1, dim = c(2, 3, 1)),
          scale_param = array(1, dim = c(2, 3, 1))
        )
      ),
      list(stage = "predicted")
    )
  }
  trained <- structure(list(env = env, model_type = "NeuralODE"), class = "ndm_trained_model")

  latents <- ndm_extract_temporal_latents(
    trained,
    batch = ndm_test_named_batch(),
    update_state = FALSE
  )

  expect_equal(nrow(latents$feature_map$decoder$head_input), 0L)
  local_map <- latents$feature_map$ode$local
  expect_equal(local_map$index[match("beta_l", local_map$feature)], 3L)
  expect_equal(local_map$transform[match("beta_l", local_map$feature)], "InvSoftPlus")
  expect_equal(local_map$index[match("p_l", local_map$feature)], 4L)
  expect_equal(local_map$transform[match("p_l", local_map$feature)], "Identity")

  local_hidden <- ndm_latent_matrix(latents, "ode_local_hidden")
  expect_equal(dim(local_hidden), c(6L, 2L))
  beta_feature <- ndm_latent_matrix(latents, "ode_local_features", features = "beta_l")
  expect_equal(
    as.numeric(beta_feature[, 1]),
    as.numeric(ndm:::.ndm_softplus_r(local_state[, , 3]))
  )

  delta_feature <- ndm_latent_matrix(latents, "ode_global_features", features = "delta")
  expect_equal(
    as.numeric(delta_feature[, 1]),
    as.numeric(ndm:::.ndm_sigmoid_r(global_state[, , 3]))
  )

  state_matrix <- ndm_latent_matrix(latents, "ode_states", features = "s_l")
  expect_equal(as.numeric(state_matrix[, 1]), as.numeric(s_state))
})

test_that("empty NeuralODE feature lists are unavailable rather than zero-row matrices", {
  env <- ndm_test_runtime_env()
  env$ModelType <- "NeuralODE"
  env$ModelDims <- 4L
  env$nOutcomes <- 1L
  env$LocalNeuralEmbedDim <- 2L
  env$GlobalNeuralEmbedDim <- 2L
  env$uq_encneural_vec <- "beta_l"
  env$uq_globalneural_vec <- character()

  local_state <- array(seq(-3, 20), dim = c(2, 3, 4))
  global_state <- array(seq_len(12), dim = c(2, 3, 2))
  env$GetPred_inference <- function(ModelList, x, state, PriorList, PolicyList, GetPredSaveAtInfo, seed) {
    list(
      list(
        y_mu = array(1, dim = c(2, 3, 1)),
        y_sigma = array(1, dim = c(2, 3, 1)),
        ODEParamsSampList = list(
          "diff_eq_sol_ts" = 0:2,
          "diff_eq_sol_ys.Neural1" = local_state,
          "diff_eq_sol_ys.Neural2" = global_state,
          center_param = array(1, dim = c(2, 3, 1)),
          scale_param = array(1, dim = c(2, 3, 1))
        )
      ),
      list(stage = "predicted")
    )
  }
  trained <- structure(list(env = env, model_type = "NeuralODE"), class = "ndm_trained_model")

  latents <- ndm_extract_temporal_latents(
    trained,
    batch = ndm_test_named_batch(),
    update_state = FALSE
  )

  local_features <- ndm_latent_matrix(latents, "ode_local_features")
  global_hidden <- ndm_latent_matrix(latents, "ode_global_hidden")
  expect_equal(dim(local_features), c(6L, 1L))
  expect_equal(dim(global_hidden), c(6L, 2L))
  expect_error(
    ndm_latent_matrix(latents, "ode_global_features"),
    "No features are available for this latent source"
  )
})

test_that("prediction and loss surface missing runtime bindings clearly", {
  env <- ndm_new_runtime_env()
  env$ModelList <- list(model = TRUE)
  env$state <- list(stage = "built")

  expect_error(
    ndm_predict(env, batch = ndm_test_named_batch()),
    "Runtime environment is missing required objects for prediction"
  )

  env <- ndm_test_runtime_env()
  batch <- ndm_test_named_batch()
  batch$YTrue_out <- NULL

  expect_error(
    ndm_loss(env, batch = batch),
    "could not be inferred from the batch"
  )
})

test_that("ndm_fit threads runtime, data, build, and train stages for real workflows", {
  calls <- character()
  seen <- new.env(parent = emptyenv())
  config <- ndm_create_config(
    model_type = "DecoderOnly",
    backbone = "transformer",
    float_type = "32",
    force_to_gpu = FALSE
  )
  spec <- ndm_model_spec(preset = "seirs_dynamic_beta")

  local_mocked_bindings(
    ndm_prepare_runtime = function(config, runtime_env = ndm_new_runtime_env(), runtime_globals = list()) {
      calls <<- c(calls, "prepare_runtime")
      seen$runtime_globals <- runtime_globals
      env <- ndm_new_runtime_env()
      assign("stage", "runtime", envir = env)
      env
    },
    ndm_prepare_data = function(runtime_env, generator = c("sim", "real", "multidisease"), runtime_globals = list()) {
      calls <<- c(calls, paste0("prepare_data:", generator))
      seen$data_globals <- runtime_globals
      assign("data_generator", generator, envir = runtime_env)
      runtime_env
    },
    ndm_build_model = function(runtime_env,
                               model_type = c("DecoderOnly", "NeuralODE"),
                               model_spec = NULL,
                               backbone = "transformer",
                               runtime_globals = list()) {
      calls <<- c(calls, "build_model")
      seen$build_globals <- runtime_globals
      seen$build_model_type <- model_type
      seen$build_backbone <- backbone
      seen$model_spec <- model_spec
      structure(list(env = runtime_env), class = "ndm_model")
    },
    ndm_train = function(x, run_define = TRUE, run_loop = TRUE) {
      calls <<- c(calls, "train")
      seen$train_globals <- mget("train_flag", envir = x$env, inherits = FALSE)
      seen$run_define <- run_define
      seen$run_loop <- run_loop
      structure(list(env = x$env, calls = calls), class = "ndm_trained_model")
    },
    .package = "ndm"
  )

  trained <- ndm_fit(
    config = config,
    model_spec = spec,
    data_generator = "real",
    runtime_globals = list(runtime_flag = TRUE),
    data_globals = list(data_flag = TRUE),
    build_globals = list(build_flag = TRUE),
    train_globals = list(train_flag = TRUE),
    run_define = FALSE,
    run_loop = TRUE
  )

  expect_s3_class(trained, "ndm_trained_model")
  expect_equal(calls, c("prepare_runtime", "prepare_data:real", "build_model", "train"))
  expect_equal(seen$runtime_globals, list(runtime_flag = TRUE))
  expect_equal(seen$data_globals, list(data_flag = TRUE))
  expect_equal(seen$build_globals, list(build_flag = TRUE))
  expect_equal(seen$build_model_type, "DecoderOnly")
  expect_equal(seen$build_backbone, "transformer")
  expect_identical(seen$model_spec, spec)
  expect_equal(seen$train_globals$train_flag, TRUE)
  expect_false(seen$run_define)
  expect_true(seen$run_loop)
})
