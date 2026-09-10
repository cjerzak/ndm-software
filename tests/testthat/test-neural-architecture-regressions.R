test_that("final attention residual aggregation preserves the input and its gradient", {
  env <- ndm_test_architecture_env(backbone = TRUE)
  params <- env$TransformerList
  for (layer in c("d1", "d2")) {
    params[[layer]]$Multihead$W_o <- env$jnp$zeros_like(params[[layer]]$Multihead$W_o)
    params[[layer]]$FFN$OutProj1 <- env$jax$tree_util$tree_map(
      function(x) x * 0, params[[layer]]$FFN$OutProj1
    )
  }
  x <- env$jax$random$normal(env$jax$random$PRNGKey(2L), list(4L, 16L))
  mask <- env$jnp$array(matrix(c(0, 1, 1, 1), ncol = 1L))
  forward <- function(x) env$RunTransformerBackbone(x, mask, params)
  # Four zero sublayer outputs plus the embedding: uniform attention gives x/5.
  expected <- x * mask / 5
  expect_equal(ndm_test_architecture_array(forward(x), env),
               ndm_test_architecture_array(expected, env), tolerance = 1e-6)
  grad <- env$jax$grad(function(x) env$jnp$sum(forward(x)))(x)
  expect_equal(ndm_test_architecture_array(grad, env),
               ndm_test_architecture_array(env$jnp$ones_like(x) * mask / 5, env),
               tolerance = 1e-6)
})

test_that("learned output residual mixing agrees across full, prefill and decode paths", {
  env <- ndm_test_architecture_env(backbone = TRUE)
  params <- env$TransformerList
  params$AttnResOutput$PseudoQuery <- env$jax$random$normal(
    env$jax$random$PRNGKey(3L), list(16L)
  ) * 0.2
  x <- env$jax$random$normal(env$jax$random$PRNGKey(4L), list(6L, 16L))
  mask <- env$jnp$array(matrix(c(0, 1, 0, 1, 0, 0), ncol = 1L))
  prefill <- env$transformer_prefill_kv(x, mask, params)
  full <- env$RunTransformerBackbone(x, mask, params)
  expect_equal(ndm_test_architecture_array(prefill$xt_last, env),
               ndm_test_architecture_array(env$jnp$take(full, 3L, axis = 0L), env),
               tolerance = 1e-5)
  cached_forward <- function(p) {
    cache <- env$transformer_prefill_kv(x, mask, p)$cache
    for (pos in 4L:5L) {
      decoded <- env$transformer_decode_step_kv(
        env$jnp$take(x, pos, axis = 0L), env$jnp$array(pos), p, cache
      )
      cache <- decoded$cache
    }
    decoded$token_out
  }
  full_mask <- env$jnp$array(matrix(c(0, 1, 0, 1, 1, 1), ncol = 1L))
  full_forward <- function(p) {
    env$jnp$take(env$RunTransformerBackbone(x, full_mask, p), 5L, axis = 0L)
  }
  expect_equal(ndm_test_architecture_array(cached_forward(params), env),
               ndm_test_architecture_array(full_forward(params), env), tolerance = 1e-5)
  cached_grad <- env$eq$filter_grad(function(p) env$jnp$sum(cached_forward(p)^2))(params)
  full_grad <- env$eq$filter_grad(function(p) env$jnp$sum(full_forward(p)^2))(params)
  cached_leaves <- env$jax$tree_util$tree_leaves(cached_grad)
  full_leaves <- env$jax$tree_util$tree_leaves(full_grad)
  expect_length(cached_leaves, length(full_leaves))
  for (i in seq_along(cached_leaves)) {
    expect_equal(ndm_test_architecture_array(cached_leaves[[i]], env),
                 ndm_test_architecture_array(full_leaves[[i]], env), tolerance = 2e-5)
  }
  expect_gt(as.numeric(env$np$array(env$jnp$linalg$norm(
    full_grad$AttnResOutput$PseudoQuery
  ))), 0)
})

test_that("convolution kernels preserve the requested width on both sides of the input width", {
  env <- ndm_test_architecture_env()
  for (name in c("init_orthogonal_kernel", "manual_conv1d")) {
    ndm_test_architecture_function("ModelDefiners/SuperLModel_BuildML.R", name, env)
  }
  for (input_width in c(8L, 16L, 20L)) {
    for (kernel_size in c(1L, 3L, 7L)) {
      kernel <- env$init_orthogonal_kernel(
        env$jax$random$PRNGKey(5L), kernel_size, input_width, 16L
      )
      expect_equal(as.integer(unlist(kernel$shape)), c(kernel_size, input_width, 16L))
      output <- env$manual_conv1d(env$jnp$ones(list(5L, input_width)), kernel, kernel_size)
      expect_equal(as.integer(unlist(output$shape)), c(5L, 16L))
      expect_true(all(is.finite(ndm_test_architecture_array(output, env))))
    }
  }
})

test_that("runtime losses and gradients ignore masked non-finite targets", {
  env <- ndm_test_architecture_env()
  env$.ndm_select_model_targets <- ndm:::.ndm_select_model_targets
  env$ndm_student_t_masked_nll <- ndm:::.ndm_student_t_masked_nll
  env$nOutcomes <- 1L
  env$ObservationScaleFloor <- 1e-5
  env$neuralode_variational <- FALSE
  env$neuralode_kl_weight <- 0
  env$outcome_loss_scale <- 2
  env$GetPred_train <- function(ModelList, ...) list(list(
    y_mu = ModelList, y_sigma = env$jnp$ones_like(ModelList),
    solver_diagnostics = list(success = env$jnp$ones(list(1L), dtype = env$jnp$bool_))
  ), NULL)
  ndm_test_architecture_function("ModelDefiners/SuperLModel_BuildML.R", "getLoss_train", env)
  mu <- env$jnp$ones(list(1L, 4L, 1L))
  mask <- env$jnp$array(array(c(TRUE, FALSE, FALSE, FALSE), c(1L, 4L, 1L)))
  poisoned <- env$jnp$array(array(c(2, NaN, Inf, -Inf), c(1L, 4L, 1L)))
  clean <- env$jnp$array(array(c(2, 0, 0, 0), c(1L, 4L, 1L)))
  for (model_type in c("DecoderOnly", "NeuralODE")) {
    env$ModelType <- model_type
    for (objective in c("student_t_nll", "scaled_mse")) {
      env$training_objective <- objective
      env$neuralode_mean_loss_weight <- if (objective == "student_t_nll") 0.5 else 0
      loss <- function(mu, y) env$getLoss_train(mu, NULL, y, mask, NULL, NULL, NULL, NULL, NULL, NULL)[[1L]]
      expect_equal(ndm_test_architecture_array(loss(mu, poisoned), env),
                   ndm_test_architecture_array(loss(mu, clean), env), tolerance = 1e-6)
      actual_grad <- env$jax$grad(function(mu) loss(mu, poisoned))(mu)
      expected_grad <- env$jax$grad(function(mu) loss(mu, clean))(mu)
      expect_true(all(is.finite(ndm_test_architecture_array(actual_grad, env))))
      expect_equal(ndm_test_architecture_array(actual_grad, env),
                   ndm_test_architecture_array(expected_grad, env), tolerance = 1e-6)
    }
    # An overflowing diagnostic MSE must not enter a disabled auxiliary loss.
    env$training_objective <- "student_t_nll"
    env$neuralode_mean_loss_weight <- 0
    extreme <- env$jnp$array(array(c(1e30, 0, 0, 0), c(1L, 4L, 1L)))
    expect_true(is.finite(as.numeric(env$np$array(loss(mu, extreme)))))
    expect_true(all(is.finite(ndm_test_architecture_array(
      env$jax$grad(function(mu) loss(mu, extreme))(mu), env
    ))))
  }
})

test_that("cosine schedule spans the full run and supports short runs", {
  env <- ndm_test_architecture_env()
  ndm_test_architecture_function("ModelTrainers/SuperLModel_TrainDefine.R", "ndm_training_lr_schedule", env)
  schedule <- env$ndm_training_lr_schedule(1000L, 1)
  value <- function(step) as.numeric(env$np$array(schedule(step)))
  expect_equal(value(0L), 0.01, tolerance = 1e-6)
  expect_equal(value(100L), 1, tolerance = 1e-6)
  expect_gt(value(900L), 0.01)
  expect_equal(value(1000L), 0.01, tolerance = 1e-6)
  for (steps in c(1L, 2L, 20L, 100L)) {
    schedule <- env$ndm_training_lr_schedule(steps, 1)
    expect_true(all(is.finite(vapply(0L:steps, value, numeric(1L)))))
    expect_equal(value(if (steps == 1L) 0L else steps - 1L), 1, tolerance = 1e-6)
  }
})

test_that("optimizer preserves query gradients while clipping ordinary weights", {
  env <- ndm_test_architecture_env()
  for (name in c("ndm_training_clip_mask", "ndm_training_optimizer")) {
    ndm_test_architecture_function("ModelTrainers/SuperLModel_TrainDefine.R", name, env)
  }
  query <- list(PseudoQuery = env$jnp$zeros(list(16L)), NormScale = env$jnp$ones(list(16L)))
  params <- list(TSList = list(TSBackbone = list(
    d1 = list(AttnRes1 = query, AttnRes2 = query, weight = env$jnp$ones(list(16L))),
    AttnResOutput = query
  )))
  grads <- env$jax$tree_util$tree_map(function(x) env$jnp$ones_like(x), params)
  clipped <- env$optax$masked(
    env$optax$adaptive_grad_clip(0.1, eps = 0.0001), env$ndm_training_clip_mask
  )
  updates <- clipped$update(grads, clipped$init(params), params)[[1L]]
  expect_equal(ndm_test_architecture_array(updates$TSList$TSBackbone$d1$AttnRes1$PseudoQuery, env), as.array(rep(1, 16L)))
  expect_equal(ndm_test_architecture_array(updates$TSList$TSBackbone$d1$AttnRes2$PseudoQuery, env), as.array(rep(1, 16L)))
  expect_equal(ndm_test_architecture_array(updates$TSList$TSBackbone$AttnResOutput$PseudoQuery, env), as.array(rep(1, 16L)))
  expect_equal(ndm_test_architecture_array(updates$TSList$TSBackbone$d1$weight, env), as.array(rep(0.1, 16L)), tolerance = 1e-6)
  expect_equal(ndm_test_architecture_array(updates$TSList$TSBackbone$AttnResOutput$NormScale, env), as.array(rep(0.1, 16L)), tolerance = 1e-6)
  optimizer <- env$ndm_training_optimizer(1e-3)
  step <- env$eq$filter_jit(function(p, state) optimizer$update(grads, state, p))
  result <- step(params, optimizer$init(params))
  query_update <- result[[1L]]$TSList$TSBackbone$AttnResOutput$PseudoQuery
  expect_gt(as.numeric(env$np$array(env$jnp$linalg$norm(query_update))), 1e-3)
})
