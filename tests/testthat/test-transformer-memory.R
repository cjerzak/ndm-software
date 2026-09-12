test_that("native grouped attention matches repeated heads and their gradients", {
  env <- ndm_test_architecture_env(backbone = TRUE)
  p <- list(
    q = env$jax$random$normal(env$jax$random$PRNGKey(10L), list(5L, 4L, 8L)),
    k = env$jax$random$normal(env$jax$random$PRNGKey(11L), list(7L, 1L, 8L)),
    v = env$jax$random$normal(env$jax$random$PRNGKey(12L), list(7L, 1L, 8L))
  )
  mask <- env$jnp$array(array(c(FALSE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE), c(1L, 1L, 7L)))
  grouped <- function(p) env$dot_product_attention_unified(p$q, p$k, p$v, mask, FALSE, "xla")
  repeated <- function(p) env$jax$nn$dot_product_attention(
    p$q, env$jnp$`repeat`(p$k, 4L, axis = 1L), env$jnp$`repeat`(p$v, 4L, axis = 1L),
    mask = mask, implementation = "xla"
  )
  expect_equal(ndm_test_architecture_array(grouped(p), env), ndm_test_architecture_array(repeated(p), env), tolerance = 1e-6)
  a <- env$jax$grad(function(p) env$jnp$sum(grouped(p)^2))(p)
  b <- env$jax$grad(function(p) env$jnp$sum(repeated(p)^2))(p)
  for (name in names(p)) expect_equal(ndm_test_architecture_array(a[[name]], env), ndm_test_architecture_array(b[[name]], env), tolerance = 2e-5)
})

test_that("context-only prefill preserves positions, predictions and gradients", {
  env <- ndm_test_architecture_env(backbone = TRUE)
  x <- env$jax$random$normal(env$jax$random$PRNGKey(20L), list(5L, 16L))
  mask <- env$jnp$array(matrix(c(0, 1, 0, 1, 1), ncol = 1L))
  padded_x <- env$jnp$pad(x, list(c(0L, 3L), c(0L, 0L)))
  padded_mask <- env$jnp$pad(mask, list(c(0L, 3L), c(0L, 0L)))
  compact <- function(p) env$transformer_prefill_kv(x, mask, p, max_len = 8L)
  reference <- function(p) env$transformer_prefill_kv(padded_x, padded_mask, p)
  a <- compact(env$TransformerList)
  b <- reference(env$TransformerList)
  expect_equal(as.numeric(env$np$array(a$next_pos)), 5)
  expect_equal(ndm_test_architecture_array(a$xt_last, env), ndm_test_architecture_array(b$xt_last, env), tolerance = 1e-5)
  for (layer in c("d1", "d2")) {
    expect_equal(as.integer(unlist(a$cache[[layer]]$k$shape)), c(8L, 1L, 8L))
    expect_equal(ndm_test_architecture_array(a$cache[[layer]]$valid, env), ndm_test_architecture_array(b$cache[[layer]]$valid, env))
  }
  rollout <- function(p, prefill) {
    first <- prefill(p)
    step <- env$transformer_decode_step_kv(first$xt_last, first$next_pos, p, first$cache)
    env$jnp$sum(step$token_out^2)
  }
  ga <- env$eq$filter_grad(function(p) rollout(p, compact))(env$TransformerList)
  gb <- env$eq$filter_grad(function(p) rollout(p, reference))(env$TransformerList)
  la <- env$jax$tree_util$tree_leaves(ga)
  lb <- env$jax$tree_util$tree_leaves(gb)
  for (i in seq_along(la)) expect_equal(ndm_test_architecture_array(la[[i]], env), ndm_test_architecture_array(lb[[i]], env), tolerance = 3e-5)
})

test_that("cuDNN input preparation pads sequence axes without expanding KV heads", {
  env <- ndm_test_architecture_env(backbone = TRUE, compute_dtype = "bfloat16")
  real_attention <- env$jax$nn$dot_product_attention
  env$cuda_attention_available <- env$TRY_FLASH <- TRUE
  seen <- new.env(parent = emptyenv())
  # Exercise the CUDA dispatch contract on CPU, then use XLA for arithmetic.
  # Actual CUDA kernel execution is a separate hardware qualification.
  env$jax <- list(nn = list(dot_product_attention = function(q, k, v, mask, is_causal, implementation) {
    seen$q <- as.integer(unlist(q$shape))
    seen$k <- as.integer(unlist(k$shape))
    seen$dtype <- q$dtype$name
    seen$implementation <- implementation
    real_attention(q, k, v, mask = mask, is_causal = is_causal, implementation = "xla")
  }))
  for (batched in c(FALSE, TRUE)) {
    q_shape <- c(if (batched) 2L, 5L, 4L, 8L)
    k_shape <- c(if (batched) 2L, 7L, 1L, 8L)
    out <- env$dot_product_attention_unified(
      env$jnp$ones(as.list(q_shape), dtype = env$jnp$bfloat16),
      env$jnp$ones(as.list(k_shape), dtype = env$jnp$bfloat16),
      env$jnp$ones(as.list(k_shape), dtype = env$jnp$bfloat16),
      mask = env$jnp$ones(list(1L, 1L, 7L), dtype = env$jnp$bool_)
    )
    expect_identical(as.integer(unlist(out$shape)), q_shape)
    expect_identical(seen$q, c(if (batched) 2L, 8L, 4L, 8L))
    expect_identical(seen$k, c(if (batched) 2L, 8L, 1L, 8L))
    expect_identical(seen$dtype, "bfloat16")
    expect_identical(seen$implementation, "cudnn")
    expect_true(all(ndm_test_architecture_array(out$astype(env$jnp$float32), env) == 1))
  }
})

test_that("BF16 transformer activations retain FP32 masters and finite gradients", {
  env <- ndm_test_architecture_env(backbone = TRUE, compute_dtype = "bfloat16")
  x <- env$jax$random$normal(env$jax$random$PRNGKey(30L), list(5L, 16L))
  mask <- env$jnp$ones(list(5L, 1L))
  forward <- function(p) env$transformer_prefill_kv(x, mask, p, max_len = 7L)
  output <- forward(env$TransformerList)
  expect_identical(output$xt_last$dtype$name, "bfloat16")
  expect_identical(output$cache$d1$k$dtype$name, "bfloat16")
  expect_identical(output$cache$d1$v$dtype$name, "bfloat16")
  expect_identical(env$TransformerList$d1$Multihead$W_q$dtype$name, "float32")
  grad <- env$eq$filter_grad(function(p) env$jnp$sum(forward(p)$xt_last$astype(env$jnp$float32)^2))(env$TransformerList)
  for (leaf in env$jax$tree_util$tree_leaves(grad)) {
    expect_identical(leaf$dtype$name, "float32")
    expect_true(all(is.finite(ndm_test_architecture_array(leaf, env))))
  }
  native <- ndm_test_architecture_env(backbone = TRUE, compute_dtype = "native")
  ref <- native$transformer_prefill_kv(x, mask, env$TransformerList, max_len = 7L)
  actual <- ndm_test_architecture_array(output$xt_last$astype(env$jnp$float32), env)
  expected <- ndm_test_architecture_array(ref$xt_last, env)
  expect_lt(sqrt(sum((actual - expected)^2) / sum(expected^2)), 0.04)
})

test_that("residual source rematerialization reduces saved arrays without changing gradients", {
  env <- ndm_test_architecture_env(backbone = TRUE, checkpointing = TRUE, depth = 4L)
  reference <- ndm_test_architecture_env(backbone = TRUE, checkpointing = FALSE, depth = 4L)
  x <- env$jax$random$normal(env$jax$random$PRNGKey(40L), list(12L, 16L))
  mask <- env$jnp$ones(list(12L, 1L))
  loss <- function(p) env$jnp$sum(env$RunTransformerBackbone(x, mask, p)^2)
  ref <- function(p) reference$jnp$sum(reference$RunTransformerBackbone(x, mask, p)^2)
  a <- env$eq$filter_value_and_grad(loss)(env$TransformerList)
  b <- env$eq$filter_value_and_grad(ref)(env$TransformerList)
  la <- env$jax$tree_util$tree_leaves(a)
  lb <- env$jax$tree_util$tree_leaves(b)
  for (i in seq_along(la)) expect_equal(ndm_test_architecture_array(la[[i]], env), ndm_test_architecture_array(lb[[i]], env), tolerance = 3e-5)
  # Inspect reverse-mode residuals, rather than assuming scan/remat reduces them.
  inspector <- reticulate::py_run_string(paste(
    "def saved_bytes(f, x):",
    "    from jax._src.ad_checkpoint import saved_residuals",
    "    import math",
    "    return sum(math.prod(a.shape) * a.dtype.itemsize for a, _ in saved_residuals(f, x))",
    sep = "\n"
  ), local = TRUE)
  before <- inspector$saved_bytes(ref, env$TransformerList)
  after <- inspector$saved_bytes(loss, env$TransformerList)
  expect_lt(after, before)
  message(sprintf("Transformer saved residuals: %s -> %s bytes", before, after))
})

test_that("donated updates reject bad loss and solver failures with usable original values", {
  env <- ndm_test_architecture_env()
  env$TrackBlockUpdateNorms <- FALSE
  env$optax_optimizer <- env$optax$adabelief(1e-3)
  env$switch_filter_jit <- env$eq$filter_jit
  env$DonateTrainingState <- TRUE
  env$getLoss_train <- function(ModelList, batch_pkg, y_true, y_mask, iteration,
                                state, PriorList, PolicyList, GetPredSaveAtInfo, seed) {
    list(env$jnp$sum((ModelList$weight - y_true)^2), list(
      model_state = state, solver_diagnostics = list(success = y_mask), loss_components = list()
    ))
  }
  for (name in c(
    "ndm_training_tree_finite",
    "ndm_training_solver_diagnostic_fields", "ndm_training_loss_component_fields",
    "ndm_training_pack_diagnostics", "ndm_training_pack_scalars",
    "ndm_training_observation_mask_valid",
    "train_step_impl", "train_step_owned_compiled"
  )) {
    ndm_test_architecture_function("ModelTrainers/SuperLModel_TrainDefine.R", name, env)
  }
  for (failure in c("none", "nan_loss", "solver")) {
    model <- list(weight = env$jnp$ones(list(64L)))
    state <- env$optax_optimizer$init(model)
    state <- env$jax$tree_util$tree_map(function(x) env$jnp$array(x, copy = TRUE), state)
    original_moments <- lapply(env$jax$tree_util$tree_leaves(state), ndm_test_architecture_array, env = env)
    owned <- list(model = model, opt_state = state)
    fixed <- list(batch_pkg = NULL, y_true = env$jnp$array(if (failure == "nan_loss") NaN else 0),
                  y_mask = env$jnp$array(failure != "solver"), iteration = env$jnp$array(1L),
                  state = env$jnp$array(1.), PriorList = NULL, PolicyList = NULL,
                  GetPredSaveAtInfo = NULL, seed = NULL)
    result <- env$train_step_owned_compiled(fixed, owned)
    accepted <- isTRUE(as.logical(env$np$array(result$update_accepted)))
    expect_identical(accepted, failure == "none")
    expect_true(isTRUE(model$weight$is_deleted()))
    weights <- ndm_test_architecture_array(result$model$weight, env)
    if (accepted) expect_true(all(weights < 1)) else expect_equal(weights, as.array(rep(1, 64L)))
    expect_true(all(is.finite(weights)))
    if (!accepted) {
      returned_moments <- lapply(env$jax$tree_util$tree_leaves(result$opt_state), ndm_test_architecture_array, env = env)
      expect_identical(returned_moments, original_moments)
    }
    # Returned buffers remain usable on a second donated call, including rejection.
    fixed$y_true <- env$jnp$array(0.)
    fixed$y_mask <- env$jnp$array(TRUE)
    resumed <- env$train_step_owned_compiled(fixed, list(model = result$model, opt_state = result$opt_state))
    expect_true(isTRUE(as.logical(env$np$array(resumed$update_accepted))))
  }
})
