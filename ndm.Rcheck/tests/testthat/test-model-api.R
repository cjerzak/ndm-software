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
