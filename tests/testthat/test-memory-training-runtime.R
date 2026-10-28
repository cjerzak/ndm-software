test_that("BF16 decoder and NeuralODE training keep heads and optimizer moments at FP32", {
  ndm_skip_if_no_sim_backend()
  for (model_type in c("DecoderOnly", "NeuralODE")) {
    frozen_before <- NULL
    details <- ndm_test_fit_sim_case(
      model_type = model_type, endogeneity = 0, n_sgd = 2L,
      config_overrides = list(transformer_compute_dtype = "bfloat16", neuralode_variational = FALSE),
      return_details = TRUE,
      before_train = function(runtime_env, model) {
        frozen <- runtime_env$eq$partition(runtime_env$ModelList, runtime_env$TrainabilityMask)[[2L]]
        frozen_before <<- lapply(runtime_env$jax$tree_util$tree_leaves(runtime_env$eq$filter(frozen, runtime_env$eq$is_array)),
          function(x) as.array(runtime_env$np$array(x)))
      }
    )
    env <- details$runtime_env
    frozen <- env$eq$partition(env$ModelList, env$TrainabilityMask)[[2L]]
    frozen_after <- lapply(env$jax$tree_util$tree_leaves(env$eq$filter(frozen, env$eq$is_array)),
      function(x) as.array(env$np$array(x)))
    expect_identical(frozen_after, frozen_before)
    expect_lt(env$nParamsTrainable, env$nParamsTotal)
    expect_equal(env$nParamsTrainable + env$nParamsNonTrainable, env$nParamsTotal)
    expect_identical(env$TransformerComputeDtypeResolved, "bfloat16")
    expect_true(env$DonateTrainingState)
    expect_true(env$TrainingStateUsable)
    expect_false(exists("TransformerList", envir = env, inherits = FALSE))
    expect_true(is.finite(details$summary$first_loss[[1L]]))
    expect_true(is.finite(details$summary$last_loss[[1L]]))
    for (leaf in env$jax$tree_util$tree_leaves(env$opt_state)) {
      if (env$eq$is_inexact_array(leaf)) expect_identical(leaf$dtype$name, "float32")
    }
    pred <- ndm_predict(details$trained, batch = details$batch, inference = TRUE, seed = 71L, update_state = FALSE)
    expect_identical(pred$y_mu$dtype$name, "float32")
    expect_identical(pred$y_sigma$dtype$name, "float32")
    expect_true(all(is.finite(as.array(env$np$array(pred$y_mu)))))
    if (model_type == "NeuralODE") {
      states <- pred$ODEParamsSampList$diff_eq_sol_ys
      # R concatenation flattens the named ODE-state list into dotted entries.
      if (is.null(states)) states <- pred$ODEParamsSampList[grepl("^diff_eq_sol_ys", names(pred$ODEParamsSampList))]
      expect_gt(length(env$jax$tree_util$tree_leaves(states)), 0L)
      for (leaf in env$jax$tree_util$tree_leaves(states)) expect_identical(leaf$dtype$name, "float32")
    }
  }
})
