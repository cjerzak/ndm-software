test_that("corrected NeuralODE training preserves frozen arrays and round trips artifacts", {
  ndm_skip_if_no_sim_backend()
  frozen_before <- NULL
  details <- ndm_test_fit_sim_case(
    model_type = "NeuralODE", endogeneity = 0, n_sgd = 2L, model_dims = 16L,
    attention_head_dim = 8L, n_times_lookahead = 3L, return_details = TRUE,
    config_overrides = list(neuralode_train_max_substeps = 8L),
    before_train = function(runtime_env, model) {
      frozen <- runtime_env$eq$partition(runtime_env$ModelList, runtime_env$TrainabilityMask)[[2L]]
      frozen_before <<- lapply(runtime_env$jax$tree_util$tree_leaves(runtime_env$eq$filter(frozen, runtime_env$eq$is_array)),
                              function(x) as.array(runtime_env$np$array(x)))
    }
  )
  env <- details$runtime_env
  expect_equal(env$ExecuteUpdateCounter, 2L)
  expect_true(env$TrainingStateUsable)
  expect_lt(env$nParamsTrainable, env$nParamsTotal)
  expect_equal(env$nParamsTrainable + env$nParamsNonTrainable, env$nParamsTotal)
  frozen <- env$eq$partition(env$ModelList, env$TrainabilityMask)[[2L]]
  frozen_after <- lapply(env$jax$tree_util$tree_leaves(env$eq$filter(frozen, env$eq$is_array)), function(x) as.array(env$np$array(x)))
  expect_identical(frozen_after, frozen_before)
  expect_false(env$TrainabilityMask$InitProcessList$TimeEmbeds)
  expect_true(env$TrainabilityMask$TSList$TSBackbone$d1$AttnRes1$PseudoQuery)
  expect_equal(length(env$InitStateLogitParamNames), length(env$InitStateTerms) - 1L)
  expect_false(any(env$NeuralDefinitions_jax_STOCHASTIC["DefClean", ] %in% env$InitStateTerms))
  expect_true(env$LastSolverAccuracy$fallback || env$LastSolverAccuracy$scaled_discrepancy <= 1)
  # Public training-mode loss uses the same controller without corrupting state.
  value <- ndm_loss(details$trained, batch = details$batch, seed = 101L, update_state = TRUE)
  expect_true(is.finite(as.numeric(env$np$array(value))))
  pred <- ndm_predict(details$trained, batch = details$batch, seed = 12L, update_state = FALSE)
  artifact <- tempfile("ndm-corrected-artifact-")
  on.exit(unlink(artifact, recursive = TRUE), add = TRUE)
  ndm_save_model(details$trained, artifact)
  restored <- ndm_load_model(artifact)
  expect_equal(restored$env$ParameterCounts, env$ParameterCounts)
  actual <- ndm_predict(restored, batch = details$batch, seed = 12L, update_state = FALSE)
  expect_equal(as.array(env$np$array(actual$y_mu)), as.array(env$np$array(pred$y_mu)), tolerance = 0)
  # Run another optimizer update using the restored partition and moments.
  re <- restored$env
  x <- ndm_batch_to_model_inputs(details$batch)
  keys <- re$jax$random$split(re$JaxKey(3L), re$nBatch)
  info <- ndm:::.ndm_select_training_solver(re, re$ModelList, x, re$state, re$PriorList, re$PolicyList, re$GetPredSaveAtInfo_default, keys)
  result <- re$train_step_compiled(re$ModelList, x, details$batch$YTrue_out, details$batch$YTrue_out_mask,
    re$jnp$array(3.), re$state, re$PriorList, re$PolicyList, info, keys, re$opt_state)
  expect_true(as.logical(re$np$array(result$update_accepted)))
  metadata_file <- file.path(artifact, "metadata.rds")
  metadata <- readRDS(metadata_file); metadata$version <- 1L; saveRDS(metadata, metadata_file)
  digest_before <- tools::md5sum(metadata_file)
  expect_error(ndm_load_model(artifact), "retrain")
  expect_identical(tools::md5sum(metadata_file), digest_before)
})
