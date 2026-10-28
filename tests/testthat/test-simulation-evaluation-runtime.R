test_that("simulation checkpoints cover the same panel across batch boundaries", {
  ndm_require_backend_test_stack("complete simulation inference")
  fit <- ndm_test_fit_sim_case("NeuralODE", 0, n_sgd = 1L, n_checkpoints = 1L,
    runtime_globals = list(InferenceMCDraws = 3L), return_details = TRUE)
  env <- fit$runtime_env
  snapshot <- ndm_test_copy_env(env)
  on.exit(ndm_test_restore_env(snapshot), add = TRUE)
  file <- env$ndm_last_evaluation_file
  before <- data.table::fread(file)
  expect_equal(before$inference_examples, 128L)
  expect_equal(before$inference_mc_draws_requested, 3L)
  expect_equal(before$inference_mc_draws_effective, 3L)
  expect_gte(before$KLDiv_init, 0)
  fingerprint <- digest::digest(file = file, algo = "sha256")
  set.seed(119L); rng <- .Random.seed
  ndm_source_runtime_results_get(env = env)
  expect_identical(.Random.seed, rng)
  expect_identical(digest::digest(file = file, algo = "sha256"), fingerprint)
  # The final batch contains one example; keys remain canonical ordinals.
  env$TFDataset_inference <- env$TFDataset_inference$unbatch()$batch(127L)
  env$ndm_last_evaluated_step <- NULL
  ndm_source_runtime_results_get(env = env)
  after <- data.table::fread(file)
  expect_identical(after$policy_status, "ok")
  fields <- c("inference_examples", "KLDiv_init", grep("^RSS(Pred|Baseline)Time", names(before), value = TRUE))
  expect_equal(as.numeric(before[, ..fields]), as.numeric(after[, ..fields]), tolerance = 1e-5)
  expect_identical(.Random.seed, rng)
})
