#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ndm)
  library(testthat)
})

source("tests/testthat/helper-sim-fit.R")

run_pair <- function(label,
                     shared_runtime_globals = ndm_test_sim_parity_shared_runtime_globals(4242L),
                     neuralode_runtime_globals_after_setup = ndm_test_sim_parity_neural_runtime_globals_after_setup(),
                     neuralode_runtime_globals = NULL,
                     decoder_runtime_globals = NULL,
                     n_sgd = 300L,
                     model_dims = 64L) {
  cat(sprintf("\n== %s ==\n", label))
  started <- Sys.time()
  pair <- ndm_test_collect_week10_relative_accuracy_pair(
    endogeneity = 0.0,
    shared_seed = 4242L,
    n_times_lookahead = 10L,
    n_sgd = as.integer(n_sgd),
    model_dims = as.integer(model_dims),
    shared_runtime_globals = shared_runtime_globals,
    neuralode_runtime_globals_after_setup = neuralode_runtime_globals_after_setup,
    neuralode_runtime_globals = neuralode_runtime_globals,
    decoder_runtime_globals = decoder_runtime_globals,
    return_details = TRUE
  )
  elapsed <- difftime(Sys.time(), started, units = "secs")
  state_metrics <- ndm_test_capture_initial_state_metrics(pair$neuralode_details, seed = 99L)
  moved_blocks <- stats::aggregate(
    update_norm ~ block,
    data = pair$neuralode_details$block_update_log,
    FUN = function(x) max(x, na.rm = TRUE)
  )
  print(list(
    info = pair$info,
    decoder_over_neural = pair$decoder_relative_accuracy_10 / pair$neuralode_relative_accuracy_10,
    neural_over_decoder = pair$neuralode_relative_accuracy_10 / pair$decoder_relative_accuracy_10,
    decoder_summary = pair$decoder_details$summary,
    neural_summary = pair$neuralode_details$summary,
    state_metrics = state_metrics[c("mean_entropy", "mean_max_component", "mean_components")],
    moved_blocks = moved_blocks,
    elapsed_seconds = as.numeric(elapsed)
  ))
  invisible(pair)
}

backend_conda_env <- Sys.getenv(
  "NDM_SOFTWARE_CONDA_ENV",
  unset = Sys.getenv(
    "NDM_CONDA_ENV",
    unset = if (grepl(version$os, pattern = "darwin")) "jax_cpu" else "ndm_software_env"
  )
)
ndm_initialize_backend(
  conda_env = backend_conda_env,
  import_tensorflow = TRUE
)

mode <- if (length(commandArgs(trailingOnly = TRUE)) > 0L) {
  commandArgs(trailingOnly = TRUE)[[1L]]
} else {
  "baseline"
}

results <- switch(
  mode,
  baseline = list(
    baseline = run_pair("baseline", n_sgd = 300L, model_dims = 64L)
  ),
  probe = list(
    baseline = run_pair("baseline-quick", n_sgd = 80L, model_dims = 64L),
    smaller_dt0 = run_pair(
      "smaller-dt0",
      neuralode_runtime_globals_after_setup = utils::modifyList(
        ndm_test_sim_parity_neural_runtime_globals_after_setup(),
        list(dt0_init_optim = 1e-3)
      ),
      n_sgd = 80L,
      model_dims = 64L
    ),
    no_sampling = run_pair(
      "testWithoutSampling",
      neuralode_runtime_globals = list(testWithoutSampling = TRUE),
      n_sgd = 80L,
      model_dims = 64L
    ),
    diag_cov = run_pair(
      "diag-cov",
      neuralode_runtime_globals = list(UseDiagonalLMatVCov = TRUE),
      n_sgd = 80L,
      model_dims = 64L
    )
  ),
  stop(sprintf("Unknown mode: %s", mode))
)

saveRDS(results, file = sprintf("Tmp/repro_week10_parity_%s.rds", mode))
