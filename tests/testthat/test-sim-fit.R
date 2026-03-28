test_that("simulated pandemic fits improve across model families and endogeneity levels", {
  ndm_skip_if_no_sim_backend()

  cases <- data.frame(
    model_type = c("DecoderOnly", "DecoderOnly", "NeuralODE", "NeuralODE"),
    endogeneity = c(0.0, 0.6, 0.0, 0.6),
    stringsAsFactors = FALSE
  )

  results <- do.call(
    rbind,
    lapply(seq_len(nrow(cases)), function(i) {
      ndm_test_fit_sim_case(
        model_type = cases$model_type[[i]],
        endogeneity = cases$endogeneity[[i]]
      )
    })
  )
  results_info <- paste(capture.output(print(results)), collapse = "\n")
  neural_ode_results <- results[results$model_type == "NeuralODE", , drop = FALSE]

  expect_true(all(is.finite(results$first_loss)))
  expect_true(all(is.finite(results$last_loss)))
  expect_true(all(results$spec_preset == "seirs_dynamic_beta"), info = results_info)
  expect_true(all(results$loss_delta >= 0), info = results_info)
  expect_true(mean(results$loss_delta) > 1e-3, info = results_info)
  expect_true(min(neural_ode_results$loss_delta) > 1e-3, info = results_info)
})

test_that("safe log diagnostics plots tolerate non-finite loss histories", {
  pdf_file <- tempfile(fileext = ".pdf")
  grDevices::pdf(pdf_file)
  on.exit({
    try(grDevices::dev.off(), silent = TRUE)
  }, add = TRUE)

  expect_invisible(ndm:::.ndm_plot_log_series_safe(c(Inf, NA_real_, 0, -1), main = "Loss"))
  expect_invisible(ndm:::.ndm_plot_log_series_safe(c(NA_real_, NaN, Inf), main = "Gradients"))
})

test_that("decoder cache and non-cache predictions agree in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    return_details = TRUE
  )

  details$runtime_env$EnableKVCaching <- FALSE
  pred_no_cache <- ndm_predict(
    details$trained,
    batch = details$batch,
    seed = 123L,
    update_state = FALSE
  )
  details$runtime_env$EnableKVCaching <- TRUE
  pred_cache <- ndm_predict(
    details$trained,
    batch = details$batch,
    seed = 123L,
    update_state = FALSE
  )

  pred_no_cache_mu <- details$runtime_env$np$asanyarray(pred_no_cache$y_mu)
  pred_cache_mu <- details$runtime_env$np$asanyarray(pred_cache$y_mu)

  expect_equal(dim(pred_cache_mu), dim(pred_no_cache_mu))
  expect_lt(max(abs(pred_cache_mu - pred_no_cache_mu)), 1e-4)
})

test_that("decoder transformer resolves GQA topology and KV cache shapes in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  # Prime the session with a smaller decoder model so the GQA assertions below
  # verify that helper bindings are scoped to the current runtime.
  small_details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    model_dims = 32L,
    return_details = TRUE
  )

  details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    model_dims = 256L,
    return_details = TRUE
  )

  env <- details$runtime_env
  expect_equal(as.integer(env$TransformerHeads), 4L)
  expect_equal(as.integer(env$TransformerKVHeads), 1L)
  expect_equal(as.integer(env$TransformerHeadDim), 64L)
  expect_equal(as.integer(env$TransformerKVGroupSize), 4L)

  layer <- details$model$env$ModelList$TSList$TSBackbone$d1$Multihead
  small_layer <- small_details$model$env$ModelList$TSList$TSBackbone$d1$Multihead
  wq_shape <- as.integer(unlist(reticulate::py_to_r(layer$W_q$shape)))
  wk_shape <- as.integer(unlist(reticulate::py_to_r(layer$W_k$shape)))
  wv_shape <- as.integer(unlist(reticulate::py_to_r(layer$W_v$shape)))
  wo_shape <- as.integer(unlist(reticulate::py_to_r(layer$W_o$shape)))
  qnorm_shape <- as.integer(unlist(reticulate::py_to_r(layer$QNormScale$shape)))
  knorm_shape <- as.integer(unlist(reticulate::py_to_r(layer$KNormScale$shape)))
  small_qnorm_shape <- as.integer(unlist(reticulate::py_to_r(small_layer$QNormScale$shape)))
  small_knorm_shape <- as.integer(unlist(reticulate::py_to_r(small_layer$KNormScale$shape)))
  expect_equal(wq_shape, c(256L, 256L))
  expect_equal(wk_shape, c(256L, 64L))
  expect_equal(wv_shape, c(256L, 64L))
  expect_equal(wo_shape, c(256L, 256L))
  expect_equal(qnorm_shape, c(4L, 1L))
  expect_equal(knorm_shape, c(1L, 1L))
  expect_equal(small_qnorm_shape, c(1L, 1L))
  expect_equal(small_knorm_shape, c(1L, 1L))

  cache <- env$kv_cache_allocate(
    max_len = 11L,
    num_layers = 1L,
    num_kv_heads = env$TransformerKVHeads,
    head_dim = env$TransformerHeadDim,
    dtype = details$batch$XPred$dtype
  )
  cache_shape <- as.integer(unlist(reticulate::py_to_r(cache$d1$k$shape)))
  expect_equal(cache_shape, c(11L, 1L, 64L))

  topology_cases <- list(
    "8" = c(1L, 1L, 8L),
    "32" = c(1L, 1L, 32L),
    "64" = c(1L, 1L, 64L),
    "96" = c(2L, 1L, 48L),
    "128" = c(2L, 1L, 64L),
    "192" = c(3L, 1L, 64L),
    "256" = c(4L, 1L, 64L)
  )

  for (case_name in names(topology_cases)) {
    resolved <- env$resolve_transformer_topology(
      model_dims = as.integer(case_name),
      target_head_dim = 64L,
      kv_heads_override = NULL
    )
    expect_equal(
      c(
        as.integer(resolved$num_query_heads),
        as.integer(resolved$num_kv_heads),
        as.integer(resolved$head_dim)
      ),
      topology_cases[[case_name]]
    )
    expect_equal(as.integer(resolved$head_dim) %% 2L, 0L)
    expect_equal(as.integer(resolved$num_query_heads) %% as.integer(resolved$num_kv_heads), 0L)
  }
})

test_that("decoder inference tolerates missing QK norm gains in older model trees", {
  ndm_skip_if_no_sim_backend()

  details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    return_details = TRUE
  )

  multihead <- details$trained$env$ModelList$TSList$TSBackbone$d1$Multihead
  multihead$QNormScale <- NULL
  multihead$KNormScale <- NULL
  details$trained$env$ModelList$TSList$TSBackbone$d1$Multihead <- multihead
  details$runtime_env$ModelList <- details$trained$env$ModelList

  pred <- ndm_predict(
    details$trained,
    batch = details$batch,
    seed = 123L,
    update_state = FALSE
  )
  pred_mu <- details$runtime_env$np$asanyarray(pred$y_mu)

  expect_true(all(is.finite(pred_mu)))
})

test_that("checkpointed sim runs emit analytics artifacts in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    n_checkpoints = 1L,
    return_details = TRUE
  )

  csv_files <- list.files(details$holder_folder, pattern = "^res.*\\.csv$", full.names = TRUE)
  rdata_files <- list.files(details$holder_folder, pattern = "^res.*\\.Rdata$", full.names = TRUE)

  expect_length(csv_files, 1L)
  expect_length(rdata_files, 1L)

  metrics <- as.data.frame(data.table::fread(csv_files[[1]]))
  metric_names <- names(metrics)

  expect_true("PolicySkill1" %in% metric_names)
  expect_true("RSSBaselineTime1" %in% metric_names)
  expect_true(is.finite(as.numeric(details$trained$env$Skill8SanityCheck)))
})

test_that("checkpointed NeuralODE sim runs emit structural analytics artifacts in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  details <- ndm_test_fit_sim_case(
    model_type = "NeuralODE",
    endogeneity = 0.0,
    n_sgd = 1L,
    n_checkpoints = 1L,
    return_details = TRUE
  )

  csv_files <- list.files(details$holder_folder, pattern = "^res.*\\.csv$", full.names = TRUE)
  rdata_files <- list.files(details$holder_folder, pattern = "^res.*\\.Rdata$", full.names = TRUE)

  expect_length(csv_files, 1L)
  expect_length(rdata_files, 1L)

  metrics <- as.data.frame(data.table::fread(csv_files[[1]]))
  metric_names <- names(metrics)

  expect_true("AbsDiff_init" %in% metric_names)
  expect_true("AbsDiff_gamma" %in% metric_names)
  expect_true("PolicySkill1" %in% metric_names)
  expect_true(is.finite(as.numeric(metrics$AbsDiff_init[[1L]])))
  expect_true(is.finite(as.numeric(metrics$AbsDiff_gamma[[1L]])))
  expect_true(is.finite(as.numeric(details$trained$env$Skill8SanityCheck)))
})

test_that("sim analytics reader selects the final checkpoint CSV", {
  holder_folder <- tempfile("ndm-sim-metrics-")
  dir.create(holder_folder, recursive = TRUE)
  on.exit(unlink(holder_folder, recursive = TRUE, force = TRUE), add = TRUE)

  data.table::fwrite(
    data.frame(iteration = 10L, marker = "early"),
    file.path(holder_folder, "res1_i10.csv")
  )
  data.table::fwrite(
    data.frame(iteration = 50L, marker = "final"),
    file.path(holder_folder, "res1_i50.csv")
  )

  metrics <- ndm_test_read_single_sim_metrics(holder_folder)

  expect_equal(as.integer(metrics$iteration[[1L]]), 50L)
  expect_equal(as.character(metrics$marker[[1L]]), "final")
})

test_that("sim analytics reader rejects malformed checkpoint CSV names when multiple files exist", {
  holder_folder <- tempfile("ndm-sim-metrics-")
  dir.create(holder_folder, recursive = TRUE)
  on.exit(unlink(holder_folder, recursive = TRUE, force = TRUE), add = TRUE)

  data.table::fwrite(
    data.frame(iteration = 10L),
    file.path(holder_folder, "res1_i10.csv")
  )
  data.table::fwrite(
    data.frame(iteration = 50L),
    file.path(holder_folder, "res_bad.csv")
  )

  expect_error(
    ndm_test_read_single_sim_metrics(holder_folder),
    "Offending files: res_bad.csv"
  )
})

test_that("tuned scientific NeuralODE week-10 parity stays within 25% of decoder on the same sim case", {
  ndm_skip_if_no_sim_backend()

  pair <- ndm_test_collect_week10_relative_accuracy_pair(
    endogeneity = 0.0,
    shared_seed = 4242L,
    n_times_lookahead = 10L,
    n_sgd = 250L,
    model_dims = 64L,
    shared_runtime_globals = ndm_test_sim_parity_shared_runtime_globals(4242L),
    neuralode_runtime_globals_after_setup = ndm_test_sim_parity_neural_runtime_globals_after_setup(),
    return_details = TRUE
  )
  decoder_over_neural <- pair$decoder_relative_accuracy_10 / pair$neuralode_relative_accuracy_10
  neural_over_decoder <- pair$neuralode_relative_accuracy_10 / pair$decoder_relative_accuracy_10
  ratio_info <- paste(
    pair$info,
    sprintf("decoder/neuralode ratio=%.6f", decoder_over_neural),
    sprintf("neuralode/decoder ratio=%.6f", neural_over_decoder),
    sep = "; "
  )

  expect_true(is.finite(pair$decoder_relative_accuracy_10), info = ratio_info)
  expect_true(is.finite(pair$neuralode_relative_accuracy_10), info = ratio_info)
  expect_true(is.finite(pair$decoder_iterations_per_second), info = ratio_info)
  expect_true(is.finite(pair$neuralode_iterations_per_second), info = ratio_info)
  expect_true(pair$decoder_relative_accuracy_10 > 0, info = ratio_info)
  expect_true(pair$neuralode_relative_accuracy_10 > 0, info = ratio_info)
  expect_true(pair$decoder_iterations_per_second > 0, info = ratio_info)
  expect_true(pair$neuralode_iterations_per_second > 0, info = ratio_info)
  expect_true(decoder_over_neural >= 0.75, info = ratio_info)
  expect_true(decoder_over_neural <= 1.25, info = ratio_info)
  expect_true(neural_over_decoder >= 0.75, info = ratio_info)
  expect_true(neural_over_decoder <= 1.25, info = ratio_info)

  neural_summary <- pair$neuralode_details$summary
  neural_info <- paste(
    ratio_info,
    sprintf("neural first_loss=%.6f", neural_summary$first_loss[[1L]]),
    sprintf("neural last_loss=%.6f", neural_summary$last_loss[[1L]]),
    sep = "; "
  )
  expect_true(
    neural_summary$last_loss[[1L]] < neural_summary$first_loss[[1L]],
    info = neural_info
  )

  block_log <- pair$neuralode_details$block_update_log
  expect_true(is.data.frame(block_log), info = neural_info)
  expect_true(nrow(block_log) > 0L, info = neural_info)
  expect_true(
    all(c("iter", "block", "param_norm", "grad_norm", "update_norm", "rel_update") %in% names(block_log)),
    info = neural_info
  )
  expect_true(all(is.finite(block_log$param_norm)), info = neural_info)
  expect_true(all(is.finite(block_log$grad_norm)), info = neural_info)
  expect_true(all(is.finite(block_log$update_norm)), info = neural_info)
  expect_true(all(is.finite(block_log$rel_update)), info = neural_info)
  expect_true(
    all(c("InitProcessList", "LocalNeural", "GlobalNeural", "ScaleList", "TSList") %in% unique(block_log$block)),
    info = neural_info
  )
  moved_blocks <- stats::aggregate(
    update_norm ~ block,
    data = block_log,
    FUN = function(x) max(x, na.rm = TRUE)
  )
  expect_true(
    any(moved_blocks$block %in% c("LocalNeural", "GlobalNeural", "TSList") &
          moved_blocks$update_norm > 0),
    info = neural_info
  )

  state_metrics <- ndm_test_capture_initial_state_metrics(pair$neuralode_details, seed = 99L)
  state_info <- paste(
    neural_info,
    sprintf("mean_entropy=%.6f", state_metrics$mean_entropy),
    sprintf("mean_max_component=%.6f", state_metrics$mean_max_component),
    sep = "; "
  )
  expect_true(state_metrics$mean_entropy > 0.1, info = state_info)
  expect_true(state_metrics$mean_max_component < 0.98, info = state_info)
})

test_that("non-finite sim training fails fast and writes a debug artifact", {
  ndm_skip_if_no_sim_backend()

  details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    after_train_define = function(runtime_env, model) {
      runtime_env$train_step_compiled <- function(ModelList,
                                                  batch_pkg,
                                                  y_true,
                                                  y_mask,
                                                  iteration,
                                                  state,
                                                  PriorList,
                                                  PolicyList,
                                                  GetPredSaveAtInfo,
                                                  seed,
                                                  opt_state) {
        list(
          loss = runtime_env$jnp$array(Inf),
          state = state,
          grad_norm = runtime_env$jnp$array(Inf),
          model = ModelList,
          opt_state = opt_state
        )
      }
    },
    expect_train_error = TRUE,
    return_details = TRUE
  )

  expect_match(
    ndm:::.ndm_condition_message(details$train_error),
    "Non-finite training state"
  )

  artifact_files <- list.files(
    details$holder_folder,
    pattern = "^nonfinite_outer1_i1\\.rds$",
    full.names = TRUE
  )
  expect_length(artifact_files, 1L)

  report <- readRDS(artifact_files[[1L]])
  expect_identical(report$outer_iteration, 1L)
  expect_identical(report$base_id, 1L)
  expect_true(is.infinite(report$loss))
  expect_true("batch_YTrue_out" %in% names(report$tensor_summaries))
  expect_true("prediction_center_param" %in% names(report$tensor_summaries))
})

test_that("GetPred save-at horizon remains a static host integer in sim runtimes", {
  ndm_skip_if_no_sim_backend()

  details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    return_details = TRUE
  )

  saveat_info <- details$runtime_env$GetPredSaveAtInfo_default
  expect_true(is.list(saveat_info))
  expect_false("python.builtin.object" %in% class(saveat_info[[1L]]))
  expect_identical(as.integer(saveat_info[[1L]]), 4L)
})
