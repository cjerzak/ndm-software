test_that("canonical sim fixtures preserve legacy scaling workload", {
  expect_identical(
    ndm_test_sim_fit_scaling_batches(1L, 2L, 8L),
    1L
  )
  expect_identical(
    ndm_test_sim_fit_scaling_batches(2L, 12L, 8L),
    12L
  )
  expect_identical(
    ndm_test_sim_fit_scaling_batches(1L, 1L, 17L),
    2L
  )

  grid <- ndm_test_sim_fit_grid(
    sim_entry = list(),
    n_samples_train = 4L,
    n_times_lookahead = 4L,
    scaling_outer_loops = 2L,
    scaling_inner_loops = 12L,
    n_batch_sim_grid_gen = 8L
  )
  expect_identical(grid$scaling_batches, 12L)
})

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
    n_sgd = 300L,
    model_dims = 64L,
    shared_runtime_globals = ndm_test_sim_parity_shared_runtime_globals(4242L),
    neuralode_config_overrides = ndm_test_sim_parity_neural_config_overrides(),
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
  expect_equal(
    pair$decoder_week10$rss_baseline,
    pair$neuralode_week10$rss_baseline,
    tolerance = 1e-12,
    info = ratio_info
  )
  expect_equal(
    pair$decoder_week10$truth,
    pair$neuralode_week10$truth,
    tolerance = 0,
    info = ratio_info
  )

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

test_that("NeuralODE latent dims follow ModelDims by default and honor explicit overrides in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  had_local_dim <- exists("LocalNeuralEmbedDim", envir = .GlobalEnv, inherits = FALSE)
  old_local_dim <- if (had_local_dim) {
    get("LocalNeuralEmbedDim", envir = .GlobalEnv, inherits = FALSE)
  } else {
    NULL
  }
  had_global_dim <- exists("GlobalNeuralEmbedDim", envir = .GlobalEnv, inherits = FALSE)
  old_global_dim <- if (had_global_dim) {
    get("GlobalNeuralEmbedDim", envir = .GlobalEnv, inherits = FALSE)
  } else {
    NULL
  }
  on.exit({
    if (had_local_dim) {
      assign("LocalNeuralEmbedDim", old_local_dim, envir = .GlobalEnv)
    } else if (exists("LocalNeuralEmbedDim", envir = .GlobalEnv, inherits = FALSE)) {
      rm("LocalNeuralEmbedDim", envir = .GlobalEnv)
    }
    if (had_global_dim) {
      assign("GlobalNeuralEmbedDim", old_global_dim, envir = .GlobalEnv)
    } else if (exists("GlobalNeuralEmbedDim", envir = .GlobalEnv, inherits = FALSE)) {
      rm("GlobalNeuralEmbedDim", envir = .GlobalEnv)
    }
  }, add = TRUE)

  assign("LocalNeuralEmbedDim", 32L, envir = .GlobalEnv)
  assign("GlobalNeuralEmbedDim", 32L, envir = .GlobalEnv)

  default_details <- ndm_test_fit_sim_case(
    model_type = "NeuralODE",
    endogeneity = 0.0,
    n_sgd = 1L,
    model_dims = 64L,
    return_details = TRUE
  )
  expect_equal(as.integer(default_details$runtime_env$LocalNeuralEmbedDim), 64L)
  expect_equal(as.integer(default_details$runtime_env$GlobalNeuralEmbedDim), 64L)

  override_details <- ndm_test_fit_sim_case(
    model_type = "NeuralODE",
    endogeneity = 0.0,
    n_sgd = 1L,
    model_dims = 64L,
    config_overrides = list(
      neuralode_local_latent_dim = 48L,
      neuralode_global_latent_dim = 40L
    ),
    return_details = TRUE
  )
  expect_equal(as.integer(override_details$runtime_env$LocalNeuralEmbedDim), 48L)
  expect_equal(as.integer(override_details$runtime_env$GlobalNeuralEmbedDim), 40L)
})

ndm_test_expect_neuralode_smoke <- function(spec,
                                            label,
                                            model_dims = 32L,
                                            n_times_lookahead = 2L) {
  details <- ndm_test_fit_sim_case(
    model_type = "NeuralODE",
    endogeneity = 0.0,
    n_sgd = 1L,
    model_dims = model_dims,
    n_times_lookahead = n_times_lookahead,
    model_spec = spec,
    return_details = TRUE
  )
  pred <- ndm_predict(
    details$trained,
    batch = details$batch,
    seed = 123L,
    update_state = FALSE
  )
  pred_mu <- details$runtime_env$np$asanyarray(pred$y_mu)
  grad_norm <- as.numeric(details$trained$env$grad_norm_vec[[1L]])
  loss_value <- as.numeric(details$summary$last_loss[[1L]])

  expect_equal(details$summary$spec_preset[[1L]], spec$preset, info = label)
  expect_identical(as.character(details$runtime_env$InitStateTerms), spec$init_state_terms, info = label)
  expect_true(all(is.finite(pred_mu)), info = label)
  expect_true(length(dim(pred_mu)) >= 1L, info = label)
  expect_true(is.finite(loss_value), info = label)
  expect_true(is.finite(grad_norm), info = label)
  expect_false(is.null(details$trained$opt_state), info = label)
  expect_equal(
    as.integer(sum(details$runtime_env$local_base_prior_mask_matched)),
    as.integer(details$runtime_env$nODEParams_base),
    info = label
  )
  expect_equal(
    as.integer(sum(details$runtime_env$local_neural_prior_mask_matched)),
    as.integer(details$runtime_env$nODEParams_neural),
    info = label
  )
  details$smoke_pred <- pred

  details
}

test_that("NeuralODE init-state mapping survives interleaved dynamic-state order in custom tex specs", {
  ndm_skip_if_no_sim_backend()

  base_spec <- ndm_model_spec(preset = "seirs_dynamic_beta", model_type = "NeuralODE")
  lines <- strsplit(ndm_model_spec_to_tex(base_spec), "\n", fixed = TRUE)[[1L]]
  idx_s <- grep("Evolve{s_l}", lines, fixed = TRUE)[[1L]]
  idx_e <- grep("Evolve{e_l}", lines, fixed = TRUE)[[1L]]
  idx_i <- grep("Evolve{i_l}", lines, fixed = TRUE)[[1L]]
  idx_r <- grep("Evolve{r_l}", lines, fixed = TRUE)[[1L]]
  idx_beta <- grep("Evolve{\\beta_l}", lines, fixed = TRUE)[[1L]]
  idx_p <- grep("Evolve{p_l}", lines, fixed = TRUE)[[1L]]
  block_idx <- c(idx_s, idx_e, idx_i, idx_r, idx_beta, idx_p)
  lines[block_idx] <- lines[c(idx_s, idx_beta, idx_e, idx_i, idx_r, idx_p)]

  interleaved_path <- tempfile(fileext = ".tex")
  writeLines(lines, interleaved_path, useBytes = TRUE)
  interleaved_spec <- ndm_model_spec_from_tex(interleaved_path, model_type = "NeuralODE")

  details <- ndm_test_fit_sim_case(
    model_type = "NeuralODE",
    endogeneity = 0.0,
    n_sgd = 1L,
    config_overrides = list(
      neuralode_init_state_logit_offset = c(s_l = 20, e_l = -20, i_l = -20, r_l = -20),
      neuralode_init_state_logit_scale_max = 1.0
    ),
    model_spec = interleaved_spec,
    return_details = TRUE
  )

  expect_identical(as.character(details$runtime_env$InitStateTerms), c("s_l", "e_l", "i_l", "r_l"))
  state_metrics <- ndm_test_capture_initial_state_metrics(details, seed = 11L)
  expect_gt(unname(state_metrics$mean_components[["s"]]), 0.8)
  expect_lt(unname(state_metrics$mean_components[["e"]]), 0.1)
  expect_lt(unname(state_metrics$mean_components[["i"]]), 0.1)
  expect_lt(unname(state_metrics$mean_components[["r"]]), 0.1)
})

test_that("custom NeuralODE init-state layouts train in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  custom_spec <- ndm_model_spec_from_structure(
    list(
      preset = "custom_slqi",
      description = "Custom four-state latent structure with a holding compartment.",
      states = c("s_l", "l_l", "q_l", "i_l"),
      parameters = list(
        lambda = ndm:::.ndm_param_spec("InvSoftPlus", prior_mean = 0.08, prior_sd = 0.25),
        c = ndm:::.ndm_param_spec("InvSoftPlus", prior_mean = 0.05, prior_sd = 0.25),
        h = ndm:::.ndm_param_spec("InvSoftPlus", prior_mean = 0.03, prior_sd = 0.25)
      ),
      equations = c(
        s_l = "- lambda * s_l",
        l_l = "lambda * s_l - (c + h) * l_l",
        q_l = "h * l_l",
        i_l = "c * l_l"
      ),
      observations = "i_l"
    ),
    model_type = "NeuralODE"
  )

  details <- ndm_test_expect_neuralode_smoke(
    spec = custom_spec,
    label = "custom_slqi"
  )

  expect_true("q_l" %in% as.character(details$runtime_env$InitStateTerms))
})

test_that("NeuralODE local-only dynamic parameters do not emit empty args lists in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  base_spec <- ndm_model_spec_from_structure(
    list(
      preset = "custom_local_only_neural_args",
      description = "Minimal two-state model with all structural parameters in Neural1.",
      states = c("s_l", "i_l"),
      parameters = list(
        lambda_l = ndm:::.ndm_param_spec("InvSoftPlus", prior_mean = 0.08, prior_sd = 0.25)
      ),
      equations = c(
        s_l = "- lambda_l * s_l",
        i_l = "lambda_l * s_l"
      ),
      observations = "i_l"
    ),
    model_type = "NeuralODE"
  )
  lines <- strsplit(ndm_model_spec_to_tex(base_spec), "\n", fixed = TRUE)[[1L]]
  lines <- sub("% local_dynamic_terms:.*", "% local_dynamic_terms: lambda_l", lines)
  lines <- sub("% time_varying_terms:.*", "% time_varying_terms: lambda_l", lines)
  lines <- sub("% endogenous_terms:.*", "% endogenous_terms: lambda_l", lines)
  insert_after <- grep("Evolve{i_l}", lines, fixed = TRUE)[[1L]]
  lines <- append(
    lines,
    "\\item $\\Evolve{lambda_l} = \\Neural1{ lambda_l , s_l , i_l }$",
    after = insert_after
  )

  local_only_path <- tempfile(fileext = ".tex")
  writeLines(lines, local_only_path, useBytes = TRUE)
  local_only_spec <- ndm_model_spec_from_tex(local_only_path, model_type = "NeuralODE")

  details <- ndm_test_expect_neuralode_smoke(
    spec = local_only_spec,
    label = "custom_local_only_neural_args",
    model_dims = 16L
  )

  expect_identical(as.character(details$runtime_env$uq_args_vec), character(0))
  expect_identical(as.character(details$runtime_env$uq_globalneural_vec), character(0))
  expect_identical(as.character(details$runtime_env$uq_encneural_vec), "lambda_l")
})

test_that("NeuralODE generalized structures smoke-test across SEIRS and the 12 TB forms", {
  ndm_skip_if_no_sim_backend()

  cases <- list(
    list(label = "seirs_dynamic_beta", spec = ndm_model_spec(preset = "seirs_dynamic_beta", model_type = "NeuralODE")),
    list(label = "tb_a", spec = ndm_model_spec(preset = "tb_a", model_type = "NeuralODE")),
    list(label = "tb_b_n2", spec = ndm_model_spec(preset = "tb_b", model_type = "NeuralODE", family_args = list(n = 2L))),
    list(label = "tb_b_n5", spec = ndm_model_spec(preset = "tb_b", model_type = "NeuralODE", family_args = list(n = 5L))),
    list(label = "tb_c", spec = ndm_model_spec(preset = "tb_c", model_type = "NeuralODE")),
    list(label = "tb_d", spec = ndm_model_spec(preset = "tb_d", model_type = "NeuralODE")),
    list(label = "tb_e", spec = ndm_model_spec(preset = "tb_e", model_type = "NeuralODE")),
    list(label = "tb_f", spec = ndm_model_spec(preset = "tb_f", model_type = "NeuralODE")),
    list(label = "tb_g", spec = ndm_model_spec(preset = "tb_g", model_type = "NeuralODE")),
    list(label = "tb_h", spec = ndm_model_spec(preset = "tb_h", model_type = "NeuralODE")),
    list(label = "tb_i", spec = ndm_model_spec(preset = "tb_i", model_type = "NeuralODE")),
    list(label = "tb_j_n2", spec = ndm_model_spec(preset = "tb_j", model_type = "NeuralODE", family_args = list(n = 2L))),
    list(label = "tb_j_n5", spec = ndm_model_spec(preset = "tb_j", model_type = "NeuralODE", family_args = list(n = 5L))),
    list(label = "tb_k", spec = ndm_model_spec(preset = "tb_k", model_type = "NeuralODE")),
    list(label = "tb_l", spec = ndm_model_spec(preset = "tb_l", model_type = "NeuralODE"))
  )

  results <- lapply(
    cases,
    function(case) {
      details <- ndm_test_expect_neuralode_smoke(
        spec = case$spec,
        label = case$label
      )
      if (identical(case$spec$compartments, "tb")) {
        observed_expr <- as.character(details$runtime_env$observed_vec_final)
        expect_equal(length(observed_expr), length(case$spec$execution_spec$observations), info = case$label)
        expect_false(any(grepl("diff_eq_sol$ys$i_l", observed_expr, fixed = TRUE)), info = case$label)
      }
      data.frame(
        label = case$label,
        preset = case$spec$preset,
        n_states = length(case$spec$state_terms),
        last_loss = details$summary$last_loss[[1L]],
        grad_norm = as.numeric(details$trained$env$grad_norm_vec[[1L]]),
        stringsAsFactors = FALSE
      )
    }
  )
  results_df <- do.call(rbind, results)
  results_info <- paste(capture.output(print(results_df)), collapse = "\n")

  expect_equal(nrow(results_df), length(cases))
  expect_true(all(is.finite(results_df$last_loss)), info = results_info)
  expect_true(all(is.finite(results_df$grad_norm)), info = results_info)
  expect_true(all(results_df$n_states >= 2L), info = results_info)
})

test_that("TB NeuralODE structures complete finite forward/backward passes with invariant-respecting state trajectories", {
  ndm_skip_if_no_sim_backend()

  cases <- list(
    list(label = "tb_a", spec = ndm_model_spec(preset = "tb_a", model_type = "NeuralODE")),
    list(label = "tb_b_n3", spec = ndm_model_spec(preset = "tb_b", model_type = "NeuralODE", family_args = list(n = 3L))),
    list(label = "tb_c", spec = ndm_model_spec(preset = "tb_c", model_type = "NeuralODE")),
    list(label = "tb_d", spec = ndm_model_spec(preset = "tb_d", model_type = "NeuralODE")),
    list(label = "tb_e", spec = ndm_model_spec(preset = "tb_e", model_type = "NeuralODE")),
    list(label = "tb_f", spec = ndm_model_spec(preset = "tb_f", model_type = "NeuralODE")),
    list(label = "tb_g", spec = ndm_model_spec(preset = "tb_g", model_type = "NeuralODE")),
    list(label = "tb_h", spec = ndm_model_spec(preset = "tb_h", model_type = "NeuralODE")),
    list(label = "tb_i", spec = ndm_model_spec(preset = "tb_i", model_type = "NeuralODE")),
    list(label = "tb_j_n3", spec = ndm_model_spec(preset = "tb_j", model_type = "NeuralODE", family_args = list(n = 3L))),
    list(label = "tb_k", spec = ndm_model_spec(preset = "tb_k", model_type = "NeuralODE")),
    list(label = "tb_l", spec = ndm_model_spec(preset = "tb_l", model_type = "NeuralODE"))
  )

  results <- vector("list", length(cases))
  for (i in seq_along(cases)) {
    case <- cases[[i]]
    details <- ndm_test_expect_neuralode_smoke(
      spec = case$spec,
      label = case$label
    )
    diagnostics <- ndm_test_capture_tb_structure_diagnostics(
      details = details,
      spec = case$spec,
      label = case$label
    )

    case_info <- paste(
      case$label,
      sprintf("preset=%s", case$spec$preset),
      sprintf("n_states=%s", length(case$spec$state_terms)),
      sprintf("min_state=%.6g", diagnostics$min_state),
      sprintf("max_rel_mass_drift=%.6g", diagnostics$max_rel_mass_drift),
      sprintf("last_loss=%.6g", details$summary$last_loss[[1L]]),
      sprintf("grad_norm=%.6g", as.numeric(details$trained$env$grad_norm_vec[[1L]])),
      sep = "; "
    )

    expect_true(all(is.finite(diagnostics$pred_mu)), info = case_info)
    expect_true(all(is.finite(diagnostics$pred_sigma)), info = case_info)
    expect_true(all(is.finite(diagnostics$state_cube)), info = case_info)
    expect_true(diagnostics$min_state >= -diagnostics$state_epsilon, info = case_info)
    expect_true(diagnostics$max_rel_mass_drift <= 5e-3, info = case_info)
    if (length(case$spec$time_varying_terms) > 0L) {
      expect_true(diagnostics$time_varying_all_finite, info = case_info)
      expect_true(diagnostics$time_varying_min >= -diagnostics$state_epsilon, info = case_info)
    }

    results[[i]] <- data.frame(
      label = case$label,
      preset = case$spec$preset,
      n_states = length(case$spec$state_terms),
      min_state = diagnostics$min_state,
      state_epsilon = diagnostics$state_epsilon,
      max_rel_mass_drift = diagnostics$max_rel_mass_drift,
      last_loss = details$summary$last_loss[[1L]],
      grad_norm = as.numeric(details$trained$env$grad_norm_vec[[1L]]),
      n_time_varying_terms = length(case$spec$time_varying_terms),
      time_varying_all_finite = diagnostics$time_varying_all_finite,
      time_varying_min = diagnostics$time_varying_min,
      stringsAsFactors = FALSE
    )
  }

  results_df <- do.call(rbind, results)
  results_info <- paste(capture.output(print(results_df)), collapse = "\n")

  expect_equal(nrow(results_df), length(cases))
  expect_true(all(is.finite(results_df$last_loss)), info = results_info)
  expect_true(all(is.finite(results_df$grad_norm)), info = results_info)
  expect_true(all(results_df$min_state >= (-1 * results_df$state_epsilon)), info = results_info)
  expect_true(all(results_df$max_rel_mass_drift <= 5e-3), info = results_info)
  expect_true(
    all(results_df$time_varying_all_finite[results_df$n_time_varying_terms > 0L]),
    info = results_info
  )
  expect_true(
    all(is.finite(results_df$time_varying_min[results_df$n_time_varying_terms > 0L])),
    info = results_info
  )
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
