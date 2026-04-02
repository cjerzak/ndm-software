# Shared transformer regression tests that are narrower than the heavier
# end-to-end sim-fit suite.

ndm_test_transformer_source_root <- function() {
  source_root <- try(ndm:::.ndm_runtime_source_root(), silent = TRUE)
  if (inherits(source_root, "try-error") || is.null(source_root) || !dir.exists(source_root)) {
    start_paths <- unique(c(
      getwd(),
      tryCatch(testthat::test_path("..", ".."), error = function(...) ""),
      system.file(package = "ndm")
    ))
    candidate_roots <- unique(unlist(lapply(start_paths[nzchar(start_paths)], function(path) {
      path <- normalizePath(path, winslash = "/", mustWork = FALSE)
      parents <- character(7L)
      parents[[1L]] <- path
      for (i in 1:6) {
        parents[[i + 1L]] <- normalizePath(
          file.path(parents[[i]], ".."),
          winslash = "/",
          mustWork = FALSE
        )
      }
      unique(parents[nzchar(parents)])
    })))

    for (root in candidate_roots) {
      candidate <- file.path(root, "tools", "runtime_source", "ndm_runtime")
      if (dir.exists(candidate)) {
        source_root <- candidate
        break
      }
    }
  }
  if (inherits(source_root, "try-error") || is.null(source_root) || !dir.exists(source_root)) {
    skip("runtime source tree is not available in this installed-package context")
  }
  normalizePath(source_root, winslash = "/", mustWork = TRUE)
}

ndm_test_py_numeric <- function(x) {
  as.numeric(reticulate::py_to_r(x))
}

ndm_test_max_abs_diff <- function(lhs, rhs) {
  max(abs(lhs - rhs))
}

ndm_test_capture_decoder_prediction <- function(runtime_env, batch, seed = 1L) {
  getpred_saveat_info <- if (exists("ndm_runtime_normalize_getpred_saveat_info", envir = runtime_env, inherits = TRUE)) {
    runtime_env$ndm_runtime_normalize_getpred_saveat_info(runtime_env$GetPredSaveAtInfo_default)
  } else {
    runtime_env$GetPredSaveAtInfo_default
  }
  seed_matrix <- runtime_env$jax$random$split(
    runtime_env$jax$random$PRNGKey(as.integer(seed)),
    as.integer(runtime_env$nBatch)
  )
  if (exists("ndm_runtime_data_to_device", envir = runtime_env, inherits = TRUE)) {
    seed_matrix <- runtime_env$ndm_runtime_data_to_device(seed_matrix)
  }

  pred <- runtime_env$GetPred_train_jit(
    runtime_env$ModelList,
    runtime_env$batch2package(batch),
    runtime_env$state,
    runtime_env$PriorList,
    runtime_env$PolicyList,
    getpred_saveat_info,
    seed_matrix
  )[[1L]]

  list(
    y_mu = ndm_test_py_numeric(runtime_env$np$asanyarray(pred$y_mu)),
    center_param = ndm_test_py_numeric(runtime_env$np$asanyarray(pred$ODEParamsSampList$center_param))
  )
}

ndm_test_capture_metadata_snapshot <- function(init_process_list, runtime_env) {
  list(
    place_fixed = isTRUE(init_process_list$PlaceEmbedsFixed),
    place_embeds = ndm_test_py_numeric(runtime_env$np$asanyarray(init_process_list$PlaceEmbeds)),
    time_embeds = ndm_test_py_numeric(runtime_env$np$asanyarray(init_process_list$TimeEmbeds)),
    place_proj = ndm_test_py_numeric(runtime_env$np$asanyarray(init_process_list$PlaceEmbeds_Proj$weight)),
    time_proj = ndm_test_py_numeric(runtime_env$np$asanyarray(init_process_list$TimeEmbeds_Proj$weight))
  )
}

test_that("transformer runtime defaults to full attention residuals and preserves backbone output contracts in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  captured <- new.env(parent = emptyenv())
  details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    return_details = TRUE,
    before_train = function(runtime_env, model) {
      layer <- model$env$ModelList$TSList$TSBackbone$d1
      captured$use_full_attnres <- isTRUE(model$env$ModelList$TSList$TSBackbone$UseFullAttentionResiduals)
      captured$num_heads <- as.integer(runtime_env$TransformerHeads)
      captured$num_kv_heads <- as.integer(runtime_env$TransformerKVHeads)
      captured$attn_query <- ndm_test_py_numeric(runtime_env$np$asanyarray(layer$AttnRes1$PseudoQuery))
      captured$ffn_query <- ndm_test_py_numeric(runtime_env$np$asanyarray(layer$AttnRes2$PseudoQuery))
      captured$q_norm_shape <- as.integer(unlist(reticulate::py_to_r(layer$Multihead$QNormScale$shape)))
      captured$k_norm_shape <- as.integer(unlist(reticulate::py_to_r(layer$Multihead$KNormScale$shape)))
      captured$q_norm <- ndm_test_py_numeric(runtime_env$np$asanyarray(layer$Multihead$QNormScale))
      captured$k_norm <- ndm_test_py_numeric(runtime_env$np$asanyarray(layer$Multihead$KNormScale))
    }
  )

  expect_true(captured$use_full_attnres)
  expect_equal(captured$attn_query, rep(0, as.integer(details$runtime_env$ModelDims)))
  expect_equal(captured$ffn_query, rep(0, as.integer(details$runtime_env$ModelDims)))
  expect_equal(captured$q_norm_shape, c(captured$num_heads, 1L))
  expect_equal(captured$k_norm_shape, c(captured$num_kv_heads, 1L))
  expect_equal(captured$q_norm, rep(1, captured$num_heads))
  expect_equal(captured$k_norm, rep(1, captured$num_kv_heads))

  hidden_state <- details$runtime_env$jnp$zeros(list(as.integer(details$runtime_env$ModelDims)))
  bridge_out <- details$runtime_env$DecoderBackboneToOutput(
    TSList = details$model$env$ModelList$TSList,
    hidden_state = hidden_state
  )
  bridge_shape <- as.integer(unlist(reticulate::py_to_r(bridge_out$shape)))
  proj_out <- details$model$env$ModelList$TSList$TSBackbone$DecoderProj(bridge_out)
  proj_shape <- as.integer(unlist(reticulate::py_to_r(proj_out$shape)))

  expect_equal(bridge_shape, as.integer(details$runtime_env$ModelDims))
  expect_equal(proj_shape, as.integer(details$runtime_env$nOutcomes))
})

test_that("default full attention residual transformer builds and trains through the no-cache decoder path in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  captured <- new.env(parent = emptyenv())
  details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    enable_kv_cache = FALSE,
    return_details = TRUE,
    before_train = function(runtime_env, model) {
      layer <- model$env$ModelList$TSList$TSBackbone$d1
      captured$use_full_attnres <- isTRUE(model$env$ModelList$TSList$TSBackbone$UseFullAttentionResiduals)
      captured$attn_query <- ndm_test_py_numeric(runtime_env$np$asanyarray(layer$AttnRes1$PseudoQuery))
      captured$ffn_query <- ndm_test_py_numeric(runtime_env$np$asanyarray(layer$AttnRes2$PseudoQuery))
    }
  )

  expect_true(captured$use_full_attnres)
  expect_equal(captured$attn_query, rep(0, as.integer(details$runtime_env$ModelDims)))
  expect_equal(captured$ffn_query, rep(0, as.integer(details$runtime_env$ModelDims)))
  expect_true(is.finite(details$summary$first_loss[[1L]]))
  expect_true(is.finite(details$summary$last_loss[[1L]]))
  expect_true(is.finite(details$iterations_per_second))
  expect_gt(details$iterations_per_second, 0)
})

test_that("cached and uncached full attention residual decoder predictions stay aligned in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  cached_capture <- new.env(parent = emptyenv())
  uncached_capture <- new.env(parent = emptyenv())

  ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    case_seed = 314L,
    n_sgd = 1L,
    enable_kv_cache = TRUE,
    after_train_define = function(runtime_env, model) {
      batch <- if (exists("batch_l_cal", envir = runtime_env, inherits = FALSE)) {
        get("batch_l_cal", envir = runtime_env, inherits = FALSE)
      } else {
        runtime_env$TFConst2JAXArray(
          reticulate::iter_next(reticulate::as_iterator(get("TFDataset_train", envir = runtime_env, inherits = FALSE)))
        )
      }
      cached_capture$pred <- ndm_test_capture_decoder_prediction(runtime_env, batch, seed = 17L)
    }
  )

  ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    case_seed = 314L,
    n_sgd = 1L,
    enable_kv_cache = FALSE,
    after_train_define = function(runtime_env, model) {
      batch <- if (exists("batch_l_cal", envir = runtime_env, inherits = FALSE)) {
        get("batch_l_cal", envir = runtime_env, inherits = FALSE)
      } else {
        runtime_env$TFConst2JAXArray(
          reticulate::iter_next(reticulate::as_iterator(get("TFDataset_train", envir = runtime_env, inherits = FALSE)))
        )
      }
      uncached_capture$pred <- ndm_test_capture_decoder_prediction(runtime_env, batch, seed = 17L)
    }
  )

  expect_lt(
    ndm_test_max_abs_diff(cached_capture$pred$y_mu, uncached_capture$pred$y_mu),
    1e-4
  )
  expect_lt(
    ndm_test_max_abs_diff(cached_capture$pred$center_param, uncached_capture$pred$center_param),
    1e-4
  )
})

test_that("legacy residual transformer remains available through UseFullAttentionResiduals = FALSE in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  captured <- new.env(parent = emptyenv())
  details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    return_details = TRUE,
    runtime_globals = list(
      UseFullAttentionResiduals = FALSE
    ),
    before_train = function(runtime_env, model) {
      layer <- model$env$ModelList$TSList$TSBackbone$d1
      capture_pair <- function(pair) {
        list(
          skip = ndm_test_py_numeric(
            runtime_env$np$asanyarray(runtime_env$jax$nn$softplus(pair$WtSkipPath))
          ),
          resid = ndm_test_py_numeric(
            runtime_env$np$asanyarray(runtime_env$jax$nn$softplus(pair$WtResidPath))
          )
        )
      }

      captured$model_depth <- as.integer(runtime_env$ModelDepth)
      captured$use_full_attnres <- isTRUE(model$env$ModelList$TSList$TSBackbone$UseFullAttentionResiduals)
      captured$has_attnres_1 <- !is.null(layer$AttnRes1)
      captured$has_attnres_2 <- !is.null(layer$AttnRes2)
      captured$resid_con_1 <- capture_pair(layer$ResidCon1)
      captured$resid_con_2 <- capture_pair(layer$ResidCon2)
    }
  )

  expected_resid <- sqrt(1 / (1 + 0.01^2 * captured$model_depth))
  expected_skip <- sqrt(1 - expected_resid^2)

  expect_false(captured$use_full_attnres)
  expect_false(captured$has_attnres_1)
  expect_false(captured$has_attnres_2)

  for (pair in list(captured$resid_con_1, captured$resid_con_2)) {
    expect_true(all(is.finite(pair$skip)))
    expect_true(all(is.finite(pair$resid)))
    expect_gt(mean(pair$resid), mean(pair$skip))
    expect_lt(max(abs(pair$skip - expected_skip)), 1e-4)
    expect_lt(max(abs(pair$resid - expected_resid)), 1e-4)
  }

  expect_true(is.finite(details$summary$first_loss[[1L]]))
  expect_true(is.finite(details$summary$last_loss[[1L]]))
  expect_true(is.finite(details$iterations_per_second))
  expect_gt(details$iterations_per_second, 0)
})

test_that("maintained transformer sources keep cache and rotary guardrails", {
  source_root <- ndm_test_transformer_source_root()
  buildml_source <- paste(
    readLines(
      file.path(source_root, "ModelDefiners", "SuperLModel_BuildML.R"),
      warn = FALSE
    ),
    collapse = "\n"
  )
  backbone_source <- paste(
    readLines(
      file.path(source_root, "ModelDefiners", "SuperLModel_BackboneTransformer.R"),
      warn = FALSE
    ),
    collapse = "\n"
  )

  expect_match(buildml_source, "DecoderBackboneToOutput <- function\\(TSList, hidden_state\\)")
  expect_match(buildml_source, "jnp\\$expand_dims\\(xt_last, 0L\\)")
  expect_match(buildml_source, "jnp\\$expand_dims\\(embed_out, 0L\\)")
  expect_match(buildml_source, "DecoderProj\\(embed_out\\)")
  expect_match(buildml_source, "PlaceEmbeds_Proj\\(place_embed\\)")
  expect_match(buildml_source, "TimeEmbeds_Proj\\(time_embed\\)")
  expect_false(grepl("InitProcessList\\$PlaceEmbeds <- jax\\$lax\\$stop_gradient", buildml_source))
  expect_false(grepl("InitProcessList\\$TimeEmbeds <- jax\\$lax\\$stop_gradient", buildml_source))
  expect_match(backbone_source, "qk_normalize_heads <- function")
  expect_match(backbone_source, "full_attnres_reduce_buffer <- function")
  expect_match(backbone_source, "attnres_append <- function")
  expect_match(backbone_source, "QNormScale")
  expect_match(backbone_source, "KNormScale")
  expect_false(grepl("jnp\\$stack\\(layer_outputs", backbone_source))
  expect_false(grepl("RotaryPositionalEmbedding", paste(buildml_source, backbone_source), fixed = TRUE))
})

test_that("optional residual attention benchmark warms up and syncs before timing", {
  if (!identical(tolower(Sys.getenv("NDM_RUN_BENCHMARKS", unset = "false")), "true")) {
    skip("Set NDM_RUN_BENCHMARKS=true to run optional transformer microbenchmarks")
  }
  ndm_skip_if_no_sim_backend()

  details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    return_details = TRUE
  )

  env <- details$trained$env
  model_list <- env$ModelList
  batch <- details$batch
  x_sample <- list(
    env$jnp$take(batch$XPred, 0L, axis = 0L),
    env$jnp$take(batch$XPred_mask, 0L, axis = 0L)
  )
  place_idx <- env$jnp$squeeze(env$jnp$take(batch$location_id_numeric, 0L, axis = 0L)$astype(env$jnp$int32))
  time_idx <- env$jnp$squeeze(env$jnp$take(batch$time_id_numeric, 0L, axis = 0L)$astype(env$jnp$int32))
  processed <- env$ProcessEncoderInput(
    InitProcessList = model_list$InitProcessList,
    TSList = model_list$TSList,
    xt = x_sample,
    time = time_idx,
    place = place_idx,
    BNList = model_list$BNList,
    state = env$jnp$array(1.),
    inference = TRUE
  )
  gen_cap <- as.integer(env$nTimesLookahead)
  xt_running <- list(
    env$jnp$concatenate(list(processed[[1]], env$jnp$zeros(list(gen_cap, processed[[1]]$shape[[2]]))), 0L),
    env$jnp$concatenate(list(processed[[2]], env$jnp$zeros(list(gen_cap, 1L))), 0L)
  )
  prefix_len <- env$jnp$sum(xt_running[[2]])$astype(env$jnp$int32)

  backbone_bench <- env$ndm_benchmark_helpers$benchmark(
    function() {
      env$RunTransformerBackbone(
        xt = processed[[1]],
        x_mask = processed[[2]],
        TransformerList = model_list$TSList$TSBackbone
      )
    },
    warmup = 1L,
    runs = 2L
  )
  prefill_bench <- env$ndm_benchmark_helpers$benchmark(
    function() {
      env$transformer_prefill_kv(
        xt = xt_running[[1]],
        x_mask = xt_running[[2]],
        TransformerList = model_list$TSList$TSBackbone,
        prefix_len = prefix_len
      )
    },
    warmup = 1L,
    runs = 2L
  )
  prefill_ret <- env$transformer_prefill_kv(
    xt = xt_running[[1]],
    x_mask = xt_running[[2]],
    TransformerList = model_list$TSList$TSBackbone,
    prefix_len = prefix_len
  )
  xt_last <- env$DecoderBackboneToOutput(
    TSList = model_list$TSList,
    hidden_state = prefill_ret$xt_last
  )
  xt_running[[1]] <- env$jax$lax$dynamic_update_slice(
    xt_running[[1]],
    env$jnp$expand_dims(xt_last, 0L),
    env$jnp$array(c(prefix_len, 0L), dtype = env$jnp$int32)
  )
  xt_running[[2]] <- env$jax$lax$dynamic_update_slice(
    xt_running[[2]],
    env$jnp$ones(list(1L, 1L), dtype = xt_running[[2]]$dtype),
    env$jnp$array(c(prefix_len, 0L), dtype = env$jnp$int32)
  )
  decode_bench <- env$ndm_benchmark_helpers$benchmark(
    function() {
      env$transformer_decode_step_kv(
        token_in = env$jnp$take(xt_running[[1]], prefix_len, axis = 0L),
        pos = prefix_len,
        TransformerList = model_list$TSList$TSBackbone,
        cache = prefill_ret$cache
      )
    },
    warmup = 1L,
    runs = 2L
  )

  for (bench in list(backbone_bench, prefill_bench, decode_bench)) {
    expect_equal(bench$warmup, 1L)
    expect_equal(bench$runs, 2L)
    expect_true(is.finite(bench$mean_seconds))
    expect_gt(bench$mean_seconds, 0)
  }
})

test_that("metadata tokens preserve append order while using projected embeddings in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    return_details = TRUE
  )

  env <- details$runtime_env
  model_list <- details$trained$env$ModelList
  init_process <- model_list$InitProcessList
  batch <- details$batch
  x_sample <- list(
    env$jnp$take(batch$XPred, 0L, axis = 0L),
    env$jnp$take(batch$XPred_mask, 0L, axis = 0L)
  )
  place_idx <- env$jnp$squeeze(env$jnp$take(batch$location_id_numeric, 0L, axis = 0L)$astype(env$jnp$int32))
  time_idx <- env$jnp$squeeze(env$jnp$take(batch$time_id_numeric, 0L, axis = 0L)$astype(env$jnp$int32))

  processed <- env$ProcessEncoderInput(
    InitProcessList = init_process,
    TSList = model_list$TSList,
    xt = x_sample,
    time = time_idx,
    place = place_idx,
    BNList = model_list$BNList,
    state = env$jnp$array(1.),
    inference = TRUE
  )

  content_len <- as.integer(x_sample[[1]]$shape[[1]])
  expect_false(isTRUE(init_process$PlaceEmbedsFixed))
  expect_equal(as.integer(processed[[1]]$shape[[1]]), content_len + 3L)
  expect_equal(as.integer(processed[[2]]$shape[[1]]), content_len + 3L)

  place_embed <- env$jnp$take(init_process$PlaceEmbeds, indices = place_idx, axis = 0L)
  place_embed <- init_process$PlaceEmbeds_Proj(place_embed)
  time_embed <- env$jax$lax$stop_gradient(env$jnp$take(init_process$TimeEmbeds, indices = time_idx, axis = 0L))
  time_embed <- init_process$TimeEmbeds_Proj(time_embed)

  appended_place <- env$jnp$take(processed[[1]], content_len, axis = 0L)
  appended_time <- env$jnp$take(processed[[1]], content_len + 1L, axis = 0L)
  appended_cls <- env$jnp$take(processed[[1]], content_len + 2L, axis = 0L)

  expect_lt(
    ndm_test_max_abs_diff(
      ndm_test_py_numeric(env$np$asanyarray(appended_place)),
      ndm_test_py_numeric(env$np$asanyarray(place_embed))
    ),
    1e-6
  )
  expect_lt(
    ndm_test_max_abs_diff(
      ndm_test_py_numeric(env$np$asanyarray(appended_time)),
      ndm_test_py_numeric(env$np$asanyarray(time_embed))
    ),
    1e-6
  )
  expect_lt(
    ndm_test_max_abs_diff(
      ndm_test_py_numeric(env$np$asanyarray(appended_cls)),
      ndm_test_py_numeric(env$np$asanyarray(env$jnp$squeeze(model_list$TSList$InitialCLS, 0L)))
    ),
    1e-6
  )
  expect_equal(ndm_test_py_numeric(env$np$asanyarray(env$jnp$take(processed[[2]], content_len, axis = 0L))), 1)
  expect_equal(ndm_test_py_numeric(env$np$asanyarray(env$jnp$take(processed[[2]], content_len + 1L, axis = 0L))), 1)
  expect_equal(ndm_test_py_numeric(env$np$asanyarray(env$jnp$take(processed[[2]], content_len + 2L, axis = 0L))), 1)
})

test_that("metadata projections learn while fixed metadata bases stay frozen in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  captured_default <- new.env(parent = emptyenv())
  default_details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    return_details = TRUE,
    before_train = function(runtime_env, model) {
      captured_default$init <- ndm_test_capture_metadata_snapshot(
        model$env$ModelList$InitProcessList,
        runtime_env
      )
    }
  )
  default_final <- ndm_test_capture_metadata_snapshot(
    default_details$trained$env$ModelList$InitProcessList,
    default_details$runtime_env
  )

  captured_fixed <- new.env(parent = emptyenv())
  fixed_details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    return_details = TRUE,
    runtime_globals = list(
      coordinates_mat = data.frame(
        location_id_numeric = 0L,
        lat = 47.6062,
        long = -122.3321
      )
    ),
    before_train = function(runtime_env, model) {
      captured_fixed$init <- ndm_test_capture_metadata_snapshot(
        model$env$ModelList$InitProcessList,
        runtime_env
      )
    }
  )
  fixed_final <- ndm_test_capture_metadata_snapshot(
    fixed_details$trained$env$ModelList$InitProcessList,
    fixed_details$runtime_env
  )

  expect_false(captured_default$init$place_fixed)
  expect_true(captured_fixed$init$place_fixed)

  expect_gt(ndm_test_max_abs_diff(default_final$time_proj, captured_default$init$time_proj), 1e-8)
  expect_lt(ndm_test_max_abs_diff(default_final$time_embeds, captured_default$init$time_embeds), 1e-12)
  expect_gt(ndm_test_max_abs_diff(default_final$place_proj, captured_default$init$place_proj), 1e-8)
  expect_gt(ndm_test_max_abs_diff(default_final$place_embeds, captured_default$init$place_embeds), 1e-8)

  expect_gt(ndm_test_max_abs_diff(fixed_final$time_proj, captured_fixed$init$time_proj), 1e-8)
  expect_lt(ndm_test_max_abs_diff(fixed_final$time_embeds, captured_fixed$init$time_embeds), 1e-12)
  expect_gt(ndm_test_max_abs_diff(fixed_final$place_proj, captured_fixed$init$place_proj), 1e-8)
  expect_lt(ndm_test_max_abs_diff(fixed_final$place_embeds, captured_fixed$init$place_embeds), 1e-12)
})
