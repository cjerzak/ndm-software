# Shared transformer regression tests that are narrower than the heavier
# end-to-end sim-fit suite.

ndm_test_py_numeric <- function(x) {
  as.numeric(reticulate::py_to_r(x))
}

ndm_test_max_abs_diff <- function(lhs, rhs) {
  max(abs(lhs - rhs))
}

ndm_test_decoder_batch_mask <- function(batch, runtime_env, left_pad = 0L) {
  left_pad <- as.integer(left_pad)
  mask <- ndm_test_py_numeric_array(runtime_env$np$asanyarray(batch$XPred_mask))
  if (length(dim(mask)) != 3L) {
    stop("Expected decoder XPred_mask to have shape [batch, time, 1].")
  }
  mask[] <- 1
  if (left_pad > 0L) {
    mask[, seq_len(left_pad), ] <- 0
  }
  batch$XPred_mask <- runtime_env$jnp$array(
    mask,
    dtype = batch$XPred_mask$dtype
  )
  batch
}

ndm_test_capture_model_prediction <- function(x, batch, seed = 1L, inference = FALSE) {
  runtime_env <- x$env
  pred <- ndm_predict(
    x,
    batch = batch,
    inference = inference,
    seed = seed,
    update_state = FALSE
  )

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
    enable_kv_cache_training = FALSE,
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

test_that("decoder-only transformer preserves requested kv cache setting in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  cached_capture <- new.env(parent = emptyenv())
  uncached_capture <- new.env(parent = emptyenv())

  ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    enable_kv_cache = TRUE,
    before_train = function(runtime_env, model) {
      cached_capture$enable_kv_caching <- get("EnableKVCaching", envir = runtime_env, inherits = FALSE)
      cached_capture$enable_kv_caching_training <- get(
        "EnableKVCachingTraining", envir = runtime_env, inherits = FALSE
      )
    }
  )

  ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    enable_kv_cache = FALSE,
    enable_kv_cache_training = FALSE,
    before_train = function(runtime_env, model) {
      uncached_capture$enable_kv_caching <- get("EnableKVCaching", envir = runtime_env, inherits = FALSE)
    }
  )

  expect_true(isTRUE(cached_capture$enable_kv_caching))
  expect_true(isTRUE(cached_capture$enable_kv_caching_training))
  expect_false(isTRUE(uncached_capture$enable_kv_caching))
})

test_that("cached and uncached full attention residual decoder predictions stay aligned in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  cached_details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    case_seed = 314L,
    enable_kv_cache = TRUE,
    enable_kv_cache_training = TRUE,
    return_details = TRUE
  )
  uncached_details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    case_seed = 314L,
    enable_kv_cache = FALSE,
    enable_kv_cache_training = FALSE,
    return_details = TRUE
  )

  expect_true(isTRUE(cached_details$runtime_env$EnableKVCaching))
  expect_true(isTRUE(cached_details$runtime_env$EnableKVCachingTraining))
  expect_false(isTRUE(uncached_details$runtime_env$EnableKVCaching))

  cached_capture <- ndm_test_capture_model_prediction(
    cached_details$trained,
    batch = cached_details$batch,
    seed = 17L,
    inference = FALSE
  )
  uncached_capture <- ndm_test_capture_model_prediction(
    uncached_details$trained,
    batch = uncached_details$batch,
    seed = 17L,
    inference = FALSE
  )

  expect_lt(
    ndm_test_max_abs_diff(cached_capture$y_mu, uncached_capture$y_mu),
    1e-4
  )
  expect_lt(
    ndm_test_max_abs_diff(cached_capture$center_param, uncached_capture$center_param),
    1e-4
  )
})

test_that("cached decoder supports a one-step horizon without an empty scan in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  cached_details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    n_times_lookahead = 1L,
    case_seed = 1414L,
    enable_kv_cache = TRUE,
    enable_kv_cache_training = FALSE,
    return_details = TRUE
  )
  uncached_details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    n_times_lookahead = 1L,
    case_seed = 1414L,
    enable_kv_cache = FALSE,
    enable_kv_cache_training = FALSE,
    return_details = TRUE
  )

  cached <- ndm_test_capture_model_prediction(
    cached_details$trained,
    batch = cached_details$batch,
    seed = 31L,
    inference = TRUE
  )
  uncached <- ndm_test_capture_model_prediction(
    uncached_details$trained,
    batch = uncached_details$batch,
    seed = 31L,
    inference = TRUE
  )

  expect_length(cached$y_mu, length(uncached$y_mu))
  expect_true(all(is.finite(cached$y_mu)))
  expect_lt(ndm_test_max_abs_diff(cached$y_mu, uncached$y_mu), 1e-4)
  expect_lt(
    ndm_test_max_abs_diff(cached$center_param, uncached$center_param),
    1e-4
  )
})

test_that("decoder KV cache preserves physical positions and agrees with no-cache under left padding in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  scale_capture <- new.env(parent = emptyenv())
  cached_details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    case_seed = 2718L,
    enable_kv_cache = TRUE,
    before_train = function(runtime_env, model) {
      scale_list <- model$env$ModelList$ScaleList$ScaleBayes
      scale_capture$var_learned <- ndm_test_py_numeric(
        runtime_env$np$asanyarray(runtime_env$SoftPlus(scale_list$VarInit))
      )
      scale_capture$decoder_learned <- ndm_test_py_numeric(
        runtime_env$np$asanyarray(runtime_env$SoftPlus(scale_list$DecoderObservationScale))
      )
      scale_capture$floor <- as.numeric(runtime_env$ObservationScaleFloor)
    },
    return_details = TRUE
  )
  uncached_details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    case_seed = 2718L,
    enable_kv_cache = FALSE,
    enable_kv_cache_training = FALSE,
    return_details = TRUE
  )
  env <- cached_details$runtime_env
  uncached_env <- uncached_details$runtime_env
  expect_true(all(is.finite(scale_capture$var_learned)))
  expect_true(all(is.finite(scale_capture$decoder_learned)))
  expect_equal(
    scale_capture$var_learned,
    rep(1 - scale_capture$floor, length(scale_capture$var_learned)),
    tolerance = 1e-7
  )
  expect_equal(
    scale_capture$decoder_learned,
    rep(1 - scale_capture$floor, length(scale_capture$decoder_learned)),
    tolerance = 1e-7
  )
  expect_equal(
    scale_capture$floor + scale_capture$var_learned,
    rep(1, length(scale_capture$var_learned)),
    tolerance = 1e-7
  )
  expect_equal(
    scale_capture$floor + scale_capture$decoder_learned,
    rep(1, length(scale_capture$decoder_learned)),
    tolerance = 1e-7
  )

  for (left_pad in c(0L, 3L)) {
    cached_batch <- ndm_test_decoder_batch_mask(
      cached_details$batch,
      runtime_env = env,
      left_pad = left_pad
    )
    uncached_batch <- ndm_test_decoder_batch_mask(
      uncached_details$batch,
      runtime_env = uncached_env,
      left_pad = left_pad
    )
    cached_capture <- ndm_test_capture_model_prediction(
      cached_details$trained,
      batch = cached_batch,
      seed = 23L,
      inference = TRUE
    )
    uncached_capture <- ndm_test_capture_model_prediction(
      uncached_details$trained,
      batch = uncached_batch,
      seed = 23L,
      inference = TRUE
    )

    expect_lt(
      ndm_test_max_abs_diff(cached_capture$y_mu, uncached_capture$y_mu),
      1e-4,
      label = sprintf("cache/no-cache y_mu difference with left_pad=%s", left_pad)
    )
    expect_lt(
      ndm_test_max_abs_diff(cached_capture$center_param, uncached_capture$center_param),
      1e-4,
      label = sprintf("cache/no-cache center_param difference with left_pad=%s", left_pad)
    )
  }

  left_batch <- ndm_test_decoder_batch_mask(
    cached_details$batch,
    runtime_env = env,
    left_pad = 3L
  )
  model_list <- cached_details$trained$env$ModelList
  x_sample <- list(
    env$jnp$take(left_batch$XPred, 0L, axis = 0L),
    env$jnp$take(left_batch$XPred_mask, 0L, axis = 0L)
  )
  place_idx <- env$jnp$squeeze(
    env$jnp$take(left_batch$location_id_numeric, 0L, axis = 0L)$astype(env$jnp$int32)
  )
  time_idx <- env$jnp$squeeze(
    env$jnp$take(left_batch$time_id_numeric, 0L, axis = 0L)$astype(env$jnp$int32)
  )
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
    env$jnp$concatenate(
      list(processed[[1]], env$jnp$zeros(list(gen_cap, processed[[1]]$shape[[2]]))),
      0L
    ),
    env$jnp$concatenate(
      list(processed[[2]], env$jnp$zeros(list(gen_cap, 1L))),
      0L
    )
  )
  prefill <- env$transformer_prefill_kv(
    xt = xt_running[[1]],
    x_mask = xt_running[[2]],
    TransformerList = model_list$TSList$TSBackbone
  )
  expected_valid <- ndm_test_py_numeric(
    env$np$asanyarray(env$jnp$squeeze(xt_running[[2]], 1L))
  ) > 0
  cache_valid <- ndm_test_py_numeric(
    env$np$asanyarray(prefill$cache$d1$valid)
  ) > 0
  expected_last <- max(which(expected_valid)) - 1L

  expect_identical(cache_valid, expected_valid)
  expect_equal(
    as.integer(ndm_test_py_numeric(env$np$asanyarray(prefill$last_valid))),
    expected_last
  )
  expect_equal(
    as.integer(ndm_test_py_numeric(env$np$asanyarray(prefill$next_pos))),
    expected_last + 1L
  )
  expect_true(cache_valid[[expected_last + 1L]])
  expect_false(any(cache_valid[seq_len(3L)]))

  decoder_input <- env$DecoderBackboneToOutput(
    TSList = model_list$TSList,
    hidden_state = prefill$xt_last
  )
  decoded <- env$transformer_decode_step_kv(
    token_in = decoder_input,
    pos = prefill$next_pos,
    TransformerList = model_list$TSList$TSBackbone,
    cache = prefill$cache
  )
  decoded_valid <- ndm_test_py_numeric(
    env$np$asanyarray(decoded$cache$d1$valid)
  ) > 0
  expected_decoded_valid <- expected_valid
  expected_decoded_valid[[expected_last + 2L]] <- TRUE
  expect_identical(decoded_valid, expected_decoded_valid)

  expect_error({
    overflow <- env$transformer_decode_step_kv(
      token_in = decoder_input,
      pos = as.integer(prefill$cache$d1$k$shape[[1]]),
      TransformerList = model_list$TSList$TSBackbone,
      cache = prefill$cache
    )
    env$np$array(overflow$token_out)
  }, "outside the allocated capacity")

  legacy_cached_details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    case_seed = 1618L,
    enable_kv_cache = TRUE,
    runtime_globals = list(UseFullAttentionResiduals = FALSE),
    return_details = TRUE
  )
  legacy_uncached_details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    case_seed = 1618L,
    enable_kv_cache = FALSE,
    enable_kv_cache_training = FALSE,
    runtime_globals = list(UseFullAttentionResiduals = FALSE),
    return_details = TRUE
  )
  legacy_cached_batch <- ndm_test_decoder_batch_mask(
    legacy_cached_details$batch,
    runtime_env = legacy_cached_details$runtime_env,
    left_pad = 3L
  )
  legacy_uncached_batch <- ndm_test_decoder_batch_mask(
    legacy_uncached_details$batch,
    runtime_env = legacy_uncached_details$runtime_env,
    left_pad = 3L
  )
  legacy_cached <- ndm_test_capture_model_prediction(
    legacy_cached_details$trained,
    batch = legacy_cached_batch,
    seed = 29L,
    inference = TRUE
  )
  legacy_uncached <- ndm_test_capture_model_prediction(
    legacy_uncached_details$trained,
    batch = legacy_uncached_batch,
    seed = 29L,
    inference = TRUE
  )
  expect_lt(
    ndm_test_max_abs_diff(legacy_cached$y_mu, legacy_uncached$y_mu),
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
  buildml_source <- ndm_test_runtime_source_text(
    "ModelDefiners/SuperLModel_BuildML.R"
  )
  backbone_source <- ndm_test_runtime_source_text(
    "ModelDefiners/SuperLModel_BackboneTransformer.R"
  )

  expect_match(buildml_source, "DecoderBackboneToOutput <- function\\(TSList, hidden_state\\)")
  expect_match(buildml_source, "jnp\\$expand_dims\\(xt_last, 0L\\)")
  expect_match(buildml_source, "init = list\\(xt_last, kv_cache, insert_pos\\)")
  expect_match(buildml_source, "UseKVCachingForCall")
  expect_match(buildml_source, "EnableKVCachingTraining")
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
  expect_match(backbone_source, "\"valid\" =")
  expect_match(backbone_source, "\"last_valid\" = last_valid")
  expect_match(backbone_source, "\"next_pos\" = next_pos")
  expect_match(backbone_source, "eq\\$error_if")
  expect_false(grepl("pos_layer <- jnp\\$clip", backbone_source))
  expect_match(buildml_source, "insert_pos <- prefill_ret\\$next_pos")
  expect_false(grepl("prefix_len <- jnp\\$sum", buildml_source))
  expect_match(
    buildml_source,
    "ndm_runtime_get0\\(\"InitialObservationScale\", ifnotfound = 1(?:\\.0)?\\)"
  )
  expect_match(
    buildml_source,
    "ndm_runtime_get0\\(\"ObservationScaleFloor\", ifnotfound = 1e-0?5\\)"
  )
  expect_match(
    buildml_source,
    "InitialObservationScaleLearned <- InitialObservationScale - ObservationScaleFloor"
  )
  expect_match(
    buildml_source,
    "InitialObservationScaleLearned \\+[\n ]+log\\(-expm1\\(-InitialObservationScaleLearned\\)\\)"
  )
  expect_match(
    buildml_source,
    "ObservationScaleFloor \\+ SoftPlus\\(ModelList\\$ScaleList\\$ScaleBayes\\$DecoderObservationScale\\)"
  )
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
        TransformerList = model_list$TSList$TSBackbone
      )
    },
    warmup = 1L,
    runs = 2L
  )
  prefill_ret <- env$transformer_prefill_kv(
    xt = xt_running[[1]],
    x_mask = xt_running[[2]],
    TransformerList = model_list$TSList$TSBackbone
  )
  next_pos <- prefill_ret$next_pos
  xt_last <- env$DecoderBackboneToOutput(
    TSList = model_list$TSList,
    hidden_state = prefill_ret$xt_last
  )
  xt_running[[1]] <- env$jax$lax$dynamic_update_slice(
    xt_running[[1]],
    env$jnp$expand_dims(xt_last, 0L),
    env$jnp$array(c(next_pos, 0L), dtype = env$jnp$int32)
  )
  xt_running[[2]] <- env$jax$lax$dynamic_update_slice(
    xt_running[[2]],
    env$jnp$ones(list(1L, 1L), dtype = xt_running[[2]]$dtype),
    env$jnp$array(c(next_pos, 0L), dtype = env$jnp$int32)
  )
  decode_bench <- env$ndm_benchmark_helpers$benchmark(
    function() {
      env$transformer_decode_step_kv(
        token_in = env$jnp$take(xt_running[[1]], next_pos, axis = 0L),
        pos = next_pos,
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

test_that("cached training rollout matches uncached training gradients in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  cached_details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 3L,
    case_seed = 4242L,
    enable_kv_cache = TRUE,
    enable_kv_cache_training = TRUE,
    return_details = TRUE
  )
  uncached_details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 3L,
    case_seed = 4242L,
    enable_kv_cache = TRUE,
    enable_kv_cache_training = FALSE,
    return_details = TRUE
  )

  cached_first <- cached_details$summary$first_loss[[1L]]
  uncached_first <- uncached_details$summary$first_loss[[1L]]
  expect_true(is.finite(cached_first))
  expect_true(is.finite(uncached_first))
  # Same weights, same batch: the two training rollouts compute the same loss
  # up to float32 reordering noise.
  expect_lt(
    abs(cached_first - uncached_first) / (abs(uncached_first) + 1e-8),
    1e-4
  )

  # Gradient parity: after identical SGD steps through the two training paths
  # the loss trajectories stay aligned.
  cached_last <- cached_details$summary$last_loss[[1L]]
  uncached_last <- uncached_details$summary$last_loss[[1L]]
  expect_true(is.finite(cached_last))
  expect_true(is.finite(uncached_last))
  expect_lt(
    abs(cached_last - uncached_last) / (abs(uncached_last) + 1e-8),
    1e-3
  )

  cached_pred <- ndm_test_capture_model_prediction(
    cached_details$trained,
    batch = cached_details$batch,
    seed = 47L,
    inference = TRUE
  )
  uncached_pred <- ndm_test_capture_model_prediction(
    uncached_details$trained,
    batch = uncached_details$batch,
    seed = 47L,
    inference = TRUE
  )
  expect_lt(ndm_test_max_abs_diff(cached_pred$y_mu, uncached_pred$y_mu), 1e-4)
})
