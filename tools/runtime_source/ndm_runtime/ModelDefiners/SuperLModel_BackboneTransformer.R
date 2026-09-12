# Content of SuperLModel_BackboneTransformer.R
print("Done with SuperLModel_BackboneTransformer.R")
ndm_transformer_initialization_keys <- function(key) {
  split_keys <- jax$random$split(key, 5L)
  # reticulate forwards integer subscripts on JAX arrays as Python indices.
  # Keep these explicitly zero-based: four projection keys and one key that
  # advances initialization to the next layer.
  list(
    "W_q" = split_keys[0L],
    "W_k" = split_keys[1L],
    "W_v" = split_keys[2L],
    "W_o" = split_keys[3L],
    "next_layer" = split_keys[4L]
  )
}

if(backbonePath == "initialize"){
  backbone_runtime_lookup_env <- environment()
  backbone_runtime_get0 <- function(name, ifnotfound = NULL) {
    get0(name, envir = backbone_runtime_lookup_env, inherits = FALSE, ifnotfound = ifnotfound)
  }
  cuda_attention_available <- isTRUE(backbone_runtime_get0(
    "NDM_CUDA_ATTENTION_AVAILABLE",
    ifnotfound = FALSE
  ))
  TRY_FLASH <- cuda_attention_available && tryCatch(
    !any(grepl("V100", sapply(selected_devices, function(d) d$device_kind))),
    error = function(e) FALSE
  )
  EnableKVCachingRequested <- isTRUE(backbone_runtime_get0(
    "EnableKVCaching",
    ifnotfound = TRUE
  ))
  EnableKVCachingTrainingRequested <- isTRUE(backbone_runtime_get0(
    "EnableKVCachingTraining",
    ifnotfound = TRUE
  ))
  if (EnableKVCachingTrainingRequested && !EnableKVCachingRequested) {
    stop(
      "EnableKVCachingTraining=TRUE requires EnableKVCaching=TRUE.",
      call. = FALSE
    )
  }
  EnableKVCaching <- EnableKVCachingRequested && (ModelType == "DecoderOnly")
  EnableKVCachingTraining <- EnableKVCaching && EnableKVCachingTrainingRequested
  EnableKVCachingInferenceEffective <- EnableKVCaching
  EnableKVCachingTrainingEffective <- EnableKVCachingTraining
  print(sprintf(
    paste(
      "KV cache policy: requested=%s; training_requested=%s;",
      "inference_effective=%s; training_effective=%s"
    ),
    EnableKVCachingRequested,
    EnableKVCachingTrainingRequested,
    EnableKVCachingInferenceEffective,
    EnableKVCachingTrainingEffective
  ))
  # Full attention residuals are now the default transformer residual path.
  # Set UseFullAttentionResiduals = FALSE in runtime globals to opt back into
  # the legacy additive residual implementation for compatibility testing.
  UseFullAttentionResiduals <- isTRUE(backbone_runtime_get0("UseFullAttentionResiduals", ifnotfound = TRUE))
  FullAttentionResidualEps <- as.numeric(backbone_runtime_get0("FullAttentionResidualEps", ifnotfound = 1e-6))
  num_heads <- TransformerHeads
  num_kv_heads <- TransformerKVHeads
  head_dim <- TransformerHeadDim
  kv_group_size <- TransformerKVGroupSize
  if ((num_heads %% num_kv_heads) != 0L) {
    stop("Transformer query heads must be divisible by KV heads.", call. = FALSE)
  }
  
  # Keep master parameters and ODE arithmetic at jaxFloatType. Only the
  # transformer projections, residual sources and K/V caches use this dtype.
  TransformerComputeDtype <- match.arg(backbone_runtime_get0(
    "TransformerComputeDtype", "auto"
  ), c("auto", "native", "bfloat16", "float32"))
  native_dtype <- jnp$dtype(jaxFloatType)$name
  TransformerComputeDtypeResolved <- if (TransformerComputeDtype == "auto") {
    if (cuda_attention_available && native_dtype == "float32") "bfloat16" else native_dtype
  } else if (TransformerComputeDtype == "native") native_dtype else TransformerComputeDtype
  transformer_dtype <- jnp$dtype(TransformerComputeDtypeResolved)
  TransformerActivationCheckpointing <- backbone_runtime_get0("TransformerActivationCheckpointing", FALSE)
  if (!is.logical(TransformerActivationCheckpointing) ||
      length(TransformerActivationCheckpointing) != 1L || is.na(TransformerActivationCheckpointing)) {
    stop("TransformerActivationCheckpointing must be one non-missing logical value.", call. = FALSE)
  }
  transformer_checkpoint <- function(f, prevent_cse = TRUE) {
    # Layers are unrolled: prevent compiler CSE from retaining recomputed
    # intermediates. A surrounding recurrent scan can omit that barrier.
    if (TransformerActivationCheckpointing) eq$filter_checkpoint(f, prevent_cse = prevent_cse) else f
  }

  choose_attention_impl <- function(prefer = "auto") {
    # CPU and non-CUDA devices always fall back to portable XLA.
    if (prefer == "xla" || !cuda_attention_available || !TRY_FLASH) "xla" else "cudnn"
  }
  normalize_mask_for_dpa <- function(mask) {
    if (is.null(mask)) NULL else jnp$greater(mask, 0)
  }
  dot_product_attention_unified <- function(q, k, v, mask = NULL,
                                            is_causal = FALSE, prefer = "auto") {
    impl <- choose_attention_impl(prefer)
    mask <- normalize_mask_for_dpa(mask)
    # Native GQA accepts fewer K/V heads than query heads. Never tile K/V.
    dtype <- if (q$dtype$name == "bfloat16") jnp$bfloat16 else if (impl == "cudnn") jnp$float16 else jnp$float32
    if (impl == "cudnn") {
      seq_axis <- length(q$shape) - 2L
      tq <- as.integer(q$shape[[seq_axis]])
      sk <- as.integer(k$shape[[seq_axis]])
      tq_pad <- as.integer((8L - tq %% 8L) %% 8L)
      sk_pad <- as.integer((8L - sk %% 8L) %% 8L)
      pad_sequence <- function(x, amount) {
        pads <- rep(list(c(0L, 0L)), length(x$shape))
        pads[[seq_axis]] <- c(0L, amount)
        jnp$pad(x, pads)
      }
      q <- pad_sequence(q, tq_pad)
      k <- pad_sequence(k, sk_pad)
      v <- pad_sequence(v, sk_pad)
      if (!is.null(mask)) {
        pads <- rep(list(c(0L, 0L)), length(mask$shape))
        if (mask$shape[[length(pads) - 1L]] != 1L) pads[[length(pads) - 1L]] <- c(0L, tq_pad)
        pads[[length(pads)]] <- c(0L, sk_pad)
        mask <- jnp$pad(mask, pads, constant_values = FALSE)
      } else if (sk_pad > 0L) {
        mask <- jnp$less(jnp$arange(sk + sk_pad), sk)
        mask <- jnp$reshape(mask, list(1L, 1L, sk + sk_pad))
      }
    }
    out <- jax$nn$dot_product_attention(
      q$astype(dtype), k$astype(dtype), v$astype(dtype), mask = mask,
      is_causal = is_causal, implementation = impl
    )
    if (impl == "cudnn") out <- jnp$take(out, jnp$arange(tq, dtype = jnp$int32), axis = as.integer(seq_axis - 1L))
    out$astype(q$dtype)
  }

      rope_apply_single <- function(x_d, pos, head_dim) {
        # x_d: [H] for one head (not all heads), H == head_dim
        # Split into even/odd (cos/sin) halves
        half <- as.integer(head_dim %/% 2L)
        # Frequencies as in standard RoPE
        freqs <- 1 / (10000^(seq(0, half - 1) / half))
        angle <- jnp$array(freqs) * jnp$array(pos, dtype = jnp$float32)
        c_ <- jnp$cos(angle); s_ <- jnp$sin(angle)
        
        # Pair rotation: (x_even, x_odd)
        x_even <- jnp$take(x_d, jnp$array(0:(half - 1), dtype = jnp$int32))
        x_odd  <- jnp$take(x_d, jnp$array(half:(2L*half - 1L), dtype = jnp$int32))
        x_rot_even <- x_even * c_ - x_odd * s_
        x_rot_odd  <- x_even * s_ + x_odd * c_
        jnp$concatenate(list(x_rot_even, x_rot_odd), axis = 0L)
      }

      repeat_kv_heads <- function(x, group_size) {
        if (group_size == 1L) {
          return(x)
        }
        if (length(x$shape) == 3L) {
          x_expanded <- jnp$expand_dims(x, 2L)
          x_tiled <- jnp$tile(x_expanded, list(1L, 1L, group_size, 1L))
          return(jnp$reshape(x_tiled, list(x$shape[[1]], x$shape[[2]] * group_size, x$shape[[3]])))
        }
        if (length(x$shape) == 2L) {
          x_expanded <- jnp$expand_dims(x, 1L)
          x_tiled <- jnp$tile(x_expanded, list(1L, group_size, 1L))
          return(jnp$reshape(x_tiled, list(x$shape[[1]] * group_size, x$shape[[2]])))
        }
        stop("repeat_kv_heads expects a [T, N, H] or [N, H] tensor.", call. = FALSE)
      }

      resolve_qk_norm_scale <- function(scale, num_local_heads, dtype = jnp$float32) {
        if (is.null(scale)) {
          return(jnp$ones(list(as.integer(num_local_heads), 1L), dtype = dtype))
        }
        scale <- scale$astype(dtype)
        if (length(scale$shape) == 1L) {
          return(jnp$reshape(scale, list(as.integer(num_local_heads), 1L)))
        }
        scale
      }

      qk_normalize_heads <- function(x, scale = NULL, eps = 1e-6) {
        rank <- length(x$shape)
        if (!(rank %in% c(2L, 3L, 4L))) {
          stop("qk_normalize_heads expects a [N, H], [T, N, H], or [B, T, N, H] tensor.", call. = FALSE)
        }

        num_local_heads <- as.integer(x$shape[[rank - 1L]])
        scale_shape <- rep(1L, rank)
        scale_shape[[rank - 1L]] <- num_local_heads
        scale_shape[[rank]] <- 1L

        x_f32 <- x$astype(jnp$float32)
        scale_f32 <- resolve_qk_norm_scale(scale, num_local_heads, dtype = jnp$float32)
        scale_f32 <- jnp$reshape(scale_f32, scale_shape)
        rms <- jnp$sqrt(jnp$mean(jnp$square(x_f32), axis = -1L, keepdims = TRUE) + eps)

        jnp$multiply(jnp$divide(x_f32, rms), scale_f32)$astype(x$dtype)
      }

      resolve_attnres_norm_scale <- function(scale, width, dtype = jnp$float32) {
        if (is.null(scale)) {
          stop("Full attention residual layers require AttnRes NormScale.", call. = FALSE)
        }
        scale <- scale$astype(dtype)
        if (length(scale$shape) != 1L) {
          scale <- jnp$reshape(scale, list(as.integer(width)))
        }
        scale
      }

      attnres_normalize_sources <- function(sources, scale, eps = FullAttentionResidualEps) {
        rank <- length(sources$shape)
        if (!(rank %in% c(2L, 3L))) {
          stop("attnres_normalize_sources expects a [N, D] or [N, T, D] tensor.", call. = FALSE)
        }

        width <- as.integer(sources$shape[[rank]])
        scale_shape <- rep(1L, rank)
        scale_shape[[rank]] <- width

        sources_f32 <- sources$astype(jnp$float32)
        scale_f32 <- resolve_attnres_norm_scale(scale, width, dtype = jnp$float32)
        scale_f32 <- jnp$reshape(scale_f32, scale_shape)
        rms <- jnp$sqrt(jnp$mean(jnp$square(sources_f32), axis = -1L, keepdims = TRUE) + eps)

        jnp$multiply(jnp$divide(sources_f32, rms), scale_f32)
      }

      mask_sequence_rows_2d <- function(x, mask_rows_bool) {
        jnp$where(
          jnp$expand_dims(mask_rows_bool, 1L),
          x,
          jnp$zeros_like(x)
        )
      }

      mask_sequence_rows_3d <- function(x, mask_rows_bool) {
        jnp$where(
          jnp$expand_dims(jnp$expand_dims(mask_rows_bool, 1L), 2L),
          x,
          jnp$zeros_like(x)
        )
      }

      attnres_max_sources <- as.integer(1L + 2L * ModelDepth)

      attnres_init_buffer <- function(seq_len, width, dtype) {
        jnp$zeros(list(attnres_max_sources, seq_len, width), dtype = dtype)
      }

      attnres_append <- function(buffer, source_count, source_txd) {
        update_idx <- jnp$array(c(source_count, 0L, 0L), dtype = jnp$int32)
        updated_buffer <- jax$lax$dynamic_update_slice(
          buffer,
          jnp$expand_dims(source_txd, 0L),
          update_idx
        )
        list(
          "buffer" = updated_buffer,
          "count" = jnp$add(source_count, jnp$array(1L, dtype = jnp$int32))
        )
      }

      full_attnres_reduce_buffer <- function(buffer, source_count, pseudo_query, norm_scale, eps = FullAttentionResidualEps) {
        if (length(buffer$shape) != 3L) {
          stop("full_attnres_reduce_buffer expects a [N, T, D] tensor.", call. = FALSE)
        }

        eps_f32 <- jnp$array(as.numeric(eps), dtype = jnp$float32)
        sources_f32 <- buffer$astype(jnp$float32)
        query_f32 <- pseudo_query$astype(jnp$float32)
        keys_f32 <- attnres_normalize_sources(sources_f32, scale = norm_scale, eps = eps_f32)
        logits <- jnp$einsum("ntd,d->nt", keys_f32, query_f32)
        valid_sources <- jnp$less(
          jnp$arange(buffer$shape[[1]], dtype = jnp$int32),
          jnp$astype(source_count, jnp$int32)
        )
        logits <- jnp$where(
          jnp$expand_dims(valid_sources, 1L),
          logits,
          jnp$array(-1e30, dtype = logits$dtype)
        )
        weights <- jax$nn$softmax(logits, axis = 0L)
        jnp$einsum("nt,ntd->td", weights, sources_f32)$astype(buffer$dtype)
      }

      full_attnres_output <- function(buffer, source_count, TransformerList) {
        output_params <- TransformerList$AttnResOutput
        if (is.null(output_params)) {
          stop("Full attention residual models require AttnResOutput; rebuild models created before the final aggregation was added.", call. = FALSE)
        }
        full_attnres_reduce_buffer(
          buffer, source_count,
          output_params$PseudoQuery, output_params$NormScale
        )
      }

      rope_freqs <- jnp$array(
        1 / (10000^(seq(0, as.integer(head_dim %/% 2L) - 1L) / as.integer(head_dim %/% 2L))),
        dtype = jnp$float32
      )

      apply_rope_batched <- function(x_tnh, pos_ids, head_dim) {
        if (length(x_tnh$shape) != 3L) {
          stop("apply_rope_batched expects a [T, N, H] tensor.", call. = FALSE)
        }

        half <- as.integer(head_dim %/% 2L)
        even_idx <- jnp$array(0:(half - 1L), dtype = jnp$int32)
        odd_idx <- jnp$array(half:(2L * half - 1L), dtype = jnp$int32)
        x_even <- jnp$take(x_tnh, even_idx, axis = 2L)
        x_odd <- jnp$take(x_tnh, odd_idx, axis = 2L)
        angles <- jnp$reshape(pos_ids$astype(jnp$float32), list(-1L, 1L)) *
          jnp$reshape(rope_freqs, list(1L, half))
        c_ <- jnp$expand_dims(jnp$cos(angles), 1L)
        s_ <- jnp$expand_dims(jnp$sin(angles), 1L)
        x_rot_even <- x_even * c_ - x_odd * s_
        x_rot_odd <- x_even * s_ + x_odd * c_
        jnp$concatenate(list(x_rot_even, x_rot_odd), axis = 2L)$astype(x_tnh$dtype)
      }
      
      # Allocate KV cache: per layer, K/V are [max_len, num_kv_heads, head_dim].
      # `valid` records the physical sequence positions that contain tokens. It
      # must not be replaced by a token count because decoder inputs are
      # left-padded and therefore need not start at position zero.
      kv_cache_allocate <- function(max_len, num_layers, num_kv_heads, head_dim, dtype) {
        make_one <- function() {
          list(
            "k" = jnp$zeros(list(max_len, num_kv_heads, head_dim), dtype = dtype),
            "v" = jnp$zeros(list(max_len, num_kv_heads, head_dim), dtype = dtype),
            "valid" = jnp$greater(
              jnp$zeros(list(max_len), dtype = jnp$int32),
              jnp$array(0L, dtype = jnp$int32)
            )
          )
        }
        out <- replicate(num_layers, make_one(), simplify = FALSE)
        names(out) <- paste0("d", as.character(1:num_layers))
        out
      }
      

  # A residual source carries its inverse RMS from creation, computed once in
  # float32. Every later aggregation reuses it, so the normalised keys are
  # never materialised: logits = (source . (scale * query)) * inv_rms.
  attnres_source <- function(x, eps = FullAttentionResidualEps) {
    x_f32 <- x$astype(jnp$float32)
    mean_square <- jnp$mean(jnp$square(x_f32), axis = -1L)
    inv_rms <- jnp$reciprocal(jnp$sqrt(mean_square + jnp$array(as.numeric(eps), dtype = jnp$float32)))
    list("value" = x, "inv_rms" = inv_rms)
  }
  # Aggregate residual sources without a [2*depth+1, T, D] stack: one matvec
  # per source for the logits (the source is upcast for that matvec; a
  # bfloat16 dot loses gradient precision) and a fused weighted sum.
  full_attnres_combine <- function(sources, pseudo_query, norm_scale) {
    if (length(sources) == 0L) {
      stop("Full attention residual aggregation needs at least one source.", call. = FALSE)
    }
    width <- as.integer(sources[[1L]]$value$shape[[2L]])
    query_scaled <- jnp$multiply(
      resolve_attnres_norm_scale(norm_scale, width, dtype = jnp$float32),
      pseudo_query$astype(jnp$float32)
    )
    logits <- jnp$stack(lapply(sources, function(s) {
      jnp$multiply(jnp$matmul(s$value$astype(jnp$float32), query_scaled), s$inv_rms)
    }), axis = 0L)
    weights <- jax$nn$softmax(logits, axis = 0L)
    out <- NULL
    for (n in seq_along(sources)) {
      term <- jnp$multiply(
        jnp$expand_dims(jnp$take(weights, n - 1L, axis = 0L), 1L),
        sources[[n]]$value$astype(jnp$float32)
      )
      out <- if (is.null(out)) term else jnp$add(out, term)
    }
    out$astype(sources[[1L]]$value$dtype)
  }
  full_attnres_reduce_sources <- transformer_checkpoint(function(sources, query, scale) {
    full_attnres_combine(sources, query, scale)
  })
  transformer_norm <- function(x) {
    if (x$dtype$name == "bfloat16") NormFxn(x$astype(jnp$float32))$astype(x$dtype) else NormFxn(x)
  }
  transformer_linear <- function(x, layer) {
    y <- jnp$dot(x, jnp$transpose(layer$weight$astype(x$dtype)))
    if (!is.null(layer$bias)) y <- y + layer$bias$astype(x$dtype)
    y$astype(x$dtype)
  }
  transformer_ffn <- transformer_checkpoint(function(x, L) {
    x <- transformer_norm(x) * L$NormScalerPostMultiHead$astype(x$dtype)
    x <- jax$nn$swish(transformer_linear(x, L$FFN$WideProj1)) *
      transformer_linear(x, L$FFN$WideProj2)
    transformer_linear(x, L$FFN$OutProj1)
  })
  transformer_attention <- transformer_checkpoint(function(x, L, positions, mask,
                                                            causal, prefer, cache, pos) {
    dtype <- x$dtype
    x <- transformer_norm(x) * L$NormScalerInput$astype(dtype)
    project <- function(w, heads) jnp$reshape(jnp$dot(x, w$astype(dtype)), list(x$shape[[1]], heads, head_dim))
    q <- qk_normalize_heads(apply_rope_batched(project(L$Multihead$W_q, num_heads), positions, head_dim), L$Multihead$QNormScale)
    k <- qk_normalize_heads(apply_rope_batched(project(L$Multihead$W_k, num_kv_heads), positions, head_dim), L$Multihead$KNormScale)
    v <- project(L$Multihead$W_v, num_kv_heads)
    if (!is.null(cache)) {
      write_pos <- if (is.null(pos)) jnp$array(0L, dtype = jnp$int32) else pos
      idx <- jnp$array(c(write_pos, 0L, 0L), dtype = jnp$int32)
      cache$k <- jax$lax$dynamic_update_slice(cache$k, k, idx)
      cache$v <- jax$lax$dynamic_update_slice(cache$v, v, idx)
    }
    if (!is.null(pos)) {
      k <- cache$k
      v <- cache$v
    }
    out <- dot_product_attention_unified(q, k, v, mask, causal, prefer)$astype(dtype)
    list(value = jnp$dot(jnp$reshape(out, list(x$shape[[1]], num_heads * head_dim)), L$Multihead$W_o$astype(dtype)), cache = cache)
  })
  transformer_skip <- function(x, branch, weights) {
    x * jax$nn$softplus(weights$WtSkipPath$astype(x$dtype)) +
      branch * jax$nn$softplus(weights$WtResidPath$astype(x$dtype))
  }

  # Restrict a residual source, the running state and the row mask to one
  # sequence position. Every remaining computation (FFN, residual skip, source
  # aggregation) is position-wise, so the result at that position is unchanged.
  attnres_select_position <- function(sources, xt, rows, position) {
    index <- jnp$reshape(position$astype(jnp$int32), list(1L))
    list(
      sources = lapply(sources, function(s) list(
        "value" = jnp$take(s$value, index, axis = 0L),
        "inv_rms" = jnp$take(s$inv_rms, index, axis = 0L)
      )),
      xt = jnp$take(xt, index, axis = 0L),
      rows = jnp$take(rows, index, axis = 0L)
    )
  }

  # `select_position`: when given, only that sequence position is carried past
  # the last layer's attention (whose K/V come from the layer's input and are
  # written to the cache before the slice), so the final FFN and output
  # aggregation run on one token instead of the whole sequence. Callers that
  # consume a single token use this; RunTransformerBackbone keeps the full
  # sequence. The returned `xt` then has one row.
  transformer_run <- function(xt, x_mask, TransformerList, mode = "full", cache = NULL, pos = NULL, max_len = NULL,
                              select_position = NULL) {
    xt <- xt$astype(transformer_dtype)
    rows <- jnp$squeeze(jnp$greater(x_mask, 0), 1L)
    positions <- if (mode == "decode") jnp$reshape(pos, list(1L)) else jnp$arange(xt$shape[[1]], dtype = jnp$int32)
    xt <- mask_sequence_rows_2d(xt, rows)
    if (mode == "prefill") {
      if (is.null(max_len)) max_len <- as.integer(xt$shape[[1]])
      if (max_len < xt$shape[[1]]) stop("KV cache capacity must cover the prefill context.", call. = FALSE)
      cache <- kv_cache_allocate(max_len, ModelDepth, num_kv_heads, head_dim, xt$dtype)
    }
    sources <- if (UseFullAttentionResiduals) list(attnres_source(xt)) else list()
    for (layer_name in paste0("d", seq_len(ModelDepth))) {
      L <- TransformerList[[layer_name]]
      source <- if (UseFullAttentionResiduals) full_attnres_reduce_sources(sources, L$AttnRes1$PseudoQuery, L$AttnRes1$NormScale) else xt
      layer_cache <- if (mode == "full") NULL else cache[[layer_name]]
      if (mode == "prefill") {
        layer_cache$valid <- jax$lax$dynamic_update_slice(layer_cache$valid, rows, list(0L))
      }
      if (mode == "decode") {
        layer_cache$valid <- jax$lax$dynamic_update_slice(layer_cache$valid, jnp$ones(list(1L), dtype = jnp$bool_), jnp$reshape(pos, list(1L)))
        valid_keys <- jnp$logical_and(layer_cache$valid, jnp$less_equal(jnp$arange(layer_cache$k$shape[[1]]), pos))
      } else valid_keys <- rows
      mask <- jnp$reshape(valid_keys, list(1L, 1L, valid_keys$shape[[1]]))
      attention <- transformer_attention(source, L, positions, mask,
        ModelType == "DecoderOnly" && mode != "decode", if (mode == "decode") "xla" else "auto",
        layer_cache, if (mode == "decode") pos else NULL)
      if (mode != "full") cache[[layer_name]] <- attention$cache
      branch <- mask_sequence_rows_2d(attention$value, rows)
      if (!is.null(select_position) && layer_name == paste0("d", ModelDepth)) {
        selected <- attnres_select_position(sources, xt, rows, select_position)
        sources <- selected$sources
        xt <- selected$xt
        rows <- selected$rows
        branch <- jnp$take(branch, jnp$reshape(select_position$astype(jnp$int32), list(1L)), axis = 0L)
      }
      if (UseFullAttentionResiduals) {
        sources[[length(sources) + 1L]] <- attnres_source(branch)
        source <- full_attnres_reduce_sources(sources, L$AttnRes2$PseudoQuery, L$AttnRes2$NormScale)
      } else {
        xt <- transformer_skip(xt, branch, L$ResidCon1)
        source <- xt
      }
      branch <- mask_sequence_rows_2d(transformer_ffn(source, L), rows)
      if (UseFullAttentionResiduals) sources[[length(sources) + 1L]] <- attnres_source(branch) else xt <- transformer_skip(xt, branch, L$ResidCon2)
    }
    if (UseFullAttentionResiduals) {
      output <- TransformerList$AttnResOutput
      if (is.null(output)) stop("Full attention residual models require AttnResOutput; rebuild models created before the final aggregation was added.", call. = FALSE)
      xt <- full_attnres_reduce_sources(sources, output$PseudoQuery, output$NormScale)
    }
    list(xt = xt, cache = cache)
  }
  RunTransformerBackbone <- function(xt, x_mask, TransformerList) {
    transformer_run(xt, x_mask, TransformerList)$xt
  }
  RunTransformerBackbone_FullAttnRes <- RunTransformerBackbone
  transformer_prefill_kv <- function(xt, x_mask, TransformerList, max_len = NULL) {
    valid <- jnp$squeeze(jnp$greater(x_mask, 0), 1L)
    last_valid <- jnp$max(jnp$where(valid, jnp$arange(xt$shape[[1]], dtype = jnp$int32), -1L))
    next_pos <- last_valid + 1L
    # Only the last valid token's output is used, so the final layer's FFN and
    # output aggregation run on that one position.
    result <- transformer_run(xt, x_mask, TransformerList, mode = "prefill", max_len = max_len,
                              select_position = jnp$maximum(last_valid, 0L))
    list(xt_last = jnp$squeeze(result$xt, 0L), cache = result$cache,
         "last_valid" = last_valid, "next_pos" = next_pos)
  }
  # One-token backbone output for encoders that keep a single position
  # (SelectBackboneOutputToken semantics), without computing the rest.
  RunTransformerBackboneAt <- function(xt, x_mask, TransformerList, position) {
    jnp$squeeze(transformer_run(xt, x_mask, TransformerList, select_position = position)$xt, 0L)
  }
  transformer_decode_step_kv <- function(token_in, pos, TransformerList, cache) {
    pos <- jnp$astype(pos, jnp$int32)
    pos <- eq$error_if(pos, jnp$logical_or(pos < 0L, pos >= cache[[1L]]$k$shape[[1]]),
                      "KV cache decode position is outside the allocated capacity.")
    result <- transformer_run(jnp$expand_dims(token_in, 0L), jnp$ones(list(1L, 1L)), TransformerList,
                              mode = "decode", cache = cache, pos = pos)
    list(token_out = jnp$squeeze(result$xt, 0L), cache = result$cache)
  }

  print("Generating TransformerList objects...")
  for(l_ in 1L:length(TransformerList)){
    if( UseLatentAttention ){ 
      print("Defining latent attention helpers...")
      stop("Latent attention is not included in ndm.", call. = FALSE)
      TransformerList[[l_]]$LatentMultihead <- LatentMultiheadAttentionInitialize(
                                          query_size = ModelDims,
                                          output_size = ModelDims,
                                          num_heads = TransformerHeads,
                                          latent_dim = LatentDim,
                                          use_output_bias = F,
                                          key = key)
      key <- jax$random$split(key)[[1]]
    }
    if( !UseLatentAttention ){
      {
        # - Define GQA projections. Query heads span the model width; KV heads are grouped.
        head_dim <- TransformerHeadDim
        num_heads = TransformerHeads
        num_kv_heads = TransformerKVHeads
        q_proj_dim <- num_heads * head_dim
        kv_proj_dim <- num_kv_heads * head_dim
        
        init_std <- sqrt(2.0 / as.numeric(ModelDims + ModelDims))
        make_w <- function(shape, seed_key) {
          oryx$Normal(loc = 0., scale = jnp$array(init_std))$
            sample(shape, seed = seed_key)$astype(jaxFloatType)
        }
        
        print("Generating Multihead objects...")
        multihead_keys <- ndm_transformer_initialization_keys(key)
        TransformerList[[l_]]$Multihead <- list(
          "W_q" = make_w(list(ModelDims, q_proj_dim), multihead_keys$W_q),
          "W_k" = make_w(list(ModelDims, kv_proj_dim), multihead_keys$W_k),
          "W_v" = make_w(list(ModelDims, kv_proj_dim), multihead_keys$W_v),
          "W_o" = make_w(list(q_proj_dim, ModelDims), multihead_keys$W_o),
          "QNormScale" = jnp$ones(list(num_heads, 1L), dtype = jaxFloatType),
          "KNormScale" = jnp$ones(list(num_kv_heads, 1L), dtype = jaxFloatType)
        )
        if (isTRUE(UseFullAttentionResiduals)) {
          TransformerList[[l_]]$AttnRes1 <- list(
            "PseudoQuery" = jnp$zeros(list(ModelDims), dtype = jaxFloatType),
            "NormScale" = jnp$ones(list(ModelDims), dtype = jaxFloatType)
          )
          TransformerList[[l_]]$AttnRes2 <- list(
            "PseudoQuery" = jnp$zeros(list(ModelDims), dtype = jaxFloatType),
            "NormScale" = jnp$ones(list(ModelDims), dtype = jaxFloatType)
          )
        }
        key <- multihead_keys$next_layer
      }
    }
    TransformerList[[l_]]$NormScalerInput <- oryx$Normal(loc = 1.,scale =  0.0001)$sample( list(1L,ModelDims), seed = key*21134L*l_)$astype(jaxFloatType)
    {
      # swiglu FFN
      print(sprintf("Generating FNN objects...[layer %s of %s]",l_,length(TransformerList)-1L))
      TransformerList[[l_]]$FFN <- list("WideProj1"=eq$nn$Linear(in_features = ModelDims,
                                                      out_features = ai(ModelDims*WideMultiplicationFactor),
                                                      use_bias = F, # hidden bias
                                                      key = 3L+key*3L*l_),
                                       "WideProj2"= eq$nn$Linear(in_features = ModelDims,
                                                      out_features = ai(ModelDims*WideMultiplicationFactor),
                                                      use_bias = F, # swiglu bias
                                                      key =  4L+key*l_*7L),
                                       "OutProj1"=eq$nn$Linear(in_features = ai(ModelDims*WideMultiplicationFactor),
                                                      out_features = ModelDims,
                                                      use_bias = F,  # output bias
                                                      key =  56L+key*l_*23L))
    }
    print(sprintf("Generating final scaling outputs...[layer %s of %s]",l_,length(TransformerList)-1L))
    TransformerList[[l_]]$NormScalerPostMultiHead <- oryx$Normal(loc = 1., scale =  0.0001)$sample( list(1L,ModelDims),seed = key*l_*21344L)$astype(jaxFloatType)
    # WtSkipPathInit <- (2*ModelDepth)^(1/(2*ModelDepth)); WtResidPathInit <- 1. # DeepNorm (Wang 2022)
    #WtSkipPathInit <- sqrt( 0.1 / ( l_ + 2*ModelDepth)); WtResidPathInit <- rep(sqrt(1-WtSkipPathInit^2),time=ModelDims)
    # https://proceedings.neurips.cc/paper_files/paper/2019/file/e520f70ac3930490458892665cda6620-Paper.pdf
    #WtSkipPathInit <- 1; WtResidPathInit <- (1/ModelDepth)^(1/2)
    #WtSkipPathInit <- (1/ModelDepth)^(1/1); WtResidPathInit <- 1. # https://arxiv.org/abs/2203.00555
    #WtSkipPathInit <- sqrt( (l_-1+ModelDepth) / (l_+ModelDepth)); WtResidPathInit <- sqrt(1./(l_+ModelDepth)) # https://proceedings.neurips.cc/paper/2020/file/9b8619251a19057cff70779273e95aa6-Paper.pdf
    #WtResidPathInit <- sqrt( 1 / ( l_ + 0.05^2*ModelDepth)); WtSkipPathInit <- sqrt(1-WtResidPathInit^2) # https://iclr-blog-track.github.io/2022/03/25/unnormalized-resnets/
    WtResidPathInit <- sqrt( 1 / ( l_ + 0.01^2*ModelDepth)); 
    WtSkipPathInit <- sqrt(1-WtResidPathInit^2) # https://iclr-blog-track.github.io/2022/03/25/unnormalized-resnets/
    WtSkipPathInit_inv <- np$array( InvSoftPlus(jnp$array( WtSkipPathInit )))
    WtResidPathInit_inv <- np$array( InvSoftPlus(jnp$array( WtResidPathInit )))
    # WtSkipPathInit <- 1; WtResidPathInit <- sqrt( 1 / ModelDepth)
    # plot( WtSkipPathInit,ylim = c(0,1),type="b"); points(WtResidPathInit,type="b"); WtResidPathInit + WtSkipPathInit
    TransformerList[[l_]]$ResidCon1 <- list("WtSkipPath"=jnp$squeeze(oryx$Normal(loc =WtSkipPathInit_inv,
                                                                      scale =  0.0000001)$sample( list(ModelDims), seed = 4000L+key)$astype(jaxFloatType),1L),
                                            "WtResidPath"=jnp$squeeze(oryx$Normal(loc = WtResidPathInit_inv,
                                                                 scale =  0.0000001)$sample( list(ModelDims), seed = 4001L+key)$astype(jaxFloatType),1L))
    TransformerList[[l_]]$ResidCon2 <-  list("WtSkipPath"=jnp$squeeze(oryx$Normal(loc = WtSkipPathInit_inv,
                                                                 scale =  0.0000001)$sample( list(ModelDims), seed = 4002L+ key)$astype(jaxFloatType),1L),
                                            "WtResidPath"=jnp$squeeze(oryx$Normal(loc = WtResidPathInit_inv,
                                                                  scale =  0.0000001)$sample( list(ModelDims), seed =4003L+ key)$astype(jaxFloatType),1L))
  }
  names(TransformerList) <- paste0("d",as.character( 1L:length(TransformerList) ))
  if (isTRUE(UseFullAttentionResiduals)) {
    TransformerList$AttnResOutput <- list(
      "PseudoQuery" = jnp$zeros(list(ModelDims), dtype = jaxFloatType),
      "NormScale" = jnp$ones(list(ModelDims), dtype = jaxFloatType)
    )
  }
  print("Generating decoder head...")
  TransformerList$DecoderProj <- eq$nn$Linear(in_features = ModelDims,
                                              out_features = ai(nOutcomes),
                                              use_bias = T, key = 993L+key*233L)
  TransformerList$UseFullAttentionResiduals <- isTRUE(UseFullAttentionResiduals)
  print("Done with init path in SuperLModel_BackboneTransformer.R...")
}

if(backbonePath == "run"){ # note: there is no caching here; caching is applied in *BuildML.R
  if (!exists("RunTransformerBackbone", inherits = TRUE)) {
    stop("Transformer runtime helpers were not initialized before the run path.", call. = FALSE)
  }
  xt <- RunTransformerBackbone(xt = xt, x_mask = x_mask, TransformerList = TSList$TSBackbone)
}
print("Done sourcing SuperLModel_BackboneTransformer.R")
