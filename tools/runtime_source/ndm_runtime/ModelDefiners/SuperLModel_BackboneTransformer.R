# Content of SuperLModel_BackboneTransformer.R
print("Done with SuperLModel_BackboneTransformer.R")
if(backbonePath == "initialize"){
  backbone_runtime_lookup_env <- environment()
  backbone_runtime_get0 <- function(name, ifnotfound = NULL) {
    get0(name, envir = backbone_runtime_lookup_env, inherits = FALSE, ifnotfound = ifnotfound)
  }
  TRY_FLASH <- tryCatch(!any(grepl("V100", sapply(jax$devices(), function(d) d$device_kind))), error = function(e) FALSE)
  EnableKVCaching <- backbone_runtime_get0("EnableKVCaching", ifnotfound = (ModelType == "DecoderOnly"))
  EnableKVCaching <- isTRUE(EnableKVCaching) && (ModelType == "DecoderOnly")
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
  
  # --- Unified dot-product attention helper (flash or XLA) ----------------------
  if (!exists(".__unified_attn_defined", inherits = FALSE)) {
    print("Defining Unified dot-product attention helper...")
    # Decide implementation once per host (not traced)
    choose_attention_impl <- function(prefer = "auto"){
      if (prefer == "xla")   return("xla")
      if (prefer == "cudnn") return("cudnn")
      # auto: prefer CUDA/cuDNN when a GPU is visible and TRY_FLASH is TRUE
      has_gpu <- any(sapply(jax$devices(), function(d) d$platform == "gpu"))
      if (isTRUE(has_gpu) && isTRUE(TRY_FLASH)) "cudnn" else "xla"
    }
    
    # Normalize masks: accept numeric/bool; pass through existing broadcast shapes.
    # We don't reshape aggressively to avoid surprises—your code already forms [N,T,S] or [1,N,1,S].
    normalize_mask_for_dpa <- function(mask){
      if (is.null(mask)) return(NULL)
      # ensure boolean
      if (mask$dtype$`__str__`() != "bool"){ mask <- jnp$greater(mask, 0) }
      return( mask ) 
    }
    
    # q,k,v expected as [T, N, H] (your current convention) or [B, T, N, H]
    # If mask != NULL we use XLA path (flash_jax doesn’t support arbitrary masks);
    # if mask == NULL we use flash on GPU, XLA otherwise.
    dot_product_attention_unified <- function(q, k, v, 
                                              mask = NULL, 
                                              is_causal = FALSE, 
                                              prefer = "auto"){
      
      # try automatic routing of attention type 
      impl <- choose_attention_impl( prefer )
      
      # Route: force XLA when a mask is provided + on flash-flash branch (not cudnn-flash branch)
      #if (!is.null(mask)){ impl <- "xla" } 
      
      if (impl == "cudnn"){
        if(FALSE){  
          # flash-flash path (doesn't allow masking)
          # Flash/cuDNN path (no mask): compute in fp16 for speed, cast back to input dtype
          # Notes: also supports sliding/ring attention for long series 
          # Notes: mask
          # Note: remove padding by flattening tokens from every batch into a single (total_tokens, n_heads, head_dim) tensor and pass cu_seqlens (cumulative lengths). 
          # The official FlashAttention API supports this (and it's what gives you both correctness and speed). The flash_attn_jax repo exposes flash MHA functionality
          
          # Support both [T,N,H] and [B,T,N,H]; vmap over batch if needed.
          if (length(q$shape) == 3L) {
            # requires fp16, then cast back
            out <- flash_mha(q$astype(jnp$float16), 
                             k$astype(jnp$float16), 
                             v$astype(jnp$float16), is_causal = is_causal)
          }
          if (length(q$shape) != 3L) {
            vmap_flash <- jax$vmap(flash_one <- function(qb, kb, vb) flash_mha(qb, kb, vb, is_causal = is_causal), 
                                   in_axes = list(0L, 0L, 0L))
            out <- vmap_flash(q$astype(jnp$float16),
                              k$astype(jnp$float16),
                              v$astype(jnp$float16))
          }
        }
        if(FALSE){ 
          # cudnn-flash path: supports masks; requires fp16, then cast back
          # no padding 
          mask_dpa <- normalize_mask_for_dpa(mask)
          out <- jax$nn$dot_product_attention(
            q$astype(jnp$float16),
            k$astype(jnp$float16),
            v$astype(jnp$float16),
            mask = mask_dpa,
            is_causal = is_causal,
            implementation = "cudnn"
          )
        }
        if(TRUE){
          # cudnn-flash path: supports masks; requires fp16, then cast back
          # with padding 
          next8 <- function(n) as.integer(((n + 7L) %/% 8L) * 8L)
          pad_TNH <- function(x, newT){
            pad <- newT - x$shape[[1]]
            if (pad <= 0L) return(x)
            jnp$pad(x, list(c(0L, pad), c(0L, 0L), c(0L, 0L)))
          }
          
          Tq <- as.integer(ifelse(length(q$shape) == 3L, yes =q$shape[[1]],no = q$shape[[2]]))
          Sk <- as.integer(ifelse(length(k$shape) == 3L, yes = k$shape[[1]],no= k$shape[[2]]))
          Tq8 <- next8(Tq); Sk8 <- next8(Sk)
          
          # Pad Q/K/V
          q_pad <- pad_TNH(q, Tq8)
          k_pad <- pad_TNH(k, Sk8)
          v_pad <- pad_TNH(v, Sk8)
          
          # Pad mask on the S axis if present
          if (!is.null(mask)) {
            pad_S <- Sk8 - Sk
            if( pad_S > 0L ) {
              if( Tq != 1 ){ 
                mask <- jnp$pad(mask,
                        list(c(0L, 0L), c(0L, pad_S), c(0L, pad_S)),
                        mode = "constant", 
                        constant_values = FALSE)
              }
              if( Tq==1 ){ 
                mask <- jnp$pad(mask,
                                list(c(0L, 0L), c(0L, 0L), c(0L, pad_S)),
                                mode = "constant", 
                                constant_values = FALSE)
              }
            }
          }
          
          out <- jax$nn$dot_product_attention(
            q_pad$astype(jnp$float16),
            k_pad$astype(jnp$float16),
            v_pad$astype(jnp$float16),
            mask = mask,
            implementation = "cudnn",
            is_causal = is_causal
          )
          
          # Slice back to original T
          out <- jnp$take(out, jnp$arange(Tq, dtype = jnp$int32), axis = 0L)
        }
        return(out$astype(k$dtype))
      }
      if (impl != "cudnn"){
        # XLA path: supports masks; requires fp32, then cast back
        mask_dpa <- normalize_mask_for_dpa(mask)
        out <- jax$nn$dot_product_attention(
          q$astype(jnp$float32),
          k$astype(jnp$float32),
          v$astype(jnp$float32),
          mask = mask_dpa,
          is_causal = is_causal,
          implementation = "xla"
        )
        return(out$astype(k$dtype))
      }
    }
    
    .__unified_attn_defined <- TRUE
  }
  
  # Kv cache helpers
  {
    if (!exists(".__kv_helpers_defined", inherits = FALSE)) {
      print("Defining KV cache helpers...")
      # Single-position rotary embedding (RoPE) for [H]-dim token vector
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
        jnp$concatenate(list(x_rot_even, x_rot_odd), axis = 2L)
      }
      
      # Allocate KV cache: per layer, K/V are [max_len, num_kv_heads, head_dim]
      kv_cache_allocate <- function(max_len, num_layers, num_kv_heads, head_dim, dtype) {
        make_one <- function() {
          list(
            "k"   = jnp$zeros(list(max_len, num_kv_heads, head_dim), dtype = dtype),
            "v"   = jnp$zeros(list(max_len, num_kv_heads, head_dim), dtype = dtype),
            "len" = jnp$array(0L, dtype = jnp$int32)
          )
        }
        out <- replicate(num_layers, make_one(), simplify = FALSE)
        names(out) <- paste0("d", as.character(1:num_layers))
        out
      }
      
      # Prefill K/V for the known prefix (length = prefix_len)
      # Returns: list("xt_last"=[D], "cache"=cache_updated, "prefix_len"=prefix_len)
      transformer_prefill_kv <- function(xt, x_mask, TransformerList, prefix_len) {
        T_full <- xt$shape[[1]]
        D <- xt$shape[[2]]
        dtype <- xt$dtype
        layer_names <- paste0("d", as.character(seq_len(ModelDepth)))
        pos_ids <- jnp$arange(T_full, dtype = jnp$int32)

        cache <- kv_cache_allocate(
          max_len = T_full,
          num_layers = ModelDepth,
          num_kv_heads = num_kv_heads,
          head_dim = head_dim,
          dtype = dtype
        )

        pm <- make_prefix_index_mask(prefix_len, T_full)
        x_mask_pref <- mask_prefix_rows(x_mask, pm$mask)
        xt <- mask_prefix_rows(xt, pm$mask)
        keys_mask_bool <- jnp$greater(jnp$squeeze(x_mask_pref, 1L), 0)
        mask_keys_prefill <- jnp$expand_dims(keys_mask_bool, 0L)
        mask_keys_prefill <- jnp$expand_dims(mask_keys_prefill, 0L)
        mask_keys_prefill <- jnp$broadcast_to(mask_keys_prefill, list(num_heads, T_full, T_full))
        q_mask_T11 <- jnp$expand_dims(jnp$expand_dims(pm$mask, 1L), 2L)
        is_causal_flag <- (ModelType == "DecoderOnly")

        if (!isTRUE(UseFullAttentionResiduals)) {
          for (l_ in seq_along(layer_names)) {
            L <- TransformerList[[layer_names[[l_]]]]
            xt <- jnp$multiply(NormFxn(xtminu1 <- xt), L$NormScalerInput)
            q <- jnp$dot(xt, L$Multihead$W_q)
            k <- jnp$dot(xt, L$Multihead$W_k)
            v <- jnp$dot(xt, L$Multihead$W_v)
            qh <- jnp$reshape(q, list(T_full, num_heads, head_dim))
            kh_kv <- jnp$reshape(k, list(T_full, num_kv_heads, head_dim))
            vh_kv <- jnp$reshape(v, list(T_full, num_kv_heads, head_dim))
            qh <- apply_rope_batched(qh, pos_ids, head_dim)
            kh_kv <- apply_rope_batched(kh_kv, pos_ids, head_dim)
            qh <- qk_normalize_heads(qh, L$Multihead$QNormScale)
            kh_kv <- qk_normalize_heads(kh_kv, L$Multihead$KNormScale)
            cache[[l_]]$k <- jax$lax$dynamic_update_slice(
              cache[[l_]]$k,
              kh_kv,
              jnp$array(c(0L, 0L, 0L), dtype = jnp$int32)
            )
            cache[[l_]]$v <- jax$lax$dynamic_update_slice(
              cache[[l_]]$v,
              vh_kv,
              jnp$array(c(0L, 0L, 0L), dtype = jnp$int32)
            )
            cache[[l_]]$len <- pm$len
            kh <- repeat_kv_heads(kh_kv, kv_group_size)
            vh <- repeat_kv_heads(vh_kv, kv_group_size)
            attn_out <- dot_product_attention_unified(
              qh,
              kh,
              vh,
              mask = mask_keys_prefill,
              is_causal = is_causal_flag,
              prefer = "auto"
            )$astype(dtype)
            attn_out <- jnp$where(q_mask_T11, attn_out, jnp$zeros_like(attn_out))
            attn_TD <- jnp$reshape(attn_out, list(T_full, num_heads * head_dim))
            attn_proj <- jnp$dot(attn_TD, L$Multihead$W_o)
            xt <- (xtminu1 * jax$nn$softplus(L$ResidCon1$WtSkipPath)) +
              (attn_proj * jax$nn$softplus(L$ResidCon1$WtResidPath))
            xt <- NormFxn(xtminu1 <- xt) * L$NormScalerPostMultiHead
            xt <- jax$nn$swish(ffmap(L$FFN$WideProj1, xt)) *
              ffmap(L$FFN$WideProj2, xt)
            xt <- ffmap(L$FFN$OutProj1, xt)
            xt <- (xtminu1 * jax$nn$softplus(L$ResidCon2$WtSkipPath)) +
              (xt * jax$nn$softplus(L$ResidCon2$WtResidPath))
            xt <- mask_prefix_rows(xt, pm$mask)
          }

          last_nonmasked_i <- jnp$maximum(
            pm$len - jnp$array(1L, dtype = jnp$int32),
            jnp$array(0L, dtype = jnp$int32)
          )
          xt_last <- jnp$take(xt, last_nonmasked_i, axis = 0L)
          return(list("xt_last" = xt_last, "cache" = cache, "prefix_len" = pm$len))
        }

        initial_append <- attnres_append(
          attnres_init_buffer(T_full, D, dtype),
          jnp$array(0L, dtype = jnp$int32),
          xt
        )
        carry_init <- list(xt, initial_append$buffer, initial_append$count, cache)
        layer_branches <- lapply(seq_along(layer_names), function(branch_idx) {
          L <- TransformerList[[layer_names[[branch_idx]]]]
          function(carry_in) {
            xt_in <- carry_in[[1]]
            source_buffer <- carry_in[[2]]
            source_count <- carry_in[[3]]
            cache_in <- carry_in[[4]]

            attn_source <- full_attnres_reduce_buffer(
              source_buffer,
              source_count,
              L$AttnRes1$PseudoQuery,
              L$AttnRes1$NormScale
            )
            xt_norm <- jnp$multiply(NormFxn(attn_source), L$NormScalerInput)
            q <- jnp$dot(xt_norm, L$Multihead$W_q)
            k <- jnp$dot(xt_norm, L$Multihead$W_k)
            v <- jnp$dot(xt_norm, L$Multihead$W_v)
            qh <- jnp$reshape(q, list(T_full, num_heads, head_dim))
            kh_kv <- jnp$reshape(k, list(T_full, num_kv_heads, head_dim))
            vh_kv <- jnp$reshape(v, list(T_full, num_kv_heads, head_dim))
            qh <- apply_rope_batched(qh, pos_ids, head_dim)
            kh_kv <- apply_rope_batched(kh_kv, pos_ids, head_dim)
            qh <- qk_normalize_heads(qh, L$Multihead$QNormScale)
            kh_kv <- qk_normalize_heads(kh_kv, L$Multihead$KNormScale)

            cache_in[[branch_idx]]$k <- jax$lax$dynamic_update_slice(
              cache_in[[branch_idx]]$k,
              kh_kv,
              jnp$array(c(0L, 0L, 0L), dtype = jnp$int32)
            )
            cache_in[[branch_idx]]$v <- jax$lax$dynamic_update_slice(
              cache_in[[branch_idx]]$v,
              vh_kv,
              jnp$array(c(0L, 0L, 0L), dtype = jnp$int32)
            )
            cache_in[[branch_idx]]$len <- pm$len

            kh <- repeat_kv_heads(kh_kv, kv_group_size)
            vh <- repeat_kv_heads(vh_kv, kv_group_size)
            attn_out <- dot_product_attention_unified(
              qh,
              kh,
              vh,
              mask = mask_keys_prefill,
              is_causal = is_causal_flag,
              prefer = "auto"
            )$astype(dtype)
            attn_out <- jnp$where(q_mask_T11, attn_out, jnp$zeros_like(attn_out))
            attn_proj <- jnp$dot(
              jnp$reshape(attn_out, list(T_full, num_heads * head_dim)),
              L$Multihead$W_o
            )
            attn_proj <- mask_prefix_rows(attn_proj, pm$mask)
            attn_append <- attnres_append(source_buffer, source_count, attn_proj)

            mlp_source <- full_attnres_reduce_buffer(
              attn_append$buffer,
              attn_append$count,
              L$AttnRes2$PseudoQuery,
              L$AttnRes2$NormScale
            )
            xt_out <- NormFxn(mlp_source) * L$NormScalerPostMultiHead
            xt_out <- jax$nn$swish(ffmap(L$FFN$WideProj1, xt_out)) *
              ffmap(L$FFN$WideProj2, xt_out)
            xt_out <- ffmap(L$FFN$OutProj1, xt_out)
            xt_out <- mask_prefix_rows(xt_out, pm$mask)
            ffn_append <- attnres_append(attn_append$buffer, attn_append$count, xt_out)

            list(xt_out, ffn_append$buffer, ffn_append$count, cache_in)
          }
        })
        scan_body <- function(carry_in, i) {
          carry_next <- jax$lax$switch(index = i, branches = layer_branches, operand = carry_in)
          list(carry_next, jnp$array(0L, dtype = jnp$int32))
        }
        scan_result <- jax$lax$scan(
          f = scan_body,
          init = carry_init,
          xs = jnp$arange(start = 0L, stop = as.integer(ModelDepth), dtype = jnp$int32)
        )
        final_carry <- scan_result[[1]]
        xt_final <- final_carry[[1]]
        cache_final <- final_carry[[4]]
        last_nonmasked_i <- jnp$maximum(
          pm$len - jnp$array(1L, dtype = jnp$int32),
          jnp$array(0L, dtype = jnp$int32)
        )
        xt_last <- jnp$take(xt_final, last_nonmasked_i, axis = 0L)
        list("xt_last" = xt_last, "cache" = cache_final, "prefix_len" = pm$len)
      }
      
      # Single decode step with KV cache update for position `pos`
      # token_in: [D] (embedding at index pos), returns: list("token_out"=[D], "cache"=updated)
      transformer_decode_step_kv <- function(token_in, pos, TransformerList, cache) {
        D <- token_in$shape[[1]]
        dtype <- token_in$dtype
        layer_names <- paste0("d", as.character(seq_len(ModelDepth)))
        pos_i32 <- jnp$astype(pos, jnp$int32)

        if (!isTRUE(UseFullAttentionResiduals)) {
          xt <- token_in
          for (l_ in seq_along(layer_names)) {
            L <- TransformerList[[layer_names[[l_]]]]
            xt <- jnp$multiply(
              NormFxn(xtminus1 <- xt),
              jnp$squeeze(L$NormScalerInput, 0L)
            )
            q_full <- jnp$dot(xt, L$Multihead$W_q)
            k_full <- jnp$dot(xt, L$Multihead$W_k)
            v_full <- jnp$dot(xt, L$Multihead$W_v)
            q_NH <- jnp$squeeze(
              apply_rope_batched(
                jnp$reshape(q_full, list(1L, num_heads, head_dim)),
                jnp$reshape(pos_i32, list(1L)),
                head_dim
              ),
              0L
            )
            k_KH <- jnp$squeeze(
              apply_rope_batched(
                jnp$reshape(k_full, list(1L, num_kv_heads, head_dim)),
                jnp$reshape(pos_i32, list(1L)),
                head_dim
              ),
              0L
            )
            v_KH <- jnp$reshape(v_full, list(num_kv_heads, head_dim))
            q_NH <- qk_normalize_heads(q_NH, L$Multihead$QNormScale)
            k_KH <- qk_normalize_heads(k_KH, L$Multihead$KNormScale)

            max_len <- cache[[l_]]$k$shape[[1]]
            pos_layer <- jnp$clip(
              pos_i32,
              jnp$array(0L, dtype = jnp$int32),
              jnp$array(max_len - 1L, dtype = jnp$int32)
            )
            write_idx <- jnp$array(c(pos_layer, 0L, 0L), dtype = jnp$int32)
            cache[[l_]]$k <- jax$lax$dynamic_update_slice(
              cache[[l_]]$k,
              jnp$expand_dims(k_KH, 0L),
              write_idx
            )
            cache[[l_]]$v <- jax$lax$dynamic_update_slice(
              cache[[l_]]$v,
              jnp$expand_dims(v_KH, 0L),
              write_idx
            )
            cache[[l_]]$len <- jnp$maximum(
              cache[[l_]]$len,
              jnp$array(pos_layer + 1L, dtype = jnp$int32)
            )

            q_TNH <- jnp$expand_dims(q_NH, 0L)
            K_SNH <- repeat_kv_heads(cache[[l_]]$k, kv_group_size)
            V_SNH <- repeat_kv_heads(cache[[l_]]$v, kv_group_size)
            idx_full <- jnp$arange(max_len, dtype = jnp$int32)
            keys_mask_1d <- jnp$logical_and(
              jnp$less(idx_full, cache[[l_]]$len),
              jnp$less_equal(idx_full, pos_layer)
            )
            mask_keys_decode <- jnp$expand_dims(keys_mask_1d, 0L)
            mask_keys_decode <- jnp$expand_dims(mask_keys_decode, 0L)
            mask_keys_decode <- jnp$broadcast_to(mask_keys_decode, list(num_heads, 1L, max_len))
            attn <- dot_product_attention_unified(
              q = q_TNH,
              k = K_SNH,
              v = V_SNH,
              mask = mask_keys_decode,
              is_causal = FALSE,
              prefer = "xla"
            )$astype(dtype)
            mha_out <- jnp$dot(
              jnp$squeeze(jnp$reshape(attn, list(1L, num_heads * head_dim)), 0L),
              L$Multihead$W_o
            )
            xt <- (xtminus1 * jax$nn$softplus(L$ResidCon1$WtSkipPath)) +
              (mha_out * jax$nn$softplus(L$ResidCon1$WtResidPath))
            xt <- NormFxn(xtminus1 <- xt) * jnp$squeeze(L$NormScalerPostMultiHead, 0L)
            xt <- jax$nn$swish(L$FFN$WideProj1(xt)) * L$FFN$WideProj2(xt)
            xt <- L$FFN$OutProj1(xt)
            xt <- (xtminus1 * jax$nn$softplus(L$ResidCon2$WtSkipPath)) +
              (xt * jax$nn$softplus(L$ResidCon2$WtResidPath))
          }
          return(list("token_out" = xt, "cache" = cache))
        }

        xt <- jnp$expand_dims(token_in, 0L)
        initial_append <- attnres_append(
          attnres_init_buffer(1L, D, dtype),
          jnp$array(0L, dtype = jnp$int32),
          xt
        )
        carry_init <- list(xt, initial_append$buffer, initial_append$count, cache)
        layer_branches <- lapply(seq_along(layer_names), function(branch_idx) {
          L <- TransformerList[[layer_names[[branch_idx]]]]
          function(carry_in) {
            xt_in <- carry_in[[1]]
            source_buffer <- carry_in[[2]]
            source_count <- carry_in[[3]]
            cache_in <- carry_in[[4]]

            attn_source <- full_attnres_reduce_buffer(
              source_buffer,
              source_count,
              L$AttnRes1$PseudoQuery,
              L$AttnRes1$NormScale
            )
            xt_norm <- jnp$multiply(NormFxn(attn_source), L$NormScalerInput)
            q_full <- jnp$dot(xt_norm, L$Multihead$W_q)
            k_full <- jnp$dot(xt_norm, L$Multihead$W_k)
            v_full <- jnp$dot(xt_norm, L$Multihead$W_v)
            q_TNH <- jnp$reshape(q_full, list(1L, num_heads, head_dim))
            k_KH <- jnp$reshape(k_full, list(1L, num_kv_heads, head_dim))
            v_KH <- jnp$reshape(v_full, list(1L, num_kv_heads, head_dim))
            q_TNH <- apply_rope_batched(q_TNH, jnp$reshape(pos_i32, list(1L)), head_dim)
            k_KH <- apply_rope_batched(k_KH, jnp$reshape(pos_i32, list(1L)), head_dim)
            q_TNH <- qk_normalize_heads(q_TNH, L$Multihead$QNormScale)
            k_KH <- qk_normalize_heads(k_KH, L$Multihead$KNormScale)

            max_len <- cache_in[[branch_idx]]$k$shape[[1]]
            pos_layer <- jnp$clip(
              pos_i32,
              jnp$array(0L, dtype = jnp$int32),
              jnp$array(max_len - 1L, dtype = jnp$int32)
            )
            write_idx <- jnp$array(c(pos_layer, 0L, 0L), dtype = jnp$int32)
            cache_in[[branch_idx]]$k <- jax$lax$dynamic_update_slice(
              cache_in[[branch_idx]]$k,
              k_KH,
              write_idx
            )
            cache_in[[branch_idx]]$v <- jax$lax$dynamic_update_slice(
              cache_in[[branch_idx]]$v,
              v_KH,
              write_idx
            )
            cache_in[[branch_idx]]$len <- jnp$maximum(
              cache_in[[branch_idx]]$len,
              jnp$array(pos_layer + 1L, dtype = jnp$int32)
            )

            K_SNH <- repeat_kv_heads(cache_in[[branch_idx]]$k, kv_group_size)
            V_SNH <- repeat_kv_heads(cache_in[[branch_idx]]$v, kv_group_size)
            idx_full <- jnp$arange(max_len, dtype = jnp$int32)
            keys_mask_1d <- jnp$logical_and(
              jnp$less(idx_full, cache_in[[branch_idx]]$len),
              jnp$less_equal(idx_full, pos_layer)
            )
            mask_keys_decode <- jnp$expand_dims(keys_mask_1d, 0L)
            mask_keys_decode <- jnp$expand_dims(mask_keys_decode, 0L)
            mask_keys_decode <- jnp$broadcast_to(mask_keys_decode, list(num_heads, 1L, max_len))
            attn <- dot_product_attention_unified(
              q = q_TNH,
              k = K_SNH,
              v = V_SNH,
              mask = mask_keys_decode,
              is_causal = FALSE,
              prefer = "xla"
            )$astype(dtype)
            attn_proj <- jnp$dot(
              jnp$reshape(attn, list(1L, num_heads * head_dim)),
              L$Multihead$W_o
            )
            attn_append <- attnres_append(source_buffer, source_count, attn_proj)

            mlp_source <- full_attnres_reduce_buffer(
              attn_append$buffer,
              attn_append$count,
              L$AttnRes2$PseudoQuery,
              L$AttnRes2$NormScale
            )
            xt_out <- NormFxn(mlp_source) * L$NormScalerPostMultiHead
            xt_out <- jax$nn$swish(ffmap(L$FFN$WideProj1, xt_out)) *
              ffmap(L$FFN$WideProj2, xt_out)
            xt_out <- ffmap(L$FFN$OutProj1, xt_out)
            ffn_append <- attnres_append(attn_append$buffer, attn_append$count, xt_out)

            list(xt_out, ffn_append$buffer, ffn_append$count, cache_in)
          }
        })
        scan_body <- function(carry_in, i) {
          carry_next <- jax$lax$switch(index = i, branches = layer_branches, operand = carry_in)
          list(carry_next, jnp$array(0L, dtype = jnp$int32))
        }
        scan_result <- jax$lax$scan(
          f = scan_body,
          init = carry_init,
          xs = jnp$arange(start = 0L, stop = as.integer(ModelDepth), dtype = jnp$int32)
        )
        final_carry <- scan_result[[1]]
        list("token_out" = jnp$squeeze(final_carry[[1]], 0L), "cache" = final_carry[[4]])
      }
      
      # Returns a full static index [0, 1, ..., max_len-1] and a boolean mask (idx < prefix_len).
      # Shapes are static; only *values* depend on traced prefix_len.
      make_prefix_index_mask <- function(prefix_len, max_len){
        idx_full  <- jnp$arange(max_len, dtype = jnp$int32)                         # [max_len]
        mask_full <- jnp$less(idx_full, jnp$astype(prefix_len, jnp$int32))          # [max_len] bool
        list("idx" = idx_full, "mask" = mask_full, "len" = jnp$astype(prefix_len, jnp$int32))
      }
      
      # Utility: mask rows of a [T, ...] tensor keeping static T, zeroing out rows >= prefix_len
      mask_prefix_rows <- function(x_T_any, mask_rows_bool){
        # x_T_any: [T, ...], mask_rows_bool: [T] bool
        jnp$where(
          jnp$expand_dims(mask_rows_bool, 1L),
          x_T_any,
          jnp$zeros_like(x_T_any)
        )
      }
      .__kv_helpers_defined <- TRUE
    }
  }

  TransformerStep_NoCache <- function(xt, TransformerList_d, x_mask_attn) {
    xt <- jnp$multiply(NormFxn(xtm1 <- xt), TransformerList_d$NormScalerInput)

    if (UseLatentAttention) {
      stop("Latent Attention not double checked -- do not use")
    }

    q_ <- jnp$dot(xt, TransformerList_d$Multihead$W_q)
    k_ <- jnp$dot(xt, TransformerList_d$Multihead$W_k)
    v_ <- jnp$dot(xt, TransformerList_d$Multihead$W_v)

    q_ <- jnp$reshape(q_, list(q_$shape[[1]], num_heads, head_dim))
    k_ <- jnp$reshape(k_, list(k_$shape[[1]], num_kv_heads, head_dim))
    v_ <- jnp$reshape(v_, list(v_$shape[[1]], num_kv_heads, head_dim))

    pos_ids <- jnp$arange(q_$shape[[1]], dtype = jnp$int32)
    q_ <- apply_rope_batched(q_, pos_ids, head_dim)
    k_ <- apply_rope_batched(k_, pos_ids, head_dim)
    q_ <- qk_normalize_heads(q_, TransformerList_d$Multihead$QNormScale)
    k_ <- qk_normalize_heads(k_, TransformerList_d$Multihead$KNormScale)
    k_ <- repeat_kv_heads(k_, kv_group_size)
    v_ <- repeat_kv_heads(v_, kv_group_size)

    mask_bool <- jnp$greater(x_mask_attn, 0)
    mask_bool <- jnp$broadcast_to(
      mask_bool,
      list(num_heads, mask_bool$shape[[1]], mask_bool$shape[[2]])
    )

    xt_attn <- dot_product_attention_unified(
      q = q_,
      k = k_,
      v = v_,
      mask = mask_bool,
      is_causal = (ModelType == "DecoderOnly"),
      prefer = "auto"
    )$astype(k_$dtype)

    xt_attn <- jnp$reshape(xt_attn, list(xt_attn$shape[[1]], num_heads * head_dim))
    xt <- jnp$dot(xt_attn, TransformerList_d$Multihead$W_o)
    xt <- xtm1 <- xtm1 * jax$nn$softplus(TransformerList_d$ResidCon1$WtSkipPath) +
      xt * jax$nn$softplus(TransformerList_d$ResidCon1$WtResidPath)
    xt <- NormFxn(xt) * TransformerList_d$NormScalerPostMultiHead
    xt <- jax$nn$swish(ffmap(TransformerList_d$FFN$WideProj1, xt)) *
      ffmap(TransformerList_d$FFN$WideProj2, xt)
    xt <- ffmap(TransformerList_d$FFN$OutProj1, xt)
    xt <- xtm1 <- xtm1 * jax$nn$softplus(TransformerList_d$ResidCon2$WtSkipPath) +
      xt * jax$nn$softplus(TransformerList_d$ResidCon2$WtResidPath)
    xt
  }

  RunTransformerBackbone_FullAttnRes <- function(xt, x_mask, TransformerList) {
    if (UseLatentAttention) {
      stop("Latent Attention not double checked -- do not use")
    }

    row_mask_bool <- jnp$squeeze(jnp$greater(x_mask, 0), 1L)
    x_mask_attn <- jnp$matmul(x_mask, jnp$transpose(x_mask))
    mask_bool <- jnp$greater(x_mask_attn, 0)
    mask_bool <- jnp$broadcast_to(
      mask_bool,
      list(num_heads, mask_bool$shape[[1]], mask_bool$shape[[2]])
    )
    xt <- mask_sequence_rows_2d(xt, row_mask_bool)
    layer_names <- paste0("d", as.character(seq_len(ModelDepth)))
    pos_ids <- jnp$arange(xt$shape[[1]], dtype = jnp$int32)
    initial_append <- attnres_append(
      attnres_init_buffer(xt$shape[[1]], xt$shape[[2]], xt$dtype),
      jnp$array(0L, dtype = jnp$int32),
      xt
    )
    carry_init <- list(xt, initial_append$buffer, initial_append$count)
    layer_branches <- lapply(layer_names, function(layer_name) {
      L <- TransformerList[[layer_name]]
      function(carry_in) {
        source_buffer <- carry_in[[2]]
        source_count <- carry_in[[3]]

        attn_source <- full_attnres_reduce_buffer(
          source_buffer,
          source_count,
          L$AttnRes1$PseudoQuery,
          L$AttnRes1$NormScale
        )
        xt_norm <- jnp$multiply(NormFxn(attn_source), L$NormScalerInput)
        q_ <- jnp$dot(xt_norm, L$Multihead$W_q)
        k_ <- jnp$dot(xt_norm, L$Multihead$W_k)
        v_ <- jnp$dot(xt_norm, L$Multihead$W_v)
        q_ <- jnp$reshape(q_, list(q_$shape[[1]], num_heads, head_dim))
        k_ <- jnp$reshape(k_, list(k_$shape[[1]], num_kv_heads, head_dim))
        v_ <- jnp$reshape(v_, list(v_$shape[[1]], num_kv_heads, head_dim))
        q_ <- apply_rope_batched(q_, pos_ids, head_dim)
        k_ <- apply_rope_batched(k_, pos_ids, head_dim)
        q_ <- qk_normalize_heads(q_, L$Multihead$QNormScale)
        k_ <- qk_normalize_heads(k_, L$Multihead$KNormScale)
        k_ <- repeat_kv_heads(k_, kv_group_size)
        v_ <- repeat_kv_heads(v_, kv_group_size)

        xt_attn <- dot_product_attention_unified(
          q = q_,
          k = k_,
          v = v_,
          mask = mask_bool,
          is_causal = (ModelType == "DecoderOnly"),
          prefer = "auto"
        )$astype(k_$dtype)

        attn_proj <- jnp$dot(
          jnp$reshape(xt_attn, list(xt_attn$shape[[1]], num_heads * head_dim)),
          L$Multihead$W_o
        )
        attn_proj <- mask_sequence_rows_2d(attn_proj, row_mask_bool)
        attn_append <- attnres_append(source_buffer, source_count, attn_proj)

        mlp_source <- full_attnres_reduce_buffer(
          attn_append$buffer,
          attn_append$count,
          L$AttnRes2$PseudoQuery,
          L$AttnRes2$NormScale
        )
        xt_out <- NormFxn(mlp_source) * L$NormScalerPostMultiHead
        xt_out <- jax$nn$swish(ffmap(L$FFN$WideProj1, xt_out)) *
          ffmap(L$FFN$WideProj2, xt_out)
        xt_out <- ffmap(L$FFN$OutProj1, xt_out)
        xt_out <- mask_sequence_rows_2d(xt_out, row_mask_bool)
        ffn_append <- attnres_append(attn_append$buffer, attn_append$count, xt_out)

        list(xt_out, ffn_append$buffer, ffn_append$count)
      }
    })
    scan_body <- function(carry_in, i) {
      carry_next <- jax$lax$switch(index = i, branches = layer_branches, operand = carry_in)
      list(carry_next, jnp$array(0L, dtype = jnp$int32))
    }
    scan_result <- jax$lax$scan(
      f = scan_body,
      init = carry_init,
      xs = jnp$arange(start = 0L, stop = as.integer(ModelDepth), dtype = jnp$int32)
    )
    scan_result[[1]][[1]]
  }

  RunTransformerBackbone <- function(xt, x_mask, TransformerList) {
    if (isTRUE(UseFullAttentionResiduals)) {
      return(RunTransformerBackbone_FullAttnRes(
        xt = xt,
        x_mask = x_mask,
        TransformerList = TransformerList
      ))
    }

    x_mask_attn <- jnp$matmul(x_mask, jnp$transpose(x_mask))
    layer_names <- paste0("d", as.character(seq_len(ModelDepth)))
    layer_branches <- lapply(layer_names, function(layer_name) {
      layer_params <- TransformerList[[layer_name]]
      function(xt_in) TransformerStep_NoCache(xt_in, layer_params, x_mask_attn)
    })
    if (length(layer_branches) == 0L) {
      return(xt)
    }
    scan_body <- function(carry_xt, i) {
      xt_next <- jax$lax$switch(index = i, branches = layer_branches, operand = carry_xt)
      list(xt_next, jnp$array(0L, dtype = jnp$int32))
    }
    jax$lax$scan(
      f = scan_body,
      init = xt,
      xs = jnp$arange(start = 0L, stop = as.integer(ModelDepth), dtype = jnp$int32)
    )[[1]]
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
        # Properly split keys for reproducible random initialization
        multihead_keys <- jax$random$split(key, 4L)
        TransformerList[[l_]]$Multihead <- list(
          "W_q" = make_w(list(ModelDims, q_proj_dim), multihead_keys[1]),
          "W_k" = make_w(list(ModelDims, kv_proj_dim), multihead_keys[2]),
          "W_v" = make_w(list(ModelDims, kv_proj_dim), multihead_keys[3]),
          "W_o" = make_w(list(q_proj_dim, ModelDims), multihead_keys[4]),
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
        key <- jax$random$split(key)[[1]]  # advance key for next use
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
