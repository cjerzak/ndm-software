# Content of SuperLModel_BackboneTransformer.R
print("Done with SuperLModel_BackboneTransformer.R")
if(backbonePath == "initialize"){
  TRY_FLASH <- tryCatch(!any(grepl("V100", sapply(jax$devices(), function(d) d$device_kind))), error = function(e) FALSE)
  EnableKVCaching <- TRUE & (ModelType == "DecoderOnly")
  # Full attention residuals are now the default transformer residual path.
  # Set UseFullAttentionResiduals = FALSE in runtime globals to opt back into
  # the legacy additive residual implementation for compatibility testing.
  UseFullAttentionResiduals <- isTRUE(get0("UseFullAttentionResiduals", ifnotfound = TRUE))
  FullAttentionResidualEps <- as.numeric(get0("FullAttentionResidualEps", ifnotfound = 1e-6))
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
      print(sprintf("Using attention compute path: [%s]",impl))
      
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

      full_attnres_reduce <- function(sources, pseudo_query, eps = FullAttentionResidualEps) {
        rank <- length(sources$shape)
        if (!(rank %in% c(2L, 3L))) {
          stop("full_attnres_reduce expects a [N, D] or [N, T, D] tensor.", call. = FALSE)
        }

        eps_f32 <- jnp$array(as.numeric(eps), dtype = jnp$float32)
        sources_f32 <- sources$astype(jnp$float32)
        query_f32 <- pseudo_query$astype(jnp$float32)
        rms <- jnp$sqrt(jnp$mean(jnp$square(sources_f32), axis = -1L, keepdims = TRUE) + eps_f32)
        keys_f32 <- jnp$divide(sources_f32, rms)

        if (rank == 2L) {
          logits <- jnp$einsum("nd,d->n", keys_f32, query_f32)
          weights <- jax$nn$softmax(logits, axis = 0L)
          return(jnp$einsum("n,nd->d", weights, sources_f32)$astype(sources$dtype))
        }

        logits <- jnp$einsum("ntd,d->nt", keys_f32, query_f32)
        weights <- jax$nn$softmax(logits, axis = 0L)
        jnp$einsum("nt,ntd->td", weights, sources_f32)$astype(sources$dtype)
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
        # xt: [T, D], x_mask: [T, 1]; prefill over logical prefix [0:prefix_len)
        T_full <- xt$shape[[1]]
        D      <- xt$shape[[2]]
        dtype  <- xt$dtype
        
        # Heads info (shared across layers)
        num_heads <- num_heads
        num_kv_heads <- num_kv_heads
        head_dim  <- head_dim
        kv_group_size <- kv_group_size
        
        # Allocate KV cache for the full static capacity T_full
        cache <- kv_cache_allocate(
          max_len    = T_full,
          num_layers = ModelDepth, #length(TransformerList) - 1L,  # minus DecoderProj
          num_kv_heads  = num_kv_heads,
          head_dim   = head_dim,
          dtype      = dtype
        )
        
        # --- Static mask setup (no dynamic shapes) ----------------------------------
        # Build [T_full]-length mask with TRUE for t < prefix_len
        pm <- make_prefix_index_mask(prefix_len, T_full)  # pm$mask: [T_full] bool, pm$len: int32
        
        # Zero-out rows >= prefix_len for inputs and x_mask; keep static shapes
        x_mask_pref <- mask_prefix_rows(x_mask, pm$mask)  # [T_full, 1]
        xt    <- mask_prefix_rows(xt,     pm$mask)  # [T_full, D]
        layer_outputs <- if (UseFullAttentionResiduals) list(xt) else NULL
        
        # Keys-only mask (avoid all-False query rows):
        # Start from [T_full] 1D keep-mask for keys, then make [1, N, 1, S] (B=1, broadcast over T)
        keys_mask_1d   <- jnp$squeeze(x_mask_pref, 1L)               # [T_full]
        keys_mask_bool <- jnp$greater(keys_mask_1d, 0)               # [T_full] bool
        
        mask_keys_prefill   <- jnp$expand_dims(keys_mask_bool, 0L)        # [1, T_full]
        #mask_keys_prefill   <- jnp$expand_dims(mask_keys_prefill, 0L)          # [1, 1, T_full]
        mask_keys_prefill   <- jnp$expand_dims(mask_keys_prefill, 0L)          # [1, 1, 1, T_full]
        #mask_keys_prefill   <- jnp$broadcast_to(mask_keys_prefill, list(1L, num_heads, 1L, T_full))  # [1, N, 1, S]
        #mask_keys_prefill   <- jnp$broadcast_to(mask_keys_prefill, list(1L, num_heads, T_full, T_full)) # EXPERIMENTAL999
        mask_keys_prefill   <- jnp$broadcast_to(mask_keys_prefill, list(num_heads, T_full, T_full)) # EXPERIMENTAL999 
        
        # EXPERIMENTAL999
        #mask_TS      <- jnp$broadcast_to(keys_mask_bool, list(T_full, T_full))        # [T, S]
        #mask_keys_prefill <- jnp$broadcast_to(jnp$expand_dims(mask_TS, 0L),                # [1, T, S]
                                         #list(num_heads, T_full, T_full))             # [N,
        #mask_keys_prefill <- jnp$broadcast_to(jnp$expand_dims(mask_TS, 0L), list(num_heads, T_full, T_full))   
        
        # Query-row mask (to zero attention outputs for rows >= prefix_len) -> [T,1,1]
        q_mask_T11 <- jnp$expand_dims(jnp$expand_dims(pm$mask, 1L), 2L)  # [T_full, 1, 1]
        
        # Causality flag consistent with your "run" path
        is_causal_flag <- (ModelType == "DecoderOnly")
        
        # --- Layer loop --------------------------------------------------------------
        for (l_ in 1L:ModelDepth) {
          L <- eval(parse(text = sprintf("TransformerList$d%s", l_)))
          
          # Pre-norm + scale
          if (UseFullAttentionResiduals) {
            attn_source <- full_attnres_reduce(
              jnp$stack(layer_outputs, axis = 0L),
              L$AttnRes1$PseudoQuery
            )
            xt <- jnp$multiply(NormFxn(attn_source), L$NormScalerInput)  # [T_full, D]
          } else {
            xt <- jnp$multiply(NormFxn(xtminu1 <- xt), L$NormScalerInput)  # [T_full, D]
          }
          
          # Q/K/V projections from *unrotated* activations
          q <- jnp$dot(xt,  L$Multihead$W_q)                          # [T_full, D]
          k <- jnp$dot(xt,  L$Multihead$W_k)                          # [T_full, D_kv]
          v <- jnp$dot(xt,  L$Multihead$W_v)                          # [T_full, D_kv]
          
          # Reshape to query and KV heads separately.
          qh <- jnp$reshape(q, list(T_full, num_heads, head_dim))           # [T, N, H]
          kh_kv <- jnp$reshape(k, list(T_full, num_kv_heads, head_dim))     # [T, N_kv, H]
          vh_kv <- jnp$reshape(v, list(T_full, num_kv_heads, head_dim))     # [T, N_kv, H]
          
          # Apply RoPE *after* projection, per time-step and per head
          pos_ids <- jnp$arange(T_full, dtype = jnp$int32)                  # [T]
          
          apply_rope_one_row <- function(NH_row, p) {
            # NH_row: [N, H] for one timestep; p: scalar position
            jax$vmap(function(hvec){ rope_apply_single(hvec, p, head_dim) }, in_axes = 0L)(NH_row)  # [N, H]
          }
          apply_rope_all <- jax$vmap(apply_rope_one_row, in_axes = list(0L, 0L))                    # over T
          
          # rotate queries and keys 
          qh <- apply_rope_all(qh, pos_ids)                                 # [T, N, H]
          kh_kv <- apply_rope_all(kh_kv, pos_ids)                           # [T, N_kv, H]
          qh <- qk_normalize_heads(qh, L$Multihead$QNormScale)
          kh_kv <- qk_normalize_heads(kh_kv, L$Multihead$KNormScale)
          
          # Save full (masked) slices to cache; logical length is pm$len
          k_slice_idx <- jnp$array(c(0L, 0L, 0L), dtype = jnp$int32)
          v_slice_idx <- jnp$array(c(0L, 0L, 0L), dtype = jnp$int32)
          cache[[l_]]$k   <- jax$lax$dynamic_update_slice(cache[[l_]]$k, kh_kv, k_slice_idx)
          cache[[l_]]$v   <- jax$lax$dynamic_update_slice(cache[[l_]]$v, vh_kv, v_slice_idx)
          cache[[l_]]$len <- pm$len

          kh <- repeat_kv_heads(kh_kv, kv_group_size)                       # [T, N, H]
          vh <- repeat_kv_heads(vh_kv, kv_group_size)                       # [T, N, H]

          # Attention (NO transpose). API expects [T, N, H] (or [B,T,N,H]).
          attn_out <- dot_product_attention_unified(
            qh,      # [T, N, H]
            kh,      # [S, N, H]  (S == T here)
            vh,      # [S, N, H]
            mask = mask_keys_prefill,         # [1, N, 1, S] (broadcast over T)
            is_causal = is_causal_flag, 
            prefer = "auto"
          )$astype(dtype)                # [T, N, H]
          
          # Zero out query rows >= prefix_len
          attn_out <- jnp$where(q_mask_T11, attn_out, jnp$zeros_like(attn_out))  # [T, N, H]
          
          # Merge heads back to [T_full, D] and output projection
          attn_TNH <- attn_out                                              # [T, N, H]
          attn_TD  <- jnp$reshape(attn_TNH, list(T_full, num_heads * head_dim))  # [T, D]
          attn_proj <- jnp$dot(attn_TD, L$Multihead$W_o)                     # [T, D]
          
          # Residual + pre-FFN norm
          if (UseFullAttentionResiduals) {
            xt <- mask_prefix_rows(attn_proj, pm$mask)
            layer_outputs[[length(layer_outputs) + 1L]] <- xt
            mlp_source <- full_attnres_reduce(
              jnp$stack(layer_outputs, axis = 0L),
              L$AttnRes2$PseudoQuery
            )
            xt <- NormFxn(mlp_source) * L$NormScalerPostMultiHead
          } else {
            xt <- (xtminu1 * jax$nn$softplus(L$ResidCon1$WtSkipPath)) +
              (attn_proj * jax$nn$softplus(L$ResidCon1$WtResidPath))
            xt <- NormFxn(xtminu1 <- xt) * L$NormScalerPostMultiHead
          }
          
          # SwiGLU FFN
          xt <- jax$nn$swish( ffmap(L$FFN$WideProj1, xt) ) * 
                              ffmap(L$FFN$WideProj2, xt)
          xt <- ffmap(L$FFN$OutProj1, xt)
          
          # Final residual of this layer
          if (UseFullAttentionResiduals) {
            xt <- mask_prefix_rows(xt, pm$mask)  # [T_full, D]
            layer_outputs[[length(layer_outputs) + 1L]] <- xt
          } else {
            xt <- (xtminu1 * jax$nn$softplus(L$ResidCon2$WtSkipPath)) +
                          (xt    * jax$nn$softplus(L$ResidCon2$WtResidPath))
             
            # Safety: keep padded rows identically zero
            xt <- mask_prefix_rows(xt, pm$mask)  # [T_full, D]
          }
        }
        
        # Last valid prefix hidden: index = max(prefix_len-1, 0)
        last_nonmasked_i <- jnp$maximum(
          pm$len - jnp$array(1L, dtype = jnp$int32),
          jnp$array(0L, dtype = jnp$int32)
        )
        xt_last <- jnp$take(xt, last_nonmasked_i, axis = 0L)  # [D]
        
        list("xt_last" = xt_last, "cache" = cache, "prefix_len" = pm$len)
      }
      
      # Single decode step with KV cache update for position `pos`
      # token_in: [D] (embedding at index pos), returns: list("token_out"=[D], "cache"=updated)
      transformer_decode_step_kv <- function(token_in, pos, TransformerList, cache) {
        D          <- token_in$shape[[1]]
        dtype      <- token_in$dtype
        num_heads  <- num_heads
        num_kv_heads <- num_kv_heads
        head_dim   <- head_dim
        kv_group_size <- kv_group_size
        
        xt <- token_in
        layer_outputs <- if (UseFullAttentionResiduals) list(token_in) else NULL
        for (l_ in 1:ModelDepth) {
          L <- eval(parse(text = sprintf("TransformerList$d%s", l_)))
          
          # Pre-norm
          if (UseFullAttentionResiduals) {
            attn_source <- full_attnres_reduce(
              jnp$stack(layer_outputs, axis = 0L),
              L$AttnRes1$PseudoQuery
            )
            xt <- jnp$multiply(
              NormFxn(attn_source),
              jnp$squeeze(L$NormScalerInput, 0L)
            )  # [D]
          } else {
            xt <- jnp$multiply(
              NormFxn(xtminus1 <- xt),
              jnp$squeeze(L$NormScalerInput, 0L)
            )  # [D]
          }
          
          # Project Q/K/V for this single token
          q_full <- jnp$dot(xt, L$Multihead$W_q)               # [D]
          k_full <- jnp$dot(xt, L$Multihead$W_k)               # [D_kv]
          v_full <- jnp$dot(xt, L$Multihead$W_v)               # [D_kv]
          
          # Reshape to query and KV heads.
          q_NH <- jnp$reshape(q_full, list(num_heads, head_dim))
          k_KH <- jnp$reshape(k_full, list(num_kv_heads, head_dim))
          v_KH <- jnp$reshape(v_full, list(num_kv_heads, head_dim))
          
          # Apply RoPE with absolute position = pos (vectorized over heads)
          apply_rope <- jax$vmap(function(hvec){rope_apply_single(hvec, pos, head_dim)},
                                 in_axes = 0L)
          q_NH <- apply_rope(q_NH)  # [N, H]
          k_KH <- apply_rope(k_KH)  # [N_kv, H]
          q_NH <- qk_normalize_heads(q_NH, L$Multihead$QNormScale)
          k_KH <- qk_normalize_heads(k_KH, L$Multihead$KNormScale)
          
          # --- Write K/V for this pos into cache (static shapes; dynamic index ok) ---
          max_len  <- cache[[l_]]$k$shape[[1]]                   # static
          pos_i32  <- jnp$astype(pos, jnp$int32)
          pos_i32  <- jnp$clip(pos_i32, jnp$array(0L, jnp$int32), 
                               jnp$array(max_len - 1L, jnp$int32))
          
          k_write_idx <- jnp$array(c(pos_i32, 0L, 0L), dtype = jnp$int32)
          v_write_idx <- jnp$array(c(pos_i32, 0L, 0L), dtype = jnp$int32)
          
          cache[[l_]]$k <- jax$lax$dynamic_update_slice(
            cache[[l_]]$k, jnp$expand_dims(k_KH, 0L), k_write_idx)  # [max_len, N_kv, H]
          cache[[l_]]$v <- jax$lax$dynamic_update_slice(
            cache[[l_]]$v, jnp$expand_dims(v_KH, 0L), v_write_idx)  # [max_len, N_kv, H]
          
          # Logical cache length: at least pos+1
          cache[[l_]]$len <- jnp$maximum(cache[[l_]]$len,
                                         jnp$array(pos_i32 + 1L, dtype = jnp$int32))
          
          # --- Attention: q=[T=1,N,H], K/V=[S=max_len,N,H] (no transposes) ----------
          # Pack the single-token query to [1, N, H]
          q_TNH <- jnp$expand_dims(q_NH, 0L)                     # [1, N, H]
          
          # Use full K/V (static S=max_len); restrict with a keys-only mask.
          K_SNH <- repeat_kv_heads(cache[[l_]]$k, kv_group_size) # [max_len, N, H]
          V_SNH <- repeat_kv_heads(cache[[l_]]$v, kv_group_size) # [max_len, N, H]
          
          # Keys mask from logical length and current pos (strict causality: keys <= pos)
          idx_full       <- jnp$arange(max_len, dtype = jnp$int32)        # [max_len]
          len_mask_1d    <- jnp$less(idx_full, cache[[l_]]$len)           # [max_len] bool
          past_mask_1d   <- jnp$less_equal(idx_full, pos_i32)             # [max_len] bool
          keys_mask_1d   <- jnp$logical_and(len_mask_1d, past_mask_1d)    # [max_len] bool
          
          # 4D mask: [1, N, 1, S]  (B=1, broadcast over T=1) - OLD 
          #mask_keys_4d <- jnp$expand_dims(keys_mask_1d, 0L)      # [1, S]
          #mask_keys_4d <- jnp$expand_dims(mask_keys_4d, 0L)      # [1, 1, S]
          #mask_keys_4d <- jnp$expand_dims(mask_keys_4d, 0L)      # [1, 1, 1, S]
          #mask_keys_4d <- jnp$broadcast_to(mask_keys_4d, list(1L, num_heads, 1L, max_len))  # [1, N, 1, S]
          
          # new - 999
          mask_keys_decode <- jnp$expand_dims(keys_mask_1d, 0L)  # [1, S]
          mask_keys_decode <- jnp$expand_dims(mask_keys_decode, 0L)  # [1, 1, S]
          mask_keys_decode <- jnp$broadcast_to(mask_keys_decode, list(num_heads, 1L, max_len))  # [N, 1, S]
          
          # EXPERIMENTAL999
          #q_TNH <- jnp$expand_dims(q_TNH, 0L)  # [1, T, N, H]
          #K_SNH <- jnp$expand_dims(K_SNH, 0L)  # [1, T, N, H]
          #V_SNH <- jnp$expand_dims(V_SNH, 0L)  # [1, T, N, H]
          
          attn <- dot_product_attention_unified(
            q = q_TNH,   # [1, N, H]
            k = K_SNH,   # [S, N, H]
            v = V_SNH,   # [S, N, H]
            mask = mask_keys_decode,         # [1, N, 1, S]
            # Do NOT set is_causal=True here; the built-in triangular mask would assume
            # the query index is 0..T-1. We’re at an arbitrary absolute `pos`, so we
            # enforce causality via keys_mask_4d instead.
            is_causal = FALSE,
            prefer = "xla" # flash breaking on 1 to S attention 
          )$astype(dtype)                                        # [1, N, H]

          attn <- jnp$squeeze(attn, 0L)  # [1, T, N, H]
          
          # Merge heads back to [D] and project out
          attn_TNH <- attn                                        # [1, N, H]
          attn_TD  <- jnp$reshape(attn_TNH, list(1L, num_heads * head_dim))  # [1, D]
          mha_out  <- jnp$dot(jnp$squeeze(attn_TD, 0L), L$Multihead$W_o)     # [D]
          
          # Residual + pre-FFN norm
          if (UseFullAttentionResiduals) {
            layer_outputs[[length(layer_outputs) + 1L]] <- mha_out
            mlp_source <- full_attnres_reduce(
              jnp$stack(layer_outputs, axis = 0L),
              L$AttnRes2$PseudoQuery
            )
            xt <- NormFxn(mlp_source) * jnp$squeeze(L$NormScalerPostMultiHead, 0L)
          } else {
            xt <- (xtminus1 * jax$nn$softplus(L$ResidCon1$WtSkipPath)) +
              (mha_out * jax$nn$softplus(L$ResidCon1$WtResidPath))
            xt <- NormFxn(xtminus1 <- xt) * jnp$squeeze(L$NormScalerPostMultiHead, 0L)
          }
          
          # SwiGLU FFN
          xt <- jax$nn$swish(L$FFN$WideProj1(xt)) * L$FFN$WideProj2(xt)
          xt <- L$FFN$OutProj1(xt)
          
          # Final residual
          if (UseFullAttentionResiduals) {
            layer_outputs[[length(layer_outputs) + 1L]] <- xt
          } else {
            xt <- (xtminus1 * jax$nn$softplus(L$ResidCon2$WtSkipPath)) +
                      (xt * jax$nn$softplus(L$ResidCon2$WtResidPath))
          }
        }
        
        list("token_out" = xt, "cache" = cache)
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
    apply_rope_one_row <- function(NH_row, p) {
      jax$vmap(function(hvec) {
        rope_apply_single(hvec, p, head_dim)
      }, in_axes = 0L)(NH_row)
    }
    apply_rope_all <- jax$vmap(apply_rope_one_row, in_axes = list(0L, 0L))
    q_ <- apply_rope_all(q_, pos_ids)
    k_ <- apply_rope_all(k_, pos_ids)
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
    layer_outputs <- list(xt)

    for (l_ in 1L:ModelDepth) {
      L <- TransformerList[[paste0("d", as.character(l_))]]
      attn_source <- full_attnres_reduce(
        jnp$stack(layer_outputs, axis = 0L),
        L$AttnRes1$PseudoQuery
      )
      xt_norm <- jnp$multiply(NormFxn(attn_source), L$NormScalerInput)

      q_ <- jnp$dot(xt_norm, L$Multihead$W_q)
      k_ <- jnp$dot(xt_norm, L$Multihead$W_k)
      v_ <- jnp$dot(xt_norm, L$Multihead$W_v)

      q_ <- jnp$reshape(q_, list(q_$shape[[1]], num_heads, head_dim))
      k_ <- jnp$reshape(k_, list(k_$shape[[1]], num_kv_heads, head_dim))
      v_ <- jnp$reshape(v_, list(v_$shape[[1]], num_kv_heads, head_dim))

      pos_ids <- jnp$arange(q_$shape[[1]], dtype = jnp$int32)
      apply_rope_one_row <- function(NH_row, p) {
        jax$vmap(function(hvec) {
          rope_apply_single(hvec, p, head_dim)
        }, in_axes = 0L)(NH_row)
      }
      apply_rope_all <- jax$vmap(apply_rope_one_row, in_axes = list(0L, 0L))
      q_ <- apply_rope_all(q_, pos_ids)
      k_ <- apply_rope_all(k_, pos_ids)
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

      xt_attn <- jnp$reshape(xt_attn, list(xt_attn$shape[[1]], num_heads * head_dim))
      attn_proj <- jnp$dot(xt_attn, L$Multihead$W_o)
      attn_proj <- mask_sequence_rows_2d(attn_proj, row_mask_bool)
      layer_outputs[[length(layer_outputs) + 1L]] <- attn_proj

      mlp_source <- full_attnres_reduce(
        jnp$stack(layer_outputs, axis = 0L),
        L$AttnRes2$PseudoQuery
      )
      xt_norm <- NormFxn(mlp_source) * L$NormScalerPostMultiHead
      xt <- jax$nn$swish(ffmap(L$FFN$WideProj1, xt_norm)) *
        ffmap(L$FFN$WideProj2, xt_norm)
      xt <- ffmap(L$FFN$OutProj1, xt)
      xt <- mask_sequence_rows_2d(xt, row_mask_bool)
      layer_outputs[[length(layer_outputs) + 1L]] <- xt
    }

    xt
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
            "PseudoQuery" = jnp$zeros(list(ModelDims), dtype = jaxFloatType)
          )
          TransformerList[[l_]]$AttnRes2 <- list(
            "PseudoQuery" = jnp$zeros(list(ModelDims), dtype = jaxFloatType)
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
                                                                 scale =  0.0000001)$sample( list(ModelDims), seed = 4002L+ key)$astype(jaxFloatType,1L)),
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
