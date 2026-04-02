# Content of SuperLModel_BackboneTransformer.R
print("Done with SuperLModel_BackboneTransformer.R")
if(backbonePath == "initialize"){
  TRY_FLASH <- tryCatch(!any(grepl("V100", sapply(jax$devices(), function(d) d$device_kind))), error = function(e) FALSE)
  EnableKVCaching <- get0("EnableKVCaching", ifnotfound = (ModelType == "DecoderOnly"))
  EnableKVCaching <- isTRUE(EnableKVCaching) && (ModelType == "DecoderOnly")
  
  # --- Unified dot-product attention helper (flash or XLA) ----------------------
  if (!exists(".__unified_attn_defined", inherits = TRUE)) {
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
    if (!exists(".__kv_helpers_defined", inherits = TRUE)) {
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
      
      # Allocate KV cache: per layer, K/V are [max_len, num_heads, head_dim]
      kv_cache_allocate <- function(max_len, num_layers, num_heads, head_dim, dtype) {
        make_one <- function() {
          list(
            "k"   = jnp$zeros(list(max_len, num_heads, head_dim), dtype = dtype),
            "v"   = jnp$zeros(list(max_len, num_heads, head_dim), dtype = dtype),
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
        head_dim  <- head_dim
        
        # Allocate KV cache for the full static capacity T_full
        cache <- kv_cache_allocate(
          max_len    = T_full,
          num_layers = ModelDepth, #length(TransformerList) - 1L,  # minus DecoderProj
          num_heads  = num_heads,
          head_dim   = head_dim,
          dtype      = dtype
        )
        
        # --- Static mask setup (no dynamic shapes) ----------------------------------
        # Build [T_full]-length mask with TRUE for t < prefix_len
        pm <- make_prefix_index_mask(prefix_len, T_full)  # pm$mask: [T_full] bool, pm$len: int32
        
        # Zero-out rows >= prefix_len for inputs and x_mask; keep static shapes
        x_mask_pref <- mask_prefix_rows(x_mask, pm$mask)  # [T_full, 1]
        xt    <- mask_prefix_rows(xt,     pm$mask)  # [T_full, D]
        
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
          xt <- jnp$multiply(NormFxn( xtminu1 <- xt), L$NormScalerInput)  # [T_full, D]
          
          # Q/K/V projections from *unrotated* activations
          q <- jnp$dot(xt,  L$Multihead$W_q)                          # [T_full, D]
          k <- jnp$dot(xt,  L$Multihead$W_k)                          # [T_full, D]
          v <- jnp$dot(xt,  L$Multihead$W_v)                          # [T_full, D]
          
          # Reshape to heads: [T, D] -> [T, N, H]
          qh <- jnp$reshape(q, list(T_full, num_heads, head_dim))           # [T, N, H]
          kh <- jnp$reshape(k, list(T_full, num_heads, head_dim))           # [T, N, H]
          vh <- jnp$reshape(v, list(T_full, num_heads, head_dim))           # [T, N, H] (no RoPE on V)
          
          # Apply RoPE *after* projection, per time-step and per head
          pos_ids <- jnp$arange(T_full, dtype = jnp$int32)                  # [T]
          
          apply_rope_one_row <- function(NH_row, p) {
            # NH_row: [N, H] for one timestep; p: scalar position
            jax$vmap(function(hvec){ rope_apply_single(hvec, p, head_dim) }, in_axes = 0L)(NH_row)  # [N, H]
          }
          apply_rope_all <- jax$vmap(apply_rope_one_row, in_axes = list(0L, 0L))                    # over T
          
          # rotate queries and keys 
          qh <- apply_rope_all(qh, pos_ids)                                 # [T, N, H]
          kh <- apply_rope_all(kh, pos_ids)                                 # [T, N, H]
          
          # Save full (masked) slices to cache; logical length is pm$len
          k_slice_idx <- jnp$array(c(0L, 0L, 0L), dtype = jnp$int32)
          v_slice_idx <- jnp$array(c(0L, 0L, 0L), dtype = jnp$int32)
          cache[[l_]]$k   <- jax$lax$dynamic_update_slice(cache[[l_]]$k, kh, k_slice_idx)
          cache[[l_]]$v   <- jax$lax$dynamic_update_slice(cache[[l_]]$v, vh, v_slice_idx)
          cache[[l_]]$len <- pm$len

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
          xt   <- jnp$dot(attn_TD, L$Multihead$W_o)                     # [T, D]
          
          # Residual + pre-FFN norm
          xt <- (xtminu1 * jax$nn$softplus(L$ResidCon1$WtSkipPath)) +
            (xt    * jax$nn$softplus(L$ResidCon1$WtResidPath))
          xt <- NormFxn(xtminu1 <- xt) * L$NormScalerPostMultiHead
          
          # SwiGLU FFN
          xt <- jax$nn$swish( ffmap(L$FFN$WideProj1, xt) ) * 
                              ffmap(L$FFN$WideProj2, xt)
          xt <- ffmap(L$FFN$OutProj1, xt)
          
          # Final residual of this layer
          xt <- (xtminu1 * jax$nn$softplus(L$ResidCon2$WtSkipPath)) +
                        (xt    * jax$nn$softplus(L$ResidCon2$WtResidPath))
           
          # Safety: keep padded rows identically zero
          xt <- mask_prefix_rows(xt, pm$mask)  # [T_full, D]
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
        head_dim   <- head_dim
        
        xt <- token_in
        for (l_ in 1:ModelDepth) {
          L <- eval(parse(text = sprintf("TransformerList$d%s", l_)))
          
          # Pre-norm
          xt <- jnp$multiply(
            NormFxn(xtminus1 <- xt),
            jnp$squeeze(L$NormScalerInput, 0L)
          )  # [D]
          
          # Project Q/K/V for this single token
          q_full <- jnp$dot(xt, L$Multihead$W_q)               # [D]
          k_full <- jnp$dot(xt, L$Multihead$W_k)               # [D]
          v_full <- jnp$dot(xt, L$Multihead$W_v)               # [D]
          
          # Reshape to [N, H]
          q_NH <- jnp$reshape(q_full, list(num_heads, head_dim))
          k_NH <- jnp$reshape(k_full, list(num_heads, head_dim))
          v_NH <- jnp$reshape(v_full, list(num_heads, head_dim))
          
          # Apply RoPE with absolute position = pos (vectorized over heads)
          apply_rope <- jax$vmap(function(hvec){rope_apply_single(hvec, pos, head_dim)},
                                 in_axes = 0L)
          q_NH <- apply_rope(q_NH)  # [N, H]
          k_NH <- apply_rope(k_NH)  # [N, H]
          
          # --- Write K/V for this pos into cache (static shapes; dynamic index ok) ---
          max_len  <- cache[[l_]]$k$shape[[1]]                   # static
          pos_i32  <- jnp$astype(pos, jnp$int32)
          pos_i32  <- jnp$clip(pos_i32, jnp$array(0L, jnp$int32), 
                               jnp$array(max_len - 1L, jnp$int32))
          
          k_write_idx <- jnp$array(c(pos_i32, 0L, 0L), dtype = jnp$int32)
          v_write_idx <- jnp$array(c(pos_i32, 0L, 0L), dtype = jnp$int32)
          
          cache[[l_]]$k <- jax$lax$dynamic_update_slice(
            cache[[l_]]$k, jnp$expand_dims(k_NH, 0L), k_write_idx)  # [max_len, N, H]
          cache[[l_]]$v <- jax$lax$dynamic_update_slice(
            cache[[l_]]$v, jnp$expand_dims(v_NH, 0L), v_write_idx)  # [max_len, N, H]
          
          # Logical cache length: at least pos+1
          cache[[l_]]$len <- jnp$maximum(cache[[l_]]$len,
                                         jnp$array(pos_i32 + 1L, dtype = jnp$int32))
          
          # --- Attention: q=[T=1,N,H], K/V=[S=max_len,N,H] (no transposes) ----------
          # Pack the single-token query to [1, N, H]
          q_TNH <- jnp$expand_dims(q_NH, 0L)                     # [1, N, H]
          
          # Use full K/V (static S=max_len); restrict with a keys-only mask.
          K_SNH <- cache[[l_]]$k                                 # [max_len, N, H]
          V_SNH <- cache[[l_]]$v                                 # [max_len, N, H]
          
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
          #mask_keys_decode <- jnp$expand_dims(mask_keys_decode, 0L)  # [1, 1, 1, S]
          mask_keys_decode <- jnp$broadcast_to(mask_keys_decode, list(1L, 1L, max_len))  # [1, 1, 1, S]
          
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
          xt <- (xtminus1 * jax$nn$softplus(L$ResidCon1$WtSkipPath)) +
            (mha_out * jax$nn$softplus(L$ResidCon1$WtResidPath))
          xt <- NormFxn(xtminus1 <- xt) * jnp$squeeze(L$NormScalerPostMultiHead, 0L)
          
          # SwiGLU FFN
          xt <- jax$nn$swish(L$FFN$WideProj1(xt)) * L$FFN$WideProj2(xt)
          xt <- L$FFN$OutProj1(xt)
          
          # Final residual
          xt <- (xtminus1 * jax$nn$softplus(L$ResidCon2$WtSkipPath)) +
                    (xt * jax$nn$softplus(L$ResidCon2$WtResidPath))
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
        # - Define Q/K/V/O weights for Flash attention
        # head_dim must divide ModelDims by TransformerHeads (already ensured elsewhere)
        head_dim <- as.integer(ModelDims / TransformerHeads)
        num_heads = TransformerHeads
        
        init_std <- sqrt(2.0 / as.numeric(ModelDims + ModelDims))
        make_w <- function(shape, seed_key) {
          oryx$Normal(loc = 0., scale = jnp$array(init_std))$
            sample(shape, seed = seed_key)$astype(jaxFloatType)
        }
        
        print("Generating Multihead objects...")
        # Properly split keys for reproducible random initialization
        multihead_keys <- jax$random$split(key, 4L)
        TransformerList[[l_]]$Multihead <- list(
          # [D, D] projections: Q=xt_pos·W_q, K=xt_pos·W_k, V=xt·W_v; O merges heads back
          "W_q" = make_w(list(ModelDims, ModelDims), multihead_keys[1]),
          "W_k" = make_w(list(ModelDims, ModelDims), multihead_keys[2]),
          "W_v" = make_w(list(ModelDims, ModelDims), multihead_keys[3]),
          "W_o" = make_w(list(ModelDims, ModelDims), multihead_keys[4])
        )
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
  print("Done with init path in SuperLModel_BackboneTransformer.R...")
}

if(backbonePath == "run"){ # note: there is no caching here; caching is applied in *BuildML.R
  # multihead attn part
  x_mask_attn <- jnp$matmul(x_mask, jnp$transpose(x_mask)) #length (query_seq_length, kv_seq_length)
  # x_mask_1oversums <- jnp$reciprocal( jnp$add(0.000001, jnp$sum(x_mask,0L,keepdims=T) ) )
  # causalimages::image2(np$array(x_mask_attn$val)[1,,] )
  
  print2(sprintf("Starting Transformer block [depth: %s]...", ModelDepth))
  TransformerStep_NoCache <- switch_filter_jit(function(xt, TransformerList_d, x_mask_attn){
        
        # standardize
        xt <- jnp$multiply(NormFxn( xtm1 <- xt  ), TransformerList_d$NormScalerInput)
        
        # multihead attention block
        if(UseLatentAttention){
          stop("Latent Attention not double checked -- do not use")
          xt <- LatentMultiheadAttention(
            Multihead = TransformerList_d$LatentMultihead,
            query = (xt_pos <- eq$nn$RotaryPositionalEmbedding(ModelDims)( xt )),
            key_  = xt_pos,
            value = xt, 
            mask = x_mask_attn )
        }
        if(!UseLatentAttention){
          {
          # 1) Linear projections: [T, D] @ [D, D] -> [T, D]
          q_ = jnp$dot(xt, TransformerList_d$Multihead$W_q)
          k_ = jnp$dot(xt, TransformerList_d$Multihead$W_k)
          v_ = jnp$dot(xt, TransformerList_d$Multihead$W_v)
          
          # 2) Reshape to heads: [T, D] -> [T, N, H]
          q_ = jnp$reshape(q_, list(q_$shape[[1]], num_heads, head_dim))
          k_ = jnp$reshape(k_, list(k_$shape[[1]], num_heads, head_dim))
          v_ = jnp$reshape(v_, list(v_$shape[[1]], num_heads, head_dim))
          
          # 3) Apply RoPE after Q/K projections so cached and uncached paths match.
          pos_ids <- jnp$arange(q_$shape[[1]], dtype = jnp$int32)
          apply_rope_one_row <- function(NH_row, p) {
            jax$vmap(function(hvec){ rope_apply_single(hvec, p, head_dim) }, in_axes = 0L)(NH_row)
          }
          apply_rope_all <- jax$vmap(apply_rope_one_row, in_axes = list(0L, 0L))
          q_ <- apply_rope_all(q_, pos_ids)
          k_ <- apply_rope_all(k_, pos_ids)
          
          # 4) Mask: make boolean and broadcast to [N, T, S].
          #     Your x_mask_attn is [T, S] with 1=keep, 0=mask. Convert to bool and add head axis.
          #mask_bool <- jnp$greater(x_mask_attn, 0)
          #mask_bool <- jnp$expand_dims(mask_bool, 0L)                            # [1, T, S]
          #mask_bool <- jnp$broadcast_to(mask_bool, list(num_heads, q$shape[[1]], k$shape[[1]]))  # [N, T, S]
          
          mask_bool  <- jnp$greater(x_mask_attn, 0)                              # [T, S]
          #mask_bool  <- jnp$expand_dims(jnp$expand_dims(mask_bool, 0L), 0L)      # [1, 1, T, S]
          #mask_bool  <- jnp$expand_dims(mask_bool, 0L)      # [1, 1, T, S]
          mask_bool <- jnp$broadcast_to(mask_bool,
                                        list(num_heads, mask_bool$shape[[1]], mask_bool$shape[[2]])
                                        )  # [N, T, S]

          # attention 
          xt_attn <- dot_product_attention_unified(
              q = q_,
              k = k_,
              v = v_,
              mask = mask_bool,
              is_causal = (ModelType == "DecoderOnly"),
              prefer = "auto"
            )$astype(k_$dtype)
          
          # 6) Merge heads back: [T, N, H] -> [T, D]
          xt_attn <- jnp$reshape(xt_attn, list(xt_attn$shape[[1]], num_heads*head_dim))
          
          # 7) Output projection: [T, D] @ [D, D] -> [T, D]
          xt <- jnp$dot(xt_attn, TransformerList_d$Multihead$W_o)
          }
        }
        
        # residual + normalization, see https://arxiv.org/pdf/2302.00453.pdf Eq 2
        #xt <- xtm1 <- xtm1 + xt * jax$nn$softplus( TransformerList_d[[5]][[1]] ) 
        xt <- xtm1 <- xtm1 * jax$nn$softplus( TransformerList_d$ResidCon1$WtSkipPath ) + 
                      xt   * jax$nn$softplus( TransformerList_d$ResidCon1$WtResidPath )
        xt <- NormFxn( xt ) * TransformerList_d$NormScalerPostMultiHead
        
        # feed forward with swiglu FFN
        xt <- jax$nn$swish(ffmap(TransformerList_d$FFN$WideProj1, xt)) * 
                           ffmap(TransformerList_d$FFN$WideProj2, xt) # swiglu proj
        xt <- ffmap(  TransformerList_d$FFN$OutProj1, xt  ) # final projection

        # return with residual connection to previous state
        xt <- xtm1 <- xtm1 * jax$nn$softplus( TransformerList_d$ResidCon2$WtSkipPath ) +
                      xt   * jax$nn$softplus( TransformerList_d$ResidCon2$WtResidPath )
        return( xt )
      })
  
  # manual loop 
  if(TRUE){
    for(d_ in 1:ModelDepth){ print(d_); xt <- TransformerStep_NoCache( xt, eval(parse(text=sprintf("TSList$TSBackbone$d%s",d_))), x_mask_attn )  }
  }
  
  # loop with scan
  if(FALSE){
    layer_branches <- lapply(seq_len(ModelDepth), function(d_) {
      L_d <- TSList$TSBackbone[[paste0("d", d_)]]
      function(xt_in) TransformerStep_NoCache(xt_in, L_d, x_mask_attn)
    })
    
    scan_body <- function(carry_xt, i) {
      xt_next <- jax$lax$switch(index = i, branches = layer_branches, operand = carry_xt)
      list(xt_next, jnp$array(0L)) # dummy per-step output
    }
    
    xt <- jax$lax$scan(
      f    = scan_body,
      init = xt,
      xs   = jnp$arange(start = 0L, stop = as.integer(ModelDepth), dtype = jnp$int32)
    )[[1]]
  }
  
  # plot(np$array(xt$val$val)[1,1,,d_<-sample(1:50,1)]) # sample a dimension
  # plot(np$array(xt$val$val)[sample(1:20,1),1,,d_]) # look at its evolution across different matrix slices
  print2(sprintf("Done with Transformer block [depth: %s]...", ModelDepth))
}
print("Done sourcing SuperLModel_BackboneTransformer.R")
