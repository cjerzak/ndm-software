# Content of SuperLModel_BuildML.R
{
print("Starting SuperLModel_BuildML.R")
DisappList <- function(.){ret_<-.;if(is.list(.)){if(length(unlist(.))==0){ret_ <- NULL}};return(ret_)}

# Special tokens config 
doGeoInfo <- FALSE
AppendPlaceEmbeds <- isTRUE(get0("AppendPlaceEmbeds", ifnotfound = TRUE))
AppendTimeEmbeds <- isTRUE(get0("AppendTimeEmbeds", ifnotfound = TRUE))

# Latent Attention config
DecodingInNeuralODE <- FALSE
UseLatentAttention <- FALSE  # Set to TRUE to enable Multi-Head Latent Attention; this is not supported yet
LatentDim <- as.integer(ModelDims / 4)  # Latent dimension for compression (1/4 of ModelDims by default)

# setup model
{
  ODE_NORM_TYPE <- "pre-activation1" # "post-activation", "none" 

  ## structural parameters
  nTimeStepsInHistory <- batch_l$XPred$shape[[2]][[1]]
  TSInputSize <- batch_l$XPred$shape[[3]][[1]]
  batch_axis_name <- "batch" # -> must name if using batchnorm
  InitTransform_CLS <- InitTransform_EMBED <- function(x){  sqrt(2/x)  }
  #InitTransform_CLS <- InitTransform_EMBED <- function(x){  1/sqrt(2)  }

  # model architecture
  #ModelDims <- ai(2^7); ModelDepth <- 2L # for testing only 
  ModelDims <- ai(ModelDims); ModelDepth <- as.integer(ModelDepth) # now defined in SuperLModel_BuildML.R
  WideMultiplicationFactor <- 3.75
  ln_bias <- F; useBias_odeLN <- F; finalBias_ODE <- F
  nExperts <- 1L

  resolve_neuralode_latent_dim <- function(config_name, legacy_name, fallback_value) {
    lookup_env <- parent.env(environment())
    dim_value <- get0(
      config_name,
      envir = lookup_env,
      inherits = FALSE,
      ifnotfound = NULL
    )
    if (is.null(dim_value)) {
      dim_value <- get0(
        legacy_name,
        envir = lookup_env,
        inherits = FALSE,
        ifnotfound = fallback_value
      )
    }
    dim_value <- suppressWarnings(as.integer(dim_value))
    if (length(dim_value) != 1L || is.na(dim_value) || dim_value < 1L) {
      stop(sprintf("`%s` must resolve to a positive integer.", config_name), call. = FALSE)
    }
    ai(dim_value)
  }

  #encneuralType <- "l"; LocalNeuralEmbedDim <- 24L;  nWidthODEHidden_ts_local <- ai(LocalNeuralEmbedDim*1); 
  #encneuralType <- "g"; LocalNeuralEmbedDim <-  64L;  nWidthODEHidden_ts_local <- ai(LocalNeuralEmbedDim*WideMultiplicationFactor);
  encneuralType <- "g"; LocalNeuralEmbedDim <- resolve_neuralode_latent_dim("neuralode_local_latent_dim", "LocalNeuralEmbedDim", ModelDims);  nWidthODEHidden_ts_local <- ai(LocalNeuralEmbedDim*WideMultiplicationFactor);
  GlobalNeuralEmbedDim <- resolve_neuralode_latent_dim("neuralode_global_latent_dim", "GlobalNeuralEmbedDim", ModelDims); nWidthODEHidden_ts_global <- ai(GlobalNeuralEmbedDim*WideMultiplicationFactor)
  nDepth_nODE <- 1L;

  # hyperparameters
  bn_momentum <- 0.90
  ep_LN <- ep_BN <- 1e-4
  widthCycle <- 50;
  doGeoInfo <- F
  BackboneType <- "transformer"; 
  resolve_transformer_topology <- function(model_dims,
                                           target_head_dim = 64L,
                                           kv_heads_override = NULL) {
    target_head_dim <- max(1L, ai(target_head_dim))
    divisors <- (1:model_dims)[which(model_dims %% (1:model_dims) == 0L)]
    head_dims <- model_dims %/% divisors
    even_mask <- (head_dims %% 2L) == 0L
    if (!any(even_mask)) {
      stop("ModelDims must admit an even attention head dimension for RoPE.", call. = FALSE)
    }
    divisors <- divisors[even_mask]
    head_dims <- head_dims[even_mask]
    best_q_idx <- order(abs(head_dims - target_head_dim), divisors)[1]
    num_query_heads <- ai(divisors[[best_q_idx]])
    head_dim <- ai(head_dims[[best_q_idx]])

    if (is.null(kv_heads_override)) {
      max_kv_heads <- max(1L, ai(floor(num_query_heads / 4L)))
      kv_candidates <- (1:num_query_heads)[which(num_query_heads %% (1:num_query_heads) == 0L)]
      kv_candidates <- kv_candidates[kv_candidates <= max_kv_heads]
      if (length(kv_candidates) == 0L) {
        kv_candidates <- 1L
      }
      num_kv_heads <- ai(max(kv_candidates))
    } else {
      num_kv_heads <- ai(kv_heads_override)
      if (num_kv_heads < 1L || (num_query_heads %% num_kv_heads) != 0L) {
        stop("AttentionKVHeads must be a positive divisor of the resolved query head count.", call. = FALSE)
      }
    }

    list(
      "num_query_heads" = num_query_heads,
      "num_kv_heads" = num_kv_heads,
      "head_dim" = head_dim,
      "kv_group_size" = ai(num_query_heads %/% num_kv_heads)
    )
  }
  AttentionHeadDim <- max(1L, ai(get0("AttentionHeadDim", ifnotfound = 64L)))
  AttentionKVHeads <- get0("AttentionKVHeads", ifnotfound = NULL)
  if (!is.null(AttentionKVHeads)) {
    AttentionKVHeads <- ai(AttentionKVHeads)
  }
  TransformerTopology <- resolve_transformer_topology(
    model_dims = ModelDims,
    target_head_dim = AttentionHeadDim,
    kv_heads_override = AttentionKVHeads
  )
  TransformerHeads <- TransformerTopology$num_query_heads
  TransformerKVHeads <- TransformerTopology$num_kv_heads
  TransformerHeadDim <- TransformerTopology$head_dim
  TransformerKVGroupSize <- TransformerTopology$kv_group_size

  #BackboneType <- "mamba"; StateSize <- 4L
  DoPriorSamplingAutoDiff <- F
  UseDiagonalLMatVCov <- isTRUE(get0("UseDiagonalLMatVCov", ifnotfound = TRUE))

  # define model architecture via input
  Constrained2Unconstrained <- function(target_mean_of_transformated_x, transformation, sd){
    ex_seq <- seq(-100,100,length.out = 10000)
    fx_jax <- jax$jit(function(center){
      x_draw <- oryx$Normal(loc = center, scale = sd)$sample(list(10000L),seed=JaxKey(jnp$array(ai(runif(1,0,100000)))))
      jnp$mean(transformation(x_draw))})
    fx_jax_v <- jax$vmap(function(cs){ fx_jax(cs) }, in_axes = list(0L))
    efx <- np$asanyarray( fx_jax_v(jnp$array(as.matrix(ex_seq))) )
    best_ex <- ex_seq[ which.min(abs(efx - target_mean_of_transformated_x)) ]
    return( list("best_ex" = best_ex,  "sd"= sd))
  }

  # RMS norm
  RMSNorm <- function(x_){ jnp$divide( x_, jnp$sqrt(1e-5+jnp$mean(jnp$square(x_), -1L, keepdims=T))) }
  LayerNorm <- function(x_){ jax$nn$standardize( x_, -1L, epsilon = 1e-5) }
  #NormFxn <- LayerNorm
  NormFxn <- RMSNorm

  print2("Entering SuperLModel_ParseDynamicODE.R")
  ndm_source_extracted("ModelDefiners/SuperLModel_ParseDynamicODE.R")

  # define all parameter arrays
  DenseLayerInputSize <- ai(1L*ModelDims)

  print2("Setting up ts part...")
  {
      # positional encoding scaler
      InitProcessList <- list()
      InitProcessList$PositionalEncoderScaler <- jnp$array( InvSoftPlus(.1) )

      # initial encoding layer(s)
      InitialTransformType <- "CNN"
      #InitialTransformType <- "Linear"
      if(InitialTransformType == "Linear"){
        InitProcessList$InitialEncodingTransform <- eq$nn$Linear(in_features = TSInputSize, out_features = ModelDims,
                                                          key = jax$random$PRNGKey(ai(SEED_*658L)), use_bias = F)
      }
      if(InitialTransformType == "CNN" & FALSE){
        InitProcessList$InitialEncodingTransform <- list(
                                "Conv1d_short"=eq$nn$Conv1d(in_channels = TSInputSize, out_channels = ModelDims,
                                      padding = "SAME",  padding_mode = "REPLICATE", 
                                      kernel_size = 1L, key = jax$random$PRNGKey(ai(SEED_*658L)), use_bias = F),
                                "Conv1d_mid"=eq$nn$Conv1d(in_channels = TSInputSize, out_channels = ModelDims,
                                      padding = "SAME",  padding_mode = "REPLICATE", 
                                      kernel_size = 3L, key = jax$random$PRNGKey(ai(SEED_*615228L)), use_bias = F),
                                "Conv1d_long"=eq$nn$Conv1d(in_channels = TSInputSize, out_channels = ModelDims,
                                      padding = "SAME",  padding_mode = "REPLICATE", 
                                      kernel_size = 7L, key = jax$random$PRNGKey(ai(SEED_*6158L)), use_bias = F),
                                "EncodingCombine"=eq$nn$Linear(in_features = ModelDims*3L, out_features = ModelDims,
                                      key = jax$random$PRNGKey(ai(SEED_*411L)), use_bias = F)
                                )
      }
      
      if(InitialTransformType == "CNN" & TRUE){
        print2("Setting up InitialTransformType == 'CNN' branch...")
        manual_conv1d <- function(x, kernel, kernel_size) {
          # x: either (batch, length, in_channels) or (length, in_channels)
          # kernel: (kernel_size, in_channels, out_channels)
          
          is_unbatched <- (x$ndim == 2L)
          
          if (is_unbatched) {
            x <- jnp$expand_dims(x, 0L)  # Add batch dimension
          }
          
          # Now x is (batch, length, in_channels)
          batch_size <- x$shape[[1]]
          seq_len <- x$shape[[2]]
          in_channels <- x$shape[[3]]
          out_channels <- kernel$shape[[3]]
          
          pad_size <- kernel_size %/% 2L
          
          # Pad sequence dimension with edge mode
          x_padded <- jnp$pad(
            x, 
            list(list(0L, 0L), list(pad_size, pad_size), list(0L, 0L)), 
            mode = 'edge'
          )
          
          # Extract patches using dynamic_slice (avoids indexing issues)
          patch_list <- list()
          for (i in 0L:(kernel_size - 1L)) {
            patch <- jax$lax$dynamic_slice(
              x_padded,
              start_indices = c(0L, i, 0L),
              slice_sizes = c(batch_size, seq_len, in_channels)
            )
            patch_list[[length(patch_list) + 1]] <- patch
          }
          
          # Concatenate along channel dimension: (batch, seq_len, kernel_size * in_channels)
          patches <- jnp$concatenate(patch_list, axis = 2L)
          
          # Flatten kernel: (kernel_size * in_channels, out_channels)
          kernel_flat <- jnp$reshape(kernel, list(-1L, out_channels))
          
          # GEMM: (batch, seq_len, out_channels)
          out <- jnp$matmul(patches, kernel_flat)
          
          # Remove batch dimension if input was unbatched
          if (is_unbatched) {
            out <- jnp$squeeze(out, 0L)
          }
          
          return(out)
        }

        # Initialize kernels (orthogonal init for stability, as in Equinox defaults)
        key_short <- jax$random$PRNGKey(ai(SEED_*6581L))
        key_mid <- jax$random$PRNGKey(ai(SEED_*6152228L))
        key_long <- jax$random$PRNGKey(ai(SEED_*61584L))
        init_orthogonal_kernel <- function(key, kernel_size, in_channels, out_channels) {
          # Create orthogonal matrix for each position in kernel
          kernels <- list()
          for (k in 1:kernel_size) {
            subkey <- jax$random$fold_in(key, ai(k-1L))
            # Initialize (in_channels, out_channels) orthogonally
            if (in_channels <= out_channels) {
              #mat <- jax$random$orthogonal(subkey, n = out_channels, shape = list())[0:in_channels, ]
              mat <- jax$random$orthogonal(subkey, n = out_channels)[0:in_channels, ]
            } else {
              #mat <- jnp$transpose(jax$random$orthogonal(subkey, n = in_channels, shape = list()))[, 1:out_channels]
              mat <- jnp$transpose(jax$random$orthogonal(subkey, n = in_channels))[, 1:out_channels]
            }
            kernels[[k]] <- mat
          }
          jnp$stack(kernels, axis = 0L)
        }
        kernel_short <- init_orthogonal_kernel(key_short, 1L, TSInputSize, ModelDims)
        kernel_mid <- init_orthogonal_kernel(key_mid, 3L, TSInputSize, ModelDims)
        kernel_long <- init_orthogonal_kernel(key_long, 7L, TSInputSize, ModelDims)
        #  manual_conv1d(xt, kernel_short, 1L)
        #  manual_conv1d(xt, kernel_mid, 3L)
        
        # Store as list (no bias, as original)
        InitProcessList$InitialEncodingTransform <- list(
          "Conv1d_short" = list("kernel" = kernel_short),
          "Conv1d_mid" = list("kernel" = kernel_mid),
          "Conv1d_long" = list("kernel" = kernel_long),
          "EncodingCombine" = eq$nn$Linear(in_features = ModelDims*3L, out_features = ModelDims,
                                           key = jax$random$PRNGKey(ai(SEED_*411L)), use_bias = F)
        )
      }
      
      # place embedding
      { 
        print2("Setting up place embeddings...")
        InitProcessList$PlaceEmbedsFixed <- FALSE
        InitProcessList$PlaceEmbeds <- jnp$array(matrix(rnorm(ModelDims*nPlaces, 
                                                             sd = InitTransform_EMBED(ModelDims)), 
                                                       ncol = ModelDims, nrow = nPlaces)) # place embedding  
        if("coordinates_mat" %in% ls()){ 
          # Define sinusoidal positional encoding function for coordinates
          geo_positional_encoding <- function(longitudes, latitudes, ModelDims) {
            nPlaces <- length(longitudes)
            dim_half <- ModelDims / 2
            
            long_frequencies <- 1 / (10000^(seq(0, dim_half - 1) / dim_half))
            lat_frequencies <- 1 / (10000^(seq(0, dim_half - 1) / dim_half))
            
            embeddings <- matrix(0, nrow = nPlaces, ncol = ModelDims)
            
            for (i in 1:nPlaces) {
              embeddings[i, seq(1, ModelDims, by = 2)] <- sin(longitudes[i] * long_frequencies)
              embeddings[i, seq(2, ModelDims, by = 2)] <- cos(latitudes[i] * lat_frequencies)
            }
            
            return(jnp$array(embeddings))
          }
          
          # Initialize place embeddings
          InitProcessList$PlaceEmbeds <- geo_positional_encoding(longitudes = coordinates_mat$long, 
                                                                 latitudes = coordinates_mat$lat, 
                                                                 ModelDims)
          InitProcessList$PlaceEmbedsFixed <- TRUE
        }
      }
      
      # time embedding 
      print2("Setting up time embeddings...")
      MaxTimeIndex <- ai(max(c(0L, f2n(get0("MaxTimeIndex", ifnotfound = nTimesTotal - 1L))), na.rm = TRUE))
      InitProcessList$TimeEmbeds <- jnp$array( # Initialize TimeEmbeds with sinusoidal positional encoding
                                  do.call(rbind, lapply(0:MaxTimeIndex, function(pos) {
                                    frequencies <- 1 / (10000^(seq(0, ModelDims - 1L, by = 2L) / ModelDims))
                                    pos_vector <- numeric(ModelDims)
                                    pos_vector[seq(1, ModelDims, by = 2L)] <- sin(pos * frequencies)
                                    pos_vector[seq(2, ModelDims, by = 2L)] <- cos(pos * frequencies)
                                    return(pos_vector)
                                  }))
                                )
      InitProcessList$TimeEmbeds_Proj <- eq$nn$Linear(in_features = ModelDims,
                                                      out_features = ModelDims, use_bias = FALSE,
                                                      key = jax$random$PRNGKey(ai(SEED_*123123)))
      InitProcessList$PlaceEmbeds_Proj <- eq$nn$Linear(in_features = ModelDims,
                                                       out_features = ModelDims, use_bias = FALSE,
                                                       key = jax$random$PRNGKey(ai(SEED_*23411)))
      # hist(np$array( InitProcessList$TimeEmbeds ))
      # hist(np$array( InitProcessList$PlaceEmbeds ))

      print("Setting up some initial neural variational parameters...")
      nDimOutput_neuralODE <-  999L
      {
        NeuralVariationalInitEncLocalMean_red <- NeuralVariationalInitEncLocalMean[names(NeuralVariationalInitEncLocalMean) != "sd_l"]
        nODEParams_base <- nODEParams <- length( NeuralVariationalInitEncLocalMean_red ) -
                     sum(is.na(NeuralVariationalInitEncLocalMean_red))   # the context terms
        nParams_DiagMat_neural <- nODEParams_neural <- 0;
        { # adjust param count
          nODEParams_base <- nODEParams - (nODEParams_neural <- sum(names((NeuralVariationalInitEncLocalMean_red)) %in% 
                                                                      c(uq_encneural_vec,c("Neural1") )))
          nParams_DiagMat_neural <- nODEParams_neural # params for diagonal mat - clarify 
        }
        if(isTRUE(UseDiagonalLMatVCov)){
          nParams_LMat_base <- nODEParams_base
        }
        if(!isTRUE(UseDiagonalLMatVCov)){
          nParams_LMat_base <- ifelse(nODEParams_base > 0L,
                                      yes = sum(seq_len(nODEParams_base)),
                                      no = 0L)
        }
        nDimOutput_dense <- ai(nODEParams_total <- (nODEParams_base +
                                                      nODEParams_neural +
                                                      nParams_LMat_base +
                                                      nParams_DiagMat_neural # params for diagonal terms in vcov
                       )) + nOutcomes  # add nOutcomes for sd_l components
        if(ModelType == "DecoderOnly"){ nDimOutput_dense <- ModelDims }

        # fixed local terms - unsure if will work with > 1 term
        print("Setting up fixed local terms...")
        nPlaces <- ifelse(SimMode, yes = 1L, no = length(unique(  truth_df_red$location_id  ) ))
        nFixedLocal <- length( VariationalInitNonEncLocalMean )
        FixedLocalTerms <- jnp$array(0.); if(nFixedLocal > 0){
          FixedLocalMeans <- jnp$array(   ifelse(length(VariationalInitNonEncLocalMean)==1,yes=as.matrix,no=t)(
            replicate(nPlaces, VariationalInitNonEncLocalMean) ))
          FixedLocalInvSD <- jnp$array(  
            ifelse(length(VariationalInitNonEncLocalSD)==1,yes=as.matrix,no=t)(
              replicate(nPlaces, np$asanyarray((InvSoftPlus_r(VariationalInitNonEncLocalSD)) ))  
            ) )
          FixedLocalTerms <- c(FixedLocalMeans, FixedLocalInvSD)
        }

        # global terms
        print("Setting up global terms...")
        nGlobalParams <- 0L
        GlobalMeans <- GlobalLTerms <- jnp$array(0.)
        if(length(NeuralVariationalInitGlobalMean)>0){
          nGlobalParams <- length( GlobalMeans <- NeuralVariationalInitGlobalMean )
          nDimOutput_global <- ai(nGlobalParams + sum(nGlobalParams:1)) # fancy sum does covariances
          GlobalLTerms <- mean(abs(GlobalMeans))*runif(sum(nGlobalParams:1),-0.1,0.1)
        }
        GlobalTerms <- jnp$array(c(GlobalMeans, GlobalLTerms))
      }
  }

  Ik <- function(key){ return( key <- key + 1L) }
  SampleConst <- function(const=1., shape, key){ oryx$Normal(loc = const, scale =  0.000001)$sample(shape, seed = key) }
  #TSList <- eq$filter_vmap(function(key){
  print("Setting up rest of TSList...")
  TSList <- {key <- jax$random$PRNGKey(ai(123L*SEED_))
  
    if("transformer" %in% BackboneType){
      print("Initializing transformer objects")
      # setup transformers list
      TransformerList <- replicate(list(list()), n = ( (ModelDepth)))
      
      # transformer seq (1 is initial processing)
      backbonePath <- "initialize"; ndm_source_extracted("ModelDefiners/SuperLModel_BackboneTransformer.R")
      print("Done sourcing SuperLModel_BackboneTransformer.R in initialize path")
    }

    if("mamba" %in% BackboneType){
      print("Initializing mamba objects")
      stop("Mamba is not included in ndm.", call. = FALSE)
      print("Done sourcing SuperLModel_BackboneMamba.R in initialize path")
    }
    
    RNNList <- oryx$Normal(loc = 1, scale = 0.001)$sample(list(), seed = key+40L)
    
    print("Defining TSList object...")
    TSList <- c("TSBackbone"=list(TransformerList), # Transformer list
                "InitialCLS" = oryx$Normal(loc = 0., scale =  InitTransform_CLS(ModelDims))$sample( list(1L,ModelDims), seed = key*234L)$astype(jaxFloatType),  # initial hidden state for agg token
                "FinalNormScaler" = jnp$array( oryx$Normal( loc = 1., scale =  0.0001)$sample( list(1L,ModelDims), seed = key*2334L) ),  # final RMS norm re-weightor
                "OutputProcess" = list(list("Proj1"=eq$nn$Linear( in_features = ModelDims, 
                                                                  out_features = nDimOutput_dense,
                                        use_bias = ifelse(ModelType == "DecoderOnly",
                                                          yes = TRUE,
                                                          no = FALSE),# if neuralODE, then using manual bias 
                                        key = key*34000L), # final projection 
                       "ManualBias"=oryx$Normal(
                         loc = c(NeuralVariationalInitEncLocalMean, 
                                 rep(0,times = max(c(0,nDimOutput_dense - length(NeuralVariationalInitEncLocalMean))))),
                            scale =  0.000001)$sample( list(1L), seed = key*234L*SEED_ )$astype(jaxFloatType)$flatten(),  # final projection bias 
                       "Proj2" = eq$nn$Linear( in_features = ModelDims, out_features = ModelDims,
                                     use_bias = F, key = key*340L*SEED_) # extra projection for LN if used
                       )
                ))
  #return(TSList) }, in_axes = 0L)(
              #jax$random$split( jax$random$PRNGKey(1235L),
              #        ifelse(nExperts==1,yes=1L,no=ai(nExperts+1L)) ))
  }
  print("Past setting up rest of TSList...")
  }
  
    # scale list
    print2("Setting up scale list...")
    ScaleList <- list(
                 "ScaleBayes" = {list(
                            "VarInit" = InvSoftPlus( jnp$array(rep(1,times=nOutcomes) )),  # Var Init Scale -> this is later calibrated
                            
                            "DirichletScale1"=InvSoftPlus( jnp$array(c(1, 1, 1, 1) ) ),  # dirichletparams 
                            "DirichletScale2"=InvSoftPlus( jnp$array(rep(1, times=4)) ),  # dirichletparams 
                            
                            "LmatLocalDiagScaler"=InvSoftPlusLargeInputApprox(LMat_DiagScale <- 100), # Lmat diag scaler
                            "LmatGlobalScaler"=InvSoftPlus(1/LMat_DiagScale * 0.0001), # Lmat global scaler (if too small, becomes -inf)
                            "globalLmatDiagScaler"=InvSoftPlusLargeInputApprox(LMat_DiagScale_g <- 100), # globalLmat diag scaler
                            "globalLmatGlobalScaler"=InvSoftPlus(1/LMat_DiagScale_g * 0.0001) # globalLmat global scaler (if too small, becomes -inf)
                  )},
                "nODE_scale"= c(jnp$array(1.), 
                                DisappList(replicate(nDepth_nODE-1L, jnp$array(t(rep(1,times = nWidthODEHidden_ts_local))))), # nODE weight norms
                                jnp$array(1.)),

                "nODE_out_scale" = c(jnp$array(rep(1,times = nWidthODEHidden_ts_local)),jnp$array(rep(0,times = nWidthODEHidden_ts_local))),# layer norm scaler

                "GlobalNorm"=c( InvSoftPlus( jnp$array(rep( .1 * 1/sqrt(1*nTimesTotal ), times = outputDim_nODE_local ))) ) # global norm
                )

  print2("Set up model lists...")
  {
      ListIndices <- c( "InitProcessList", "TSList", "BNList",
                       "ScaleList", "FixedLocalTerms", "GlobalTerms",
                       "GlobalNeural", "LocalNeural" ) 
      # note: GlobalNeural defined in ParseDynamicODE.R
      names(ListIndices) <- ListIndices; ListIndices[] <- 1:length( ListIndices )
      ListIndices <- sapply(ListIndices,f2n)
      state <- jnp$array(0.)
      ModelList <- list( "InitProcessList"=InitProcessList,  
                         "TSList"=TSList, 
                         "BNList"=jnp$array(0.), 
                         "ScaleList"=ScaleList, 
                         "FixedLocalTerms"=FixedLocalTerms, 
                         "GlobalTerms"=GlobalTerms,
                         "GlobalNeural"=GlobalNeural, 
                         "LocalNeural"=LocalNeural)
      # head(names(ModelList))
      
      print2("Obtaining approximate parameter count...")
      nParamsModel <- sum(unlist(lapply(jax$tree$leaves(eq$partition(ModelList, eq$is_array)[[1]]), function(zer){zer$size})))
      rm( list = names(ListIndices) )
  }

  print2("Define functional components of model...")
  {
  # feed forward map
  ffmap <- jax$vmap(function(L_,x){ L_(x) }, in_axes = list(NULL,0L))

  ProcessEncoderInput <- function(InitProcessList, TSList, 
                                  xt, time, 
                                  place, BNList, 
                                  state, inference){
    x_mask <- xt[[2]]; xt <- xt[[1]]
    
    # initial projection to hiddenDim dimensionality - either linear or MLP
    if(InitialTransformType=="Linear"){ 
      xt <-  ffmap( InitProcessList$InitialEncodingTransform,  xt ) 
    }
    
    if(InitialTransformType=="CNN"& FALSE){ 
      # plot(np$array(xt$val)[1,,5])
      # points(np$array(x_mask$val)[1,,])
      xt <-  ffmap(InitProcessList$InitialEncodingTransform$EncodingCombine,
                   jnp$concatenate(list(
                     jnp$transpose(InitProcessList$InitialEncodingTransform$Conv1d_short( jnp$transpose(xt) )), 
                     jnp$transpose(InitProcessList$InitialEncodingTransform$Conv1d_mid(   jnp$transpose(xt) )),
                     jnp$transpose(InitProcessList$InitialEncodingTransform$Conv1d_long(  jnp$transpose(xt) ))
                   ), 1L))
      # plot(np$array(xt$val)[1,,5])
      # points(np$array(x_mask$val)[1,,])
    }
    if(InitialTransformType=="CNN"& TRUE){ 
      # Apply manual convs directly 
      xt_short <- manual_conv1d(xt, InitProcessList$InitialEncodingTransform$Conv1d_short$kernel, 1L)
      xt_mid <- manual_conv1d(xt, InitProcessList$InitialEncodingTransform$Conv1d_mid$kernel, 3L)
      xt_long <- manual_conv1d(xt, InitProcessList$InitialEncodingTransform$Conv1d_long$kernel, 7L)
      
      xt <- ffmap(InitProcessList$InitialEncodingTransform$EncodingCombine,
                  jnp$concatenate(list(xt_short, xt_mid, xt_long), 1L))  # Concat on channel dim
    }
    
    # jnp$mean(xt$val, 0L:1L); # confirm near 0 
    # jnp$std(xt$val, 0L:1L) # confirm near 1
    # jnp$abs(InitProcessList$PlaceEmbeds)$mean()
    # jnp$abs(InitProcessList$PlaceEmbeds)$mean()
    # jnp$abs(InitProcessList$TimeEmbeds )$mean()
    place_embed <- jnp$take(
      InitProcessList$PlaceEmbeds,
      indices = place,
      axis = 0L
    )
    if (isTRUE(InitProcessList$PlaceEmbedsFixed)) {
      place_embed <- jax$lax$stop_gradient(place_embed)
    }
    place_embed <- InitProcessList$PlaceEmbeds_Proj(place_embed)

    time_embed <- jax$lax$stop_gradient(jnp$take(
      InitProcessList$TimeEmbeds,
      indices = time,
      axis = 0L
    ))
    time_embed <- InitProcessList$TimeEmbeds_Proj(time_embed)
    if(endAppend){ 
      if(AppendPlaceEmbeds){
        x_mask <- jnp$concatenate(list(x_mask, jnp$ones(list(1L,1L))), 0L)
        xt <- jnp$concatenate(list(xt, jnp$expand_dims(place_embed, 0L)), axis = 0L)
      }
      if(AppendTimeEmbeds){
        x_mask <- jnp$concatenate(list(x_mask, jnp$ones(list(1L,1L))), 0L)
        xt <- jnp$concatenate(list(xt, jnp$expand_dims(time_embed, 0L)), axis = 0L)
      }
      
      # append CLS representation to LAST position 
      x_mask <- jnp$concatenate(list(x_mask, jnp$ones(list(1L,1L))), 0L)
      xt <- jnp$concatenate(list(xt, TSList$InitialCLS), axis = 0L)
    }
    if(!endAppend){ 
      if(AppendPlaceEmbeds){
        x_mask <- jnp$concatenate(list(jnp$ones(list(1L,1L)), x_mask), 0L)
        xt <- jnp$concatenate(list(jnp$expand_dims(place_embed, 0L), xt ), axis = 0L)
      }
      if(AppendTimeEmbeds){
        x_mask <- jnp$concatenate(list(jnp$ones(list(1L,1L)), x_mask), 0L)
        xt <- jnp$concatenate(list(jnp$expand_dims(time_embed, 0L), xt ), axis = 0L)
      }
      
      # append CLS representation to FIRST position
      x_mask <- jnp$concatenate(list(jnp$ones(list(1L,1L)), x_mask), 0L)
      xt <- jnp$concatenate(list(TSList$InitialCLS, xt), axis = 0L)
    }
    # compare CLS with place embeds 
    # plot(c(np$array(jnp$take(InitProcessList$PlaceEmbeds, indices = place, axis = 0L)$val)[1,]), c(np$array(TSList[[3]]$val)))
    
    # check correlation of CLS with initialization 
    # plot( np$array(TSList[[3]]$val)[1,1,], np$array(jnp$take(InitProcessList$PlaceEmbeds,  indices = place, axis = 0L)$val)[1,])
    
    # plot(np$array(xt$val)[1,1,d_<-sample(1:50,1)]) # sample a dimension
    # plot(np$array(xt$val)[sample(1:20,1),1,,d_]) # look at its evolution across different matrix slices
    
    # analyze moments of initial inputs - across units 
    # plot(apply(np$array(xt$val),c(2),mean) )
    # plot(apply(np$array(xt$val),c(2),sd) )
    
    # plot(apply(np$array(xt$val),c(2),sd) )
    # np$array(xt$val)[1,,1:6] -  np$array(xt$val)[2,,1:6]
    # plot(np$array(xt$val)[sample(1:32,1),,1],ylim = c(-1,1))
    return( list(xt, x_mask) )
  }  
    
  # define dense + ts functions
  DecoderBackboneToOutput <- function(TSList, hidden_state){
    hidden_state <- jnp$squeeze(
      LayerNorm(jnp$expand_dims(hidden_state, 0L)) * TSList$FinalNormScaler
    )
    hidden_state <- TSList$OutputProcess$Proj1(hidden_state)
    if(ModelType != "DecoderOnly"){
      hidden_state <- hidden_state + TSList$OutputProcess$ManualBias
    }
    hidden_state
  }

  SelectBackboneOutputToken <- function(xt, x_mask){
    if( ModelType == "NeuralODE"){
      if( T == F ){ 
        plot(np$array(x_mask$val)[,4,1])
      }
      if(endAppend){  xt <- jnp$take(xt, indices = xt$aval$shape[[1]]-1L, axis = 0L) }
      if(!endAppend){ xt <- jnp$take(xt, indices = 0L, axis = 0L)}
    }

    if( ModelType == "DecoderOnly"){
      last_nonmasked_i <- jnp$argmax(jnp$where(jnp$equal(x_mask, 1),
                                                jnp$expand_dims(jnp$arange(x_mask$shape[[1]]),1L),
                                               -jnp$inf
                                               ) )
      xt <- jnp$take(xt,  indices = last_nonmasked_i, axis = 0L )

      if(paddingMethod == "right"){ stop("Not yet implemented in BuildML.R")}
    }

    xt
  }

  Encoder2Output <- function(TSList, 
                             xt, time, 
                             place, BNList, 
                             state, inference){
    x_mask <- xt[[2]]; xt <- xt[[1]]

    if("transformer" %in% BackboneType){
      xt <- RunTransformerBackbone(
        xt = xt,
        x_mask = x_mask,
        TransformerList = TSList$TSBackbone
      )
    }
    if("mamba" %in% BackboneType){
      stop("Mamba is not included in ndm.", call. = FALSE)
      print("Done sourcing SuperLModel_BackboneMamba.R in run path")
    }

    xt <- SelectBackboneOutputToken(xt = xt, x_mask = x_mask)
    xt <- DecoderBackboneToOutput(TSList = TSList, hidden_state = xt)
    return( xt ) 
  }

  Samp2Params <- ( function(localze,
                            fixedlocalze,
                            dynamicglobalze,
                            globalze,
                            dirichletparams,
                            context_,
                            tze,
                            seed){
    # local parameters - vector sequence - referenced to localze (these are dynamic local)
    lDepParamsVec_ <- NeuralDefinitions_jax_STOCHASTIC["DefClean",][as.logical(NeuralDefinitions_jax_STOCHASTIC["LDep",])]
    lDepParamsVec_ <- lDepParamsVec_[!lDepParamsVec_ %in% FixedLocalParams_defns]

    # fixed local parameters - vector sequence - indexed to fixedlocalze
    lFixedDepParamsVec_ <- FixedLocalParams_defns

    # global parameters - vector sequence - not indexed
    gDepParamsVec_ <- NeuralDefinitions_jax_STOCHASTIC["DefClean",][!as.logical(NeuralDefinitions_jax_STOCHASTIC["LDep",])]
    dgDepParamsVec_ <- gDepParamsVec_[gDepParamsVec_ %in% uq_globalneural_vec] # uq_globalneural_vec
    gDepParamsVec_ <- gDepParamsVec_[!gDepParamsVec_ %in% uq_globalneural_vec]

    # handle args
    nameArgs_ <- c(uq_encneural_vec, uq_args_vec, "Neural1")
    iArgs_ <- which(NeuralDefinitions_jax_STOCHASTIC["DefClean",] %in% nameArgs_)
    theArgs_ <- NeuralDefinitions_jax_STOCHASTIC["DefClean",][which(NeuralDefinitions_jax_STOCHASTIC["DefClean",] %in% nameArgs_)]
    theArgs_[theArgs_ %in% uq_encneural_vec] <- "Neural1"

    # get the arg samples
    args_samp_vec <- tapply(iArgs_, theArgs_, function(whereArg_i){
                              #whereArg_i <- iArgs_[theArgs_ == "Neural1"]

                              # set name
                              name_zap <- unique( gsub(NeuralDefinitions_jax_STOCHASTIC["DefClean",whereArg_i],pattern=" ",replace = "") )
                              if(all(name_zap %in% c("Neural1",uq_encneural_vec))){ name_zap <- "Neural1" }

                              # set transformation
                              transform_zap <- unique( gsub(NeuralDefinitions_jax_STOCHASTIC["Transformation",whereArg_i],pattern="Inv",replace="") )
                              if(name_zap %in% c("Neural1",uq_allneural_vec)){ transform_zap <- "" }# no transformation

                              # take local
                              if(name_zap %in% c("Neural1", ArgLDeps)){
                                # dynamic local
                                if(name_zap %in% "Neural1"){
                                eval(parse(text = sprintf("
                    %s_samp <- %s( jnp$take(localze,indices = jnp$array(tmp_i <- (ai(which(lDepParamsVec_ %%in%% c(uq_encneural_vec, 'Neural1'))-1L))), axis=0L) )
                ", name_zap, transform_zap) ))
                                }

                                # fixed local
                                if(name_zap %in% FixedLocalParams_defns){
                                  eval(parse(text = sprintf("
                    %s_samp <- %s( jnp$take(fixedlocalze,indices = jnp$array(tmp_i <- (ai(which(lFixedDepParamsVec_ == '%s')-1L))), axis=0L) )
                ",name_zap, transform_zap, name_zap)))
                                }
                              }

                              # further transform time dep parameters (only compute once, all params returned)
                              #if(name_zap %in% ArgTDeps & !(name_zap %in% uq_globalneural_vec)){
                              if( name_zap == "Neural1" ){
                                #eligibleTDepNames <- name_zap[name_zap %in% ArgTDeps & !(name_zap %in% uq_globalneural_vec)]
                                #if( name_zap == eligibleTDepNames[1] )
                                {
                                  if( encneuralType == "g" ){
                                    nODE_wt_resid <- NULL; for(d_a in 1:(nDepth_nODE+1)){
                                      eval(parse(text = sprintf("nODE_ln%s_center <- nODE_ln%s_scale <- nODE_bias%s <- nODE_wt%s <-  NULL",d_a, d_a,d_a, d_a)))
                                    }
                                  }
                                  if( encneuralType == "l" ){
                                    if(!residCon_neuralODE){ eval(parse(text = sprintf("nODE_wt_resid <- NULL")))  }
                                    if(residCon_neuralODE){
                                      eval(parse(text = sprintf("nODE_wt_resid <-  jnp$take(%s_samp,nODE_loc_list_param_mean$wts_resid, axis=0L)",name_zap)))
                                      if(!neuralODE_lowRank){ nODE_wt_resid <-  jnp$reshape(nODE_wt_resid, list(-1L, outputDim_nODE_local))}
                                      if(neuralODE_lowRank){
                                        left_ <-  jnp$reshape(jnp$take(nODE_wt_resid,
                                                                       indices = jnp$array( 0L:(tmp_<- (inputDim_nODE_local*lowRankDims-1L)))
                                                                       ), list(-1L,lowRankDims))
                                        m_transf <- function(m_){ return( jnp$linalg$svd(m_, full_matrices = F, compute_uv = T)[[1]] ) }
                                        left_ <- m_transf( left_ )
                                        mid_ <-  jnp$diag( jnp$reshape( SoftPlus(jnp$take(nODE_wt_resid,
                                                                       indices = jnp$array( (tmp_+1L):(tmp_ <- (tmp_+1L+lowRankDims-1L)))    )),
                                                             list(-1L)) )
                                        right_ <-  jnp$reshape(jnp$take(nODE_wt_resid,
                                                                  indices = jnp$array( (tmp_+1L):(nODE_wt_resid$shape[[1]]-1L))  ),
                                                               list(-1L,lowRankDims))
                                        right_ <- jnp$transpose(m_transf(right_))
                                        nODE_wt_resid <- jnp$matmul(jnp$matmul(left_, mid_),right_)
                                      }
                                      # rescale
                                      #nODE_wt_resid <- jnp$multiply(nODE_wt_resid, jnp$array(    sqrt(0.5 / (inputDim_nODE_local + inputDim_nODE_local)   )))
                                    }
                                    for(d_a in 1:(nDepth_nODE+1)){
                                      if(d_a == 1L){ wt_init_ <-  sqrt(0.5 / (inputDim_nODE_local + nWidthODEHidden_ts_local)) }
                                      if(d_a > 1 & d_a <= nDepth_nODE){ wt_init_ <-  sqrt(0.5 / (nWidthODEHidden_ts_local + nWidthODEHidden_ts_local)) }
                                      if(d_a > nDepth_nODE){ wt_init_ <-  sqrt(0.5 / (nWidthODEHidden_ts_local + outputDim_nODE_local)) }
                                      if(d_a <= nDepth_nODE & ODE_NORM_TYPE != "none"){
                                        eval(parse(text = sprintf("nODE_ln%s_scale <-  SoftPlus(jnp$add(jnp$multiply(jnp$take(%s_samp,nODE_loc_list_param_mean$ln%s_scale, axis=0L), 0.10),InvSoftPlus(1)))",d_a,name_zap,d_a)))
                                        eval(parse(text = sprintf("nODE_ln%s_center <-  jnp$multiply(jnp$take(%s_samp,nODE_loc_list_param_mean$ln%s_scale, axis=0L), 0.10)",d_a,name_zap,d_a)))
                                      }
                                      if(d_a > nDepth_nODE | ODE_NORM_TYPE == "none"){
                                        eval(parse(text = sprintf("nODE_ln%s_scale <-  jnp$array(0.)",d_a)))
                                        eval(parse(text = sprintf("nODE_ln%s_center <-  jnp$array(0.)",d_a)))
                                      }
                                      eval(parse(text = sprintf("nODE_wt%s <-  jnp$take(%s_samp,nODE_loc_list_param_mean$wts%s, axis=0L)",d_a,name_zap,d_a)))
                                      if(!neuralODE_lowRank){
                                        if(d_a <= nDepth_nODE){ eval(parse(text = sprintf("nODE_wt%s <-  jnp$reshape(nODE_wt%s, list(-1L,nWidthODEHidden_ts_local))",d_a,d_a))) }
                                        if(d_a > nDepth_nODE){ eval(parse(text = sprintf("nODE_wt%s <-  jnp$reshape(nODE_wt%s, list(-1L,outputDim_nODE_local))",d_a,d_a))) }
                                      }
                                      if(neuralODE_lowRank){
                                        eval(parse(text = sprintf("nODE_wt_ <- nODE_wt%s", d_a)))
                                        left_dim <- ifelse(d_a == 1L, yes = inputDim_nODE_local, no = nWidthODEHidden_ts_local) # 1 is hidden layer, 2 is final projection
                                        right_dim <- ifelse(d_a == 1L, yes = nWidthODEHidden_ts_local, no = outputDim_nODE_local)
                                        left_ <-  jnp$reshape(jnp$take(nODE_wt_,
                                                                       indices = jnp$array( 0L:(tmp_<- (left_dim*lowRankDims-1L)))  ),
                                                              list(-1L,lowRankDims))
                                        left_ <- m_transf( left_ )
                                        mid_ <-  jnp$diag( jnp$reshape( SoftPlus(
                                                                   jnp$take(nODE_wt_,
                                                                      indices = jnp$array( (tmp_+1L):(tmp_ <- (tmp_+1L+lowRankDims-1L)))   )),
                                                                    list(-1L)) )
                                        right_ <-  jnp$reshape(jnp$take(nODE_wt_,
                                                                        indices = jnp$array( (tmp_+1L):(nODE_wt_$shape[[1]]-1L))   ),
                                                               list(-1L,lowRankDims))
                                        right_ <- jnp$transpose( m_transf(right_) )
                                        nODE_wt_ <- jnp$matmul( jnp$matmul(left_, mid_), right_ )
                                        eval(parse(text = sprintf("nODE_wt%s <- nODE_wt_", d_a)))
                                      }
                                      #eval(parse(text = sprintf("nODE_wt%s <-  jnp$multiply(nODE_wt%s, jnp$array(   wt_init_    ))",d_a,d_a)))
                                      eval(parse(text = sprintf("nODE_bias%s <-  jnp$multiply(jnp$take(%s_samp,nODE_loc_list_param_mean$bias%s,axis=0L), jnp$array(  1/1e2  ))",d_a,name_zap,d_a)))
                                    }
                                  }
                                  tmp <- eval(parse(text =
                                  paste("list(",
                                    "'nODE_wt_resid' = nODE_wt_resid,",
                                    paste(sapply(1:(nDepth_nODE+1),function(d_r) sprintf(
                                      "'nODE_ln%s_scale' = nODE_ln%s_scale,
                                       'nODE_wt%s' = nODE_wt%s,
                                       'nODE_bias%s' = nODE_bias%s", d_r, d_r, d_r, d_r, d_r, d_r)), collapse = ","),

                                    # wt norm scaling
                                    # NOTE: nODE SCALE DEPRECIATED
                                    ",'nODE_scale' = ModelList$ScaleList$nODE_scale, 

                                    # scaling output
                                    'nODE_out_scale' = SoftPlus( ModelList$ScaleList$nODE_out_scale[[1]] ),",

                                    # initial condition - extra ) closes the group
                                    sprintf("'%s_0' = jnp$take(%s_samp,jnp$array( (length(tmp_i)-LocalNeuralEmbedDim-length(uq_encneural_vec)):
                                                                                      (length(tmp_i)-1L)  )))",name_zap,name_zap),

                                    # close paste
                                    collapse="") ) )
                                  eval(parse(text = sprintf("%s_samp <- tmp", name_zap)))
                                }
                              }

                              # take global
                              if(!name_zap %in% ArgLDeps & !(name_zap %in% uq_globalneural_vec) & !name_zap %in% "Neural1"){
                                eval(parse(text = sprintf("
                    %s_samp <- %s( jnp$squeeze(jnp$take(globalze,indices = jnp$array(tmp_i <- (ai(which(gDepParamsVec_ == '%s')-1L))), axis=0L)) )
                ",name_zap, transform_zap, name_zap)))
                              }

                              # take dynamic global (ODE solved outside of Samp2Params)
                              if(!name_zap %in% ArgLDeps & (name_zap %in% uq_globalneural_vec)){
                                eval(parse(text = sprintf("
                    %s_samp <- %s( jnp$squeeze(jnp$take(dynamicglobalze,indices = jnp$array(tmp_i <- (ai(which(dgDepParamsVec_ == '%s')-1L))), axis=0L)) )
                ",name_zap, transform_zap, name_zap)))
                              }

                              # return content
                              return( eval(parse(text = sprintf("list('%s_samp' = %s_samp)",name_zap, name_zap)))  )
                            })

    # context parameters - DEPRECIATED
    iContext_ <- which(NeuralDefinitions_jax_CONTEXT["DefClean",] %in% c(uq_encneural_vec,uq_args_vec))
    theContext_ <- NeuralDefinitions_jax_CONTEXT["DefClean",][which(NeuralDefinitions_jax_CONTEXT[5,] %in%
                                                                        c(uq_encneural_vec,uq_args_vec) )]

    args_context_vec <- tapply(iContext_, theContext_, function(whereContext_i){
                                 name_zap <- unique( gsub(NeuralDefinitions_jax_CONTEXT[5,whereContext_i],pattern=" ",replace = "") )
                                 transform_zap <- unique( gsub(NeuralDefinitions_jax_CONTEXT[1,whereContext_i],pattern="Inv",replace="") )
                                 eval(parse(text = sprintf("%s_samp <- %s", name_zap, transform_zap)))
                                 if(name_zap %in% VI_param_terms_TDependency){
                                   eval(parse(text = sprintf("%s_samp <- diffraxInterpolator(interp_ts, %s_samp)",name_zap,name_zap)))
                                 }
                                 return(eval(parse(text = sprintf("list('%s_samp' = %s_samp)",name_zap,name_zap))))
                               })
    args_samp_vec <- c(args_samp_vec, args_context_vec)
    names(args_samp_vec) <- paste(names(args_samp_vec),"_samp",sep="")

    # select initial conditions
    uq_initcond_vec <- get0(
      "InitStateTerms",
      inherits = TRUE,
      ifnotfound = uq_y_vec[!uq_y_vec %in% uq_allneural_vec]
    )
    uq_initcond_vec <- as.character(uq_initcond_vec)
    init_state_supported_terms <- unique(uq_initcond_vec)
    if (length(init_state_supported_terms) == 0L) {
      stop(
        "NeuralODE init-state softmax requires at least one declared init-state term.",
        call. = FALSE
      )
    }
    if (length(init_state_supported_terms) != length(uq_initcond_vec)) {
      stop("NeuralODE init-state softmax requires unique `InitStateTerms`.", call. = FALSE)
    }
    init_logit_param_names <- get0(
      "InitStateLogitParamNames",
      inherits = TRUE,
      ifnotfound = paste0("InitStateLogit_", uq_initcond_vec)
    )
    init_logit_param_names <- as.character(init_logit_param_names)
    init_scale_param_name <- as.character(
      get0("InitStateScaleParamName", inherits = TRUE, ifnotfound = "InitStateLogitScale")
    )
    init_param_indices <- match(init_logit_param_names, lDepParamsVec_)
    if (anyNA(init_param_indices)) {
      stop(
        sprintf(
          "Missing NeuralODE init-state logit parameters: %s",
          paste(init_logit_param_names[is.na(init_param_indices)], collapse = ", ")
        ),
        call. = FALSE
      )
    }
    init_scale_index <- match(init_scale_param_name, lDepParamsVec_)
    if (is.na(init_scale_index)) {
      stop(
        sprintf("Missing NeuralODE init-state scale parameter: %s", init_scale_param_name),
        call. = FALSE
      )
    }
   
   { # performs better in tests 
      init_m <- jnp$take(localze, jnp$array(init_param_indices - 1L))
      init_logit_offset_default <- c(10, rep(-1, max(0L, length(uq_initcond_vec) - 1L)))
      if (length(init_logit_offset_default) > 0L) {
        names(init_logit_offset_default) <- uq_initcond_vec
      }
      init_logit_offset_raw <- get0(
        "InitStateLogitOffset",
        inherits = TRUE,
        ifnotfound = init_logit_offset_default
      )
      init_logit_offset_names <- names(init_logit_offset_raw)
      init_logit_offset <- as.numeric(init_logit_offset_raw)
      if(length(init_logit_offset) == 1L){
        init_logit_offset <- rep(init_logit_offset, length(uq_initcond_vec))
      } else if(!is.null(init_logit_offset_names) && any(nzchar(init_logit_offset_names))){
        names(init_logit_offset) <- init_logit_offset_names
        if(!all(uq_initcond_vec %in% init_logit_offset_names)){
          stop(
            sprintf(
              "InitStateLogitOffset names must cover {%s}.",
              paste(uq_initcond_vec, collapse = ", ")
            ),
            call. = FALSE
          )
        }
        init_logit_offset <- init_logit_offset[uq_initcond_vec]
      } else {
        if(length(init_logit_offset) != length(init_state_supported_terms)){
          stop(
            sprintf(
              "InitStateLogitOffset must have length 1 or %s, or be named by init-state terms.",
              length(init_state_supported_terms)
            ),
            call. = FALSE
          )
        }
        names(init_logit_offset) <- init_state_supported_terms
        init_logit_offset <- init_logit_offset[uq_initcond_vec]
      }
      init_logit_offset <- jnp$array(as.numeric(init_logit_offset))$astype(init_m$dtype)
      init_logit_scale_max <- suppressWarnings(as.numeric(get0("InitStateLogitScaleMax", inherits = TRUE, ifnotfound = Inf)))
      if(length(init_logit_scale_max) == 0L){ init_logit_scale_max <- Inf }
      if(length(init_logit_scale_max) != 1L){
        stop("InitStateLogitScaleMax must be scalar.")
      }
      init_logit_scale_max <- init_logit_scale_max[[1L]]
      init_scale <- SoftPlus( InvSoftPlus(1.) + 
                      jnp$take(localze, jnp$array(init_scale_index - 1L ))*1  )
      if(is.finite(init_logit_scale_max)){
        init_scale <- jnp$minimum(
          init_scale,
          jnp$array(init_logit_scale_max)$astype(init_scale$dtype)
        )
      }
      # plot(np$array(init_m$val)[,1])
     
      #init_m <- jnp$take(localze, jnp$array( which(lDepParamsVec_ %in% uq_initcond_vec)[1:3] - 1L ))
      #init_m <- jnp$concatenate(list( init_m, jnp$zeros(1L)$astype(init_m$dtype) ), 0L)
      #init_m <-  ( init_m  + jnp$array(c(10,0,0,0)) ) *
      init_m <-  ( ( init_m  + init_logit_offset )) * init_scale # 0.1,1 before 
      init_samp <- jnp$exp(  jax$nn$log_softmax( init_m  ) + jnp$log(CONST_N)   )
      # plot(np$array(init_samp$val)[,1])
    }
    # init_samp$sum()
    # cor(np$array( init_m_T$val$val )[,1,])
    # cor(np$array( init_samp$val$val )[,1,])
    # plot(np$array( init_samp$val$val )[,1,1] )
    # plot(np$array( init_samp$val$val )[,1,2:3] )

    if("N_samp" %in% names(args_samp_vec)){ init_samp <- jnp$multiply( init_samp,  args_samp_vec$N_samp ) }

    for(j_ in 1L:length(uq_initcond_vec)){
      eval(parse(text = sprintf("%s_samp = jnp$take(init_samp, j_ - 1L)", uq_initcond_vec[j_])))
    }
    tmp <- paste0(uq_initcond_vec, "_samp")
    tmp <- paste("c(args_samp_vec,", paste(paste("'",tmp,"'=",tmp,sep = ""),collapse=","), ")",collapse="")
    
    return(   eval(parse(text = tmp)) )
  } )

  PolicyList <- list( # policy indicator
                      diffrax$LinearInterpolation(jnp$array(as.numeric(0:(nTimesTotal-1L))),
                                                  jnp$array(rep(0,nTimesTotal))),

                       # scenario (defines the smooth shift trajectory in the policy from the previous state)
                       diffrax$LinearInterpolation(jnp$array(as.numeric(0:NTimeSteps_SIM)),
                                                   jnp$array(rep(-0.10,times=NTimeSteps_SIM+1))))
  
  GetPred <- function(ModelList, # not externally vectorized 
                      x, # vectorized
                      state, # not vectorized
                      inference, # not vectorized
                      PriorList, # not vectorized
                      PolicyList, # not vectorized
                      GetPredSaveAtInfo, # not vectorized
                      seed){ # vectorized

    # parcel out context information
    context <- x[[2]][[1]]
    loc_indices <- jnp$squeeze( x[[3]][[1]]$astype( jnp$int32 ) )
    time_indices <- jnp$squeeze( x[[4]][[1]]$astype( jnp$int32 ) )
    x  <- x[[1]]

    # Modeling of outcome variance + parameter KL div
    #TSList_str <- jax$tree_util$tree_flatten(  ModelList[[ListIndices["TSList"]]] )[[2]]  
    # return_v <- eq$filter_vmap(function(TSList_){
    return_v <- {
      state <- jnp$array(1.); 

      # initial processing 
      x <- ProcessEncoderInput(
                          InitProcessList = ModelList$InitProcessList ,# 
                          TSList = ModelList$TSList, 
                          xt = x,
                          time = time_indices,
                          place = loc_indices,
                          BNList = ModelList$BNList, 
                          state = state,
                          inference = inference)
      # plot( np$array(x[[1]]$val)[i_<-sample(1:10,1),,1] )
      # plot( np$array(x[[2]]$val)[i_,,1] )
      
      if(TRUE){ # decoder with prefilling
      if( ModelType == "DecoderOnly"){
          if(paddingMethod == "right"){ stop("paddingMethod must be 'left' for decoder-only models!")}
          
          # Extend running buffers by forecast horizon
          oldx <- x[[1]]$shape[[1]]
          GEN_CAP <- GetPredSaveAtInfo[[1]]
          xt_running <- list(
            jnp$concatenate(list(x[[1]], jnp$zeros(list(GEN_CAP, x[[1]]$shape[[2]]))), 0L),  # [T_total, D]
            jnp$concatenate(list(x[[2]], jnp$zeros(list(GEN_CAP, 1L))), 0L)                  # [T_total, 1]
          )
          
          # Count known prefix length from mask (ones)
          prefix_len <- jnp$sum(xt_running[[2]])$astype(jnp$int32)
          
          # When KV caching cannot be used, fall back to original full pass/scan
          if (!EnableKVCaching) {
            # ---- ORIGINAL PATH (unchanged) ----
            decoder_step <- function(xt_running, t_){
              xt_new <- Encoder2Output(
                TSList = ModelList$TSList, 
                xt = xt_running,
                time = time_indices,
                place = loc_indices,
                BNList = ModelList$BNList, 
                state = state,
                inference = inference)
              start_idx <- jnp$argmax(jnp$where(jnp$equal(xt_running[[2]], 1),
                                                jnp$expand_dims(jnp$arange(xt_running[[2]]$shape[[1]]),1L),
                                                -jnp$inf)) + 1L
              new_input <- jax$lax$dynamic_update_slice(
                xt_running[[1]], jnp$expand_dims(xt_new, 0L),
                jnp$array(c(start_idx, 0L), dtype = jnp$int32)
              )
              new_mask <- jax$lax$dynamic_update_slice(
                xt_running[[2]], jnp$ones(list(1L,1L), dtype = xt_running[[2]]$dtype), jnp$array(c(start_idx, 0L), dtype = jnp$int32)
              )
              xt_running <- list(new_input, new_mask)
              list(xt_running, ModelList$TSList$TSBackbone$DecoderProj(xt_new))
            }
            K_static <- as.integer(nTimesLookahead) 
            scan_out <- jax$lax$scan(
              f    = decoder_step,
              init = xt_running,
              #xs   = jnp$arange(start = 1L, stop = GetPredSaveAtInfo[[1]]+1)
              xs = NULL, length = K_static
            )
            y_mean  <- jax$nn$softplus(scan_out[[2]])
            y_sigma <- jnp$ones_like(y_mean)
            KL_TERM <- jnp$array(0.)
            ODEParamsSampList_y0 <- ODEParamsSampList_args <- NULL
            diff_eq_sol <- NULL; diff_eq_sol$ts <- diff_eq_sol$ys <- NULL
          }
          if( EnableKVCaching) {
            # ---- KV-CACHED PATH ----
            
            # 1) Prefill KV cache once for the known prefix subset.
            #    This uses the per-layer TRY_FLASH projections (W_q/W_k/W_v/W_o).
            prefill_ret <- transformer_prefill_kv(
              xt        = xt_running[[1]],
              x_mask    = xt_running[[2]],
              TransformerList = ModelList$TSList$TSBackbone,
              prefix_len = prefix_len
            )
            kv_cache   <- prefill_ret$cache
            xt_last_raw <- prefill_ret$xt_last
            xt_last <- DecoderBackboneToOutput(
              TSList = ModelList$TSList,
              hidden_state = xt_last_raw
            )
            
            # 2) First prediction y_1 uses representation at last known token.
            y_first    <- ModelList$TSList$TSBackbone$DecoderProj(xt_last)  # [nOutcomes]
            
            # 3) Insert xt_last as the embedding for the *next* position (prefix_len)
            insert_pos <- prefix_len                         # index to write next token (0-based)
            xt_running[[1]] <- jax$lax$dynamic_update_slice(
              xt_running[[1]],
              jnp$expand_dims(xt_last, 0L),                 # [1, D]
              jnp$array(c(insert_pos, 0L), dtype = jnp$int32)
            )
            xt_running[[2]] <- jax$lax$dynamic_update_slice(
              xt_running[[2]],
              jnp$ones(list(1L,1L), dtype = xt_running[[2]]$dtype),
              jnp$array(c(insert_pos, 0L), dtype = jnp$int32)
            )
            
            # 4) Decode subsequent steps with cache.
            # Carry: (xt_running, kv_cache, pos)
            decode_step_cached <- function(carry, t_) {
              xt_run <- carry[[1]]; cache <- carry[[2]]; pos <- carry[[3]] # pos is index of current last known token
              # Take the input embedding at 'pos' (it was inserted in the previous step)
              token_in <- jnp$take(xt_run[[1]], jnp$array(pos, dtype = jnp$int32), axis = 0L)  # [D]
              
              # Run a single-layer stack forward for this one token, updating per-layer K/V at 'pos'
              sret <- transformer_decode_step_kv(
                token_in       = token_in,
                pos            = pos,
                TransformerList= ModelList$TSList$TSBackbone,
                cache          = cache
              )
              embed_out <- DecoderBackboneToOutput(
                TSList = ModelList$TSList,
                hidden_state = sret$token_out
              )
              cache     <- sret$cache
              
              # Insert embed_out as the input for the next (pos+1) position and open its mask
              write_pos <- pos + 1L
              xt_next <- jax$lax$dynamic_update_slice(
                xt_run[[1]], jnp$expand_dims(embed_out, 0L),
                jnp$array(c(write_pos, 0L), dtype = jnp$int32)
              )
              m_next <- jax$lax$dynamic_update_slice(
                xt_run[[2]], jnp$ones(list(1L,1L), dtype = xt_run[[2]]$dtype),
                jnp$array(c(write_pos, 0L), dtype = jnp$int32)
              )
              
              # Project to prediction (y_t at next step)
              y_t <- ModelList$TSList$TSBackbone$DecoderProj(embed_out) # [nOutcomes]
              
              list(list(list(xt_next, m_next), cache, write_pos),
                   y_t)
            }
            
            scan_out2 <- jax$lax$scan(
              f    = decode_step_cached,
              init = list(xt_running, kv_cache, insert_pos),
              xs   = jnp$arange(start = 0L, stop = GEN_CAP-1L) # first entry is already in the prefill, hence -1L
            )
            y_tail <- scan_out2[[2]]  # [num_steps, nOutcomes, nFeatures]
            
            # Concatenate first + tail => [nTimesLookahead, nOutcomes]
            # Goal:
            # y_first: [D]
            # y_tail : [K, D]   (K is static)
            # We want: [1 + count, D] logically,
            # but we keep static shape [1 + K, D] by zero-masking tail rows >= count.
            count_i32   <- jnp$astype(GEN_CAP-1L, jnp$int32)
            K           <- scan_out2[[2]]$shape[[1]]
            idxK        <- jnp$arange(K, dtype = jnp$int32)       # OK: K is static
            keep_tail   <- jnp$less(idxK, count_i32)              # [K]
            masked_tail <- jnp$where(jnp$expand_dims(keep_tail, 1L),
                                       scan_out2[[2]],
                                       jnp$zeros_like(scan_out2[[2]]))
            y_all <- jnp$concatenate(list(jnp$expand_dims(y_first, 0L), masked_tail), 0L)
            
            y_mean  <- jax$nn$softplus(y_all)   # preserve your softplus post-proj
            y_sigma <- jnp$ones_like(y_mean)
            KL_TERM <- jnp$array(0.)
            ODEParamsSampList_y0 <- ODEParamsSampList_args <- NULL
            diff_eq_sol <- NULL; diff_eq_sol$ts <- diff_eq_sol$ys <- NULL
          }
        }
      }
      if(FALSE){  # decoder no prefilling 
      if( ModelType == "DecoderOnly" ){ 
        if(paddingMethod == "right"){stop("paddingMethod must be 'left' for decoder-only models!")}
        # decoder architecture 
        # 3) We'll keep a "running" input. Start with your existing context/historical x:
        oldx <- x[[1]]$shape[[1]]
        xt_running <- list(jnp$concatenate(list(x[[1]], 
                            jnp$zeros(list(nTimesLookahead, x[[1]]$shape[[2]]))), 0L),
                          jnp$concatenate(list(x[[2]], 
                            jnp$zeros(list(nTimesLookahead, 1L))), 0L))
    
        # trial step to ensure correctness
        if(T == F){ 
          Encoder2Output(
            TSList = ModelList$TSList, 
            xt = xt_running,
            time = time_indices,
            place = loc_indices,
            BNList = ModelList$BNList, 
            state = state,
            inference = inference)
        }
        
        # 4) for t in 1..T_forecast:
        decoder_step <- function(xt_running, t_){
          # 4a) run the transformer *causally* on running_input
          #     must pass a causal mask that matches length(running_input)
          #     so the model sees only prior steps for each token.
          xt_new <- Encoder2Output(
                        TSList = ModelList$TSList, 
                        xt = xt_running,
                        time = time_indices,
                        place = loc_indices,
                        BNList = ModelList$BNList, 
                        state = state,
                        inference = inference)
          
          # 4c) now insert into to running_input: 
          # Insert the new prediction into the running input at the current timestep position
          start_idx <- (nTimeStepsInHistory + 
                                1 + # ONE FOR PLACE EMBEDDINGS 
                                + t_ - 1L)  # Position to update in sequence
          new_input <- jax$lax$dynamic_update_slice(
            xt_running[[1]],
            jnp$expand_dims(xt_new, axis = 0L),
            jnp$array(c(start_idx, 0L))$astype(jnp$int32)  # Update [seq_pos, batch, features]
          )
          # plot( np$array(xt_running[[1]]$val)[1,,1] ) 
          # plot( np$array(new_input$val)[1,,1] ) 
          
          # Update mask to 1 at the current timestep position
          new_mask_val <- jnp$ones(list(1L, 1L), 
                                   dtype = xt_running[[2]]$dtype)
          new_mask <- jax$lax$dynamic_update_slice(
            xt_running[[2]],
            new_mask_val,
            jnp$array(c(start_idx, 0L))$astype(jnp$int32)  # Update [seq_pos, mask_dim]
          )
          # c(np$array(xt_running[[2]]$val[1,]))
          # c(np$array(new_mask$val[1,]))
          
          # Update running_input with new prediction and mask
          xt_running <- list(new_input, new_mask)
          return( list(xt_running, 
                       TransformerList$DecoderProj(xt_new) ) )  
          # 1st element represents a new value for the loop carry 
          # and the second represents a slice of the output.
        }
        # xt_running[[1]]$shape; xt_running[[2]]$shape
        scan_out <- jax$lax$scan(
          f    = decoder_step,
          init = xt_running,
          xs   = jnp$arange(start = 1L, stop = GetPredSaveAtInfo[[1]]+1)
        )
        # TransformerList$DecoderProj(xt_new)
        
        xt_running_final <- scan_out[[1]]   # final updated carry
        y_mean          <- jax$nn$softplus( scan_out[[2]] )# * 0.1 
        y_sigma         <- jnp$ones_like(y_mean)

        # 6) If your code expects a big T dimension including historical time,
        #    you can also combine them, or keep them separate. 
        #    Return in the same shape as "NeuralODE" does:
        KL_TERM <- jnp$array(0.)
        ODEParamsSampList_y0 <- ODEParamsSampList_args <- NULL
        diff_eq_sol <- NULL
        diff_eq_sol$ts <- diff_eq_sol$ys <- NULL
      }
      }
  
      if( ModelType == "NeuralODE" ){ 
      x <- Encoder2Output(#TSList = jax$tree_util$tree_unflatten(TSList_str, TSList_),# push it to axis 0
                            TSList = ModelList$TSList, 
                            xt = x,
                            time = time_indices,
                            place = loc_indices,
                            BNList = ModelList$BNList, 
                            state = state,
                            inference = inference)
      # set initial null parameter vectors (replaced if needed)
      globalLMat <- globalx_mu_params_ODE <- globalx_sigma_params_ODE <- jnp$array(0.)
      localx_mu_neural_params_ODE <- localx_Diag_params_ODE <- jnp$array(0.)
      globalLMat <- jnp$expand_dims(jnp$expand_dims(globalLMat,0L),1L)
  
      # get params
      IndicesLocalMuParams <- ai(0:(stop_ <- (nODEParams-1L)))
      IndicesLocalLParams <- ai((stop_+1):(stop_<-(stop_+1L+nParams_LMat_base-1L)))
      localx_LMat_params_ODE <- jnp$take(x, jnp$array( IndicesLocalLParams ))
      if(nParams_DiagMat_neural > 0){
        IndicesLocalDiagParams <- ai((stop_+1):(stop_ <- (stop_+1L+nParams_DiagMat_neural-1L)))
        localx_Diag_params_ODE <- jnp$divide(SoftPlus( jnp$take(x, jnp$array(IndicesLocalDiagParams)) ),10000.)
      }
  
      # scale initial sd
      IndicesSDParams <- jnp$array(IndicesSDParams_R <- (nODEParams_total+1L):nDimOutput_dense - 1L)
      ySigma_init_nODE_untransformed <- ModelList$ScaleList$ScaleBayes$VarInit + 
                                              jnp$take(x, jnp$array(  IndicesSDParams_R ))
      if( length(IndicesSDParams$shape) == 0 ){ 
        ySigma_init_nODE_untransformed <- jnp$expand_dims(ySigma_init_nODE_untransformed, 0L)
      }
  
      # get params based on indices
      localx_mu_params_ODE <- jnp$take(x, jnp$array(IndicesLocalMuParams))
      NeuralDefinitions_jax_STOCHASTIC_LDep <- NeuralDefinitions_jax_STOCHASTIC[,
                                                  as.logical(NeuralDefinitions_jax_STOCHASTIC["LDep",])]
      NeuralDefinitions_jax_STOCHASTIC_LDep <- NeuralDefinitions_jax_STOCHASTIC_LDep[,
         !NeuralDefinitions_jax_STOCHASTIC_LDep["DefClean",] %in% FixedLocalParams_defns]
      
      ref_i_base <- which(!as.logical(NeuralDefinitions_jax_STOCHASTIC_LDep["TDep",]) &
                            !NeuralDefinitions_jax_STOCHASTIC_LDep["DefClean",] %in% FixedLocalParams_defns) - 1L
      ref_i_neural <- which(as.logical(NeuralDefinitions_jax_STOCHASTIC_LDep["TDep",])) - 1L
      
      localx_mu_base_params_ODE <- jnp$take(localx_mu_params_ODE, jnp$array(ref_i_base))
      localx_mu_neural_params_ODE <- jnp$take(localx_mu_params_ODE, jnp$array(ref_i_neural))
  
      # checks:
      # IndicesLocalMuParams; IndicesLocalLParams; IndicesLocalDiagParams; IndicesSDParams
  
      # obtain full Sigma mat
      # create lower triangular matrix manually
      LMat_diag_scale <- SoftPlus( ModelList$ScaleList$ScaleBayes$LmatLocalDiagScaler )
      LMat_global_scale <- SoftPlus( ModelList$ScaleList$ScaleBayes$LmatGlobalScaler )
      if(isTRUE(UseDiagonalLMatVCov)){ localx_DiagOfSigma <- SoftPlus(localx_LMat_params_ODE)/LMat_diag_scale } 
      if(!isTRUE(UseDiagonalLMatVCov)){ 
        OrigDiag_LMat <- jnp$diag(  LMat <- oryx$math$fill_triangular( localx_LMat_params_ODE  )   ) # XXX
        DiagMat1 <- jnp$diag(  OrigDiag_LMat )
        DiagMat2 <- jnp$diag( jnp$add(jax$nn$softplus(OrigDiag_LMat),LMat_diag_scale))
        localx_LMat <- jnp$add(jnp$subtract(LMat, DiagMat1), DiagMat2)
        localx_LMat <- jnp$multiply( localx_LMat, LMat_global_scale )
      }
  
      # create sigma mat if needed
      #localSigmaMat <- jnp$matmul(LMat,jnp$transpose(LMat)) # Original Cov +
      #localSigmaMat <- jnp$add(localSigmaMat, jnp$multiply(jnp$array(Sigma_LAMBDA <- 0.001), jnp$eye(LMat$shape[[1]]))  ) # lambda * I
  
      # work thru global parameters
      globalLMat_diag_scale <- SoftPlus( ModelList$ScaleList$ScaleBayes$globalLmatDiagScaler )
      globalLMat_global_scale <- SoftPlus( ModelList$ScaleList$ScaleBayes$globalLmatGlobalScaler )
      
      GlobalTerms <- ModelList$GlobalTerms
      if(length(ArgNoDeps) > 0 && nGlobalParams > 0L){
        IndicesGlobalMuParams <- ai(seq_len(nGlobalParams) - 1L)
        IndicesGlobalSigmaParams <- ai((max(IndicesGlobalMuParams)+1L):(GlobalTerms$shape[[1]]-1L))
        globalx_mu_params_ODE <- jnp$take(GlobalTerms, jnp$array(IndicesGlobalMuParams))
        globalx_sigma_params_ODE <- jnp$take(GlobalTerms, jnp$array(IndicesGlobalSigmaParams))
  
        if(length(IndicesGlobalSigmaParams) == 1L){
          globalLMat <- jnp$square(globalx_sigma_params_ODE)
          globalLMat <- jnp$expand_dims(globalLMat,0L)
          globalLMat <- jnp$expand_dims(globalLMat,1L)
        }
        if(length(IndicesGlobalSigmaParams) > 1){
          globalLMat <- oryx$math$fill_triangular( globalx_sigma_params_ODE  ) # XXX
          globalOrigDiag_LMat <- jnp$diag(  globalLMat   )
          globalDiagMat1 <- jnp$diag(  globalOrigDiag_LMat )
          globalDiagMat2 <- jnp$diag( jnp$add(jax$nn$softplus(globalOrigDiag_LMat),globalLMat_diag_scale))
          globalLMat <- ((globalLMat-  globalDiagMat1) + globalDiagMat2)
          globalLMat <- globalLMat * globalLMat_global_scale 
  
          # create sigma mat if needed
          #globalSigmaMat <- jnp$matmul(globalLMat,jnp$transpose(globalLMat)) # Original Cov +
          #globalSigmaMat <- jnp$add(globalSigmaMat, jnp$multiply(jnp$array(Sigma_LAMBDA), jnp$eye(globalLMat$shape[[1]]))  )
        }
      }
  
      # Define param distributions
      if( !prior_sampling ){
        # sample local non-neural parameters
        if(!isTRUE(UseDiagonalLMatVCov)){ 
          localParamD <- oryx$MultivariateNormalTriL( loc = localx_mu_base_params_ODE, 
                                                      scale_tril = localx_LMat)
        }
        if(isTRUE(UseDiagonalLMatVCov)){ 
          localParamD <- oryx$MultivariateNormalDiag( loc = localx_mu_base_params_ODE, 
                                                      scale_diag = localx_DiagOfSigma)
        }
  
        # sample local neural parameters
        localParamD_neural <- oryx$Normal( loc = localx_mu_neural_params_ODE,
                                           scale = localx_Diag_params_ODE)
  
        # obtain fixed local (check?)
        fixedLocal_means <- fixedLocal_sds <- fixedlocal_x_params_samp <- jnp$array(0.)
        if(nFixedLocal > 0){
          fixedLocal_means <- jnp$take( ModelList$FixedLocalTerms[[1]], loc_indices, 0L)
          fixedLocal_sds <- SoftPlus( jnp$take( ModelList$FixedLocalTerms[[2]], loc_indices, 0L) )
          fixedLocalParamD <- oryx$Normal( loc = fixedLocal_means, scale = fixedLocal_sds )
        }
        global_x_params_samp <- jnp$array(0.)
        if(length(ArgNoDeps) > 0 && nGlobalParams > 0L){
          globalParamD <- oryx$MultivariateNormalTriL(loc = globalx_mu_params_ODE,
                                                      scale_tril = globalLMat)
        }
      }
  
      # Define prior distributions if doing prior sampling
      if( prior_sampling ){
        localPriorList <- PriorList[[1]]; globalPriorList <- PriorList[[2]]
  
        localx_mu_params_ODE <- localPriorList[[1]]; localSigmaMat <- localPriorList[[2]]
  
        if(length(ArgNoDeps) > 0 && nGlobalParams > 0L){
          globalx_mu_params_ODE <- globalPriorList[[1]]; globalSigmaMat <- globalPriorList[[2]]
        }

        localParamD <- oryx$MultivariateNormalFullCovariance(loc = localx_mu_params_ODE, covariance_matrix = localSigmaMat)
        if(length(ArgNoDeps) > 0 && nGlobalParams > 0L){
          globalParamD <- oryx$MultivariateNormalFullCovariance(loc = globalx_mu_params_ODE,
                                                                covariance_matrix = globalSigmaMat)
        }
      }
  
      # KL divergences -- set to 0 for now
      KL_TERM <- jnp$array(0.)
      #KL_TERM <- oryx$kl_divergence(localParamD, localParamPrior_base)
      #KL_TERM <- jnp$add(KL_TERM, jnp$divide(oryx$kl_divergence(globalParamD,globalParamPrior_base), jnp$array(as.numeric(nBatch ))))
      # for KL, must specify separately for l + non-l-dep params
  
      testWithoutSampling <- isTRUE(get0("testWithoutSampling", ifnotfound = FALSE))
      if(testWithoutSampling){ warning("Testing with no sampling!")}
  
      # sample local
      if(!testWithoutSampling){ local_x_base_params_samp <- localParamD$sample(seed = jnp$add(seed, jnp$array(25L))) }
      if(testWithoutSampling){ local_x_base_params_samp <- localParamD$parameters$loc }
  
      # sample local neural
      if(!testWithoutSampling){ local_x_neural_params_samp <- localParamD_neural$sample(seed = jnp$add(seed, jnp$array(2435L))) }
      if(testWithoutSampling){ local_x_neural_params_samp <- localParamD_neural$parameters$loc }
  
      # check this
      if(any(diff(c(ref_i_neural,ref_i_base))!=1)){ stop("Stop: Is Assumed neural/Base ordering right?") }
      local_x_params_samp <- jnp$concatenate( list(local_x_neural_params_samp, local_x_base_params_samp) )
  
      # sample global parameters (consider incorporating one draw for entire batch)
      if(length(ArgNoDeps) > 0 && nGlobalParams > 0L){
        if(!testWithoutSampling){ global_x_params_samp <- globalParamD$sample(seed = jnp$add(seed, jnp$array(77245L)))  }
        if(testWithoutSampling){ global_x_params_samp <- globalParamD$parameters$loc }
      }

      # sample fixed local parameters
      if(nFixedLocal > 0){
        if(!testWithoutSampling){ fixedlocal_x_params_samp <- fixedLocalParamD$sample(seed = jnp$add(seed, jnp$array(295L))) }
        if(testWithoutSampling){ fixedlocal_x_params_samp <- fixedLocalParamD$parameters$loc }
      }
      
      # define structure of the neural networks 
      coreNeuralText <- "list('MLP' = ModelList$LocalNeural$MLP,
                              #'TSList' = ModelList$TSList, # input in all Transformer params
                              'ResidConProj' = ModelList$LocalNeural$ResidConProj,
                              'LNScaler' = ModelList$LocalNeural$LNScaler,
                              'NonLinearPathScaler' = SoftPlus(ModelList$LocalNeural$NonLinearPathScaler),
                              'MagDtScaler' = SoftPlus(ModelList$LocalNeural$MagDtScaler) )"
  
      # global evolution - we need to solve the first global ODE once to get starting values for all initial conditions of diff starting times 
      if(length(uq_globalneural_vec) == 0L){ dynamicglobal_x_params_samp <- jnp$array(0.) }
      dynamicglobal_x0_samp <- NULL; if(length(uq_globalneural_vec) > 0){
        #GlobalNeuralInitialCondInfoIndices <- (length(ModelList$GlobalNeural)-1):length(ModelList$GlobalNeural) - 1L
        #GlobalNeuralDynamicIndices <- 1:length(ModelList$GlobalNeural)
        #GlobalNeuralDynamicIndices <- GlobalNeuralDynamicIndices[!GlobalNeuralDynamicIndices %in% GlobalNeuralInitialCondInfoIndices]
        #y0 =  {send2cpu(list("Neural2"=ModelList$GlobalNeural[[ GlobalNeuralInitialCondInfoIndices[1] ] ]))},
  
        #f_VI_ODE_TERM_globalOnly(jnp$array(0.), tmpy, tmpargs) # t, y, args
        #plot( np$array(ModelList$GlobalNeural$NeuralInitialConditions) ) 
        dynamicglobal_x_params_samp <- diffrax$diffeqsolve(
                                           terms = VI_ODE_term_globalOnly,
                                           solver = VI_diff_eq_solver_optim,
                                           saveat = VI_SaveAt_ODE_GlobalNeural,
                                           # for structure of this, see definitions in transform_vec_final 
                                           args = {ndm_runtime_replicate_tree(eval(parse(text = sprintf('list("Neural2" = %s)',
                                                      gsub(coreNeuralText,pattern="LocalNeural",replace="GlobalNeural") ))))},
                                           y0 =  {ndm_runtime_replicate_tree(list("Neural2"=ModelList$GlobalNeural$NeuralInitialConditions ))},
                                           max_steps = MaxSteps,
                                           t0 = 0.,
                                           t1 = f2n(NTimeGlobalNeuralMax),
                                           dt0 = (dt0_init_optim),
                                           stepsize_controller = stepsize_controller_optim )
        dynamicglobal_x0_samp <- jnp$take(dynamicglobal_x_params_samp$ys$Neural2, time_indices, axis = 0L)
        # plot( np$array(dynamicglobal_x_params_samp$ys$Neural2)[,sample(1:100,1)] )
        # saveat is zero-based so time_indices line up directly with the solved path
      }
  
      # sampling parameters
      ODEParamsSampList_args <- Samp2Params(  
                                         localze = local_x_params_samp,
                                         fixedlocalze = fixedlocal_x_params_samp,
                                         globalze = global_x_params_samp,
                                         dynamicglobalze = dynamicglobal_x0_samp,
                                         dirichletparams = list( ( ModelList$ScaleList$ScaleBayes$DirichletScale1 ),
                                                                 ( ModelList$ScaleList$ScaleBayes$DirichletScale2 ) ), 
                                         context_ = context,
                                         tze = time_indices,
                                         seed = jnp$add(seed, jnp$array(4223131L))  )
      # explicit name for initial neural condition & drop NULLs
      names(ODEParamsSampList_args$Neural1_samp)[length(names(ODEParamsSampList_args$Neural1_samp))] <- "Neural1_0"
      ODEParamsSampList_args$Neural1_samp <- ODEParamsSampList_args$Neural1_samp[!unlist(lapply(ODEParamsSampList_args$Neural1_samp,is.null))]
  
      # divide args
      ODEParamsSampList_y0 <- ODEParamsSampList_args[c(paste(uq_y_vec,"_samp",sep=""),"Neural1_samp")]
      ODEParamsSampList_y0 <- ODEParamsSampList_y0[!is.na(names(ODEParamsSampList_y0))]
      ODEParamsSampList_y0$Neural1_samp <-  ODEParamsSampList_y0$Neural1_samp$Neural1_0
      ODEParamsSampList_args <- ODEParamsSampList_args[!names(ODEParamsSampList_args) %in% names(ODEParamsSampList_y0) |
                                                        grepl(names(ODEParamsSampList_args),pattern="Neural")]
      ODEParamsSampList_args$Neural1_samp$Neural1_0 <- NULL
  
      # scale initial states of neural ODE
      if(encneuralType == "l"){
          # insert initials
        ODEParamsSampList_y0$Neural1_samp <- jnp$concatenate(list(
  
              # hidden + bio init terms
              ODEParamsSampList_y0$Neural1_samp,
  
              # append sd init
              ySigma_init_nODE_untransformed), 0L)
      }
      if(encneuralType == "g" ){
        if( length(uq_encneural_vec) == 0 ){
          # scale only 
          ODEParamsSampList_y0$Neural1_samp <- jnp$concatenate(list(
            # scale local embedding terms
            jnp$multiply( jnp$take(ODEParamsSampList_y0$Neural1_samp,
                                   jnp$array(0L:(LocalNeuralEmbedDim-1L))),
                          SoftPlus(ModelList$LocalNeural$InitConScaler ) )
          ) )
        }
        if( length(uq_encneural_vec) > 0 ){
          # scale and append contents 
          ODEParamsSampList_y0$Neural1_samp <- jnp$concatenate(list(
              # scale local embedding terms
              jnp$multiply( jnp$take(ODEParamsSampList_y0$Neural1_samp,
                                     jnp$array(0L:(LocalNeuralEmbedDim-1L))),
                            SoftPlus(ModelList$LocalNeural$InitConScaler) ),
  
              # append local explicit term (e.g., beta)
              jnp$squeeze(jnp$take(ODEParamsSampList_y0$Neural1_samp,
                                   jnp$array(as.matrix(LocalNeuralEmbedDim:(LocalNeuralEmbedDim+length(uq_encneural_vec)-1L)))),1L)
            ) )
        }  

        # concatenate sigma terms
        #ODEParamsSampList_y0$Neural1_samp <- jnp$concatenate(list(
                                #  ODEParamsSampList_y0$Neural1_samp, # neural + param terms
                                #  ySigma_init_nODE_untransformed # sigma terms
                                #) )
        # Store the original before concatenating sigma
        Neural1_samp_before_sigma <- ODEParamsSampList_y0$Neural1_samp
        ODEParamsSampList_y0$Neural1_samp <- jnp$concatenate(
          lapply( list(ODEParamsSampList_y0$Neural1_samp, ySigma_init_nODE_untransformed),
                   function(x){ jnp$ravel(jnp$asarray(x, dtype = jnp$float32)) }
                 ), axis = 0L)
        # plot(np$array(NormFxn( ODEParamsSampList_y0$Neural1_samp$val$val ))[,sample(1:30,2)])
      }
      
  
      # setup final text for ODE transformations
      globalNeuralArgsText <- ""; if( length(uq_globalneural_vec)>0 ){
        # initial hidden states and scaling are last two states
        ModelList_GlobalNeural <- ModelList$GlobalNeural[ !(1:length(ModelList$GlobalNeural) %in% c(6,7)) ] # drop initial conditions at t=0, the initial scaling
        globalNeuralArgsText = gsub(coreNeuralText,pattern="LocalNeural",replace = "GlobalNeural")
        globalNeuralArgsText <- sprintf("'Neural2' = %s", globalNeuralArgsText)
        globalNeuralArgsText <- gsub(globalNeuralArgsText,
                                     pattern="ModelList\\[\\[ListIndices\\['GlobalNeural'\\]\\]\\]",
                                     replace="ModelList_GlobalNeural")
        ODEParamsSampList_y0 <- c(ODEParamsSampList_y0, "Neural2_samp" =  dynamicglobal_x0_samp)
      }
      neuralArgsText <- ifelse(encneuralType=="g",
        yes = sprintf("list('Neural1' = %s, %s", coreNeuralText, globalNeuralArgsText),
        no  = sprintf("list('Neural1' = ODEParamsSampList_args$Neural1_samp, %s ", globalNeuralArgsText ) )
      y0_names <- c(uq_y_vec[!uq_y_vec %in% uq_globalneural_vec],
                    ifelse(length(uq_globalneural_vec)>0, no = list(NULL), yes = list("Neural2"))[[1]] )
      y0_names <- c(y0_names[!y0_names %in%  uq_encneural_vec], "Neural1")
      
      # plot( colMeans( np$array( tmp$Neural1$val$val )[,1,] )  ) 
      # plot( apply( np$array( tmp$Neural1$val$val )[,1,], 2, sd )  ) 
      diff_eq_sol <- diffrax$diffeqsolve(terms = VI_ODE_term,
                                         solver = VI_diff_eq_solver_optim,
                                         args = 
                                          { 
                                          ndm_runtime_replicate_tree(
                                         c(eval(parse(text = paste(
                                           # dynamic params
                                           neuralArgsText,
  
                                           # conditional comma
                                           ifelse(length(uq_args_vec) > 0 & length(uq_globalneural_vec) > 0, yes = ",",no = ""),
  
                                           # non-dynamic params
                                           ifelse(length(uq_args_vec) > 0,
                                           yes = list(paste(sapply(c(uq_args_vec),function(ze){
                                                sprintf("'%s' = ODEParamsSampList_args$%s_samp",ze,ze) }),
                                                 collapse=",")),
                                            no = list(""))[[1]],
  
                                           # close group
                                            ")", collapse="")  )),
                                                "policy_scenario_indicator" = PolicyList[[1]],
                                                "policy_scenario" = PolicyList[[2]]))
                                        },
                                         y0 = 
                                          {  
                                            ndm_runtime_replicate_tree(
                                             eval(parse(text = paste("list(",paste(
                                           sapply(y0_names, function(ze){
                                           sprintf("'%s' = ODEParamsSampList_y0$%s_samp%s",
                                                  ze, ze,
                                                  ifelse(ze %in% uq_encneural_vec,
                                                         yes = paste("$",ze,"_0",sep=""),
                                                         no = "")
                                                  ) }),collapse=","),")",collapse="")  )))
                                          },
                                         max_steps = (MaxSteps),
                                         t0 = (jnp$array(0)), 
                                         t1 = GetPredSaveAtInfo[[1]], 
                                         saveat = GetPredSaveAtInfo[[2]],
                                         dt0 = dt0_init_optim,
                                         stepsize_controller = stepsize_controller_optim )
      # np$array(tmp$Neural1$val$val)[,1,]
      # does this speed up or slow down execution? TEST
      # diff_eq_sol$ys$e_l$devices() # analyze chip placement 
      
      # plot( np$array(diff_eq_sol$ys$i_l$val)[j_<-sample(1:40,1),] );
      # plot( np$array(diff_eq_sol$ys$Neural1$val)[1,,j_<-sample(1:40,1)] );
      # plot( np$array(diff_eq_sol$ys$Neural2$val)[1,,j_<-sample(1:40,1)] );
  
      # extract observed data mean
      if( length(ODEParamsSampList_args$p_l_samp$shape)==0 ){
        #ODEParamsSampList_args$p_l_samp <- ODEParamsSampList_args$p_l_samp *
        ODEParamsSampList_args$p_l_samp <- jnp$array(1.) * jnp$ones_like(diff_eq_sol$ys$i_l)
      }
      y_mean <- eval(parse(text = paste("jnp$concatenate(list(",
                          paste( paste("jnp$expand_dims(", observed_vec_final, ", 1L)"), collapse=","),"),1L)",collapse = "") ))
      # plot( c(np$array(y_mean$val$val)[,,,1]) )
      
      # sanity check - to confirm gradient passage through raw transformer part (x)
      #y_mean <- jnp$expand_dims(jnp$take(x, jnp$array(0L:(y_mean$shape[[1]]-1L))),1L) * jnp$ones_like(y_mean)
      #y_mean <- jnp$mean(ModelList[[8]][[6]],0L) * jnp$ones_like(y_mean)
  
      y_sigma <- 0.001+SoftPlus( jnp$take(diff_eq_sol$ys$Neural1,
                              jnp$array((diff_eq_sol$ys$Neural1$aval$shape[[2]]-nOutcomes):
                                  (diff_eq_sol$ys$Neural1$aval$shape[[2]]-1L)), axis = 1L) )
      if( nOutcomes == 1 ){ y_sigma <- jnp$expand_dims(y_sigma,1L) }
      }
  
      return_v <- list(list("y_mu" = y_mean,
                            "y_sigma" = y_sigma,
                            "KL_TERM" = KL_TERM,
                            "ODEParamsSampList" = c(ODEParamsSampList_args,
                                                    ODEParamsSampList_y0,
                                                    "center_param" = y_mean,
                                                    "scale_param" = y_sigma,
                                                    "diff_eq_sol_ts" = diff_eq_sol$ts,
                                                    "diff_eq_sol_ys" = diff_eq_sol$ys)),
                        state)
      return( return_v )
    }
    #,in_axes = eq$if_array(0L))( jax$tree_util$tree_flatten(  ModelList[[ListIndices["TSList"]]] )[[1]])
    if(T == F){ 
      if(nExperts == 1L){
          return_v <- jax$tree_util$tree_map(function(aer){jnp$squeeze(aer, axis = 0L)}, return_v)
      }
      if(nExperts > 1){
        WhereSelector <- 0L
        Selector <- jnp$take(jnp$take(return_v[[1]]$ODEParamsSampList$diff_eq_sol_ys.Neural1,
                             jnp$array(WhereSelector),
                             axis = 0L), jnp$array(0L:(nExperts - 1L)), axis = 1L)
        Selector <- jax$nn$softmax( Selector - jnp$max(Selector,1L, keepdims = T), 1L)
        Selector <- jnp$expand_dims(jnp$transpose(Selector),2L)
  
        return_v[[1]]$y_mu <- jnp$take(return_v[[1]]$y_mu, jnp$array(1L:(nExperts)), 0L)
        return_v[[1]]$y_sigma <- jnp$take(return_v[[1]]$y_sigma, jnp$array(1L:(nExperts)), 0L)
        return_v[[1]]$y_mu <- jnp$sum(return_v[[1]]$y_mu*Selector, 0L)
        return_v[[1]]$y_sigma <- jnp$sum(return_v[[1]]$y_sigma*Selector, 0L)
  
        for(r_ in 1:length(return_v[[1]]$ODEParamsSampList)){
          rShape <- length(return_v[[1]]$ODEParamsSampList[[r_ ]]$shape)
          if(rShape  == 2){
            return_v[[1]]$ODEParamsSampList[[r_]] <- jnp$sum(jnp$take(return_v[[1]]$ODEParamsSampList[[r_]],jnp$array(1L:(nExperts)), 0L) *
                                                               jnp$mean(Selector,1L), 0L)
          }
          if( rShape == 3 ){
            return_v[[1]]$ODEParamsSampList[[r_]] <- jnp$sum(jnp$take(return_v[[1]]$ODEParamsSampList[[r_]],jnp$array(1L:(nExperts)), 0L) *
                                                               Selector, 0L)
          }
        }
      }
    }

    return(   return_v   )
  }

  #############################################
  # setup key vectorized functions
  #############################################
  for(prior_sampling in unique(c(DoPriorSamplingAutoDiff,F))){
    GetPred_inference <- switch_filter_jit( jax$vmap(function(ModelList, x, state, PriorList, PolicyList, GetPredSaveAtInfo, seed){
      GetPred(ModelList, x, state, inference=T, PriorList=PriorList, PolicyList=PolicyList, GetPredSaveAtInfo=GetPredSaveAtInfo, seed=seed) },
      in_axes = list(NULL,0L,NULL,NULL,NULL,NULL,0L), # null indicates no vmap over that input
      out_axes = list(0L,0L), # null indicates no vmap over output
      axis_name = batch_axis_name )
      )
    GetPred_train_jit <- switch_filter_jit( GetPred_train <- jax$vmap(function(ModelList, x, state, PriorList, PolicyList, GetPredSaveAtInfo, seed){
      GetPred(ModelList, x, state, inference=F, PriorList=PriorList, PolicyList=PolicyList, GetPredSaveAtInfo=GetPredSaveAtInfo, seed=seed) },
        in_axes = list(NULL,0L,NULL,NULL,NULL,NULL,0L), # null indicates no vmap over parameters
        out_axes = list(0L,0L), # null indicates no vmap over output
        axis_name = batch_axis_name
      )
      )

    PriorListLocInGetLoss <- 5L
    getLoss_train <- function(ModelList, x, y, y_mask, 
                              i, state, PriorList, PolicyList, 
                              GetPredSaveAtInfo, seed){
        GetPred_output <- GetPred_train(ModelList, x, state, PriorList, PolicyList, GetPredSaveAtInfo, seed)
        state <- GetPred_output[[2]]; GetPred_output <- GetPred_output[[1]]
        {
          #y_mask <- y[[2]];
          #y <- y[[1]]
          # plot(  np$array(GetPred_output$y_mu)[sample(1:40,1),,1] )
          # plot(  np$array(GetPred_output$y_sigma)[sample(1:40,1),,1] )
          
          #likelihood_d <- oryx$Normal(loc = GetPred_output$ODEParamsSampList$center_param,
                                                      #scale = GetPred_output$ODEParamsSampList$scale_param)$log_prob( y )*y_mask
          likelihood_d <- jnp$negative((GetPred_output$ODEParamsSampList$center_param - y)^2) # loss with MSE 
          #likelihood_d <- jnp$negative( jnp$log(jnp$cosh(GetPred_output$ODEParamsSampList$center_param - y) )) # loss with log cosh (logcosh)
          
          # plot(np$array(GetPred_output$ODEParamsSampList$center_param)[5,,2])
          # hist(np$array(jnp$sum(likelihood_d,0L))[,1])

          # which(is.na(rowMeans(np$array(likelihood_d))))
          # minThis <- jnp$add(jnp$negative(jnp$sum(likelihood_d)), KL_div)
          minThis <- jnp$negative( jnp$sum(likelihood_d*y_mask$astype(likelihood_d$dtype)) /
                                     (0.001+jnp$sum(y_mask$astype(likelihood_d$dtype)) ) )
        }
        return( list(minThis, state)  )
      }

    GetPredSaveAtInfo_default <- ndm_runtime_normalize_getpred_saveat_info(
      list(as.integer(VI_TotalTimesInLikelihood), VI_SaveAt_ODE_optim)
    )
    gradLoss_jax <- switch_filter_jit( eq$filter_value_and_grad( getLoss_train, has_aux = T) )
    if(prior_sampling){
      gradLoss_jax <- switch_filter_jit( jax$value_and_grad(getLoss_train, ai(PriorListLocInGetLoss)) )
      GetPred_inference_prior_sampling <- GetPred_inference
      GetPred_train_prior_sampling <- GetPred_train
      gradLoss_jax_prior_sampling <- gradLoss_jax
      getLoss_train_prior_sampling <- getLoss_train

      # prior analysis step - must do this here to correctly initialize jitted fxn
      prior_grad_checks <- gradLoss_jax_prior_sampling(
        ModelList,
        jax_batchx,
        jax_batchy,
        jnp$array(1.),
        state,
        PriorList,
        PolicyList,
        GetPredSaveAtInfo_default,
        jax$random$split(JaxKey(ai(6L*SEED_)),nBatch))
      plot( np$asanyarray( prior_grad_checks[[2]][[1]]), cex = 0)
        text( np$asanyarray( prior_grad_checks[[2]][[1]]), labels = PriorDefinitions_jax[2,])
        image2( np$asanyarray( prior_grad_checks[[2]][[2]]) )
    }
  }
  predict_step_compiled <- GetPred_inference
  predict_train_step_compiled <- GetPred_train_jit
  loss_step_compiled <- switch_filter_jit(getLoss_train)
  if (exists("ndm_runtime_replicate_tree", inherits = TRUE)) {
    ModelList <- ndm_runtime_replicate_tree(ModelList)
    state <- ndm_runtime_replicate_tree(state)
    PriorList <- ndm_runtime_replicate_tree(PriorList)
    PolicyList <- ndm_runtime_replicate_tree(PolicyList)
    GetPredSaveAtInfo_default <- ndm_runtime_replicate_tree(GetPredSaveAtInfo_default)
  }
  }
  print("Done with this round of SuperLModel_BuildML.R!")
}
