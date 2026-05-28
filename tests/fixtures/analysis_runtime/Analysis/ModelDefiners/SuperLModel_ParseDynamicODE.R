# Content of SuperLModel_ParseDynamicODE.R
{
  print("Sarting SuperLModel_ParseDynamicODE.R")
  # Note: Neural1 -> Time evolving parameters like SD, beta_l
  #       Neural2 -> Global evolving parameters like gamma

  # clean environment
  gc();py_gc$collect()
  NTimeGlobalNeuralMax <- 200L

  # initializers
  NeuralODEClipAt <- jnp$array(100.)

  NeuralODE_InitScaler_SliceContext <- InvSoftPlus(  1 * 1/sqrt( LocalNeuralEmbedDim ) )$tolist()  
  NeuralODE_InitScaler_PandemicContext <- InvSoftPlus( 1 * 1/sqrt( GlobalNeuralEmbedDim ) )$tolist() 

  NeuralODE_NonLinearPathScaler_SliceContext <- InvSoftPlus(  1 * 1/sqrt( nTimesTotal)  )$tolist()
  NeuralODE_NonLinearPathScaler_PandemicContext <- InvSoftPlus( 1 * 1/sqrt( NTimeGlobalNeuralMax )  )$tolist()

  NeuralODE_OutputScaler_SliceContext <- InvSoftPlus( 1 * 1/sqrt(  nTimesTotal ) )$tolist()
  NeuralODE_OutputScaler_PandemicContext <- InvSoftPlus( 1 * 1/sqrt( NTimeGlobalNeuralMax ) )$tolist()

  PriorText <- unlist(read.table(model_tex_loc, header = F,  sep="\t"))
  names(PriorText) <- NULL
  PriorText <- unlist(strsplit(PriorText,split="\\n"))
  PriorText <- PriorText[PriorText!=""]

  start_i <- which(grepl(PriorText,pattern="%%%START BAYES ODE"))
  start_i <- start_i[length(start_i)]
  end_i <- which(grepl(PriorText,pattern="%%%END BAYES ODE"))[1]
  PriorText <- PriorText[start_i:end_i]
  PriorText <- PriorText[substr(PriorText,start=1,stop=1)!='%']
  PriorText <- strsplit(PriorText,split="\\$")
  PriorText <- c(na.omit(unlist(lapply(PriorText,function(zer){ c(zer,NA)[2] }))))
  ConstantsDefs <- PriorText[grep(PriorText,pattern="\\\\leftarrow")]
  ConstantsNames <- unlist(lapply(strsplit(ConstantsDefs,split="\\\\leftarrow"),function(l_){
    name_ <- paste(gsub(l_[[1]],pattern=" ",replace=""), sep = "")
  } ))

  # import code with global side effects
  lapply(strsplit(ConstantsDefs,split="\\\\leftarrow"),function(l_){
    name_ <- paste("CONST_", gsub(l_[[1]],pattern=" ",replace=""), sep = "")
    eval_ <- paste("jnp$array(f2n(", gsub(l_[[2]], pattern=" " ,replace =""), "))", collapse = "")
    eval.parent(parse(text=sprintf("%s <<- %s",name_,eval_)))
  })

  PriorDefinitions <- PriorText[grep(PriorText,pattern="\\\\sim")]
  TexTransformationRules <- PriorText[grep(PriorText,pattern="=")]
  TexTransformationRules <- gsub(TexTransformationRules,pattern="\\\\",replace="")
  nOutcomes_tex <- sum( grepl(TexTransformationRules, pattern = "Observe"))
  stopifnot( nOutcomes_tex == nOutcomes )

  PriorDefinitions <- gsub(PriorDefinitions,pattern="\\\\",replace="")
  PriorDefinitions_jax <- sapply(PriorDefinitions,function(zer){
    zer <- strsplit(zer,split = "sim")[[1]]

    # component 1 processing
    zer1 <- gsub(strsplit(zer[[1]],split="\\(")[[1]],pattern="\\)",replace="")
    if(length(zer1) == 1){zer1 <- c("Identity",zer1)}
    zer[[1]] <- list(zer1)

    # component 2 processing
    if(IsContext <- grepl(zer[2],pattern="Context_")){
      zer2_names <- strsplit(zer2 <- zer[[2]], split="_")[[1]][2]
      zer2_names <- gsub(zer2_names,pattern="textrm",replace="")
      zer2_names <- gsub(zer2_names,pattern="text",replace="")
      zer2_names <- gsub(zer2_names,pattern="\\{",replace="")
      zer2_names <- gsub(zer2_names,pattern="\\}",replace="")
      zer2_names_loc <- which(zer2_names == context_variables) - 1L # 0 indexed
      zer2_names_loc <- paste(zer2_names_loc,'L',sep="")
      if(grepl(zer2_names,pattern="_t")){
        zer2 <- eval(sprintf("jnp$take(context_, jnp$array(%s), axis = 1L)",zer2_names_loc))
      }
      if(!grepl(zer2_names,pattern="_t")){
        zer2 <- eval(sprintf("jnp$take(jnp$take(context_, jnp$array(%s), axis = 1L),0L,axis=0L)",zer2_names_loc))
      }
      zer1[1] <- zer2
      zer1[2] <- gsub(zer1[2],pattern=" ",replace="")
      zer2 <- c(NA,NA)
    }
    if(!IsContext){
      zer2 <- gsub(zer2 <- zer[[2]],
                   pattern="Dist\\{Normal\\}",
                   replace ="oryx$Normal" )
      zer2 <- gsub(gsub(".*\\((.*)\\).*", "\\1", zer2),pattern="\\)",replace="")
      zer2 <- unlist(sapply(strsplit(zer2,split=",")[[1]],function(zr){ eval(parse(text=zr)) }))
      zer2[1] <- ifelse(grepl(zer[[2]],pattern="gst\\("),
                        yes = Constrained2Unconstrained(target_mean_of_transformated_x = zer2[1],
                                                        transformation = eval(parse(text = gsub(zer1[1],pattern="Inv",replace=""))),
                                                        sd = zer2[2])$best_ex,
                        no = zer2[1])
    }
    ret_ <- try(c(zer1, zer2, IsContext),T)
    if("try-error" %in% class(ret_)){browser()}
    return( ret_ )
  })

  PriorDefinitions_jax <- rbind(PriorDefinitions_jax,
                                gsub(PriorDefinitions_jax[2,],pattern = "_\\{l0\\}",replace="_l"))
  PriorDefinitions_jax[nrow(PriorDefinitions_jax),] <- gsub(PriorDefinitions_jax[nrow(PriorDefinitions_jax),],pattern="\\}",replace = "")
  PriorDefinitions_jax[nrow(PriorDefinitions_jax),] <- gsub(PriorDefinitions_jax[nrow(PriorDefinitions_jax),],pattern="\\{",replace = "")
  PriorDefinitions_jax[nrow(PriorDefinitions_jax),] <- gsub(PriorDefinitions_jax[nrow(PriorDefinitions_jax),],pattern=" ",replace = "")

  # re-order rows not entries
  PriorDefinitions_jax <- PriorDefinitions_jax[c(1:4,6,5),]
  PriorDefinitions_jax[2,] <- gsub(PriorDefinitions_jax[2,],pattern=" ",replace = "")
  PriorDefinitions_jax[5,] <- gsub(PriorDefinitions_jax[5,],pattern=" ",replace = "")

  # define unique terms in vector field
  uq_y_vec <- gsub(( gsub(pattern="Evolve\\{", replace = "",
                        x = regmatches(TexTransformationRules,
                                       gregexpr("Evolve\\{.*?\\}", TexTransformationRules))  )), pattern = " ", replac = "")
  uq_y_vec <- gsub(uq_y_vec,pattern="\\}",replace = "")
  uq_y_vec <- uq_y_vec[uq_y_vec!="character(0)"]
  uq_encneural_vec <-  unique( uq_y_vec[grepl(TexTransformationRules,pattern="Evolve\\{.*_l\\}") &
                                          grepl(TexTransformationRules,pattern="Neural")] )
  uq_globalneural_vec <-  unique( uq_y_vec[grepl(TexTransformationRules,pattern="Evolve\\{.*\\}") &
                                          grepl(TexTransformationRules,pattern="Neural")] )
  uq_globalneural_vec <- uq_globalneural_vec[!uq_globalneural_vec %in% uq_encneural_vec]
  uq_allneural_vec <- c(uq_encneural_vec,uq_globalneural_vec)
  uq_y_vec <- unique( nonuq_y_vec <- uq_y_vec  )

  # define unique terms as named parameter inputs to vector field
  # (initial conditions are *not* defined here)
  uq_args_vec <- as.vector(PriorDefinitions_jax[5,])
  uq_args_vec <- gsub(uq_args_vec,pattern = "-",replace =" ")
  uq_args_vec <- gsub(uq_args_vec,pattern="frac\\{1",replace="")
  uq_args_vec <- gsub(uq_args_vec,pattern="\\}",replace="")
  uq_args_vec <- gsub(uq_args_vec,pattern="\\{",replace="")
  uq_args_vec <- gsub(uq_args_vec,pattern="\\)",replace="")
  uq_args_vec <- gsub(uq_args_vec,pattern="SoftPlus\\(",replace="")
  uq_args_vec <- gsub(uq_args_vec,pattern="\\(",replace="")
  uq_args_vec <- gsub(uq_args_vec,pattern = "cdot",replace =" ")
  uq_args_vec <- gsub(uq_args_vec,pattern = "\\+",replace =" ")
  uq_args_vec <- unique(unlist(strsplit(uq_args_vec,split=" ")))
  uq_args_vec <- uq_args_vec[!grepl(uq_args_vec,pattern="Evolve")]
  uq_args_vec <- uq_args_vec[uq_args_vec!=""]
  uq_args_vec <- uq_args_vec[(!uq_args_vec %in% uq_y_vec) & (!uq_args_vec %in% uq_encneural_vec)]
  uq_args_vec <- uq_args_vec[! uq_args_vec %in% ConstantsNames]
  uq_args_vec <- uq_args_vec[! (gsub(uq_args_vec,pattern="_lt",replace="_l") %in% uq_y_vec)]
  uq_args_vec <- uq_args_vec[! (grepl(uq_args_vec,pattern="Neural"))]
  uq_args_vec <- uq_args_vec[is.na(f2n(uq_args_vec))]

  # define time series
  temporalModelType <- "neuralODE";
  if(temporalModelType == "neuralODE"){
    TDependencyResolution <- nTimesTotal
    string_ <- TexTransformationRules[grepl(TexTransformationRules,pattern="Neural")]
    pattern_ <- "\\{([^{}]+)\\}"
    encneural_inputs <- c(); if(length(string_) > 0){
      matches_ <- gsub(regmatches(string_, gregexpr(pattern_, string_))[[1]][[2]],pattern="\\}",replace="")
      matches_ <- gsub(matches_,pattern="\\{",replace="")
      encneural_inputs <- gsub(unlist(strsplit(matches_, ", ")),pattern=" ", replace="")
    }
    if(encneuralType == "l"){
      nODE_loc_list_param_mean <- nODE_indices(nDimInput_nODE = (inputDim_nODE_local <- ai(length( encneural_inputs ) + LocalNeuralEmbedDim + nOutcomes   )),
                                          nWidth_nODE = nWidthODEHidden_ts_local,
                                          nDepth_nODE = nDepth_nODE,
                                          nDimOutput_nODE = (outputDim_nODE_local <- ai(length(uq_encneural_vec) + LocalNeuralEmbedDim + nOutcomes ) ))
      nDimODEOutput_ts_mean <- nDimODEOutput_ts <- ai( sum( unlist(  lapply(nODE_loc_list_param_mean,function(zer){
        nDim_ <- zer$shape; if(length(nDim_) == 0){nDim_<-1}; nDim_ }))) + nOutcomes  + LocalNeuralEmbedDim ) # experimental
    }
    if( length(uq_encneural_vec) == 0 ){ encneural_inputs <- c("s_l","e_l","i_l","r_l") }

    LocalNeuralMLP <- LocalNeuralLinear <- NULL
    GlobalNeural <- LocalNeural <- list( jnp$array(1.) )
    if(encneuralType == "g"){
      outputDim_nODE_local <- 1L # unused in g case
      nDimODEOutput_ts <- nDimODEOutput_ts_mean <- ai( LocalNeuralEmbedDim + length(uq_encneural_vec) + nOutcomes )
      
      LocalNeuralMLP <- list(
                          "WideProj1" = eq$nn$Linear(
                              in_features = inputDim_nODE_local <- 4L + 
                                       ai(LocalNeuralEmbedDim + 
                                           length(encneural_inputs[!encneural_inputs %in% c("s_l","e_l","i_l","r_l")]) +
                                          nOutcomes+2*grepl(model_tex_loc,pattern="DynamicBeta_DynamicGlobal")# understand why needed better 
                                          ),
                              out_features = ai(nWidthODEHidden_ts_local),
                              use_bias = F,
                              key = jax$random$PRNGKey(ai(4444L*34L + 5225L))),
                         "WideProj2" = eq$nn$Linear(
                              in_features = inputDim_nODE_local ,
                              out_features = nWidthODEHidden_ts_local,
                              use_bias = F,
                              key = jax$random$PRNGKey(ai(43*3443L + 5325L))),
                          "OutProj1" =  eq$nn$Linear(
                              in_features = nWidthODEHidden_ts_local ,
                              out_features = nDimODEOutput_ts_mean,
                              use_bias = finalBias_ODE,
                              key = jax$random$PRNGKey(ai(4111*34L + 5325L)) )
                          )
      LocalNeuralLinearResid <- eq$nn$Linear(
                                 in_features = inputDim_nODE_local,
                                 out_features = nDimODEOutput_ts_mean,
                                 use_bias = F,
                                 key = jax$random$PRNGKey(ai(43*34L + 55L)))
      if(ODE_NORM_TYPE == "none"){ NODE_LN_dims <- 1L }
      if(ODE_NORM_TYPE == "pre-activation1"){ NODE_LN_dims <- inputDim_nODE_local }
      if(ODE_NORM_TYPE == "pre-activation2"){ NODE_LN_dims <- nWidthODEHidden_ts_local }
      if(ODE_NORM_TYPE == "post-activation"){ NODE_LN_dims <- outputDim_nODE_local }
      LocalNeural <- list(
                          "MLP"=LocalNeuralMLP,
                          "ResidConProj"=LocalNeuralLinearResid,
                          "LNScaler"=jnp$array(rep(1,times=NODE_LN_dims)), # layer norm within neural ODE block

                          "NonLinearPathScaler"=jnp$array(rep(NeuralODE_NonLinearPathScaler_SliceContext, times = 1L)), 
                          "MagDtScaler"=jnp$array(rep(NeuralODE_OutputScaler_SliceContext, times = nDimODEOutput_ts_mean)), #SoftPlus applied to; controls scale of dt transformations
                          "InitConScaler"=jnp$array(rep(NeuralODE_InitScaler_SliceContext, times = LocalNeuralEmbedDim)) # last bc shifted; SoftPlus applied to; controls scale of initial cond
                          )
    }
    if(length(uq_globalneural_vec)>0){
      nDimODEOutput_ts_global <- (GlobalNeuralEmbedDim) + length( uq_globalneural_vec )
      
      GlobalNeuralMLP_global <- list(
        "WideProj1" = eq$nn$Linear(
            in_features = nDimODEInput_ts_global <- (GlobalNeuralEmbedDim+length(uq_globalneural_vec)),
            out_features = nWidthODEHidden_ts_global,
            use_bias = F,
            key = jax$random$PRNGKey(ai(4*34L + 56234L))),
        "WideProj2" = eq$nn$Linear(
            in_features = nDimODEInput_ts_global ,
            out_features = nWidthODEHidden_ts_global,
            use_bias = F,
            key = jax$random$PRNGKey(ai(43*3443L + 5325L))),
        "OutProj1" = eq$nn$Linear(
            in_features = nWidthODEHidden_ts_global,
            out_features = nDimODEOutput_ts_global,
            use_bias = finalBias_ODE,
            key = jax$random$PRNGKey(ai(4*34L + 332L)))
        )
      GlobalNeuralLinear_global <- eq$nn$Linear(
        in_features = nDimODEOutput_ts_global,
        out_features = nDimODEOutput_ts_global,
        use_bias = F, key = jax$random$PRNGKey(ai(4*34L + 544L)))
      GlobalNeuralLinearResid_global <- eq$nn$Linear(
        in_features = GlobalNeuralEmbedDim+length(uq_globalneural_vec),
        out_features = nDimODEOutput_ts_global,
        use_bias = F, key = jax$random$PRNGKey(ai(4334L + 5642345L))
      )

      # setup ODE
      VI_SaveAt_ODE_GlobalNeural <- diffrax$SaveAt(ts = jnp$array(  1:(NTimeGlobalNeuralMax )))
    }
  }
  if(temporalModelType == "linearInterpolation"){
    TDependencyResolutionFrac <- 0.25
    nDimODEOutput_ts <- TDependencyResolution <- round(nTimesTotal * TDependencyResolutionFrac)
    interp_ts = jnp$array(seq(0L,nTimesTotal,length.out = TDependencyResolution))
  }

  # process
  z <- transform_vec <- unlist(lapply(strsplit(TexTransformationRules,split = "="),function(zer){zer[2]}))
  transform_vec <- gsub(transform_vec, pattern="cdot", replace="\\*")
  transform_vec <- gsub(transform_vec, pattern = "frac\\{([^\\}]+)\\}\\{([^\\}]+)\\}",
                                       replace = "jnp$divide( \\1 , \\2 )")
  transform_vec <- gsub(transform_vec,pattern="[-]",replace=" - ")
  transform_vec <- gsub(transform_vec,pattern="[\\+]",replace=" \\+ ")
  transform_vec <- gsub(transform_vec,pattern="[*]",replace=" * ")
  transform_vec <- gsub(transform_vec,pattern="Evolve",replace="Evolve___")
  transform_vec <- gsub(transform_vec,pattern="Neural([[:alnum:]]+)",replace="Neural\\1___")
  transform_vec <- gsub(transform_vec,pattern="\\{",replace="")
  transform_vec <- gsub(transform_vec,pattern="\\}",replace="")
  transform_vec <- unlist(lapply(strsplit(transform_vec,split=" "),function(zer){
    if(length(uq_y_vec) > 0){
      tmp <- sapply(uq_y_vec,function(zerr){
        zer[zer == zerr]  <- eval(parse(text = sprintf("'YGOY%s'",zerr)))
        return( zer )  })
      zer <- apply(tmp,1,function(ra){ ra[which.max(nchar(ra))] })
    }

    if(length(uq_args_vec) > 0){
      tmp <- sapply(uq_args_vec,function(zerr){
        zer[zer == zerr]  <- eval(parse(text = sprintf("'ARGSGO%s'",zerr)))
        return( zer )  })
      zer <- try( apply(tmp,1,function(ra){ ra[which.max(nchar(ra))] }) , T)
    }

    if(length(ConstantsNames) > 0){
      tmp <- sapply(ConstantsNames,function(zerr){
        zer[zer == zerr]  <- eval(parse(text = sprintf("'CONSTSGO%s'",zerr)))
        return( zer )  })
    }

    zer <- try(apply(tmp,1,function(ra){ ra[which.max(nchar(ra))] }), T)

    paste(zer,collapse = " ")
  }))

  Transform2jnp <- function(text, r_transform = "\\*",
                            jnp_transform = "jnp$multiply"){
    text_ <- strsplit(text,split=r_transform)[[1]]
    if(length(text_) == 1){ }
    if(length(text_) > 1){
      ok_ <- F;while(ok_ == F){
        text_ <- Collapse2jnp(text_,jnp_transform = jnp_transform)
        if(length(text_) == 1){ok_ <- T}
      } }
    return(  text_ )
  }
  Collapse2jnp <- function(text__, jnp_transform = "jnp$multiply"){
    indices_ <- 1:length(text__)
    text_new <- text__[1]
    for(i in indices_){
      doNext <- try(text__[i+1],T)
      if("try-error" %in% class(doNext)){doNext <- F}
      if(is.na(doNext)){doNext <- F}
      if(is.character(doNext)){doNext <- T}
      if(doNext){
        text_new <- c(text_new,paste(jnp_transform, "(",text_new[i], ",", text__[i+1], ")",sep =""))
      }
    }
    return( text_new[length(text_new)]  )
  }
  transform_vec_final <- sapply(transform_vec, function(zer){
      # zer <- transform_vec[1]
      # deal with negative sign at the start
      zer_a <- strsplit(zer,split="-")[[1]]
      zer_b <- strsplit(zer,split="-\\(")[[1]]
      if( gsub(zer_a[1],pattern=" ",replace="") == "" &
          gsub(zer_b[1],pattern=" ",replace="") != "" ){
        zer <- strsplit(zer,split="\\*")[[1]]
        zer[1] <- gsub(zer[1],pattern="-",replace="jnp$negative(")
        zer[1] <- paste(zer[1],")",sep = "")
        zer <- paste(zer,collapse = " * ")
      }

      # reciprocol block - works only with one reciprocol!
      zer <- sapply(zer,function(rez){
        rez_replace_with <- qdapRegex::rm_between_multiple(rez,
                                                           left = "frac{1}{ ", right = " }",
                                                           extract = T)
        rez_replace_with <- paste("jnp$reciprocal(",rez_replace_with,")",collapse="")
        rez <- qdapRegex::rm_between_multiple(rez,
                                              left = "frac{1}{ ", right = " }",
                                              replace = " REPLACE_HERE ")
        rez <- gsub(rez,pattern="REPLACE_HERE",replace = rez_replace_with)
      })

      # multiplication with subtract blocks
      zer <- unlist(strsplit(zer,split="-"))
      if(length(zer) > 1){
        zer <- sapply(zer,function(rez){ Transform2jnp(rez,jnp_transform = "jnp$multiply")})
        zer <- Collapse2jnp(zer,jnp_transform = "jnp$subtract")
      }

      # multiplication within add block
      zer <- unlist(strsplit(zer,split="\\+"))
      if(length(zer) > 1){
        zer <- sapply(zer,function(rez){ Transform2jnp(rez,jnp_transform = "jnp$multiply")})
        zer <- Collapse2jnp(zer,jnp_transform = "jnp$add")
      }

      # residual multiplication block & return 
      return( sapply(zer,function(rez){ Transform2jnp(rez,jnp_transform = "jnp$multiply")}) ) 
  })
  names(transform_vec_final) <- NULL
  transform_vec_final <- gsub(transform_vec_final,pattern="ARGSGO",replace="args$")
  transform_vec_final <- gsub(transform_vec_final,pattern="YGOY",replace="y$")
  transform_vec_final <- gsub(transform_vec_final,pattern="CONSTSGO",replace="CONST_")
  if(temporalModelType == "linearInterpolation"){
    transform_vec_final <- gsub(transform_vec_final,pattern="_lt",replace="_l$evaluate(t)")
    transform_vec_final <- gsub(transform_vec_final,pattern="_t",replace="$evaluate(t)")
  }

  neural_key_vec <- ifelse(grepl("Neural1", transform_vec_final), "Neural1",
                   ifelse(grepl("Neural2", transform_vec_final), "Neural2",
                          ifelse(grepl("Neural3", transform_vec_final), "Neural3",
                                 ifelse(grepl("Neural4", transform_vec_final), "Neural4", NA))))

  if( length(uq_encneural_vec) == 0 ){
    #"Neural1___$s_l , y$e_l , y$i_l , y$r_l"
    # add SD evolution if no otherwise specified neural ODE term
    neural_key_vec <- c(neural_key_vec, "Neural1")
    transform_vec_final <- c(transform_vec_final, "Neural1___ $s_l , y$e_l , y$i_l , y$r_l")
  }
  transform_vec_final_split <- strsplit(transform_vec_final, split = "Neural([[:alnum:]]+)___")
  transform_vec_final <- unlist(sapply(1:length(transform_vec_final_split), function(i_){
    l_ <- transform_vec_final_split[[i_]]
    if(length(l_) == 1){ ret_l <- l_ }

    if(length(l_) > 1){
      if(neural_key_vec[i_] == "Neural1"){
        YARGS_NEURAL_TEXT_PARTIAL_0 <- c("y$Neural", paste("jnp$divide(y$",
                                                           c("s_l","e_l","i_l","r_l"),
                                                           ",CONST_N) - 1/4",sep=""))
        if(length(encneural_inputs[!encneural_inputs %in% c(uq_encneural_vec,"s_l","e_l","i_l","r_l")]) > 0){ 
          YARGS_NEURAL_TEXT_PARTIAL_0 <- c(YARGS_NEURAL_TEXT_PARTIAL_0, 
                                           paste(
                                           "jnp$take(y$XXX2 / sqrt(LocalNeuralEmbedDim), ai(GlobalNeuralEmbedDim+",
                                           1:length(encneural_inputs[!encneural_inputs %in% uq_encneural_vec]),"L",
                                           "-1L))" , sep = "") ) 
        }
        YARGS_NEURAL_TEXT_PARTIAL_0 <- c(YARGS_NEURAL_TEXT_PARTIAL_0)
        YARGS_NEURAL_TEXT_PARTIAL_0[-1] <- paste("jnp$expand_dims(", YARGS_NEURAL_TEXT_PARTIAL_0[-1], ",0L)")

        YARGS_NEURAL_TEXT <- paste("jnp$concatenate(list(", YARGS_NEURAL_TEXT_PARTIAL_1 <- paste(
          YARGS_NEURAL_TEXT_PARTIAL_0, collapse=","),"))",collapse="")
      }
      if(neural_key_vec[i_] == "Neural2"){ YARGS_NEURAL_TEXT_PARTIAL_1 <- YARGS_NEURAL_TEXT <- "y$Neural" }

      if(encneuralType == "l"){
         ret_l <- sprintf("f_neuralODE_R(t=t, y=%s, args=args$%s )", YARGS_NEURAL_TEXT, "Neural")
      }
      
      if(encneuralType == "g" | neural_key_vec[i_] == "Neural2"){
          # args$Neural$NonLinearPathScaler is ln layer
          # note: this normalization choice only affects globally defined terms
        if(ODE_NORM_TYPE == "none"){  # no normalization
            ret_l <- sprintf("jnp$squeeze(
                  args$Neural$MLP$OutProj1(
                        jax$nn$swish( args$Neural$MLP$WideProj1(jnp$concatenate(list(%s))) )  ) )
                               ",YARGS_NEURAL_TEXT_PARTIAL_1)
        }
        if(ODE_NORM_TYPE == "pre-activation1"){  # pre-activation normalization
          if(T == F){ # swish
              ret_l <- sprintf("jnp$squeeze( 
             args$Neural$MLP$OutProj1(
                 jax$nn$swish( args$Neural$MLP$WideProj1(
                     raw_xt <- (args$Neural$LNScaler * NormFxn(jnp$concatenate(list(%s)))) )) ) )",YARGS_NEURAL_TEXT_PARTIAL_1)
          }
           if(T == T){ # swiglu
           ret_l <- sprintf("jnp$squeeze( 
             args$Neural$MLP$OutProj1(
                 jax$nn$swish( args$Neural$MLP$WideProj1(
                     raw_xt <- (args$Neural$LNScaler * NormFxn(jnp$concatenate(list(%s)))) )) * 
                        args$Neural$MLP$WideProj2(raw_xt)  ) )",YARGS_NEURAL_TEXT_PARTIAL_1)
           }
        }
        if(ODE_NORM_TYPE == "pre-activation2"){  # pre-activation normalization with swish
          ret_l <- sprintf("jnp$squeeze( args$Neural$MLP$OutProj1( jax$nn$swish( (args$Neural$LNScaler*NormFxn(args$Neural$MLP$WideProj1(jnp$concatenate(list(%s))))))  ) )",YARGS_NEURAL_TEXT_PARTIAL_1)
        }
        if(ODE_NORM_TYPE == "post-activation"){  # post activation normalization
            ret_l <- sprintf("jnp$squeeze( (args$Neural$LNScaler * NormFxn( args$Neural$MLP$OutProj1(jax$nn$swish(args$Neural$MLP$WideProj1(jnp$concatenate(list(%s)))) ))))",YARGS_NEURAL_TEXT_PARTIAL_1)
          }

          ret_l_resid <- sprintf("args$Neural$ResidConProj(jnp$concatenate(list(%s)))", YARGS_NEURAL_TEXT_PARTIAL_1)
          ret_l <- sprintf("jnp$clip(
                           (%s + args$Neural$NonLinearPathScaler * %s)*args$Neural$MagDtScaler, 
                           -NeuralODEClipAt, NeuralODEClipAt)",ret_l_resid, ret_l)
      }
      
      ret_l <- gsub(ret_l, pattern = "\\$Neural", replace = paste("\\$",neural_key_vec[i_], sep = "") )
      ret_l <- gsub(ret_l, pattern ="XXX2", replace = "Neural2")
    }
    if( length(uq_encneural_vec) > 0 ){
      uq_counter <- 0L; for(uq__ in uq_encneural_vec){
        uq_counter <- uq_counter + 1L
        ret_l <- gsub(ret_l, pattern =  sprintf('y\\$%s',uq_encneural_vec[uq_counter]), replacement = sprintf('jnp$take(y$%s, LocalNeuralEmbedDim + %sL)',"Neural1",  - 1L + uq_counter ))
      }
    }
   return(ret_l) }))

  # define final ODE field
  Neural2_vec <- nonuq_y_vec[ neural_key_vec %in% c("Neural2") ]
  transform_vec_final <- gsub(transform_vec_final, pattern = "args\\$(?=[0-9])", replace="", perl = T)
  transform_vec_final[ !uq_y_vec %in% uq_encneural_vec ] <- sapply(transform_vec_final[ !uq_y_vec %in% uq_encneural_vec ],
   function(zer){
        if(length(Neural2_vec) > 0){
           for(ja in 1:length(Neural2_vec)){
            zer <- gsub(zer,pattern =  sprintf('y\\$%s',Neural2_vec[ja]), replacement = sprintf('jnp$take(y$Neural2, ai(LocalNeuralEmbedDim+%s-1L))',ja))
          } }
        return( zer )
   } )

  # define left side
  lefthandside_vec <- uq_y_vec
  lefthandside_vec[lefthandside_vec %in% uq_encneural_vec] <- "Neural1"
  if(!"Neural1" %in% lefthandside_vec){
    lefthandside_vec <- c(lefthandside_vec,"Neural1")
    TexTransformationRules <- c(TexTransformationRules,"Neural1")
  }
  lefthandside_vec[lefthandside_vec %in% uq_globalneural_vec] <- "Neural2"

  # define right side
  righthandside_vec <- transform_vec_final[!grepl(TexTransformationRules,pattern = "Observe")]
  if(length(lefthandside_vec) != length(righthandside_vec)){ print("left right here"); browser() }

  # combine left and right
  ode_pairs <- data.frame(
    lefthandside_vec = lefthandside_vec,
    righthandside_vec = righthandside_vec,
    stringsAsFactors = FALSE
  )
  ode_pairs <- ode_pairs[!duplicated(ode_pairs), , drop = FALSE]
  lefthandside_vec <- ode_pairs$lefthandside_vec
  righthandside_vec <- ode_pairs$righthandside_vec
  ode_transformation_text <- paste(paste("'", lefthandside_vec, "'=", righthandside_vec, sep = ""),collapse = ",")

  #ode_transformation_text <- gsub(ode_transformation_text, pattern = uq_encneural_vec, replace = "Neural1")
  VI_ODE_term <- eval(parse(text = sprintf("diffrax$ODETerm(  f_VI_ODE_TERM <- function(t, y, args){
            # browser();

            # evolve ODE
            y <- list( %s )

            # perform surgery
            if('p_l' %%in%% uq_encneural_vec & T == T){
              p_l_vector_loc <- jnp$array( LocalNeuralEmbedDim - 1L + which(uq_encneural_vec == 'p_l') );
              y$Neural1 <- y$Neural1$at[
                           p_l_vector_loc
                          ]$set(  jnp$add(jnp$multiply(
                                          jnp$take(y$Neural1, p_l_vector_loc),
                                          jnp$subtract(1,args$policy_scenario_indicator$evaluate(t)) ),
                                    jnp$multiply(args$policy_scenario$evaluate(t),
                                          args$policy_scenario_indicator$evaluate(t) ) ) );
            }

            return( y  ) }   )", ode_transformation_text )))

  VI_ODE_term_globalOnly <- NULL
  if( length(righthandside_vec[lefthandside_vec=="Neural2"]) > 0   ){
    VI_ODE_term_globalOnly <- eval(parse(text = sprintf("diffrax$ODETerm(  f_VI_ODE_TERM_globalOnly <- function(t, y, args){
              # print('exploring globalOnly'); browser();
              return( y <- list( %s ) ) }   )", paste(paste("'", lefthandside_vec[lefthandside_vec=="Neural2"],
                    "'=", righthandside_vec[lefthandside_vec=="Neural2"], sep = ""),collapse = ",") )))
  }

  # adjust for different neural embedding dims in Neural2 (really is NeuralGlobal)
  transform_vec_final <- gsub(transform_vec_final,
                              pattern = "jnp\\$take\\(y\\$Neural2, ai\\(LocalNeuralEmbedDim",
                              replace = "jnp$take(y$Neural2, ai(GlobalNeuralEmbedDim")

  # observed outcome transforms
  observed_name <- transform_vec_final
  observed_name <- observed_name[grepl(TexTransformationRules, pattern = "Observe")]
  observed_name <- lapply(observed_name,function(l_){
                  return( l_ <- gsub(l_,pattern="y\\$",replace = "diff_eq_sol\\$ys\\$") ) })
  observed_vec_final <- unlist(observed_name)
  observed_vec_final <- gsub(observed_vec_final, pattern="args\\$", replace = "ODEParamsSampList_args\\$")
  for(z_ in unique( uq_args_vec )){
    observed_vec_final <- gsub(
      observed_vec_final,
      pattern = sprintf("\\$%s(?![[:alnum:]_])", z_),
      replace = sprintf("\\$%s_samp", z_),
      perl = TRUE
    )
  }
  observed_vec_final <- gsub(observed_vec_final,pattern=" ", replace = "")
  observed_vec_final <- gsub(observed_vec_final,
                             pattern = "(jnp\\$take\\(diff_eq_sol\\$ys\\$Neural1,)([^)]+)\\)",
                             replace = "\\1\\2,axis=1L)") # expand dims for extraction
  if(!"delta" %in% uq_globalneural_vec){ 
    observed_vec_final <- gsub(observed_vec_final,pattern = "Sigmoid",
                               replace="") # FIND MORE GENERAL WAY TO HANDLE CONSTRAINTS HANDLED IN MULTIPLE WAYS 
  }
  transform_vec_final <- transform_vec_final[!grepl(TexTransformationRules,pattern = "Observe")]

  # define dependencies
  LDeps <- unique(c( grep(uq_args_vec, pattern="_l", value=T),grep(uq_y_vec, pattern="_l", value=T)))
  TDeps <- unique(c(grep(uq_args_vec, pattern="_lt", value=T),
                    grep(uq_args_vec, pattern="_t", value=T),
                    uq_allneural_vec,
                    grep(uq_y_vec, pattern="_lt", value=T),
                    grep(uq_y_vec, pattern="_t", value=T)))
  NoDeps <- c(uq_args_vec,uq_y_vec)[!c(uq_args_vec,uq_y_vec) %in% c(LDeps,TDeps)]

  # args defns
  ArgNoDeps <- NoDeps[NoDeps %in% c(uq_args_vec,uq_allneural_vec)]
  ArgLDeps <- LDeps[LDeps %in% c(uq_args_vec,uq_allneural_vec)]
  ArgTDeps <- TDeps[TDeps %in% c(uq_args_vec,uq_allneural_vec)]

  # setup args vec
  VI_param_terms <- c(uq_args_vec, uq_y_vec)
  VI_param_terms <- sapply(1:length(VI_param_terms),function(zr){
    TDep_ <- VI_param_terms[zr] %in% TDeps
    TEncNeural <- VI_param_terms[zr] %in% uq_allneural_vec
    if(TDep_ ){ret_ <- replicate( LocalNeuralEmbedDim+1L, VI_param_terms[zr]) }
    if(!TDep_){ret_ <- c(VI_param_terms[zr]) }
    return( ret_ ) })
  if("list" %in% class(VI_param_terms)){
    VI_param_terms <- do.call(c, VI_param_terms)
  }

  # setup big helper matrices
  PriorDefinitions_jax_orig <- PriorDefinitions_jax
  type_counter <- 0; for(type_ in c("Neural","Prior")){
  type_counter <- type_counter + 1
  PriorDefinitions_jax <- sapply(1:ncol(PriorDefinitions_jax_orig),function(zr){
    TDep_ <- PriorDefinitions_jax_orig[5,zr] %in% TDeps
    if(TDep_){ ret_ <- replicate(ifelse(type_ == "Prior",yes = TDependencyResolution, no = nDimODEOutput_ts),
                                 PriorDefinitions_jax_orig[,zr]) }
    if(!TDep_){ ret_ <- as.matrix(PriorDefinitions_jax_orig[,zr]) }
     return( ret_ )
  })
  if("list" %in% class(PriorDefinitions_jax)){
    PriorDefinitions_jax <- do.call(cbind,PriorDefinitions_jax)
  }
  PriorDefinitions_jax <- rbind(PriorDefinitions_jax,

                                # argIndicator
                                PriorDefinitions_jax[5,] %in% unique(uq_args_vec),

                                # LDep
                                PriorDefinitions_jax[5,] %in% LDeps,

                                # TDep
                                PriorDefinitions_jax[5,] %in% TDeps,

                                # NoDep
                                PriorDefinitions_jax[5,] %in% NoDeps
                                )
  row.names(PriorDefinitions_jax) <- c("Transformation","DefOrig",
                                       "PriorMean","PriorSD","DefClean","IsContext",
                                       "argIndicator","LDep","TDep","NoDep")

  # add ODE term if needed (for variance)
  if(length( uq_encneural_vec ) == 0){
    ADDON_ <- PriorDefinitions_jax[,1:2]
    ADDON <- c(); for(rer in 1:(LocalNeuralEmbedDim)){
      addon_mean <- c("Transformation" = "Identity",
                      "DefOrig" = "Neural1", 
                      "PriorMean" = "0", "PriorSD"="1",
                      "DefClean" = "Neural1",
                      "IsContext" = "FALSE",
                      "argIndicator" = "FALSE",
                      "LDep" = "TRUE",
                      "TDep" = "TRUE", 
                      "NoDep"="FALSE")
      if(any(names(addon_mean) != row.names( ADDON_ ))){stopping("names(addon_mean) != row.names( ADDON_ ) in ParseDynamicODE.R")}
      ADDON <- cbind(ADDON, addon_mean)
    }
    PriorDefinitions_jax <- cbind(ADDON, PriorDefinitions_jax)
  }

  # add the SEIR power weights
  # initial conditions mixture
  for(add_k in 1L:(nSoftMaxMix <- 1)){ # one of these is defined implicitly in the above!
    if(add_k < nSoftMaxMix){
      #PowerTypeInvTransform <- "SoftPlus"; PowerWtInvTransform <- InvSoftPlus_r;
      PowerTypeInvTransform <- "Identity"; PowerWtInvTransform <- function(x){x};
      EIRM <- SM <- 0; Width <- 0.001
      PowerWeightAdds <- cbind(c(PowerTypeInvTransform, "s_{l0}",
                  as.character(PowerWtInvTransform(   rnorm(1, SM, 0.1*SM )   )), "1",
                            "s_l", "FALSE","FALSE","TRUE","FALSE","FALSE"),
                          c(PowerTypeInvTransform, "e_{l0}",as.character(PowerWtInvTransform( rnorm(1,EIRM,1*Width) )), "1",
                            "e_l", "FALSE","FALSE","TRUE","FALSE","FALSE"),
                          c(PowerTypeInvTransform, "i_{l0}",as.character(PowerWtInvTransform( rnorm(1,EIRM,1*Width) )), "1",
                            "i_l", "FALSE","FALSE","TRUE","FALSE","FALSE"),
                          c(PowerTypeInvTransform, "r_{l0}",as.character(PowerWtInvTransform( rnorm(1,EIRM,1*Width) )), "1",
                            "r_l", "FALSE","FALSE","TRUE","FALSE","FALSE") )
    }
    if(add_k == nSoftMaxMix){ # temperature or master mixture controller parameters
      PowerTypeInvTransform <- "Identity"; PowerWtInvTransform <- function(x){x};
      PowerWeightAdds <- replicate(max(c(4,nSoftMaxMix)), # max 4 so has at least as many as SEIR for wts transforms 
                                  c(PowerTypeInvTransform, "s_{l0}",as.character(PowerWtInvTransform(  0  )), "1",
                                        "s_l", "FALSE","FALSE","TRUE","FALSE","FALSE"))
    }
    PriorDefinitions_jax <- cbind(PriorDefinitions_jax, PowerWeightAdds)
  }

  # setup entries
  PriorDefinitions_jax_CONTEXT <- PriorDefinitions_jax[,is.na(PriorDefinitions_jax[3,])]
  if("character" %in% class( PriorDefinitions_jax_CONTEXT)){PriorDefinitions_jax_CONTEXT<-as.matrix(PriorDefinitions_jax_CONTEXT)}
  PriorDefinitions_jax_STOCHASTIC <- PriorDefinitions_jax[,!is.na(PriorDefinitions_jax[3,])]
  VI_param_terms <- VI_param_terms[! VI_param_terms %in% PriorDefinitions_jax_CONTEXT[2,]]

  # all
  VariationalInitAllMean <- f2n(PriorDefinitions_jax_STOCHASTIC["PriorMean",])
  VariationalInitAllSD <- f2n(PriorDefinitions_jax_STOCHASTIC["PriorSD",])
  names(VariationalInitAllSD) <- names(VariationalInitAllMean) <- gsub(PriorDefinitions_jax_STOCHASTIC["DefClean",],pattern="_0",replace="")

  # evolving local parameters
  EncodedIndi <- as.logical(PriorDefinitions_jax_STOCHASTIC["LDep",])
  VariationalInitEncLocalMean <- f2n(PriorDefinitions_jax_STOCHASTIC["PriorMean",])[EncodedIndi]
  VariationalInitEncLocalSD <- f2n(PriorDefinitions_jax_STOCHASTIC["PriorSD",])[EncodedIndi]
  names(VariationalInitEncLocalSD) <- names(VariationalInitEncLocalMean) <- gsub(PriorDefinitions_jax_STOCHASTIC["DefClean",],pattern="_0",replace="")[EncodedIndi]

  # non-evolving non-temporal local parameters (these are all args)
  FixedLocalParams_defns <- grep(uq_args_vec, pattern="_l$",value = T)
  VariationalInitNonEncLocalMean <- VariationalInitAllMean[  FixedLocalParams_defns  ]
  VariationalInitNonEncLocalSD <- VariationalInitAllSD[ FixedLocalParams_defns  ]

  # non-evolving temporal global parameters (these are all args, too)
  NonEncGlobalParams_defns <- uq_globalneural_vec#grep(uq_args_vec,pattern="_t$",value = T)
  VariationalInitTemporalGlobalMean <- VariationalInitAllMean[  names(VariationalInitAllMean) %in% NonEncGlobalParams_defns  ]
  VariationalInitTemporalGlobalSD <- VariationalInitAllSD[ names(VariationalInitAllSD) %in% NonEncGlobalParams_defns  ]
  
  # distinguish fixed from evolving local variables 
  VariationalInitEncLocalMean <- VariationalInitEncLocalMean[! names(VariationalInitEncLocalMean) %in% FixedLocalParams_defns  ]
  VariationalInitEncLocalSD <- VariationalInitEncLocalSD[ !names(VariationalInitEncLocalSD) %in% FixedLocalParams_defns  ]

  # non-evolving non-temporal global parameters
  NoDepIndi <- as.logical(PriorDefinitions_jax_STOCHASTIC["NoDep",])==T
  VariationalInitGlobalMean <- f2n(PriorDefinitions_jax_STOCHASTIC["PriorMean",])[NoDepIndi]
  VariationalInitGlobalSD <- f2n(PriorDefinitions_jax_STOCHASTIC["PriorSD",])[NoDepIndi]
  names(VariationalInitGlobalMean) <- names(VariationalInitGlobalSD) <- gsub(PriorDefinitions_jax_STOCHASTIC["DefClean",],pattern="_0",replace="")[NoDepIndi]

  # reset initializations of neural parameters
  if(type_ == "Neural"){
    if(length( VariationalInitTemporalGlobalMean ) > 0){
      VariationalInitTemporalGlobalMean <- c(rnorm(GlobalNeuralEmbedDim,sd  =  1/sqrt(GlobalNeuralEmbedDim)),
                                             VariationalInitTemporalGlobalMean[!duplicated(VariationalInitTemporalGlobalMean)])
      VariationalInitTemporalGlobalSD <- VariationalInitTemporalGlobalMean
      VariationalInitTemporalGlobalSD[] <- 1
      GlobalNeuralTakes <- which(names(VariationalInitTemporalGlobalMean)!="")-1L
      names(GlobalNeuralTakes) <- unique(names(VariationalInitTemporalGlobalMean))[-1]

      if(ODE_NORM_TYPE == "none"){ NODE_LN_dims <- 1L }
      if(ODE_NORM_TYPE == "pre-activation1"){ NODE_LN_dims <- nDimODEInput_ts_global }
      if(ODE_NORM_TYPE == "pre-activation2"){ NODE_LN_dims <- nWidthODEHidden_ts_global }
      if(ODE_NORM_TYPE == "post-activation"){ NODE_LN_dims <- nDimODEOutput_ts_global }
      GlobalNeural <- list(
                                # evolution parameters
                                "MLP" = GlobalNeuralMLP_global,  #
                                "ResidConProj" = GlobalNeuralLinearResid_global,  #
                                "LNScaler" = jnp$array(rep(1,times=NODE_LN_dims)), # layer norm within neural ODE block
                                
                                "NonLinearPathScaler" = jnp$array(rep(NeuralODE_NonLinearPathScaler_PandemicContext, times = 1L)), #
                                "MagDtScaler" = jnp$array(rep(NeuralODE_OutputScaler_PandemicContext, times = nDimODEOutput_ts_global)), # SoftPlus applied to; controls scale of dt transformations
                                
                                "InitConScaler" = jnp$array(rep(NeuralODE_InitScaler_PandemicContext, times = nDimODEOutput_ts_global)), #NB: last bc shifted. SoftPlus applied to; controls scale of augmented embeddings, (not in ode fxn)
                                "NeuralInitialConditions" = jnp$array( VariationalInitTemporalGlobalMean ) # initial values
                              )
    }

    # perhaps take the previous strategy to the following:
    VariationalInitEncLocalMean[
                        names(VariationalInitEncLocalMean) %in% TDeps &  # neural mean = 0
                          rev(duplicated(rev(names(VariationalInitEncLocalMean)))) # except for initial cond
                        ] <- 0
    VariationalInitEncLocalSD[ names(VariationalInitEncLocalSD) %in% TDeps ] <- 0.1
    VariationalInitGlobalMean[ names(VariationalInitGlobalMean) %in% TDeps ] <- 0
    VariationalInitGlobalSD[ names(VariationalInitGlobalSD) %in% TDeps ] <- 1
  }

    # prior sampling 
    PriorList <- list() 
    if(T == F){
    PriorList <- list(
      # locals
      list(jnp$array(f2n(VariationalInitEncLocalMean)),
           jnp$array(diag(f2n(VariationalInitEncLocalSD^2)))),
  
      # globals
      list(jnp$array(f2n(VariationalInitGlobalMean)),
           jnp$array(
             ifelse(length(VariationalInitGlobalSD) == 1,
                    yes = list(as.matrix(VariationalInitGlobalSD^2)),
                    no = list(diag(f2n(VariationalInitGlobalSD)^2) ))[[1]]
      )))
    }

    # rename as desired
    if(type_ == "Neural"){
      VariationalInitTemporalGlobalMean <- VariationalInitTemporalGlobalMean
      VariationalInitTemporalGlobalSD <- VariationalInitTemporalGlobalSD

      NeuralVariationalInitEncLocalMean <- VariationalInitEncLocalMean
      NeuralVariationalInitEncLocalSD <- VariationalInitEncLocalSD

      NeuralVariationalInitGlobalMean <- VariationalInitGlobalMean
      NeuralVariationalInitGlobalSD <- VariationalInitGlobalSD

      NeuralDefinitions_jax_CONTEXT <- PriorDefinitions_jax_CONTEXT;
      NeuralDefinitions_jax_STOCHASTIC <- PriorDefinitions_jax_STOCHASTIC
      NeuralDefinitions_jax_STOCHASTIC <- NeuralDefinitions_jax_STOCHASTIC[,(!duplicated(NeuralDefinitions_jax_STOCHASTIC["DefClean",]) &
                            NeuralDefinitions_jax_STOCHASTIC["DefClean",] %in% uq_globalneural_vec) | (!NeuralDefinitions_jax_STOCHASTIC["DefClean",] %in% uq_globalneural_vec)]
    }
  }

  # setup prior mean
  if(length(VariationalInitNonEncLocalMean) > 0){
    fixedLocalParamPrior_base <- oryx$Normal( jnp$array( VariationalInitNonEncLocalMean ),
                                                                       jnp$array(VariationalInitNonEncLocalSD) )
  }
  if(length(VariationalInitEncLocalMean) > 0){
    localParamPrior_base <- oryx$MultivariateNormalDiag( jnp$array(VariationalInitEncLocalMean[!names(VariationalInitEncLocalMean) %in% TDeps] ),
                                                                jnp$array(VariationalInitEncLocalSD[!names(VariationalInitEncLocalSD) %in% TDeps]) )
  }
  if(length(VariationalInitGlobalMean[!names(VariationalInitGlobalMean) %in% TDeps]) > 0){
    globalParamPrior_base <- oryx$MultivariateNormalDiag( jnp$array(VariationalInitGlobalMean[!names(VariationalInitGlobalMean) %in% TDeps]),
                                                                 jnp$array(ifelse(length(VariationalInitGlobalSD[!names(VariationalInitGlobalMean) %in% TDeps])==1,
                                                                                  yes = list(as.matrix(VariationalInitGlobalSD[!names(VariationalInitGlobalMean) %in% TDeps])),
                                                                                  no = list(VariationalInitGlobalSD[!names(VariationalInitGlobalMean) %in% TDeps]))[[1]] ))
  }
  #- define four priors (neural + non-neural for local + global )
  if(length(VariationalInitEncLocalMean[names(VariationalInitEncLocalMean) %in% TDeps] ) > 0){
    localParamPrior_neural <- oryx$Normal( jnp$array(VariationalInitEncLocalMean[names(VariationalInitEncLocalMean) %in% TDeps] ),
                                                                scale = jnp$array(VariationalInitEncLocalSD[names(VariationalInitEncLocalSD) %in% TDeps]) )
  }
  if(length(VariationalInitGlobalMean[names(VariationalInitGlobalMean) %in% TDeps]) > 0){
    globalParamPrior_neural <- oryx$Normal( jnp$array(VariationalInitGlobalMean[names(VariationalInitGlobalMean) %in% TDeps]),
                                                        scale =  jnp$array(ifelse(length(VariationalInitGlobalSD[names(VariationalInitGlobalMean) %in% TDeps])==1,
                                                                                         yes = list(as.matrix(VariationalInitGlobalSD[names(VariationalInitGlobalMean) %in% TDeps])),
                                                                                         no = list(VariationalInitGlobalSD[names(VariationalInitGlobalMean) %in% TDeps]))[[1]] ))
  }
  # NB: NITIAL DIRICHLET FROM SIM IS DIFF THAN INIT FROM TIME t
  #init_s_prior <- oryx$Dirichlet(concentration = 1*rep(1,times = 4))
  #oryx$kl_divergence(oryx$Dirichlet(concentration =rep(10,4)),init_s_prior)
  CONST_N_r <- np$asarray( CONST_N )
  print2("Done initializing ParseDynamicODE.R")
}
