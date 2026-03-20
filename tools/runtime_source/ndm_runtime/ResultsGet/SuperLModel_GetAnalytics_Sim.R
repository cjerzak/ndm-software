# Content of SuperLModel_GetAnalytics_Sim.R
if(SimMode == T){
  print("Starting SuperLModel_GetAnalytics_Sim.R")
  ndm_plot_log_series_safe <- getFromNamespace(".ndm_plot_log_series_safe", "ndm")
  analytics_truth_matrix <- function(x, name, nrow_expected, ncol_expected){
    arr <- np$array(x)
    dims <- dim(arr)
    if(length(dims) == 3L && dims[[2]] == 1L){
      arr <- arr[,1,]
      dims <- dim(arr)
    }
    if(length(dims) == 2L && dims[[1]] == nrow_expected && dims[[2]] == ncol_expected){
      return(arr)
    }
    stop(
      sprintf(
        "Expected `%s` to resolve to a [%s x %s] batch matrix; got [%s]",
        name,
        nrow_expected,
        ncol_expected,
        paste(dims, collapse = " x ")
      )
    )
  }
  analytics_truth_vector <- function(x, name, nrow_expected){
    arr <- np$array(x)
    dims <- dim(arr)
    if(length(dims) == 3L && dims[[2]] == 1L && dims[[3]] == 1L){
      arr <- arr[,1,1]
    } else if(length(dims) == 2L && dims[[2]] == 1L){
      arr <- arr[,1]
    }
    arr <- as.numeric(arr)
    if(length(arr) == nrow_expected){
      return(arr)
    }
    stop(
      sprintf(
        "Expected `%s` to resolve to a length-%s batch vector; got [%s]",
        name,
        nrow_expected,
        paste(dims, collapse = " x ")
      )
    )
  }
  res_list <- replicate({list()}, n = nMonteEval)
  for(nj in 1:nMonteEval){
  print2( sprintf("nj %s of %s in GetAnalytics_Sim.R", nj, nMonteEval  )) 
  #GetPredSaveAtInfo_inference <- list(jnp$array(tmp_ <- nTimesTotal-nTimesLookahead+nTimesLookValidation),
  #GetPredSaveAtInfo_inference <- list(jnp$array(tmp_ <- nTimesLookValidationInference),
  GetPredSaveAtInfo_inference <- list(tmp_ <- nTimesLookValidationInference,
                                      diffrax$SaveAt(ts = jnp$array(  0L:(tmp_-1L) )))
  ok_<-F; ok_counter_ <- 0; while(!ok_){
    ok_counter_ <- ok_counter_ + 1
    batch_l <- try(reticulate::iter_next( TFDatasetIterator_inference ), T) 
    if("try-error" %in% class(batch_l)){ batch_l <- reticulate::iter_next( TFDatasetIterator_inference <- reticulate::as_iterator( TFDataset_inference )  )  }
    if(length(batch_l)[[1]] == 0){ batch_l <- reticulate::iter_next( TFDatasetIterator_inference <- reticulate::as_iterator( TFDataset_inference )  )  }
    if( np$array(batch_l[[1]]$shape)[[1]] != nBatch | is.null(batch_l) ){  
      TFDatasetIterator_inference <- reticulate::as_iterator( TFDataset_inference ) 
    }
    if( any(unlist(lapply(batch_l, function(l_){ np$array(l_$shape)[[1]] } ) ) != nBatch) ){  
      badshape2_ctr <- 0; badshape2 <- T; while(badshape2){ 
        print2("Resetting type: bad shape (2) in GetAnalytics_Sim.R")
        badshape2_ctr <- badshape2_ctr + 1; if(badshape2_ctr > 100){stop("Too many bad shape (2)'s in GetAnalytics_Sim.R")}
        batch_l <- reticulate::iter_next(TFDatasetIterator_inference <- reticulate::as_iterator(TFDataset_inference))
        badshape2 <- any(unlist(lapply(batch_l, function(l_){ np$array(l_$shape)[[1]] } ) ) != nBatch)  
      }
    }
    batch_l <- TFConst2JAXArray( batch_l )
    if(ok_counter_ > 100){ stop("Problem in SuperLModel_GetAnalytics_Sim.R") }
    if(!"try-error" %in% class(batch_l)){ ok_ <- T }
  }
  pred_l <- replicate(10,list(GetPred_inference(ModelList,
                                                batch2package(batch_l),
                                                state,
                                                PriorList,
                                                PolicyList,
                                                GetPredSaveAtInfo_inference,
                                                jax$random$split(JaxKey(as.integer(3L+runif(1,1,10000))),nBatch) )[[1]]))
  pred_l_mean <- lapply(pred_l,function(zer){  np$asanyarray( zer$y_mu )  })
  pred_l_mean_full <- pred_l_mean <- 1/length(pred_l_mean) * Reduce("+",pred_l_mean)
  out_y_indices <- (1:ncol(pred_l_mean))[ !1:ncol(pred_l_mean) %in% 1:nTimesPast]
  #pred_l_mean <- pred_l_mean[,out_y_indices <- (ncol(pred_l_mean) - nTimesLookahead+1):ncol(pred_l_mean),1]
  #pred_l_mean <- pred_l_mean[,out_y_indices <- (1:ncol(pred_l_mean))[ !1:ncol(pred_l_mean) %in% 1:nTimesPast],1]
  pred_l_mean <- pred_l_mean_full[,,1]
  pred_l_sd <- lapply(pred_l,function(zer){ np$asanyarray( zer$y_sigma )[,,1] })
  pred_l_sd <- 1/length(pred_l_sd) * Reduce("+",pred_l_sd)
  #plot(pred_l_mean[sample(1:dim(pred_l_mean)[1],1),])

  # analysis of means
  #l_true <- np$asanyarray( batch_l$YTrue) [,out_y_indices,1]
  l_true <- np$asanyarray( batch_l$YTrue_out )[,,1]
  #l_true_full <- np$asanyarray( batch_l$YTrue) [,,1]
  pred_l_baselineVal <- np$asanyarray( batch_l$YTrue ) [,(nTimesTotal - nTimesLookahead),1]
  pred_l_baselineVal_mat <- matrix(1,nrow = nrow(pred_l_mean), 
                                   ncol=ncol(pred_l_mean)) * 
                                pred_l_baselineVal
  LastOutCor <- cor(c(l_true),c(pred_l_mean))
  # plot(c(l_true),c(pred_l_mean_full));abline(a=0,b=1)
  
  ##############################
  # R-squared analysis
  ##############################
  {
  ### with predicted values
  RawR2 <- function(yhat,ytrue){ 1 - sum( (yhat-ytrue)^2) / sum((ytrue-mean(ytrue))^2) }
  MedianWithinR2_Raw <- median( sapply(1:nrow(l_true),function(zer){
    WithinR2 <- RawR2(c(pred_l_mean[zer,]),l_true[zer,])
  }) )
  MedianWithinR2_Fit <- median( sapply(1:nrow(l_true),function(zer){
    WithinR2 <- lm(c(l_true[zer,])~c(pred_l_mean[zer,]))
    WithinR2 <- summary(WithinR2)$adj.r.squared
  }) )
  GlobalR2_Raw <- RawR2(c(pred_l_mean),c(l_true))
  GlobalR2_Fit <- summary(lm(c(l_true)~c(pred_l_mean)))$adj.r.squared

  ### compare against baseline
  MedianWithinR2_Raw_baseline <- median( sapply(1:nrow(l_true),function(zer){
    WithinR2 <- RawR2(
      rep(pred_l_baselineVal[zer],times = ncol(l_true)),
      l_true[zer,])
  }) )
  MedianWithinR2_Fit_baseline <- median( sapply(1:nrow(l_true),function(zer){
    WithinR2 <- lm(c(l_true[zer,])~1)
    WithinR2 <- summary(WithinR2)$adj.r.squared
  }) )
  GlobalR2_Raw_baseline <- RawR2(c(pred_l_baselineVal_mat),c(l_true))
  GlobalR2_Fit_baseline <- summary(lm(c(l_true)~c(pred_l_baselineVal_mat)))$adj.r.squared

  FNorm_Raw <- mean(sqrt((c(pred_l_mean) - c(l_true))^2))
  FNorm_Rel <- mean(sqrt((c(pred_l_mean) - c(l_true))^2)) / mean(sqrt(l_true^2))
  FNorm_Raw_baseline <- mean(sqrt((c(pred_l_baselineVal_mat) - c(l_true))^2))
  FNorm_Rel_baseline <- FNorm_Raw_baseline / mean(sqrt(l_true^2))

  # agg norm
  AggNorm_Raw <- mean(sqrt((rowSums(pred_l_mean) - rowSums(l_true))^2))
  AggNorm_Rel <- mean(sqrt((rowSums(pred_l_mean) - rowSums(l_true))^2)) / 
                                    mean(sqrt(rowSums(l_true)^2))
  AggNorm_Raw_baseline <- mean(sqrt((rowSums(pred_l_baselineVal_mat) - rowSums(l_true))^2))
  AggNorm_Rel_baseline <- AggNorm_Raw_baseline / mean(sqrt(rowSums(l_true)^2))

  # Skill measure
  #plot(l_true[rer<-sample(1:10,1),],ylim=c(0,max(l_true))); points(pred_l_mean[rer,],pch="^")
  #plot(l_true[1,]);plot(pred_l_mean[1,])
  RSS_baseline <- apply( (pred_l_baselineVal_mat - l_true)^2, 2, function(zr){ return(mean(clipAt(zr))) } )
  RSS_pred <- apply( (pred_l_mean - l_true)^2, 2, function(zr){ return(mean(clipAt(zr) )) } )
  skill_vec <- 1 - (0.001+RSS_pred^0.5) / (0.001+RSS_baseline^0.5)
  names(skill_vec) <- paste("SkillTime",1:length(skill_vec), sep = "")
  names(RSS_pred) <- paste("RSSPredTime",1:length(skill_vec), sep = "")
  names(RSS_baseline) <- paste("RSSBaselineTime",1:length(skill_vec), sep = "")

  sd_outcome <- median( apply(l_true,1,sd) )
  }

  ##############################
  # analyze structural parameters
  ##############################
  init_s <- init_e <- init_i <- init_r <- init_hat <-
    AbsDiff_init <- KLDiv_init <- sigma_hat <- gamma_hat <-
    AbsDiff_gamma <- RelAbsDiff_gamma <- AbsDiff_sigma <- RelAbsDiff_sigma <- NA
  beta_hat <- beta_MedianWithinR2_Fit <- beta_MedianWithinR2_Raw <- beta_true <- NA;
  beta_GlobalR2_Raw <- beta_GlobalR2_Fit <- beta_FNorm_Rel <- beta_FNorm_Raw <- beta_AggNorm_Rel <- beta_AggNorm_Raw <- NA
  if(ModelType != "DecoderOnly"){
    init_true_mat <- analytics_truth_matrix(
      batch_l$init_true,
      name = "init_true",
      nrow_expected = nrow(pred_l_mean),
      ncol_expected = 4L
    )
    gamma_true_vec <- analytics_truth_vector(
      batch_l$gamma_true,
      name = "gamma_true",
      nrow_expected = nrow(pred_l_mean)
    )
    sigma_true_vec <- analytics_truth_vector(
      batch_l$sigma_true,
      name = "sigma_true",
      nrow_expected = nrow(pred_l_mean)
    )

    # initial values
    init_s <- colMeans(do.call(rbind,lapply(pred_l,function(zer){np$asanyarray( zer$ODEParamsSampList$s_l_samp) })))
    init_e <- colMeans(do.call(rbind,lapply(pred_l,function(zer){np$asanyarray( zer$ODEParamsSampList$e_l_samp) })))
    init_i <- colMeans(do.call(rbind,lapply(pred_l,function(zer){np$asanyarray( zer$ODEParamsSampList$i_l_samp) })))
    init_r <- colMeans(do.call(rbind,lapply(pred_l,function(zer){np$asanyarray( zer$ODEParamsSampList$r_l_samp) })))
    init_hat <- cbind((init_s),(init_e),(init_i),(init_r))
    AbsDiff_init <- median(rowSums( abs(init_hat - init_true_mat )))
    KLDiv_init <- median( rowSums(  init_true_mat * (log(init_hat) - log(init_true_mat) ) ))

    sigma_hat <- colMeans(do.call(rbind,lapply(pred_l,function(zer){np$asanyarray( zer$ODEParamsSampList$sigma_samp) })))
    gamma_hat <- colMeans(do.call(rbind,lapply(pred_l,function(zer){np$asanyarray( zer$ODEParamsSampList$gamma_samp) })))

    AbsDiff_gamma <- median(  abs( gamma_true_vec - gamma_hat) )
    RelAbsDiff_gamma <- median(  abs( gamma_true_vec - gamma_hat)/gamma_true_vec)

    AbsDiff_sigma <- median(  abs( sigma_true_vec - sigma_hat) )
    RelAbsDiff_sigma <- median(  abs( sigma_true_vec - sigma_hat)/sigma_true_vec)

    ### beta analysis
    if(ModelType != "DecoderOnly"){ 
      beta_hat <- lapply(pred_l, function(l_){ ExtractBetaDraw(l_$ODEParamsSampList)$BetaDraw })
      beta_hat <- 1/length(beta_hat) * Reduce("+",beta_hat)
    }
    if(T == F){ 
      beta_true <- t( apply(np$array(batch_l$inv_beta_true),1,function(zer){
         #np$asanyarray( SoftPlus( zer[[1]]$evaluate(jnp$array(1.:max(out_y_indices),dtype=jnp$float64)) ) )
          #zer 
          zer[1:max(out_y_indices)]
      }) )
      beta_MedianWithinR2_Raw <- median( sapply(1:nrow(beta_true),function(zer){
        WithinR2 <- RawR2(c(beta_hat[zer,]),beta_true[zer,])
      }) )
      beta_MedianWithinR2_Fit <- median( sapply(1:nrow(beta_true),function(zer){
        WithinR2 <- lm(c(beta_true[zer,])~c(beta_hat[zer,]))
        WithinR2 <- summary(WithinR2)$adj.r.squared }) )
      
      beta_GlobalR2_Raw <- RawR2(c(beta_hat),c(beta_true))
      beta_GlobalR2_Fit <- summary(lm(c(beta_true)~c(beta_hat)))$adj.r.squared
  
      beta_FNorm_Raw <- mean(sqrt((c(beta_hat) - c(beta_true))^2))
      beta_FNorm_Rel <- mean(sqrt((c(beta_hat) - c(beta_true))^2)) / mean(sqrt(beta_true^2))
  
      beta_AggNorm_Raw <- mean(sqrt((rowSums(beta_hat) - rowSums(beta_true))^2))
      beta_AggNorm_Rel <- mean(sqrt((rowSums(beta_hat) - rowSums(beta_true))^2)) / mean(sqrt(rowSums(beta_true)^2))
    }
  }

  ##############################
  # policy eval metrics
  ##############################
  {
  # data gen part  is indexed to: t0 = 0, t1 = NTimeSteps_SIM,
  # model part  is indexed to: t0 = 0, t1 = nTimesTotal,
  InterpolatorBasis_ <- rep(0.1, times = NTimeSteps_SIM+1)
  #ShiftFactor <- min( np$array( NTimeSteps_SIM - nTimesPast )  )
  ShiftFactor <- min( np$array( 0 + nTimesPast )  )

  startIntervention <- max( np$array( ShiftFactor )  ) + 2L
  IndicatorBasis_un <- rep(0, NTimeSteps_SIM+1)
  IndicatorBasis_un[-c(1:startIntervention)] <- 1
  # IndicatorBasis_un[-c(1:ShiftFactor)][1:nTimesTotal]
  # PolicyList_GetBatch_unnatural[[1]]$evaluate(jnp$array((startIntervention-1):startIntervention))
  PolicyList_GetBatch_natural <- list(
    "ComputeScenario" = F,

    # scenario (defines the smooth shift trajectory in the policy from the previous state)
    "Scenario" = jnp$array(InterpolatorBasis_[1:NTimeSteps_SIM])
  )

  PolicyList_GetBatch_unnatural <- list(
    "ComputeScenario" = T,

    # scenario (defines the smooth shift trajectory in the policy from the previous state)
    "Scenario" = jnp$array(InterpolatorBasis_[1:NTimeSteps_SIM])
    )

  model_policy_ts <- jnp$array(as.numeric(0:(nTimesTotal-1L)))
  model_policy_scenario <- jnp$array(InterpolatorBasis_[-c(1:ShiftFactor)][1:(nTimesTotal+0)])
  PolicyList_Model_natural <- list(
    # policy indicator
    diffrax$LinearInterpolation(
      model_policy_ts,
      jnp$array(rep(0, nTimesTotal + 0))
    ),

    # scenario (defines the smooth shift trajectory in the policy from the previous state)
    diffrax$LinearInterpolation(
      model_policy_ts,
      model_policy_scenario
    )
  )

  PolicyList_Model_unnatural <- list(
    # policy indicator
    diffrax$LinearInterpolation(
      model_policy_ts,
      jnp$array(IndicatorBasis_un[-c(1:ShiftFactor)][1:(nTimesTotal+0)])
    ),

    # scenario (defines the smooth shift trajectory in the policy from the previous state)
    diffrax$LinearInterpolation(
      model_policy_ts,
      model_policy_scenario
    )
  )
  }

  
  # get factual and counterfactual data
  PolicyScenarioSkillRes <- PolicyScenarioSkillBaselineRes <- NULL; nCounterfactuals <- 0; policy_eval_failed <- FALSE; while( nCounterfactuals < 2 ){
    nCounterfactuals <- nCounterfactuals + 1
    counterf_ <- ai(runif(1,1,10000))
    ok_<-F; ok_counter_ <- 0; while(!ok_){
      ok_counter_ <- ok_counter_ + 1
      set.seed(PolicyEvalSeed <- (12456L + counterf_)); batch_l_natural <- try(GetBatch(
                                      nBatch = nBatch, 
                                      INPUT_REF_DAT = input_df_red_in,
                                      PolicyList = PolicyList_GetBatch_natural,
                                      nTimes = NTimeSteps_SIM,
                                      nTimesLook = nTimesLookValidation), T)
      set.seed(PolicyEvalSeed);batch_l_unnatural <- try(GetBatch(nBatch = nBatch,  
                                      INPUT_REF_DAT = input_df_red_in,
                                      nTimes = NTimeSteps_SIM,
                                      nTimesLook = nTimesLookValidation,
                                      PolicyList = PolicyList_GetBatch_unnatural), T) 
                                      #PolicyList = PolicyList_GetBatch_natural),T) # for debugging
      if(ok_counter_ > 100){
        warning("Skipping policy scenario analytics because natural/counterfactual batches could not be generated")
        policy_eval_failed <- TRUE
        break
      }
      if(!"try-error" %in% class(batch_l_natural) & !"try-error" %in% class(batch_l_unnatural) ){ ok_ <- T }
    }
    if(policy_eval_failed){ break }
    {
      # get factual and counterfactual predictions
      # note: t is differnt in meaning for getting data and for modeling
      hat_y_natural <- GetPred_inference(
        ModelList, batch2package(batch_l_natural),
        state, PriorList, PolicyList_Model_natural,
        GetPredSaveAtInfo_inference,
        jax$random$split(JaxKey(9L+counterf_), nBatch))[[1]]
      hat_y_unnatural <- GetPred_inference(
        ModelList, batch2package(batch_l_unnatural),
        state, PriorList, PolicyList_Model_unnatural,
        GetPredSaveAtInfo_inference,
        jax$random$split(JaxKey(9L), nBatch))[[1]]

      # skill baseline -predicting unnatural outcomes with natural outcomes
      NUM <- (0.001+colMeans(abs(np$array( batch_l_unnatural$YTrue_out )[,,1] -
                            np$array( batch_l_natural$YTrue_out )[,,1])))
      DENOM <- (0.001+colMeans(abs( np$array( batch_l_unnatural$YTrue_out )[,,1] -
                            matrix(np$array( batch_l_natural$YTrue_out )[,1,1],
                                   ncol = ncol(np$array( batch_l_natural$YTrue_out )),
                                   nrow = nrow(np$array( batch_l_natural$YTrue_out )),byrow=F)   )) )
      PolicyScenarioSkill_baseline <- NUM / DENOM

      # skill - predicting unnatural outcomes using natural predictions
      NUM <- (0.001+colMeans(abs(np$array( batch_l_unnatural$YTrue_out )[,,1] -#out_y_indices
                            np$array( hat_y_unnatural$y_mu )[,,1] )))
                            #np$array( hat_y_unnatural$y_mu )[,out_y_indices,1] )))
      DENOM <- (0.001+colMeans(abs(np$array( batch_l_unnatural$YTrue_out )[,,1] -
                                np$array( hat_y_natural$y_mu )[,,1])) )
                                     #np$array( hat_y_natural$y_mu )[,out_y_indices,1])) )
      PolicyScenarioSkill <- NUM  / DENOM

      names(PolicyScenarioSkill) <- paste("PolicySkill", 1:length(PolicyScenarioSkill), sep="")
      names(PolicyScenarioSkill_baseline) <- paste("PolicySkillBaseline", 1:length(PolicyScenarioSkill_baseline), sep="")
      PolicyScenarioSkillRes <- rbind(PolicyScenarioSkillRes,PolicyScenarioSkill)
      PolicyScenarioSkillBaselineRes <- rbind(PolicyScenarioSkillBaselineRes,PolicyScenarioSkill_baseline)

      if(PolicyPlot <- F){
        # sample an observation
        i_ <- sample(1:batch_l_natural$YTrue[[1]]$shape[[1]], 1)

        # data-generation outcome scenario checks
        par(mfrow=c(2,2))
        plot( np$array( batch_l_natural$YTrue[[1]] )[i_,,1],main = "Disease Outcomes", ylim = c(0, max( c(np$array( batch_l_unnatural$YTrue[[1]] )[i_,,1] ,np$array( batch_l_natural$YTrue[[1]] )[i_,,1]))))
        #abline(v =  batch_l_natural$YTrue[[1]]$shape[[2]] - nTimesLookahead-1,col="red")
        points( np$array( batch_l_unnatural$YTrue[[1]] )[i_,,1] , col = "gray",pch = 3)

        # data-generation policy scenario checks
        plot( tmp <- np$array( batch_l_natural$YTrue[[1]] )[i_,,2], main = "Policy Stringency", ylim = summary(c(np$array( batch_l_unnatural$YTrue[[1]] )[i_,,2], np$array( batch_l_natural$YTrue[[1]] )[i_,,2]) )[c(1,6)])
        #abline(v =  batch_l_natural$YTrue[[1]]$shape[[2]] - nTimesLookahead,col="red")
        points( np$array( batch_l_unnatural$YTrue[[1]] )[i_,,2],  col = "gray", pch = 3)

        # data-generation beta scenario checks
        plot( np$array( SoftPlus(batch_l_natural$inv_beta_true))[i_,1:length(tmp)], main = "Beta", ylim = summary(c(np$array( SoftPlus(batch_l_natural$inv_beta_true) )[i_,], np$array( SoftPlus(batch_l_unnatural$inv_beta_true) )[i_,]) )[c(1,6)])
        #abline(v =  batch_l_natural$YTrue[[1]]$shape[[2]] - nTimesLookahead,col="red")
        points( np$array( SoftPlus(batch_l_unnatural$inv_beta_true) )[i_,1:length(tmp)],  col = "gray", pch = 3)


        # prediction scenario outcome checks
        plot( ( np$array( hat_y_natural$y_mu )[i_,,1]), main="Pred Truths",ylim = summary(c(( np$array( hat_y_natural$y_mu )[i_,,1]),c(np$array( hat_y_unnatural$y_mu )[i_,,1])))[c(1,6)])
        points( np$array( hat_y_unnatural$y_mu )[i_,,1],  col = "gray", pch = 3)
        # plot(np$array( hat_y_natural$y_mu )[i_,,1] - np$array( hat_y_unnatural$y_mu )[i_,,1])

        # prediction scenario policy checks
        plot( ( np$array( hat_y_natural$y_mu )[i_,,2]), main="Pred Policies", ylim = c(-10, 10))
        points( np$array( hat_y_unnatural$y_mu )[i_,,2],  col = "gray", pch = 3)
        PolicyList_Model_unnatural[[1]]$evaluate(jnp$array(0:(nTimesTotal-1L)))
        par(mfrow=c(1,1))

        # prediction beta policy checks
        #plot( ( np$array( hat_y_natural$ODEParamsSampList$diff_eq_sol_ys.Neural1 )[i_,,LocalNeuralEmbedDim+1]), ylim = summary( np$array( hat_y_natural$ODEParamsSampList$diff_eq_sol_ys.Neural1 )[i_,,LocalNeuralEmbedDim+1])[c(1,6)])
        #points( np$array( hat_y_unnatural$ODEParamsSampList$diff_eq_sol_ys.Neural1 )[i_,,LocalNeuralEmbedDim+1],  col = "gray", pch = 3)

        # analysis
        # plot( np$array(hat_y_natural$ODEParamsSampList$diff_eq_sol_ys.Neural1)[i_,,LocalNeuralEmbedDim+1] )
        # hist( apply(np$array(hat_y_natural$ODEParamsSampList$diff_eq_sol_ys.Neural1)[,,LocalNeuralEmbedDim+1], 1,  function(zer){mean(diff(zer))}), xlab = ""); abline(v = 0, lwd = 2)
      }
    }
  }
  if(is.null(PolicyScenarioSkillRes) || is.null(PolicyScenarioSkillBaselineRes)){
    PolicyScenarioSkillRes <- rep(NA_real_, nTimesLookValidationInference)
    names(PolicyScenarioSkillRes) <- paste("PolicySkill", seq_len(nTimesLookValidationInference), sep="")
    PolicyScenarioSkillBaselineRes <- rep(NA_real_, nTimesLookValidationInference)
    names(PolicyScenarioSkillBaselineRes) <- paste("PolicySkillBaseline", seq_len(nTimesLookValidationInference), sep="")
  } else {
    PolicyScenarioSkillRes <- colMeans(PolicyScenarioSkillRes)
    PolicyScenarioSkillBaselineRes <- colMeans(PolicyScenarioSkillBaselineRes)
  }
  rm_list_ <- c("batch_l_unnatural","batch_l_natural","hat_y_natural","hat_y_unnatural")
  rm_list_ <- rm_list_[vapply(rm_list_, exists, logical(1), inherits = FALSE)]
  if(length(rm_list_) > 0L){ rm(list = rm_list_) }

  ##############################
  # setup save variable names
  ##############################
  if(nj == 1){
    SaveQuantityNames <- c(outcome2 <- c("MedianWithinR2_Raw","MedianWithinR2_Fit",
                                         "GlobalR2_Raw","GlobalR2_Fit",
                                         "FNorm_Raw","FNorm_Rel",
                                         "AggNorm_Raw","AggNorm_Rel"),
                           paste(outcome2, "_baseline",sep = ""))
    SaveQuantityNames <- c(SaveQuantityNames,"beta_MedianWithinR2_Raw",
                  "beta_MedianWithinR2_Fit",
                  "beta_AggNorm_Raw",
                  "beta_AggNorm_Rel",
                  "beta_FNorm_Raw",
                  "sd_outcome",
                  "beta_FNorm_Rel",
                  "beta_GlobalR2_Raw",
                  "beta_GlobalR2_Fit")
    SaveQuantityNames <- c(SaveQuantityNames,
                           "AbsDiff_init","KLDiv_init",
                           "AbsDiff_gamma","RelAbsDiff_gamma",
                           "AbsDiff_sigma","RelAbsDiff_sigma")
  }

  tmp <- sapply(SaveQuantityNames,function(ze){
    list(eval(parse(text = sprintf("c('%s' = eval(parse(text = %s)))",ze,ze))))
  })
  names(tmp) <- NULL
  res_list[[nj]] <- c(do.call(c,tmp), skill_vec, RSS_pred, RSS_baseline, PolicyScenarioSkillRes, PolicyScenarioSkillBaselineRes)
  }
  
  #  write loss fig for debugging 
  pdf(file.path(HolderFolder, sprintf("diagnostics%s.pdf", af)), height = 10, width = 5)
  {
  par(mfrow=c(2,2))
  loss_obs <- na.omit(in_loss_vec)
  grad_obs <- na.omit(grad_norm_vec)
  ndm_plot_log_series_safe(loss_obs, main = "Loss")
  if(length(loss_obs) > 0L){ plot(rank(loss_obs), log = "") } else { plot.new() }
  ndm_plot_log_series_safe(grad_obs, main = "Gradients")
  dev.off()
  }
  par(mfrow = c(1,1))
  
  
  # --- Skill sanity check (aggregate RMSE vs. persistence baseline) ---
  {
    eps <- 1e-3
    skill_horizon <- max(1L, min(8L, ncol(pred_l_mean)))
    rmse_pred <- sqrt(mean(clipAt((c(pred_l_mean[,skill_horizon]) - c(l_true[,skill_horizon]))^2), na.rm = TRUE))
    rmse_base <- sqrt(mean(clipAt((c(pred_l_baselineVal_mat[,skill_horizon]) - c(l_true[,skill_horizon]))^2), na.rm = TRUE))
    Skill8SanityCheck <- 1 - (eps + rmse_pred) / (eps + rmse_base)
    print2(sprintf("Skill sanity: %.3f", Skill8SanityCheck))
  }
  
  # save results to disk 
  res_vec <- t(c("sim_index" = af,
                 "i_in_sgd" = i,
                 "nTrainingSamplesSeen" = i*nBatch,
                 "atEpoch" = i*nBatch/nSamplesTrain, 
                 "nSGD" = nSGD_model,
                 "nParamsModel" = nParamsModel,
                 "te_total" = te_total, 
                 "te_grads" = te_grads, 
                 unlist(SimEntry),
                 colMeans(do.call(rbind,res_list))))
  print(sprintf("WRITING af of %s", af))
  data.table::fwrite(file = file.path(HolderFolder, sprintf("res%s_i%s.csv", af, i)), res_vec)
  save.image(file = file.path(HolderFolder, sprintf("res%s_i%s.Rdata", af, i)))
  print("Done with SuperLModel_GetAnalytics_Sim.R")
}
