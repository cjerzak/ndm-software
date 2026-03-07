# Content of SuperLModel_TrainDo.R
print("Sarting SuperLModel_TrainDo.R")
for(i in i_:nSGD_model){
  if(Sys.info()["sysname"] != "Darwin" & i > 2){  
    #write.csv( file="./i_.csv",data.frame("i" = i, "te_total" = as.numeric(te_total, units = "secs"), "te_grad" = as.numeric(te_grads, units = "secs") ))  
  }
  # save model checkpoints
  if(nCheckpoints > 0){
  CheckPointSaveAt <- unique(c(round(10^(seq(log(10,base=10), 
                                             log(nSGD_MASTER,base=10), 
                                             length.out = nCheckpoints)),0L),
                               nSGD_MASTER))
  if(  i %in% CheckPointSaveAt ){
    print2( "Saving checkpoint (Optimizer, Model, Model-generating Code)..." )
    saveCheckpointCounter <- saveCheckpointCounter + 1

    if(saveCheckpointCounter == 1){
      zip::zip(zipfile = sprintf("./SavedModels/%s/Model_%s_%s/AnalysisR_%s_%s.zip",
                            ifelse(grepl(tolower(AnalysisName), pattern = "sim"),
                                   yes = "FromSim",no="FromReal"), 
                            AnalysisName, AnalysisDate,
                            AnalysisName, AnalysisDate), files = NDM_INTERNAL_ANALYSIS_DIR)
    }

    if(i %in% CheckPointSaveAt){ save_i <- i }
    if(SaveEqx <- F){ 
      # save_i <- "last"
      eq$tree_serialise_leaves(sprintf("./SavedModels/%s/Model_%s_%s/ModelList_i%s_%s_%s.eqx",
                                       ifelse(grepl(tolower(AnalysisName), pattern = "sim"), yes = "FromSim",no="FromReal"), 
                                       AnalysisName, AnalysisDate,
                                       save_i,
                                       AnalysisName, AnalysisDate),
                               list( ModelList, state, opt_state, 
                                    jnp$array(list(SIM_GLOBAL_SCALE_MEAN,SIM_GLOBAL_SCALE_SD))))
      if("ModelList_notshared_set" %in% ls()){
        eq$tree_serialise_leaves(sprintf("./SavedModels/%s/Model_%s_%s/ModelList_notshared_set_i%s_%s_%s.eqx",
                                         ifelse(grepl(tolower(AnalysisName), pattern = "sim"), yes = "FromSim",no="FromReal"), 
                                         AnalysisName, AnalysisDate,
                                         save_i,
                                         AnalysisName, AnalysisDate),
                                 list(ModelList_notshared_set, # nonshared parameters 
                                      lapply(sim_dat_norm_list,jnp$array)) # means for recovery 
                                 ) 
      }
    }

    # recover trained model
    if(T == F){
      # nSGD_model / 1:4 # defines i's
      RecoverAt <- "last"; ModelList_recovered <- eq$tree_deserialise_leaves(
        sprintf("./SavedModels/%s/Model_%s_%s/ModelList_i%s_%s_%s.eqx",
                ifelse(grepl(tolower(AnalysisName), pattern = "sim"), yes = "FromSim",no="FromReal"), 
                AnalysisName, AnalysisDate, RecoverAt,
                AnalysisName, AnalysisDate),
        list( ModelList, state, opt_state,
             jnp$array(list(SIM_GLOBAL_SCALE_MEAN,SIM_GLOBAL_SCALE_SD)) ) ) 
      if("ModelList_notshared_set" %in% ls()){
        ModelList_notshared_set_recovered <- eq$tree_deserialise_leaves(sprintf("./SavedModels/%s/Model_%s_%s/ModelList_notshared_set_i%s_%s_%s.eqx",
                                         ifelse(grepl(tolower(AnalysisName), pattern = "sim"), yes = "FromSim",no="FromReal"), 
                                         AnalysisName, AnalysisDate,
                                         RecoverAt,
                                         AnalysisName, AnalysisDate),
                                 list(ModelList_notshared_set, # non shared parameters 
                                      lapply(sim_dat_norm_list,jnp$array) ) # norm factors 
                                 )
        sim_dat_norm_list <- ModelList_notshared_set_recovered[[2]] 
        sim_dat_norm_list <- lapply(sim_dat_norm_list,function(l_){
          SIM_GLOBAL_SCALE_MEAN_ <- jnp$take(l_,0L,axis=0L)$tolist()
          SIM_GLOBAL_SCALE_SD_ <- jnp$take(l_,1L,axis=0L)$tolist()
          list(SIM_GLOBAL_SCALE_MEAN_, SIM_GLOBAL_SCALE_SD_)
        })
        SIM_GLOBAL_SCALE_MEAN <- jnp$take(ModelList_recovered[[4]],0L,axis=0L)$tolist()
        SIM_GLOBAL_SCALE_SD <- jnp$take(ModelList_recovered[[4]],1L,axis=0L)$tolist()
        ModelList_notshared_set <- ModelList_notshared_set_recovered[[1]]
      }

      # confirm 
      SIM_GLOBAL_SCALE_MEAN <- jnp$take(ModelList_recovered[[4]],0L,axis=0L)$tolist()
      SIM_GLOBAL_SCALE_SD <- jnp$take(ModelList_recovered[[4]],1L,axis=0L)$tolist()
      # opt_state[[1]]
      # opt_state[[3]][[1]]
      opt_state <- ModelList_recovered[[3]];
      state <- ModelList_recovered[[2]];
      ModelList <- ModelList_recovered[[1]]
      i_ <- np$array( opt_state[[4]][[2]]$count )
      rm( ModelList_recovered )
    }
  }

  # save analytics
  if( i %in% CheckPointSaveAt  ){
    print2( sprintf("Starting GetAnalytics.R at %s of %s", i, nSGD_model) )
    outSampCounter <- outSampCounter+1;ndm_source_extracted("ResultsGet/SuperLModel_GetAnalytics.R")
  }
  }
  
  i_ <- i; fulliter_timer <- Sys.time()
  if(i %% 10 == 0){  print(i); gc(); py_gc$collect() }

  # get batch
  if(nSGD_pretrain > 0L | nSGD_posttrain > 0){ 
  dat_ <- try(reticulate::iter_next( TFDatasetIterator_train ), T)
  # batch_l_train <-  TFConst2JAXArray(dat_)
  # par(mfrow=c(2,2));i_<-sample(1:50,1)
  # plot(np$array(dat_$XPred)[i_,,c_<-sample(1:dat_$XPred$shape[2],1)],main = dataInputs_colnames[c_],cex.main=0.5)
  if("try-error" %in% class(dat_)){ 
    print2("Resetting type: try-error in TrainDo.R...")
    dat_ <- reticulate::iter_next(TFDatasetIterator_train <- reticulate::as_iterator(TFDataset_train))
  }
  if(is.null(dat_)){ 
    print2("Resetting type: is.null in TrainDo.R...")
    dat_ <- reticulate::iter_next(TFDatasetIterator_train <- reticulate::as_iterator(TFDataset_train))
  }
  if(length(dat_)[[1]] == 0){ 
    print2("Resetting type: length 0 dat_ in TrainDo.R...")
    dat_ <- reticulate::iter_next(TFDatasetIterator_train <- reticulate::as_iterator(TFDataset_train))
  }
  if(np$array(dat_[[1]]$shape)[[1]] != nBatch){  
    print2("Resetting type: bad shape (1) in TrainDo.R...")
    dat_ <- reticulate::iter_next(TFDatasetIterator_train <- reticulate::as_iterator(TFDataset_train))
  }
  if( any(unlist(lapply(dat_, function(l_){ np$array(l_$shape)[[1]] } ) ) != nBatch) ){  
    badshape2_ctr <- 0; badshape2 <- T; while(badshape2){ 
      print2("Resetting type: bad shape (2) in TrainDo.R...")
      badshape2_ctr <- badshape2_ctr + 1; if(badshape2_ctr > 100){stop("Too many bad shape (2)'s in TrainDo.R")}
      dat_ <- reticulate::iter_next(TFDatasetIterator_train <- reticulate::as_iterator(TFDataset_train))
      badshape2 <- any(unlist(lapply(dat_, function(l_){ np$array(l_$shape)[[1]] } ) ) != nBatch)  
    }
  }
  dat_ <- TFConst2JAXArray( dat_ )

  # update step
  {
    gd_timer <- Sys.time()
    if( i == 1 ){ print2( "At first gradLoss_jax()" ) }
    keys_mat <- jax$random$split(JaxKey(ai(i)),nBatch)
    GetPredSaveAtInfo_default[[1]] <- as.integer(np$array(GetPredSaveAtInfo_default[[1]]) )# prevents scan compilation issues
    GradientUpdatePackage <- gradLoss_jax(ModelList,
                                          batch2package(dat_),
                                          dat_$YTrue_out,
                                          dat_$YTrue_out_mask,
                                          jnp$array(as.numeric(i)),
                                          state,
                                          PriorList,
                                          PolicyList,
                                          GetPredSaveAtInfo_default,
                                          keys_mat)

    #if("try-error" %in% class(GradientUpdatePackage)){ print(GradientUpdatePackage); stop( "Failure at GradientUpdatePackage" ) }
    if(!"try-error" %in% class(GradientUpdatePackage)){
      # update state
      state_tmp <- GradientUpdatePackage[[1]][[2]]
      GradientUpdatePackage[[1]] <- GradientUpdatePackage[[1]][[1]]

      # update loss
      myLoss_fromGrad <- GradientUpdatePackage[[1]]$tolist()
      GradientUpdatePackage <- GradientUpdatePackage[[2]];
      GradientUpdatePackage <- eq$partition(GradientUpdatePackage, eq$is_inexact_array)
      GradientUpdatePackage_aux <- GradientUpdatePackage[[2]]; GradientUpdatePackage <- GradientUpdatePackage[[1]]

      # get information about gradient norms
      #if(i == 1){
      if(i == 1 | i %% 10 == 0){
        grad_vec <- unlist(lapply(jax$tree_util$tree_leaves(GradientUpdatePackage), function(ze){ return( np$array(jnp$abs(ze)$mean()) ) }))
        values3 <- rrapply::rrapply(GradientUpdatePackage, condition = function(x){TRUE},
                                   function(x){
                                     tmp_ <- unlist(lapply(jax$tree_util$tree_leaves(x),
                                                   function(ze){ return( np$array(jnp$abs(ze)$mean()) ) }))
                                     return(tmp_)
                                   },how="list")
        if(is.na(COMMAND_ARG_INPUT)){
          grad_norm_mat <- rbind(grad_norm_mat, unlist(values3))
          # PCNorms <- predict(prcomp(t(grad_norm_mat)))[,1:2]
          #row.names(PCNorms) <- colnames(grad_norm_mat)
          #sort(PCNorms[,1]);sort(PCNorms[,2])
          # View(grad_norm_mat)
          # grad_norm_mat <- as.data.frame(grad_norm_mat)
          # sort(colMeans(grad_norm_mat))
          # sort(apply(grad_norm_mat,2,sd))
          # plot(sort(colMeans(grad_norm_mat)))
          # plot(grad_norm_mat$InitProcessList.InitialEncodingTransform.Conv1d_long)
        }
        print(tail(sort(unlist(values3),decreasing=F),25))
        mean(duplicated(names(sort(unlist(values3)))))
        plot(sort(unlist(values3)))
        
        if(i > 1){ 
          plot(unlist(values3), unlist(values3_past)); abline(a=0,b=1)
        }
        values3_past <- values3
      
        grad_list <- jax$tree_util$tree_map(function(ze){
          leave_grad <- try(as.numeric(np$array(jnp$abs( 
            jax$tree_util$tree_leaves(ze)[[1]] )$mean()) ),T)
          if(class(leave_grad) == "try-error"){browser()}
          return(leave_grad)}, GradientUpdatePackage)
        #unlist(GradientUpdatePackage); unlist(grad_list)
        #plot(unlist(grad_vec)+0.00001, main = sprintf("Grad mag: {%.3f%% are zero}...",100*mean(unlist(grad_vec)==0)), log = "y")
      }
      GradNorm_i <- grad_norm_vec[i] <- np$array( optax$global_norm(  jax$tree_util$tree_leaves(GradientUpdatePackage) )  )

      # update parameters if passing thru all checks
      if(is.na(GradNorm_i)){print("NA in gradient!")}
      UpdateParametersCond <- !is.na(myLoss_fromGrad) & !is.na(GradNorm_i)
      if(! UpdateParametersCond ){ print("Warning: Not updating parameters; check code...") }
      if( UpdateParametersCond){
        ExecuteUpdateCounter <- ExecuteUpdateCounter + 1

        # updates + setup for net iteration
        GradientUpdatePackage <- jit_get_update(
          updates = GradientUpdatePackage,
          state = opt_state,
          params = eq$partition(ModelList, eq$is_array)[[1]] )
        
        opt_state <- GradientUpdatePackage[[2]]
        GradientUpdatePackage <- eq$combine(GradientUpdatePackage[[1]], GradientUpdatePackage_aux)

        # perform update
        if(i == 1){ 
          updatefxn_ <- eq$filter_jit( function(ml,gup){ 
               eq$combine( jit_apply_updates(
                            params = GlobalPartition(eq$partition(ml, eq$is_array)[[1]],
                                                     PartFxn)[[1]],
                            updates = GlobalPartition(gup,PartFxn)[[1]]),
                            eq$partition(ModelList, eq$is_array)[[2]] )
              })
        }
        ModelList <- updatefxn_(ModelList, GradientUpdatePackage)
        in_loss_vec[i] <- as.numeric( myLoss_fromGrad )
        rm(GradientUpdatePackage, state_tmp, dat_)
      }

      # plotting sequence (do this after all parameter + state updates complete)
      {
      if( (i == 1 | i %% 10 == 0)  & is.na(COMMAND_ARG_INPUT) ){
          plottingSeq_counter <- plottingSeq_counter + 1
          if(plottingSeq_counter == 1){ print2( "First plotting seq analysis"  ) }
          par(mfrow = c(1,2))
          for(reri in 1:2){
            # tmp2 <- GetPred_train_jit( ModelList, jax_batchx, state, PriorList, PolicyList, jax$random$split(JaxKey(ai(runif(1,1,100000))), nBatch))
            # plot( np$array( tmp2[[1]]$y_mu )[,,1] ) ; plot( c(np$array( tmp2[[1]]$y_sigma[,,1] ) ))
            ( {tmp555 <- np$array(GetPred_train_jit(
              ModelList, batch2package(batch_l_cal),
              state, PriorList, PolicyList,
              GetPredSaveAtInfo_default,
              jax$random$split(JaxKey(ai(runif(1,1,100000))), nBatch))[[1]]$y_mu)[,,1] })
            if(reri == 1){ tmp_ <- tmp555 }
            if(reri == 1){ try(plot(tmp555[1,],ylim = c(0,max(tmp555)),type = 'l',main="Y-hat"),T) }
            for(i3 in 2:nrow(tmp555)){ points(tmp555[i3,],type = 'l',col = ifelse(reri == 1, yes = "black", no = "gray"),
                                           lwd = ifelse(reri == 1, yes = 2, no = 2),
                                           lty = ifelse(reri == 1, yes = 1, no = 2)) }
          }

          # plot compare against truth
          tmp_true <- np$array(batch_l_cal$YTrue_out)[,,1]
          try(plot(tmp_true[1,],ylim = c(0,max(tmp_true)),type = 'l',main="Y-true"),T)
          for(i3 in 2:nrow(tmp555)){ points(tmp_true[i3,],type = 'l',col = "black",
                                         lwd = 2,
                                         lty = 1) }
          print(sprintf("Current cor with truth: %.3f", cor(c(tmp555),c(tmp_true))))
          
          plot(c(tmp555),c(tmp_true),
               xlim = summary(c(tmp555,tmp_true))[c(1,6)],
               ylim = summary(c(tmp555,tmp_true))[c(1,6)]); abline(a=0,b=1); 
          plot(c(tmp555),c(tmp_true),
               xlim = summary(c(tmp555,tmp_true))[c(1,6)],
               ylim = summary(c(tmp555,tmp_true))[c(1,6)]);abline(a=0,b=1)

          print2(sprintf("Within iteration mean correlation: %.10f", mean( diag( cor(t(tmp_), t(tmp555))) ) ))
          if(plottingSeq_counter == 1){ sharedXSamp <- sample(1:length(tmp555),length(tmp555)/10) }
          if(plottingSeq_counter > 1){
            crossIterCor_vec <- c(crossIterCor_vec, cor(tmp555[sharedXSamp], tmpSamp_minus1))
            print2( sprintf("Across iteration correlation: %.10f",crossIterCor_vec[ length(crossIterCor_vec) ]) )
          }
          if(i == 1){ tmpSamp_t0 <- tmp555[sharedXSamp] }
          tmpSamp_minus1 <- tmp555[sharedXSamp] # update minus 1 value
          #rm(tmp555,tmp_,tmp_true)
          rm(tmp_,tmp_true)
      }
      if( (i > 10 & i %% 10 == 5) & is.na(COMMAND_ARG_INPUT) ){
        par(mfrow=c(1,2)); try({plot(rank(na.omit(in_loss_vec)),main="Loss"); points(smooth.spline(rank(na.omit(in_loss_vec))),type = "l",lwd=3)},T)
        try({plot(grad_norm_vec[1:i],log = 'y',main="Gradients")},T)
      }
      }

      # print results
      if (i == 1 || (is.na(COMMAND_ARG_INPUT) && i %% 4 == 0L) ||
            (!is.na(COMMAND_ARG_INPUT) && i %% 20 == 0L)) {
          print2( sprintf("Iter: %s of %s -- Loss: %.4f (%.3f%%) -- Last OutCor %.2f: -- Last OutSkill: %.2f -- Total (s): %s -- Grad & updates (s): %.3f",
                            i, nSGD_model, 
                            myLoss_fromGrad,
                            100*(1-rank(in_loss_vec[1:i])[i]/i),
                            LastOutCor, Skill8SanityCheck,
                            te_total <- round(difftime(Sys.time(), fulliter_timer, units = "secs"),2L), 
                            te_grads <- round(difftime(Sys.time(), gd_timer, units = "secs"),2L) )   )
      }
      write.csv(file = './TrainDoStepTime.csv', as.matrix(te_total))
    }
  }
  }
}
print("Done with SuperLModel_TrainDo.R")
