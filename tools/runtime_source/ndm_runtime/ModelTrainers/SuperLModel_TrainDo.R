# Content of SuperLModel_TrainDo.R
print("Sarting SuperLModel_TrainDo.R")
SavedModelDir <- get0(
  "SavedModelDir",
  ifnotfound = sprintf("./SavedModels/%s/Model_%s_%s",
                       ifelse(grepl(tolower(AnalysisName), pattern = "sim"),
                              yes = "FromSim", no = "FromReal"),
                       AnalysisName,
                       AnalysisDate)
)
checkpoint_file <- function(prefix, step){
  file.path(SavedModelDir, sprintf("%s_i%s_%s_%s.eqx", prefix, step, AnalysisName, AnalysisDate))
}
reset_train_iterator <- function(){
  TFDatasetIterator_train <<- reticulate::as_iterator(TFDataset_train)
  TFDatasetIterator_train
}
batch_has_expected_shape <- function(dat_, expected_batch_size){
  if("try-error" %in% class(dat_) || is.null(dat_) || length(dat_) == 0L){
    return(FALSE)
  }
  batch_dims <- try(vapply(dat_, function(l_){ as.integer(np$array(l_$shape)[[1]]) }, integer(1)), TRUE)
  if("try-error" %in% class(batch_dims) || length(batch_dims) == 0L || any(is.na(batch_dims))){
    return(FALSE)
  }
  batch_size <- batch_dims[[1]]
  batch_size > 0L && batch_size == expected_batch_size && all(batch_dims == batch_size)
}
next_train_batch <- function(max_attempts = 100L){
  for(attempt_i in seq_len(max_attempts)){
    dat_ <- try(reticulate::iter_next(TFDatasetIterator_train), TRUE)
    if(batch_has_expected_shape(dat_, nBatch)){
      return(TFConst2JAXArray(dat_))
    }
    print2(sprintf("Resetting train iterator in TrainDo.R [attempt %s]", attempt_i))
    reset_train_iterator()
  }
  stop("Too many malformed batches in TrainDo.R")
}
save_eqx_enabled <- isTRUE(get0("SaveEqx", ifnotfound = TRUE))
recover_checkpoint_at <- get0("RecoverCheckpointAt", ifnotfound = NULL)
if(isTRUE(recover_checkpoint_at)){
  recover_checkpoint_at <- "last"
}
checkpoint_has_scale_state <- function(){
  exists("SIM_GLOBAL_SCALE_MEAN", inherits = TRUE) &&
    exists("SIM_GLOBAL_SCALE_SD", inherits = TRUE)
}
checkpoint_scale_state <- function(){
  if(!checkpoint_has_scale_state()){
    return(NULL)
  }
  jnp$array(list(SIM_GLOBAL_SCALE_MEAN, SIM_GLOBAL_SCALE_SD))
}
checkpoint_model_payload <- function(){
  payload <- list(ModelList, state, opt_state)
  scale_state <- checkpoint_scale_state()
  if(!is.null(scale_state)){
    payload[[4]] <- scale_state
  }
  payload
}
restore_checkpoint_scale_state <- function(recovered_payload){
  if(length(recovered_payload) < 4L || is.null(recovered_payload[[4]])){
    return(invisible(NULL))
  }
  SIM_GLOBAL_SCALE_MEAN <<- jnp$take(recovered_payload[[4]], 0L, axis = 0L)$tolist()
  SIM_GLOBAL_SCALE_SD <<- jnp$take(recovered_payload[[4]], 1L, axis = 0L)$tolist()
  invisible(NULL)
}
ndm_condition_message <- utils::getFromNamespace(".ndm_condition_message", "ndm")
ndm_numeric_summary <- utils::getFromNamespace(".ndm_numeric_summary", "ndm")
ndm_first_nonfinite_name <- utils::getFromNamespace(".ndm_first_nonfinite_name", "ndm")
ndm_write_nonfinite_report <- utils::getFromNamespace(".ndm_write_nonfinite_report", "ndm")
nonfinite_empty_summary <- function(error = NA_character_){
  list(
    error = error,
    dims = integer(),
    length = 0L,
    finite_fraction = NA_real_,
    n_nonfinite = NA_integer_,
    n_nan = NA_integer_,
    n_inf = NA_integer_,
    min = NA_real_,
    max = NA_real_,
    mean = NA_real_,
    first_nonfinite_index = integer()
  )
}
nonfinite_capture_summary <- function(expr){
  value <- try(eval.parent(substitute(expr)), silent = TRUE)
  if(inherits(value, "try-error")){
    return(nonfinite_empty_summary(ndm_condition_message(value)))
  }
  ndm_numeric_summary(value, np = np)
}
capture_nonfinite_learning_rate <- function(iteration){
  if(!exists("LR_schedule_vec", inherits = TRUE) || length(LR_schedule_vec) < iteration){
    return(NA_real_)
  }
  suppressWarnings(as.numeric(LR_schedule_vec[[iteration]])[[1L]])
}
capture_nonfinite_base_id <- function(){
  if(!exists("SimEntry", inherits = TRUE) || is.null(SimEntry) || is.null(SimEntry$BaseID)){
    return(NA_integer_)
  }
  suppressWarnings(as.integer(as.numeric(SimEntry$BaseID)[[1L]]))
}
capture_nonfinite_report <- function(batch_l,
                                     loss_value,
                                     grad_norm_value,
                                     iteration,
                                     keys_mat){
  prediction <- try(
    GetPred_train_jit(
      ModelList,
      batch2package(batch_l),
      state,
      PriorList,
      PolicyList,
      GetPredSaveAtInfo_default,
      keys_mat
    ),
    silent = TRUE
  )

  tensor_summaries <- list(
    batch_YTrue_out = nonfinite_capture_summary(batch_l$YTrue_out),
    batch_YTrue_out_mask = nonfinite_capture_summary(batch_l$YTrue_out_mask),
    batch_YTrue = nonfinite_capture_summary(batch_l$YTrue),
    batch_YTrue_mask = nonfinite_capture_summary(batch_l$YTrue_mask),
    batch_XPred = nonfinite_capture_summary(batch_l$XPred),
    batch_XPred_mask = nonfinite_capture_summary(batch_l$XPred_mask)
  )

  if(inherits(prediction, "try-error")){
    tensor_summaries$prediction_forward_pass <- nonfinite_empty_summary(ndm_condition_message(prediction))
  } else {
    pred_obj <- prediction[[1]]
    tensor_summaries$prediction_y_mu <- nonfinite_capture_summary(pred_obj$y_mu)
    tensor_summaries$prediction_y_sigma <- nonfinite_capture_summary(pred_obj$y_sigma)
    tensor_summaries$prediction_center_param <- nonfinite_capture_summary(pred_obj$ODEParamsSampList$center_param)
    tensor_summaries$prediction_scale_param <- nonfinite_capture_summary(pred_obj$ODEParamsSampList$scale_param)
    tensor_summaries$prediction_s_l <- nonfinite_capture_summary(pred_obj$ODEParamsSampList$s_l_samp)
    tensor_summaries$prediction_e_l <- nonfinite_capture_summary(pred_obj$ODEParamsSampList$e_l_samp)
    tensor_summaries$prediction_i_l <- nonfinite_capture_summary(pred_obj$ODEParamsSampList$i_l_samp)
    tensor_summaries$prediction_r_l <- nonfinite_capture_summary(pred_obj$ODEParamsSampList$r_l_samp)
    tensor_summaries$prediction_raw_beta_t <- nonfinite_capture_summary(pred_obj$ODEParamsSampList$diff_eq_sol_ys$raw_beta_t)
    tensor_summaries$prediction_raw_policy_t <- nonfinite_capture_summary(pred_obj$ODEParamsSampList$diff_eq_sol_ys$raw_policy_t)
    tensor_summaries$prediction_neural1 <- nonfinite_capture_summary(pred_obj$ODEParamsSampList$diff_eq_sol_ys$Neural1)
  }

  outer_iteration <- get0("OUTER_ITERATION", ifnotfound = NA_integer_)
  report <- list(
    analysis_name = get0("AnalysisName", ifnotfound = NA_character_),
    outer_iteration = outer_iteration,
    iteration = as.integer(iteration),
    base_id = capture_nonfinite_base_id(),
    model_type = get0("ModelType", ifnotfound = NA_character_),
    n_samples_train = suppressWarnings(as.integer(get0("nSamplesTrain", ifnotfound = NA_integer_))),
    model_depth = suppressWarnings(as.integer(get0("ModelDepth", ifnotfound = NA_integer_))),
    model_dims = suppressWarnings(as.integer(get0("ModelDims", ifnotfound = NA_integer_))),
    context_length = suppressWarnings(as.integer(get0("nTimesPast", ifnotfound = NA_integer_))),
    learning_rate = capture_nonfinite_learning_rate(iteration),
    loss = loss_value,
    grad_norm = grad_norm_value,
    sim_entry = if(exists("SimEntry", inherits = TRUE)) SimEntry else NULL,
    tensor_summaries = tensor_summaries,
    first_bad_tensor = ndm_first_nonfinite_name(tensor_summaries)
  )
  report_path <- ndm_write_nonfinite_report(
    report = report,
    holder_folder = HolderFolder,
    outer_iteration = outer_iteration,
    iteration = iteration
  )
  list(report = report, report_path = report_path)
}
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
      zip::zip(zipfile = file.path(SavedModelDir, sprintf("AnalysisR_%s_%s.zip", AnalysisName, AnalysisDate)),
               files = NDM_INTERNAL_ANALYSIS_DIR)
    }

    if(i %in% CheckPointSaveAt){ save_i <- i }
    if(save_eqx_enabled){ 
      # save_i <- "last"
      eq$tree_serialise_leaves(checkpoint_file("ModelList", save_i),
                               checkpoint_model_payload())
      if("ModelList_notshared_set" %in% ls()){
        eq$tree_serialise_leaves(checkpoint_file("ModelList_notshared_set", save_i),
                                 list(ModelList_notshared_set, # nonshared parameters 
                                      lapply(sim_dat_norm_list,jnp$array)) # means for recovery 
                                 ) 
      }
    }

    # recover trained model
    if(save_eqx_enabled && !is.null(recover_checkpoint_at)){
      # nSGD_model / 1:4 # defines i's
      RecoverAt <- recover_checkpoint_at
      ModelList_recovered <- eq$tree_deserialise_leaves(
        checkpoint_file("ModelList", RecoverAt),
        checkpoint_model_payload()) 
      if("ModelList_notshared_set" %in% ls()){
        ModelList_notshared_set_recovered <- eq$tree_deserialise_leaves(checkpoint_file("ModelList_notshared_set", RecoverAt),
                                 list(ModelList_notshared_set, # non shared parameters 
                                      lapply(sim_dat_norm_list,jnp$array) ) # norm factors 
                                 )
        sim_dat_norm_list <- ModelList_notshared_set_recovered[[2]] 
        sim_dat_norm_list <- lapply(sim_dat_norm_list,function(l_){
          SIM_GLOBAL_SCALE_MEAN_ <- jnp$take(l_,0L,axis=0L)$tolist()
          SIM_GLOBAL_SCALE_SD_ <- jnp$take(l_,1L,axis=0L)$tolist()
          list(SIM_GLOBAL_SCALE_MEAN_, SIM_GLOBAL_SCALE_SD_)
        })
        restore_checkpoint_scale_state(ModelList_recovered)
        ModelList_notshared_set <- ModelList_notshared_set_recovered[[1]]
      }

      # confirm 
      restore_checkpoint_scale_state(ModelList_recovered)
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
  dat_ <- next_train_batch()

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
      state <- state_tmp

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
        if(is.na(COMMAND_ARG_INPUT)){
          print(tail(sort(unlist(values3),decreasing=F),25))
          mean(duplicated(names(sort(unlist(values3)))))
          plot(sort(unlist(values3)))
          
          if(i > 1){ 
            plot(unlist(values3), unlist(values3_past)); abline(a=0,b=1)
          }
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
      Loss_i <- in_loss_vec[i] <- suppressWarnings(as.numeric(myLoss_fromGrad)[[1L]])
      GradNorm_i <- grad_norm_vec[i] <- suppressWarnings(as.numeric(np$array(
        optax$global_norm(jax$tree_util$tree_leaves(GradientUpdatePackage))
      ))[[1L]])

      # update parameters if passing thru all checks
      UpdateParametersCond <- is.finite(Loss_i) & is.finite(GradNorm_i)
      if(! UpdateParametersCond ){
        nonfinite_capture <- capture_nonfinite_report(
          batch_l = dat_,
          loss_value = Loss_i,
          grad_norm_value = GradNorm_i,
          iteration = i,
          keys_mat = keys_mat
        )
        stop(
          sprintf(
            paste(
              "Non-finite training state in %s",
              "[outer=%s, BaseID=%s, ModelType=%s, iter=%s, loss=%s, grad_norm=%s].",
              "Saved debug report to %s.",
              sep = " "
            ),
            get0("AnalysisName", ifnotfound = "Analysis2"),
            get0("OUTER_ITERATION", ifnotfound = NA_integer_),
            capture_nonfinite_base_id(),
            get0("ModelType", ifnotfound = NA_character_),
            i,
            format(Loss_i, scientific = TRUE),
            format(GradNorm_i, scientific = TRUE),
            nonfinite_capture$report_path
          )
        )
      }
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
        rm(GradientUpdatePackage, dat_)
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
