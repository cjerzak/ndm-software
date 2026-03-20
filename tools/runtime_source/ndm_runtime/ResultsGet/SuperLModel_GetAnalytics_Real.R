# Content of SuperLModel_GetAnalytics_Real.R
if( !SimMode ){
    print("Starting SuperLModel_GetAnalytics_Real.R")
    ndm_plot_log_series_safe <- getFromNamespace(".ndm_plot_log_series_safe", "ndm")
    # get out-of-sample results
    #try(plot(rank(na.omit(in_loss_vec))), T) ; 
    #try(points(smooth.spline(rank(na.omit(in_loss_vec))),type = "l",lwd=3), T) 
  
    # reset iterator and do analysis 
    gc(); py_gc$collect()
    TFDatasetIterator_inference <- reticulate::as_iterator( TFDataset_inference )
    inference_batch_dims <- function(batch_l){
      if("try-error" %in% class(batch_l) || is.null(batch_l) || length(batch_l) == 0L){
        return(integer())
      }
      try(vapply(batch_l, function(l_){ as.integer(np$array(l_$shape)[[1]]) }, integer(1)), TRUE)
    }
    sl_dat <- c();ok_<-F; ok_counter_ <- 0; while(!ok_){ # !ok_ means do a batch
          ok_counter_ <- ok_counter_ + 1
          batch_l <- try(reticulate::iter_next( TFDatasetIterator_inference ), T) 
          
          # reasons not to go on 
          if( "try-error" %in% class(batch_l) | is.null(batch_l) ){ ok_ <- T }
          if( !ok_ ){
            batch_dims <- inference_batch_dims(batch_l)
            if("try-error" %in% class(batch_dims) || length(batch_dims) == 0L || batch_dims[[1]] == 0L){
              ok_ <- T
            }
            if(!ok_ && any(batch_dims != batch_dims[[1]])){
              print2("Skipping malformed inference batch in GetAnalytics_Real.R...")
              ok_ <- T
            }
            if(!ok_){
              batch_l <- TFConst2JAXArray( batch_l )
            }
          }
          
          # if going in do this
          if(!ok_){
          print2(sprintf("GetAnalytics_Real [@ sgd iter: %s, analytics iter: %s]", i, ok_counter_))
          #plot(as.matrix(batch_l$XPred_Mask[1,,])[,100],as.matrix(batch_l$XPred[1,,])[,100])
          #input_df_red_full[input_df_red_full$location_id %in% loc_id & input_df_red_full$time_id %in% time_id,]
          if(class(batch_l)!="try-error"){
            #if(!is.null(batch_l) & !all(batch_l$is_null_indicator) & (length(batch_l_prior)>0)){
            #if(!is.null(batch_l)  & (length(batch_l_prior)>0)){
            #if(place_counter__ == 1 & time_counter__ == 1 & outSampCounter == 1){
            {
                # lapply(unlist(batch2package(batch_l)), function(l_){paste0(l_$shape)})
                 add_pred_all <- replicate(5,list(try(GetPred_inference(
                                    ModelList, batch2package(batch_l),
                                    #ModelList, batch2package(batch_l_train),
                                    state, PriorList, PolicyList,
                                    GetPredSaveAtInfo_default,
                                    jax$random$split(JaxKey(999L), batch_l$XPred$shape[[1]]))[[1]],T)))
                                    #jax$random$split(JaxKey(999L), batch_l_train$XPred$shape[[1]]))[[1]],T)))
                 add_pred_all <- add_pred_all[!(unlist(lapply(add_pred_all,class)) %in% 'try-error')]
                 plot( np$array(add_pred_all[[1]]$y_mu)[,,1], 
                       np$array(batch_l$YTrue_out)[,,1],
                       main = sprintf("Just1 at %s [Cor: %.3f]",ok_counter_,
                               cor(c(np$array(add_pred_all[[1]]$y_mu)[,,1]),
                                   c(np$array(batch_l$YTrue_out)[,,1]))));abline(a=0,b=1)
                 # plot( np$array(add_pred_all[[1]]$y_mu)[,,1], np$array(batch_l_train$YTrue_out)[,,1]);abline(a=0,b=1)
                 # add_pred_all[[1]]$y_mu$shape
                 # batch_l$XPred$shape
                 # batch_l$XPred_mask
                 # batch_l$XPred$mean(axis = 0L:1L)
  
                # analyze time dependent parameters
                at_t <- try(colMeans(do.call(rbind,lapply(add_pred_all,function(zer){
                                        c(ExtractBetaDraw(zer$ODEParamsSampList)$TSDraw) }))),T)
                beta_hat_at_t <- try(colMeans(do.call(rbind,lapply(add_pred_all,function(zer){c(ExtractBetaDraw(zer$ODEParamsSampList)$BetaDraw)}))),T)
                s_average <- try(colMeans(do.call(rbind,lapply(add_pred_all,function(zer){c(np$asanyarray(zer$ODEParamsSampList$s_l_samp) )}))),T)
                e_average <- try(colMeans(do.call(rbind,lapply(add_pred_all,function(zer){c(np$asanyarray(zer$ODEParamsSampList$e_l_samp) )}))),T)
                i_average <- try(colMeans(do.call(rbind,lapply(add_pred_all,function(zer){c(np$asanyarray(zer$ODEParamsSampList$i_l_samp) )}))),T)
                r_average <- try(colMeans(do.call(rbind,lapply(add_pred_all,function(zer){c(np$asanyarray(zer$ODEParamsSampList$r_l_samp) )}))),T)
  
                add_sd_all <- colMeans(do.call(rbind,lapply(add_pred_all,function(zer){np$asanyarray(zer$y_sigma) })))
                #if(mean(abs(add_sd_all)) < 0.01){browser() }
                # plot(  sort(  input_df_red_full[input_df_red_full$location_id %in% loc_id,]$time_id ) )
                # plot(  sort(  input_df_red_full[input_df_red_full$location_id %in% place_pool,]$time_id ) )
                
                #mean(input_df_red_in$time_id %in% input_df_red_full$time_id)
                if(!UseShortOutcomes){ 
                  add_sd_past <- t(add_sd_all[c(1:abs(maxTimesPast))])
                  add_sd_out <- t(add_sd_all[-c(1:abs(maxTimesPast))])
                }
                if(UseShortOutcomes){ 
                  add_sd_past <- t(add_sd_all); add_sd_out <- t(add_sd_all)
                }
  
                # obtain posterior predicted mean
                add_pred_all_reduced <- Reduce(`+`,
                                               lapply(add_pred_all, 
                                                      function(l_){np$asanyarray(l_$y_mu)[,,1] })) / 
                                                 length(add_pred_all)
                add_pred_all_reduced[add_pred_all_reduced<0] <- 0
  
                if(UseShortOutcomes){ 
                  add_pred_past <- (add_pred_all_reduced)
                  add_pred_out <- (add_pred_all_reduced)
                }
                if(!UseShortOutcomes){ 
                  add_pred_past <- (add_pred_all_reduced[,c(1:abs(maxTimesPast))])
                  add_pred_out <- (add_pred_all_reduced[,-c(1:abs(maxTimesPast))])
                }
  
                # subset truth (no need to change subsetting here for short outcomes [using YTrue])
                add_true_prior <- (np$asanyarray(batch_l$YTrue)[,c(1:abs(maxTimesPast)),])
                add_true_prior_mask <- (np$asanyarray(batch_l$YTrue_mask)[,c(1:abs(maxTimesPast)),])
                
                add_true_out <- (np$asanyarray(batch_l$YTrue)[,-c(1:abs(maxTimesPast)),])
                # max(abs(add_true_out - np$asanyarray(batch_l$YTrue_out)[,,1])) # sanity 0 value 
                add_true_out_mask <- (np$asanyarray(batch_l$YTrue_mask)[,-c(1:abs(maxTimesPast)),])
                add_true_prior[add_true_prior_mask == 0] <- NA
                add_true_out[add_true_out_mask == 0] <- NA
                
                if(length(dim(add_true_out)) > 2 & tail(dim(add_true_out),1)>1){ add_true_out <- add_true_out[,,1] }
                if(length(dim(add_true_prior)) > 2 & tail(dim(add_true_prior),1)>1){ add_true_prior <- add_true_prior[,,1] }
                # View(add_true_out[,,1])
                # View(add_true_out_mask[,,1])
  
                # barplot
                #par(mfrow = c(1,1))
                #seir_ <- c(s_average,e_average,i_average,r_average)
                #names(seir_) <- c("S[0]","E[0]","I[0]","R[0]")
                #barplot(seir_ , add = F, density = 0.3, horiz  = F)
  
                # beta plot
                #try(plot(at_t , beta_hat_at_t,type = "b",xlab="Time (t) in Times"),T)
              }
              if("try-error" %in% class(add_true_prior)){
                add_true_prior <- add_true_out
                add_true_prior[] <- NA
              }
              # par(mfrow=c(1,1));plot(x_<-c(add_true_out), y_<-c(add_pred_out));abline(a=0,b=1); print(cor(x_,y_, use = "pairwise.complete"))
              colnames(add_pred_out) <- c(paste("Pred_l",1:ncol(add_pred_out),sep=""))
              colnames(add_true_out) <- c(paste("Truth_l",1:ncol(add_true_out),sep=""))
              colnames(add_true_prior) <- c(paste("PreviousSequenceTruth_l",1:ncol(add_true_prior),sep=""))
              
              # depreciate PredBase
              add_pred_out_baseline <- add_pred_out
              #add_pred_out_baseline[] <- add_true_prior[length(add_true_prior)]
              add_pred_out_baseline[] <- NA
              colnames(add_pred_out_baseline) <- gsub(colnames(add_pred_out_baseline),
                                                   pattern = "Pred", replace = "PredBase")
            
              # jnp$mean(jnp$array(batch_l$XPred),axis = 0L:1L)
              # jnp$std(jnp$array(batch_l$XPred),axis = 0L:1L)
              plot(add_true_out, add_pred_out, main=sprintf("%s [Cor: %.3f]",ok_counter_, LastOutCor <- cor(c(add_true_out),c(add_pred_out)))); abline(a=0,b=1)
              
              # combine results 
              sl_dat <- rbind(sl_dat, cbind("location_id" = np$array(batch_l$location_id_numeric),
                                         "time_anchor_id" = np$array(batch_l$time_id_numeric),#[!batch_l$is_null_indicator],
                                         cbind(add_true_out,
                                               add_pred_out,
                                               add_pred_out_baseline,
                                               add_true_prior) ) )
              # plot(f2n(sl_dat[,"Truth_l8"]), f2n(sl_dat[,"Pred_l8"]));abline(a=0,b=1)
              # plot(f2n(sl_dat[,"Truth_l8"]), f2n(sl_dat[,"Truth_l1"]));abline(a=0,b=1)
          print2(sprintf("Skill sanity: %.3f",
              (Skill8SanityCheck <- 1 - (sum(( f2n(sl_dat[,"Truth_l8"]) - f2n(sl_dat[,"Pred_l8"]) )^2)+0.01)/
                  ( sum(( f2n(sl_dat[,"Truth_l8"]) - f2n(sl_dat[,"Truth_l1"]) )^2) +0.01))))
          }
        }
    }
  
    sl_dat <- eval(parse(text = sprintf('cbind(sl_dat,
                    %s,
                    "modelingStrategy_name" = modelingStrategyNameKey,
                    "nSGD" = nSGD_model,
                    "nBatch" = nBatch,
                    "maxTimesPast" = maxTimesPast, # past context
                    "evaluationTime" = evaluationTime,
                    "evaluationMethod" = evaluationMethod,
                    "OUTER_ITERATION" = OUTER_ITERATION,
                    "i_in_sgd" = i,
                    "te_total" = te_total, 
                    "te_grads" = te_grads, 
                    "nTrainingSamplesSeen" = i*nBatch, 
                    "Skill8SanityCheck" = Skill8SanityCheck, 
                    "atEpoch" = i*nBatch/nSamplesTrain, 
                    "nParamsModel" = nParamsModel, 
                    "maxInSampleTime_id" = max( input_df_red_in$time_id ),
                    "model_id" = rlang::hash(modelingStrategyNameKey) )',
                      paste(paste("'",names(unlist(RealEntry)), "'='", unlist(RealEntry),"'", sep=""),collapse=",")
                      ) ))
    sl_dat <- as.data.frame( sl_dat )
    sl_dat$time_anchor_id <- unlist( sl_dat$time_anchor_id )
    #plot(abs(f2n(sl_dat$PredBase_l6) - f2n(sl_dat$Truth_l6)), abs(f2n(sl_dat$Pred_l6) - f2n(sl_dat$Truth_l6))); abline(a=0,b=1)
    #plot(abs(f2n(sl_dat$PredBase_l12) - f2n(sl_dat$Truth_l12)), abs(f2n(sl_dat$Pred_l12) - f2n(sl_dat$Truth_l12))); abline(a=0,b=1)

    #summary(abs(f2n(sl_dat$PredBase_l4) - f2n(sl_dat$Truth_l4)))
    #summary(abs(f2n(sl_dat$Pred_l4) - f2n(sl_dat$Truth_l4)))

    #summary(abs(f2n(sl_dat$PredBase_l6) - f2n(sl_dat$Truth_l6)))
    #summary(abs(f2n(sl_dat$Pred_l6) - f2n(sl_dat$Truth_l6)))

    #summary(abs(f2n(sl_dat$PredBase_l12) - f2n(sl_dat$Truth_l12)))
    #summary(abs(f2n(sl_dat$Pred_l12) - f2n(sl_dat$Truth_l12)))

    #np$array(ModelList[[1]][[3]][[1]])
    #np$array(ModelList[[1]][[3]][[2]])
    #plot( predict(prcomp(np$array(ModelList[[1]][[3]][[1]])))[,1:2])
    #plot( predict(prcomp(np$array(ModelList[[1]][[3]][[2]])))[,1:2])
    #tapply(truth_df_red$location_id_numeric,truth_df_red$location_id_numeric,unique)

    #  write loss fig for debugging (note: files are re-written)
    pdf(file.path(
      HolderFolder,
      sprintf("diagnostics_%s_EvalTime%s_IsPretraining%s_OuterKey%s.pdf",
              rlang::hash(modelingStrategyNameKey),
              evaluationTime,
              IsPretraining,
              OUTER_ITERATION)
    ), height = 10, width = 5)
    {
    par(mfrow=c(2,1 + is.na(COMMAND_ARG_INPUT) ))
    loss_obs <- na.omit(in_loss_vec)
    grad_obs <- na.omit(grad_norm_vec)
    ndm_plot_log_series_safe(loss_obs, main = "Loss")
    if(length(loss_obs) > 0L){ plot(rank(loss_obs), log = "") } else { plot.new() }
    ndm_plot_log_series_safe(grad_obs, main = "Gradients")
    if(is.na(COMMAND_ARG_INPUT)){
      tmp555 <- as.matrix(tmp555);try(plot(tmp555[1,],ylim = c(0,max(c(tmp555[1:10,]))),type = 'l',main="Y-hat"),T)
      for(i3 in 2:10){ try(points(tmp555[i3,],type = 'l',col = "black", lwd = 2, lty = 1),T) }
    }
    }
    dev.off(); par(mfrow = c(1,1))
    
    # save results 
    data.table::fwrite(
      as.matrix(sl_dat),
      file = as.character(file.path(
        HolderFolder,
        sprintf("predicted_df_out_%s_EvalTime%s_IsPretraining%s_TrainIter%sof%s_OuterIter%s.csv",
                rlang::hash(modelingStrategyNameKey),
                evaluationTime,
                IsPretraining,
                i,
                nSGD_model,
                OUTER_ITERATION)
      ))
    )
    print("Done with SuperLModel_GetAnalytics_Real.R")
}

# load(file = "./tmp_input_df_red_full.Rdata");
# full sample
#place_pool <- sample(unique(input_df_red_out$location_id[input_df_red_out$location_id %in% input_df_red_in$location_id]))

# partial sample
# place_pool <- sample(unique(input_df_red_out$location_id[input_df_red_out$location_id %in% input_df_red_in$location_id]), 50)
# place_counter__ <- 0; for(loc_id in place_pool){
# out of sample targets - how to  gen the input data for the base learners?
#for(time_iter in time_iter_pool <- times_out){
