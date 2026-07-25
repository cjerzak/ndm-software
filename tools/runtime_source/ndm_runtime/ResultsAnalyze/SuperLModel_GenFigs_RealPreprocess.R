# Content of SuperLModel_GenFigs_RealPreprocess.R
{
  # get some parameters 
  nMax_PreviousSequenceTruths <- length(grep(names(res_mat),pattern="PreviousSequenceTruth_l", value = T))
  nMax_FutureTruth <- length(grep(names(res_mat),pattern="^Truth_l", value = T))
  
  # create aggregate measures across model_id (based on modelingStrategyNameKey, unique for each grid entry)
  # note: model_id is unique within grid rows; confirm via: 
  # table((res_mat[which(res_mat$model_id==res_mat$model_id[1]),])$ModelType)
  # table((res_mat[which(res_mat$model_id==res_mat$model_id[1]),])$ModelDims)
  res_mat$master_conditioning_id <- paste0(res_mat$model_id,
                                           "_", res_mat$i_in_sgd)
  table(table(res_mat$master_conditioning_id))

  # process baselines 
  if(FALSE){
  # future.apply version -> hurts memory 
  options(future.globals.maxSize = 26000 * 1024^2) # Set to 7 GB
  #DT_skill <- future.apply::future_lapply(1:nrow(res_mat), # parallelize, but memory problems
  DT_skill <- lapply(1:nrow(res_mat),
                              function(zer){ 
                                #if(zer %% 1000 == 0){ print2(sprintf("At %s of %s [%.2f%%]",zer, nrow(res_mat), 100*zer/nrow(res_mat))) } 
                                if(zer %% 1000 == 0){ write.csv(c(zer, nrow(res_mat)), file = '~/Downloads/i_.csv') }
                                res_mat_red <- res_mat[zer,]
                                
                                # sanity checks 
                                # mean(is.na(res_mat$PreviousSequenceTruth_l1))
                                # hist(res_mat$PreviousSequenceTruth_l1)
                                # grep(names(res_mat),pattern="PreviousSequenceTruth_", value = T)
                                # grep(names(res_mat),pattern="^Truth_l", value = T)
                                # grep(names(res_mat),pattern="^Pred_l", value = T)
                                # plot(unlist(res_mat_red[,paste0("PreviousSequenceTruth_l",1:nMax_PreviousSequenceTruths)]))
                                # plot(na.omit(unlist(res_mat_red[,paste0("PreviousSequenceTruth_l",1:nMax_PreviousSequenceTruths)])))
                                # plot(na.omit(unlist(res_mat_red[,paste0("Truth_l",1:nMax_FutureTruth)])))
                                
                                # sanity confirm continuity -- YES 
                                # plot(c(na.omit(unlist(res_mat_red[,paste0("PreviousSequenceTruth_l",1:nMax_PreviousSequenceTruths)])), na.omit(unlist(res_mat_red[,paste0("Truth_l",1:nMax_FutureTruth)]))))
                                
                                pred_l_baselineVal <- na.omit(unlist(res_mat_red[,paste0("PreviousSequenceTruth_l",
                                                                                         1:nMax_PreviousSequenceTruths)]))
                                pred_l_baselineVal <- pred_l_baselineVal[length(pred_l_baselineVal)]
                                if(length(pred_l_baselineVal) == 0){ pred_l_baselineVal <- NA }
                                l_true <- unlist(res_mat_red[,paste0("Truth_l",1:nMax_FutureTruth)])
                                pred_l_mean <- unlist(res_mat_red[,paste0("Pred_l",1:nMax_FutureTruth)])
                                
                                RSS_baseline <-  (pred_l_baselineVal - l_true)^2
                                RSS_pred <- (pred_l_mean - l_true)^2
                                SliceSkill <- 1 - (0.01+RSS_pred^0.5) / (0.01+RSS_baseline^0.5)
                                names(SliceSkill) <- paste("SliceSkillTime",1:nMax_FutureTruth, sep = "")
                                names(RSS_pred) <- paste("RSSPredTime",1:nMax_FutureTruth, sep = "")
                                names(RSS_baseline) <- paste("RSSBaselineTime",1:nMax_FutureTruth, sep = "")
                                return( c(RSS_baseline, RSS_pred, SliceSkill) )
                              })
  DT_skill <- as.data.frame(do.call(rbind, DT_skill))
  # plot(res_mat$i_in_sgd[1:100], DT_skill$SliceSkillTime12 ) 
  # mean(colnames(res_mat) %in% colnames(DT_skill))
  }

  # vectorized version 
  if(TRUE){   
    mem.maxVSize(96000) # -> forces swapping 
    #mem.maxVSize(36000)
    DT <- as.data.table(res_mat)           # data.table is fast because of shallow copies
    
    ## ------------------------------------------------------------------
    ## 1. Pull the three blocks of columns into dense matrices ------------
    ## ------------------------------------------------------------------
    prev_cols  <- grep("^PreviousSequenceTruth_l",  names(DT), value = TRUE)
    truth_cols <- grep("^Truth_l",                  names(DT), value = TRUE)
    pred_cols  <- grep("^Pred_l",                   names(DT), value = TRUE)
    
    prev_mat  <- (as.matrix(DT[, ..prev_cols]))       # n × P
    truth_mat <- (as.matrix(DT[, ..truth_cols]))      # n × H
    pred_mat  <- (as.matrix(DT[, ..pred_cols]))       # n × H
    
    H <- ncol(truth_mat)                            # number of forecast times
    
    ## ------------------------------------------------------------------
    ## 2. One vector that holds "y[t‑1]" for each row --------------------
    ## ------------------------------------------------------------------
    # index of last non‑NA element in each row
    last_idx <- max.col(!is.na(prev_mat), ties.method = "last")
    baseline_vec <- prev_mat[cbind(seq_len(nrow(prev_mat)), last_idx)]   # length = n
    
    ## replicate once; the next steps are pure BLAS‑level matrix ops
    baseline_mat <- matrix(baseline_vec, nrow = nrow(prev_mat), ncol = H)
    
    ## ------------------------------------------------------------------
    ## 3. Compute RSS and skill all at once ------------------------------
    ## ------------------------------------------------------------------
    rss_base <- (baseline_mat - truth_mat)^2
    rss_pred <- (pred_mat     - truth_mat)^2
    
    skill_mat <- 1 - (0.01 + sqrt(rss_pred)) / (0.01 + sqrt(rss_base))
    
    ## ------------------------------------------------------------------
    ## 4. Assemble result ------------------------------------------------
    ## ------------------------------------------------------------------
    DT_skill <- cbind(
      as.data.table(rss_base, keep.rownames = FALSE),
      as.data.table(rss_pred, keep.rownames = FALSE),
      as.data.table(skill_mat, keep.rownames = FALSE)
    )
    
    setnames(DT_skill,
             c(paste0("RSSBaselineTime", 1:H),
               paste0("RSSPredTime",     1:H),
               paste0("SliceSkillTime",  1:H)))
    
    # add to the main table (shallow copy; fast)
    #Sys.setenv(R_MAX_VSIZE = "40Gb")
    #library(float);DT <- fl(DT); out <- fl(out)
    #DT <- cbind(DT, DT_skill)
    #rm(out)
    #res_mat <- as.data.frame(DT)
    # rm(DT)
  }
  
  if(dim(res_mat)[1] != dim(DT_skill)[1]){ stop("dim(res_mat) != dim(res_mat) in RealPreprocess.R") }
  
  # combine skill with res_mat - combined object takes up too much mem 
  # res_mat <- cbind(res_mat, DT_skill)
  # rm(DT_skill);gc()
  
  # future.apply -> memory unsafe 
  if(FALSE){ 
  #res_mat_agg <- tapply(X = 1:nrow(res_mat), simplify = T, 
  res_mat_agg <- future.apply::future_tapply(X = 1:nrow(res_mat), 
                        INDEX = res_mat$master_conditioning_id, 
                        FUN = function(zer){ 
    if( any(zer %% 1000 == 0) ) { 
      write.csv(c( max(zer), nrow(res_mat)), file = "~/Downloads/i_.csv")
      #print2(sprintf("At %s of %s [%.2f%%]", max(zer), nrow(res_mat), 100*max(zer)/nrow(res_mat))) 
    } 
    res_mat_zer <- res_mat[zer,]
    DT_skill_zer <- DT_skill[zer,]
    skill_metrics <- vapply(seq_len(nMax_FutureTruth), function(horizon) {
      ndm_paired_squared_error_skill(
        squared_prediction_error = f2n(
          DT_skill_zer[[paste0("RSSPredTime", horizon)]]
        ),
        squared_baseline_error = f2n(
          DT_skill_zer[[paste0("RSSBaselineTime", horizon)]]
        )
      )
    }, numeric(3L))
    skill_vec <- skill_metrics["skill",]
    RSS_pred <- skill_metrics["rss_pred",]
    RSS_baseline <- skill_metrics["rss_baseline",]
    
    # rename 
    names(skill_vec) <- paste("SkillTime",1:length(skill_vec), sep = "")
    names(RSS_pred) <- paste("RSSPredTime",1:length(skill_vec), sep = "")
    names(RSS_baseline) <- paste("RSSBaselineTime",1:length(skill_vec), sep = "")
    
    # investigate PredBase_l -- what is it? 
    meta_data <- res_mat_zer[1,which(apply(res_mat_zer,2,function(z){length(unique(z))})==1)]
    meta_data <- meta_data[,!grepl(colnames(meta_data),pattern="^PreviousSequenceTruth_l")]
    meta_data <- meta_data[,!grepl(colnames(meta_data),pattern="^SliceSkillTime")]
    meta_data <- meta_data[,!grepl(colnames(meta_data),pattern="^PredBase_l")]
    meta_data <- meta_data[,!grepl(colnames(meta_data),pattern="^Pred_l")]
    meta_data <- meta_data[,!grepl(colnames(meta_data),pattern="^RSSPredTime")]
    meta_data <- meta_data[,!grepl(colnames(meta_data),pattern="^RSSBaselineTime")]
    meta_data <- meta_data[,!grepl(colnames(meta_data),pattern="^Truth_l")]
    meta_data <- meta_data[,!grepl(colnames(meta_data),pattern="^location_id")]
    meta_data <- meta_data[,!grepl(colnames(meta_data),pattern="^time_anchor_id")]
    
    # return package 
    ret_package <- t(unlist(c(unlist(meta_data), RSS_baseline, RSS_pred, skill_vec)))
    return( list(list(ret_package)) )
  }); print2("Done generating res_mat_agg!")
  
  res_mat_dims <- unlist(lapply(res_mat_agg,function(l_){length(unlist(l_))}))
  DominantDim <- names(table(res_mat_dims)[which.max(table(res_mat_dims))])
  # table(res_mat_dims)
  if(length(unique(res_mat_dims)) != 1){warning("In RealPreprocess.R: More than 1 dim here!")}
  # if(length(unique(res_mat_dims)) != 1){stop("In RealPreprocess.R: More than 1 dim here!")}
  res_mat_agg <- res_mat_agg[res_mat_dims==DominantDim] # drop files with off dims
  # colnames((res_mat_agg[[ g_<-which(res_mat_dims == DominantDim)[1]] ][[1]]))
  # colnames((res_mat_agg[[ b_<-which(res_mat_dims != DominantDim)[1] ]][[1]]))
  # setdiff(colnames((res_mat_agg[[g_]][[1]])), colnames((res_mat_agg[[b_]][[1]]))) 
  # which(res_mat_dims != 56 )
  # incongruent colnames: 
  # table(res_mat_dims)
  # setdiff(colnames(res_mat_agg[res_mat_dims!=DominantDim][[1]][[1]]), colnames(res_mat_agg[res_mat_dims==DominantDim][[1]][[1]]))
  res_mat_agg <- as.data.frame(do.call(rbind, unlist(res_mat_agg, recursive=F )))
  #res_mat_agg <- as.data.frame(do.call(plyr::rbind.fill, res_mat_agg ))
  }
  # vectorized -> memory safe 
  if(TRUE){
    ## --- 1.  Convert to data.table and add the ID ---------------------------
    DT [ , master_conditioning_id := paste0(model_id, "_", i_in_sgd) ]
    DT_skill[ , master_conditioning_id := DT$master_conditioning_id ]  # align rows
    
    ## --- 2.  Identify the columns we need -----------------------------------
    rss_base_cols <- grep("^RSSBaselineTime", names(DT_skill), value = TRUE)
    rss_pred_cols <- grep("^RSSPredTime",     names(DT_skill), value = TRUE)
    
    ## columns that are constant within a group
    meta_cols <- setdiff(
      names(DT),
      c(grep("^PreviousSequenceTruth_l", names(DT), value = TRUE),
        grep("^SliceSkillTime",          names(DT), value = TRUE),
        grep("^Pred(Base|_l)",           names(DT), value = TRUE),
        grep("^RSS(Pred|Baseline)Time",  names(DT), value = TRUE),
        grep("^Truth_l",                 names(DT), value = TRUE),
        c("location_id", "time_anchor_id"))
    )
    
    ## --- 3.  Compute paired group-level mean squared errors -----------------
    RSS_agg <- DT_skill[ , {
      paired_metrics <- vapply(seq_along(rss_base_cols), function(horizon) {
        ndm_paired_squared_error_skill(
          squared_prediction_error = f2n(.SD[[rss_pred_cols[[horizon]]]]),
          squared_baseline_error = f2n(.SD[[rss_base_cols[[horizon]]]])
        )
      }, numeric(3L))
      aggregate_values <- c(
        paired_metrics["rss_baseline",],
        paired_metrics["rss_pred",]
      )
      stats::setNames(
        as.list(aggregate_values),
        c(rss_base_cols, rss_pred_cols)
      )
    },
    by = master_conditioning_id,
    .SDcols = c(rss_base_cols, rss_pred_cols) ]
    
    ## --- 4.  Derive the SkillTime* columns ----------------------------------
    H <- length(grep("^Truth_l", names(DT)))          # number of forecast times
    RSS_base_mat <- as.matrix(RSS_agg[ , ..rss_base_cols ])
    RSS_pred_mat <- as.matrix(RSS_agg[ , ..rss_pred_cols ])
    
    skill_mat <- 1 - (0.001 + sqrt(RSS_pred_mat)) /
                (0.001 + sqrt(RSS_base_mat))
    
    set(
      RSS_agg,
      j = paste0("SkillTime", 1:H),
      value = as.data.table(skill_mat)
    )
    
    ## --- 5.  Attach the invariant meta‑data (first row of each group) -------
    Meta_agg <- DT[ , .SD[1], by = master_conditioning_id, .SDcols = meta_cols ]
    
    ## --- 6.  Final aggregate table ------------------------------------------
    Meta_agg$master_conditioning_id_META <- Meta_agg$master_conditioning_id
    Meta_agg[, master_conditioning_id := NULL]
    res_mat_agg <- merge(x = Meta_agg, by.x = "master_conditioning_id_META",   
                         y = RSS_agg, by.y = "master_conditioning_id")
    print2("Done generating res_mat_agg!")
  }
  
  # convert to data.frame and calculate agg skill
  res_mat_agg <- as.data.frame(res_mat_agg)
  res_mat_agg$AggSkill_internal <- rowMeans(cbind(f2n(res_mat_agg$SkillTime8),
                f2n(res_mat_agg$SkillTime9),
                f2n(res_mat_agg$SkillTime10),
                f2n(res_mat_agg$SkillTime11),
                f2n(res_mat_agg$SkillTime12)), na.rm = T)
  
  res_mat_agg_maxIters <- res_mat_agg[res_mat_agg$i_in_sgd==(f2n(res_mat_agg$nSGD)) ,]
  
  if(nrow(res_mat_agg_maxIters) > 0 & FALSE){ 
    # sanity checks - exploring data qualitatively 
    tapply(f2n(res_mat_agg_maxIters$AggSkill_internal),res_mat_agg_maxIters$model_tex_loc,summary)
    tapply(f2n(res_mat_agg_maxIters$AggSkill_internal),res_mat_agg_maxIters$paddingMethod,summary)
    (tmp_<-tapply(f2n(res_mat_agg_maxIters$AggSkill_internal),res_mat_agg_maxIters$i_in_sgd, summary2))[order(f2n(names(tmp_)))]
    # plot(f2n(names(do.call(rbind,tmp_)[,'Mean'])), do.call(rbind,tmp_)[,'Mean'])
    (tmp_ <- tapply(f2n(res_mat_agg_maxIters$AggSkill_internal),res_mat_agg_maxIters$atEpoch, summary2))[ order(f2n(names(tmp_))) ] 
    (tmp_ <- tapply(f2n(res_mat_agg_maxIters$AggSkill_internal),res_mat_agg_maxIters$dataInputs, summary2))[
                    order(unlist(lapply(tmp_,function(l_){l_[4]})),na.last = F) ]
    (tmp_ <- tapply(f2n(res_mat_agg_maxIters$AggSkill_internal),
                    paste(res_mat_agg_maxIters$dataInputs,
                          res_mat_agg_maxIters$i_in_sgd,res_mat_agg_maxIters$evaluationTime,sep="_"), 
                    summary2))[ order(unlist(lapply(tmp_,function(l_){l_[4]}))) ]
    (tmp_ <- tapply(f2n(res_mat_agg_maxIters$AggSkill_internal),res_mat_agg_maxIters$ModelDepth, summary2))[ order(f2n(names(tmp_)),na.last = F) ]
    tapply(f2n(res_mat_agg_maxIters$AggSkill_internal),res_mat_agg_maxIters$evaluationTime, summary2)
    (tmp_<-tapply(f2n(res_mat_agg_maxIters$AggSkill_internal),res_mat_agg_maxIters$ContextLength, summary2))[order(f2n(names(tmp_)))]
    # View(sort(tapply(f2n(res_mat_agg$SkillTime10),res_mat_agg$dataInputs, mean)))
    
    # looking at some predictions 
    res_mat_MaxIterOnly <- res_mat_orig[res_mat_orig$i_in_sgd==(res_mat_orig$nSGD),]
    #res_mat_MaxIterOnly <- res_mat_orig[res_mat_orig$i_in_sgd==min(res_mat_orig$i_in_sgd),]
    par(mfrow=c(2,2));for(irr in sample(1:nrow(res_mat_MaxIterOnly),2*2)){
      plot(unlist(res_mat_MaxIterOnly[irr,paste0("Truth_l",1:nMax_FutureTruth)]),ylim = c(0,4), 
           main = paste0("MD",res_mat_MaxIterOnly$ModelDepth[irr],"_",
                         "CL",res_mat_MaxIterOnly$ContextLength[irr],"_",
                         res_mat_MaxIterOnly$i_in_sgd[irr]))
      points(unlist(res_mat_MaxIterOnly[irr,paste0("Pred_l",1:nMax_FutureTruth)]),pch="^")
    }
    par(mfrow=c(1,1));
    
    hist(res_mat_agg_maxIters$AggSkill_internal[res_mat_agg_maxIters$AggSkill_internal>0])
    # View(res_mat_agg_maxIters[res_mat_agg_maxIters$AggSkill_internal>0,])
    
    tapply(res_mat_agg[res_mat_agg$i_in_sgd==(f2n(res_mat_agg$nSGD)),]$SkillTime11,
           res_mat_agg[res_mat_agg$i_in_sgd==(f2n(res_mat_agg$nSGD)),]$floatType,summary)
    tapply(f2n(res_mat_agg[res_mat_agg$i_in_sgd==(f2n(res_mat_agg$nSGD)),]$te_grads),
           res_mat_agg[res_mat_agg$i_in_sgd==(f2n(res_mat_agg$nSGD)),]$floatType,summary)
  }
  
  # coercing to numeric
  res_mat_agg[,paste0("SkillTime",1:nMax_FutureTruth)] <- 
    apply(res_mat_agg[,paste0("SkillTime",1:nMax_FutureTruth)],  2, f2n)
  res_mat_agg[,paste0("RSSBaselineTime",1:nMax_FutureTruth)] <- 
    apply(res_mat_agg[,paste0("RSSBaselineTime",1:nMax_FutureTruth)],  2, f2n)
  res_mat_agg[,paste0("RSSPredTime",1:nMax_FutureTruth)] <- 
    apply(res_mat_agg[,paste0("RSSPredTime",1:nMax_FutureTruth)],  2, f2n)
  
  tapply( res_mat_agg[res_mat_agg$i_in_sgd==max(res_mat_agg$i_in_sgd),]$SkillTime11,
          res_mat_agg[res_mat_agg$i_in_sgd==max(res_mat_agg$i_in_sgd),]$evaluationTime, summary)
  hist(res_mat_agg$SkillTime11,xlim = c(-10,1))
  
  tapply( res_mat_agg[res_mat_agg$i_in_sgd==max(res_mat_agg$i_in_sgd),]$SkillTime11,
          res_mat_agg[res_mat_agg$i_in_sgd==max(res_mat_agg$i_in_sgd),]$dataInputs, summary)
  tapply( res_mat_agg[res_mat_agg$i_in_sgd==max(res_mat_agg$i_in_sgd),]$SkillTime11,
          grepl(res_mat_agg[res_mat_agg$i_in_sgd==max(res_mat_agg$i_in_sgd),]$dataInputs,
                pattern= "sqrt"), summary)
  
  print2("Done with RealPreprocessing.R")
}
