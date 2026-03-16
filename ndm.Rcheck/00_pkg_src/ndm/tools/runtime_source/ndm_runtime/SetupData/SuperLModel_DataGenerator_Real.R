# **** BEGIN SuperLModel_DataGenerator_Real.R script ****
{
  print2("Setup data infrastructure in DataGenerator_Real")
  {
    context_variables <- NULL
    #context_variables <- c("Pop", "ihme_true_value_inc_hosp_per_capita")
    
    # normalize data inputs 
    input_df_red_tmp <- truth_df_red
    input_df_red_tmp[,dataInputs_colnames] <- apply(as.matrix(input_df_red_tmp[,dataInputs_colnames]),2,f2n)
  
    #
    dataInputs_colnames_TO_NORMALIZE <- dataInputs_colnames
    in_sample_red_tmp_indices <- which(input_df_red_tmp$time_id <= max(times_in))
  
    # data normalization
    #colnames(input_df_red_tmp)[colnames(input_df_red_tmp) %in% dataInputs_colnames_TO_NORMALIZE]
    input_df_red_tmp[,dataInputs_colnames_TO_NORMALIZE] <- 
          sapply(1:length(dataInputs_colnames_TO_NORMALIZE),function(col_to_norm){
      col_vals <- f2n(unlist(truth_df_red[,dataInputs_colnames_TO_NORMALIZE[col_to_norm]]))
      
      # yeojohnson for normality 
      if(initialTransform == "yeoJohnson"){
        print2("---yeojohnson for normality---")
        col_vals <- predict(bestNormalize::yeojohnson(col_vals))
      }
      
      # if demeaning based on time t
      if(initialNormType == "at_t"){
        demeaner <- tapply(col_vals[in_sample_red_tmp_indices], 
                           truth_df_red$time_id[in_sample_red_tmp_indices], 
                           function(x){mean(x,na.rm=T)})
        descaler <- tapply(col_vals[in_sample_red_tmp_indices], 
                           truth_df_red$time_id[in_sample_red_tmp_indices], 
                           function(x){sd(x,na.rm=T)})
        demeaner <- demeaner[!is.na(demeaner)]
        descaler <- descaler[!is.na(descaler)]
        #plot(demeaner); plot(descaler)
        descaler <- as.data.frame(cbind(descaler,"time_id"=f2n(names(descaler))))
        demeaner <- as.data.frame(cbind(demeaner,"time_id"=f2n(names(demeaner))))
        demeaner <- predict(smooth.spline(x=demeaner$time_id,
                                          y=demeaner$demeaner), 
                            x = max(f2n(demeaner$time_id)))$y
        descaler <- predict(smooth.spline(x=descaler$time_id,
                                          y=descaler$descaler), 
                            x = max(f2n(descaler$time_id)))$y
      }
      
      # if demeaning based on entire context uniformly 
      if(initialNormType == "all"){ 
        demeaner <- mean(col_vals[in_sample_red_tmp_indices],na.rm=T)
        descaler <- sd(col_vals[in_sample_red_tmp_indices],na.rm=T)
      }
  
      # mean / var norm
      col_vals_normed <- (col_vals - demeaner) / (0.001+descaler)
      
      return( col_vals_normed )   
    })
  
    # setup train / test
    input_df_red_full <- input_df_red_tmp
    
    print2(sprintf("Splitting data according to OSSType='%s'", OSSType))
    if (OSSType == "OutOfTime") {
      # -- Standard "out of time" logic (default) -------------------------------
      input_df_red_out <- input_df_red_full[input_df_red_full$time_id >= min(times_out), ]
      input_df_red_in  <- input_df_red_full[input_df_red_full$time_id <= max(times_in), ]
      
      # Basic checks on in/out
      if (any(input_df_red_out$time_id <= max(times_in))) stop("In-out mismatch-1")
      if (!all(input_df_red_in$time_id <= max(times_in))) stop("In-out mismatch-2")
      if (!all(input_df_red_out$time_id >= min(times_out))) stop("In-out mismatch-3")
      if (any(input_df_red_in$time_id >= min(times_out))) stop("In-out mismatch-4")
      
      print2("Using default out-of-time train/test split.")
    } 
    if (OSSType == "OutOfPlace") {
      # -- "out_of_place": different locations for train vs test ----------------
      # Use *all* times for both sets.
      
      locs <- unique(input_df_red_full$location_id)
      
      # For example, 80% of locations for training, 20% for test:
      frac_train_locs <- 0.8
      n_train_locs <- floor(length(locs) * frac_train_locs)
      
      pre_samp <- sample(1:1000,1)
      set.seed(BaseID) # for reproducible location splitting
      train_locs <- sample(locs, n_train_locs, replace = FALSE)
      set.seed(pre_samp) # reset seed 
      
      test_locs  <- setdiff(locs, train_locs)
      
      # We do *not* filter by times_in or times_out here
      # because "out_of_place" is about new places, not new times.
      # Optionally, you may want to ensure all times are included:
      
      # Reset times_in and times_out to use all available times
      times_in <- unique(input_df_red_full$time_id)
      times_out <- times_in
      
      # subset to within range governed by times_in
      input_df_red_full  <- input_df_red_full[input_df_red_full$time_id <= max(times_in), ]
      
      # set training / test based on location ids. 
      input_df_red_in  <- input_df_red_full[input_df_red_full$location_id %in% train_locs, ]
      input_df_red_out <- input_df_red_full[input_df_red_full$location_id %in% test_locs, ]
      
      print2("Using out-of-place train/test split (same times, different places).")
    } 
    if( OSSType == "OutOfPlacetime") {
      
      # -- "out_of_placetime": different locations *and* out-of-time ------------
      # 1) We first select a subset of places for training, the rest for test.
      # 2) Among the training places, we only keep times_in (like usual).
      # 3) Among the test places, we only keep times_out.
      
      locs <- unique(input_df_red_full$location_id)
      
      frac_train_locs <- 0.8
      n_train_locs <- floor(length(locs) * frac_train_locs)
      
      pre_samp <- sample(1:1000,1)
      set.seed(BaseID) # for reproducible location splitting
      train_locs <- sample(locs, n_train_locs, replace = FALSE)
      set.seed(pre_samp) # reset seed 
      
      test_locs  <- setdiff(locs, train_locs)
      
      # train = (train_locs) AND (time in times_in range)
      # test  = (test_locs)  AND (time in times_out range)
      in_mask  <- (input_df_red_full$location_id %in% train_locs) & 
        (input_df_red_full$time_id <= max(times_in)) 
      out_mask <- (input_df_red_full$location_id %in% test_locs)  &
        (input_df_red_full$time_id >= min(times_out)) 
      
      input_df_red_in  <- input_df_red_full[in_mask, ]
      input_df_red_out <- input_df_red_full[out_mask, ]
      
      print2("Using out-of-place-time train/test split (different places + future times).")
    } 
    
    if( !OSSType %in% c("OutOfTime","OutOfPlace","OutOfPlacetime")){
      # ------------------------------------------------------------------------
      stop(sprintf("Unknown OSSType='%s'. Must be one of: {OutOfTime, OutOfPlace, OutOfPlacetime}", OSSType))
    }
    rm(input_df_red_tmp)
    
    context_df_red <- truth_df_red[,c("location_id","time_id"
                                      ,"targetTime_id",context_variables)]
    truth_df_red <- truth_df_red[,c("location_id",
                                    "location_id_numeric", # note: 0 indexed 
                                    "location_name",
                                     all_true_value_names,
                                    "time_id", # note: 0 indexed 
                                    "targetTime_id")]
  
    NAHolder_X <- array(NA, dim = c(abs(maxTimesPast),
                                    nInputCovars <- length(dataInputs_colnames) ))
    NAHolder_C <- array(NA, dim = c(abs(maxTimesPast),
                                    nInputContext <- length(context_variables) ))
    NAHolder_Y <- array(NA, dim = c(abs(maxTimesPast)+nTimesLookahead, 
                                    nOutcomes))
    {
      GetBatch_real <- GetBatch_base <- GetBatch <- function(
                                            nBatch=NULL, 
                                            training = T,
                                            INPUT_REF_DAT = NULL, 
                                            finalGenLocID = NULL, 
                                            finalGenTimeID = NULL,
                                            nTimesLook = NULL, 
                                            openBrowser=F){
        if(openBrowser){browser()}
        
        # cast to df to prevent unexpected behavior 
        INPUT_REF_DAT <- as.data.frame( INPUT_REF_DAT )
  
        if(!is.null(finalGenLocID)){
          anchoring_time_vec <- finalGenTimeID
          INPUT_places <- INPUT_REF_DAT[INPUT_REF_DAT$location_id %fin% finalGenLocID,]
          loc_indices_pool <- tapply(1:nrow(INPUT_places), INPUT_places$location_id, c)
          nBatch <- length( loc_indices_pool )
        }
        
        # print2("Grabbing batch via while loop in SuperLModel_DataGenerator_Real.R...")
        batch_dat <- replicate(nBatch,list())
        if("sampled_times" %in% ls(.GlobalEnv)){ 
          #print(dim(cbind(sampled_times,sampled_places)))
          #placetimes_ <- paste(sampled_times,sampled_places,sep="_")
          #length(unique(placetimes_)) / length(placetimes_)
          #if(length(sampled_times) > 3000){browser()} 
        }
        if(!"sampled_times" %in% ls(.GlobalEnv)){ sampled_places <<- sampled_times <<- c() }
        whileLoop_counter <- ii_ <- 0; while(ii_ < nBatch){ 
          if(whileLoop_counter == 10){ warning("Slow batch generation: whileLoop_counter reached 10 in DataGenerator_Real.R") }
          if(whileLoop_counter >= 1000*nBatch){ stop(sprintf("whileLoop_counter exceeded %d in DataGenerator_Real.R", 1000*nBatch)) }
          
          # note: finalGenLocID should be referenced to location_id NOT location_id_numeric
          whileLoop_counter <- whileLoop_counter + 1 
          
          # training mode
          # if(whileLoop_counter > 10000){print("opening browser in bad while loop");browser()}
          if(is.null(finalGenLocID)){
            # max(INPUT_REF_DAT$time_id) # -> max time id defined in MasterReal.R
            INPUT_places <- INPUT_REF_DAT
            sampled_place <- as.character(f2n(sample(as.character(unique(INPUT_REF_DAT$location_id_numeric)),
                                                   1,  # note: IDs are 1 indexed as R objects 
                                                   replace = FALSE )))
            loc_indices <- tapply(1:nrow(INPUT_places),
                                       INPUT_places$location_id_numeric,
                                       c)[sampled_place][[1]]
            sampled_places <<- c(sampled_places, sampled_place)
          }
          
          # inference mode 
          if(!is.null(finalGenLocID)){
            loc_indices <- loc_indices_pool[[whileLoop_counter]]
          }
        
          # loc_indices <- PLACES_POOL_LIST[[1]]
          # anchoring_time  <- 20;perspective_ <- "past"
          # obtain predicted data pool
          INPUT_place <- INPUT_places[loc_indices,]
          
          if( training & (!is.null(finalGenTimeID) |!is.null(finalGenLocID))){ 
            stop("Case not implemented")
          }

          # sample an anchoring time (point at which predictions are made)
          if( training & is.null(finalGenTimeID) ){
            tmp_ <- sort( unique( INPUT_place$time_id ) )
            tmp_ <- tmp_[   tmp_ >= as.integer( minAnchoringTimeID ) & 
                              tmp_ < max(INPUT_place$time_id) ]  
            
            # find new placestimes
            {
              # Compute used times for this location
              if( length(sampled_places) > 1 ){
                prev_sampled_places <- sampled_places[1:(length(sampled_places)-1)]
                used_times_for_loc <- sampled_times[prev_sampled_places == sampled_place]
              }
              else { 
                used_times_for_loc <- c()
              }
              
              # Find available unique anchors
              available_unique_anchors <- setdiff(tmp_, used_times_for_loc)
              
              # Sample anchoring time: prefer unique if available, else allow duplicates
              if(length(available_unique_anchors) > 0){
                anchoring_time_vec <- f2n(sample(as.character(available_unique_anchors), 1))
              }
              if(length(available_unique_anchors) == 0){
                anchoring_time_vec <- sample(tmp_, 1)
              }
            }
            
            # Uniform sampling of anchoring times
            #anchoring_time_vec <- f2n(sample(as.character(tmp_), 1))
            
            # Sampling based on proximity to t
            #ProximityWts <- exp( -abs(tmp_ - times_out+nTimesLookahead) / 20)
            #ProximityWts <- ProximityWts/sum(ProximityWts) #plot(ProximityWts)
            #anchoring_time_vec <- f2n(sample(as.character(tmp_), 1,prob = ProximityWts))
            
            sampled_times <<- c(sampled_times, anchoring_time_vec)
          }
          if( !training & is.null(finalGenTimeID) ){
            stop("CASE NOT IMPLEMENTED XZ42412")
            tmp_ <- sort( unique( INPUT_place$time_id ) )
            
            # minimum context defined here, with anchoring_time_vec - > point at which predictions are made
            tmp_ <- tmp_[   tmp_ >= as.integer( minAnchoringTimeID )    ] 
            anchoring_time_vec <- f2n(sample(as.character(tmp_), 1))
          }
  
          results_by_anchor <- sapply(anchoring_time_vec, function(anchoring_time){
  
            # determine times into the past
            INPUT_place$TimesRelativeToAnchor <- INPUT_place$time_id - anchoring_time
  
            # obtain mask
            ValuesHolder_X <- NAHolder_X
            ValuesHolder_Y <- NAHolder_Y
            ValuesHolder_C <- NAHolder_C
  
            ##################
            # setup prior infection data
            ##################
            perspective_counter <- 0
            #for(perspective_ in perspective_pool){
            for(perspective_ in "past"){
              perspective_counter <- perspective_counter +  1
              if(perspective_ == "past"){
                # select those predictions falling within time lookahead
                dataInputs_colnames_ <- dataInputs_colnames_past
                INPUT_place_ <- INPUT_place[INPUT_place$TimesRelativeToAnchor <= 0 &
                                              INPUT_place$TimesRelativeToAnchor > -maxTimesPast,]
                time_placement_indices_x <- (INPUT_place_$TimesRelativeToAnchor+abs(maxTimesPast))
                if(paddingMethod == "right"){
                  time_placement_indices_x <- time_placement_indices_x - (xNAAdj <- (min(time_placement_indices_x)-1) )
                }
                covar_placement <- 1:length(dataInputs_colnames_past)
              }
              if(perspective_ == "future"){
                # select those predictions falling within time lookahead
                dataInputs_colnames_ <- dataInputs_colnames_future
                INPUT_place_ <- INPUT_place[INPUT_place$TimesRelativeToAnchor > 0 &
                                              INPUT_place$TimesRelativeToAnchor <= -maxTimesPast,]
                time_placement_indices_x <- INPUT_place_$TimesRelativeToAnchor
                covar_placement <- (length(dataInputs_colnames_past)+1):
                  (length(dataInputs_colnames_past)+length(dataInputs_colnames_future))
                stop("*CHECK THIS CASE*")
              }
  
              # insert predicted data into correct location
              {
                # insert data
                {
                  try(ValuesHolder_X[time_placement_indices_x,] <- (as.matrix( INPUT_place_[,dataInputs_colnames] )), T)
  
                  # Create a detailed mask for each covariate at each time step
                  MaskHolder_X_full <- 1 - 2*is.na(ValuesHolder_X)
                  
                  # mask if ALL covariates missing
                  MaskHolder_X <- as.matrix(1*(rowMeans(1*(!is.na(ValuesHolder_X))) > 0))
  
                  #print("ADD MISSINGNESS TOKEN HERE?")
                  ValuesHolder_X[is.na(ValuesHolder_X)] <- 0 #approximate mean imputation considering normalization
                }
              }
            }
  
            # setup truth data
            # determine times into the past
            truth_df_red$TimesRelativeToAnchor <- truth_df_red$time_id - anchoring_time
  
            # setup truth
            t_df_red_loc <- truth_df_red[truth_df_red$location_id == INPUT_place$location_id[1],]
            if( !training ){ # -> take full outcome sequence
              t_df_red_loc <- t_df_red_loc[t_df_red_loc$targetTime_id %fin%
                                             c((anchoring_time-abs(maxTimesPast)+1):(anchoring_time+nTimesLookahead)),]
            }
            if( training ){ # -> take outcome sequence up to max time in training pool
              t_df_red_loc <- t_df_red_loc[t_df_red_loc$targetTime_id %fin%
                                          c((anchoring_time-abs(maxTimesPast)+1):
                                              min(c(max(times_in), (anchoring_time+nTimesLookahead)) )) , ]
            }
            time_placement_indices_y <- (t_df_red_loc$TimesRelativeToAnchor + abs(maxTimesPast))
            if(paddingMethod == "right"){ time_placement_indices_y <- time_placement_indices_y - xNAAdj }

            if(nOutcomes == 1){
              ValuesHolder_Y[time_placement_indices_y,] <- t_df_red_loc[[true_value_names]]
            }
            if(nOutcomes > 1){
              ValuesHolder_Y[time_placement_indices_y,] <- t_df_red_loc[,true_value_names]
            }
            
            MaskHolder_Y <- 1*(!is.na(ValuesHolder_Y))
            ValuesHolder_Y[is.na(ValuesHolder_Y)] <- 0
  
            ##################
            # setup context data
            ##################
            c_df_red_loc <- NULL
            MaskHolder_C <- jnp$array(1.)
            if(length(context_variables)>0){
              context_df_red$TimesRelativeToAnchor <- context_df_red$time_id - anchoring_time
              c_df_red_loc <- context_df_red[context_df_red$location_id == INPUT_place$location_id[1],]
              c_df_red_loc <- c_df_red_loc[c_df_red_loc$targetTime_id %fin% 
                                      c((anchoring_time-abs(maxTimesPast)+1):(anchoring_time+nTimesLookahead)),]
              time_placement_indices_c <- (c_df_red_loc$TimesRelativeToAnchor + abs(maxTimesPast))
              if(paddingMethod == "right"){
                time_placement_indices_c <- time_placement_indices_c - xNAAdj
              }
              try(ValuesHolder_C[time_placement_indices_c,] <- (as.matrix(c_df_red_loc[,context_variables])),T)
              MaskHolder_C <- as.matrix( 1*(rowMeans(1*(!is.na(ValuesHolder_C))) == 1))
              ValuesHolder_C[is.na(ValuesHolder_C)] <- 0
            }
  
            targetDims <- abs(maxTimesPast) + nTimesLookahead
            {
              # plot(ValuesHolder_X[,1]); plot(ValuesHolder_X[,2]); plot(ValuesHolder_X[,3])
              ret_ <- {list(
                           "XPred" = cbind(ValuesHolder_X, 
                                           MaskHolder_X_full), # concatenate with mask to model missingness via attention
                           "XPred_mask" = MaskHolder_X,
  
                           "YTrue" = ValuesHolder_Y,
                           "YTrue_mask" = MaskHolder_Y,
                           
                           "YTrue_out" = as.matrix(ValuesHolder_Y[ 
                                                    (nrow(ValuesHolder_Y)-nTimesLookahead+1):nrow(ValuesHolder_Y),]),
                           "YTrue_out_mask" = as.matrix(MaskHolder_Y[ 
                                                    (nrow(ValuesHolder_Y)-nTimesLookahead+1):nrow(ValuesHolder_Y),]),
  
                           "Context" = ValuesHolder_C,
                           "Context_mask" = MaskHolder_C,
  
                           # location id
                           "location_id_numeric" = as.integer(t_df_red_loc$location_id_numeric[1]),
  
                           # time id (start of data tracing period)
                           "time_id_numeric" =  as.integer( anchoring_time  )
                           #"time_id_numeric" =  as.integer( anchoring_time - maxTimesPast - (minAnchoringTimeID - maxTimesPast) )
                          )}
            }
            return(  ret_  )
          })
  
          # drop bad draws 
          # plot(results_by_anchor["XPred",][[1]][,1])
          # plot(results_by_anchor["XPred_mask",][[1]][,1])
          drop_indicator <- (mean(results_by_anchor["XPred_mask",][[1]]) < MIN_NA_ACCEPT_FRAC )
          if(!is.null(finalGenLocID)){  ii_ <- ii_ + 1  } 
          if(!drop_indicator){
            if(is.null(finalGenLocID)){  ii_ <- ii_ + 1  } 
            batch_dat[[ii_]] <- results_by_anchor
          }
        }
  
        print2("Structuring batch...")
        batch_dat <- do.call(cbind,batch_dat)
        ret_list <- NULL
        if(!is.null(batch_dat)){
          for(name_ in row.names(batch_dat)){
            ndims_ <- length(dim(batch_dat[name_,][[1]]))
            which_name_ <- which(name_ == row.names(batch_dat))
            if(ncol(batch_dat) == 1){ 
              eval(parse(text = sprintf("%s <- jnp$expand_dims((%s)[[1]],0L)",
                                        name_, paste(paste("batch_dat[which_name_,]"),collapse=","))))
            }
            if(ncol(batch_dat) > 1){ 
              eval(parse(text = sprintf("%s <- jnp$%s((%s),0L)",
                                                  name_,
                                                  ifelse(ndims_==1, yes = "concatenate", no = "stack"),
                                                  paste(paste("batch_dat[which_name_,]"),collapse=","))))
            }
          }
          
          # if(!"XPred"%in% ls()){browser()} # for debugging 
          if(!"Context" %in% row.names(batch_dat)){ Context <- XPred; Context_mask <- XPred_mask }
          
          ret_list <- list( "XPred" = XPred$astype( jaxFloatType ),
                            "XPred_mask" = XPred_mask$astype( jnp$bool ), 
                            
                            "Context" = Context$astype( jaxFloatType ),
                            "Context_mask" = Context_mask$astype( jnp$bool ), 
                            
                            "location_id_numeric" = location_id_numeric$astype( jaxFloatType ),
                            "time_id_numeric" = time_id_numeric$astype( jaxFloatType ),
                            
                            "YTrue" = YTrue$astype( jaxFloatType ),
                            "YTrue_mask" = YTrue_mask$astype( jnp$bool ), 
                            
                            "YTrue_out" = YTrue_out$astype( jaxFloatType ),
                            "YTrue_out_mask" = YTrue_out_mask$astype( jnp$bool )
                            )
        }
        return( ret_list )
      }
      # NB: XPred is pre-normalized 
    }
  
    # run tests
    ndm_source_extracted("SetupData/SuperLModel_DataGenerator_Real_GetBatch_Checks.R")
    
    # sample data 
    ok_ctt_ <- 0; ok <- F; while(!ok){
      ok_ctt_ <- ok_ctt_ + 1 
      if(ok_ctt_ == 10000){ stop("ok_ctt_ > 10000 in DataGenerator_Real.R") }
      batch_l <- try(GetBatch(nBatch = nBatch, 
                              INPUT_REF_DAT = input_df_red_in ),T)
      if(class(batch_l) != "try-error"){ ok<-T }
      if(class(batch_l) == "try-error"){ print("Failed GetBatch()"); write.csv(as.character(batch_l), file = "./FailedInDataGenReal.csv")}
      
      # sanity checks 
      if(T == F){ 
      np$array(batch_l$XPred_mask)[,,1]  
      np$array(batch_l$YTrue_out)[,,1]
      np$array(batch_l$YTrue_out_mask)[,,1]
      np$array(batch_l$YTrue)[,,1]    
      np$array(batch_l$YTrue_mask)[,,1]    
      np$array(batch_l$YTrue)[,,1]*(1-np$array(batch_l$YTrue_mask)[,,1])
        
      i_<-2
      # data check
      batch_l$location_id_numeric
      unique(truth_df_red[truth_df_red$location_id_numeric%in%np$array(batch_l$location_id_numeric),]$location_id_numeric)
      check_index <- 11
      unique(truth_df_red[ truth_df_red$location_id_numeric==check_index, ]$location_name)
      check_index_into_batch <- 1:20#which( np$array(batch_l$XPred[[3]]) == check_index )[1]
      np$array(batch_l$location_id_numeric)[check_index_into_batch]
      
      # inputs check 
      check_index <- sample(1:batch_l$XPred$shape[[3]],1)
      for(irre in 1:(batch_l$XPred$shape[[3]])){
        plot((np$array(batch_l$XPred)[check_index,,irre]), main = irre,
             pch = ifelse( (np$array(batch_l$XPred_mask)[check_index,,1])==1,yes="O",no="M"),cex =0.5) 
        Sys.sleep(0.25)
      }
      
      check_index <- sample(1:batch_l$XPred$shape[[3]],1)
      plot((np$array(batch_l$YTrue)[check_index,,1])) 
      # plot((np$array(batch_l$YTrue)[check_index,,2])) 
      
      plot((np$array(batch_l$YTrue_out)[check_index,,1])) 
    
      #tmp <- cor(np$array(batch_l$XPred[[1]])[check_index_into_batch,,])
      #diag(tmp)<-NA; summary( c(tmp) )
      # which( rowSums(apply(np$array(batch_l$XPred[[1]][[1]]),1:2,sd) )  == 0)
  
      #batch_l$XPred[[1]][[1]]
      #apply(batch_l$XPred[[1]]$to_py(),3,mean)
      #apply(batch_l$XPred[[1]]$to_py(),3,sd)
      #sd(batch_l$YTrue$to_py()[,1])
      }
    }
    jax_batchx <-  batch_l$XPred
    jax_batchy <- batch_l$YTrue
    nCovars_real <- jax_batchx$shape[[3]]
  }

  print2("Apply infrastructure to save tfrecords in DataGenerator_Real.R")
  { 
      # warning("Forcing small sample size in tfrecord for testing!"); nSamplesTrain <- sample(c(10,100,1000),1);Sys.sleep(10)
      tf <- reticulate::import("tensorflow")
      try(tf$config$experimental$set_memory_growth(T),T)
      RESHUFFLE_EACH_ITERATION <- F
      
      # release holds on times and places from tests
      rm(sampled_places,sampled_times)
      
      # helper fxn 
      bytes_feature <- function(value){
        tf$train$Feature(bytes_list=tf$train$BytesList(value=list(value$numpy())))
      }
      SerializeFromListEntry <- function(l_){
        feature = dict(
          'data' = bytes_feature(lapply(l_, function(l__) {tf$io$serialize_tensor(l__) })),
          'shape' = bytes_feature(tf$io$serialize_tensor(l_$shape))
        )
        example_proto <- tf$train$Example(features = tf$train$Features(feature = feature))
        return( example_proto$SerializeToString() )
      }
      parse_single_example_fxn_OLD <- function(example_proto){
        # Define the features to be extracted.
        parseText <- paste(sapply(names(batch_l),function(nl){
          sprintf("
          '%s' = tf$io$FixedLenFeature(list(), tf$string),
          '%s_shape' = tf$io$FixedLenFeature(list(), tf$string)",nl, nl) }),collapse = ",")
        eval(parse(text = sprintf("feature_description <- list( %s )", parseText )))
        
        # Parse the input `tf.train.Example` proto using the dictionary above.
        parseText <- paste(sapply(names(batch_l),function(nl){
          sprintf("
          '%s' = tf$cast(tf$reshape(tf$io$parse_tensor(example[['%s']], out_type = tf$float32), 
                                    tf$io$parse_tensor(example[['%s_shape']], out_type = tf$int32)), 
                                    dtype = %s)
          ",nl, nl, nl, DefaultDtypeTf) }),collapse = ",")
        example <- tf$io$parse_single_example(example_proto, feature_description)
        eval(parse(text = sprintf("data <- list(%s)",parseText)))
        return(data)
      }
      parse_single_example_fxn <- function(example_proto){
        # Names to parse: take from the last sampled batch (same fields as TFRecords)
        feat_names <- names(batch_l)
        
        # 1) FixedLenFeature spec: data (bytes) + shape (bytes) for each feature
        fd_text <- paste(
          sapply(feat_names, function(nm) {
            sprintf("'%s' = tf$io$FixedLenFeature(list(), tf$string),
               '%s_shape' = tf$io$FixedLenFeature(list(), tf$string)", nm, nm)
          }),
          collapse = ","
        )
        eval(parse(text = sprintf("feature_description <- list(%s)", fd_text)))
        
        # 2) Infer the **saved** dtype for each feature (match the writer's downcast)
        #    float64/float32 -> tf$float32; int64/int32 -> tf$int32; bool -> tf$bool
        saved_out_types <- sapply(feat_names, function(nm) {
          d <- try(as.character(eval(parse(text = sprintf("batch_l$%s$dtype", nm)))), silent = TRUE)
          if (!inherits(d, "try-error")) {
            if (grepl("bool", d)) "tf$bool"
            else if (grepl("int",  d)) "tf$int32"
            else                       "tf$float32"
          } else {
            # Fallback if batch_l isn't around: masks -> bool; everything else -> float32
            if (grepl("mask", nm)) "tf$bool" else "tf$float32"
          }
        }, USE.NAMES = TRUE)
        
        # 3) Choose cast dtype exposed to the model:
        #    - floats -> DefaultDtypeTf (e.g., tf$float32 / tf$bfloat16)
        #    - ints   -> keep tf$int32
        #    - bools  -> keep tf$bool
        cast_types <- vapply(names(saved_out_types), function(nm) {
          out_t <- saved_out_types[[nm]]
          if (out_t == "tf$float32") DefaultDtypeTf else out_t
        }, "", USE.NAMES = TRUE)
        
        # 4) Parse example and build per-field parse ops with correct dtypes
        example <- tf$io$parse_single_example(example_proto, feature_description)
        
        parse_text <- paste(
          mapply(function(nm, out_t, cast_t) {
            sprintf("
        '%s' = tf$cast(
                 tf$reshape(
                   tf$io$parse_tensor(example[['%s']], out_type = %s),
                   tf$io$parse_tensor(example[['%s_shape']], out_type = tf$int32)
                 ),
                 dtype = %s
               )",
                    nm, nm, out_t, nm, cast_t)
          }, names(saved_out_types), unname(saved_out_types), unname(cast_types)),
          collapse = ","
        )
        
        eval(parse(text = sprintf("data <- list(%s)", parse_text)))
        return(data)
      }
      
      read_from_tfrecord <- function(file, batchSize, 
                                     TfRecords_BufferScaler = 10L, nTake = NULL, shuffle = T){
        raw_dataset = tf$data$TFRecordDataset( file )
        parsed_dataset = raw_dataset$map( parse_single_example_fxn )
        if(!is.null(nTake)){ parsed_dataset <- parsed_dataset$take(nTake) } # take only first nTake elements 
        if(shuffle == T){ 
          parsed_dataset <- parsed_dataset$shuffle(buffer_size = tf$constant(as.integer(TfRecords_BufferScaler*batchSize), 
                                                                             dtype=tf$int64),
                                                   reshuffle_each_iteration = RESHUFFLE_EACH_ITERATION)
        }
        parsed_dataset <- parsed_dataset$batch( batchSize )
        return( parsed_dataset )
      }
      
      nBatch_RealGridGen <- 16L # smaller gives more precise value relative to target 
      if(!dir.exists(TfRecordDir)){ dir.create(TfRecordDir) }
      tfrecord_file_train <- sprintf('%s/%s_%s.tfrecord', TfRecordDir, "train", as.character(RealEntry$BaseID)) 
      tfrecord_file_inference <- sprintf('%s/%s_%s.tfrecord', TfRecordDir, "inference", as.character(RealEntry$BaseID)) 
      nObsInference <- as.integer(get0("nObsInference", inherits = TRUE, ifnotfound = 1024L))
      nTimesLookValidation <- nTimesLookValidationInference
      
      # resave runs 
      # nSamples_max <- 100; 
      if( RealEntry$nSamplesTrain == nSamples_max & ReSaveTfRecords & (RealEntry$ResaveThisTFRecord==1) ){
        for(type_ in c("train","inference")){
          if(type_ == "train"){
            tf_record_writer = tf$io$TFRecordWriter( tfrecord_file_train  ) 
            nTimesLookValidation_ <- nTimesLookahead
            input_df_red_use_ <- input_df_red_in
          }
          if(type_ == "inference"){
            tf_record_writer = tf$io$TFRecordWriter( tfrecord_file_inference  ) 
            nTimesLookValidation_ <- nTimesLookValidationInference
            input_df_red_use_ <- input_df_red_out 
          }
          nOuter <- ifelse(type_ == "train", 
                           yes = c(ceiling(nSamplesTrain/nBatch_RealGridGen)), # go in batches 
                           no = c(nObsInference) # go 1 by 1 
                           )[[1]]
          for(b_ in 1:nOuter){ 
            print2(sprintf("%s of %s [%.2f%%], saving tfrecored [%s]",b_, nOuter, 100*b_/nOuter, type_))
            if(type_ == "train"){ 
              ok_ctr_ <- 0; ok_ <- F; while(ok_ == F){ 
                ok_ctr_ <- ok_ctr_ + 1; if(ok_ctr_ == 10000){ stop("ok_ctr_ > 10000 in DataGenerator_Real.R [at for(b_ in 1:nOuter)]") }
                batch_l <- try(GetBatch_real( nBatch = nBatch_RealGridGen, 
                                              INPUT_REF_DAT = input_df_red_use_, 
                                              nTimesLook = nTimesLookValidation_, 
                                              training = (type_ == "train") ), T)
                # tf$reduce_mean(batch_l$XPred, 0L:1L)
                if("try-error" %in% class(batch_l)){ print("Error in GetBatch_real() [in for(b_ in 1:nOuter)]...") } # go on
                if(!"try-error" %in% class(batch_l)){ ok_ <- T } # go on
                if(b_ %% 2 == 0){ gc(); py_gc$collect()  } 
              }
            }
            if(type_ == "inference"){
              ok_ctr_ <- 0; ok_ <- F; while(ok_ == F){ 
              ok_ctr_ <- ok_ctr_ + 1; if(ok_ctr_ == 10000){ stop("ok_ctr_ > 10000 in DataGenerator_Real.R [at 'inference']") }
              # partial sample
              #input_df_red_out$location_id_numeric
              place_pool <- sample(as.character(unique(input_df_red_out$location_id)), 1)
              sl_dat <- c(); place_counter__ <- 0; for(loc_id in place_pool){
                time_counter__ <- 0;
                # out of sample targets - how to  gen the input data for the base learners?
                for(time_iter in sample(as.character(times_out), 1)){
                  #for(time_iter in time_iter_pool <- times_in){ print2("DOING IN SAMPLE TEST")
                  # check this - why not right dimensions?
                  # max( input_df_red_in$time_id ); times_out
                  # plot(sort(input_df_red_full[input_df_red_full$location_id %fin% loc_id,]$time_id - time_iter))
                  # loc_id <- sample(as.character(unique(input_df_red_out$location_id[input_df_red_out$location_id %in% input_df_red_in$location_id])), 1)
                  # time_iter <- sample(as.character(times_out), 1)
                  batch_l <- try(GetBatch(training = F,
                                          finalGenLocID = loc_id, # place_pool
                                          finalGenTimeID = f2n(time_iter),
                                          INPUT_REF_DAT = input_df_red_full,
                                          nTimesLook = nTimesLookValidation_, 
                                          openBrowser = F), T)
                  # tf$reduce_mean(batch_l$XPred, 0L:1L)
                  if(T == F){ 
                         stop("NORMALIZATION ANALYSIS")
                         # sanity checks 
                         colMeans(input_df_red_full[,dataInputs_colnames_TO_NORMALIZE],na.rm=T)
                         colMeans(input_df_red_in[,dataInputs_colnames_TO_NORMALIZE],na.rm=T)
                         colMeans(input_df_red_out[,dataInputs_colnames_TO_NORMALIZE],na.rm=T)
                         table(input_df_red_out$time_id)
                                          
                         plot(input_df_red_in[,dataInputs_colnames_TO_NORMALIZE[i_<-2]],na.rm=T,ylim = c(-5,5))
                         plot(input_df_red_out[,dataInputs_colnames_TO_NORMALIZE[i_]],na.rm=T,ylim = c(-5,5))
                         input_df_red_tmp
                         plot(tapply(input_df_red_full[,dataInputs_colnames_TO_NORMALIZE[i_]], 
                                input_df_red_full$time_id,function(x){mean(x,na.rm=T)}))
                         abline(v=times_out,h=0)
                         plot(tapply(input_df_red_out[,dataInputs_colnames_TO_NORMALIZE[i_]], 
                                     input_df_red_out$time_id,function(x){mean(x,na.rm=T)}))
                         plot(tapply(input_df_red_in[,dataInputs_colnames_TO_NORMALIZE[i_]], 
                                     input_df_red_in$time_id,function(x){mean(x,na.rm=T)}))
                         input_df_red_use_ 
                  } 
                }
              }
              if("try-error" %in% class(batch_l)){ print("Error in GetBatch_real() [in type_=='inference' arm]...") } # go on
              if(!"try-error" %in% class(batch_l)){ ok_ <- T } # go on
              if(b_ %% 2 == 0){ gc(); py_gc$collect()  }
              }
            }
            
            #print2("DEBUG SUBSETTING OF THE DATA..."); batch_l <- batch_l[1:4]
            #for(ii_ in 0L:(nBatch_RealGridGen-1L)){ 
            for( ii_ in 0L:( batch_l[[1]]$shape[[1]] -1L) ){ 
              # sapply(batch_l, function(l_){l_$dtype})
              
              tfrecordExampleText <- sapply(1:length(batch_l), function(l_){ 
                THE_TYPE_ <- as.character(eval(parse(text = sprintf("batch_l$%s$dtype", names(batch_l)[l_]))))
                THE_SAVETYPE_ <- if(grepl(THE_TYPE_, pattern="float")) {
                  "tf$float32"
                } else if(grepl(THE_TYPE_, pattern="int")) {
                  "tf$int32"
                } else if(grepl(THE_TYPE_, pattern="bool")) {
                  "tf$bool"
                } else {
                  warning(sprintf("Unknown dtype '%s', defaulting to tf$float32", THE_TYPE_))
                  "tf$float32"
                } 
                sprintf(" %s = tf$train$Feature(bytes_list=
                                tf$train$BytesList(value=list( 
                                  tf$io$serialize_tensor(tf$constant( jnp$take(batch_l$%s,
                                       ii_, axis = 0L), %s ))$numpy()))),
                        %s_shape = tf$train$Feature(bytes_list=tf$train$BytesList(value=list( tf$io$serialize_tensor(tf$constant( jnp$take(batch_l$%s, ii_, axis = 0L) )$shape)$numpy())))",
                        names(batch_l)[l_], names(batch_l)[l_], 
                        THE_SAVETYPE_,
                        names(batch_l)[l_],  names(batch_l)[l_] ) }) 
              tfrecordExampleText <- paste(tfrecordExampleText, collapse = ",")
              
              # Create a feature & example 
              eval(parse(text = sprintf("example <- tf$train$Example(features=tf$train$Features(feature=dict( %s ) ))", tfrecordExampleText )))
              
              # Write the Example to the TFRecord file
              tf_record_writer$write( example$SerializeToString() )
            }
            # rm(batch_l)
          }
          rm(input_df_red_use_)
          tf_record_writer$close()
          print2(sprintf("Done writing tfrecords for: %s!",type_))
          
          # run tests    
          #TFDataset <- read_from_tfrecord(file = tfrecord_file, batchSize = nBatch)
          #TFDatasetIterator <- reticulate::as_iterator( TFDataset )
          #TFDataBatch <- reticulate::iter_next( TFDatasetIterator )
        }
        
        # Run checks
        ndm_source_extracted("SetupData/SuperLModel_DataGenerator_Real_TFR_Checks.R")
      }
      TFDatasetIterator_train <- reticulate::as_iterator(
        TFDataset_train <- read_from_tfrecord(file = tfrecord_file_train, 
                                              batchSize = nBatch, 
                                              shuffle = F,
                                              nTake = ai(RealEntry$nSamplesTrain))$shuffle(
                                                                        buffer_size = tf$constant(as.integer(10*nBatch), 
                                                                                                  dtype=tf$int64),
                                                                        reshuffle_each_iteration = T)
      )
      TFDatasetIterator_inference <- reticulate::as_iterator( 
        TFDataset_inference <- read_from_tfrecord(file = tfrecord_file_inference, 
                                                  batchSize = nBatch, 
                                                  shuffle = F) )
  }
  print2("Done with this round of SuperLModel_DataGenerator_Real.R!")
}

# stop("XXX")
