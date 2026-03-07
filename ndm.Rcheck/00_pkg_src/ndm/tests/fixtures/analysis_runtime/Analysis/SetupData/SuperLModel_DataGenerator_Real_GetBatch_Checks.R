# **** BEGIN SuperLModel_DataGenerator_Real_GetBatch_Checks.R script ****
{
  print2("Starting a test in SuperLModel_DataGenerator_Real_GetBatch_Checks.R...")
  # Test function to verify data generation correctness in GetBatch
  test_data_generation <- function(batch_l, training = TRUE, INPUT_REF_DAT, truth_df_red, 
                                   dataInputs_colnames, context_variables, maxTimesPast, 
                                   nTimesLookahead, paddingMethod, nOutcomes) {
    # Purpose: Validate time slices, IDs, and values/masks for XPred, YTrue, and YTrue_out
    
    # Convert jax arrays to R arrays using np$array (assuming reticulate is set up)
    XPred_r <- np$array(batch_l$XPred)          # Shape: [nBatch, maxTimesPast, nCovars]
    XPred_mask_r <- np$array(batch_l$XPred_mask) # Shape: [nBatch, maxTimesPast, 1]
    YTrue_r <- np$array(batch_l$YTrue)          # Shape: [nBatch, maxTimesPast + nTimesLookahead, nOutcomes]
    YTrue_mask_r <- np$array(batch_l$YTrue_mask) # Shape: [nBatch, maxTimesPast + nTimesLookahead, nOutcomes]
    YTrue_out_r <- np$array(batch_l$YTrue_out)  # Shape: [nBatch, nTimesLookahead, nOutcomes]
    YTrue_out_mask_r <- np$array(batch_l$YTrue_out_mask) # Shape: [nBatch, nTimesLookahead, nOutcomes]
    location_id_numeric <- np$array(batch_l$location_id_numeric) # Shape: [nBatch]
    time_id_numeric <- np$array(batch_l$time_id_numeric)        # Shape: [nBatch]
    
    # Loop over each batch
    for (b in 1:dim(XPred_r)[1]) {
      loc_id_num <- location_id_numeric[b]      # 0-based index
      anchoring_time <- time_id_numeric[b]      # Anchoring time ID
      
      # Map location_id_numeric to location_id (assuming 0-based indexing)
      #location_id <- unique(truth_df_red$location_id)[loc_id_num + 1]
      
      # Subset data for this location
      input_place <- INPUT_REF_DAT[INPUT_REF_DAT$location_id_numeric == loc_id_num, ]
      t_df_red_loc <- truth_df_red[truth_df_red$location_id_numeric == loc_id_num, ]
      
      # Compute TimesRelativeToAnchor
      input_place$TimesRelativeToAnchor <- input_place$time_id - anchoring_time
      t_df_red_loc$TimesRelativeToAnchor <- t_df_red_loc$targetTime_id - anchoring_time
      
      ### Test 1: XPred (Past Covariates)
      # Expected times: anchoring_time - maxTimesPast + 1 to anchoring_time
      input_place_past <- input_place[input_place$TimesRelativeToAnchor <= 0 & 
                                        input_place$TimesRelativeToAnchor > -maxTimesPast, ]
      time_placement_indices_x <- input_place_past$TimesRelativeToAnchor + maxTimesPast
      if (paddingMethod == "right") {
        xNAAdj <- min(time_placement_indices_x) - 1
        time_placement_indices_x <- time_placement_indices_x - xNAAdj
      } else { 
        xNAAdj <- 0 
      }
      
      # Build expected XPred
      expected_XPred <- matrix(NA,
                               nrow = maxTimesPast, 
                               ncol = length(dataInputs_colnames))
      for (i in 1:nrow(input_place_past)) {
        pos <- time_placement_indices_x[i]
        expected_XPred[pos, ] <- as.numeric(input_place_past[i, dataInputs_colnames])
      }
      
      # Validate XPred values and masks
      for( t in 1:maxTimesPast ){
        if( XPred_mask_r[b, t, 1] == 1 ){
          if( any( abs(XPred_r[b, t, 1:ncol(expected_XPred)] -
                       expected_XPred[t, ]) > 1e-6, na.rm = T)  ){
            browser()
            if(FALSE){ 
             XPred_r[b, t, 1:ncol(expected_XPred)]
             XPred_r[, t, 1:ncol(expected_XPred)]
             XPred_r[b,, 1:ncol(expected_XPred)]
             plot(c(XPred_r[b,, 1:ncol(expected_XPred)]),c(expected_XPred));abline(a=0,b=1)
             expected_XPred[t, ] -  XPred_r[b, t, 1:ncol(expected_XPred)]
             expected_XPred[t, ] - as.numeric(unlist(input_place_past[, dataInputs_colnames]))
             Pred_r[b,,1] - as.numeric(unlist(input_place_past[, dataInputs_colnames]))
             expected_XPred[t, ] -  XPred_r[, t, 1:ncol(expected_XPred)]
             expected_XPred[t, ] -  XPred_r[b, , 1:ncol(expected_XPred)]
             min(abs(expected_XPred[t, ] -  XPred_r[, , 1]))
             min(abs(c(expected_XPred) -  XPred_r[, , 1]))
             min(abs(expected_XPred[1, ] -  XPred_r[, , 1]))
             min(abs(expected_XPred[2, ] -  XPred_r[, , 1]))
             min(abs(expected_XPred[3, ] -  XPred_r[, , 1]))
             round(abs(expected_XPred[3, ] -  XPred_r[, , 1]), 4)
             min(abs(expected_XPred[3, ] -  XPred_r[4, , 1]))
             min(abs(expected_XPred[3, ] -  XPred_r[4, , 2]))
            }
            stop( sprintf("Mismatch in XPred at batch %d, time %d", b, t) )
          }
        } 
      }
      
      ### Test 2: YTrue (Truth Data)
      # Define target time range
      min_target <- anchoring_time - maxTimesPast + 1
      max_target <- if (training) {
        min(max(INPUT_REF_DAT$time_id), anchoring_time + nTimesLookahead)
      } else {
        anchoring_time + nTimesLookahead
      }
      t_df_red_loc_filtered <- t_df_red_loc[t_df_red_loc$targetTime_id >= min_target & 
                                              t_df_red_loc$targetTime_id <= max_target, ]
      
      # Calculate time placement indices
      time_placement_indices_y <- t_df_red_loc_filtered$TimesRelativeToAnchor + maxTimesPast
      if (paddingMethod == "right") {
        time_placement_indices_y <- time_placement_indices_y - xNAAdj
      }
      
      # Build expected YTrue
      expected_YTrue <- matrix(NA, nrow = maxTimesPast + nTimesLookahead, ncol = nOutcomes)
      for (i in 1:nrow(t_df_red_loc_filtered)) {
        pos <- time_placement_indices_y[i]
        if (pos >= 1 && pos <= maxTimesPast + nTimesLookahead) {
          if (nOutcomes == 1) {
            expected_YTrue[pos, 1] <- t_df_red_loc_filtered[[true_value_names[1]]][i]
          } else if (nOutcomes == 2) {
            expected_YTrue[pos, 2] <- t_df_red_loc_filtered[[true_value_names[2]]][i]
          } else if (nOutcomes == 3) {
            expected_YTrue[pos, 3] <- t_df_red_loc_filtered[[true_value_names[3]]][i]
          }
        }
      }
      
      # Validate YTrue values and masks
      for (t in 1:(maxTimesPast + nTimesLookahead)) {
        for (o in 1:nOutcomes) {
          if (YTrue_mask_r[b, t, o] == 1) {
            if (abs(YTrue_r[b, t, o] - expected_YTrue[t, o]) > 1e-6) {
              #plot(c(YTrue_r[b,,]), c(expected_YTrue))
              stop(sprintf("Mismatch in YTrue at batch %d, time %d, outcome %d\n", b, t, o))
            }
          } else {
            if (!is.na(expected_YTrue[t, o])) {
              cat(sprintf("Expected NA in YTrue at batch %d, time %d, outcome %d\n", b, t, o))
            }
          }
        }
      }
      
      ### Test 3: YTrue_out (Future Truth Data)
      expected_YTrue_out <- as.matrix(expected_YTrue[(maxTimesPast + 1):(maxTimesPast + nTimesLookahead), ])
      for (t in 1:nTimesLookahead) {
        for (o in 1:nOutcomes) {
          if (YTrue_out_mask_r[b, t, o] == 1) {
            if (abs(YTrue_out_r[b, t, o] - expected_YTrue_out[t, o]) > 1e-6) {
              stop(sprintf("Mismatch in YTrue_out at batch %d, time %d, outcome %d\n", b, t, o))
            }
          } else {
            if (!is.na(expected_YTrue_out[t, o])) {
              stop(sprintf("Expected NA in YTrue_out at batch %d, time %d, outcome %d\n", b, t, o))
            }
          }
        }
      }
      
      ### Test 4: Location and Time IDs
      if (loc_id_num != t_df_red_loc$location_id_numeric[1]) {
        stop(sprintf("Mismatch in location_id_numeric at batch %d\n", b))
      }
      if (anchoring_time != time_id_numeric[b]) {
        stop(sprintf("Mismatch in time_id_numeric at batch %d\n", b))
      }
    }
    cat("Data generation test passed.\n")
  }
  
  # Example usage of the test function
  
  # Prerequisites (ensure these are defined in your environment)
  # library(reticulate)  # For np$array
  # source("./Analysis/SetupData/SuperLModel_DataGenerator_Real.R")  # Load GetBatch
  
  # iterate over test runs
  for(i in 1:20){ 
    # Test in training mode
    print(sprintf("Test %s of %s",i,20))
    batch_train <- GetBatch(nBatch = 20, 
                            training = TRUE, 
                            INPUT_REF_DAT = input_df_red_in)
    test_data_generation(
      batch_l = batch_train,
      training = TRUE,
      INPUT_REF_DAT = input_df_red_in,
      truth_df_red = truth_df_red,
      dataInputs_colnames = dataInputs_colnames,
      context_variables = context_variables,
      maxTimesPast = maxTimesPast,     
      nTimesLookahead = nTimesLookahead,
      paddingMethod = paddingMethod,
      nOutcomes = nOutcomes
    )
    
    # Test in inference mode
    r_ <- FALSE; while(!r_){ 
    loc_id <- sample(unique(input_df_red_out$location_id), 1)
    anchoring_time <- as.integer(sample(as.character(times_out[times_out > minAnchoringTimeID]), 1))
    
    #input_df_red_out$time_id[ input_df_red_out$location_id %in% loc_id ]
    if( min(input_df_red_full$time_id[ input_df_red_full$location_id %in% loc_id ] - anchoring_time)  < -2 ){
      # only go on where enough training data prior to anchor (>2)
      r_ <- TRUE 
    }
    } 
  
    # if( input_df_red_out$location_id %in% loc_id )
    # loc_id <- "35494"; anchoring_time <- 3
    batch_inf <- GetBatch(
      training = FALSE,
      finalGenLocID = loc_id,
      finalGenTimeID = anchoring_time,
      INPUT_REF_DAT = input_df_red_full,
      nTimesLook = nTimesLookValidationInference
    )
    test_data_generation(
      batch_l = batch_inf,
      training = FALSE,
      INPUT_REF_DAT = input_df_red_full,
      truth_df_red = truth_df_red,
      dataInputs_colnames = dataInputs_colnames,
      context_variables = context_variables,
      maxTimesPast = maxTimesPast,   
      nTimesLookahead = nTimesLookValidationInference,
      paddingMethod = paddingMethod,
      nOutcomes = nOutcomes
    )
  }

  # clear holds created during test 
  rm(sampled_places,sampled_times)
  
  print2("Done with a test in SuperLModel_DataGenerator_Real_GetBatch_Checks.R...")
}


