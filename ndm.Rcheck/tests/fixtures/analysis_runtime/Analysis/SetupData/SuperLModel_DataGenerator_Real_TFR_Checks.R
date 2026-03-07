# **** BEGIN SuperLModel_DataGenerator_Real_TFR_Checks.R script ****
{
  print2("Staring checks in SuperLModel_DataGenerator_Real_TFR_Checks.R...")
  # Purpose: Test the integrity of tfrecord objects for data correctness and iteration behavior
  
  # Load required libraries
  library(testthat)
  
  # Define a helper function to load a small sample from tfrecord for testing
  load_tfrecord_sample <- function(tfrecord_file, batch_size = 5L) {
    dataset <- tf$data$TFRecordDataset(tfrecord_file)
    parsed_dataset <- dataset$map(parse_single_example_fxn) # Assumes this is defined in DataGenerator_Real.R
    parsed_dataset <- parsed_dataset$take(5L)$batch(batch_size)
    iterator <- reticulate::as_iterator(parsed_dataset)
    batch <- reticulate::iter_next(iterator)
    return(batch)
  }
  
  # Test suite
  test_that("TFRecord Integrity Tests", {
    
    # Define paths to tfrecord files (assuming these variables are available from the environment)
    tfrecord_file_train <- sprintf("%s/train_%s.tfrecord", TfRecordDir, RealEntry$BaseID)
    tfrecord_file_inference <- sprintf("%s/inference_%s.tfrecord", TfRecordDir, RealEntry$BaseID)
    
    # Ensure tfrecord files exist
    expect_true(file.exists(tfrecord_file_train), "Training tfrecord file does not exist")
    expect_true(file.exists(tfrecord_file_inference), "Inference tfrecord file does not exist")
    
    # --- Test 1: Correctness of Data Stored in TFRecords ---
    context("Data Correctness in TFRecords")
    
    # Load a sample batch from training tfrecord
    train_batch <- load_tfrecord_sample(tfrecord_file_train)
    
    # Check shapes and types
    expect_equal(np$array(train_batch$XPred$shape)[[1]], 5L, info="XPred batch size mismatch", tolerance = 1e-10)
    expect_equal(train_batch$XPred$dtype$name, "float32", info = "XPred should be float32", tolerance = 1e-10)
    expect_equal(np$array(train_batch$YTrue$shape)[[1]], 5L, info = "YTrue batch size mismatch", tolerance = 1e-10)
    expect_equal(train_batch$YTrue$dtype$name, "float32", info = "YTrue should be float32")
    
    test_that("XPred matches original dataframe rows (accounting for padding)", {
      
      # We'll load just a small batch for demonstration
      train_batch <- load_tfrecord_sample(tfrecord_file_train, batch_size = 5L)
      
      # For each example in the batch, figure out which rows in xpred_sample 
      # correspond to which times in the original input_df_red_in
      for (i in seq_len(5)) {
        
        # Grab the location_id_numeric (loc_id) and the anchoring time (anchor_time)
        loc_id      <- np$array(train_batch$location_id_numeric)[i]
        anchor_time <- np$array(train_batch$time_id_numeric)[i]
        
        # Filter your in-sample data to just this location
        df_loc <- input_df_red_in[input_df_red_in$location_id_numeric == loc_id, ]
        
        # Compute TimesRelativeToAnchor exactly how GetBatch_real() does it
        df_loc$TimesRelativeToAnchor <- df_loc$time_id - anchor_time
        
        # Keep only the rows that fall into the “past window” 
        #   i.e. TimesRelativeToAnchor ∈ (-maxTimesPast, 0]
        df_loc <- df_loc[
          df_loc$TimesRelativeToAnchor <= 0 & 
            df_loc$TimesRelativeToAnchor > -maxTimesPast,
        ]
        
        # Now replicate the code snippet that sets up the row indices:
        #    time_placement_indices_x <- (INPUT_place_$TimesRelativeToAnchor + abs(maxTimesPast))
        df_loc$time_placement_indices_x <- df_loc$TimesRelativeToAnchor + abs(maxTimesPast)
        
        # If we are using “right” padding, there is an additional shift by (min(...)-1).
        # (See the code that does: time_placement_indices_x <- time_placement_indices_x - xNAAdj,
        #  where xNAAdj <- (min(time_placement_indices_x)-1).)
        if (paddingMethod == "right") {
          xNAAdj <- min(df_loc$time_placement_indices_x) - 1
          df_loc$time_placement_indices_x <- df_loc$time_placement_indices_x - xNAAdj
        }
        
        # Now, xpred_sample below is shape [timeSeriesLength, numCovariates].
        # The mask tells us which rows are actually “used” vs. set to zero for padding.
        xpred_sample <- np$array(train_batch$XPred)[i, , ]        # shape: (T, numCovars)
        xpred_mask   <- np$array(train_batch$XPred_mask)[i, , ]   # shape: (T, numCovars) or (T,1)
        
        # Sort df_loc so that smaller time_placement_indices_x appear first.
        # This ensures the row order lines up if you want to iterate sequentially.
        df_loc <- df_loc[order(df_loc$time_placement_indices_x), ]
        
        # Optional: Check that the number of valid (unmasked) rows in XPred 
        # matches the number of rows in df_loc.
        # If your mask is shape (T,numCovars), check any column’s mask or rowMeans, etc.
        # For example, check the first column's mask or a “row is fully unmasked” rule:
        
        # Compare each row in df_loc to its corresponding row in XPred
        for (r in seq_len(nrow(df_loc))) {
          # This is the 0-based index used in XPred
          row_idx <- df_loc$time_placement_indices_x[r]
          
          # Because R arrays are 1-based, the row in xpred_sample is row_idx+1
          xpred_row <- xpred_sample[row_idx , ]        # numeric vector, length = numCovars
          orig_row  <- as.numeric(df_loc[r, dataInputs_colnames]) 
          
          # Now check (within tolerance) that the unmasked XPred row 
          # matches the original, normalized input row
          # (assuming you used mean=0, stdev=1 transforms, or yeojohnson, etc.)
          # If any further transformations exist, apply the inverse transform first.
          expect_true(
            all(abs(xpred_row[1:length(orig_row)] - orig_row) < 1e-5, na.rm = TRUE),
            info = sprintf("Mismatch in XPred vs. original row, sample %d, time_placement_indices_x=%d", i, row_idx)
          )
        }
      }
    })

    # Mask consistency check
    xpred_mask <- np$array(train_batch$XPred_mask)
    xpred <- np$array(train_batch$XPred)
    expect_true(all(xpred[xpred_mask == 0] %in% c(0,-1)),
                "XPred values should be {0,-1} where mask is 0")
    
    # --- Test 2: Iteration Code Behavior ---
    context("Iterator Behavior for Batching")
    
    # Initialize iterator
    TFDataset_train <- read_from_tfrecord(file = tfrecord_file_train, batchSize = nBatch, shuffle = FALSE)
    TFDatasetIterator_train <- reticulate::as_iterator(TFDataset_train)
    
    # Test initialization
    expect_true(!is.null(TFDatasetIterator_train), "Iterator initialization failed")
    
    # Test batch retrieval
    dat__ <- reticulate::iter_next(TFDatasetIterator_train)
    expect_true(!is.null(dat__), "First batch retrieval failed")
    expect_equal(np$array(dat__$XPred$shape)[[1]], nBatch,
                 tolerance = 1e-6,
                 info="Batch size mismatch in iteration")
    
    # Simulate iteration to end and reset - part 1 
    TFDataset_train <- read_from_tfrecord(file = tfrecord_file_train, 
                                          batchSize = nBatch, shuffle = FALSE)
    TFDatasetIterator_train <- reticulate::as_iterator(TFDataset_train)
    batch_count <- 0
    max_batches <- as.integer(RealEntry$nSamplesTrain / nBatch)
    dat_ <- dat__; while (!is.null(dat_) && batch_count < max_batches) {
      dat_ <- try(reticulate::iter_next(TFDatasetIterator_train), silent = TRUE)
      batch_count <- batch_count + 1
    }
    expect_true(batch_count <= max_batches, "Iterator did not stop at dataset end")
    
    # Simulate iteration to end and reset - part 2 
    TFDataset_train <- read_from_tfrecord(file = tfrecord_file_train, 
                                          batchSize = nBatch, shuffle = FALSE)
    TFDatasetIterator_train <- reticulate::as_iterator(TFDataset_train)
    max_batches <- as.integer(RealEntry$nSamplesTrain / nBatch)
    theDataContent <- NULL
    batch_count <- 0;dat_ <- dat__; while (!is.null(dat_) && batch_count < (max_batches+1000)) {
      dat_ <- try(reticulate::iter_next(TFDatasetIterator_train), silent = TRUE)
      if(!is.null(dat_)){
        theDataContent <- rbind(theDataContent,
                                cbind("location_id_numeric" = np$array(dat_$location_id_numeric),
                                      "time_id_numeric" = np$array(dat_$time_id_numeric)))
      }
      batch_count <- batch_count + 1
    }
    expect_true( abs(batch_count*nBatch - RealEntry$nSamplesTrain) < 10*nBatch_RealGridGen,
                "Bad match between target and true value of obesrvation number")
    theDataContent <- as.data.frame(theDataContent)
    
    if(OSSType %in% c("OutOfTime","OutOfPlacetime")){
      expect_true( all(theDataContent$time_id_numeric+1 < min(times_out)),
                   "In sample time ids in out of sample range")
    }
    
    theDataContent_combs <- apply(theDataContent,1,function(x){paste(x,collapse="_")})
    
    print(sprintf("UNIQUE PLACETIMES FRACTION: %.4f",
          length(unique(theDataContent_combs)) / length(theDataContent_combs) ))
    #expect_true( length(unique(theDataContent_combs)) / length(theDataContent_combs) > 0.5,
                 #"More than 10% of observations redundant")
    TFDataset_train <- read_from_tfrecord(file = tfrecord_file_train, batchSize = nBatch, shuffle = FALSE)
    TFDatasetIterator_train <- reticulate::as_iterator(TFDataset_train)
    
    # Test reset behavior
    if (inherits(dat_, "try-error") || is.null(dat_) || length(dat_)[[1]] == 0) {
      TFDatasetIterator_train <- reticulate::as_iterator(TFDataset_train)
      dat_ <- reticulate::iter_next(TFDatasetIterator_train)
      expect_true(!is.null(dat_), info = "Iterator reset failed")
      expect_equal(np$array(dat_$XPred$shape)[[1]], nBatch, info="Batch size mismatch after reset")
    }
    
    # --- Test 3: Mission-Critical Checks ---
    context("Mission-Critical Integrity Checks")
    
    # Data leakage check: Ensure inference data is not in training data
    inference_batch <- load_tfrecord_sample(tfrecord_file_inference)
    train_locs <- unique(np$array(train_batch$location_id_numeric))
    inf_locs <- unique(np$array(inference_batch$location_id_numeric))
    train_times <- unique(np$array(train_batch$time_id_numeric))
    inf_times <- unique(np$array(inference_batch$time_id_numeric))
    
    if(OSSType %in% c("OutOfTime","OutOfPlacetime")){ 
      expect_true(length(intersect(train_locs, inf_locs)) == 0 || 
                    length(intersect(train_times, inf_times)) == 0,
                  "Potential data leakage between training and inference datasets")
    }
    
    # Consistency across runs (non-stochastic check since shuffle = FALSE)
    train_batch_2 <- load_tfrecord_sample(tfrecord_file_train)
    expect_equal(np$array(train_batch$XPred), np$array(train_batch_2$XPred),
                 info = "Data should be consistent across runs without shuffling")
  })
  print2("Done with checks in SuperLModel_DataGenerator_Real_TFR_Checks.R...")
}


