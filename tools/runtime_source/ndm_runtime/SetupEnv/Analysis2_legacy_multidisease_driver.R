#!/usr/bin/env Rscript
# Internal compatibility driver for multidisease real-data runs.
# Public callers should go through `analysis2_run_real_multidisease()` so all
# option parsing and path resolution happens in one place.
{
{
options(error = NULL)

if (!exists("analysis2_multidisease_spec", inherits = TRUE)) {
  stop("`analysis2_multidisease_spec` must be defined before sourcing the multidisease driver.", call. = FALSE)
}

analysis2_multidisease_spec <- get("analysis2_multidisease_spec", inherits = TRUE)
AnalysisName <- analysis2_multidisease_spec$analysis_name
AnalysisDate <- Sys.Date()
DiseaseNameVec <- analysis2_multidisease_spec$disease_names
NDM_INTERNAL_ANALYSIS_DIR <- normalizePath(
  analysis2_multidisease_spec$paths$analysis_root,
  winslash = "/",
  mustWork = TRUE
)
ndm_source_extracted <- function(relative_path, env_target = parent.frame()) {
  sys.source(
    file.path(NDM_INTERNAL_ANALYSIS_DIR, relative_path),
    envir = env_target,
    keep.source = FALSE
  )
  invisible(env_target)
}
COMMAND_ARG_INPUT <- if (length(analysis2_multidisease_spec$outer) == 1L) {
  analysis2_as_int(analysis2_multidisease_spec$outer[[1]])
} else {
  NA_integer_
}
OUTER_ITERATION_SEQUENCE <- as.integer(analysis2_multidisease_spec$outer)
setwd(analysis2_multidisease_spec$project_root)
print(sprintf("wd is: {%s}", getwd()))
set.seed(theInitialSeed <- 12L)

# Maintainer note:
# This driver still uses sourced globals for the actual training loop. Only the
# run surface has been modernized; change model/data/training internals here
# only when the package-backed runners cannot express the needed behavior.

# critical hyperparameters
ReSaveTfRecords <- isTRUE(analysis2_multidisease_spec$resave_tfrecords)
force2GPU <- TRUE
nRealGridSeed <- 128L
nExamplesPerCell <- 10L
nRealGrid <- 704
GPU_MEM_FRAC <- NULL
UseShortOutcomes <- TRUE

nTimesLookahead <- (4L * 3L)
nTimesLookValidationInference <- nTimesLookahead
OverDoDataFrac <- 0.90
DecoderInNeuralODE <- FALSE
endAppend <- TRUE
print(ifelse(endAppend, yes = "Appending special tokens to **END**", no = "Appending special tokens to **START**"))

# setup command arguments
nOutcomes <- 1L
LEARNING_RATE_MAX_model <- 0.0002
LEARNING_RATE_MAX_pretrain <- 1e-5
useLSTM <- FALSE
doGrid <- TRUE
SimMode <- FALSE
nBatch <- as.integer(32L)
nSGD_pretrain <- 0L
nCheckpointsDefault <- 10L
nCheckpoints <- nCheckpointsDefault
nEpochesMax <- 9L
nSamples_max <- 20000L
nSGD_DefiningLRSeq <- nSGD_model <- as.integer(round(nEpochesMax * (nSamples_max / nBatch)))
PreTrain <- ifelse(nSGD_pretrain == 0, yes = FALSE, no = TRUE)
nTotalDiseases <- (nSynthDiseases <- 3L) + 1L
nSGD_posttrain <- nSGD_model
if (ReSaveTfRecords) {
  nSGD_pretrain <- nSGD_DefiningLRSeq <- nSGD_model <- 0L
}
specificOptState <- TRUE
SharedListNames <- c("TS")

# set results holder
HolderFolder <- sprintf("./SavedResults/Real/Results_%s", AnalysisName)
if (!dir.exists(sprintf("%s", HolderFolder))) {
  try(dir.create(sprintf("%s", HolderFolder)), TRUE)
}

# Phase 1: runtime helpers and multidisease data loading
ndm_source_extracted("SetupEnv/SuperLModel_helperFxns.R")
dataFormat <- analysis2_multidisease_spec$data_format
ndm_source_extracted("SetupData/MultiDiseaseRuns/SuperL_UniversalDataReader.R")

data_subset <- analysis2_multidisease_spec$data_subset

# setup data inputs pool
dataInputs_pool_orig <- dataInputs_pool <- dataInputs_colnames_past
if(any(!dataInputs_pool_orig %in% colnames(truth_df_red))){
  stop(sprintf("Stopping: %s not in colnames(truth_df_red) in MasterReal.R",
               paste(dataInputs_pool_orig[!dataInputs_pool_orig %in% colnames(truth_df_red) ],collapse = ",")))
}
# dataInputs_pool_orig[!dataInputs_pool_orig %in% colnames(input_df_red_full) ]
#dataInputs_pool <- c("FUTURE_NPI_DAT_bsg_stringency_index","NPI_DAT_bsg_stringency_index")

# setup data inputs pool (cont'd)
dataInputs_pool <- dataInputs_pool[dataInputs_pool != outcome_metric]
#dataInputs_pool_DropNone <- dataInputs_pool[dataInputs_pool!=""]
#dataInputs_pool_use <- c()
#nMonte_DataInterSpace <- 3L; for(comb_ in 3:length(dataInputs_pool_DropNone)){
#   comb_grid <- combn(length(dataInputs_pool_DropNone),comb_)
#   comb_grid <- as.matrix(comb_grid[,sample(1:ncol(comb_grid),min(ncol(comb_grid), nMonte_DataInterSpace),replace=FALSE)])
#   dataInputs_pool_use <- unique(c(dataInputs_pool_use,apply(comb_grid,2,function(ser){
#           paste(dataInputs_pool_DropNone[ser],collapse = "__") })))
#}
#dataInputs_pool <- dataInputs_pool

# uncomment if using ALL covariates 
# dataInputs_pool <- dataInputs_pool[which.max(nchar(dataInputs_pool))]

dataInputs_colnames <- dataInputs_pool

# evaluation periods 
evaluation_seq <- c(1L,2L,3L,4L)

# regenerate simulation grid 
if( RegenGrid <- FALSE ){
  print2("Generating real grid..."); 
  set.seed(999L); ndm_source_extracted("SetupData/SuperLModel_GenRealGrid.R")
}
if(!RegenGrid){ warning("Not re-generating real grid") }
print2("Reading in simulation grid...")
if (exists("analysis2_multidisease_grid", inherits = TRUE)) {
  RealGrid <- get("analysis2_multidisease_grid", inherits = TRUE)
} else {
  RealGrid <- as.data.frame(data.table::fread(sprintf("./Data/RunGrids/RealGrids/RealGrid_%s.csv", AnalysisName)))
}
nsgd_calibration <- get0("analysis2_nsgd_calibration", inherits = TRUE, ifnotfound = NULL)
if (is.null(nsgd_calibration)) {
  nsgd_resolver <- utils::getFromNamespace(".ndm_resolve_nsgd_calibration", "ndm")
  nsgd_calibration <- nsgd_resolver(
    mode = "multidisease",
    project_root = analysis2_multidisease_spec$project_root,
    analysis_name = AnalysisName,
    n_epoches_max = nEpochesMax,
    grid = RealGrid,
    grid_file = analysis2_multidisease_spec$grid_file,
    fallback_n_samples_train = nSamples_max
  )
}
nSamples_max <- as.integer(nsgd_calibration$anchor_max_n_samples_train)
nSGD_DefiningLRSeq <- nSGD_model <- as.integer(nsgd_calibration$resolved_n_sgd)
nSGD_posttrain <- nSGD_model
nCheckpoints <- analysis2_small_run_n_checkpoints(nSamples_max, nSGD_model, nCheckpointsDefault)
nSGDPolicy <- as.character(nsgd_calibration$policy)
nSGDAnchorScope <- as.character(nsgd_calibration$anchor_scope)
nSGDAnchorMaxSamplesTrain <- as.integer(nsgd_calibration$anchor_max_n_samples_train)
nSGDAnchorBatch <- as.integer(nsgd_calibration$anchor_n_batch)
nRealGrid <- nrow(RealGrid)

summary(which(RealGrid$ResaveThisTFRecord==1))
length(which(RealGrid$ResaveThisTFRecord==1))
dim(RealGrid)

#source("./Analysis2/SetupData/SuperLModel_MakeMasterData.R", encoding="utf-8") 

# scramble simgrid if not resaving tf record
if( !ReSaveTfRecords ){
  #RealGrid <- RealGrid[order(apply(RealGrid,1,function(zer){ rlang::hash(paste(zer,collapse="_")) })),]# scramble 
  # RealGrid <- RealGrid[order(RealGrid$BaseID),]# sort by base id to enable better comparisions
  RealGrid <- RealGrid[order(RealGrid$BaseID),] 
  if( RealGrid$BaseID[OUTER_ITERATION_SEQUENCE[1]] %% 2 == 1 ){ 
    # for ODD - order based on BaseID to enable better pairwise comparisons
  } 
  if( RealGrid$BaseID[OUTER_ITERATION_SEQUENCE[1]] %% 2 == 0 ){ 
    RealGrid[RealGrid$BaseID %% 2 == 0,] <- RealGrid[RealGrid$BaseID %% 2 == 0,][
      order(apply(RealGrid[RealGrid$BaseID %% 2 == 0,],
                  1, function(zer){ rlang::hash(paste(zer,collapse="_")) })), ]
    # plot(RealGrid$BaseID)
  } 
}

# which(RealGrid$ResaveThisTFRecord==1)
# dim(RealGrid);

for(OUTER_ITERATION in OUTER_ITERATION_SEQUENCE){
  print2(sprintf("STARTING outer iteration sequence %s...", OUTER_ITERATION))
  {
  set.seed( SEED_ <- ai(OUTER_ITERATION) )
    
  # process grid entry 
  RealEntry <- RealGrid[OUTER_ITERATION,]
  for(e_ in names(RealEntry)){ 
    eval(parse(text = sprintf("tmp_ <- f2n(RealEntry['%s'])",e_)))
    if(is.na(tmp_)){ eval(parse(text = sprintf("tmp_ <- as.character(RealEntry['%s'])",e_))) }
    eval(parse(text = sprintf("%s <- tmp_",e_)))
    eval(parse(text = sprintf("RealEntry['%s'] <- tmp_",e_)))
  }
  if(exists("nSamplesTrain") && !is.na(nSamplesTrain) && nSamplesTrain > 0){
    nBatch <- max(1L, min(as.integer(32L), as.integer(nSamplesTrain)))
    nSamples_max <- as.integer(nsgd_calibration$anchor_max_n_samples_train)
    nSGD_DefiningLRSeq <- nSGD_model <- as.integer(nsgd_calibration$resolved_n_sgd)
    nSGD_posttrain <- nSGD_model
    nCheckpoints <- analysis2_small_run_n_checkpoints(nSamples_max, nSGD_model, nCheckpointsDefault)
    nObsInference <- analysis2_small_run_n_obs_inference(
      n_samples_train = nSamplesTrain,
      n_batch = nBatch,
      configured = get0("nObsInference", inherits = FALSE, ifnotfound = NULL)
    )
  }
  modelingStrategyNameKey <- paste(c("RealMode", paste(names(RealEntry), 
                                                       RealEntry, sep = "_")), collapse = "__")
  
  # setup for tfrecord 
  TfRecordDir<-sprintf("./Data/RunTFRecords/RealTFRecords/%s",AnalysisName)
  if(!dir.exists(TfRecordDir)){
    dir.create(TfRecordDir, recursive = TRUE, showWarnings = FALSE)
  }
  need_canonical_tfrecords <- !all(file.exists(c(
    sprintf('%s/%s_%s.tfrecord', TfRecordDir, "train", RealEntry$BaseID),
    sprintf('%s/%s_%s.tfrecord', TfRecordDir, "inference", RealEntry$BaseID)
  )))
  if(!ReSaveTfRecords){
    if(need_canonical_tfrecords){
      warning(sprintf("Canonical TFRecords missing for BaseID %s; generating them on demand in DataGenerator_Real.R", RealEntry$BaseID))
    }
  }
  
  # Forces 
   #print( sprintf("Forcing model str %s", 
      #model_tex_loc <- "./Analysis2/ModelStructureTex/bayes_ode_SEIRS_FixedBeta_FixedGlobal.tex" ) )
      #model_tex_loc <- "./Analysis2/ModelStructureTex/bayes_ode_SEIRS_DynamicBeta_FixedGlobal.tex" ))
      #model_tex_loc <- "./Analysis2/ModelStructureTex/bayes_ode_SEIRS_DynamicBeta_DynamicGlobal.tex" ))
  # print( sprintf("Forcing float: %s", floatType <- "32" ))
  # print( sprintf("Forcing padding method: %s", paddingMethod <- "right" ))
  ModelType <- analysis2_model_type(analysis2_multidisease_spec, RealEntry$ModelType, default = "DecoderOnly")
  print(sprintf("Using model type: %s", ModelType))
  
   # setup master ODE solution parameters
   ndm_source_extracted("SetupEnv/SuperLModel_MasterImports.R")
  
   # setup some parameters
   {
     # summary( sort( unique( input_df_red_in$time_id ) )  )
     # summary( sort( unique( input_df_red_out$time_id ) )  )
     # make sure  maxTimesPast > past time in input_df_red_in
     maxTimesPast <- ContextLength
     # maxTimesPast <- 4L; warning("FORCING CONTEXT LENGTH OF 4 FOR TESTING")  # Disabled for production
     nTimesTotal <- nTimesLookahead + maxTimesPast
     minAnchoringTimeID <- 4L
     MIN_NA_ACCEPT_FRAC <- 4 / maxTimesPast
   }
   
   {
     NTimeSteps_SIM <- (nTimesLookahead+abs(maxTimesPast)  )*2L
     MaxSteps <- 100000L
     #dt0_init <- 10^(-7)
     dt0_init <- 10^(-1)
     #VI_TotalTimesInLikelihood <- (nTimesLookahead+abs(maxTimesPast)  ) # if predicing  full context + lookahead 
     VI_TotalTimesInLikelihood <- nTimesLookahead # if using lookahead context only 
     VI_SaveAt_ODE <- diffrax$SaveAt(ts = jnp$array(  1:VI_TotalTimesInLikelihood  ))
     diff_eq_solver <- VI_diff_eq_solver <- diffrax$Dopri8() # If you need accurate solutions at high tolerances then try diffrax.Dopri8.
     stepsize_controller = diffrax$PIDController(rtol = 1e-7, atol = 1e-9)
     diffraxInterpolator <- diffrax$LinearInterpolation
     
     MaxSteps <- ai( 10^6 )
     VI_SaveAt_ODE_sim <- diffrax$SaveAt(ts = jnp$array(  0L:(NTimeSteps_SIM-1L) ))
     VI_SaveAt_ODE_optim <- diffrax$SaveAt(ts = jnp$array(  0L:(VI_TotalTimesInLikelihood-1L) ))
     #VI_diff_eq_solver_optim <- VI_diff_eq_solver_dgp <- diffrax$Dopri8() # If you need accurate solutions at high tolerances then try diffrax.Dopri8.
     VI_diff_eq_solver_optim <- VI_diff_eq_solver_dgp <- diffrax$Tsit5() # good general solver
     dt0_init_dgp <- 1e-3; stepsize_controller_dgp = diffrax$PIDController(rtol = 1e-6 , atol = 1e-7) # required tolerance seems to be at least 1e-5
     if(!DecoderInNeuralODE){ 
        dt0_init_optim <- 1e-3; stepsize_controller_optim = diffrax$PIDController(rtol = 1e-5, atol = 1e-7) # required tolerance seems to be at least 1e-5
     }
     if(DecoderInNeuralODE){ 
       dt0_init_optim <- 1e-3; stepsize_controller_optim = diffrax$ConstantStepSize() # required tolerance seems to be at least 1e-5
     }
     #dt0_init_optim <- 10^(-1); stepsize_controller_optim <- diffrax$ConstantStepSize()
     #dt0_init_dgp <- dt0_init_optim <- 10^(-1); stepsize_controller_dgp <- stepsize_controller_optim <- diffrax$ConstantStepSize()
     diffraxInterpolator <- diffrax$LinearInterpolation
  }
     
  ### Setup data
  # drop data from ineligible places
  if(data_subset == "all"){ keep_place_ids <- unique(  truth_df_red$location_id ) }
  if(data_subset == "high_income"){
    # Check if LOC2_region_name column exists for high-income filtering
    if(!"LOC2_region_name" %in% colnames(truth_df_red)){
      warning("high_income subset requested but LOC2_region_name not available; using all locations")
      keep_place_ids <- unique(truth_df_red$location_id)
    } else {
      keep_place_ids <- unique(truth_df_red[
                                  truth_df_red$LOC2_region_name %in% c(
                                      "Western Europe","High-income North America",
                                       "Central Europe","High-income Asia Pacific"),]$location_id)
    }
  }
  truth_df_red <- truth_df_red[truth_df_red$location_id %in% keep_place_ids, ]
  # Handle Pop column - use POP_population if available, else NA
  if("POP_population" %in% colnames(truth_df_red)){
    truth_df_red$Pop <- truth_df_red$POP_population
  } else {
    truth_df_red$Pop <- NA
    warning("POP_population column not available; Pop set to NA")
  }
  truth_df_red <- truth_df_red[!is.na(truth_df_red$location_name),]
  #predicted_df_red <- predicted_df_red[predicted_df_red$location_id %in% keep_place_ids, ]
  
  # define numeric identifiers for place 
  truth_df_red$location_id_numeric <- as.integer(as.factor( truth_df_red$location_id ) ) - 1L # minus 1 for 0 indexing
  if(!any(truth_df_red$time_id==0)){stop("time_id seems to be non-zero indexed!")}
  
  # for place embedding 
  if( FALSE ){ 
    loc_name_id_walk <- tapply(truth_df_red$location_id_numeric, truth_df_red$location_name, unique)
    loc_name_id_walk <- cbind("location_name"=names(loc_name_id_walk),"location_id_numeric"=loc_name_id_walk)
    row.names(loc_name_id_walk) <- NULL 
    
    coordinates_mat <- tidygeocoder::geo(address = loc_name_id_walk[,"location_name"])
    coordinates_mat[coordinates_mat$address=="King and Snohomish Counties",c("lat","long")] <- data.frame(47.62, -122.40)
    coordinates_mat[coordinates_mat$address=="Washington except for King, Snohomish, and Spokane Counties",c("lat","long")] <- data.frame(47.60, -120.59)
    coordinates_mat[coordinates_mat$address=="Washington except for King, Snohomish, and Spokane Counties",c("lat","long")] <- data.frame(47.60, -120.59)
    coordinates_mat[coordinates_mat$address=="Catalonia",c("lat","long")] <- data.frame(41.8167, 1.4667)
    coordinates_mat[coordinates_mat$address=="Galicia",c("lat","long")] <- data.frame(42.8, -7.9)
    coordinates_mat <- cbind(loc_name_id_walk, coordinates_mat)
    coordinates_mat <- coordinates_mat[order(f2n(coordinates_mat$location_id_numeric)),]
    coordinates_mat[is.na(coordinates_mat$lat),]
    if(sum(is.na(coordinates_mat$lat)) > 0){stop("Stop: New NA lat")}
    write.csv(file = "./Data/coordinates_mat.csv", coordinates_mat)
    rm(loc_name_id_walk)
  }
  
  # use hard coded place embeddings? - NOTE: must be aligned with data generator 
  # coordinates_mat <- read.csv(file = "./Data/coordinates_mat.csv")
  
  # drop other SL models (in terms of # of Times predicting)
  nModelTimesThres <- 2
  #keep_models <- names(table(predicted_df_red$model_name)[table(predicted_df_red$model_name)>=nModelTimesThres])
  #print(" Dropping models: ")
  #print(names(table(predicted_df_red$model_name)[table(predicted_df_red$model_name)<nModelTimesThres]))
  #predicted_df_red <- predicted_df_red[predicted_df_red$model_name %fin% keep_models,]
  #mean(predicted_df_red$model_name %fin% keep_models == predicted_df_red$model_name %fin% keep_models)
  
  #unique_models <- unique( predicted_df_red$model_id )
  #predicted_df_red$model_id_red <- as.numeric(as.factor(predicted_df_red$model_id))
  #unique_models_key <- tapply(predicted_df_red$model_name,predicted_df_red$model_id_red,function(zer){ unique(zer)  })
  #nModels <- length( unique_models )
  #irlBasis_t_nModels <- tf$transpose(tf$constant( compositions::ilrBase(D = nModels), tfFloatType))
  #unique_locations <- unique( predicted_df_red$location_id )
  
  ## obtain test set given evaluation design
  {
    # sapply(evaluation_seq,function(e_){round(quantile(sort( unique( truth_df_red$time_id )),prob = e_/(max(evaluation_seq)+1)))})
    in_out_cutpoint <- round(quantile(sort( unique( truth_df_red$time_id )),
                                      prob = evaluationTime/(max(evaluation_seq)+1)))
    #times_out <- times_out # if just one prospective time
    #times_out <- c(in_out_cutpoint+1, (in_out_cutpoint+nTimesLookahead*c(1/4,2/4,3/4,4/4))) # look at four times into the future 
    times_out <- in_out_cutpoint+1
    times_out <- times_out[times_out<=max(truth_df_red$time_id)]
    times_in <- 0L:(in_out_cutpoint)
  }
  
  # some normalization factors
  AVERAGE_TRUTH <- apply( as.matrix(truth_df_red[,outcome_metric]),2, function(x){mean(x,  na.rm = TRUE ) })
  
  # setup data generator
  print2( "Defining data acquisition process..." )
  if(need_canonical_tfrecords){
    ReSaveTfRecords <- TRUE
  }
  ndm_source_extracted("SetupData/SuperLModel_DataGenerator_Real.R")
  if(need_canonical_tfrecords){
    ReSaveTfRecords <- FALSE
  }

  if(!ReSaveTfRecords){
  if(any(!sapply(unique(RealGrid$BaseID), function(s_){
    file.exists(sprintf('%s/%s_%s.tfrecord', TfRecordDir, "train", s_)) }))){
    warning(sprintf("Some tf records missing... check data generation pipeline!
           Make sure you're generating {%s} unique records", length(unique(RealGrid$BaseID))))
  }
  }
  #tmp_ <- paste0(TfRecordDir, "/train_",unique(RealGrid$BaseID),".tfrecord")
  #tmp_[!file.exists(tmp_)]; file.exists(tmp_)
  #tmp__ <- paste0(TfRecordDir, "/inference_",unique(RealGrid$BaseID),".tfrecord")
  #tmp__[!file.exists(tmp__)]; file.exists(tmp__)     
  
  if(nSGD_model > 0){ 
  # run build model sequence
  print2( "Building core ML model...")
  nOutcomes_ <- nOutcomes 
  ndm_source_extracted("ModelDefiners/SuperLModel_BuildML.R")
  if(nOutcomes_ != nOutcomes){stop(".tex mismatches nOutcomes vs. data - check in SuperLModel_MasterRealMultiDisease.R")}
  ModelList_init <- ModelList; state_init <- state
  ListIndices_shared <- ListIndices[  SharedListNames  ]
  ListIndices_notshared <- ListIndices[-ListIndices_shared] # not shared in pre-training
  
  # run training sequence
  TrainSeq <- ifelse(PreTrain, yes = list(c(rep("pre",times = 1),"real")), no = list("real"))[[1]]
  nSGD_model_init <- nSGD_model
  print2( "Starting training sequence...")
  trainCounter <- 0; for(TrainType_ in TrainSeq){
    IsPretraining <- ifelse(TrainType_ == "pre", yes = TRUE, no = FALSE)
    trainCounter <- trainCounter + 1
    if(IsPretraining){
      print2("Starting pre-training run...");
      
      print2("restart model list")
      ModelList[ ListIndices_notshared ] <- ModelList_init[ ListIndices_notshared ]
      nSGD_DefiningLRSeq <- nSGD_pretrain
      rollCompute_window <- 52L / 2L
      GLOBAL_ODE_NPOP <- as.vector( CONST_N_r )
      source( "./Analysis2/SetupData/SuperLModel_GenSimGrid.R" );
      SimGrid <- SimGrid[SimGrid$c_endogeneous > 0,]
      SimGrid <- SimGrid[sample(1:nrow(SimGrid),nSynthDiseases),]
      SimEntry <- SimGrid[1,]
  
      simCovariates <- c(
        "XPred_d_log" = "inc_death_per_capita_log",  # log incidental
        "XPred_c_log" = "inc_case_per_capita_log",
        "XPred_h_log" = "inc_hosp_per_capita_log",
  
        "XPred_rmd_log" = "rollmean_death_per_capita_log", # log  roll mean
        "XPred_rmc_log" = "rollmean_case_per_capita_log",
        "XPred_rmh_log" = "rollmean_hosp_per_capita_log",
  
        "XPred_rsd_log" = "rollstd_death_per_capita_log",  # log roll std
        "XPred_rsc_log" = "rollstd_case_per_capita_log",
        "XPred_rsh_log" = "rollstd_hosp_per_capita_log",
  
        "XPred_rxd_log" = "rollmedian_death_per_capita_log",  # log roll max
        "XPred_rxc_log" = "rollmedian_case_per_capita_log",
        "XPred_rxh_log" = "rollmedian_hosp_per_capita_log",
  
        #"XPred_rxd" = "rollmedian_death_per_capita",  # roll max
        #"XPred_rxc" = "rollmedian_case_per_capita",
        #"XPred_rxh" = "rollmedian_hosp_per_capita",
  
        #"XPred_d" = "inc_death_per_capita",  # incidental
        #"XPred_c" = "inc_case_per_capita",
        #"XPred_h" = "inc_hosp_per_capita",
  
        #"XPred_rmd" = "rollmean_death_per_capita", # roll mean
        #"XPred_rmc" = "rollmean_case_per_capita",
        #"XPred_rmh" = "rollmean_hosp_per_capita",
  
        #"XPred_rsd" = "rollstd_death_per_capita",  # roll std
        #"XPred_rsc" = "rollstd_case_per_capita",
        #"XPred_rsh" = "rollstd_hosp_per_capita",
  
        #"XPred_rxd" = "rollmedian_death_per_capita",  # roll max
        #"XPred_rxc" = "rollmedian_case_per_capita",
        #"XPred_rxh" = "rollmedian_hosp_per_capita",
  
        "zeros1" = "zeros1"
        #"zeros2" = "zeros2"
      )
      # cbind(simCovariates, dataInputs_colnames_past)
  
      # initialize holders for non-shared
      ModelList_notshared_pool <- replicate(list(ModelList_init[ ListIndices_notshared ]), n = nTotalDiseases )
      state_list <- replicate(list(state_init), n = nTotalDiseases )
      sim_dat_norm_list <- replicate(list(), n = nTotalDiseases )
      LEARNING_RATE_MAX <- LEARNING_RATE_MAX_pretrain
      ndm_source_extracted("ModelTrainers/SuperLModel_TrainDefine.R")
      if( specificOptState ){ opt_state_list <- replicate(list(opt_state), n = nTotalDiseases) }
  
      # unclear how to pass shared opt state for shared model params (?)
      ndm_source_extracted("SetupData/SuperLModel_DataGenerator_Sim.R")
  
      i_ <- 0; synthDisease_vec <- rep(NA,times = nSGD_pretrain)
      i__ <- (i_+1) ; for(jr in i__:nSGD_pretrain){
        # sample a disease
        synthDisease_vec[jr] <- sampledDiseaseNum <- sample(1:nTotalDiseases,1)
  
        # sampledDiseaseNum <- nTotalDiseases
  
        # specify batch generation scheme
        if(sampledDiseaseNum <= nSynthDiseases){
          GetBatch <- GetBatch_sim; SimEntry <- SimGrid[sampledDiseaseNum,]
        }
        if(sampledDiseaseNum > nSynthDiseases){ GetBatch <- GetBatch_real; SimEntry <- NULL }
  
        # slot in model list
        ModelList[  ListIndices_notshared  ] <- ModelList_notshared_pool[[ sampledDiseaseNum ]]
        state <- state_list[[ sampledDiseaseNum ]]
  
        # calibrate if not already in pool
        if(!sampledDiseaseNum %in% synthDisease_vec[c(0:(jr-1))] ){
          print2( sprintf("CALIBRATING DISEASE %s", sampledDiseaseNum) )
          if(sampledDiseaseNum <= nSynthDiseases){
            # used to get new SIM_GLOBAL_SCALE_MEAN and SIM_GLOBAL_SCALE_SD
            ndm_source_extracted("SetupData/SuperLModel_DataGenerator_Sim.R")
            sim_dat_norm_list[[ sampledDiseaseNum ]] <- list(  SIM_GLOBAL_SCALE_SD,
                                                               SIM_GLOBAL_SCALE_MEAN  )
          }
  
          ndm_source_extracted("SetupData/SuperLModel_CalibrateML.R")
        }
  
        if(sampledDiseaseNum <= nSynthDiseases){
          SIM_GLOBAL_SCALE_SD <- sim_dat_norm_list[[ sampledDiseaseNum ]][[1]]
          SIM_GLOBAL_SCALE_MEAN <- sim_dat_norm_list[[ sampledDiseaseNum ]][[2]]
        }
        if(specificOptState){ opt_state  <- opt_state_list[[ sampledDiseaseNum ]] }
  
        # training step and save
        print2("Train step...")
        nSGD_model <- i_ <- i_ + 1L
        ndm_source_extracted("ModelTrainers/SuperLModel_TrainDo.R")
  
        # slot in updated model lists, model states, optimizer states
        ModelList_notshared_pool[[ sampledDiseaseNum ]] <- ModelList[ ListIndices_notshared ]
        state_list[[ sampledDiseaseNum ]] <- state
        if(specificOptState){ opt_state_list[[ sampledDiseaseNum ]] <- opt_state }
  
        print2( tapply(in_loss_vec, synthDisease_vec[1:length(in_loss_vec)],function(x){ (tmp_<-zoo::rollmean(x,k = 5, na.rm=TRUE))[length(tmp_)] })  )
      }
      print2( sprintf("Total time: %s", (Sys.time()) - st0) )
    }
    if(!IsPretraining){
      # restart model list
      if(nSGD_pretrain == 0){
        # ModelList[ ListIndices_notshared ] <- ModelList_init[ ListIndices_notshared ]
        # state <- state_init
        ndm_source_extracted("SetupData/SuperLModel_CalibrateML.R")
      }
  
      # restart with learned state
      if(nSGD_pretrain > 0){
        GetBatch <- GetBatch_real
        ModelList[ ListIndices_notshared ] <- ModelList_notshared_pool[[ nTotalDiseases ]]
        state <- state_list[[ nTotalDiseases ]]
        if( specificOptState ){ opt_state  <- opt_state_list[[ nTotalDiseases ]] }
      }
  
      if( nSGD_model > 0 ){
        # remove from memory pretrained pool
        rm ( ModelList_notshared_pool , state_list , ModelList_init, state_init ); gc(); py_gc$collect()
  
        print2( "Starting final training run...")
        nSGD_DefiningLRSeq <- nSGD_model <- nSGD_model_init
  
        # must re-initialize unless TrainDefine.R initialized with nSGD_model + nSGD_pretrain
        LEARNING_RATE_MAX <- LEARNING_RATE_MAX_model
  
        print2( "Starting final training definition..." )
        ndm_source_extracted("ModelTrainers/SuperLModel_TrainDefine.R")
  
        print2( "Starting final train do..." )
        ndm_source_extracted("ModelTrainers/SuperLModel_TrainDo.R")
  
        print2(sprintf("Total time: %s", (st1 <- Sys.time()) - st0)); 
      }
    }
  }
  }
  }
  print2(sprintf("DONE with outer iteration sequence %s...", OUTER_ITERATION))
}
}
if( FALSE ){
 setwd(analysis2_multidisease_spec$project_root)
 ndm_source_extracted("ResultsAnalyze/SuperLModel_GenFigs.R")
}
analysis2_multidisease_result <- TRUE
}


####################################
# agent update: 
# talked with Serkan et al. today re: agent pipeline to validate IHME collaborators
# will get a generalized package version of the code end of week 








####################################
# paper update - where to go next: Nature? 
# idea: 
# paper now is a bit complicated in framing 
# Neural ODE part 
# Decoder-only part 
# Scaling part 




# Should we split them into 2 or even 3? 

# Possibly, for Nature, target: 
# decoder-only + scaling 

# then, in followup, do Neural ODE?

# Keeping the 3 bundled migth be ok, worry is really mostly pragmatic
# (how to communicate ideas as simply as possible). 
# Focsuing on one compute arm at a time also speeds up iterations/runs 

####################################
# Data Update
# IHME HIV data injested 

# Notes
# HIV - interesting patterns only start at year 10 out 
# case study of downturn in new cases in the early 2000s
# prevalence surveys are solid (incidence infered from these)
# high income countries (time series of prevalence)
# death data for HIV for high income countries
# HIV notifications (for high / middle income countries)
# flu data on github 

# Codebase - get ready for new runs, but faster 
# updated removing some depreciated dependencies
# new jax release (25.08; August 29, 2025) 
## allows flash attention support (possibly 9x faster decoding)
## jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation='cudnn')
#  3.33 
##
