# Content of SuperLModel_GenFigs.R
{
  rm(  list= ls() );options(error=NULL)
  setwd("~/Dropbox/CovidSuperlearner/")
  # install.packages( "~/Documents/helpeRs-software/helpeRs",repos = NULL, type = "source",force = F)
  print("Starting SuperLModel_GenFigs.R")

  print("Define data to use in analysis")
  #SimMode<-FALSE;AnalysisName <- "RealApril15";resultsFolder<-sprintf("./SavedResults/Real/Results_%s",AnalysisName); FiguresTag <- "RealTag"; TheGrid <- as.data.frame( data.table::fread( sprintf("./Data/RunGrids/RealGrids/RealGrid_%s.csv",AnalysisName) ) ) 
  #SimMode<-FALSE;AnalysisName <- "RealApril15-2000iters";resultsFolder<-sprintf("./SavedResults/Real/Results_%s",AnalysisName); FiguresTag <- "RealTag"; TheGrid <- as.data.frame( data.table::fread( sprintf("./Data/RunGrids/RealGrids/RealGrid_%s.csv",strsplit(AnalysisName, split="-")[[1]][1]) ) ) 
  
  SimMode<-TRUE;AnalysisName <- "BigSimsJuly15";resultsFolder<-sprintf("./SavedResults/Sim/Results_%s",AnalysisName); FiguresTag <- "SimTag"; TheGrid <- as.data.frame( data.table::fread( theFileName <- sprintf("./Data/RunGrids/SimGrids/SimGrid_%s.csv",strsplit(AnalysisName, split="-")[[1]][1]) ) )
  #SimMode<-TRUE;AnalysisName <- "BigSimsApril15-9000iters";resultsFolder<-sprintf("./SavedResults/Sim/Results_%s",AnalysisName); FiguresTag <- "SimTag"; TheGrid <- as.data.frame( data.table::fread( theFileName <- sprintf("./Data/RunGrids/SimGrids/SimGrid_%s.csv",strsplit(AnalysisName, split="-")[[1]][1]) ) )
  
  print("Starting: --Load in and preprocess--")
  {
    # helper imports
    library(dplyr);library(data.table)
    #floatType <- "32";force2GPU <- TRUE; source("./Analysis/SetupEnv/SuperLModel_MasterImports.R")
    ndm_source_extracted("SetupEnv/SuperLModel_helperFxns.R")
    
    # other key hyperparameters
    SaveFolder <- "./Figures/"; cexAxis <- 1.5; cexLab <- 2; cexMain = 2
    
    # load csvs
    print2("---Loading in csv's---")
    library(future.apply); plan(multisession(workers = 10L)) # plan() is needed to set up multicore use 
    dt_list <- future.apply::future_lapply(
                    grep(list.files(resultsFolder), pattern="\\.csv",value=T), 
                          function(csv_file_){ 
                        data.table::fread(sprintf("%s/%s", resultsFolder, csv_file_))})
    future::plan(sequential)   # go back to a single process
    res_mat <- data.table::rbindlist(dt_list, fill = TRUE)
    dim(res_mat <- as.data.frame( res_mat ))
    # colnames(res_mat) <- gsub(colnames(res_mat),pattern="Week",replace="Time")
    
    table(res_mat$i_in_sgd)
    # res_mat <- res_mat[res_mat$i_in_sgd <= 28125,]
    
    # generate pair id 
    res_mat$PairID_recon <- paste(
                             as.character(res_mat$BaseID), 
                             as.character(res_mat$i_in_sgd), 
                             as.character(res_mat$nSamplesTrain), 
                             as.character(res_mat$ModelDepth), 
                             as.character(res_mat$ModelDims), sep = "_")
    # res_mat <- res_mat[1:100000,]
    #table(res_mat$OSSType)
    
    # sanity checks 
    # table(res_mat$i_in_sgd)
    # length(unique( res_mat$sim_index ) ) ; length(unique( res_mat$BaseID ) ) 
    # (table(res_mat$nSamplesTrain))
    
    # preprocessing 
    if(SimMode){  nTimesLookValidationInferenceTrainedTo <- 12L; nTimesLookValidationInference <- 50L }
    if(!SimMode){ 
      print2("---Pre-processing Real csv's (individ -> agg)---")
      res_mat_orig <- res_mat
      nTimesLookValidationInferenceTrainedTo <- nTimesLookValidationInference <- 12L 
      ndm_source_extracted("ResultsAnalyze/SuperLModel_GenFigs_RealPreprocess.R")
      res_mat <- res_mat_agg # override individual with agg results 
    } 
    
    # create summary measures
    StartTime_SkillSummary <- round(nTimesLookValidationInferenceTrainedTo*0.75)
    res_mat$SkillSummary <- rowMeans(res_mat[,paste0("SkillTime",StartTime_SkillSummary : nTimesLookValidationInferenceTrainedTo)], na.rm=T)
    res_mat$RSSBaselineSummary <- rowMeans(res_mat[,paste0("RSSBaselineTime",StartTime_SkillSummary : nTimesLookValidationInferenceTrainedTo)], na.rm=T)
    res_mat$RSSPredSummary <- rowMeans(res_mat[,paste0("RSSPredTime",StartTime_SkillSummary : nTimesLookValidationInferenceTrainedTo)], na.rm=T)
    
    # save to disk for shiny app 
    data.table::fwrite(res_mat, file = sprintf("~/Downloads/%s.csv", ifelse(SimMode, yes = "res_mat_sim",no = "res_mat_real")))
  }
  
  print("Some initial analyses")
  {
    # subset last itter 
    res_mat_ <- res_mat[res_mat$i_in_sgd == max(res_mat$i_in_sgd),]
    print(sprintf("---Marginal skill at training's end at week 11 with matrix dimensions %s---", nrow(res_mat_)))
    print( tapply(res_mat_$SkillTime11,res_mat_$ModelType,summary) ) 
    #print(tapply(res_mat_$SkillTime11,res_mat_$ModelType,function(x){plot(ecdf(x))}))
    
    # quick check of perf
    myForest_ <- randomForest::randomForest(
      y = res_mat_$SkillSummary, 
      x = res_mat_[,setdiff(paste0(colnames(TheGrid),""), "BaseID")])
    myForest_$importance[order(myForest_$importance),]
    #tapply(res_mat_$SkillSummary,res_mat_$initialNormType,summary)
    #tapply(res_mat_$SkillSummary,res_mat_$nSamplesTrain,summary)
    #tapply(res_mat_$SkillSummary,res_mat_$initialTransform,summary)
    #tapply(res_mat_$SkillSummary,res_mat_$ModelType,summary)
    #tapply(res_mat_$SkillSummary,res_mat_$dataInputs,summary)
    
    # variable transformations 
    res_mat$nParamsModel_log <- log(f2n(res_mat$nParamsModel), base = 10)
    res_mat$nSamplesTrain_log <- log(f2n(res_mat$nSamplesTrain), base = 10)
    res_mat$i_in_sgd_log <- log(f2n(res_mat$i_in_sgd), base = 10)
    res_mat$atEpoch_log <- log(f2n(res_mat$atEpoch), base = 10)
    res_mat$ContextLength_log <- log(f2n(res_mat$ContextLength), base = 2)
    res_mat$ModelDepth_log <- log(f2n(res_mat$ModelDepth), base = 2)
    res_mat$ModelDims_log <- log(f2n(res_mat$ModelDims), base = 2)
    res_mat$TimesSeen      <- f2n(res_mat$ContextLength) * f2n(res_mat$nSamplesTrain)
    res_mat$TimesSeen_log  <- log10(res_mat$TimesSeen)
    res_mat$UpdatesPerParam <- res_mat$i_in_sgd / res_mat$nParamsModel
    res_mat$UpdatesPerParam_log <- log10(res_mat$UpdatesPerParam)
    res_mat$DataParamRatio             <- res_mat$nSamplesTrain / res_mat$nParamsModel
    res_mat$DataParamRatio_log10       <- log10(res_mat$DataParamRatio)
    
    Map2ConstrainedSkill <- function(x){ 1 - exp(x) }
    InvMap2ConstrainedSkill <- function(x){ -log( (1 - x)+0.001  ) }
    InvMap2ConstrainedSkill( 1)
    
    # plot(InvMap2ConstrainedSkill(x<-rnorm(100)),x)
    
    res_mat$SkillSummary_inv <- InvMap2ConstrainedSkill(res_mat$SkillSummary)
    cor(res_mat$SkillSummary, res_mat$SkillSummary_inv)
    # plot(res_mat$SkillSummary, res_mat$SkillSummary_inv)
    max(Map2ConstrainedSkill(InvMap2ConstrainedSkill(res_mat$SkillSummary)) - res_mat$SkillSummary)
    
    #Map2ConstrainedBaselineRSS <- function(x){ exp(x) }
    #InvMap2ConstrainedBaselineRSS <- function(x){ log( x )  }
    
    Map2ConstrainedBaselineRSS <- function(x) { ifelse(x > 20, x, log1p(exp(x))) }
    InvMap2ConstrainedBaselineRSS <- function(y) { log(expm1(y)) }
    
    res_mat$RSSBaselineSummary_inv <- InvMap2ConstrainedBaselineRSS(res_mat$RSSBaselineSummary)
    max(Map2ConstrainedBaselineRSS(InvMap2ConstrainedBaselineRSS(res_mat$RSSBaselineSummary)) - res_mat$RSSBaselineSummary)
    
    # sanity - this should yield *two*'s as each pair entry is used twice 
    if( any( tapply(as.character(res_mat_$model_tex_loc),
           res_mat_$PairID_recon, table) ) > 2 ){
      stop("Stopping: Some PairIDs are tripliced; REQUIRE 2 per only")
    }
    print("Starting: --Done with load in and preprocess--")
  }
  
  print("Plotting comparision for skill results") # paired by PairID_recon, with pairing by ModelType
  {
    print2("---Starting pairwise comparisions (Decoder vs. NeuralODE)---")
    library(dplyr); library(ggplot2)
    
    # Keep only the final iteration (or whichever iteration you care about)
    res_mat_forpair <- res_mat#[res_mat$i_in_sgd == min(res_mat$i_in_sgd), ]
    # note: individual results should be overridden with aggregate results in Real case
    
    # Subset by model types you want to compare: e.g., Transformer vs. NeuralODE
    res_decoder   <- subset(res_mat_forpair, 
                            ModelType == "DecoderOnly")
    res_neuralODE <- subset(res_mat_forpair, 
                            ModelType == "NeuralODE")
    
    # Merge so each row has both Decoder skill & NeuralODE skill, matched by PairID
    # for real data anlaysis - do more merging before this step  

    # memory issues here # -> should yeild ONE (one data entry per merged pair)
    table(table(res_decoder$PairID_recon) )
    table(table(res_neuralODE$PairID_recon))

    # merge data 
    print2("---Merging pairwise (Decoder vs. NeuralODE)---")
    paired_df <- merge(
      res_decoder, 
      res_neuralODE,
      by = "PairID_recon", 
      suffixes = c("_Decoder", "_NeuralODE")
    )
    # -> should be close to 1
    if( mean(res_decoder$PairID_recon %in% res_neuralODE$PairID_recon) < 0.10){
      stop("Not enough overlap between PairID_recon in Decoder vs. NeuralODE -- Check code!")
    }
    head(cbind(sort(res_decoder$PairID_recon)[1:5],sort(res_neuralODE$PairID_recon)[1:5]))
        
    #plot(paired_df$SkillSummary_Decoder, paired_df$SkillSummary_NeuralODE,
         #xlim = c(-1,1), ylim = c(-1,1)); abline(a=0,b=1)
    
    # Now make a scatterplot comparing the skills
    print2("---Comparision scatterplots - Decoder vs. NeuralODE---")
    {
    gsvs <- ggplot(paired_df[paired_df$i_in_sgd_Decoder==max(paired_df$i_in_sgd_Decoder),],
      aes(x = SkillSummary_Decoder, y = SkillSummary_NeuralODE)) +
      geom_point(alpha = 0.7) +
      geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
      theme_minimal(base_size = 14,
                    base_line_size = 0) +
      labs(
        title = "Comparison of Skill Index: Decoder vs NeuralODE",
        x = "Skill Index (Decoder)",
        y = "Skill Index (NeuralODE)"
      )
    ggsave(sprintf("./Figures/gsvs_SimMode%s.pdf", SimMode),
           plot = gsvs, 
           device = "pdf", 
           width = 8, height = 8)
  }
    
    # Define a binary outcome: 1 if Transformer skill is higher, 0 otherwise
    paired_df$WinNeuralODE <- 1*(paired_df$SkillSummary_NeuralODE >= 
                                   paired_df$SkillSummary_Decoder)
    
    # generate win variable here 
    res_mat$WinNeuralODE_indicator <- paired_df$WinNeuralODE[
      match(res_mat$PairID_recon, paired_df$PairID_recon)
    ]
    
    # CDF PLOTs
    i_seq <- NeuralWinProb <- c(); for(i___ in sort(unique(paired_df$i_in_sgd_Decoder)) ){ 
      
      # 1. Gather all skill values
      paired_df_ <- paired_df[paired_df$i_in_sgd_Decoder == i___,]
      vals <- sort(unique(c(paired_df_$SkillSummary_Decoder, 
                            paired_df_$SkillSummary_NeuralODE)))
      
      # 2. Compute empirical CDF for Decoder and NeuralODE
      decoder_cdf   <- sapply(vals, function(v) mean(paired_df_$SkillSummary_Decoder <= v))
      neuralode_cdf <- sapply(vals, function(v) mean(paired_df_$SkillSummary_NeuralODE <= v))
      
      ks_stat <- ks.test(x = paired_df_$SkillSummary_Decoder, 
                         y = paired_df_$SkillSummary_NeuralODE,
                         alternative = "two.sided")
      print2( ks_stat$p.value) 
      
      # append
      i_seq <- c(i_seq,i___)
      NeuralWinProb <- c(NeuralWinProb,
                         mean(paired_df_$SkillSummary_NeuralODE >= paired_df_$SkillSummary_Decoder))
      
      # 3. Combine into data frame
      cdf_df <- data.frame(
        value         = vals,
        DecoderCDF    = decoder_cdf,
        NeuralODECDF  = neuralode_cdf
      )
      
      # 4. Plot CDF vs CDF
      gcdfcdf <- ggplot(cdf_df, aes(x = DecoderCDF, y = NeuralODECDF)) +
        geom_line(color = "blue", size = 1) +
        # Draw the diagonal line (same distribution = x = y)
        geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
        coord_fixed() +
        theme_minimal(base_size = 14,
                      base_line_size = 0) +
        labs(
          title = sprintf("P–P (CDF-vs-CDF) plot: Decoder vs NeuralODE (%s)",i___),
          subtitle = "Below 45 degree line indicates better NeuralODE skill",
          x     = "CDF(Decoder)",
          y     = "CDF(NeuralODE)"
        )
      ggsave(sprintf("./Figures/gcdfcdf_%s.pdf",i___),
             plot = gcdfcdf, 
             device = "pdf",
             width = 8, height = 6)
      
      # Plot density of absolute skill for Decoder vs NeuralODE (dropping mean)
      { 
      SkillSummary_ <- rbind("Decoder" = summary(paired_df_$SkillSummary_Decoder)[-4],
                             "Neural ODE" =summary(paired_df_$SkillSummary_NeuralODE)[-4]
                             )
      SkillSummary_[] <- helpeRs::fixZeroEndings(as.character(round(SkillSummary_[], 3)), 3)
      latex_code <- capture.output(stargazer::stargazer(as.matrix(SkillSummary_),
                                                        label = sprintf("tab:SkillSummarySimMode%s",SimMode),
                                                        font.size = "small",
                                                        title = sprintf("
                                                        Summary statistics of skill distributions for decoder-only and neural models.
                                                                      %s",
                                                                        ifelse(SimMode, 
                                                                               yes = "Case: simulation.",
                                                                               no = "Case: COVID-19 analysis."))
      ))
      writeLines( latex_code,
                  con = sprintf("./Figures/SkillSummaryTab_SimMode%s_i%s.tex",SimMode, i___) )
      }
    }
    writeLines( latex_code,
                con = sprintf("./Figures/SkillSummaryTab_SimMode%s_i%s.tex",SimMode, "Final") )
    # pdfs to pdf 
    # https://www.coolutils.com/online/PDF-to-MP4
    # reverse mp4
    # https://www.online-convert.com/result#j=2c2ddf87-5ed3-46cb-8f12-6b7b59cc36f0
    
    # plot by iteration number 
    pdf(sprintf("./Figures/NeuralODEWinByIter_SimMode%s.pdf",SimMode))
    {
    par(mar=c(5,5,5,1))
    plot(
      i_seq, NeuralWinProb,
      type = "p", log = "x", pch = 19,
      xlab = "Iteration (log scale)", ylab = "Probability(NeuralODE Skill > Decoder)",
      main = "NeuralODE vs Decoder\nWin Probability by Iteration",lwd = 2,
      cex.lab = 1.75, cex.main = 1.75, 
      col  = "black", ylim = c(0.5-0.2,0.5+0.2)
    )
    fit <- lm(NeuralWinProb ~ log(i_seq))
    abline(h=0.5, lty = 3, col = "gray", lwd = 2)
    fit <- lm(NeuralWinProb ~ log(i_seq))
    pred <- predict(fit, newdata = data.frame(i_seq = i_seq), interval = "confidence")
    lines(i_seq, pred[,"fit"], col = "red", lwd = 2)
    lines(i_seq, pred[,"lwr"], col = "red", lty = 2)
    lines(i_seq, pred[,"upr"], col = "red", lty = 2)
    }
    dev.off()
    
    # plot by data size 
    # ---------------------------------------------------------------------------------
    # NeuralODE vs Decoder ‑‑ Win probability by DATA‑SET SIZE
    # ---------------------------------------------------------------------------------
    {
      print2("---NeuralODE vs Decoder win probability by DATA‑SET SIZE---")
      ## 1.  Keep only the final iteration for a clean comparison  --------------------
      paired_df_EndOfTraining <- paired_df[paired_df$i_in_sgd_Decoder ==
                                     max(paired_df$i_in_sgd_Decoder), ]
      
      ## 2.  Compute P( Skill_NeuralODE ≥ Skill_Decoder ) for each nSamplesTrain ------
      dsizes <- sort(unique(paired_df_EndOfTraining$nSamplesTrain_Decoder))
      NeuralWinProb_ds <- sapply(
        dsizes, function(ds) {
          rows <- paired_df_EndOfTraining$nSamplesTrain_Decoder == ds
          mean(paired_df_EndOfTraining$SkillSummary_NeuralODE[rows] >=
                 paired_df_EndOfTraining$SkillSummary_Decoder[rows]) })
      n_per_ds <- sapply(
        dsizes, function(ds) sum(paired_df_EndOfTraining$nSamplesTrain_Decoder == ds))
      se_      <- sqrt(NeuralWinProb_ds * (1 - NeuralWinProb_ds) / n_per_ds)
      ci_low  <- pmax(0, NeuralWinProb_ds - 1.96 * se_)
      ci_high <- pmin(1, NeuralWinProb_ds + 1.96 * se_)
      
      pdf(sprintf("./Figures/NeuralODEWinByDatasetSize_SimMode%s.pdf",SimMode))
      {
      par(mar = c(5, 5, 5, 1))
      plot(dsizes, NeuralWinProb_ds,
           log      = "x",           # log‑scale x‑axis
           type     = "p",
           pch      = 19,
           col      = "black",
           ylim     = c(0.3, 0.9),
           xlab     = "Training Set Size",
           ylab     = "Prob( NeuralODE Skill ≥ Decoder )",
           main     = "NeuralODE vs Decoder\nWin Probability by Training Set Size",
           cex.lab  = 1.75,
           cex.main = 1.75)
      
      ## Reference line at 0.5
      abline(h = 0.5, lty = 3, col = "gray", lwd = 2)
      
      ## Error bars: SE‑based 95 % CIs
      arrows(x0 = dsizes, y0 = ci_low,
             x1 = dsizes, y1 = ci_high,
             angle = 90, code = 3, length = 0.05,
             col = "darkred", lwd = 2)
      }
      dev.off()
      
      ##  Compute P( Skill_NeuralODE ≥ Skill_Decoder ) for each per parameter update ------
      val_ <- sort(unique(paired_df_EndOfTraining$UpdatesPerParam_Decoder))
      NeuralWinProb_ds <- sapply(
        val_, function(ds) {
          rows <- paired_df_EndOfTraining$UpdatesPerParam_Decoder == ds
          mean(paired_df_EndOfTraining$SkillSummary_NeuralODE[rows] >=
                 paired_df_EndOfTraining$SkillSummary_Decoder[rows]) })
      n_per_ds <- sapply(
        val_, function(ds) sum(paired_df_EndOfTraining$UpdatesPerParam_Decoder == ds))
      se_      <- sqrt(NeuralWinProb_ds * (1 - NeuralWinProb_ds) / n_per_ds)
      ci_low  <- pmax(0, NeuralWinProb_ds - 1.96 * se_)
      ci_high <- pmin(1, NeuralWinProb_ds + 1.96 * se_)
      
      pdf(sprintf("./Figures/NeuralODEWinByPerParamData_SimMode%s.pdf",SimMode))
      {
        par(mar = c(5, 5, 5, 1))
        plot(val_, NeuralWinProb_ds,
             log      = "x",           # log‑scale x‑axis
             type     = "p",
             pch      = 19,
             col      = "black",
             ylim     = c(0, 1.1),
             xlab     = "Training Set Size",
             ylab     = "Prob( NeuralODE Skill ≥ Decoder )",
             main     = "NeuralODE vs Decoder\nWin Probability by Training Set Size",
             cex.lab  = 1.75,
             cex.main = 1.75)
        
        ## Reference line at 0.5
        abline(h = 0.5, lty = 3, col = "gray", lwd = 2)
        
        ## Error bars: SE‑based 95 % CIs
        arrows(x0 = dsizes, y0 = ci_low,
               x1 = dsizes, y1 = ci_high,
               angle = 90, code = 3, length = 0.05,
               col = "darkred", lwd = 2)
      }
      dev.off()
      
      # summarize
      plot(tapply(paired_df_EndOfTraining$SkillSummary_Decoder,
                  paired_df_EndOfTraining$ContextLength_Decoder,
                  median))
      plot(tapply(paired_df_EndOfTraining$SkillSummary_Decoder,
                  paired_df_EndOfTraining$ModelDims_Decoder,
                  median))
      plot(tapply(paired_df_EndOfTraining$SkillSummary_Decoder,
                  paired_df_EndOfTraining$nSamplesTrain_Decoder,
                  median))
      
      # forests
      #paired_df_EndOfTraining$SkillSummary_Decoder[paired_df_EndOfTraining$SkillSummary_Decoder < -4] <- -4
      #paired_df_EndOfTraining$SkillSummary_NeuralODE[paired_df_EndOfTraining$SkillSummary_NeuralODE < -4] <- -4
      DecoderForest <- randomForest::randomForest(
        y = paired_df_EndOfTraining$SkillSummary_inv_Decoder, 
        x = paired_df_EndOfTraining[,setdiff(paste0(colnames(TheGrid),"_Decoder"), 
                                             c("BaseID_Decoder","ResaveThisTFRecord_Decoder"))])
      DecoderImport <- DecoderForest$importance#[order(DecoderForest$importance),]
      
      NeuralODEForest <- randomForest::randomForest(
        y = paired_df_EndOfTraining$SkillSummary_inv_NeuralODE, # inv 
        x = paired_df_EndOfTraining[,setdiff(paste0(colnames(TheGrid),"_NeuralODE"), 
                                             c("BaseID_NeuralODE","ResaveThisTFRecord_NeuralODE"))])
      NeuralODEImport <- NeuralODEForest$importance#[order(NeuralODEForest$importance),]
      
      DiffForest <- randomForest::randomForest(
        y = paired_df_EndOfTraining$SkillSummary_inv_NeuralODE - 
                    paired_df_EndOfTraining$SkillSummary_inv_Decoder, 
        x = paired_df_EndOfTraining[,setdiff(paste0(colnames(TheGrid),"_Decoder"), 
                                                   c("BaseID_Decoder","ResaveThisTFRecord_Decoder"))]
        )
      DiffImport <- DiffForest$importance#[order(NeuralODEForest$importance),]
      DiffForest$importance[order(NeuralODEForest$importance),]
      
      if(FALSE){ 
        # split into short vs. long contexts at the median
        med_ctx <- median(paired_df_EndOfTraining$ContextLength_Decoder)
        for(grp in c("Short","Long")) {
          df <- paired_df_EndOfTraining[
            (grp=="Short" & paired_df_EndOfTraining$ContextLength_Decoder <  med_ctx) |
              (grp=="Long"  & paired_df_EndOfTraining$ContextLength_Decoder >= med_ctx), ]
          rf <- randomForest::randomForest(
            y = df$SkillSummary_inv_NeuralODE - df$SkillSummary_inv_Decoder,
            x = df[, setdiff(paste0(colnames(TheGrid),"_Decoder"),
                             c("BaseID_Decoder","ResaveThisTFRecord_Decoder"))]
          )
          cat("\nVariable importance for", grp, "contexts:\n")
          print(rf$importance[order(rf$importance), , drop=FALSE])
          # policy_effectiveness_Decoder
        }  
      }
      
      row.names(DiffImport) <- gsub(row.names(DiffImport), pattern = "_Decoder",replace="")
      row.names(DecoderImport) <- gsub(row.names(DecoderImport), pattern = "_Decoder",replace="")
      row.names(NeuralODEImport) <- gsub(row.names(NeuralODEImport), pattern = "_NeuralODE",replace="")
      mean(row.names(NeuralODEImport) == row.names(DecoderImport)) # == 1
      plot(DecoderImport,NeuralODEImport); abline(a=0,b=1)
      plot(DecoderImport,DiffImport); abline(a=0,b=1)
      plot(NeuralODEImport,DiffImport); abline(a=0,b=1)
      DecoderImport[order(DecoderImport),]
      NeuralODEImport[order(NeuralODEImport),]
      DiffImport[order(DiffImport),]
      cor(cbind(DecoderImport, NeuralODEImport, DiffImport))
      
      # other figs 
      {
      # 1. Bin axes (feel free to pick break widths that leave ≥20 pairs per bin)
      paired_df_EndOfTraining$b_ctx <- as.factor(paired_df_EndOfTraining$ContextLength_Decoder)#cut(paired_df_EndOfTraining$ContextLength_Decoder,
                             #breaks = 2^seq(0, log2(max(paired_df_EndOfTraining$ContextLength_Decoder)), by = 1),
                             #right = FALSE)
      paired_df_EndOfTraining$b_nsamp <- as.factor(paired_df_EndOfTraining$nSamplesTrain_Decoder)#cut(paired_df_EndOfTraining$nSamplesTrain_Decoder,
                               #breaks = 10^seq(3, log10(max(paired_df_EndOfTraining$nSamplesTrain_Decoder)), by = 0.5),
                               #right = FALSE)
      
      # 2. Aggregate win‑probability & gap
      phase <- paired_df_EndOfTraining |>
        group_by(b_ctx, b_nsamp) |>
        summarise(p_win = mean(SkillSummary_NeuralODE >= SkillSummary_Decoder),
                  gap   = median(abs(SkillSummary_NeuralODE - SkillSummary_Decoder)),
                  n     = n(), .groups = "drop") |>
        filter(n >= 20)                   # reliability filter
      
      # 3. Plot
      p <- ggplot(phase, aes(x = b_ctx, y = b_nsamp, fill = p_win)) +
        geom_tile() +
        geom_contour(aes(z = p_win), breaks = 0.5, colour = "black",
                     linetype = "dashed", size = 1.2) +
        geom_text(aes(label = scales::percent(p_win, accuracy = 1)),
                  colour = "white", size = 3) +
        scale_fill_viridis_c(name = "P(NeuralODE > Decoder)",
                             limits = c(0,1)) +
        scale_x_discrete(expand = c(0,0)) +
        scale_y_discrete(expand = c(0,0)) +
        labs(title = "When does NeuralODE beat a Decoder‑only model?",
             x = "Context length", y = "Training-set size") +
        theme_minimal(base_size = 14) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
      ggsave(p, 
             file = sprintf("./Figures/PhaseDiagram_NeuralODEvsDecoder_SimMode%s.pdf",SimMode),
             width = 8, height = 6)
    }
    }
    
    #-----------------------------------------------------------------------#
    print2("---Logistic regression: Probability( NeuralODE wins | predictors )---")
    if(SimMode){
    m_logit <- glm(
      WinNeuralODE ~ 
        sigma_NeuralODE + 
        gamma_NeuralODE +
        c_endogeneous_NeuralODE + 
        policy_effectiveness_NeuralODE +
        ContextLength_NeuralODE + 
        nParamsModel_log_NeuralODE + 
        nSamplesTrain_log_NeuralODE,
      data   = paired_df[paired_df$i_in_sgd_Decoder == max(paired_df$i_in_sgd_Decoder), ],
      family = binomial("logit")
    )
    summary(m_logit)
    
    # Predicted probabilities and AUC
    paired_df$pred_prob_logit <- predict(m_logit, 
                                         newdata = paired_df, 
                                         type = "response")
    roc_logit <- pROC::roc(paired_df$WinNeuralODE, 
                           paired_df$pred_prob_logit)
    cat("AUC (Logistic) =", round(pROC::auc(roc_logit), 3), "\n")
    table(paired_df$WinNeuralODE)
    plot(paired_df$WinNeuralODE, paired_df$pred_prob_logit)
    
    # heat mappers 
    MakeHeatMap(  factor1 = c("log(Context length)" = "ContextLength_NeuralODE"),
                  factor2 = c("log(# of Samples)" = "nSamplesTrain_log_NeuralODE"),
                  outcome = "WinNeuralODE",
                  pdf_path = sprintf("./Figures/Heat_Win.pdf"),
                  useLog = "", 
                  dat = (paired_df[paired_df$i_in_sgd_Decoder==max(paired_df$i_in_sgd_Decoder),
                                   colnames(m_logit$model)]),
                  lm_obj =  m_logit,
                  extrap_factor1 = 1, extrap_factor2 = 1, 
                  OutcomeTransformFxn = function(x){e1071::sigmoid(x)}, 
                  openBrowser = F )
    
    # additional tests 
    { 
    test_ <- glm(
      SkillSummary_NeuralODE ~ 
        floatType_NeuralODE+
        sigma_NeuralODE + 
        gamma_NeuralODE +
        c_endogeneous_NeuralODE + 
        policy_effectiveness_NeuralODE +
        ContextLength_NeuralODE + 
        nParamsModel_log_NeuralODE + 
        nSamplesTrain_log_NeuralODE,
      data   = paired_df[paired_df$i_in_sgd_NeuralODE == max(paired_df$i_in_sgd_NeuralODE), ],
      family = 'gaussian'
    )
    
    df_ <- paired_df[paired_df$i_in_sgd_NeuralODE == max(paired_df$i_in_sgd_NeuralODE), ]
    myForest_NeuralODESkill <- randomForest::randomForest(
                               y = df_$SkillSummary_NeuralODE, 
                               x = df_[,paste0(colnames(TheGrid),"_NeuralODE")])
    myForest_NeuralODESkill$importance[order(myForest_NeuralODESkill$importance),]
    
    myForest_DiffSkill <- randomForest::randomForest(
      y = df_$SkillSummary_Decoder-df_$SkillSummary_NeuralODE, 
      x = df_[,paste0(colnames(TheGrid),"_NeuralODE")])
    myForest_DiffSkill$importance[order(myForest_DiffSkill$importance),]
    
    test_ <- glm(
        SkillSummary_NeuralODE ~ 
        floatType_NeuralODE+
        sigma_NeuralODE + 
        gamma_NeuralODE +
        c_endogeneous_NeuralODE + 
        policy_effectiveness_NeuralODE +
        ContextLength_NeuralODE + 
        nParamsModel_log_NeuralODE + 
        nSamplesTrain_log_NeuralODE,
      data   = df_,
      family = 'gaussian'
    )
    summary( test_ )
    paired_df$floatType_NeuralODE
    }
    }
    
    {
      # ----------------------------------------------------------------------
      # Pretty scaling plot: context length  ✕  # observations
      # ----------------------------------------------------------------------
      library(patchwork)   # side‑by‑side layouts
      library(tidyr)
      
      ## 2.  Gather into long form -------------------------------------------
      long_scaling <- paired_df_EndOfTraining |>
        ## keep only the columns we need
        dplyr::select(PairID_recon,
                      ContextLength_Decoder, SkillSummary_Decoder,
                      ContextLength_NeuralODE, SkillSummary_NeuralODE,
                      nSamplesTrain_Decoder, SkillSummary_Decoder,
                      nSamplesTrain_NeuralODE, SkillSummary_NeuralODE,
                      #c_endogeneous_Decoder, c_endogeneous_NeuralODE,
                      #policy_effectiveness_Decoder, policy_effectiveness_NeuralODE,
                      ModelDims_Decoder, ModelDims_NeuralODE
                      ) |>
        ## reshape: names like ContextLength_log_Decoder → value = ContextLength_log, Model
        pivot_longer(
          cols      = -PairID_recon,
          names_to  = c(".value", "Model"),      # the bit before the final underscore is the ".value"
          names_sep = "_(?=[^_]+$)"              # split on the last underscore
        ) |>
        ## nicer labels
        mutate(Model = factor(Model,
                              levels = c("Decoder", "NeuralODE"),
                              labels = c("Decoder", "NeuralODE")))
      
      ## 3.  Plot helpers -----------------------------------------------------
      base_theme <- theme_minimal(base_size = 14) +
        theme(panel.grid.major  = element_line(size = 0.3),
              panel.grid.minor  = element_blank(),
              legend.position   = "bottom")
      pal <- c("Decoder"   = "#e64b35",   # deep red
               "NeuralODE" = "#4da6ff")   # sky blue
      
      ## 4.  Panel A – scaling in context length -----------------------------
      lowess_span <- 5
      p_context <- ggplot(long_scaling,
                          aes(x = ContextLength,
                              y = SkillSummary,
                              colour = Model)) +
        geom_point(alpha = .65, size = 1.5) +
        geom_smooth(se = FALSE, method = "loess",
                    span = lowess_span, linewidth = 1.2) +
        scale_colour_manual(values = pal, 
                            breaks = c("Decoder", "NeuralODE"),   # legend top -> bottom
                            name = NULL) +
        guides(colour = guide_legend(nrow = 2, byrow = TRUE)) + 
        scale_x_continuous(
          trans  = scales::log2_trans(),          # base‑2 log transform
          breaks = scales::trans_breaks("log2", function(x) 2^x),
          labels = scales::trans_format("log2", scales::math_format(2^.x))
        ) +
        #scale_y_continuous(
          #limits = c(-1, 1),
          #oob    = scales::oob_squish )+         # puts anything < –0.5 right on –0.5
        coord_cartesian(ylim = c(-0.5, 1)) + 
        labs(x = "log(Context length)",
             y = "Skill index",
             title = "A. Scaling with context length") +
        base_theme
      
      ## 5.  Panel B – scaling in # observations -----------------------------
      p_nsamp <- ggplot(long_scaling,
                        aes(x = nSamplesTrain,
                            y = SkillSummary,
                            colour = Model)) +
        geom_point(alpha = .65, size = 1.5) +
        geom_smooth(se = FALSE, method = "loess", 
                    span = lowess_span, linewidth = 1.2) +
        scale_colour_manual(values = pal,
                            breaks = c("Decoder", "NeuralODE"),   # legend top -> bottom
                            name = NULL) +
        guides(colour = guide_legend(nrow = 2, byrow = TRUE)) + 
        labs(x = "Training-set size",
             y = NULL,
             title = "B. Scaling with number of observations") +
        base_theme +
        scale_x_continuous(
          trans  = scales::log10_trans(),          # base‑10 log transform
          breaks = scales::trans_breaks("log10", function(x) 10^x),
          labels = scales::trans_format("log10", scales::math_format(10^.x))
        ) +
        #scale_y_continuous(
          #limits = c(-1, 1),
          #oob    = scales::oob_squish )+         # puts anything < –0.5 right on –0.5
        coord_cartesian(ylim = c(-0.5, 1)) + 
        theme(axis.text.y = element_blank(),
              axis.ticks.y = element_blank())
      
      ## Panel C –   ------------------------------------------
      p_ <- ggplot(long_scaling,
                    aes(x = ModelDims, y = SkillSummary, colour = Model)) +
        geom_point(alpha = .65, size = 1.5) +
        geom_smooth(se = FALSE, 
                    method = "loess", 
                    span = lowess_span, linewidth = 1.2) +
        scale_colour_manual(values = pal,
                            limits = c("Decoder", "NeuralODE"),   # legend top → bottom
                            name   = NULL) +
        guides(colour = guide_legend(nrow = 2, byrow = TRUE)) +
        scale_x_continuous(
          trans  = scales::log10_trans(),          # base log transform
          breaks = scales::trans_breaks("log2", function(x) 2^x),
          labels = scales::trans_format("log2", scales::math_format(2^.x))
        ) +
        #scale_y_continuous(
          #limits = c(-1, 1),
          #oob    = scales::oob_squish ) +         # puts anything < –0.5 right on –0.5
        coord_cartesian(ylim = c(-0.5, 1)) + 
        labs(x = "",
             y = NULL,
             title = "C. Model Dimensions") +
        base_theme +
        theme(axis.text.y  = element_blank(),
              axis.ticks.y = element_blank())
      
      ## 6.  Assemble & write -------------------------------------------------
      scaling_plot <- (p_context | p_nsamp | p_) +  
                          plot_layout(guides = "collect")
      scaling_plot
      ggsave(sprintf("./Figures/Scaling_Decoder_vs_NeuralODE_%s.pdf",SimMode),
             scaling_plot, device = "pdf",
             width = 10, height = 5)
    }
    print("Done plotting comparision for skill results...") 
  }
  
  print("source the scaling results")
  ndm_source_extracted("ResultsAnalyze/SuperLModel_GenScalingResults.R")

  print("source the input stats")
  ndm_source_extracted("ResultsAnalyze/SuperLModel_GenFigs_GenInputStats.R")
  
  Decoder_ <- res_mat_[res_mat_$ModelType=="DecoderOnly",]
  NeuralODE <- res_mat_[res_mat_$ModelType=="NeuralODE",]
  
  print("Results evolution figs")
  { 
    # heat map analyses 
    GroupPolicyResultsBy_cols <- c("ContextLength","ModelDepth","nSamplesTrain")
    res_mat$GroupPolicyResultsBy <- apply(res_mat[,GroupPolicyResultsBy_cols],
                                          1, function(x){ paste(x,collapse = "_") })
    # table(res_mat$GroupPolicyResultsBy)
    
    # initialization analysis 
    res_mat_Start <- res_mat[res_mat$i_in_sgd %in% min(res_mat$i_in_sgd),]
    res_mat_End <- res_mat[res_mat$i_in_sgd %in% (res_mat$nSGD),]
    tapply(res_mat_End$SkillSummary, res_mat_End$ModelDepth, mean2)
    
    v_seq <- quantile(sort(unique(f2n(res_mat$i_in_sgd))),c(0,0.33,0.66,1))
    v_seq <- sapply(v_seq,function(x){sort(unique(f2n(res_mat$i_in_sgd)))[
                    which.min(abs(x-sort(unique(f2n(res_mat$i_in_sgd)))))]})
    v_ct_ <- 0; for(v_ in v_seq){ 
      v_ct_ <- v_ct_ + 1 
      pdf(sprintf("~/Dropbox/CovidSuperlearner/Figures/SkillAnimation_%s_%s.pdf",FiguresTag, v_ct_))
      par(mar=c(5,5,3,1))
      hist( ifelse(f2n(res_mat[j_<-res_mat$i_in_sgd==v_,]$SkillSummary)<0,0,
                   f2n(res_mat$SkillSummary)[j_]), xlim = c(0,1), 
            xlab = "Skill", ylab = "Frequency", cex.main = 2,cex.lab = 2, 
            breaks = seq(0,1,length.out=10),
            main = sprintf("Iteration %s of %s", v_, max(unique(f2n(res_mat$i_in_sgd)))))
      dev.off()
    }
      
    print2("---Baseline skill visualizations---")
    pdf(sprintf("./Figures/RSSBaseVSkillAtInitialization_%s.pdf",FiguresTag))
    {
    par(mar=c(5,5,1,1), mfrow = c(1,1))
    plot( res_mat_Start$RSSBaselineSummary,
          ytrans_ <- -log(abs(res_mat_Start$SkillSummary)), 
          log = "x", yaxt = "n",
          cex.lab = 1.5, 
          xlab = "RSS Baseline", 
          ylab = "Skill at Initialization")
    axis(side = 2,
         at = seq(summary(ytrans_)[1],
                  summary(ytrans_)[6], length.out = 4), 
         labels = round(
                  seq(summary(res_mat$SkillSummary)[1],
                      summary(res_mat$SkillSummary)[6], 
                      length.out = 4), 2))
    }
    dev.off() 
    
    pdf(sprintf("./Figures/RSSBaseline_c1_%s.pdf",FiguresTag))
    {
    par(mfrow=c(2,2),mar=c(2,2,4,1))
    theLims <- c(0,1.5*max(eval(parse(text = 
          sprintf("res_mat_Start$RSSBaselineTime%s",
                  nTimesLookValidationInferenceTrainedTo))), na.rm=T ))
    for(f_ in 4:1){ 
    hist(eval(parse(text = 
                      sprintf("res_mat_Start$RSSBaselineTime%s",
                              wk_ <- round(nTimesLookValidationInferenceTrainedTo/f_) ))), 
         xlim= theLims,  
         main = sprintf("RSS, Time %s",wk_), xlab = "", cex.main = 1.5, 
         breaks = seq(0,theLims[2],length.out = 40))
    }
    }
    dev.off()
    
    if(SimMode & T == F){ 
    for(GroupPolicyByThis_ in unique(res_mat$GroupPolicyResultsBy)){
      for(ty_ in c("BaselineRSS", "Skill")){
      res_mat_ <- res_mat[res_mat$i_in_sgd %in% max(res_mat$i_in_sgd) & 
                           res_mat$GroupPolicyResultsBy %in% GroupPolicyByThis_,]
      if(ty_ == "Skill"){ theOutcome <- "SkillSummary" } 
      if(ty_ == "BaselineRSS"){ theOutcome <- "RSSBaselineSummary" } 
      aggregated_data_median <- res_mat_ %>% group_by(c_endogeneous, policy_effectiveness) %>% 
        summarise(Ave = median(eval(parse(text=theOutcome))))
      aggregated_data_median <- reshape2::acast(aggregated_data_median, c_endogeneous ~ policy_effectiveness, value.var="Ave")
      
      aggregated_data_mean <- res_mat_ %>% group_by(c_endogeneous, policy_effectiveness) %>% summarise(Ave = mean(eval(parse(text=theOutcome))))
      aggregated_data_mean <- reshape2::acast(aggregated_data_mean, c_endogeneous ~ policy_effectiveness, value.var="Ave")
      
      aggregated_data_mean_clip <- res_mat_ %>% group_by(c_endogeneous, policy_effectiveness) %>% summarise(Ave = clippedMean(eval(parse(text=theOutcome))))
      aggregated_data_mean_clip <- reshape2::acast(aggregated_data_mean_clip, c_endogeneous ~ policy_effectiveness, value.var="Ave")
      
      aggregated_data_var <- res_mat_ %>% group_by(c_endogeneous, policy_effectiveness) %>% summarise(Ave = var(eval(parse(text=theOutcome))))
      aggregated_data_var <- reshape2::acast(aggregated_data_var, c_endogeneous ~ policy_effectiveness, value.var="Ave")
  
      aggregated_data_n <- res_mat_ %>% group_by(c_endogeneous, policy_effectiveness) %>% summarise(Ave = length(eval(parse(text=theOutcome))))
      aggregated_data_n <- reshape2::acast(aggregated_data_n, c_endogeneous ~ policy_effectiveness, value.var="Ave")
      sum(aggregated_data_n, na.rm=T)
      
      aggregated_data_se <- sqrt( aggregated_data_var / aggregated_data_n)
      aggregated_data_lb <- aggregated_data_mean - sqrt(1/aggregated_data_n * aggregated_data_var)
  
      RowKey <- c("Endo."); ColKey <- c("P. Effect")
      ProcessAggMatFxn <- function(m){
        m[] <- helpeRs::fixZeroEndings(round(m,2))
        row.names(m) <- helpeRs::fixZeroEndings(round(f2n(row.names(m)),3),roundAt=3)
        colnames(m) <- helpeRs::fixZeroEndings(round(f2n(colnames(m)),3),roundAt=3)
        row.names(m)[1] <- paste(sprintf("%s$=$",RowKey), rownames(m)[1]); 
        colnames(m)[1] <- paste(sprintf("%s$=$",ColKey), colnames(m)[1])
        return(m)
      }
      GroupPolicyByThisText_ <- paste(sapply(1:length(GroupPolicyResultsBy_cols),
                                       function(s_){ 
                                       tmp_ <- res_mat[1,GroupPolicyResultsBy_cols][s_]
                                       paste0(names(tmp_)[1],"=", 
                                              format(tmp_[1], scientific = FALSE))
                                       }),collapse=", ")
      write(file = sprintf("%s/Heat%s_%s.tex", SaveFolder, ty_, FiguresTag),
            stargazer::stargazer(ProcessAggMatFxn(aggregated_data_mean), type = "latex", 
                                   label = sprintf("tab:Heat%s", ty_), 
                                   title = sprintf("%s Rows denote endogeneity levels; columns denote policy effectiveness levels. 
                                                   %s. %s observations per grid cell.",
                                              ifelse(ty_ == "BaselineRSS", 
                                              yes = "Baseline RSS analysis. ", 
                                              no = "Skill analysis."), 
                                              GroupPolicyByThisText_,
                                              theN <- median(aggregated_data_n) )))
      write(file = sprintf("%s/Heat%sVar_%s.tex", SaveFolder, ty_, FiguresTag),
            stargazer::stargazer(ProcessAggMatFxn(aggregated_data_se), type = "latex", 
                                 label = sprintf("tab:Heat%sSE", ty_), 
                                 title = sprintf("%s Rows denote endogeneity levels; columns denote policy effectiveness levels. 
                                                   %s. %s observations per grid cell.",
                                                 ifelse(ty_ == "BaselineRSS", 
                                                        yes = "Baseline RSS analysis, standard error. ", 
                                                        no = "Skill analysis, standard error."), 
                                                 GroupPolicyByThisText_,
                                                 theN  )))
    }
    }
    }
  }
  
  stop("XXX")
  {
    print("--- XXX12344321XXX: Generating 3 Powerful Insight Figures ---")
    {
      library(ggplot2); library(dplyr)
      
      # 0. Setup: Filter for the final converged state
      # We rely on paired_df created in the previous block; if missing, we use res_mat
      if(exists("paired_df")){
        df_test <- paired_df[paired_df$i_in_sgd_Decoder == max(paired_df$i_in_sgd_Decoder), ]
      } else {
        # Fallback to creating a simple paired view from res_mat if paired_df is absent
        df_test <- res_mat[res_mat$i_in_sgd == max(res_mat$i_in_sgd), ] %>%
          filter(ModelType %in% c("DecoderOnly", "NeuralODE")) %>%
          tidyr::pivot_wider(names_from = ModelType, values_from = c(SkillSummary, RSSBaselineSummary)) %>%
          rename(SkillSummary_Decoder = SkillSummary_DecoderOnly, 
                 SkillSummary_NeuralODE = SkillSummary_NeuralODE,
                 RSSBaselineSummary_Decoder = RSSBaselineSummary_DecoderOnly)
        names(df_test) <- paste0(names(df_test), "_Decoder") # Simplified suffixing for fallback
      }
      
      # -------------------------------------------------------------------------
      # FIG 1: MECHANISTIC SCALING LAWS (The Efficiency Test)
      # Hypothesis: Hybrid has a steeper power-law exponent than Decoder.
      # Transform: g(S) = -log(1 - Skill + epsilon) as per Paper Methods.
      # -------------------------------------------------------------------------
      df_test$g_Skill_Dec  <- -log(1 - df_test$SkillSummary_Decoder + 1e-3)
      df_test$g_Skill_Node <- -log(1 - df_test$SkillSummary_NeuralODE + 1e-3)
      #df_test$g_Skill_Dec  <- (df_test$SkillSummary_Decoder + 1e-3)
      #df_test$g_Skill_Node <- (df_test$SkillSummary_NeuralODE + 1e-3)
      #df_test$g_Skill_Dec[df_test$g_Skill_Dec<0]<-0
      
      library(scales) # Required for trans_format and math_format
      p1 <- {ggplot(df_test, aes(x = nSamplesTrain_Decoder)) +
        geom_smooth(aes(y = g_Skill_Node, color = "Hybrid (NeuralODE)", fill = "Hybrid (NeuralODE)"), 
                    method = "lm", formula = y ~ log(x), alpha = 0.15) +
        geom_smooth(aes(y = g_Skill_Dec, color = "Decoder-Only", fill = "Decoder-Only"), 
                    method = "lm", formula = y ~ log(x), alpha = 0.15) +
        
        # 1. Scientific notation for x-axis (10^x)
        scale_x_log10(
          labels = scales::trans_format("log10", scales::math_format(10^.x))
        ) +
        
        # 2. Unify legend: 'name' must be identical for both scales to merge them
        scale_color_manual(
          name = NULL, 
          values = c("Hybrid (NeuralODE)" = "#e41a1c", "Decoder-Only" = "#377eb8")
        ) +
        scale_fill_manual(
          name = NULL, 
          values = c("Hybrid (NeuralODE)" = "#e41a1c", "Decoder-Only" = "#377eb8")
        ) +
        labs(title = "Data Efficiency",
             y = "Stabilized Skill -log(1-S)", 
             x = "Training Samples") + # "log10" removed from label as the ticks now show it explicitly
        
        theme_minimal(base_size = 14, base_line_size = 1) + 
        theme(legend.position = "bottom",panel.grid = element_blank() )}
      p1
      ggsave(sprintf("%s/MarginalDataScaling_%s.pdf", SaveFolder, FiguresTag), p1, width = 7, height = 6)
      
      # -------------------------------------------------------------------------
      # FIG 2: THE ENDOGENEITY PARADOX (The Robustness Test)
      # Hypothesis: Stronger feedback (Endogeneity) lowers Baseline RSS (task gets 'easier'),
      # but Decoder skill stagnates relative to Hybrid (Gap grows/sustains).
      # -------------------------------------------------------------------------
      if("c_endogeneous_Decoder" %in% names(df_test)){
        # Aggregate for clear trend lines
        agg_endo <- df_test %>% group_by(c_endogeneous_Decoder) %>%
          summarise(
            BaseRSS = median(RSSBaselineSummary_Decoder, na.rm=T),
            SkillGap = median(SkillSummary_NeuralODE - SkillSummary_Decoder, na.rm=T)
          )
        
        # Scale factor for dual axis
        coeff <- max(agg_endo$SkillGap) / max(agg_endo$BaseRSS)
        
        p2 <- ggplot(agg_endo, aes(x = c_endogeneous_Decoder)) +
          geom_line(aes(y = BaseRSS * coeff, linetype = "Baseline Difficulty (RSS)"), color = "gray50", size = 1.2) +
          geom_line(aes(y = SkillGap, color = "Hybrid Advantage (Gap)"), size = 1.5) +
          scale_y_continuous(name = "Hybrid Skill Advantage (Gap)",
                             sec.axis = sec_axis(~./coeff, name = "Baseline RSS (Lower = 'Easier')")) +
          scale_color_manual(values = c("Hybrid Advantage (Gap)" = "#e41a1c")) +
          labs(title = "2. The Endogeneity Paradox",
               subtitle = "As feedback strengthens (x), the task becomes 'easier' (RSS drops),\nyet the Hybrid's advantage persists or grows.",
               x = "Endogeneity Strength (Feedback)") +
          theme_minimal(base_size = 14) + theme(legend.position = "bottom")
        
        ggsave(sprintf("%s/Insight2_Endogeneity_%s.pdf", SaveFolder, FiguresTag), p2, width = 7, height = 6)
      }
      
      # -------------------------------------------------------------------------
      # FIG 3: REGIME ROBUSTNESS MAP (The Guardrails Test)
      # Hypothesis: Hybrid wins most consistently in the "Danger Zone": 
      # High Endogeneity (Strong Feedback) + Sparse Data.
      # -------------------------------------------------------------------------
      if("c_endogeneous_Decoder" %in% names(df_test)){
        df_test$Win <- df_test$SkillSummary_NeuralODE > df_test$SkillSummary_Decoder
        
        # Bin if continuous, or use raw if grid
        agg_map <- df_test %>% 
          group_by(c_endogeneous_Decoder, nSamplesTrain_Decoder) %>%
          summarise(WinProb = mean(Win, na.rm=T), .groups = 'drop')
        
        p3 <- ggplot(agg_map, aes(x = factor(nSamplesTrain_Decoder), y = factor(c_endogeneous_Decoder), fill = WinProb)) +
          geom_tile(color = "white") +
          scale_fill_gradient2(low = "#377eb8", mid = "white", high = "#e41a1c", midpoint = 0.5, 
                               limits = c(0,1), name = "P(Hybrid > Dec)") +
          labs(title = "3. Regime Robustness Map",
               subtitle = "Red = Hybrid Wins. The Hybrid dominates in the\nHigh-Feedback / Low-Data corner (Top Left).",
               x = "Training Samples", y = "Endogeneity Strength") +
          theme_minimal(base_size = 14)
        
        ggsave(sprintf("%s/Insight3_RegimeMap_%s.pdf", SaveFolder, FiguresTag), p3, width = 8, height = 6)
      }
      
      print("--- Finished generating 3 Insight Figures ---")
    }
  }

  print("Making heat maps")
  stop("XXX")
  {
    MakeHeatMaps <- FALSE
    ModelsUseInLms <- c("NeuralODE","DecoderOnly")
    #ModelsUseInLms <- c("NeuralODE")
    #for(TypeMultivariateModel in c("lm","randomForest")){
    for(TypeMultivariateModel in c("lm")){
      print2(sprintf("Starting multivariate analysis [type: %s]...",TypeMultivariateModel))
      if(SimMode){
        if(TypeMultivariateModel == "randomForest"){ 
          res_mat$SkillSummary[res_mat$SkillSummary < -100] <- -100
          TrainedModel_RSS <- randomForest::randomForest(
            y = res_mat$RSSBaselineSummary_inv,
            x = cols2numeric(res_mat[,theCovars_ <- c(
                                          "ContextLength_log",
                                          "ModelDepth_log",
                                          #"nSamplesTrain_log",
                                          "c_endogeneous",
                                          "policy_decay",
                                          "policy_effectiveness",
                                          "policy_responsiveness",
                                          "invbetat_sd",
                                          "gamma",
                                          "sigma",
                                          "xi",
                                          "i0_a","i0_b",
                                        "i_in_sgd")]),
            ntrees = 10000)
          TrainedModel_Skill <- randomForest::randomForest(
            y = res_mat$SkillSummary,
            x = cols2numeric(res_mat[,c(theCovars_,"RSSBaselineTime1")]),
            ntrees = 10000, 
            localImp = T, importance = T)
          TrainedModel_Skill$importance[order(TrainedModel_Skill$importance[,1],decreasing=T),]
        }
        if(TypeMultivariateModel == "lm"){ 
        # res_mat$policy_effectiveness
        ModelTrained_RSS <- lm(RSSBaselineSummary_inv~f2n(c_endogeneous)+
                                # policy factors 
                                f2n(policy_effectiveness) + 
                                f2n(policy_decay) + 
                                f2n(policy_effectiveness) * f2n(c_endogeneous) + 
                                f2n(policy_responsiveness) + 
                                 
                                # ODE factors 
                                f2n(invbetat_sd) + 
                                f2n(gamma)+
                                f2n(sigma)+
                                f2n(xi) +
                                f2n(i0_a) + 
                                #f2n(i0_a)*f2n(i0_b) + 
                                f2n(i0_b) + 
                                f2n(r0_a)+ 
                                f2n(r0_b),
                               data = res_mat[res_mat$i_in_sgd == min(res_mat$i_in_sgd) & 
                                               res_mat$ModelType == ModelsUseInLms[1], ]
                               )
        summary( ModelTrained_RSS)
        
        for(re_ in c("all", "StartOfTraining", "AtEpoch1", "EndOfTraining")){ 
          i_in_sgd_text <- ifelse(re_ == "all", yes = "*I(f2n(atEpoch_log))", no = "")
          GrabModelCond <- res_mat$ModelType %in% ModelsUseInLms
          if(re_ == "all"){ res_mat_use <- res_mat }
          if(re_ == "StartOfTraining"){ 
            res_mat_use <- res_mat[res_mat$i_in_sgd == min(res_mat$i_in_sgd) & GrabModelCond,] }
          if(re_ == "AtEpoch1"){ 
            res_mat_use <- res_mat[ res_mat$atEpoch >= 1 & res_mat$atEpoch < 2 & GrabModelCond,] }
          if(re_ == "EndOfTraining"){ 
            res_mat_use <- res_mat[res_mat$i_in_sgd == max(res_mat$i_in_sgd) & GrabModelCond,] }
          TrainedModel_Skill <- lm(eval(parse(text=sprintf("as.formula(SkillSummary_inv ~
                                     # causal parameters of interest 
                                     f2n(c_endogeneous) +
                                     f2n(policy_effectiveness) + 
                                     f2n(policy_decay) + 
                                     f2n(policy_responsiveness) + 
                                     f2n(policy_effectiveness)*f2n(c_endogeneous)%s + 
                                     
                                     # training & model parameters 
                                     f2n(nSamplesTrain_log)+
                                     f2n(nParamsModel_log)+
                                     f2n(ContextLength_log)+
                                     #I((f2n(nSamplesTrain_log)))*I((f2n(nParamsModel_log)))%s +
                                     #f2n(ContextLength_log)*(I(f2n(nParamsModel_log)))%s + 
                                     
                                     ## #I(f2n(i_in_sgd_log)) + 
                                     
                                     # other factors to marginalize over
                                     f2n(invbetat_sd) + 
                                     f2n(gamma) +
                                     f2n(sigma) +
                                     f2n(xi) +
                                     f2n(i0_a) + f2n(i0_b) + 
                                     f2n(i0_a) * f2n(i0_b) + 
                                     f2n(r0_a) + f2n(r0_b)
                                     )",i_in_sgd_text,i_in_sgd_text,i_in_sgd_text)
                                     )), data = res_mat_use)
          if(re_ == "all"){ TrainedModel_Skill_all <- TrainedModel_Skill}
          if(re_ == "StartOfTraining"){ TrainedModel_Skill_StartOfTraining <- TrainedModel_Skill}
          if(re_ == "AtEpoch1"){ TrainedModel_Skill_AtEpoch1 <- TrainedModel_Skill}
          if(re_ == "EndOfTraining"){ TrainedModel_Skill_EndOfTraining <- TrainedModel_Skill}
          rm(TrainedModel_Skill)
        }
        }
      }
      if(!SimMode){
        res_mat$nCovars <- unlist(lapply(strsplit(res_mat$dataInputs,split="__"),length))
        if(TypeMultivariateModel == "randomForest"){ 
          res_mat$SkillSummary[res_mat$SkillSummary < -100] <- -100
          TrainedModel_RSS <- randomForest::randomForest(
                                      y = res_mat$RSSBaselineSummary_inv,
                                      x = cols2numeric(res_mat[,c("ContextLength_log","ModelDepth_log",
                                                                  "nSamplesTrain_log","i_in_sgd_log",
                                                                  "evaluationTime")]),
                                      ntrees = 10000)
          CovarMat_ <- sapply(unique(unlist(strsplit(res_mat$dataInputs,split = "__"))), function(x){
                  lapply(strsplit(res_mat$dataInputs,split = "__"), function(y){
                    1*(x %in% y) }) })
          res_mat <- cbind(res_mat,CovarMat_)
          TrainedModel_Skill <- randomForest::randomForest(
                                     y = res_mat$SkillSummary_inv,
                                     x = cols2numeric(res_mat[,c("ContextLength_log",
                                                                 "ModelDepth_log",
                                                                 "nSamplesTrain",
                                                                 "RSSBaselineTime1",
                                                                 "atEpoch_log",
                                                                 #"i_in_sgd_log",
                                                                 colnames(CovarMat_),# "nCovars",
                                                                 "evaluationTime")]),
                                     ntrees = 10000, 
                                     localImp = T, importance = T)
          TrainedModel_Skill$importance[order(TrainedModel_Skill$importance[,1],decreasing=T),]
        }
        if(TypeMultivariateModel == "lm"){
          ModelTrained_RSS <- lm(RSSBaselineSummary_inv~f2n(maxInSampleTime_id), 
                                 data = res_mat)
          png( sprintf("%s/RealRSS_ByTime_%s.png",SaveFolder, FiguresTag) )
          {
          par(mfrow=c(1,1),mar=c(5,5,1,1))
          y_ <- res_mat_orig_WithSkills$RSSBaselineTime8 + res_mat_orig_WithSkills$RSSBaselineTime9 + res_mat_orig_WithSkills$RSSBaselineTime10+0.01
          colPal <- tapply(y_[res_mat_orig_WithSkills$evaluationTime==1], 
                           res_mat_orig_WithSkills$location_id[res_mat_orig_WithSkills$evaluationTime==1], 
                           function(z){mean(z,na.rm=T)})
          colPal_ordered <- viridis::magma(n=length(colPal),alpha = 0.9, begin = 0.15, end = 0.85)
          names(colPal_ordered) <- names(colPal)[order(colPal)]
          plot( x_ <- res_mat_orig_WithSkills$time_anchor_id,
                y_, yaxt = "n", 
                col = colPal_ordered[as.character(res_mat_orig_WithSkills$location_id)],
                xlab = "Evaluation Time", ylab = "RSS", pch = 19, cex.axis = 1.5, 
                log = "y",cex=0.5,cex.lab = 1.5); axis(2,at = c(0.01, 0.1, 1, 10), 
                                                       labels = c(0.01, 0.1, 1, 10))
          }
          dev.off()
          pdf( sprintf("%s/RealRSS_ByPlace_%s.pdf",SaveFolder, FiguresTag) )
          {
            par(mfrow=c(1,1),mar=c(5,5,1,1))
            colPal <- tapply(y_[res_mat_orig_WithSkills$evaluationTime==1], 
                             res_mat_orig_WithSkills$location_id[res_mat_orig_WithSkills$evaluationTime==1], 
                             function(z){sd(z,na.rm=T)})
            colPal_ordered <- viridis::magma(n=length(colPal),alpha = 0.9, begin = 0.15, end = 0.85)
            names(colPal_ordered) <- names(colPal)[order(colPal)]
            x_ <- tapply(res_mat_orig_WithSkills$location_id,
                         res_mat_orig_WithSkills$location_id, unique)
            y_ <- tapply(res_mat_orig_WithSkills$RSSBaselineTime8 + res_mat_orig_WithSkills$RSSBaselineTime9 + res_mat_orig_WithSkills$RSSBaselineTime10+0.01,
                         res_mat_orig_WithSkills$location_id, sd)
            #load("./Data/COVID_LEARNER_REALDAT_RESTART.rdata")
            #mean(res_mat_orig_WithSkills$location_id %in% predicted_df$location_id  ) 
            #PlaceNames_ <- predicted_df$location_id
            #names(PlaceNames_) <- predicted_df$location_id
            #PlaceNames_ <- PlaceNames_[as.character(names(y_))]
            #cbind(y_[order(y_)],PlaceNames_[order(y_)])
            plot( #x_[order(y_)],
                  0.001+y_[order(y_)],
                  yaxt = "n", 
                  # pch = as.numeric(as.factor(res_mat_orig_WithSkills$location_id)),
                  #col = colPal_ordered[as.character(x_[order(y_)])],
                  xlab = "Locations, Sorted", ylab = "sd(RSS)", pch = 19, 
                  cex.axis = 1.5, 
                  log = "y",cex=0.9,cex.lab = 1.5); 
            axis(2,at = c(0.01, 0.1, 1, 10), labels = c(0.01, 0.1, 1, 10))
          }
          dev.off()
          for(re_ in c("all","StartOfTraining","AtEpoch1", "EndOfTraining")){ 
            if(re_ == "all"){ res_mat_use <- res_mat }
            if(re_ == "StartOfTraining"){ 
              res_mat_use <- res_mat[res_mat$i_in_sgd == min(res_mat$i_in_sgd),] }
            if(re_ == "AtEpoch1"){ 
              res_mat_use <- res_mat[ res_mat$atEpoch >= 1 & res_mat$atEpoch < 2,] }
            if(re_ == "EndOfTraining"){ 
              res_mat_use <- res_mat[res_mat$i_in_sgd == max(res_mat$i_in_sgd),] }
            i_in_sgd_text <- ifelse(re_ == "all", yes = "*I(f2n(i_in_sgd_log))", no = "")
            #i_in_sgd_text <- ifelse(re_ == "all", yes = "*I(f2n(atEpoch_log))", no = "")
            TrainedModel_Skill <- lm(eval(parse(text=sprintf("as.formula(SkillSummary_inv ~   
                                     # training & model parameters 
                                     f2n(ContextLength_log)*(I(f2n(nParamsModel_log)))%s + 
                                     I(f2n(i_in_sgd_log)) + 
                                     I((f2n(nSamplesTrain_log)))*I((f2n(nParamsModel_log)))%s +
                                     f2n(nCovars)*(I(f2n(nParamsModel_log)))%s + 
                                     
                                     # other factors to marginalize over
                                     f2n(evaluationTime))",i_in_sgd_text,i_in_sgd_text,i_in_sgd_text))),
                                    data = res_mat_use)
            if(re_ == "all"){ TrainedModel_Skill_all <- TrainedModel_Skill}
            if(re_ == "StartOfTraining"){ TrainedModel_Skill_StartOfTraining <- TrainedModel_Skill}
            if(re_ == "AtEpoch1"){ TrainedModel_Skill_AtEpoch1 <- TrainedModel_Skill}
            if(re_ == "EndOfTraining"){ TrainedModel_Skill_EndOfTraining <- TrainedModel_Skill}
          }
        }
      }
      
      print2(sprintf("Starting MakeHeatMaps in multivariate analysis [type: %s]...",TypeMultivariateModel))
      if(MakeHeatMaps <- FALSE){ 
      options(scipen = 10)
      MakeHeatMap(  factor1 = c("log(Context length)" = "ContextLength_log"),
                    factor2 = c("log(# of Parameters)" = "nParamsModel_log"),
                    outcome = "SkillSummary_inv",
                    pdf_path = sprintf("./Figures/Heat_ContextVParams_%s_%s.pdf",FiguresTag,TypeMultivariateModel), 
                    useLog = "", 
                    dat = res_mat[res_mat$i_in_sgd==res_mat$nSGD,],
                    lm_obj =  TrainedModel_Skill_EndOfTraining,
                    extrap_factor1 = 1, extrap_factor2 = 1, 
                    OutcomeTransformFxn = Map2ConstrainedSkill, 
                    openBrowser = F )
      MakeHeatMap(  factor1 = c("log(Training Samples)" = "nSamplesTrain_log"),
                    factor2 = c("log(Context length)" = "ContextLength_log"),
                    outcome = "SkillSummary_inv",
                    pdf_path = sprintf("./Figures/Heat_DataVsContext_%s_%s.pdf",FiguresTag,TypeMultivariateModel), 
                    useLog = "", 
                    dat = res_mat[res_mat$i_in_sgd==res_mat$nSGD,],
                    #lm_obj =  TrainedModel_Skill_EndOfTraining,
                    lm_obj =  TrainedModel_Skill_AtEpoch1,
                    extrap_factor1 = 1, extrap_factor2 = 1, 
                    OutcomeTransformFxn = Map2ConstrainedSkill, 
                    openBrowser = F )
      MakeHeatMap(  factor1 = c("log(Iteration #)" = "i_in_sgd_log"),
                    factor2 = c("log(# of Parameters)" = "nParamsModel_log"),
                    outcome = "SkillSummary_inv",
                    pdf_path = sprintf("./Figures/Heat_IterNumbVDepth_%s_%s.pdf",FiguresTag,TypeMultivariateModel), 
                    dat = res_mat,
                    lm_obj =  TrainedModel_Skill_all,
                    extrap_factor1 = 1, extrap_factor2 = 1, OUTCOME_SCALER = 1, 
                    OutcomeTransformFxn = Map2ConstrainedSkill )
      MakeHeatMap(  factor1 = c("log(Iteration #)" = "i_in_sgd_log"),
                    factor2 = c("log(Context length)" = "ContextLength_log"),
                    outcome = "SkillSummary_inv",
                    pdf_path = sprintf("./Figures/Heat_IterNumbVContext_%s_%s.pdf",FiguresTag,TypeMultivariateModel), 
                    dat = res_mat,
                    lm_obj =  TrainedModel_Skill_all,
                    extrap_factor1 = 1, extrap_factor2 = 1, OUTCOME_SCALER = 1, 
                    OutcomeTransformFxn = Map2ConstrainedSkill )
      MakeHeatMap(  factor1 = c("log(Training Samples)" = "nSamplesTrain_log"),
                    factor2 = c("log(Parameter #)" = "nParamsModel_log"),
                    outcome = "SkillSummary_inv",
                    pdf_path = sprintf("./Figures/Heat_SamplesVContext_%s_%s.pdf",FiguresTag,TypeMultivariateModel), 
                    useLog = "x", 
                    dat = res_mat[f2n(res_mat$i_in_sgd)==(f2n(res_mat$nSGD)),],
                    lm_obj =  TrainedModel_Skill_EndOfTraining,
                    extrap_factor1 = 1, extrap_factor2 = 1, OUTCOME_SCALER = 1, 
                    OutcomeTransformFxn = Map2ConstrainedSkill,
                    openBrowser = F)
      plot(res_mat$RSSBaselineSummary_inv,res_mat$RSSBaselineSummary)
      plot(res_mat$c_endogeneous,res_mat$RSSBaselineSummary_inv)
      plot(res_mat$policy_effectiveness,res_mat$RSSBaselineSummary_inv)
      hist(res_mat$RSSBaselineSummary_inv)
      hist(res_mat$RSSBaselineSummary)
      #table(res_mat[res_mat$i_in_sgd==res_mat$nSGD,]$nSamplesTrain)
      if(SimMode){ 
        MakeHeatMap(  factor1 = c("Endogeneity" = "c_endogeneous"),
                      factor2 = c("Policy effectiveness" = "policy_effectiveness"),
                      outcome = "SkillSummary_inv",
                      pdf_path = sprintf("./Figures/Heat_EndogVPolicyEffectiveness_%s_%s.pdf",FiguresTag,TypeMultivariateModel), 
                      dat = res_mat[res_mat$i_in_sgd==res_mat$nSGD,],
                      lm_obj =  TrainedModel_Skill_EndOfTraining,
                      extrap_factor1 = 1, extrap_factor2 = 1, 
                      OutcomeTransformFxn = Map2ConstrainedSkill)
        MakeHeatMap(  factor1 = c("Endogeneity" = "c_endogeneous"),
                      factor2 = c("Policy effectiveness" = "policy_effectiveness"),
                      outcome = "SkillSummary_inv",
                      pdf_path = sprintf("./Figures/Heat_BaseRSS_EndogVPolicyEffectiveness_%s_%s.pdf",FiguresTag,TypeMultivariateModel), 
                      dat = res_mat,
                      lm_obj =  ModelTrained_RSS,
                      extrap_factor1 = 1, extrap_factor2 = 1, 
                      OutcomeTransformFxn = Map2ConstrainedSkill)
        MakeHeatMap(  factor1 = c("Endogeneity" = "c_endogeneous"),
                      factor2 = c("Policy effectiveness" = "policy_effectiveness"),
                      outcome = "RSSBaselineSummary_inv",
                      pdf_path = sprintf("./Figures/Heat_BaseRSS_EndogVPolicyEffectiveness_%s_%s.pdf",FiguresTag,TypeMultivariateModel), 
                      dat = res_mat,
                      lm_obj =  ModelTrained_RSS,
                      extrap_factor1 = 1, extrap_factor2 = 1, 
                      OutcomeTransformFxn = Map2ConstrainedBaselineRSS)
        print2("MakeHeatMap: Pop Mean vs. Variability...")
        MakeHeatMap(  factor1 = c("Population Mean, Initial infections" = "i0_a"),
                      factor2 = c("Population Variability, Initial infections" = "i0_b"),
                      outcome = "RSSBaselineSummary_inv",
                      pdf_path = sprintf("./Figures/Heat_BaseRSS_InitialCond_%s_%s.pdf",FiguresTag,TypeMultivariateModel), 
                      dat = res_mat,
                      lm_obj =  ModelTrained_RSS,
                      extrap_factor1 = 1, extrap_factor2 = 1, OutcomeTransformFxn = Map2ConstrainedBaselineRSS)
        print2("MakeHeatMap: Endogeneity vs. Effectiveness...")
        MakeHeatMap(  factor1 = c("Endogeneity" = "c_endogeneous"),
                      factor2 = c("Policy effectiveness" = "policy_effectiveness"),
                      outcome = "RSSBaselineSummary_inv",
                      pdf_path = sprintf("./Figures/Heat_BaseRSS_EndogVPolicyEffectiveness_%s_%s.pdf",FiguresTag,TypeMultivariateModel), 
                      dat = res_mat,
                      lm_obj =  ModelTrained_RSS,
                      extrap_factor1 = 1, extrap_factor2 = 1,
                      OutcomeTransformFxn = Map2ConstrainedBaselineRSS)
      }
      if(!SimMode){
        print2("MakeHeatMap: Iters vs. Input Covars...")
        MakeHeatMap(  factor1 = c("log(Iteration #)" = "i_in_sgd_log"),
                      factor2 = c("Input Covariates" = "nCovars"),
                      outcome = "SkillSummary_inv",
                      pdf_path = sprintf("./Figures/Heat_Test_EndogVPolicyEffectiveness_%s_%s.pdf",FiguresTag,TypeMultivariateModel), 
                      dat = res_mat,
                      lm_obj =  TrainedModel_Skill_all,
                      extrap_factor1 = 1, extrap_factor2 = 1, 
                      OutcomeTransformFxn = Map2ConstrainedSkill)
      }
      }
    
      name_conversion_matrix <- {matrix(c(
        "StartEmphDisease..ParametersEndEmph","StartEmphDisease ParametersEndEmph",
        "c_endogeneous","Endogeneity", 
        "policy_effectiveness","Policy effectiveness", 
        "policy_responsiveness","Policy responsiveness", 
        "invbetat_sd","Beta, random walk SD", 
        "gamma","Gamma",
        "sigma","Sigma", 
        "xi","Xi", 
        "i0_a","Average initial proportion infected", 
        "i0_b","SD(initial proportion infected)", 
        "r0_a","Average initial proportion recovered", 
        "r0_b","SD(initial proportion recovered)",
        
        "space1","space1",
        #"Disease..parameters..with..interactions","Disease parameters with interactions",
        #"Modeling..parameters..with..interactions","Modeling parameters with interactions",
        "StartEmphModeling..ParametersEndEmph","StartEmphModeling ParametersEndEmph",
        "nParamsModel_log","log(# of parameters)",
        "nSamplesTrain_log","log(# of data samples)", 
        #"i_in_sgd_log","log(Iteration)", 
        "ContextLength_log","log(Context Length)", 
        #"f2n(ContextLength_log)\\:I(f2n(nParamsModel_log))","log(Context) times log(# of parameters)",
        #"atEpoch_log","log(At epoch)", 
        
        "space2","space2",
        "Other statistics","Other statistics",
        "Countries","Countries",
        # "Observations","Observations",
        "Adjusted R-squared","Adjusted R-squared",
        "AIC","AIC",
        "Weak instruments","Weak instruments",
        "Wu-Hausman","Wu-Hausman"
      ), byrow = T, ncol = 2)}
      
      stop("XXX")
      if(TypeMultivariateModel == "lm"){ 
      print2("Tables2Tex: Main Table...")
      helpeRs::Tables2Tex(
        reg_list = list( ModelTrained_RSS, 
                         #TrainedModel_Skill_StartOfTraining,
                         TrainedModel_Skill_EndOfTraining
                         #TrainedModel_Skill_AtEpoch1
                         ),
        model.names = c("Baseline RSS", 
                        #"Skill, At Initialization",
                        "Skill, Epoch 1"), # include only interaction
        checkmark_list = list("Disease..parameters..with..interactions" = c(1,1),
                              "Modeling..parameters..with..interactions" = c(1,1)),
        addrow_list = list( 
                          "StartEmphDisease..ParametersEndEmph" = rep("",times = 2),
                          "StartEmphModeling..ParametersEndEmph" = rep("",times = 2)
                        ),
        clust_id = NULL,
        superunit_covariateName =  NULL, 
        superunit_label =  "",
        seType = "analytical",
        saveFull = F, DoFullTableKey = F,
        NameConversionMat = name_conversion_matrix,
        saveFolder = SaveFolder,
        nameTag = sprintf("Baselines_%s_%s", FiguresTag, TypeMultivariateModel), 
        tabCaption = "Estimator: OLS. 
                      $t$-values in parentheses, heteroskedasticity-consistent standard errors used. 
                      $*$ indicates $p<$0.05." )
      }
    }
  
    y_summary_mat <- tapply(1:nrow(res_mat),res_mat$i_in_sgd, function(i_){
                       apply(res_mat[i_,paste("SkillTime",1:nTimesLookValidationInference,sep = '')], 2, clippedMean  ) })
    names(y_summary_mat)
    grayCols <- gray.colors(length(y_summary_mat), start = 0.2, end = 0.9)
    plot(y_summary_mat[[1]], pch = "1", ylim = c(-2,1),col = grayCols[4], cex = 2)
    for(jr in 2:length(y_summary_mat)){
       text(1:length(y_summary_mat[[jr]]), y_summary_mat[[jr]], labels = as.character(jr), col = grayCols[3], cex = 1)
    }
  
    # four corners analysis 
    res_mat_MaxIters <- res_mat[res_mat$i_in_sgd==max(res_mat$i_in_sgd),]
    if(SimMode){ for(j_ in c(0:4)){
      if(j_ == 0){ 
        y_summary0 <- y_summary <- apply(  res_mat_MaxIters[,paste("SkillTime",1:nTimesLookValidationInference,sep = '')], 2,  clippedMean  )
        tag0_ <- tag_ <- "All"
      }
      if(j_ == 1){ 
        y_summary1 <- y_summary <- apply(  res_mat_MaxIters[res_mat_MaxIters$c_endogeneous==0 & 
                                                res_mat_MaxIters$policy_effectiveness==0,
                                     paste("SkillTime",1:nTimesLookValidationInference,sep = '')], 2,  clippedMean  ) 
        tag1_ <- tag_ <- "No Endogeneity; \n No Policy Effectiveness "
      }
      if(j_ == 2){ 
        y_summary2 <- y_summary <- apply(  res_mat_MaxIters[res_mat_MaxIters$c_endogeneous==0 & 
                                                res_mat_MaxIters$policy_effectiveness==max(res_mat_MaxIters$policy_effectiveness),
                                     paste("SkillTime",1:nTimesLookValidationInference,sep = '')], 2,  clippedMean  ) 
        tag2_ <- tag_ <- "No Endogeneity; \n Max Policy Effectiveness "
      }
      if(j_ == 3){ 
        y_summary3 <- y_summary <- apply(  res_mat_MaxIters[res_mat_MaxIters$c_endogeneous==max(res_mat_MaxIters$c_endogeneous) & 
                                                res_mat_MaxIters$policy_effectiveness==0,
                                     paste("SkillTime",1:nTimesLookValidationInference,sep = '')], 2,  clippedMean  ) 
        tag3_ <- tag_ <- "Max Endogeneity; \n No Policy Effectiveness "
      }
      if(j_ == 4){ 
        y_summary4 <- y_summary <- apply(  res_mat_MaxIters[res_mat_MaxIters$c_endogeneous==max(res_mat_MaxIters$c_endogeneous) & 
                                                res_mat_MaxIters$policy_effectiveness==max(res_mat$policy_effectiveness),
                                              paste("SkillTime",1:nTimesLookValidationInference,sep = '')], 2,  clippedMean  ) 
        tag4_ <- tag_ <- "Max Endogeneity; \n Max Policy Effectiveness "
      }
      
      print2("Skill fig gen...")
      pdf( sprintf("%s/SimSkill_%s_%s.pdf",SaveFolder, j_, FiguresTag) )
      {
      par(mar=c(4.5,5,6,1))
      plot(  y_summary,
             type = "b",
             main = tag_,
             ylim = c(0,1),
             xaxt = "n", 
             cex.axis = cexAxis,
             cex.lab = cexLab,
             bty = 'n', pch = 19, cex = 1.6, cex.main = 1.9,
             cex.lab = 2,
             ylab = "Skill Measure",
             xlab = "Time Lookahead")
      abline(h=0)
      axis(1, at = 1:nTimesLookValidationInference,
           labels =1:nTimesLookValidationInference,
           cex.axis = cexAxis)
      }
      dev.off()
    }
      #ifelse(y_summary1<0,yes=NA,no=y_summary1)
      #plot(y_summary2/(0.01+ifelse(y_summary1<0,yes=NA,no=y_summary1)),type = "b",log="y"); 
      #points(y_summary3/(0.01+ifelse(y_summary1<0,yes=NA,no=y_summary1)),type = "b",log="y"); 
      #points(y_summary4/(0.01+ifelse(y_summary1<0,yes=NA,no=y_summary1)),type = "b",log="y"); 
      #abline(h=1,lty=3)  
    }
    
    # counterfactual policy analyses 
    if(SimMode & FALSE){ 
    print2("Counterfactual policies...")
    y_summary_PolicySkill <- apply(  res_mat[,paste("PolicySkill",1:nTimesLookValidationInference,sep = '')],2 , clippedMean  )
    y_summary_PolicyBaseline <- apply(  res_mat[,paste("PolicySkillBaseline",1:nTimesLookValidationInference,sep = '')],2 , clippedMean  )
  
    plot( y_summary )
    plot( y_summary_PolicySkill )
    plot( y_summary_PolicyBaseline[-1] )
  
    # skill fig - via policy baseline
    y_summary_PolicyBaseline[1] <- NA
    pdf( sprintf("%s/SimSkillPolicyBaseline_%s.pdf",SaveFolder, FiguresTag) )
    {
      par(mar=c(4.5,5,1,1))
      plot(  y_summary_PolicyBaseline,
             type = "b",
             main = "",
             xaxt = "n",
             cex.axis = cexAxis,
             cex.lab = cexLab,
             bty = 'n', pch = 19, cex = 1.6,
             cex.lab = 2,
             ylab = "Skill Measure",
             xlab = "Time Lookahead")
      axis(1, at = 1:nTimesLookValidationInference,
           labels =1:nTimesLookValidationInference,
           cex.axis = cexAxis)
    }
    dev.off()
    
    plot( res_mat$policy_effectiveness, res_mat$RSSBaselineSummary ,log="")
    plot( res_mat$policy_effectiveness, res_mat$SkillSummary )
    plot( res_mat$policy_decay, res_mat$SkillSummary )
    plot( res_mat$policy_responsiveness, res_mat$SkillSummary )
  
    #PlotType <- "PolicySkill";
    PlotType <- "Skill";
    DecayCoef <- DecayR2 <- EffectivenessCoef <- EffectivenessR2 <- rep(NA,nTimesLookValidationInference-1)
    EndogeneousCoef <- EndogeneousR2 <- ResponsivenessCoef <- ResponsivenessR2 <- EffectivenessCoef
    for(t_ in 2:nTimesLookValidationInference){
    if(PlotType == "PolicySkill"){ tmp <- eval(parse(text=sprintf("summary(  lm(res_mat$PolicySkillBaseline%s ~ res_mat$policy_effectiveness) )", t_)))}
     if(PlotType == "Skill"){ tmp <- eval(parse(text=sprintf("summary(  lm(res_mat$SkillTime%s ~ res_mat$policy_effectiveness) )", t_))) }
     EffectivenessCoef[t_-1L] <- tmp$coefficients[2,1]; EffectivenessR2[t_-1L] <- tmp$r.squared
  
     if(PlotType == "PolicySkill"){ tmp <- eval(parse(text=sprintf("summary(  lm(res_mat$PolicySkillBaseline%s ~ res_mat$policy_responsiveness) )", t_))) }
     if(PlotType == "Skill"){ tmp <- eval(parse(text=sprintf("summary(  lm(res_mat$SkillTime%s ~ res_mat$policy_responsiveness) )", t_))) }
     ResponsivenessCoef[t_-1L] <- tmp$coefficients[2,1]; ResponsivenessR2[t_-1L] <- tmp$r.squared
  
     if(PlotType == "PolicySkill"){ tmp <- eval(parse(text=sprintf("summary(  lm(res_mat$PolicySkillBaseline%s ~ res_mat$policy_decay) )", t_))) }
     if(PlotType == "Skill"){  tmp <- eval(parse(text=sprintf("summary(  lm(res_mat$SkillTime%s ~ res_mat$policy_decay) )", t_))) }
     DecayCoef[t_-1L] <- tmp$coefficients[2,1]; DecayR2[t_-1L] <- tmp$r.squared
  
     #tmp <- eval(parse(text=sprintf("summary(  lm(res_mat$PolicySkillBaseline%s ~ res_mat$c_endogeneous) )", t_)))
     tmp <- eval(parse(text=sprintf("summary(  lm(res_mat$SkillTime%s ~ res_mat$c_endogeneous) )", t_)))
     EndogeneousCoef[t_-1L] <- tmp$coefficients[2,1]; EndogeneousR2[t_-1L] <- tmp$r.squared
    }
  
    # breakdown plot
    plot(x_ <- 2:c(length(EffectivenessR2)+1), EffectivenessR2, pch = "E", type = "b",
         ylim = c(0,1), xlab ="Time Lookahead", ylab = "R2 in Predicting Skill from Natural Outcomes")
    points(x_, ResponsivenessR2, pch = "R", type = "b", lty = 2)
    points(x_, DecayR2, pch = "D", type = "b", lty = 3)
    points(x_, EndogeneousR2, pch = "C", type = "b", lty = 4)
  
    plot( x_,EffectivenessCoef, pch = "E", type = "b", ylim = c(-0.25,1.6),
          xlab ="Time Lookahead", ylab = "Coefficient in Predicting Skill from Natural Outcomes")
    points( x_,ResponsivenessCoef, pch = "R", type = "b", lty = 2)
    points( x_,DecayCoef, pch = "D", type = "b", lty = 2)
    abline(h=0, lty = 2, col = "gray")
  
    library(  glmnet ); colnames( res_mat )
    pred_pool <- c("xi","sigma","gamma",
                   "invbetat_sd", "betat_init",
                   "policy_responsiveness", "policy_decay","policy_effectiveness",
                   "measurement_noise", "c_endogeneous")
    summary( lm(res_mat$SkillTime5~res_mat$policy_responsiveness) )
    summary( lm(res_mat$SkillTime5~res_mat$policy_decay) )
    summary( lm(res_mat$SkillTime5~res_mat$policy_effectiveness) )
  
    coef_list <- replicate(n = nTimesLookValidationInference,list())
    res_mat_glmnet <- res_mat
    #res_mat_glmnet <- rbind(res_mat, res_mat)
    cv_min <- rep(NA,times = nTimesLookValidationInference); for(time_ in 1:nTimesLookValidationInference){
        my_glmnet <- cv.glmnet( x = as.matrix(res_mat_glmnet[,pred_pool]),
                                y = res_mat_glmnet[,paste("SkillTime",time_,sep="")],
                                nfolds = 3L)
        cv_min[time_] <- 1 - min(my_glmnet$cvm) /  var(res_mat_glmnet[,paste("SkillTime",time_,sep="")])
        coef_list[[time_]] <- coef(my_glmnet,s="lambda.min")[-1,]
    }
    coef_list <- as.data.frame(  do.call(rbind,coef_list)  )
  
    plot(res_mat$c_endogeneous, res_mat$SkillTime12)
  
    SkillNames_vec <- paste("SkillTime",1:nTimesLookValidationInference,sep="")
    if(T == F){
    pdf(sprintf("%s/EndogeneityAnalysis_%s.pdf",SaveFolder,FiguresTag))
    {
      par(mar=c(5,8,1,1))
      plot(mean_vec, 1:length(mean_vec),
           xlim = summary(c(mean_vec-(qstar <- 1.96)*se_vec,
                            mean_vec+qstar*se_vec))[c(1,6)]*c(1,1.2),
           cex = 2, pch = 19, yaxt = "n",ylab = "",
           xlab = "Skill at 12 Times",cex.lab = cexLab)
      axis(2,at = length(mean_vec), labels = round(f2n(names(mean_vec)),2),
           cex.axis = 0.90*cexAxis, las = 1)
      axis(2,at = c(2), labels = c("Endogeneity Level"),
           cex.axis =  1.3*cexAxis, line = 5, tick = F,cex=cexLab)
      for(jr in 1:length(mean_vec)){
        points(c(mean_vec[jr] - qstar*se_vec[jr],
                 mean_vec[jr] + qstar*se_vec[jr]),
               c(jr,jr), type="l",lwd=3)
      }
    }
    dev.off()
  
    pdf(sprintf("%s/PredictR2Skill_%s.pdf",SaveFolder,FiguresTag))
    {
    par(mar=c(5,6,1,1))
    plot( cv_min,
          main = '',
          bty = "n", xaxt = "n",
          xlab = "Time Lookahead",
          ylab = expression("O.O.S."~R^2~"in Predicting Skill"),
          cex.axis = cexAxis,
          cex.main = cexMain,
          cex.lab = cexLab,
          cex = 1.5)
    points(lowess(x = 1:nTimesLookValidationInference,
                  y = cv_min),
           type = "l",lwd = 2)
    axis(1, at = 1:nTimesLookValidationInference,
         labels =1:nTimesLookValidationInference,cex.axis = cexAxis)
    }
    dev.off()
  
    colnames(coef_list)
    nameDict <- c("xi" = expression("Predictor:"~xi),
                  "sigma" = expression("Predictor:"~sigma),
                  "gamma" = expression("Predictor:"~gamma),
                  "invbetat_sdfrac" = expression('Predictor: SD('~beta[t]~')'),
                  "betat_mean" = expression('Predictor: Mean('~beta[t]~')'),
                  "measurement_noise" = "Predictor: Noise Level")
  
    for(ri in 1:ncol(coef_list)){
      pdf(sprintf("%s/OLSSumf_%s,.pdf",
                  SaveFolder,
                  colnames(coef_list)[ri],
                  FiguresTag))
      par(mar=c(5,5,3,1))
      plot( coef_list[,ri],
            main = nameDict[colnames(coef_list)[ri]],
            bty = "n", xaxt = "n",
            xlab = "Time Lookahead",
            ylab = "OLS Coefficient, Skill Outcome",
            cex.axis = cexAxis,
            cex.main = cexMain,
            cex.lab = cexLab,
            cex = 1.5)
      abline(h = 0, col = "gray", lwd = 2, lty = 3)
      axis(1, at = 1:nTimesLookValidationInference,
           labels =1:nTimesLookValidationInference,cex.axis = cexAxis)
      points(lowess(x = 1:nTimesLookValidationInference,
                           y = coef_list[,ri]),
             type = "l",lwd = 2)
      dev.off()
    }
    }
    }
  }
  print("Done with SuperLModel_GenFigs.R")
}
