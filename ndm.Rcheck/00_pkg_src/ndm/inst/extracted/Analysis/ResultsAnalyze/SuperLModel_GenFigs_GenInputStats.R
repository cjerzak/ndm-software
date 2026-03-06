# Content of SuperLModel_GenFigs_InputStats.R
{
  ##############################################################################
  # Purpose: Generate concise TeX macros (\newcommand{}) containing key summary
  #          statistics for referencing in LaTeX.  Place at the end of 
  #          SuperLModel_GenFigs.R (after res_mat has been created).
  ##############################################################################

  simModeStr <- if (SimMode) "True" else "False"
  
  table(res_mat$i_in_sgd)
  tapply(res_mat$RSSBaselineSummary, res_mat$nSamplesTrain, mean)
  # tapply(res_mat$RSSBaselineSummary, res_mat$c_endogeneous, median)
  # tapply(res_mat$RSSBaselineSummary, res_mat$policy_effectiveness, median)
  tapply(res_mat$RSSBaselineSummary, res_mat$nSamplesTrain, se)
  
  # Compute summary statistics for referencing in LaTeX (examples)
  NObservations    <- nrow(res_mat)
  NModelTypes      <- length(unique(res_mat$ModelType))
  NUniqueDiseases  <- length(unique(res_mat$BaseID))
  NTrainedModels  <- length(unique(res_mat$sim_index))
  MaxSamplesTrain  <- max(res_mat$nSamplesTrain, na.rm=TRUE)
  MedianFinalSkill  <- median(res_mat$SkillSummary[res_mat$i_in_sgd == max(res_mat$i_in_sgd)], na.rm=TRUE)
  MedianFinalSkill_Decoder  <- median(res_mat$SkillSummary[res_mat$i_in_sgd == max(res_mat$i_in_sgd) & 
                                res_mat$ModelType=="DecoderOnly"], na.rm=TRUE)
  MedianFinalSkill_NeuralODE  <- median(res_mat$SkillSummary[res_mat$i_in_sgd == max(res_mat$i_in_sgd) & 
                                  res_mat$ModelType=="NeuralODE"], na.rm=TRUE)
  MaxFinalSkill   <- max(res_mat$SkillSummary[res_mat$i_in_sgd == max(res_mat$i_in_sgd)],
                               na.rm=TRUE)
  MinContextLength <- min(res_mat$ContextLength, na.rm=TRUE)
  MaxContextLength <- max(res_mat$ContextLength, na.rm=TRUE)
  
  # generate outputs 
  output_lines <- c(
    sprintf("\\newcommand{\\NObservationsSimMode%s}{%s}",
            simModeStr, format(NObservations, big.mark=",")),
    
    sprintf("\\newcommand{\\NModelTypesSimMode%s}{%i}",
            simModeStr, NModelTypes),
    
    sprintf("\\newcommand{\\MaxSamplesTrainSimMode%s}{%s}",
            simModeStr, format(as.integer(MaxSamplesTrain),big.mark=",")),
    
    sprintf("\\newcommand{\\MedianFinalSkillSimMode%s}{%.2f}",
            simModeStr, MedianFinalSkill),
    
    sprintf("\\newcommand{\\MedianFinalSkillDecoderOnlySimMode%s}{%.2f}",
            simModeStr, MedianFinalSkill_Decoder),
    
    sprintf("\\newcommand{\\MedianFinalSkillNeuralODESimMode%s}{%.2f}",
            simModeStr, MedianFinalSkill_NeuralODE),
    
    sprintf("\\newcommand{\\MaxFinalSkillSimMode%s}{%.2f}",
            simModeStr, MaxFinalSkill),
    
    sprintf("\\newcommand{\\MinContextLengthSimMode%s}{%i}",
            simModeStr, MinContextLength),
    
    sprintf("\\newcommand{\\NUniqueDiseasesSimMode%s}{%i}",
            simModeStr, NUniqueDiseases),
    
    sprintf("\\newcommand{\\NTrainedModelsSimMode%s}{%i}",
            simModeStr, NTrainedModels),
    
    sprintf("\\newcommand{\\MaxContextLengthSimMode%s}{%i}",
            simModeStr, MaxContextLength),
    
    alpha_macros
  )
  
  # Write these stats to an output .tex file (which you can then \input in LaTeX)
  writeLines(output_lines, 
             con = sprintf("./Figures/InputStatisticsSimMode%s.tex", simModeStr))
  
  message("Finished writing Figures/InputStatistics.tex with key stats.")
}

