# Script: SuperLModel_GenScalingResults.R
{
baseSize <- 18
DropContext <- TRUE
library(stringr) # -> used to fix plotting issues with x axes 

# for setting up alpha macros 
alpha_macros <- c()
simModeStr <- if (SimMode) "True" else "False"
sanitize_for_macro <- function(s) {
  # Remove spaces, plus signs, etc. You can expand this if needed.
  s <- gsub("[+ ]", "", s, perl=TRUE)
  return( s <- gsub("\\&", "And", s, perl=TRUE) ) 
}


# Tables 
{
# ------------------------------------------------------------------------
# Scaling‑law exponents by type (including “Any”) → tidy table
# Note: Because scaling exponents are slopes on a log–log plot,
# the difference captures the architectural advantage in a way that is statistically tractable, 
# numerically stable, and easy to explain. 
# ------------------------------------------------------------------------
print2("---Collecting scaling exponents by type (and overall Any) into table---")

if (exists("res_mat") && nrow(res_mat) > 0) {
  
  # 1. Focus on final iteration
  final_iter <- max(res_mat$i_in_sgd, na.rm = TRUE)
  df_base    <- subset(res_mat, i_in_sgd == final_iter)
  # df_base <- res_mat
  # table(res_mat$ContextLength)
  
  # sanity check 
  # plot(df_base$ContextLength, df_base$SkillSummary)
  tmp <- df_base[df_base$ContextLength==62,]
  # View(tmp[order(tmp$SkillSummary,decreasing=T),] )
  sort(unique(df_base$i_in_sgd))
  # plot(df_base$ContextLength[ii_<-which(df_base$policy_effectiveness==0.00)], df_base$SkillSummary[ii_], ylim=c(-5,1))
  # plot(df_base$ContextLength[ii_<-which(df_base$i_in_sgd==448)], df_base$SkillSummary[ii_], ylim=c(-5,1))

  plot(df_base$SkillSummary,df_base$SkillSummary_inv)
  tapply(df_base$SkillSummary,df_base$nSamplesTrain,median)
  tapply(df_base$SkillSummary,df_base$ContextLength,median)
  table(df_base$nSamplesTrain,df_base$ContextLength)
  
  # 2. Include an "Any" option plus each specific type
  if(SimMode){ 
    
    # Add both a numeric rank (1 … k) and a human-readable label
    df_base$endogeneity_index <- df_base$c_endogeneous+
                                      df_base$policy_effectiveness
    analyzeThisScaling <- "endogeneity_index"
    #analyzeThisScaling <- "c_endogeneous"
    #analyzeThisScaling <- "policy_effectiveness"
    
    sort(unique(df_base[[analyzeThisScaling]]))
    df_base$c_analyze_rank   <- robust_cut(df_base[[analyzeThisScaling]], 
                                           n_bins = 3)
    print( tapply(df_base[[analyzeThisScaling]], df_base$c_analyze_rank, mean))
    df_base$c_analyze_rank <- factor(df_base$c_analyze_rank,
                                     levels = sort(unique(df_base$c_analyze_rank)),
                                     labels = c("Low", "Medium", "High"))
    cond_types <- c("Any", unique(as.character(df_base$c_analyze_rank)))
  }
  if(!SimMode){ cond_types <- c("Any", unique(as.character(df_base$OSSType))) } 
  
  # 3. Resources (log‑scale columns) and model types
  resources <- c(
                 "ContextLength_log",
                 "nSamplesTrain_log",
                 "ModelDims_log",     
                 "i_in_sgd_log"
                 )
  if(DropContext){resources <- resources[!grepl(resources,pattern="ContextLength")] }
  models    <- unique(df_base$ModelType)
  
  # 4. Prepare list to collect each row
  total_rows <- length(cond_types) * length(models) * length(resources)
  out_list   <- vector("list", total_rows)
  idx        <- 1L
  
  # 5. Loop over each type (or Any), model, and resource
  for (the_type in cond_types) {
    # choose subset or full
    if(SimMode){ df_type <- if (the_type == "Any") df_base else subset(df_base, c_analyze_rank == the_type) }
    if(!SimMode){ df_type <- if (the_type == "Any") df_base else subset(df_base, OSSType == the_type) }
    
    for (mt in models) {
      df_m <- subset(df_type, ModelType == mt)
      
      for (res in resources) {
        x        <- df_m[[res]]
        n_unique <- length(unique(x))
        
        # Initialize row
        row <- list(
          OSSType    = ifelse(SimMode, yes = NA, no = the_type),
          Endogeneity = ifelse(SimMode, yes = the_type, no = NA),
          ModelType  = mt,
          Resource   = sub("_log$", "", res),
          N_obs      = nrow(df_m),
          N_unique   = n_unique,
          p_val    = NA,
          Alpha      = NA_real_,
          Alpha_low  = NA_real_,
          Alpha_high  = NA_real_,
          Alpha_se    = NA_real_,
          R2         = NA_real_,
          Note       = NA_character_
        )
        
        # Fit if enough variation
        if (n_unique < 3) {
          row$Note <- sprintf("skipped (%d unique)", n_unique)
        }
        if( (n_unique > 3) ){ 
          print("Found a viable analysis grid cell.")
          
          grid_vars <- intersect(names(TheGrid), names(df_m))
          grid_vars <- grid_vars[sapply(grid_vars, function(v) length(unique(df_m[[v]])) > 1)]
          grid_vars <- grid_vars[!grepl(grid_vars,
                                  pattern = gsub(res,pattern="_log",replace=""))]
          fm <- as.formula(sprintf("SkillSummary_inv ~ %s + %s", 
                           res, paste(grid_vars, collapse = " + ")))
          fit_multi <- lm(fm, data = df_m)
          eval(parse(text=sprintf("fit <- lm(SkillSummary_inv ~ %s, data = df_m)",res)))
          coef(summary(fit_multi))[2,3]
          coef(summary(fit))[2,3]
          
          # use multivariate fits 
          fit <- fit_multi 
          
          cf        <- coef(fit)
          row$Alpha <- cf[2] # with LOSS, we take negative. with SKILL, we do not
          row$Alpha_low <- row$Alpha - 1.96*coef(summary(fit))[2,2]
          row$Alpha_high <- row$Alpha + 1.96*coef(summary(fit))[2,2]
          row$Alpha_se <- coef(summary(fit))[2,2]
          row$R2    <- summary(fit)$r.squared
          row$p_val <- coef(summary(fit))[2,4]
          row$Note  <- ""
          
          # append to alpha macros
          {
            # Build the macro name
            # Example: \AlphaSimModeTrueAnyDecoderContext
            macro_name <- paste0(
              "\\AlphaSimMode", stringr::str_to_title(SimMode),"VV",
               mt, "VV",
              "Resource", gsub(res,pattern="_log",replace=""),"VV",
              "CondType" ,the_type)
            
            # Generate the \newcommand line
            alpha_macros <- c(alpha_macros, 
                              sprintf("\\newcommand{%s}{%.3f}", macro_name, row$Alpha),
                              sprintf("\\newcommand{%sSE}{%.3f}", macro_name, row$Alpha_se))
          }
        }
        
        # save outputs 
        out_list[[idx]] <- row
        idx <- idx + 1L
        
      }
    }
  }
  
  # 6. Assemble into a clean data.frame
  scaling_exponents_oos <- do.call(
    rbind, lapply(out_list[1:(idx-1)], as.data.frame, stringsAsFactors = FALSE) 
  )
  rownames(scaling_exponents_oos) <- NULL
  
  # 7. Attach metadata
  scaling_exponents_oos <- as.data.frame(scaling_exponents_oos[!is.na(scaling_exponents_oos$Alpha),])
  
  if(!SimMode){
    delta_mat <- scaling_exponents_oos %>%
      filter(ModelType %in% c("NeuralODE", "DecoderOnly")) %>%
      # pivot NeuralODE / DecoderOnly α’s into their own columns
      tidyr::pivot_wider(
        id_cols   = c(OSSType, Endogeneity, Resource),
        names_from   = ModelType,
        values_from  = Alpha
      ) %>%
      # compute difference and ratio
      mutate(
        alpha_diff  = NeuralODE - DecoderOnly,
        alpha_ratio = NeuralODE / DecoderOnly
      )
  }
  
  if(SimMode){
    delta_mat <- scaling_exponents_oos |>
      select(Endogeneity, Resource, ModelType, Alpha) |>   # keep Endogeneity
      tidyr::pivot_wider(names_from = ModelType, values_from = Alpha) |>
      mutate(delta_alpha = DecoderOnly - NeuralODE)  
  }
  
  delta_mat[delta_mat$Resource=="nSamplesTrain",]
  delta_mat[delta_mat$Resource=="ModelDims",]
  
  # subset 
  scaling_exponents_tex <- scaling_exponents_oos[,!colnames(scaling_exponents_oos) %in% 
                                                   c("Note","N_obs","N_unique","p_val",
                                                     "Alpha_low","Alpha_high")]
  
  # convert to character
  scaling_exponents_tex$Alpha <- helpeRs::fixZeroEndings(as.character(round(scaling_exponents_tex$Alpha,3)),roundAt = 3)
  scaling_exponents_tex$Alpha_se <- helpeRs::fixZeroEndings(as.character(round(scaling_exponents_tex$Alpha_se,3)),roundAt = 3)
  scaling_exponents_tex$R2 <- helpeRs::fixZeroEndings(as.character(round(scaling_exponents_tex$R2,3)),roundAt = 3)

  # renames 
  scaling_exponents_tex$Endogeneity[scaling_exponents_tex$Endogeneity=="Any"] <- "Any"
  scaling_exponents_tex$OSSType[scaling_exponents_tex$OSSType=="Any"] <- "Any"
  scaling_exponents_tex$OSSType[scaling_exponents_tex$OSSType=="OutOfPlace"] <- "Out of place"
  scaling_exponents_tex$OSSType[scaling_exponents_tex$OSSType=="OutOfTime"] <- "Out of time"
  scaling_exponents_tex$OSSType[scaling_exponents_tex$OSSType=="OutOfPlacetime"] <- "Out of place+time"
  
  # --- ModelType ---
  scaling_exponents_tex$ModelType[scaling_exponents_tex$ModelType == "NeuralODE"]   <- "Neural ODE"
  scaling_exponents_tex$ModelType[scaling_exponents_tex$ModelType == "DecoderOnly"] <- "Decoder only"
  
  # --- Resource ---
  scaling_exponents_tex$Resource[scaling_exponents_tex$Resource == "ContextLength"] <- "Context length"
  scaling_exponents_tex$Resource[scaling_exponents_tex$Resource == "nSamplesTrain"] <- "Training samples"
  scaling_exponents_tex$Resource[scaling_exponents_tex$Resource == "ModelDims"]     <- "Model dims"
  
  scaling_exponents_tex$Alpha[scaling_exponents_oos$p_val > 0.05] <- paste0("RBRACKET GRAY",
                                                  scaling_exponents_tex$Alpha[scaling_exponents_oos$p_val > 0.05], "LBRACKET")
  scaling_exponents_tex$Alpha_se[scaling_exponents_oos$p_val > 0.05] <- paste0("RBRACKET GRAY",
                                                                            scaling_exponents_tex$Alpha_se[scaling_exponents_oos$p_val > 0.05], "LBRACKET")
  scaling_exponents_tex$R2[scaling_exponents_oos$p_val > 0.05] <- paste0("RBRACKET GRAY",
                                                  scaling_exponents_tex$R2[scaling_exponents_oos$p_val > 0.05], "LBRACKET")
  
  # subset 
  if(SimMode){scaling_exponents_tex <- scaling_exponents_tex[,!colnames(scaling_exponents_tex) %in% "OSSType"]}
  if(!SimMode){scaling_exponents_tex <- scaling_exponents_tex[,!colnames(scaling_exponents_tex) %in% "Endogeneity"]}
  
  # save as matrix 
  scaling_exponents_tex <- as.matrix(scaling_exponents_tex)
  colnames(scaling_exponents_tex)[colnames(scaling_exponents_tex) =="OSSType"] <- "Hold-out Regime"
  colnames(scaling_exponents_tex)[colnames(scaling_exponents_tex) =="ModelType"] <- "Model Type"
  colnames(scaling_exponents_tex)[colnames(scaling_exponents_tex) =="Alpha_se"] <- "Alpha, se"
  row.names(scaling_exponents_tex) <- NULL
  
  # write base table to disk 
  {
  latex_code <- capture.output(stargazer::stargazer(as.matrix(scaling_exponents_tex),
                       label = sprintf("tab:ExponentsSimMode%s",SimMode), 
                       title = sprintf("
                       Scaling‐law exponents $\\alpha$ and $R^2$ from fitting
                       $\\log(g(\\text{skill}))\\sim \\alpha\\;\\log(\\text{resource})+\\beta$,
                       disaggregated by hold‐out regime and model type. 
                       Gray shading results $\\alpha$ values 
                       for which the null hypothesis of zero-equality cannot be rejected. 
                                       %s",
                                        ifelse(SimMode, 
                                               yes = "Case: simulation.",
                                               no = "Case: COVID-19 analysis."))
                       ))
  tabular_header_index <- 3+grep("\\\\begin\\{tabular\\}\\{", latex_code)
  tmp <- strsplit(latex_code[tabular_header_index],split = " ")[[1]]
  for(j in 1:length(tmp)){ 
    if(!tmp[j] %in% c("&","\\\\")){
      tmp[j] <- sprintf("\\textbf{%s}",tmp[j])
  } }
  latex_code[tabular_header_index] <- paste(tmp, collapse = " ")
  latex_code <- gsub(latex_code,pattern="GRAY",replace="\\\\color{gray}")
  latex_code <- gsub(latex_code,pattern="RBRACKET",replace="{")
  latex_code <- gsub(latex_code,pattern="LBRACKET",replace="}")
  latex_code <- gsub(latex_code,pattern="R2",replace="R-squared")
  tabular_line_index <- grep("\\\\begin\\{tabular\\}\\{", latex_code)
  # Check if the line was found
  if (length(tabular_line_index) == 1) {
    # Define the desired alignment: 3 left, 2 right for the 5 columns
    # Adjust this string if the number of columns is different
    # Determine alignment: 'l' for non-numeric, 'r' for numeric
    desired_alignment <- sapply(1:ncol(scaling_exponents_tex), function(i) {
      # Access column data - using [,] works for both data frames and matrices
      column_data <- scaling_exponents_tex[1, i]
      
      # Check if the column is numeric
      if (is.na(f2n(column_data))) {
        return("r") # Left-align numeric columns
      } else {
        return("c") # Center-align non-numeric columns (character, factor, logical, etc.)
      }
    })
    
    # Collapse the individual alignment characters into a single string (e.g., "lllrr")
    desired_alignment <- paste(desired_alignment, collapse = "")
    
    # Replace the existing alignment specifier (e.g., {ccccc}) with the desired one
    replacement_ <- strsplit(latex_code[tabular_line_index], split=" ")[[1]]
    replacement_[2] <- paste0(desired_alignment,'}')
    latex_code[tabular_line_index] <- paste(replacement_,collapse = " ")
  }

  writeLines(
    latex_code,
    con = sprintf("./Figures/ScalingTable_SimMode%s.tex",SimMode)
  )
  }
  
  # write difference table to disk 
  {
    # --- Prepare delta_mat for LaTeX ---
    # Create a separate data frame for TeX output to avoid modifying delta_mat
    delta_mat_tex <- delta_mat
    
    if(!SimMode){
      # Select and reorder columns for the table
      delta_mat_tex <- delta_mat_tex[, c("OSSType", "Resource", "NeuralODE",
                                         "DecoderOnly", "alpha_diff", "alpha_ratio")]
      
      # Format numeric columns (round and ensure trailing zeros)
      delta_mat_tex$NeuralODE   <- helpeRs::fixZeroEndings(as.character(round(delta_mat_tex$NeuralODE, 3)), roundAt = 3)
      delta_mat_tex$DecoderOnly <- helpeRs::fixZeroEndings(as.character(round(delta_mat_tex$DecoderOnly, 3)), roundAt = 3)
      delta_mat_tex$alpha_diff  <- helpeRs::fixZeroEndings(as.character(round(delta_mat_tex$alpha_diff, 3)), roundAt = 3)
      delta_mat_tex$alpha_ratio <- helpeRs::fixZeroEndings(as.character(round(delta_mat_tex$alpha_ratio, 2)), roundAt = 2) # Ratio often shown with 2 decimals
      
      # Rename descriptive factor levels for readability
      delta_mat_tex$OSSType[delta_mat_tex$OSSType=="Any"] <- "Any"
      delta_mat_tex$OSSType[delta_mat_tex$OSSType=="OutOfPlace"] <- "Out of place"
      delta_mat_tex$OSSType[delta_mat_tex$OSSType=="OutOfTime"] <- "Out of time"
      delta_mat_tex$OSSType[delta_mat_tex$OSSType=="OutOfPlacetime"] <- "Out of place+time"
      
      delta_mat_tex$Resource[delta_mat_tex$Resource == "ContextLength"] <- "Context length"
      delta_mat_tex$Resource[delta_mat_tex$Resource == "nSamplesTrain"] <- "Training samples"
      delta_mat_tex$Resource[delta_mat_tex$Resource == "ModelDims"]     <- "Model dims"
      
      # Convert to matrix for stargazer
      delta_mat_tex <- as.matrix(delta_mat_tex)
      row.names(delta_mat_tex) <- NULL
      
      # Define the title for the LaTeX table
      table_title <- "Difference in scaling‐law exponents ($\\alpha$) between Neural ODE and Decoder only models. Case: COVID-19 analysis."
      delta_mat_tex <- delta_mat_tex[,!colnames(delta_mat_tex) %in% c("NeuralODE","DecoderOnly","alpha_ratio")]
      colnames(delta_mat_tex)[colnames(delta_mat_tex) == "alpha_diff"] <- "Alpha Diff"
      
      # Generate LaTeX code using stargazer
      latex_code_delta <- capture.output(stargazer::stargazer(delta_mat_tex,
                                                              title = table_title,
                                                              label = sprintf("tab:DeltaExponentsSimMode%s",SimMode), 
                                                              summary = FALSE, # Don't output summary statistics
                                                              rownames = FALSE # Don't output matrix row names
      ))
      
      # Post-process LaTeX code for better formatting and column names
      latex_code_delta <- gsub(latex_code_delta, pattern="OSSType", replace="Hold-out Regime") # More descriptive header
      latex_code_delta <- gsub(latex_code_delta, pattern="NeuralODE", replace="Neural ODE $\\\\alpha$") # Add alpha symbol
      latex_code_delta <- gsub(latex_code_delta, pattern="DecoderOnly", replace="Decoder $\\\\alpha$") # Add alpha symbol
      # Stargazer typically escapes underscores in column names with a backslash
      latex_code_delta <- gsub(latex_code_delta, pattern="alpha\\\\_diff", replace="$\\\\Delta \\\\alpha$ (ODE - Dec)") # Delta symbol and clarification
      latex_code_delta <- gsub(latex_code_delta, pattern="alpha\\\\_ratio", replace="Ratio $\\\\alpha$ (ODE / Dec)") # Clarification
      
      # Adjust column alignment:
      tabular_line_index_delta <- grep("\\\\begin\\{tabular\\}\\{", latex_code_delta)
      tmp <- strsplit(latex_code_delta[tabular_line_index_delta],split = " ")[[1]]
      tmp[2] <-  "rrc}"
      latex_code_delta[tabular_line_index_delta] <- paste(tmp, collapse = " ")
      
      # bold cols 
      subset_loc <- 3+grep("\\\\begin\\{tabular\\}\\{", latex_code_delta)
      tmp <- strsplit(latex_code_delta[subset_loc],split = " ")[[1]]
      for(j in 1:length(tmp)){ 
        if(!tmp[j] %in% c("&","\\\\")){
          tmp[j] <- sprintf("\\textbf{%s}",tmp[j])
        } }
      latex_code_delta[subset_loc] <- paste(tmp, collapse = " ")
    }
    if(SimMode){ # SimMode == TRUE
      # Select and reorder columns for the table
      # Note: delta_alpha = DecoderOnly - NeuralODE in SimMode
      delta_mat_tex <- delta_mat_tex[, c("Endogeneity", "Resource", 
                                         "DecoderOnly", "NeuralODE", "delta_alpha")]
      
      # Format numeric columns
      delta_mat_tex$DecoderOnly <- helpeRs::fixZeroEndings(as.character(round(delta_mat_tex$DecoderOnly, 3)), roundAt = 3)
      delta_mat_tex$NeuralODE   <- helpeRs::fixZeroEndings(as.character(round(delta_mat_tex$NeuralODE, 3)), roundAt = 3)
      delta_mat_tex$delta_alpha <- helpeRs::fixZeroEndings(as.character(round(delta_mat_tex$delta_alpha, 3)), roundAt = 3)
      
      # Rename descriptive factor levels (Endogeneity levels are already factors from earlier code)
      delta_mat_tex$Resource[delta_mat_tex$Resource == "ContextLength"] <- "Context length"
      delta_mat_tex$Resource[delta_mat_tex$Resource == "nSamplesTrain"] <- "Training samples"
      delta_mat_tex$Resource[delta_mat_tex$Resource == "ModelDims"]     <- "Model dims"
      
      # Convert to matrix for stargazer
      delta_mat_tex <- as.matrix(delta_mat_tex)
      row.names(delta_mat_tex) <- NULL
      
      # Define the title for the LaTeX table
      table_title <- "Difference in scaling‐law exponents ($\\alpha$) between Decoder Only and Neural ODE models. Case: simulation."
      delta_mat_tex <- delta_mat_tex[,!colnames(delta_mat_tex) %in% c("NeuralODE","DecoderOnly","alpha_ratio")]
      colnames(delta_mat_tex)[colnames(delta_mat_tex) == "alpha_diff"] <- "Alpha Diff"
      
      # Generate LaTeX code using stargazer
      latex_code_delta <- capture.output(stargazer::stargazer(delta_mat_tex,
                                                              title = table_title,
                                                              label = sprintf("tab:DeltaExponentsSimMode%s",SimMode), 
                                                              summary = FALSE,
                                                              rownames = FALSE
      ))
      
      # Post-process LaTeX code
      
      # bold cols
      subset_i <- 3+grep("\\\\begin\\{tabular\\}\\{", latex_code_delta)
      tmp <- strsplit(latex_code_delta[subset_i],split = " ")[[1]]
      for(j in 1:length(tmp)){ 
        if(!tmp[j] %in% c("&","\\\\")){
          tmp[j] <- sprintf("\\textbf{%s}",tmp[j])
        } }
      latex_code_delta[subset_i] <- paste(tmp, collapse = " ")
      
      # latex_code_delta <- gsub(latex_code_delta, pattern="Endogeneity", replace="Endogeneity Rank") # Optional: Rename if desired
      latex_code_delta <- gsub(latex_code_delta, pattern="DecoderOnly", replace="Decoder $\\\\alpha$")
      latex_code_delta <- gsub(latex_code_delta, pattern="NeuralODE", replace="Neural ODE $\\\\alpha$")
      latex_code_delta <- gsub(latex_code_delta, pattern="delta\\\\_alpha", replace="$\\\\Delta \\\\alpha$ (Dec - ODE)") # Delta symbol and clarification
      
      # Adjust column alignment:
      tabular_line_index_delta <- grep("\\\\begin\\{tabular\\}\\{", latex_code_delta)
      tmp <- strsplit(latex_code_delta[tabular_line_index_delta],split = " ")[[1]]
      tmp[2] <-  "rrc}"
      latex_code_delta[tabular_line_index_delta] <- paste(tmp, collapse = " ")
    }
    
    # Write the generated LaTeX code to a .tex file
    writeLines(
      latex_code_delta,
      con = sprintf("./Figures/ScalingDeltaTable_SimMode%s.tex", SimMode)
    )
  }
  
  # 8. Inspect or save
  print(scaling_exponents_oos)
  head(scaling_exponents_oos)
  
  tapply(scaling_exponents_oos$R2,
         scaling_exponents_oos$OSSType,mean)
}
}

# Figures
{
  # Load required library
  library(ggplot2)
  
  # Map Resource and ModelType to desired labels for consistency
  scaling_exponents_oos$Resource <- factor(scaling_exponents_oos$Resource,
                                           levels = c("ContextLength", "nSamplesTrain", "ModelDims"),
                                           labels = c("Context Length", "Training Samples", "Model Dims"))
  scaling_exponents_oos$ModelType <- factor(scaling_exponents_oos$ModelType,
                                            levels = c("NeuralODE", "DecoderOnly"),
                                            labels = c("Neural ODE", "Decoder only"))
  scaling_exponents_oos$OSSType <- factor(scaling_exponents_oos$OSSType,
                                            levels = c("Any", "OutOfPlace","OutOfTime","OutOfPlacetime"),
                                            labels = c("Any", "Out of Place","Out of Time","Out of Place + Time"))
  
  # Plot 1 or Plot 2: Grouped bar chart for 'Any' category
  if (SimMode) {
    # Simulation Results (Plot 1)
    scaling_exponents_any <- scaling_exponents_oos[scaling_exponents_oos$Endogeneity == "Any", ]
    title_text <- "Simulation Results"
  } else {
    # COVID-19 Results (Plot 2)
    scaling_exponents_any <- scaling_exponents_oos[scaling_exponents_oos$OSSType == "Any", ]
    title_text <- "COVID-19 Results"
  }
  
  # Create the main grouped bar chart
  p <- ggplot(scaling_exponents_any, 
              aes(x = Resource, y = Alpha, color = ModelType)) +
    # draw the “stick” from zero up (or down) to the point
    #geom_segment(aes(xend = Resource, y = 0, yend = Alpha),
                 #position = position_dodge(width = 0.6),
                 #size = 0.8) +
    # add the point at the top of the stick
    geom_point(position = position_dodge(width = 0.6),
               size = 3) +
    # add error bars for the confidence interval
    geom_errorbar(aes(ymin = Alpha_low, ymax = Alpha_high),
                  width = 0.15,
                  position = position_dodge(width = 0.6)) +
    # horizontal zero line
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    # labels
    labs(x    = "Resource Type",
         y    = expression("Scaling Exponent ("*alpha*")"),
         color= "Model Type",
         title= title_text, 
         subtitle= "Combined Analysis"
         ) +
    # minimal theme
    theme_minimal(base_line_size = 0, 
                  base_size = baseSize) +
    theme(legend.position = "bottom",
          plot.title      = element_text(face = "bold", hjust = 0.5),
          axis.ticks      = element_line(color = "grey80"),
          axis.line       = element_line(color = "grey80"))
  print(p)

  # Save the plot
  ggsave(ifelse(SimMode,
                "./Figures/Simulation_Scaling_Exponents.pdf",
                "./Figures/COVID19_Scaling_Exponents.pdf"), p)
  
  # scaling_exponents_oos[scaling_exponents_oos$Resource=="Context Length",]
  # Faceted plots for different conditions
  if (SimMode) {
    # plot facets by Endogeneity level
    p_faceted <- ggplot(scaling_exponents_oos, 
                      aes(x = Resource, y = Alpha, color = ModelType)) +
      # point at the tip (dodged by ModelType)
      geom_point(position = position_dodge(width = 0.6), size = 3) +
      # vertical whiskers for 95% CI
      geom_errorbar(aes(ymin = Alpha_low, ymax = Alpha_high),
                    width = 0.15,
                    position = position_dodge(width = 0.6)) +
      # zero reference line
      geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
      # facet by endogeneity level
      facet_wrap(~ Endogeneity) +
      # labels
      labs(
        x     = "Resource Type",
        y     = expression("Scaling Exponent ("*alpha*")"),
        color = "Model Type",
        title = "Simulation Results by Endogeneity Level"
      ) +
      # wrap any long Resource labels at ~15 characters
      scale_x_discrete(labels = function(x) str_wrap(x, width = 15)) +
      # preserve existing theme settings + tweak x-text for wrapping
      theme(
        legend.position = "bottom",
        plot.title      = element_text(face = "bold", hjust = 0.5),
        axis.ticks      = element_line(color = "grey80"),
        axis.line       = element_line(color = "grey80"),
        axis.text.x     = element_text(
          vjust = 0.5,
          hjust = 0.5,
          margin = margin(t = 5)
        )
      ) + 
      theme_minimal(base_line_size = 0,
                    base_size = 0.90*baseSize)
    print(p_faceted)
    ggsave("./Figures/Simulation_Scaling_Exponents_Faceted.pdf", 
           p_faceted,
           width  = 8,      # total width, in inches
           height = 6,      # total height, in inches
           units  = "in",   # units for width/height
           dpi    = 300     # only needed for raster formats (e.g. PNG))
           )
  }
  if(!SimMode){ 
    # Plot 2 with facets by Hold-out Regime
    p_faceted <- ggplot(scaling_exponents_oos, 
                        aes(x = Resource, y = Alpha, color = ModelType)) +
      # dot at each α (dodged by ModelType)
      geom_point(position = position_dodge(width = 0.6), size = 3) +
      # 95% CI error bars
      geom_errorbar(aes(ymin = Alpha_low, ymax = Alpha_high),
                    width = 0.15,
                    position = position_dodge(width = 0.6)) +
      # reference line at zero
      geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
      # separate by hold-out regime
      facet_wrap(~ OSSType) +
      # axis and title labels
      labs(
        x     = "Resource Type",
        y     = expression("Scaling Exponent ("*alpha*")"),
        color = "Model Type",
        title = "COVID-19 Results",
        subtitle = "by Hold-out Regime"
      ) +
      # wrap any long Resource labels at ~15 characters
      scale_x_discrete(labels = function(x) str_wrap(x, width = 15)) +
      # minimal “Tufte-style” theme + tweaks
      theme_minimal(base_line_size = 0, 
                    base_size = 0.80*baseSize) +
      theme(
        legend.position = "bottom",
        plot.title      = element_text(face = "bold", hjust = 0.5),
        axis.ticks      = element_line(color = "grey80"),
        axis.line       = element_line(color = "grey80"),
        axis.text.x     = element_text(
          vjust = 0.5, 
          hjust = 0.5, 
          margin = margin(t = 5)
        )
      )
    p_faceted
    ggsave("./Figures/COVID19_Scaling_Exponents_Faceted.pdf",
           p_faceted)
  }
  
  # Plot: Difference in Scaling for Simulation
  if (SimMode) {
    # Map Resource labels in delta_mat
    delta_mat$Resource <- factor(delta_mat$Resource,
                                 levels = c("ContextLength", "nSamplesTrain", "ModelDims"),
                                 labels = c("Context Length", "Training Samples", "Model Dims")
    )
    
    # (Optional) Re-factor endogeneity levels, if desired
    # Only do this if you have a known set of levels like c("None","Low","Medium","High").
    # Example:
    # delta_mat$Endogeneity <- factor(delta_mat$Endogeneity,
    #   levels = c("None","Low","Medium","High"),  # or "Any" if included
    #   labels = c("None","Low","Medium","High")
    # )
    
    # Create a bar chart for difference in alpha (DecoderOnly − NeuralODE)
    p3 <- ggplot(delta_mat, aes(x = Resource, y = delta_alpha)) +
      # draw the “stick” from zero to the difference
      geom_segment(aes(xend = Resource, y = 0, yend = delta_alpha),
                   size = 0.8) +
      # dot at the end
      geom_point(size = 3) +
      # reference line at zero
      geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
      # separate panels by endogeneity level
      facet_wrap(~ Endogeneity) +
      # labels
      labs(
        x     = "Resource Type",
        y     = expression(Delta*alpha*":  Decoder Only  -  Neural ODE"),
        title = "Diff. in Scaling Exponents",
        subtitle = "by Case"
      ) +
      # wrap any long x‐tick labels
      scale_x_discrete(labels = function(x) str_wrap(x, width = 15)) +
      # minimal theme
      theme_minimal(base_line_size = 0, 
                    base_size = baseSize) +
      theme(
        plot.title    = element_text(face = "bold", hjust = 0.5),
        axis.ticks    = element_line(color = "grey80"),
        axis.line     = element_line(color = "grey80"),
        axis.text.x   = element_text(
          vjust = 0.5,
          hjust = 0.5,
          margin = margin(t = 5)
        )
      )
    
    # Print and save
    print(p3)
    ggsave("./Figures/Simulation_Scaling_Difference.pdf", p3)
  }
  
  # Plot: Difference in Scaling for COVID-19 
  if (!SimMode) {
    # Map Resource labels in delta_mat
    delta_mat$Resource <- factor(delta_mat$Resource,
                                 levels = c("ContextLength", "nSamplesTrain", "ModelDims"),
                                 labels = c("Context Length", "Training Samples", "Model Dims"))
    delta_mat$OSSType <- factor(delta_mat$OSSType,
                                levels = c("Any", "OutOfPlace","OutOfTime","OutOfPlacetime"),
                                labels = c("Any", "Out of Place","Out of Time","Out of Place + Time"))
    
    # Create bar chart for difference in alpha
    p3 <- ggplot(delta_mat, aes(x = Resource, y = alpha_diff)) +
      # draw the “stick” from zero to the difference
      geom_segment(aes(xend = Resource, y = 0, yend = alpha_diff),
                   size = 0.8) +
      # dot at the end
      geom_point(size = 3) +
      # reference line at zero
      geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
      # separate panels by hold-out regime
      facet_wrap(~ OSSType) +
      # labels
      labs(
        x     = "Resource Type",
        y     = expression(Delta*alpha*":  Neural ODE  -  Decoder Only"),
        title = "Diff. in Scaling Exponents",
        subtitle = "by Hold-out Regime"
      ) +
      # wrap any long x‐tick labels to ~15 characters
      scale_x_discrete(labels = function(x) str_wrap(x, width = 15)) +
      # Tufte-style minimal theme
      theme_minimal(base_line_size = 0,
                    base_size = baseSize) +
      theme(
        plot.title    = element_text(face = "bold", hjust = 0.5),
        axis.ticks    = element_line(color = "grey80"),
        axis.line     = element_line(color = "grey80"),
        axis.text.x   = element_text(          # center & nudge wrapped text
          vjust = 0.5,
          hjust = 0.5,
          margin = margin(t = 5)
        )
      )
    ggsave("./Figures/COVID19_Scaling_Difference.pdf", p3)
  }
}
}
