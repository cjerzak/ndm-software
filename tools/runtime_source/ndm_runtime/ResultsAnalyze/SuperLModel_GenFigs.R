# Reporting uses the same validated tables for summaries, pairing and plots.
{
  SimMode <- isTRUE(get0("SimMode", inherits = TRUE, ifnotfound = FALSE))
  AnalysisName <- get0("AnalysisName", inherits = TRUE, ifnotfound = NA_character_)
  if (length(AnalysisName) != 1L || is.na(AnalysisName) || !nzchar(AnalysisName)) stop("Figure generation requires the resolved AnalysisName.")
  analysis2_spec <- get0("analysis2_run_spec", inherits = TRUE, ifnotfound = list(project_root = getwd()))
  resultsFolder <- file.path(analysis2_spec$project_root, "SavedResults", if (SimMode) "Sim" else "Real", paste0("Results_", AnalysisName))
  result_csv_files <- list.files(resultsFolder,
    pattern = if (SimMode) "^res.*[.]csv$" else "^predicted_df_out_.*[.]csv$",
    full.names = TRUE, recursive = TRUE)
  if (!length(result_csv_files)) stop("No mode-specific final-result CSV files were found in ", resultsFolder, ".")
  report <- utils::getFromNamespace("ndm_summarize_results", "ndm")(
    lapply(result_csv_files, data.table::fread),
    training_horizon = get0("report_training_horizon", inherits = TRUE, ifnotfound = NULL),
    experiment_id = paste(if (SimMode) "sim" else "real", AnalysisName, sep = ":"),
    output_dir = get0("SaveFolder", inherits = TRUE, ifnotfound = "./Figures/")
  )
  res_mat <- report$results
  paired_df <- report$paired
  paired_df_EndOfTraining <- report$final_pairs
  print(report$cohorts)
}
