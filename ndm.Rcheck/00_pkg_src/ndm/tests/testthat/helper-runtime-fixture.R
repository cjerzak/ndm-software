make_runtime_fixture <- function() {
  root <- tempfile("ndm-runtime-")
  analysis_root <- file.path(root, "Analysis2")

  dirs <- c(
    file.path(analysis_root, "SetupEnv"),
    file.path(analysis_root, "SetupData"),
    file.path(analysis_root, "ModelDefiners"),
    file.path(analysis_root, "ModelTrainers"),
    file.path(analysis_root, "ResultsGet"),
    file.path(analysis_root, "ResultsAnalyze"),
    file.path(analysis_root, "ModelStructureTex")
  )
  vapply(dirs, dir.create, logical(1), recursive = TRUE, showWarnings = FALSE)

  writeLines(
    c(
      "fixture_helper_loaded <- TRUE"
    ),
    file.path(analysis_root, "SetupEnv", "SuperLModel_helperFxns.R")
  )

  writeLines(
    c(
      "fixture_relative_loaded <- TRUE"
    ),
    file.path(analysis_root, "SetupEnv", "relative_helper.R")
  )

  writeLines(
    c(
      "fixture_backend_loaded <- TRUE",
      "fixture_backend_settings <- list(",
      "  floatType = get0(\"floatType\", inherits = TRUE, ifnotfound = NULL),",
      "  force2GPU = get0(\"force2GPU\", inherits = TRUE, ifnotfound = NULL),",
      "  GPU_MEM_FRAC = get0(\"GPU_MEM_FRAC\", inherits = TRUE, ifnotfound = NULL),",
      "  ReSaveTfRecords = get0(\"ReSaveTfRecords\", inherits = TRUE, ifnotfound = NULL)",
      ")"
    ),
    file.path(analysis_root, "SetupEnv", "SuperLModel_MasterImports.R")
  )

  writeLines(
    c(
      "ndm_source_extracted(file.path(\"SetupEnv\", \"relative_helper.R\"))",
      "fixture_data_generator <- \"real\""
    ),
    file.path(analysis_root, "SetupData", "SuperLModel_DataGenerator_Real.R")
  )

  writeLines(
    c(
      "ndm_source_extracted(file.path(\"SetupEnv\", \"relative_helper.R\"))",
      "fixture_data_generator <- \"sim\""
    ),
    file.path(analysis_root, "SetupData", "SuperLModel_DataGenerator_Sim.R")
  )

  writeLines(
    c(
      "fixture_calibration_loaded <- TRUE"
    ),
    file.path(analysis_root, "SetupData", "SuperLModel_CalibrateML.R")
  )

  writeLines(
    c(
      "fixture_build_model_loaded <- TRUE",
      "fixture_build_model_analysis_root <- NDM_INTERNAL_ANALYSIS_DIR",
      "ModelList <- list(source = fixture_build_model_analysis_root)",
      "state <- list(stage = \"built\")",
      "PriorList <- list(source = fixture_build_model_analysis_root)",
      "PolicyList <- list(source = fixture_build_model_analysis_root)",
      "GetPredSaveAtInfo_default <- list(1L)",
      "GetPred_inference <- function(ModelList, x, state, PriorList, PolicyList, GetPredSaveAtInfo, seed) list(list(kind = \"inference\", batch = x), state)",
      "GetPred_train_jit <- function(ModelList, x, state, PriorList, PolicyList, GetPredSaveAtInfo, seed) list(list(kind = \"train\", batch = x), state)",
      "getLoss_train <- function(ModelList, x, y, y_mask, i, state, PriorList, PolicyList, GetPredSaveAtInfo, seed) list(list(kind = \"loss\", iteration = i), state)",
      "gradLoss_jax <- function(ModelList, x, y, y_mask, i, state, PriorList, PolicyList, GetPredSaveAtInfo, seed) list(list(list(0, state)), ModelList)"
    ),
    file.path(analysis_root, "ModelDefiners", "SuperLModel_BuildML.R")
  )

  writeLines(
    c(
      "fixture_train_define_loaded <- TRUE",
      "opt_state <- list(stage = \"defined\")"
    ),
    file.path(analysis_root, "ModelTrainers", "SuperLModel_TrainDefine.R")
  )

  writeLines(
    c(
      "fixture_train_do_loaded <- TRUE",
      "state$stage <- \"trained\"",
      "opt_state <- list(stage = \"trained\")"
    ),
    file.path(analysis_root, "ModelTrainers", "SuperLModel_TrainDo.R")
  )

  writeLines(
    c(
      "fixture_results_get_loaded <- TRUE"
    ),
    file.path(analysis_root, "ResultsGet", "SuperLModel_GetAnalytics.R")
  )

  writeLines(
    c(
      "fixture_results_analyze_loaded <- TRUE"
    ),
    file.path(analysis_root, "ResultsAnalyze", "SuperLModel_GenFigs.R")
  )

  writeLines(
    c("% fixture tex"),
    file.path(analysis_root, "ModelStructureTex", "fixture.tex")
  )

  list(
    root = root,
    analysis_root = normalizePath(analysis_root, winslash = "/", mustWork = TRUE)
  )
}

runtime_fixture_analysis_root <- function() {
  ndm:::.ndm_internal_analysis_root()
}
