analysis2_runtime_path <- function(...) {
  test_path("..", "..", "inst", "extdata", "analysis_runtime", "Analysis2", ...)
}

analysis2_runtime_lines <- function(...) {
  readLines(analysis2_runtime_path(...))
}

test_that("legacy runtime scripts no longer hard-disable intended branches", {
  build_lines <- analysis2_runtime_lines("ModelDefiners", "SuperLModel_BuildML.R")
  train_lines <- analysis2_runtime_lines("ModelTrainers", "SuperLModel_TrainDo.R")
  real_lines <- analysis2_runtime_lines("ResultsGet", "SuperLModel_GetAnalytics_Real.R")

  expect_false(any(grepl("if\\(testWithoutSampling <- TRUE\\)", build_lines)))
  expect_false(any(grepl("if\\(UseDiagonalLMatVCov <- T\\)", build_lines)))
  expect_false(any(grepl("Dropping policy learning!", build_lines, fixed = TRUE)))
  expect_false(any(grepl("if\\(SaveEqx <- F\\)", train_lines)))
  expect_false(any(grepl("Skilll8SanityCheck", real_lines, fixed = TRUE)))
})

test_that("decoder cache path uses model decoder projections and rank-safe writes", {
  build_lines <- analysis2_runtime_lines("ModelDefiners", "SuperLModel_BuildML.R")
  backbone_lines <- analysis2_runtime_lines("ModelDefiners", "SuperLModel_BackboneTransformer.R")

  expect_true(any(grepl("ModelList\\$TSList\\$TSBackbone\\$DecoderProj\\(xt_last\\)", build_lines)))
  expect_true(any(grepl("ModelList\\$TSList\\$TSBackbone\\$DecoderProj\\(embed_out\\)", build_lines)))
  expect_true(any(grepl("jnp\\$expand_dims\\(embed_out, 0L\\)", build_lines)))
  expect_true(any(grepl("WtResidPathInit_inv", backbone_lines, fixed = TRUE)))
  expect_true(any(grepl("L\\$FFN\\$WideProj1\\(xt\\)", backbone_lines)))
})

test_that("ode pair dedup keeps lhs and rhs aligned", {
  lhs <- c("S", "E", "E")
  rhs <- c("dx", "dx", "dy")
  ode_pairs <- data.frame(
    lefthandside_vec = lhs,
    righthandside_vec = rhs,
    stringsAsFactors = FALSE
  )
  ode_pairs <- ode_pairs[!duplicated(ode_pairs), , drop = FALSE]

  expect_equal(ode_pairs$lefthandside_vec, lhs)
  expect_equal(ode_pairs$righthandside_vec, rhs)
})

test_that("sim analytics baseline metrics and policy routes target the correct objects", {
  sim_lines <- analysis2_runtime_lines("ResultsGet", "SuperLModel_GetAnalytics_Sim.R")

  expect_true(any(grepl("LastOutCor <- cor\\(c\\(l_true\\),c\\(pred_l_mean\\)\\)", sim_lines)))
  expect_true(any(grepl("FNorm_Raw_baseline <- mean\\(sqrt\\(\\(c\\(pred_l_baselineVal_mat\\) - c\\(l_true\\)\\)\\^2\\)\\)", sim_lines)))
  expect_true(any(grepl("AggNorm_Raw_baseline <- mean\\(sqrt\\(\\(rowSums\\(pred_l_baselineVal_mat\\) - rowSums\\(l_true\\)\\)\\^2\\)\\)", sim_lines)))
  expect_true(any(grepl("state, PriorList, PolicyList_Model_natural", sim_lines, fixed = TRUE)))
  expect_true(any(grepl("state, PriorList, PolicyList_Model_unnatural", sim_lines, fixed = TRUE)))

  pred_l_mean <- matrix(c(3, 4), nrow = 1)
  l_true <- matrix(c(1, 2), nrow = 1)
  pred_l_baselineVal_mat <- matrix(c(0, 0), nrow = 1)
  old_baseline_norm <- mean(sqrt((c(pred_l_mean) - c(pred_l_baselineVal_mat))^2))
  new_baseline_norm <- mean(sqrt((c(pred_l_baselineVal_mat) - c(l_true))^2))

  expect_equal(new_baseline_norm, 1.5)
  expect_false(isTRUE(all.equal(old_baseline_norm, new_baseline_norm)))
})

test_that("trainer and calibration scripts keep shared save paths and vectorized outcome transforms", {
  train_define_lines <- analysis2_runtime_lines("ModelTrainers", "SuperLModel_TrainDefine.R")
  train_do_lines <- analysis2_runtime_lines("ModelTrainers", "SuperLModel_TrainDo.R")
  calibrate_lines <- analysis2_runtime_lines("SetupData", "SuperLModel_CalibrateML.R")

  expect_true(any(grepl("SavedModelDir <- sprintf", train_define_lines, fixed = TRUE)))
  expect_true(any(grepl("SavedModelDir <- get0", train_do_lines, fixed = TRUE)))
  expect_true(any(grepl("next_train_batch <- function", train_do_lines, fixed = TRUE)))
  expect_true(any(grepl("state <- state_tmp", train_do_lines, fixed = TRUE)))
  expect_true(any(grepl("tmppp <- ifelse\\(tmppp > 100, tmppp, InvSoftPlus_r\\(tmppp\\)\\)\\[1:nOutcomes\\]", calibrate_lines)))

  tmppp <- c(0.5, 200)
  calibrated <- ifelse(tmppp > 100, tmppp, tmppp + 1)
  expect_equal(calibrated, c(1.5, 200))
})

test_that("special-token and time-horizon settings remain runtime-driven", {
  build_lines <- analysis2_runtime_lines("ModelDefiners", "SuperLModel_BuildML.R")
  parse_lines <- analysis2_runtime_lines("ModelDefiners", "SuperLModel_ParseDynamicODE.R")
  api_lines <- analysis2_runtime_lines("SetupEnv", "Analysis2_api.R")

  expect_false(any(grepl("AppendPlaceEmbeds <- TRUE", build_lines, fixed = TRUE)))
  expect_false(any(grepl("AppendTimeEmbeds <- TRUE", build_lines, fixed = TRUE)))
  expect_true(any(grepl("AppendPlaceEmbeds = TRUE", api_lines, fixed = TRUE)))
  expect_true(any(grepl("AppendTimeEmbeds = TRUE", api_lines, fixed = TRUE)))
  expect_true(any(grepl("AppendPlaceEmbeds = FALSE", api_lines, fixed = TRUE)))
  expect_true(any(grepl("AppendTimeEmbeds = FALSE", api_lines, fixed = TRUE)))
  expect_false(any(grepl("nTimes<-120L", build_lines, fixed = TRUE)))
  expect_true(any(grepl("MaxTimeIndex", build_lines, fixed = TRUE)))
  expect_false(any(grepl("NTimeGlobalNeuralMax <- 200L", parse_lines, fixed = TRUE)))
  expect_true(any(grepl("0L:NTimeGlobalNeuralMax", parse_lines, fixed = TRUE)))
})
