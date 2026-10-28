report_fixture <- function(h = 3L, evaluation = h) {
  data <- data.frame(ExperimentID = "e1", ReplicateID = "1", pair_id = "p1",
    ModelType = rep(c("DecoderOnly", "NeuralODE"), each = 2L),
    RunID = rep(c("d", "n"), each = 2L), RunSeed = c(11L, 11L, 29L, 29L),
    i_in_sgd = rep(c(10L, 20L), 2L), nSGD = 20L, training_horizon = h,
    evaluation_horizon = evaluation, nParamsTrainable = 100, nParamsTotal = 120,
    nParamsModel = 120, nSamplesTrain = 1000)
  for (i in seq_len(evaluation)) {
    data[[paste0("SkillTime", i)]] <- c(.4, .5, .1, .7)
    data[[paste0("RSSPredTime", i)]] <- c(.6, .5, .9, .3)
    data[[paste0("RSSBaselineTime", i)]] <- 1
  }
  data
}

test_that("reports match checkpoints and require both final budgets", {
  data <- report_fixture()
  report <- ndm_summarize_results(data)
  expect_equal(nrow(report$paired), 2L)
  expect_equal(nrow(report$final_pairs), 1L)
  expect_equal(report$cohorts$neural_win_fraction, 1)
  expect_equal(report$paired$i_in_sgd_Decoder, report$paired$i_in_sgd_NeuralODE)
  expect_equal(report$results$UpdatesPerParam, data$i_in_sgd / 100)
  data$nSGD[data$ModelType == "NeuralODE"] <- 30L
  expect_equal(nrow(ndm_summarize_results(data)$final_pairs), 0L)
  expect_error(ndm_summarize_results(rbind(data, data[1L, ])), "duplicate")
  data2 <- data; data2$ReplicateID <- "2"
  expect_equal(nrow(ndm_summarize_results(rbind(data, data2))$paired), 4L)
  data2$ReplicateID <- "1"; data2$ExperimentID <- "e2"
  expect_equal(nrow(ndm_summarize_results(rbind(data, data2))$paired), 4L)
  expect_equal(nrow(ndm_summarize_results(data[-4L, ])$unmatched), 1L)
})

test_that("horizon windows and output files are driven by explicit metadata", {
  for (h in c(1L, 3L, 12L, 50L)) {
    data <- report_fixture(h)
    for (i in seq_len(h)) data[[paste0("SkillTime", i)]] <- i
    report <- ndm_summarize_results(data[, rev(names(data))])
    window <- seq.int(max(1L, round(.75 * h)), h)
    expect_equal(report$results$SkillSummary, rep(mean(window), 4L))
    expect_equal(report$results$SkillSummaryCount, rep(length(window), 4L))
  }
  data <- report_fixture(12L, 50L); data$training_horizon <- NULL
  expect_error(ndm_summarize_results(data), "explicit training_horizon")
  expect_equal(ndm_summarize_results(data, training_horizon = 12L)$results$SkillSummaryCount, rep(4, 4))
  data <- report_fixture(); data$SkillTime2 <- NULL
  expect_error(ndm_summarize_results(data), "structural")
  data <- report_fixture(); data$Pred_l1 <- NULL; data$RSSPredTime2 <- NULL
  expect_error(ndm_summarize_results(data), "Misaligned")
  data <- report_fixture(); data[, paste0("SkillTime", 1:3)] <- NA_real_
  report <- ndm_summarize_results(data)
  expect_true(all(is.na(report$results$SkillSummary)))
  expect_false(any(is.nan(report$results$SkillSummary)))
  expect_equal(report$results$SkillSummaryCount, rep(0, 4))
  output <- tempfile(); on.exit(unlink(output, recursive = TRUE), add = TRUE)
  mixed <- ndm_summarize_results(list(report_fixture(1L), report_fixture(3L)), output_dir = output)
  expect_equal(nrow(mixed$cohorts), 2L)
  expect_equal(length(list.files(output, pattern = "[.]pdf$")), 2L)
  expect_true(all(file.info(list.files(output, full.names = TRUE))$size > 0L))
  incomplete <- report_fixture(1L)[-4L, ]
  report <- ndm_summarize_results(incomplete, output_dir = output)
  expect_equal(nrow(report$final_pairs), 0L)
  expect_gt(file.info(file.path(output, "forecast_train1_eval1.pdf"))$size, 0L)
})

test_that("real results align numeric suffixes and aggregate distinct run identities", {
  data <- report_fixture(12L)
  data <- data[rep(seq_len(4L), each = 2L), ]
  data$location_id <- rep(c("A", "B"), 4L); data$time_anchor_id <- 0L
  data[, grep("^(SkillTime|RSSPredTime|RSSBaselineTime)", names(data))] <- NULL
  data$PreviousSequenceTruth_l1 <- 0
  data$PreviousSequenceTruth_l2 <- 2
  data$PreviousSequenceTruth_l10 <- 10
  for (h in c(1:9, 11:12, 10L)) {
    data[[paste0("Truth_l", h)]] <- h
    data[[paste0("Pred_l", h)]] <- h + 1
  }
  report <- ndm_summarize_results(data[, sort(names(data))])
  expect_equal(nrow(report$results), 4L)
  expect_equal(report$results$RSSBaselineTime3, rep(49, 4))
  expect_equal(report$results$RSSPredTime10, rep(1, 4))
  expect_equal(report$results$ValidExamplesTime12, rep(2, 4))
  expect_error(ndm_summarize_results(rbind(data, data[1, ])), "Duplicate real-result")
  conflicting <- data; conflicting$nSGD[1] <- 21L
  expect_error(ndm_summarize_results(conflicting), "Conflicting metadata")
})
