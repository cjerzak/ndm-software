analytics_metric_source_text <- function(relative_path) {
  ndm_test_runtime_source_text(relative_path)
}

analytics_metric_helper_env <- function() {
  helper_env <- new.env(parent = globalenv())
  invisible(capture.output(eval(
    ndm_test_runtime_source_expressions(
      "SetupEnv/SuperLModel_helperFxns.R"
    ),
    envir = helper_env
  )))
  helper_env
}

test_that("simulation skill uses raw errors on a common observation set", {
  helper_env <- analytics_metric_helper_env()

  n <- 1000L
  n_bad <- 20L
  ordinary_error <- 0.9
  baseline_error <- rep(1, n)
  target_skill <- -1.88
  target_rmse <- (1 - target_skill) * (0.001 + 1) - 0.001
  outlier_error <- sqrt(
    (n * target_rmse^2 - (n - n_bad) * ordinary_error^2) / n_bad
  )
  model_error <- c(
    rep(ordinary_error, n - n_bad),
    rep(outlier_error, n_bad)
  )

  corrected <- helper_env$ndm_paired_rmse_skill(
    prediction = model_error,
    baseline = baseline_error,
    truth = numeric(n)
  )
  separately_clipped_skill <- 1 -
    (0.001 + sqrt(mean(helper_env$clipAt(model_error^2)))) /
      (0.001 + sqrt(mean(helper_env$clipAt(baseline_error^2))))

  expect_equal(unname(corrected[["skill"]]), target_skill, tolerance = 1e-12)
  expect_lt(corrected[["skill"]], 0)
  expect_gt(separately_clipped_skill, 0)

  paired <- helper_env$ndm_paired_rmse_skill(
    prediction = c(1, NA, 100),
    baseline = c(2, 100, NA),
    truth = c(0, 0, 0)
  )
  expect_equal(unname(paired[["rss_pred"]]), 1)
  expect_equal(unname(paired[["rss_baseline"]]), 4)
})

test_that("real aggregation compares paired raw squared errors", {
  helper_env <- analytics_metric_helper_env()
  real_preprocess_text <- analytics_metric_source_text(
    "ResultsAnalyze/SuperLModel_GenFigs_RealPreprocess.R"
  )

  corrected <- helper_env$ndm_paired_squared_error_skill(
    squared_prediction_error = c(1, NA, 100),
    squared_baseline_error = c(4, 100, NA)
  )

  expect_equal(unname(corrected[["rss_pred"]]), 1)
  expect_equal(unname(corrected[["rss_baseline"]]), 4)
  expect_match(
    real_preprocess_text,
    "ndm_paired_squared_error_skill",
    fixed = TRUE
  )
  expect_false(grepl("clipAt", real_preprocess_text, fixed = TRUE))
})

test_that("figure summaries retain the full skill distribution", {
  helper_env <- analytics_metric_helper_env()
  helper_text <- analytics_metric_source_text(
    "SetupEnv/SuperLModel_helperFxns.R"
  )
  figure_text <- analytics_metric_source_text(
    "ResultsAnalyze/SuperLModel_GenFigs.R"
  )

  set.seed(7)
  skill_sample <- c(stats::rnorm(98, 0.10, 0.05), -40, -120)

  expect_equal(helper_env$mean2(skill_sample), mean(skill_sample))
  expect_lt(helper_env$mean2(skill_sample), 0)
  expect_false(grepl("clippedMean", helper_text, fixed = TRUE))
  expect_false(grepl("clippedMean", figure_text, fixed = TRUE))
})

test_that("constrained skill transform is an order-preserving inverse", {
  helper_env <- analytics_metric_helper_env()
  skills <- c(-3, -1, -0.5, 0, 0.1, 0.5, 0.9, 1)

  unconstrained <- helper_env$ndm_skill_to_unconstrained(skills)
  round_trip <- helper_env$ndm_skill_from_unconstrained(unconstrained)

  expect_equal(round_trip, skills, tolerance = 1e-12)
  expect_true(all(diff(unconstrained) > 0))
  expect_true(all(diff(round_trip) > 0))
})

test_that("simulation and figure analytics use the corrected helpers", {
  simulation_text <- analytics_metric_source_text(
    "ResultsGet/SuperLModel_GetAnalytics_Sim.R"
  )
  figure_text <- analytics_metric_source_text(
    "ResultsAnalyze/SuperLModel_GenFigs.R"
  )

  expect_match(simulation_text, "ndm_paired_rmse_skill", fixed = TRUE)
  expect_false(grepl("mean(clipAt", simulation_text, fixed = TRUE))
  expect_match(
    figure_text,
    "Map2ConstrainedSkill <- ndm_skill_from_unconstrained",
    fixed = TRUE
  )
  expect_match(
    figure_text,
    "InvMap2ConstrainedSkill <- ndm_skill_to_unconstrained",
    fixed = TRUE
  )
})
