test_that("Analysis2 package runners are immediate deprecation stubs", {
  expect_error(
    ndm_run_analysis2_real("--dry_run=TRUE"),
    regexp = "deprecated.*Analysis2/SuperLModel_MasterReal\\.R",
    fixed = FALSE
  )

  expect_error(
    ndm_run_analysis2_sim("--dry_run=TRUE"),
    regexp = "deprecated.*Analysis2/SuperLModel_MasterSim\\.R",
    fixed = FALSE
  )
})
