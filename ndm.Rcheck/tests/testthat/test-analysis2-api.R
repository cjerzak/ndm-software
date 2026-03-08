ndm_test_dry_run_grid <- function() {
  data.frame(
    BaseID = c(1L, 2L),
    ModelType = c("DecoderOnly", "NeuralODE"),
    stringsAsFactors = FALSE
  )
}

test_that("package-native run configs support self-contained dry runs", {
  real <- ndm_run_real(
    ndm_create_real_run_config(
      project_root = tempdir(),
      grid = ndm_test_dry_run_grid(),
      outer = 1:2,
      dry_run = TRUE
    )
  )
  sim <- ndm_run_sim(
    ndm_create_sim_run_config(
      project_root = tempdir(),
      grid = ndm_test_dry_run_grid(),
      dry_run = TRUE
    )
  )
  multidisease <- ndm_run_multidisease(
    ndm_create_multidisease_run_config(
      project_root = tempdir(),
      grid = ndm_test_dry_run_grid(),
      dry_run = TRUE
    )
  )

  expect_equal(real$run_spec$mode, "real")
  expect_equal(sim$run_spec$mode, "sim")
  expect_equal(multidisease$run_spec$mode, "multidisease")
  expect_equal(real$grid_rows, 2L)
  expect_equal(sim$grid_rows, 2L)
  expect_equal(multidisease$grid_rows, 2L)
})

test_that("legacy Analysis2-branded entrypoints are not exported", {
  exports <- getNamespaceExports("ndm")

  expect_false("ndm_run_analysis2_real" %in% exports)
  expect_false("ndm_run_analysis2_sim" %in% exports)
})
