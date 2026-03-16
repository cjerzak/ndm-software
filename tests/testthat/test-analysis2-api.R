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
  expect_false("ndm_run_analysis2_multidisease" %in% exports)
})

test_that("run config helpers validate mutually exclusive grid inputs and outer rows", {
  project_root <- tempdir()

  expect_error(
    ndm:::.ndm_make_run_config(
      mode = "real",
      project_root = project_root,
      analysis_name = "Demo",
      grid = data.frame(BaseID = 1L),
      grid_file = "grid.csv",
      outer = 1L
    ),
    "Supply either `grid` or `grid_file`, not both"
  )

  expect_error(
    ndm:::.ndm_make_run_config(
      mode = "real",
      project_root = project_root,
      analysis_name = "Demo",
      outer = integer()
    ),
    "`outer` must contain at least one integer row index"
  )
})

test_that("multidisease run configs reject retired TFRecord regeneration", {
  expect_error(
    ndm_create_multidisease_run_config(
      project_root = tempdir(),
      resave_tfrecords = TRUE
    ),
    "no longer supported for multidisease workflows"
  )
})

test_that("dry-run previews support both in-memory and file-backed grids", {
  project_root <- tempdir()
  grid_path <- tempfile(fileext = ".csv")
  grid <- ndm_test_dry_run_grid()
  utils::write.csv(grid, grid_path, row.names = FALSE)

  from_memory <- ndm_run_sim(
    ndm_create_sim_run_config(
      project_root = project_root,
      analysis_name = "GridPreview",
      grid = grid,
      outer = 1:2,
      dry_run = TRUE
    )
  )
  from_file <- ndm_run_sim(
    ndm_create_sim_run_config(
      project_root = project_root,
      analysis_name = "GridPreview",
      grid_file = grid_path,
      outer = 1:2,
      dry_run = TRUE
    )
  )

  expect_equal(from_memory$grid_rows, 2L)
  expect_equal(from_file$grid_rows, 2L)
  expect_equal(from_memory$outer_rows, c(1L, 2L))
  expect_equal(from_file$outer_rows, c(1L, 2L))
  expect_equal(from_memory$grid_preview$BaseID, from_file$grid_preview$BaseID)
  expect_equal(from_memory$grid_preview$ModelType, from_file$grid_preview$ModelType)
})

test_that("package-native run configs bypass YAML manifests", {
  preview <- ndm_run_real(
    ndm_create_real_run_config(
      project_root = tempdir(),
      grid = ndm_test_dry_run_grid(),
      outer = 1:2,
      dry_run = TRUE
    )
  )

  expect_equal(preview$run_spec$mode, "real")
  expect_equal(preview$grid_rows, 2L)
  expect_false("config_file" %in% names(preview$run_spec))
})
