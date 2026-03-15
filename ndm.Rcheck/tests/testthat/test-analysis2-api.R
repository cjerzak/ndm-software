ndm_test_dry_run_grid <- function() {
  data.frame(
    BaseID = c(1L, 2L),
    ModelType = c("DecoderOnly", "NeuralODE"),
    stringsAsFactors = FALSE
  )
}

ndm_parse_cli_args <- function(args) {
  stats::setNames(
    sub("^[^=]+=", "", args),
    sub("^--([^=]+)=.*$", "\\1", args)
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

test_that("run config helpers validate inputs and normalize CLI arguments", {
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

  config <- ndm_create_real_run_config(
    project_root = project_root,
    analysis_name = "Demo",
    grid_file = file.path("Data", "RunGrids", "Real.csv"),
    outer = c(1L, 2L),
    tfrecord_dir = "TFRecords",
    raw_data_dir = "RawData",
    outcome_metric = "inc_case",
    data_subset = "all",
    dry_run = TRUE
  )
  parsed <- ndm_parse_cli_args(ndm:::.ndm_run_config_to_args(config))

  expect_equal(parsed[["project_root"]], normalizePath(project_root, winslash = "/", mustWork = TRUE))
  expect_equal(parsed[["analysis_name"]], "Demo")
  expect_equal(parsed[["outer"]], "1,2")
  expect_equal(parsed[["grid_file"]], file.path(parsed[["project_root"]], "Data", "RunGrids", "Real.csv"))
  expect_equal(parsed[["tfrecord_dir"]], file.path(parsed[["project_root"]], "TFRecords"))
  expect_equal(parsed[["raw_data_dir"]], file.path(parsed[["project_root"]], "RawData"))
  expect_equal(parsed[["dry_run"]], "TRUE")
  expect_equal(parsed[["run_figures"]], "FALSE")
})

test_that("in-memory grids are materialized before calling Analysis2", {
  config <- ndm_create_sim_run_config(
    project_root = tempdir(),
    analysis_name = "GridPreview",
    grid = ndm_test_dry_run_grid(),
    outer = 1:2,
    dry_run = TRUE
  )

  parsed <- ndm_parse_cli_args(ndm:::.ndm_run_config_to_args(config))

  expect_true(file.exists(parsed[["grid_file"]]))
  grid <- utils::read.csv(parsed[["grid_file"]], stringsAsFactors = FALSE)
  expect_equal(nrow(grid), 2L)
  expect_equal(grid$ModelType, c("DecoderOnly", "NeuralODE"))
})

test_that("Analysis2 entrypoints fail early when yaml is unavailable", {
  local_mocked_bindings(
    .ndm_namespace_available = function(package) !identical(package, "yaml"),
    .package = "ndm"
  )

  expect_error(
    ndm_run_real(
      ndm_create_real_run_config(
        project_root = tempdir(),
        grid = ndm_test_dry_run_grid(),
        dry_run = TRUE
      )
    ),
    "yaml"
  )
})
