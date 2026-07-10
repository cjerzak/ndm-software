test_that("run_seed is preserved and emitted independently of the grid row", {
  config <- ndm_create_multidisease_run_config(
    project_root = tempdir(),
    grid = data.frame(BaseID = 2007L),
    outer = 1L,
    run_seed = 202L,
    dry_run = TRUE
  )

  expect_identical(config$run_seed, 202L)
  expect_true("--run_seed=202" %in% ndm:::.ndm_run_config_to_args(config))
})

test_that("run_seed rejects values that are not non-negative integers", {
  for (bad_seed in list(-1L, 1.5, NA_integer_, Inf, "not-a-seed")) {
    expect_error(
      ndm_create_multidisease_run_config(
        project_root = tempdir(),
        grid = data.frame(BaseID = 2007L),
        outer = 1L,
        run_seed = bad_seed,
        dry_run = TRUE
      ),
      "non-negative integer"
    )
  }
})

test_that("explicit multidisease origins train only on earlier observations", {
  split <- ndm:::.ndm_multidisease_time_split(
    time_ids = 0:23,
    evaluation_time = 3L,
    evaluation_seq = 1:4,
    evaluation_origin_time_id = 13L
  )

  expect_identical(split$times_in, 0:12)
  expect_identical(split$times_out, 13L)
  expect_identical(split$in_out_cutpoint, 12L)
  expect_true(split$explicit_origin)
})

test_that("explicit multidisease origins must exist and cannot be time zero", {
  expect_error(
    ndm:::.ndm_multidisease_time_split(0:23, 3L, 1:4, 24L),
    "available non-initial"
  )
  expect_error(
    ndm:::.ndm_multidisease_time_split(0:23, 3L, 1:4, 0L),
    "available non-initial"
  )
})

test_that("training Monte Carlo keys incorporate the configured run seed", {
  runtime_source <- ndm:::.ndm_embedded_runtime_sources[[
    "ModelTrainers/SuperLModel_TrainDo.R"
  ]]

  expect_match(runtime_source, "iteration_seed <- as.integer", fixed = TRUE)
  expect_match(runtime_source, "get0(\"SEED_\"", fixed = TRUE)
  expect_match(runtime_source, "JaxKey(iteration_seed)", fixed = TRUE)

  driver_source <- ndm:::.ndm_embedded_runtime_sources[[
    "SetupEnv/Analysis2_legacy_multidisease_driver.R"
  ]]
  expect_match(driver_source, "tf$random$set_seed(as.integer(SEED_))", fixed = TRUE)
  expect_match(driver_source, "np$random$seed(as.integer(SEED_))", fixed = TRUE)
})
