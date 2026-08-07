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

test_that("run configs expose portable backend and NeuralODE KL controls", {
  config <- ndm_create_sim_run_config(
    project_root = tempdir(),
    grid = data.frame(BaseID = 1L, ModelType = "NeuralODE"),
    outer = 1L,
    gpu_mem_frac = 0.5,
    enable_kv_cache = TRUE,
    enable_kv_cache_training = TRUE,
    inference_mc_draws = 7L,
    observation_scale_floor = 2e-5,
    initial_observation_scale = 0.02,
    neuralode_variational = FALSE,
    neuralode_kl_weight = 0,
    neuralode_mean_loss_weight = 0.3,
    dry_run = TRUE
  )
  args <- ndm:::.ndm_run_config_to_args(config)

  expect_null(config$force_to_gpu)
  expect_identical(config$compute_backend, "auto")
  expect_true(config$respect_grid_model_type)
  expect_equal(config$gpu_mem_frac, 0.5)
  expect_true(config$enable_kv_cache)
  expect_true(config$enable_kv_cache_training)
  expect_identical(config$inference_mc_draws, 7L)
  expect_equal(config$observation_scale_floor, 2e-5)
  expect_equal(config$initial_observation_scale, 0.02)
  expect_false(config$neuralode_variational)
  expect_equal(config$neuralode_kl_weight, 0)
  expect_equal(config$neuralode_mean_loss_weight, 0.3)
  expect_true("--compute_backend=auto" %in% args)
  expect_false(any(grepl("^--force_to_gpu=", args)))
  expect_true("--gpu_mem_frac=0.5" %in% args)
  expect_true("--enable_kv_cache=TRUE" %in% args)
  expect_true("--enable_kv_cache_training=TRUE" %in% args)
  expect_true("--inference_mc_draws=7" %in% args)
  expect_true("--observation_scale_floor=2e-05" %in% args)
  expect_true("--initial_observation_scale=0.02" %in% args)
  expect_true("--neuralode_variational=FALSE" %in% args)
  expect_true("--neuralode_kl_weight=0" %in% args)
  expect_true("--neuralode_mean_loss_weight=0.3" %in% args)
})

test_that("run configs serialize deterministic scaled-MSE controls", {
  pinned_scales <- c(
    0.00218001845765384,
    0.0019327084723560455
  )
  config <- ndm_create_multidisease_run_config(
    project_root = tempdir(),
    grid = data.frame(BaseID = 2009L, ModelType = "NeuralODE"),
    training_objective = "scaled_mse",
    outcome_loss_scale = pinned_scales,
    neuralode_variational = FALSE,
    neuralode_kl_weight = 0,
    neuralode_mean_loss_weight = 0,
    inference_mc_draws = 1L,
    dry_run = TRUE
  )
  args <- ndm:::.ndm_run_config_to_args(config)

  expect_identical(config$training_objective, "scaled_mse")
  expect_identical(config$outcome_loss_scale, pinned_scales)
  expect_true("--training_objective=scaled_mse" %in% args)
  scale_arg <- sub(
    "^--outcome_loss_scale=", "",
    args[grepl("^--outcome_loss_scale=", args)]
  )
  serialized_scales <- strsplit(scale_arg, ",", fixed = TRUE)[[1L]]
  expect_identical(as.numeric(serialized_scales), pinned_scales)
})

test_that("runtime exports objective telemetry and a dynamic persistence skill", {
  train_source <- ndm_test_runtime_source_text(
    "ModelTrainers/SuperLModel_TrainDo.R"
  )
  analytics_source <- ndm_test_runtime_source_text(
    "ResultsGet/SuperLModel_GetAnalytics_Real.R"
  )

  for (field in c(
    "objective_data_loss", "raw_mse", "scaled_mse", "kl_local",
    "kl_global", "kl_place", "kl_weighted", "prediction_abs_mean",
    "truth_abs_mean"
  )) {
    expect_match(train_source, field, fixed = TRUE)
  }
  expect_match(analytics_source, "last_observed_context", fixed = TRUE)
  expect_match(analytics_source, "configured_horizon", fixed = TRUE)
  expect_match(analytics_source, "PredBase_l", fixed = TRUE)
  expect_false(grepl('sl_dat[,"Truth_l8"]', analytics_source, fixed = TRUE))
})

test_that("run configs preserve the legacy backend alias without ambiguity", {
  cpu_config <- suppressWarnings(ndm_create_sim_run_config(
    project_root = tempdir(),
    grid = data.frame(BaseID = 1L),
    force_to_gpu = FALSE,
    dry_run = TRUE
  ))
  expect_identical(cpu_config$compute_backend, "cpu")
  expect_false(cpu_config$force_to_gpu)
  expect_true("--compute_backend=cpu" %in% ndm:::.ndm_run_config_to_args(cpu_config))
  expect_false(any(grepl(
    "^--force_to_gpu=",
    ndm:::.ndm_run_config_to_args(cpu_config)
  )))

  gpu_config <- ndm_create_sim_run_config(
    project_root = tempdir(),
    grid = data.frame(BaseID = 1L),
    compute_backend = "gpu",
    dry_run = TRUE
  )
  expect_identical(gpu_config$compute_backend, "gpu")
  expect_true(gpu_config$force_to_gpu)
  expect_true("--compute_backend=gpu" %in% ndm:::.ndm_run_config_to_args(gpu_config))

  expect_error(
    suppressWarnings(ndm_create_sim_run_config(
      project_root = tempdir(),
      force_to_gpu = TRUE,
      compute_backend = "cpu"
    )),
    "Conflicting backend controls"
  )
  expect_error(
    ndm_create_sim_run_config(
      project_root = tempdir(),
      compute_backend = "cpu",
      gpu_mem_frac = 0.5
    ),
    "resolved compute backend is CPU"
  )
  mutated <- ndm_create_sim_run_config(
    project_root = tempdir(),
    gpu_mem_frac = 0.5,
    dry_run = TRUE
  )
  mutated$compute_backend <- "cpu"
  expect_error(
    ndm:::.ndm_run_config_to_args(mutated),
    "resolved compute backend is CPU"
  )
})

test_that("Analysis2 outer rows preserve frozen CSV order and stable row identity", {
  api <- ndm:::.ndm_new_run_impl_env()
  grid <- data.frame(
    row_id = c("fit-b", "fit-a"),
    BaseID = c(2L, 1L),
    ModelType = c("NeuralODE", "DecoderOnly"),
    stringsAsFactors = FALSE
  )

  expect_identical(api$analysis2_order_grid(grid, 1L), grid)
  expect_identical(api$analysis2_order_grid(grid, c(2L, 1L)), grid)
  expect_equal(
    api$analysis2_run_identity("sim", 1L, 2L, "NeuralODE", 99L, row_id = "fit-b"),
    api$analysis2_run_identity("sim", 999L, 2L, "NeuralODE", 99L, row_id = "fit-b")
  )
  grid$row_id[[2L]] <- "fit-b"
  expect_error(api$analysis2_order_grid(grid, 1L), "must be non-missing and unique")
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
  runtime_source <- ndm_test_runtime_source_text(
    "ModelTrainers/SuperLModel_TrainDo.R"
  )

  expect_match(
    runtime_source,
    "ndm_training_iteration_key <- function(iteration)",
    fixed = TRUE
  )
  expect_match(runtime_source, "ndm_runtime_seed_key(104729L)", fixed = TRUE)
  expect_match(runtime_source, "jax$random$fold_in(", fixed = TRUE)

  driver_source <- ndm_test_runtime_source_text(
    "SetupEnv/Analysis2_legacy_multidisease_driver.R"
  )
  expect_match(driver_source, "tf$random$set_seed(as.integer(SEED_))", fixed = TRUE)
  expect_match(driver_source, "np$random$seed(as.integer(SEED_))", fixed = TRUE)
})
