test_that("configuration objects preserve requested modeling defaults", {
  cfg <- ndm_create_config(
    model_type = "DecoderOnly",
    backbone = "transformer"
  )

  expect_s3_class(cfg, "ndm_config")
  expect_equal(cfg$model_type, "DecoderOnly")
  expect_equal(cfg$backbone, "transformer")
  expect_setequal(
    names(cfg),
    c(
      "model_type",
      "backbone",
      "float_type",
      "force_to_gpu",
      "resave_tfrecords",
      "gpu_mem_frac",
      "neuralode_local_latent_dim",
      "neuralode_global_latent_dim",
      "neuralode_init_state_logit_offset",
      "neuralode_init_state_logit_scale_max",
      "neuralode_optim_solver",
      "neuralode_optim_dt0",
      "neuralode_optim_controller",
      "neuralode_optim_rtol",
      "neuralode_optim_atol",
      "enable_kv_cache",
      "inference_mc_draws",
      "observation_scale_floor",
      "initial_observation_scale",
      "neuralode_variational",
      "neuralode_kl_weight",
      "neuralode_mean_loss_weight"
    )
  )
  expect_null(cfg$neuralode_local_latent_dim)
  expect_null(cfg$neuralode_global_latent_dim)
  expect_null(cfg$neuralode_init_state_logit_offset)
  expect_equal(cfg$neuralode_init_state_logit_scale_max, Inf)
  expect_equal(cfg$neuralode_optim_solver, "tsit5")
  expect_equal(cfg$neuralode_optim_dt0, 1e-3)
  expect_equal(cfg$neuralode_optim_controller, "pid")
  expect_equal(cfg$neuralode_optim_rtol, 1e-5)
  expect_equal(cfg$neuralode_optim_atol, 1e-7)
  expect_false(cfg$enable_kv_cache)
  expect_identical(cfg$inference_mc_draws, 5L)
  expect_equal(cfg$observation_scale_floor, 1e-5)
  expect_equal(cfg$initial_observation_scale, 1)
  expect_true(cfg$neuralode_variational)
  expect_equal(cfg$neuralode_kl_weight, 1)
  expect_equal(cfg$neuralode_mean_loss_weight, 0)
})

test_that("public config constructors share the unit observation-scale default", {
  configs <- list(
    model = ndm_create_config(),
    real = ndm_create_real_run_config(project_root = tempdir()),
    sim = ndm_create_sim_run_config(project_root = tempdir()),
    multidisease = ndm_create_multidisease_run_config(project_root = tempdir())
  )

  expect_equal(
    unname(vapply(configs, `[[`, numeric(1L), "initial_observation_scale")),
    rep(1, length(configs))
  )
})

test_that("NeuralODE remains a supported model type", {
  cfg <- ndm_create_config(
    model_type = "NeuralODE",
    backbone = "transformer"
  )

  expect_equal(cfg$model_type, "NeuralODE")
})

test_that("NeuralODE variational config validates KL weight", {
  cfg <- ndm_create_config(
    model_type = "NeuralODE",
    neuralode_variational = TRUE,
    neuralode_kl_weight = 0.25
  )

  expect_true(cfg$neuralode_variational)
  expect_equal(cfg$neuralode_kl_weight, 0.25)
  expect_error(
    ndm_create_config(model_type = "NeuralODE", neuralode_variational = NA),
    "non-missing logical"
  )
  expect_error(
    ndm_create_config(model_type = "NeuralODE", neuralode_kl_weight = -1),
    "non-negative"
  )
  expect_error(
    ndm_create_config(model_type = "NeuralODE", neuralode_kl_weight = Inf),
    "finite"
  )
})

test_that("inference and observation-scale controls validate their domains", {
  cfg <- ndm_create_config(
    enable_kv_cache = TRUE,
    inference_mc_draws = 7L,
    observation_scale_floor = 2e-5,
    initial_observation_scale = 0.02
  )

  expect_true(cfg$enable_kv_cache)
  expect_identical(cfg$inference_mc_draws, 7L)
  expect_equal(cfg$observation_scale_floor, 2e-5)
  expect_equal(cfg$initial_observation_scale, 0.02)
  expect_error(ndm_create_config(enable_kv_cache = NA), "non-missing logical")
  expect_error(ndm_create_config(inference_mc_draws = 0L), "positive integer")
  expect_error(ndm_create_config(inference_mc_draws = 1.5), "positive integer")
  expect_error(ndm_create_config(observation_scale_floor = 0), "finite positive")
  expect_error(ndm_create_config(initial_observation_scale = Inf), "finite positive")
  expect_error(
    ndm_create_config(
      observation_scale_floor = 0.01,
      initial_observation_scale = 0.01
    ),
    "greater than `observation_scale_floor`"
  )
  expect_error(
    ndm_create_config(
      observation_scale_floor = 0.02,
      initial_observation_scale = 0.01
    ),
    "greater than `observation_scale_floor`"
  )
})

test_that("NeuralODE auxiliary mean-loss weight is finite and non-negative", {
  expect_equal(
    ndm_create_config(
      model_type = "NeuralODE",
      neuralode_mean_loss_weight = 0.3
    )$neuralode_mean_loss_weight,
    0.3
  )
  expect_error(
    ndm_create_config(neuralode_mean_loss_weight = -1),
    "non-negative"
  )
  expect_error(
    ndm_create_config(neuralode_mean_loss_weight = Inf),
    "finite"
  )
})

test_that("default configs do not expose runtime-root fields", {
  cfg <- ndm_create_config()

  expect_false("analysis_root" %in% names(cfg))
})

test_that("backend resource controls reject ambiguous values", {
  expect_error(ndm_create_config(force_to_gpu = NA), "non-missing logical")
  expect_error(ndm_create_config(gpu_mem_frac = 0), "in \\(0, 1\\]")
  expect_error(ndm_create_config(gpu_mem_frac = Inf), "finite")
  expect_equal(ndm_create_config(gpu_mem_frac = 0.25)$gpu_mem_frac, 0.25)
})

test_that("generic configs reject inline TFRecord regeneration", {
  expect_error(
    ndm_create_config(resave_tfrecords = TRUE),
    "ndm_bootstrap_real_tfrecords"
  )
  expect_error(
    ndm_create_config(resave_tfrecords = 1L),
    "ndm_bootstrap_real_tfrecords"
  )
})
