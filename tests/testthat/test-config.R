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
      "neuralode_variational",
      "neuralode_kl_weight"
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
  expect_true(cfg$neuralode_variational)
  expect_equal(cfg$neuralode_kl_weight, 1)
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
    ndm_create_config(model_type = "NeuralODE", neuralode_kl_weight = -1),
    "non-negative"
  )
  expect_error(
    ndm_create_config(model_type = "NeuralODE", neuralode_kl_weight = Inf),
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
