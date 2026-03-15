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
      "gpu_mem_frac"
    )
  )
})

test_that("NeuralODE remains a supported model type", {
  cfg <- ndm_create_config(
    model_type = "NeuralODE",
    backbone = "transformer"
  )

  expect_equal(cfg$model_type, "NeuralODE")
})

test_that("default configs do not expose runtime bundle paths", {
  cfg <- ndm_create_config()

  expect_false("analysis_root" %in% names(cfg))
})
