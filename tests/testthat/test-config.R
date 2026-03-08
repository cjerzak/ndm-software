test_that("configuration objects preserve requested modeling defaults", {
  cfg <- ndm_create_config(
    model_type = "DecoderOnly",
    backbone = "transformer",
    analysis_root = "/tmp/example"
  )

  expect_s3_class(cfg, "ndm_config")
  expect_equal(cfg$model_type, "DecoderOnly")
  expect_equal(cfg$backbone, "transformer")
  expect_equal(cfg$analysis_root, normalizePath("/tmp/example", winslash = "/", mustWork = FALSE))
  expect_setequal(
    names(cfg),
    c(
      "model_type",
      "backbone",
      "analysis_root",
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
    backbone = "transformer",
    analysis_root = "/tmp/example"
  )

  expect_equal(cfg$model_type, "NeuralODE")
})

test_that("default configs use the bundled internal runtime", {
  cfg <- ndm_create_config()

  expect_equal(cfg$analysis_root, ndm:::.ndm_internal_analysis_root())
})
