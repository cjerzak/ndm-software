test_that("built-in model spec presets are available", {
  presets <- ndm_model_spec_presets()

  expect_equal(nrow(presets), 5L)
  expect_true(all(c(
    "seir_fixed",
    "seirs_dynamic_beta",
    "seirs_dynamic_beta_multi_outcome",
    "seirs_dynamic_beta_dynamic_global",
    "seirs_dynamic_beta_dynamic_global_multi_outcome"
  ) %in% presets$preset))
})

test_that("preset selection respects the requested model type", {
  spec <- ndm_model_spec(
    preset = "seirs_dynamic_beta",
    model_type = "NeuralODE"
  )

  expect_s3_class(spec, "ndm_model_spec")
  expect_equal(spec$model_type, "NeuralODE")
  expect_equal(spec$preset, "seirs_dynamic_beta")
  expect_true(file.exists(spec$source_path))
  expect_match(spec$tex_text, "Bayes")
})

test_that("built-in specs can be imported and exported", {
  spec <- ndm_model_spec(preset = "seirs_dynamic_beta_dynamic_global")
  exported <- ndm_model_spec_to_tex(spec)
  tmp <- tempfile(fileext = ".tex")
  on.exit(unlink(tmp), add = TRUE)

  path <- ndm_model_spec_to_tex(spec, path = tmp)
  roundtrip <- ndm_model_spec_from_tex(spec$source_path)

  expect_true(is.character(exported))
  expect_true(file.exists(path))
  expect_equal(roundtrip$preset, spec$preset)
  expect_equal(roundtrip$tex_text, spec$tex_text)
})
