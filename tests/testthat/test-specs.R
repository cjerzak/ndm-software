test_that("built-in model spec presets are available", {
  presets <- ndm_model_spec_presets()

  expect_equal(nrow(presets), 17L)
  expect_true(all(c(
    "seir_fixed",
    "seirs_dynamic_beta",
    "seirs_dynamic_beta_multi_outcome",
    "seirs_dynamic_beta_dynamic_global",
    "seirs_dynamic_beta_dynamic_global_multi_outcome",
    sprintf("tb_%s", letters[1:12])
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

test_that("structured TB presets expose execution metadata and family arguments", {
  spec <- ndm_model_spec(
    preset = "tb_b",
    model_type = "NeuralODE",
    family_args = list(n = 5L)
  )

  expect_s3_class(spec, "ndm_model_spec")
  expect_equal(spec$preset, "tb_b")
  expect_equal(spec$compartments, "tb")
  expect_equal(spec$family, "tb_b")
  expect_equal(spec$family_args$n, 5L)
  expect_equal(spec$source_format, "structured")
  expect_equal(length(spec$state_terms), 8L)
  expect_true(all(c("s_l", "lf1_l", "lf5_l", "ls_l", "i_l") %in% spec$state_terms))
  expect_equal(spec$init_state_terms, spec$state_terms)
  expect_true("c" %in% spec$parameter_terms)
  expect_true(all(sprintf("k%s", 1:5) %in% spec$parameter_terms))
  expect_true(all(sprintf("d%s", 1:5) %in% spec$parameter_terms))
})

test_that("structured model specs roundtrip through TeX metadata", {
  spec <- ndm_model_spec(
    preset = "tb_k",
    model_type = "NeuralODE"
  )
  tex_path <- tempfile(fileext = ".tex")
  on.exit(unlink(tex_path), add = TRUE)

  ndm_model_spec_to_tex(spec, path = tex_path)
  roundtrip <- ndm_model_spec_from_tex(tex_path, model_type = "NeuralODE")

  expect_equal(roundtrip$preset, "tb_k")
  expect_equal(roundtrip$family, "tb_k")
  expect_equal(roundtrip$state_terms, spec$state_terms)
  expect_equal(roundtrip$init_state_terms, spec$init_state_terms)
  expect_equal(roundtrip$parameter_terms, spec$parameter_terms)
  expect_equal(roundtrip$time_varying_terms, "c_t")
  expect_match(roundtrip$tex_text, "max\\(1.0, t\\)")
})

test_that("structured model declarations can be built directly", {
  spec <- ndm_model_spec_from_structure(
    list(
      preset = "toy_sl_i",
      description = "Toy two-state structure.",
      states = c("s_l", "i_l"),
      parameters = list(lambda = ndm:::.ndm_param_spec("InvSoftPlus", prior_mean = 0.1, prior_sd = 0.2)),
      equations = c(
        s_l = "- lambda * s_l",
        i_l = "lambda * s_l"
      ),
      observations = "i_l"
    ),
    model_type = "NeuralODE"
  )

  expect_s3_class(spec, "ndm_model_spec")
  expect_equal(spec$preset, "toy_sl_i")
  expect_equal(spec$state_terms, c("s_l", "i_l"))
  expect_equal(spec$compartments, "custom")
  expect_true(is.null(spec$source_path))
  expect_match(spec$tex_text, "Evolve\\{s_l\\}")
})
