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

test_that("built-in TeX specs keep init-state priors separate from dynamic priors", {
  spec <- ndm_model_spec(
    preset = "seirs_dynamic_beta",
    model_type = "NeuralODE"
  )
  roundtrip <- ndm_model_spec_from_tex(spec$source_path, model_type = "NeuralODE")

  expect_identical(spec$init_state_terms, c("s_l", "e_l", "i_l", "r_l"))
  expect_null(names(spec$init_state_terms))
  expect_false("beta_l" %in% spec$init_state_terms)
  expect_true(all(c("beta_l", "sigma", "gamma", "xi", "delta") %in% spec$parameter_terms))
  expect_identical(roundtrip$init_state_terms, spec$init_state_terms)
  expect_null(names(roundtrip$init_state_terms))
})

test_that("fallback TeX parsing distinguishes init priors from dynamic state priors", {
  tex_path <- tempfile(fileext = ".tex")
  on.exit(unlink(tex_path), add = TRUE)
  writeLines(
    c(
      "%%%START BAYES ODE%%%",
      "\\item $s_{l0} \\sim \\Dist{Normal}(0, 0.1)$",
      "\\item $\\InvSoftPlus(\\beta_{l}) \\sim \\Dist{Normal}(0, 0.1)$",
      "\\item $\\Evolve{s_l} = - \\beta_l \\cdot s_l$",
      "\\item $\\Evolve{\\beta_l} = \\Neural1{ \\beta_l , s_l }$",
      "\\item $\\Observe = s_l$",
      "%%%END BAYES ODE%%%"
    ),
    tex_path,
    useBytes = TRUE
  )

  spec <- ndm_model_spec_from_tex(tex_path, model_type = "NeuralODE")

  expect_identical(spec$init_state_terms, "s_l")
  expect_null(names(spec$init_state_terms))
  expect_identical(spec$parameter_terms, "beta_l")
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

test_that("structured TB presets observe active-TB incidence by default", {
  tb_presets <- sprintf("tb_%s", letters[1:12])

  for (preset in tb_presets) {
    family_args <- if (preset %in% c("tb_b", "tb_j")) list(n = 4L) else NULL
    spec <- ndm_model_spec(
      preset = preset,
      model_type = "NeuralODE",
      family_args = family_args
    )

    expect_equal(
      spec$execution_spec$observations,
      unname(spec$execution_spec$equations[["i_l"]]),
      info = preset
    )
    expect_false(identical(spec$execution_spec$observations, "i_l"), info = preset)
  }

  tb_b <- ndm_model_spec(
    preset = "tb_b",
    model_type = "NeuralODE",
    family_args = list(n = 4L)
  )
  tb_j <- ndm_model_spec(
    preset = "tb_j",
    model_type = "NeuralODE",
    family_args = list(n = 4L)
  )
  expect_equal(tb_b$execution_spec$observations, "d1 * lf1_l + d2 * lf2_l + d3 * lf3_l + d4 * lf4_l + c * ls_l")
  expect_equal(tb_j$execution_spec$observations, "c * l4_l")
})

test_that("structured TB presets support cumulative observation override", {
  tb_presets <- sprintf("tb_%s", letters[1:12])

  for (preset in tb_presets) {
    family_args <- if (preset %in% c("tb_b", "tb_j")) {
      list(n = 4L, observation_target = "cumulative")
    } else {
      list(observation_target = "cumulative")
    }
    spec <- ndm_model_spec(
      preset = preset,
      model_type = "NeuralODE",
      family_args = family_args
    )

    expect_equal(spec$execution_spec$observations, "i_l", info = preset)
    expect_equal(spec$family_args$observation_target, "cumulative", info = preset)
  }

  expect_error(
    ndm_model_spec(
      preset = "tb_a",
      model_type = "NeuralODE",
      family_args = list(observation_target = "stock")
    ),
    "observation_target"
  )
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
  expect_true(grepl("\\Observe = (x1 * (max(1.0, t) ^ x2)) * l_l", roundtrip$tex_text, fixed = TRUE))

  cumulative_spec <- ndm_model_spec(
    preset = "tb_k",
    model_type = "NeuralODE",
    family_args = list(observation_target = "cumulative")
  )
  cumulative_tex_path <- tempfile(fileext = ".tex")
  on.exit(unlink(cumulative_tex_path), add = TRUE)

  ndm_model_spec_to_tex(cumulative_spec, path = cumulative_tex_path)
  cumulative_roundtrip <- ndm_model_spec_from_tex(cumulative_tex_path, model_type = "NeuralODE")

  expect_equal(cumulative_roundtrip$family_args$observation_target, "cumulative")
  expect_equal(unname(cumulative_roundtrip$execution_spec$observations), "i_l")
  expect_true(grepl("\\Observe = i_l", cumulative_roundtrip$tex_text, fixed = TRUE))
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
