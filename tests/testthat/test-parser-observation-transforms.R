parser_observation_source_text <- function() {
  ndm_test_runtime_source_text(
    "ModelDefiners/SuperLModel_ParseDynamicODE.R"
  )
}

parser_observation_source_expressions <- function() {
  parse(
    text = parser_observation_source_text(),
    keep.source = FALSE
  )
}

parser_named_helper <- function(helper_name) {
  parser_exprs <- parser_observation_source_expressions()
  is_helper <- vapply(
    parser_exprs,
    function(expr) {
      is.call(expr) &&
        identical(expr[[1L]], as.name("<-")) &&
        identical(expr[[2L]], as.name(helper_name))
    },
    logical(1L)
  )
  expect_equal(sum(is_helper), 1L)

  helper_env <- new.env(parent = baseenv())
  eval(parser_exprs[[which(is_helper)]], envir = helper_env)
  get(
    helper_name,
    envir = helper_env,
    inherits = FALSE
  )
}

parser_observation_helper <- function() {
  parser_named_helper(".ndm_unwrap_pretransformed_observation_args")
}

test_that("ODE constants are imported without caller-frame evaluation", {
  import_constants <- parser_named_helper(".ndm_import_numeric_ode_constants")
  runtime_env <- new.env(parent = baseenv())
  fake_jnp <- list(array = function(value) structure(value, backend = "fake-jnp"))

  observed_names <- import_constants(
    definitions = c("N \\leftarrow 1", "Scale \\leftarrow 1e4"),
    jnp_module = fake_jnp,
    converter = function(value) as.numeric(value),
    target_env = runtime_env
  )

  expect_identical(observed_names, c("N", "Scale"))
  expect_true(exists("CONST_N", envir = runtime_env, inherits = FALSE))
  expect_true(exists("CONST_Scale", envir = runtime_env, inherits = FALSE))
  expect_equal(
    as.numeric(get("CONST_N", envir = runtime_env, inherits = FALSE)),
    1
  )
  expect_equal(
    as.numeric(get("CONST_Scale", envir = runtime_env, inherits = FALSE)),
    1e4
  )
  expect_identical(
    attr(get("CONST_N", envir = runtime_env, inherits = FALSE), "backend"),
    "fake-jnp"
  )

  expect_error(
    import_constants(
      "N;system('false') \\leftarrow 1",
      fake_jnp,
      as.numeric,
      runtime_env
    ),
    "Invalid ODE constant name"
  )
  expect_error(
    import_constants("N \\leftarrow 1/2", fake_jnp, as.numeric, runtime_env),
    "finite number"
  )
})

parser_evolving_sigmoid_spec <- function() {
  spec <- ndm_model_spec(preset = "tb_e", model_type = "NeuralODE")
  lines <- strsplit(spec$tex_text, "\n", fixed = TRUE)[[1L]]

  prior_idx <- grep("InvSigmoid(a)", lines, fixed = TRUE)
  expect_length(prior_idx, 1L)
  lines[[prior_idx]] <- sub(
    "InvSigmoid(a)",
    "InvSigmoid(a_l)",
    lines[[prior_idx]],
    fixed = TRUE
  )

  role_rule_idx <- grep("\\\\Evolve\\{[li]_l\\}|\\\\Observe =", lines)
  lines[role_rule_idx] <- gsub(
    "(?<![[:alnum:]_])a(?![[:alnum:]_])",
    "Sigmoid(a_l)",
    lines[role_rule_idx],
    perl = TRUE
  )
  active_state_idx <- grep("Evolve{i_l}", lines, fixed = TRUE)
  expect_length(active_state_idx, 1L)
  lines <- append(
    lines,
    "\\item $\\Evolve{a_l} = \\Neural1{ a_l , s_l , l_l , i_l }$",
    after = active_state_idx
  )

  source_path <- tempfile("ndm-parser-evolving-sigmoid-", fileext = ".tex")
  writeLines(lines, source_path, useBytes = TRUE)
  spec$preset <- "tb_e_evolving_sigmoid"
  spec$source_path <- source_path
  spec$tex_text <- paste(lines, collapse = "\n")
  spec
}

test_that("observation parser unwraps only explicitly pre-transformed arguments", {
  unwrap <- parser_observation_helper()
  expressions <- c(
    "Sigmoid(ODEParamsSampList_args$delta_samp)*diff_eq_sol$ys$i_l",
    "Sigmoid(jnp$take(diff_eq_sol$ys$Neural1,4L,axis=1L))*diff_eq_sol$ys$s_l",
    "Sigmoid(ODEParamsSampList_args$untouched_samp)"
  )

  observed <- unwrap(expressions, argument_names = "delta")

  expect_identical(
    observed[[1L]],
    "ODEParamsSampList_args$delta_samp*diff_eq_sol$ys$i_l"
  )
  expect_identical(observed[[2L]], expressions[[2L]])
  expect_identical(observed[[3L]], expressions[[3L]])
})

test_that("canonical parser classifies fixed InvSigmoid arguments by role", {
  parser_source <- parser_observation_source_text()

  expect_match(
    parser_source,
    "PriorDefinitions_jax[1,] == \"InvSigmoid\"",
    fixed = TRUE
  )
  expect_match(
    parser_source,
    "prior_sigmoid_args <- intersect(prior_sigmoid_args, uq_args_vec)",
    fixed = TRUE
  )
  expect_match(
    parser_source,
    "prior_sigmoid_args <- setdiff(prior_sigmoid_args, uq_allneural_vec)",
    fixed = TRUE
  )
  expect_false(grepl(
    "gsub(observed_vec_final,pattern = \"Sigmoid\"",
    parser_source,
    fixed = TRUE
  ))
})

test_that("ODE parser rejects unequal derivative vectors before pairing", {
  validate_pairs <- parser_named_helper(".ndm_validate_ode_pair_lengths")

  expect_invisible(validate_pairs(c("s_l", "i_l"), c("-flow", "flow")))
  expect_error(
    validate_pairs(c("s_l", "i_l"), "-flow"),
    paste(
      "ODE parser produced 2 state derivative name(s) but 1",
      "right-hand-side expression(s); refusing to recycle them."
    ),
    fixed = TRUE
  )
})

test_that("ODE parser failures stop without entering an interactive debugger", {
  require_success <- parser_named_helper(".ndm_require_parser_success")
  synthetic_failure <- try(
    stop("synthetic prior failure", call. = FALSE),
    silent = TRUE
  )

  expect_identical(
    require_success("parsed", "assembling a prior definition"),
    "parsed"
  )
  expect_error(
    require_success(synthetic_failure, "assembling a prior definition"),
    paste(
      "ODE parser failed while assembling a prior definition:",
      "synthetic prior failure"
    ),
    fixed = TRUE
  )

  parser_exprs <- parser_observation_source_expressions()
  contains_browser_call <- function(expr) {
    if (!is.call(expr)) {
      return(FALSE)
    }
    if (identical(expr[[1L]], as.name("browser"))) {
      return(TRUE)
    }
    any(vapply(
      as.list(expr)[-1L],
      contains_browser_call,
      logical(1L)
    ))
  }
  expect_false(any(vapply(
    parser_exprs,
    contains_browser_call,
    logical(1L)
  )))
})

test_that("canonical parser emits global state indices before compilation", {
  parser_source <- parser_observation_source_text()

  global_take <- paste0(
    "jnp$take(y$Neural2, ",
    "ai(GlobalNeuralEmbedDim+%s-1L))"
  )
  local_take <- paste0(
    "jnp$take(y$Neural2, ",
    "ai(LocalNeuralEmbedDim+%s-1L))"
  )
  global_take_position <- regexpr(
    global_take,
    parser_source,
    fixed = TRUE
  )[[1L]]
  compile_position <- regexpr(
    "VI_ODE_term <- eval",
    parser_source,
    fixed = TRUE
  )[[1L]]

  expect_gt(global_take_position, 0L)
  expect_gt(compile_position, 0L)
  expect_lt(global_take_position, compile_position)
  expect_false(grepl(local_take, parser_source, fixed = TRUE))
  expect_match(
    parser_source,
    "diff_eq_sol\\\\$ys\\\\$(?:Neural1|Neural2)",
    fixed = TRUE
  )
})

test_that("canonical parser compiles fixed and evolved Sigmoid roles consistently", {
  skip_on_cran()
  ndm_require_backend_test_stack("observation parser integration test")

  fixed_spec <- ndm_model_spec(
    preset = "seirs_dynamic_beta",
    model_type = "NeuralODE"
  )
  fixed_source_path <- tempfile(
    "ndm-parser-fixed-sigmoid-",
    fileext = ".tex"
  )
  ndm_model_spec_to_tex(fixed_spec, path = fixed_source_path)
  fixed_spec$source_path <- fixed_source_path
  details <- ndm_test_fit_sim_case(
    model_type = "NeuralODE",
    endogeneity = 0,
    n_sgd = 1L,
    model_dims = 16L,
    n_times_lookahead = 2L,
    model_spec = fixed_spec,
    return_details = TRUE
  )
  runtime_env <- details$runtime_env
  global_snapshot <- ndm_test_copy_env(runtime_env)
  on.exit(ndm_test_restore_env(global_snapshot), add = TRUE)

  # Source the canonical parser explicitly so this test covers edits before the
  # generated runtime registry is refreshed.
  suppressWarnings(eval(
    parser_observation_source_expressions(),
    envir = runtime_env
  ))
  fixed_observation <- as.character(runtime_env$observed_vec_final)
  expect_length(fixed_observation, 1L)
  expect_match(
    fixed_observation,
    "ODEParamsSampList_args$delta_samp",
    fixed = TRUE
  )
  expect_false(grepl("Sigmoid", fixed_observation, fixed = TRUE))

  fixed_eval_env <- new.env(parent = runtime_env)
  fixed_eval_env$ODEParamsSampList_args <- list(
    delta_samp = runtime_env$jnp$array(0.12)
  )
  fixed_eval_env$diff_eq_sol <- list(
    ys = list(i_l = runtime_env$jnp$array(c(0.2, 0.4)))
  )
  fixed_value <- eval(parse(text = fixed_observation), envir = fixed_eval_env)
  fixed_value <- as.numeric(reticulate::py_to_r(
    runtime_env$np$asanyarray(fixed_value)
  ))
  expect_equal(fixed_value, 0.12 * c(0.2, 0.4), tolerance = 1e-7)

  evolving_spec <- parser_evolving_sigmoid_spec()
  runtime_env$model_tex_loc <- evolving_spec$source_path
  suppressWarnings(eval(
    parser_observation_source_expressions(),
    envir = runtime_env
  ))
  evolving_observation <- as.character(runtime_env$observed_vec_final)
  expect_length(evolving_observation, 1L)
  expect_match(evolving_observation, "Sigmoid(", fixed = TRUE)
  expect_match(evolving_observation, "diff_eq_sol$ys$Neural1", fixed = TRUE)
  expect_identical(as.character(runtime_env$uq_encneural_vec), "a_l")

  local_dim <- as.integer(runtime_env$LocalNeuralEmbedDim)
  raw_a <- c(-1.25, 0.75)
  neural_path <- matrix(0, nrow = length(raw_a), ncol = local_dim + 1L)
  neural_path[, local_dim + 1L] <- raw_a
  s_path <- c(0.8, 0.6)
  l_path <- c(0.1, 0.2)
  lambda <- 0.3
  c_rate <- 0.08

  evolving_eval_env <- new.env(parent = runtime_env)
  evolving_eval_env$ODEParamsSampList_args <- list(
    lambda_samp = runtime_env$jnp$array(lambda),
    c_samp = runtime_env$jnp$array(c_rate)
  )
  evolving_eval_env$diff_eq_sol <- list(
    ys = list(
      Neural1 = runtime_env$jnp$array(neural_path),
      s_l = runtime_env$jnp$array(s_path),
      l_l = runtime_env$jnp$array(l_path)
    )
  )
  evolving_value <- eval(
    parse(text = evolving_observation),
    envir = evolving_eval_env
  )
  evolving_value <- as.numeric(reticulate::py_to_r(
    runtime_env$np$asanyarray(evolving_value)
  ))
  expected_inflow <- lambda * stats::plogis(raw_a) * s_path + c_rate * l_path
  expect_equal(evolving_value, expected_inflow, tolerance = 1e-7)

  dynamic_spec <- ndm_model_spec(
    preset = "seirs_dynamic_beta_dynamic_global",
    model_type = "NeuralODE"
  )
  runtime_env$model_tex_loc <- dynamic_spec$source_path
  runtime_env$LocalNeuralEmbedDim <- 48L
  runtime_env$GlobalNeuralEmbedDim <- 40L
  suppressWarnings(eval(
    parser_observation_source_expressions(),
    envir = runtime_env
  ))

  expect_equal(as.integer(runtime_env$LocalNeuralEmbedDim), 48L)
  expect_equal(as.integer(runtime_env$GlobalNeuralEmbedDim), 40L)
  global_rate_rhs <- runtime_env$righthandside_vec[
    grepl(
      "jnp$take(y$Neural2, ai(",
      runtime_env$righthandside_vec,
      fixed = TRUE
    )
  ]
  expect_gte(length(global_rate_rhs), 4L)
  expect_true(all(grepl(
    "ai(GlobalNeuralEmbedDim+",
    global_rate_rhs,
    fixed = TRUE
  )))
  expect_false(any(grepl(
    "ai(LocalNeuralEmbedDim+",
    global_rate_rhs,
    fixed = TRUE
  )))

  compiled_body <- paste(
    deparse(body(runtime_env$f_VI_ODE_TERM), width.cutoff = 500L),
    collapse = "\n"
  )
  expect_match(
    compiled_body,
    "take(y$Neural2, ai(GlobalNeuralEmbedDim",
    fixed = TRUE
  )
  expect_false(grepl(
    "take(y$Neural2, ai(LocalNeuralEmbedDim",
    compiled_body,
    fixed = TRUE
  ))

  local_state_size <- as.integer(unlist(
    runtime_env$LocalNeural$MagDtScaler$shape
  ))
  local_args <- list(
    ResidConProj = function(x) {
      runtime_env$jnp$zeros(local_state_size, dtype = runtime_env$jnp$float32)
    },
    MLP = list(
      WideProj1 = function(x) runtime_env$jnp$zeros(1L),
      WideProj2 = function(x) runtime_env$jnp$zeros(1L),
      OutProj1 = function(x) {
        runtime_env$jnp$zeros(local_state_size, dtype = runtime_env$jnp$float32)
      }
    ),
    LNScaler = runtime_env$jnp$array(1.0),
    NonLinearPathScaler = runtime_env$jnp$array(1.0),
    MagDtScaler = runtime_env$jnp$ones(
      local_state_size,
      dtype = runtime_env$jnp$float32
    )
  )
  global_args <- runtime_env$GlobalNeural
  global_args$NonLinearPathScaler <- runtime_env$SoftPlus(
    global_args$NonLinearPathScaler
  )
  global_args$MagDtScaler <- runtime_env$SoftPlus(global_args$MagDtScaler)
  zero_policy <- list(evaluate = function(t) runtime_env$jnp$array(0.0))
  vector_field_value <- runtime_env$f_VI_ODE_TERM(
    runtime_env$jnp$array(0.0),
    list(
      s_l = runtime_env$jnp$array(100.0),
      e_l = runtime_env$jnp$array(10.0),
      i_l = runtime_env$jnp$array(5.0),
      r_l = runtime_env$jnp$array(2.0),
      Neural2 = runtime_env$GlobalNeural$NeuralInitialConditions,
      Neural1 = runtime_env$jnp$zeros(
        local_state_size,
        dtype = runtime_env$jnp$float32
      )
    ),
    list(
      Neural1 = local_args,
      Neural2 = global_args,
      policy_scenario_indicator = zero_policy,
      policy_scenario = zero_policy
    )
  )
  for (component in c("s_l", "e_l", "i_l", "r_l")) {
    expect_true(
      as.logical(as.numeric(runtime_env$np$asarray(
        runtime_env$jnp$all(runtime_env$jnp$isfinite(
          vector_field_value[[component]]
        ))
      ))),
      info = component
    )
  }

  dynamic_observation <- as.character(runtime_env$observed_vec_final)
  expect_length(dynamic_observation, 1L)
  expect_match(dynamic_observation, "diff_eq_sol$ys$Neural2", fixed = TRUE)
  expect_match(dynamic_observation, "axis=1L", fixed = TRUE)

  n_saved_times <- 4L
  delta_path <- c(-2, -0.5, 0.75, 2.5)
  infected_path <- c(2, 3, 5, 7)
  neural2_path <- cbind(
    matrix(
      0,
      nrow = n_saved_times,
      ncol = as.integer(runtime_env$GlobalNeuralEmbedDim)
    ),
    sigma = rep(0.2, n_saved_times),
    gamma = rep(0.3, n_saved_times),
    xi = rep(0.05, n_saved_times),
    delta = delta_path
  )
  neural2_path <- runtime_env$jnp$array(neural2_path)

  observation_eval_env <- new.env(parent = runtime_env)
  observation_eval_env$diff_eq_sol <- list(
    ys = list(
      Neural2 = neural2_path,
      i_l = runtime_env$jnp$array(infected_path)
    )
  )
  dynamic_value <- eval(
    parse(text = dynamic_observation),
    envir = observation_eval_env
  )
  dynamic_value <- as.numeric(reticulate::py_to_r(
    runtime_env$np$asanyarray(dynamic_value)
  ))
  expect_equal(
    dynamic_value,
    stats::plogis(delta_path) * infected_path,
    tolerance = 1e-7
  )

  observation_function_env <- new.env(parent = runtime_env)
  observation_function_env$infected_path <- runtime_env$jnp$array(
    infected_path
  )
  observation_function <- eval(
    parse(text = sprintf(
      paste0(
        "function(neural2) {",
        "diff_eq_sol <- list(ys = list(",
        "Neural2 = neural2, i_l = infected_path));",
        "%s",
        "}"
      ),
      dynamic_observation
    )),
    envir = observation_function_env
  )
  observation_gradient <- runtime_env$jax$grad(function(neural2) {
    runtime_env$jnp$sum(observation_function(neural2))
  })(neural2_path)
  observation_gradient <- reticulate::py_to_r(
    runtime_env$np$asanyarray(observation_gradient)
  )
  delta_gradient <- observation_gradient[
    ,
    as.integer(runtime_env$GlobalNeuralEmbedDim) + 4L
  ]
  expect_equal(
    as.numeric(delta_gradient),
    infected_path *
      stats::plogis(delta_path) *
      (1 - stats::plogis(delta_path)),
    tolerance = 1e-7
  )

  fixed_seir_path <- tempfile(
    "ndm-fixed-seir-runtime-",
    fileext = ".tex"
  )
  writeLines(
    ndm_test_runtime_source_text(
      "ModelStructureTex/bayes_ode_SEIRS_FixedBeta_FixedGlobal.tex"
    ),
    fixed_seir_path,
    useBytes = TRUE
  )
  on.exit(unlink(fixed_seir_path), add = TRUE)
  runtime_env$model_tex_loc <- fixed_seir_path
  runtime_env$nOutcomes <- 2L
  suppressWarnings(eval(
    parser_observation_source_expressions(),
    envir = runtime_env
  ))
  fixed_incidence_rhs <- runtime_env$righthandside_vec[
    runtime_env$lefthandside_vec %in% c("s_l", "e_l")
  ]
  expect_length(fixed_incidence_rhs, 2L)
  expect_true(all(grepl("CONST_N", fixed_incidence_rhs, fixed = TRUE)))
})
