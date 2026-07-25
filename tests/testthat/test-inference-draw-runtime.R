load_inference_runtime_helpers <- function() {
  env <- new.env(parent = baseenv())
  env$SimMode <- TRUE
  ndm:::.ndm_eval_embedded_runtime(
    "ResultsGet/SuperLModel_GetAnalytics_Real.R",
    env
  )
  env
}

test_that("real inference only repeats genuinely stochastic NeuralODE draws", {
  env <- load_inference_runtime_helpers()

  variational <- env$ndm_inference_draw_plan(
    "NeuralODE",
    neuralode_variational = TRUE,
    requested_draws = 5L
  )
  expect_equal(variational$requested, 5L)
  expect_equal(variational$effective, 5L)
  expect_true(variational$stochastic)

  deterministic <- env$ndm_inference_draw_plan(
    "NeuralODE",
    neuralode_variational = FALSE,
    requested_draws = 5L
  )
  expect_equal(deterministic$requested, 5L)
  expect_equal(deterministic$effective, 1L)
  expect_false(deterministic$stochastic)

  decoder <- env$ndm_inference_draw_plan(
    "DecoderOnly",
    neuralode_variational = TRUE,
    requested_draws = 5L
  )
  expect_equal(decoder$effective, 1L)
  expect_false(decoder$stochastic)

  expect_error(
    env$ndm_inference_draw_plan("NeuralODE", TRUE, requested_draws = 0L),
    "positive integer"
  )
})

test_that("inference key lineage is stable by row and distinct by draw", {
  env <- load_inference_runtime_helpers()
  first <- env$ndm_inference_key_lineage(
    run_seed = 101L,
    outer_iteration = 4L,
    checkpoint_step = 900L,
    draw_id = 1L,
    location_ids = c(7L, 2L),
    time_anchor_ids = c(18L, 12L)
  )
  repeated <- env$ndm_inference_key_lineage(
    run_seed = 101L,
    outer_iteration = 4L,
    checkpoint_step = 900L,
    draw_id = 1L,
    location_ids = c(2L, 7L),
    time_anchor_ids = c(12L, 18L)
  )
  second_draw <- env$ndm_inference_key_lineage(
    run_seed = 101L,
    outer_iteration = 4L,
    checkpoint_step = 900L,
    draw_id = 2L,
    location_ids = c(7L, 2L),
    time_anchor_ids = c(18L, 12L)
  )

  expect_identical(first$global, repeated$global)
  expect_identical(first$local[[1]], repeated$local[[2]])
  expect_identical(first$local[[2]], repeated$local[[1]])
  expect_false(identical(first$global, second_draw$global))
  expect_false(identical(first$local[[1]], second_draw$local[[1]]))
})

test_that("scoped key construction shares globals and stabilizes local rows", {
  env <- load_inference_runtime_helpers()
  fake_jax <- list(
    random = list(
      fold_in = function(key, value) {
        c(
          (key[[1]] * 31 + value + 17) %% 2147483647,
          (key[[2]] * 131 + value + 29) %% 2147483647
        )
      }
    )
  )
  fake_jnp <- list(
    stack = function(values, axis = 0L) {
      if (is.null(dim(values[[1]]))) {
        return(do.call(rbind, values))
      }
      aperm(simplify2array(values), c(3L, 1L, 2L))
    }
  )
  fake_key <- function(seed) c(as.numeric(seed), as.numeric(seed) + 1)
  build_keys <- function(draw_id, locations, anchors) {
    env$ndm_inference_scoped_keys(
      jax = fake_jax,
      jnp = fake_jnp,
      JaxKey = fake_key,
      run_seed = 101L,
      outer_iteration = 4L,
      checkpoint_step = 900L,
      draw_id = draw_id,
      location_ids = locations,
      time_anchor_ids = anchors
    )
  }

  first <- build_keys(1L, c(7L, 2L), c(18L, 12L))
  repeated <- build_keys(1L, c(7L, 2L), c(18L, 12L))
  reordered <- build_keys(1L, c(2L, 7L), c(12L, 18L))
  second_draw <- build_keys(2L, c(7L, 2L), c(18L, 12L))

  expect_identical(first, repeated)
  expect_equal(first[1L, 2L, ], first[2L, 2L, ])
  expect_false(isTRUE(all.equal(first[1L, 1L, ], first[2L, 1L, ])))
  expect_equal(first[1L, 1L, ], reordered[2L, 1L, ])
  expect_equal(first[2L, 1L, ], reordered[1L, 1L, ])
  expect_false(isTRUE(all.equal(first, second_draw)))
})

test_that("all requested inference draws are required", {
  env <- load_inference_runtime_helpers()
  complete <- list(
    list(ok = TRUE, value = 1, error = NULL),
    list(ok = TRUE, value = 2, error = NULL)
  )
  expect_invisible(
    env$ndm_stop_incomplete_inference_draws(
      complete,
      requested = 2L,
      checkpoint_step = 10L
    )
  )

  incomplete <- complete
  incomplete[[2]] <- list(
    ok = FALSE,
    value = NULL,
    error = simpleError("failed draw")
  )
  expect_condition(
    env$ndm_stop_incomplete_inference_draws(
      incomplete,
      requested = 2L,
      checkpoint_step = 10L
    ),
    class = "ndm_inference_draw_failure"
  )
})

test_that("posterior prediction reduction is the arithmetic draw mean", {
  env <- load_inference_runtime_helpers()
  draws <- list(
    list(y_mu = matrix(c(1, 3), nrow = 1L)),
    list(y_mu = matrix(c(3, 5), nrow = 1L)),
    list(y_mu = matrix(c(8, 10), nrow = 1L))
  )
  reduced <- env$ndm_mean_inference_draws(draws, function(draw) draw$y_mu)
  expect_equal(reduced, matrix(c(4, 6), nrow = 1L))
  expect_error(env$ndm_mean_inference_draws(list()), "At least one")
})

test_that("inference diagnostics transfer live JAX arrays to the host", {
  conda_env <- ndm_require_backend_test_stack(
    "inference diagnostic host transfer",
    packages = "reticulate"
  )
  old_backend <- ndm:::ndm_env$backend
  on.exit(assign("backend", old_backend, envir = ndm:::ndm_env), add = TRUE)
  backend <- ndm_initialize_backend(
    conda_env = conda_env,
    float_type = "32",
    import_tensorflow = FALSE
  )
  env <- load_inference_runtime_helpers()
  value <- backend$jnp$array(c(3L, 7L), dtype = backend$jnp$int32)

  expect_identical(
    as.integer(env$ndm_inference_host_array(value, backend$np)),
    c(3L, 7L)
  )
})

test_that("inference rejects any unsuccessful solver row", {
  env <- load_inference_runtime_helpers()
  diagnostics <- list(
    success = c(TRUE, FALSE),
    failure_stage_code = c(0L, 2L),
    prediction_finite = c(TRUE, TRUE),
    global_attempted = c(FALSE, FALSE),
    global_result_success = c(TRUE, TRUE),
    global_state_finite = c(TRUE, TRUE),
    global_result_code = c(0L, 0L),
    global_num_steps = c(0L, 0L),
    global_num_accepted_steps = c(0L, 0L),
    global_num_rejected_steps = c(0L, 0L),
    global_max_steps = c(0L, 0L),
    local_attempted = c(TRUE, TRUE),
    local_result_success = c(TRUE, FALSE),
    local_state_finite = c(TRUE, FALSE),
    local_result_code = c(0L, 1L),
    local_num_steps = c(9L, 100L),
    local_num_accepted_steps = c(8L, 70L),
    local_num_rejected_steps = c(1L, 30L),
    local_max_steps = c(100L, 100L)
  )
  prediction <- list(solver_diagnostics = diagnostics)

  condition <- tryCatch(
    {
      env$ndm_validate_inference_solver_diagnostics(
        prediction = prediction,
        draw_id = 3L,
        location_ids = c(4L, 7L),
        time_anchor_ids = c(10L, 12L)
      )
      NULL
    },
    ndm_solver_failure = identity
  )
  expect_s3_class(condition, "ndm_solver_failure")
  expect_equal(condition$draw_id, 3L)
  expect_equal(condition$solver_report$location_id, 7L)
  expect_equal(condition$solver_report$time_anchor_id, 12L)
  expect_equal(condition$solver_report$failure_stage, "local_ode")
  expect_equal(condition$solver_report$local_num_rejected_steps, 30L)
})

test_that("solver diagnostics are mandatory unless legacy mode is explicit", {
  env <- load_inference_runtime_helpers()
  prediction <- list(y_mu = 1)

  expect_error(
    env$ndm_validate_inference_solver_diagnostics(
      prediction = prediction,
      draw_id = 1L,
      location_ids = 1L,
      time_anchor_ids = 2L
    ),
    "required solver diagnostics"
  )
  expect_false(
    env$ndm_validate_inference_solver_diagnostics(
      prediction = prediction,
      draw_id = 1L,
      location_ids = 1L,
      time_anchor_ids = 2L,
      require_diagnostics = FALSE
    )
  )
})

test_that("BuildML accepts scoped inference keys without changing legacy keys", {
  build_ml <- ndm:::.ndm_embedded_runtime_source(
    "ModelDefiners/SuperLModel_BuildML.R"
  )

  expect_match(build_ml, "length(seed$shape) == 2L", fixed = TRUE)
  expect_match(build_ml, "local_sampling_seed", fixed = TRUE)
  expect_match(build_ml, "global_sampling_seed", fixed = TRUE)
  expect_match(build_ml, "jax$random$fold_in", fixed = TRUE)
  expect_match(
    build_ml,
    "jnp$add(root_key, jnp$array(ai(domain_tag)))",
    fixed = TRUE
  )
})
