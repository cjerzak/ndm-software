solver_failure_assignment <- function(node, target) {
  if (is.call(node) &&
      length(node) >= 3L &&
      identical(node[[1L]], as.name("<-")) &&
      identical(node[[2L]], as.name(target))) {
    return(node)
  }
  if (!is.recursive(node)) {
    return(NULL)
  }
  for (index in seq_along(node)) {
    if (identical(node[[index]], quote(expr = ))) {
      next
    }
    child <- node[[index]]
    found <- solver_failure_assignment(child, target)
    if (!is.null(found)) {
      return(found)
    }
  }
  NULL
}

solver_failure_contains_symbol <- function(node, target) {
  if (is.symbol(node) && identical(node, as.name(target))) {
    return(TRUE)
  }
  if (!is.recursive(node)) {
    return(FALSE)
  }
  for (index in seq_along(node)) {
    if (identical(node[[index]], quote(expr = ))) {
      next
    }
    if (solver_failure_contains_symbol(node[[index]], target)) {
      return(TRUE)
    }
  }
  FALSE
}

solver_failure_guard <- function(node) {
  if (is.call(node) &&
      length(node) >= 3L &&
      identical(node[[1L]], as.name("if")) &&
      solver_failure_contains_symbol(node[[2L]], "failed_solver_examples")) {
    return(node)
  }
  if (!is.recursive(node)) {
    return(NULL)
  }
  for (index in seq_along(node)) {
    if (identical(node[[index]], quote(expr = ))) {
      next
    }
    found <- solver_failure_guard(node[[index]])
    if (!is.null(found)) {
      return(found)
    }
  }
  NULL
}

test_that("solver failures carry structured stage and counter context", {
  diagnostics <- data.frame(
    example_index = 2L,
    location_id_numeric = 41L,
    time_id_numeric = 19L,
    failure_stage_code = 2L,
    local_num_steps = 4096L,
    local_num_accepted_steps = 4030L,
    local_num_rejected_steps = 66L,
    local_max_steps = 4096L
  )
  condition <- ndm:::.ndm_solver_failure_condition(
    iteration = 17L,
    example_indices = 2L,
    stages = "local_ode",
    diagnostics = diagnostics,
    context = list(
      phase = "training",
      candidate_update_applied = FALSE,
      solver_configuration = list(
        solver = "Tsit5()",
        stepsize_controller = "PIDController(...)",
        rtol = 1e-5,
        atol = 1e-7,
        dt0 = 1e-3,
        max_steps = 4096L
      ),
      report_path = "/tmp/example-solver-report.rds"
    )
  )

  expect_s3_class(condition, "ndm_solver_failure")
  expect_s3_class(condition, "error")
  expect_identical(condition$iteration, 17L)
  expect_identical(condition$example_indices, 2L)
  expect_identical(condition$stage, "local_ode")
  expect_identical(condition$diagnostics, diagnostics)
  expect_false(condition$context$candidate_update_applied)
  expect_identical(condition$diagnostics$location_id_numeric, 41L)
  expect_identical(condition$diagnostics$time_id_numeric, 19L)
  expect_equal(condition$context$solver_configuration$rtol, 1e-5)
  expect_identical(
    condition$context$report_path,
    "/tmp/example-solver-report.rds"
  )
  expect_match(conditionMessage(condition), "no candidate state was committed")
  expect_error(stop(condition), class = "ndm_solver_failure")
})

test_that("solver failure reports are atomically published as structured RDS", {
  holder_folder <- tempfile("solver-failure-holder-")
  dir.create(holder_folder)
  on.exit(unlink(holder_folder, recursive = TRUE, force = TRUE), add = TRUE)
  report <- list(
    schema_version = 1L,
    event = "ndm_solver_failure",
    solver_configuration = list(
      solver = "Tsit5()",
      stepsize_controller = "PIDController(...)",
      rtol = 1e-5,
      atol = 1e-7,
      dt0 = 1e-3,
      max_steps = 4096L
    ),
    failed_examples = data.frame(
      location_id_numeric = 41L,
      time_id_numeric = 19L,
      local_num_steps = 4096L,
      local_num_rejected_steps = 66L
    )
  )

  report_path <- ndm:::.ndm_write_solver_failure_report(
    report = report,
    holder_folder = holder_folder,
    outer_iteration = 7L,
    iteration = 17L
  )

  expect_true(file.exists(report_path))
  expect_identical(readRDS(report_path), report)
  expect_match(basename(report_path), "solver_failure_outer7_i17_")
  expect_length(
    list.files(holder_folder, pattern = "^solver-failure-"),
    0L
  )
})

test_that("failed solver rows are JIT-safely sanitized out of the temporary loss", {
  conda_env <- ndm_require_backend_test_stack(
    "solver failure loss sanitization",
    packages = "reticulate"
  )
  old_backend <- ndm:::ndm_env$backend
  on.exit(assign("backend", old_backend, envir = ndm:::ndm_env), add = TRUE)
  backend <- ndm_initialize_backend(
    conda_env = conda_env,
    float_type = "32",
    import_tensorflow = FALSE
  )
  jnp <- backend$jnp
  sanitize_loss <- backend$jax$jit(function(y_mu, y_sigma) {
    success <- jnp$array(c(TRUE, FALSE))
    target <- jnp$zeros(list(2L, 1L, 1L))
    target_mask <- jnp$ones(list(2L, 1L, 1L), dtype = jnp$bool_)
    success_mask <- jnp$broadcast_to(
      jnp$reshape(success, list(success$shape[[1]], 1L, 1L)),
      target_mask$shape
    )
    safe_mask <- jnp$logical_and(target_mask, success_mask)
    safe_mu <- jnp$nan_to_num(
      y_mu,
      nan = 0.,
      posinf = 0.,
      neginf = 0.
    )
    safe_sigma <- jnp$nan_to_num(
      y_sigma,
      nan = 1e-5,
      posinf = 1e-5,
      neginf = 1e-5
    )
    safe_mu <- jnp$where(success_mask, safe_mu, jnp$zeros_like(safe_mu))
    safe_sigma <- jnp$where(
      success_mask,
      safe_sigma,
      jnp$ones_like(safe_sigma) * 1e-5
    )
    ndm:::.ndm_student_t_masked_nll(
      backend$jax,
      jnp,
      target,
      safe_mu,
      safe_sigma,
      safe_mask,
      df = 4,
      scale_floor = 1e-5
    )$loss
  })
  y_mu <- jnp$array(array(c(0.25, NaN), dim = c(2, 1, 1)))
  y_sigma <- jnp$array(array(c(0.1, Inf), dim = c(2, 1, 1)))

  loss <- sanitize_loss(y_mu, y_sigma)
  gradients <- backend$jax$grad(
    function(mu) sanitize_loss(mu, y_sigma)
  )(y_mu)

  expect_true(is.finite(as.numeric(backend$np$asanyarray(loss))))
  expect_true(all(is.finite(as.numeric(backend$np$asanyarray(gradients)))))
  expect_equal(
    as.numeric(backend$np$asanyarray(gradients))[[2L]],
    0,
    tolerance = 0
  )
})

test_that("training rejects failed solver rows before committing an update", {
  train_do <- ndm:::.ndm_runtime_expression(
    "ModelTrainers/SuperLModel_TrainDo.R"
  )
  failure_gate <- solver_failure_assignment(
    train_do,
    "failed_solver_examples"
  )
  failure_guard <- solver_failure_guard(train_do)
  expect_true(is.call(failure_gate))
  expect_true(is.call(failure_guard))

  gate_env <- list2env(
    list(
      solver_diagnostics_host = data.frame(
        example_index = 1:3,
        success = c(TRUE, FALSE, NA),
        failure_stage_code = c(0L, 2L, 2L),
        location_id_numeric = c(10L, 20L, 30L),
        time_id_numeric = c(100L, 200L, 300L)
      ),
      Loss_i = 1.25,
      GradNorm_i = 0.5,
      HolderFolder = tempdir(),
      i = 1L,
      AnalysisName = "SolverGateTest",
      OUTER_ITERATION = 7L,
      ModelType = "NeuralODE",
      solver_failure_stage = function(stage_code) {
        c("success", "global_ode", "local_ode", "prediction")[
          as.integer(stage_code) + 1L
        ]
      },
      solver_runtime_configuration = function() {
        list(solver = "TestSolver", max_steps = 100L)
      },
      capture_nonfinite_base_id = function() 91L,
      ndm_solver_failure_condition = ndm:::.ndm_solver_failure_condition
    ),
    parent = baseenv()
  )
  gate_env$ndm_write_solver_failure_report <- function(report, ...) {
    gate_env$written_report <- report
    "/tmp/solver-gate-test.rds"
  }

  condition <- tryCatch(
    {
      eval(failure_gate, envir = gate_env)
      eval(failure_guard, envir = gate_env)
      gate_env$candidate_update_applied <- TRUE
      NULL
    },
    ndm_solver_failure = identity
  )

  expect_identical(gate_env$failed_solver_examples, c(2L, 3L))
  expect_s3_class(condition, "ndm_solver_failure")
  expect_identical(condition$iteration, 1L)
  expect_identical(condition$example_indices, c(2L, 3L))
  expect_true(all(condition$stage == "local_ode"))
  expect_false(condition$context$candidate_update_applied)
  expect_false(exists("candidate_update_applied", envir = gate_env))
  expect_identical(condition$context$report_path, "/tmp/solver-gate-test.rds")
  expect_identical(gate_env$written_report$event, "ndm_solver_failure")
  expect_identical(gate_env$written_report$iteration, 1L)
  expect_identical(
    gate_env$written_report$failed_examples$example_index,
    c(2L, 3L)
  )
  expect_true(
    all(gate_env$written_report$failed_examples$failure_stage == "local_ode")
  )
  expect_false(gate_env$written_report$candidate_update_applied)
})
