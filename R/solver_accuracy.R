# Forward step doubling is an accuracy diagnostic, not an error bound on the
# exact ODE or its gradient. The resolution decision is outside autodiff/vmap.
.ndm_refine_rk4 <- function(probe, compare, initial, maximum, rtol, atol,
                            fallback = "diffrax", unit_grid = TRUE) {
  if (any(!is.finite(c(rtol, atol))) || rtol <= 0 || atol <= 0) {
    stop("ODE tolerances must be finite and positive.", call. = FALSE)
  }
  error <- Inf; attempts <- 0L; selected <- NA_integer_
  if (unit_grid && initial < maximum) {
    coarse <- probe("fixed_rk4", initial)
    n <- as.integer(initial)
    while (n < maximum) {
      fine_n <- as.integer(min(2 * n, maximum))
      fine <- probe("fixed_rk4", fine_n)
      attempts <- attempts + 1L
      error <- compare(coarse, fine, rtol, atol)
      if (is.finite(error) && error <= 1) {
        selected <- fine_n
        break
      }
      coarse <- fine
      n <- fine_n
    }
  }
  used_fallback <- is.na(selected)
  if (used_fallback && identical(fallback, "error")) {
    stop(sprintf("RK4 accuracy check failed (scaled discrepancy %s; budget %s); no update applied.",
                 format(error), maximum), call. = FALSE)
  }
  list(integrator = if (used_fallback) "diffrax" else "fixed_rk4",
       substeps = if (used_fallback) 1L else selected,
       scaled_discrepancy = error, comparisons = attempts,
       fallback = used_fallback, unit_grid = unit_grid)
}

.ndm_select_training_solver <- function(env, model, batch, state, prior, policy, info, seed) {
  if (!identical(env$ModelType, "NeuralODE") ||
      !identical(env$NeuralODETrainIntegrator, "fixed_rk4")) return(info)
  host <- function(x) as.numeric(env$np$asarray(x))
  grid_ok <- function(x) {
    t <- host(x)
    length(t) > 0L && all(is.finite(t)) && t[[1L]] == 0 && all(diff(t) == 1)
  }
  unit_grid <- grid_ok(info[[2L]]$subs$ts)
  if (length(env$uq_globalneural_vec)) unit_grid <- unit_grid && grid_ok(env$VI_SaveAt_ODE_GlobalNeural$subs$ts)
  controller <- env$stepsize_controller_optim
  rtol <- tryCatch(host(controller$rtol), error = function(e) env$neuralode_optim_rtol %||% 1e-5)
  atol <- tryCatch(host(controller$atol), error = function(e) env$neuralode_optim_atol %||% 1e-7)
  probe <- function(integrator, substeps) {
    probe_info <- info
    # Plain R scalars are static Equinox arguments. Each solver/resolution has
    # its own compiled specialization; no captured setting is mutated.
    probe_info[[3L]] <- list(integrator = integrator, substeps = as.integer(substeps))
    env$GetPred_train_jit(model, batch, state, prior, policy, probe_info, seed)[[1L]]
  }
  compare <- function(a, b, rtol, atol) {
    jnp <- env$jnp
    leaves_a <- env$jax$tree_util$tree_leaves(a$numerical_states)
    leaves_b <- env$jax$tree_util$tree_leaves(b$numerical_states)
    if (!length(leaves_a) || length(leaves_a) != length(leaves_b)) return(Inf)
    discrepancy <- lapply(seq_along(leaves_a), function(i) {
      x <- leaves_a[[i]]; y <- leaves_b[[i]]
      jnp$max(jnp$abs(x - y) / (atol + rtol * jnp$maximum(jnp$abs(x), jnp$abs(y))))
    })
    success <- jnp$logical_and(jnp$all(a$solver_diagnostics$success), jnp$all(b$solver_diagnostics$success))
    host(jnp$where(success, jnp$max(jnp$stack(discrepancy)), jnp$inf))
  }
  decision <- .ndm_refine_rk4(
    probe, compare, env$NeuralODETrainSubsteps,
    env$NeuralODETrainMaxSubsteps %||% 128L, rtol, atol,
    env$NeuralODETrainFallback %||% "diffrax", unit_grid
  )
  # Keep floating diagnostics separate from the integer solver-status packet.
  env$LastSolverAccuracy <- decision
  info[[3L]] <- c(decision[c("integrator", "substeps")], list(rtol = rtol, atol = atol))
  info
}
