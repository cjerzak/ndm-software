test_that("the equation compiler preserves precedence and rejects executable syntax", {
  env <- new.env(parent = baseenv())
  env$jnp <- list(add = `+`, subtract = `-`, multiply = `*`, divide = `/`, power = `^`,
                  negative = function(x) -x, exp = exp, sqrt = sqrt, log = log,
                  maximum = pmax, minimum = pmin)
  env$x <- 2
  cases <- c(`-x^2` = -4, `(-x)^2` = 4, `-(x^2)` = -4, `1e-3*x` = .002,
             `x^-2` = .25, `2^3^2` = 512, `+x` = 2, `max(1,x,3)` = 3)
  for (expr in names(cases)) {
    compiled <- ndm:::.ndm_compile_math_expr(expr, "x")
    expect_equal(eval(parse(text = compiled), env), unname(cases[[expr]]))
  }
  normalized <- ndm:::.ndm_normalize_math_text("\\frac{x^{2+1}}{\\frac{2}{4}}")
  expect_equal(eval(parse(text = ndm:::.ndm_compile_math_expr(normalized, "x")), env), 16)
  for (expr in c("x <- 1", "x;2", "system('true')", "foo(x)", "x+y", "log(x,2)")) {
    expect_error(ndm:::.ndm_compile_math_expr(expr, "x"), "Invalid ODE expression")
  }
})

test_that("Gaussian samples broadcast independent batch noise", {
  env <- ndm_test_architecture_env()
  jnp <- env$jnp; jax <- env$jax; d <- env$oryx
  ar <- function(x) as.array(env$np$array(x))
  normal <- d$Normal(jnp$array(0.), jnp$array(c(1., 2., 3.)))
  draws <- ar(normal$sample(list(10000L), seed = 17L))
  expect_equal(dim(draws), c(10000L, 3L))
  expect_lt(max(abs(cor(draws)[upper.tri(cor(draws))])), .05)
  expect_equal(unname(apply(draws, 2, sd)), c(1, 2, 3), tolerance = .06)
  factor <- jnp$array(matrix(c(1, .5, 0, 2), 2))
  tril <- d$MultivariateNormalTriL(jnp$zeros(list(3L, 2L)), factor)
  samples <- ar(tril$sample(seed = 2L))
  expect_equal(dim(samples), c(3L, 2L))
  expect_false(identical(samples[1, ], samples[2, ]))
  expect_equal(ar(tril$sample(seed = 2L)), samples, tolerance = 0)
  batched <- d$MultivariateNormalTriL(jnp$zeros(2L), jnp$broadcast_to(factor, list(3L, 2L, 2L)))
  expect_equal(dim(ar(batched$sample(list(4L, 5L), seed = 3L))), c(4L, 5L, 3L, 2L))
  diag <- d$MultivariateNormalDiag(jnp$zeros(2L), jnp$ones(list(3L, 2L)))
  expect_equal(dim(ar(diag$sample(list(7L), seed = 4L))), c(7L, 3L, 2L))
  compiled <- jax$jit(function(loc) d$MultivariateNormalTriL(loc, factor)$sample(seed = jax$random$PRNGKey(1L)))
  expect_equal(dim(ar(compiled(jnp$zeros(list(3L, 2L))))), c(3L, 2L))
  gradient <- jax$grad(function(loc) jnp$sum(compiled(loc)))(jnp$zeros(list(3L, 2L)))
  expect_equal(ar(gradient), matrix(1, 3, 2))
})

test_that("step doubling catches finite but inaccurate RK4 states and gradients", {
  env <- ndm_test_architecture_env()
  ndm_test_architecture_function("SetupEnv/SuperLModel_helperFxns.R", "ndm_runtime_fixed_step_solve", env)
  solve <- function(rate, n) env$ndm_runtime_fixed_step_solve(
    function(t, y, args) list(x = -rate * y$x), list(x = env$jnp$array(1.)),
    list(), env$jnp$array(c(0., 1.)), substeps = as.integer(n))$ys$x
  terminal <- function(rate, n) env$jnp$take(solve(rate, n), 1L)
  probe_calls <- integer()
  probe <- function(method, n) {
    probe_calls <<- c(probe_calls, n)
    as.numeric(env$np$array(solve(env$jnp$array(4.), n)))
  }
  compare <- function(a, b, rtol, atol) max(abs(b-a) / (atol + rtol * pmax(abs(a), abs(b))))
  expect_equal(tail(probe("fixed_rk4", 1L), 1L), 5)
  decision <- ndm:::.ndm_refine_rk4(probe, compare, 1L, 128L, 1e-5, 1e-7)
  expect_identical(decision$integrator, "fixed_rk4")
  expect_gt(decision$substeps, 2L)
  expect_lte(decision$scaled_discrepancy, 1)
  value <- as.numeric(env$np$array(terminal(env$jnp$array(4.), decision$substeps)))
  gradient <- env$jax$grad(function(rate) terminal(rate, decision$substeps))(env$jnp$array(4.))
  expect_equal(value, exp(-4), tolerance = 2e-6)
  expect_equal(as.numeric(env$np$array(gradient)), -exp(-4), tolerance = 2e-6)
  expect_false(anyDuplicated(probe_calls[-1L]) > 0L)
  fallback <- ndm:::.ndm_refine_rk4(probe, compare, 1L, 2L, 1e-5, 1e-7)
  expect_identical(fallback$integrator, "diffrax")
  expect_error(ndm:::.ndm_refine_rk4(probe, compare, 1L, 2L, 1e-5, 1e-7, fallback = "error"), "no update applied")
  expect_true(ndm:::.ndm_refine_rk4(probe, compare, 1L, 2L, 1e-5, 1e-7, unit_grid = FALSE)$fallback)
})

test_that("logistic-normal priors round trip and enforce identifiable coordinates", {
  terms <- c("s_l", "i_l", "r_l")
  prior <- ndm:::.ndm_default_initial_state_prior(terms)
  expect_equal(unname(prior$logratio_mean), c(-11, -11))
  expect_equal(unname(prior$logratio_cov), matrix(c(2, 1, 1, 2), 2))
  bad <- prior; bad$logratio_cov[1, 1] <- 0
  expect_error(ndm:::.ndm_validate_initial_state_prior(bad, terms), "positive definite")
  env <- ndm_test_architecture_env()
  draw <- function(z, terms, prior) ndm:::.ndm_initial_state_draw(z, prior, terms, env$jnp$array(100.), env$jax, env$jnp)
  value <- as.numeric(env$np$array(draw(env$jnp$array(prior$logratio_mean), terms, prior)))
  expect_equal(sum(value), 100, tolerance = 1e-6)
  expect_true(all(value > 0))
  singleton <- ndm:::.ndm_default_initial_state_prior("s_l")
  expect_length(singleton$logratio_mean, 0L)
  expect_equal(as.numeric(env$np$array(draw(env$jnp$array(numeric()), "s_l", singleton))), 100)
  terms <- c("s_l", "e_l", "i_l", "r_l")
  prior <- ndm:::.ndm_default_initial_state_prior(terms)
  prior$logratio_mean[["i_l"]] <- -4
  spec <- ndm_model_spec(preset = "seirs_dynamic_beta", initial_state_prior = prior)
  file <- tempfile(fileext = ".tex"); on.exit(unlink(file), add = TRUE)
  ndm_model_spec_to_tex(spec, file)
  restored <- ndm_model_spec_from_tex(file)
  expect_equal(restored$execution_spec$initial_state_prior, prior)
  expect_error(ndm_model_spec(preset = "tb_a", family_args = list(balanced_i0_prior_sd = .01)), "Obsolete")
  expect_error(ndm_model_spec(preset = "tb_a", family_args = list(initial_incidence_center = .01), initial_state_prior = prior), "either")
})

test_that("joint Gaussian KL honors correlated log-ratio covariance without jitter", {
  env <- ndm_test_architecture_env()
  covariance <- matrix(c(2, .8, .8, 1), 2L)
  q_cov <- matrix(c(1.1, -.2, -.2, .6), 2L)
  mu <- c(.3, -.4)
  p <- env$oryx$MultivariateNormalTriL(env$jnp$zeros(2L), env$jnp$array(t(chol(covariance))), stabilize = FALSE)
  q <- env$oryx$MultivariateNormalTriL(env$jnp$array(mu), env$jnp$array(t(chol(q_cov))), stabilize = FALSE)
  expected <- .5 * (sum(diag(solve(covariance, q_cov))) + drop(t(mu) %*% solve(covariance, mu)) - 2 +
                     as.numeric(determinant(covariance, logarithm = TRUE)$modulus - determinant(q_cov, logarithm = TRUE)$modulus))
  expect_equal(as.numeric(env$np$array(env$oryx$kl_divergence(q, p))), expected, tolerance = 1e-6)
  # A shared factor with batched locations still represents distinct Gaussians.
  locations <- rbind(mu, 2 * mu, -mu)
  batched <- env$oryx$MultivariateNormalTriL(env$jnp$array(locations),
    env$jnp$array(t(chol(q_cov))), stabilize = FALSE)
  expected_batch <- vapply(seq_len(nrow(locations)), function(i) {
    location <- locations[i, ]
    expected + .5 * (drop(t(location) %*% solve(covariance, location)) -
                       drop(t(mu) %*% solve(covariance, mu)))
  }, numeric(1L))
  expect_equal(as.numeric(env$np$array(env$oryx$kl_divergence(batched, p))),
               expected_batch, tolerance = 1e-6)
  sample <- as.array(env$np$array(p$sample(list(15000L), seed = 61L)))
  expect_equal(unname(cov(sample)), covariance, tolerance = .05)
  tiny <- matrix(1e-16, 1L)
  exact <- env$oryx$MultivariateNormalTriL(env$jnp$zeros(1L), env$jnp$array(t(chol(tiny))), stabilize = FALSE)
  expect_lt(abs(as.numeric(env$np$array(exact$parameters$scale_tril)) - 1e-8), 1e-14)
})
