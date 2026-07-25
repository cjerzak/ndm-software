test_that("Student-t likelihood uses the incidence-scale default floor", {
  expect_equal(formals(ndm:::.ndm_student_t_masked_nll)$scale_floor, 1e-5)
})

test_that("masked df=4 Student-t loss is a mean of per-example observed sums", {
  conda_env <- ndm_require_backend_test_stack("Student-t likelihood tests", packages = c("reticulate"))
  old_backend <- ndm:::ndm_env$backend
  on.exit(assign("backend", old_backend, envir = ndm:::ndm_env), add = TRUE)
  backend <- ndm_initialize_backend(
    conda_env = conda_env,
    float_type = "64",
    import_tensorflow = FALSE
  )
  as_r <- function(x) reticulate::py_to_r(backend$np$asanyarray(x))

  y <- array(c(0.25, -0.5, 1.25, 2, NaN, -999), dim = c(2, 3, 1))
  location <- array(c(0, 0.1, 0.5, -0.25, 9, 100), dim = c(2, 3, 1))
  scale <- array(c(0.5, 1, 2, 0.75, 1, 1), dim = c(2, 3, 1))
  mask <- array(c(1, 1, 1, 0, 0, 0), dim = c(2, 3, 1))
  result <- ndm:::.ndm_student_t_masked_nll(
    backend$jax,
    backend$jnp,
    backend$jnp$array(y),
    backend$jnp$array(location),
    backend$jnp$array(scale),
    backend$jnp$array(mask),
    df = 4,
    scale_floor = 1e-3
  )

  expected_element <- -stats::dt(
    (y - location) / scale,
    df = 4,
    log = TRUE
  ) + log(scale)
  expected_by_example <- vapply(seq_len(dim(y)[[1L]]), function(i) {
    sum(expected_element[i, , ][mask[i, , ] != 0], na.rm = FALSE)
  }, numeric(1))
  expect_equal(as.numeric(as_r(result$nll_by_example)), expected_by_example, tolerance = 1e-10)
  expect_equal(as.numeric(as_r(result$loss)), mean(expected_by_example), tolerance = 1e-10)
  expect_true(is.finite(as.numeric(as_r(result$loss))))
})

test_that("Student-t tail algebra remains finite for extreme representable residuals and gradients", {
  conda_env <- ndm_require_backend_test_stack("Student-t extreme-tail tests", packages = c("reticulate"))
  old_backend <- ndm:::ndm_env$backend
  on.exit(assign("backend", old_backend, envir = ndm:::ndm_env), add = TRUE)
  backend <- ndm_initialize_backend(
    conda_env = conda_env,
    float_type = "64",
    import_tensorflow = FALSE
  )
  jnp <- backend$jnp
  y <- jnp$array(array(1e150, dim = c(1, 1, 1)))
  mask <- jnp$array(array(1, dim = c(1, 1, 1)))
  scale <- jnp$array(array(1e-3, dim = c(1, 1, 1)))
  location <- jnp$array(array(0, dim = c(1, 1, 1)))
  loss_fn <- function(mu) {
    ndm:::.ndm_student_t_masked_nll(
      backend$jax,
      jnp,
      y,
      mu,
      scale,
      mask,
      df = 4,
      scale_floor = 1e-3
    )$loss
  }
  scale_loss_fn <- function(scale_value) {
    ndm:::.ndm_student_t_masked_nll(
      backend$jax,
      jnp,
      y,
      location,
      scale_value,
      mask,
      df = 4,
      scale_floor = 1e-3
    )$loss
  }
  loss <- loss_fn(location)
  gradient <- backend$jax$grad(loss_fn)(location)
  scale_gradient <- backend$jax$grad(scale_loss_fn)(jnp$array(array(1, dim = c(1, 1, 1))))
  expect_true(is.finite(as.numeric(backend$np$asanyarray(loss))))
  expect_true(all(is.finite(as.numeric(backend$np$asanyarray(gradient)))))
  expect_true(all(is.finite(as.numeric(backend$np$asanyarray(scale_gradient)))))
})

test_that("runtime model loss delegates to the shared Student-t helper", {
  build_source <- paste(
    deparse(ndm:::.ndm_stage_expr_ModelDefiners_SuperLModel_BuildML),
    collapse = "\n"
  )
  expect_match(build_source, "ndm_student_t_masked_nll\\(", perl = TRUE)
  expect_match(build_source, "df = 4", fixed = TRUE)
  expect_match(build_source, "scale_floor = ObservationScaleFloor", fixed = TRUE)
  expect_false(grepl("loss with MSE", build_source, fixed = TRUE))
})
