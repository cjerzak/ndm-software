ndm_make_oryx_shim <- function(jax, jnp, np) {
  stopifnot(!is.null(jax), !is.null(jnp), !is.null(np))

  scale_floor <- 1e-6
  covariance_jitter <- 1e-6

  is_py_object <- function(x) {
    "python.builtin.object" %in% class(x)
  }

  ensure_key <- function(seed) {
    if (is.null(seed)) {
      stop("seed must be provided", call. = FALSE)
    }

    if (is.numeric(seed)) {
      return(jax$random$PRNGKey(as.integer(seed)))
    }

    try({
      shp <- seed$shape
      if (!is.null(shp) && length(shp) == 1L && as.integer(shp[[1]]) == 2L) {
        return(seed)
      }
    }, silent = TRUE)

    shape <- tryCatch(seed$shape, error = function(e) NULL)
    if (!is.null(shape) && length(shape) == 0L) {
      return(jax$random$PRNGKey(as.integer(np$asarray(seed))))
    }

    jax$random$PRNGKey(as.integer(reticulate::py_to_r(seed)))
  }

  shape_of <- function(x) {
    tryCatch(as.integer(unlist(x$shape)), error = function(e) integer(0L))
  }

  as_shape <- function(x) {
    if (is.null(x)) {
      return(integer(0L))
    }
    if (is.list(x)) {
      return(as.integer(unlist(x)))
    }
    if (is_py_object(x)) {
      shp <- tryCatch(as.integer(unlist(x$shape)), error = function(e) NULL)
      if (!is.null(shp)) {
        return(shp)
      }
    }
    as.integer(x)
  }

  make_dist <- function(name, params, sample_fn, log_prob_fn = NULL) {
    list(
      name = name,
      parameters = params,
      sample = sample_fn,
      log_prob = if (is.null(log_prob_fn)) {
        function(...) stop("log_prob not implemented for ", name, call. = FALSE)
      } else {
        log_prob_fn
      }
    )
  }

  diagonal_matrix <- function(diagonal) {
    k <- as.integer(utils::tail(shape_of(diagonal), 1L))
    jnp$eye(k, dtype = diagonal$dtype) * jnp$expand_dims(diagonal, axis = -2L)
  }

  safe_scale_tril <- function(scale_tril) {
    diagonal <- jnp$diagonal(scale_tril, axis1 = -2L, axis2 = -1L)
    safe_diagonal <- jnp$maximum(jnp$abs(diagonal), scale_floor)
    scale_tril - diagonal_matrix(diagonal) + diagonal_matrix(safe_diagonal)
  }

  oryx <- list()

  oryx$Normal <- function(loc, scale) {
    if (!is_py_object(loc)) {
      loc <- jnp$array(loc)
    }
    if (!is_py_object(scale)) {
      scale <- jnp$array(scale)
    }
    scale <- jnp$maximum(jnp$abs(scale), scale_floor)

    sample_fn <- function(sample_shape = NULL, seed = NULL) {
      key <- ensure_key(seed)
      shp <- c(as_shape(sample_shape), shape_of(loc))
      eps <- jax$random$normal(key, shape = shp)
      eps * scale + loc
    }

    log_prob_fn <- function(x) {
      -0.5 * jnp$square((x - loc) / scale) - jnp$log(scale) - 0.5 * jnp$log(2 * pi)
    }

    make_dist("Normal", list(loc = loc, scale = scale), sample_fn, log_prob_fn)
  }

  oryx$Uniform <- function(low = 0, high = 1) {
    if (!is_py_object(low)) {
      low <- jnp$array(low)
    }
    if (!is_py_object(high)) {
      high <- jnp$array(high)
    }

    sample_fn <- function(sample_shape = NULL, seed = NULL) {
      key <- ensure_key(seed)
      jax$random$uniform(
        key = key,
        shape = as_shape(sample_shape),
        minval = low,
        maxval = high
      )
    }

    log_prob_fn <- function(x) {
      jnp$where((x >= low) & (x <= high), -jnp$log(high - low), -jnp$inf)
    }

    make_dist("Uniform", list(low = low, high = high), sample_fn, log_prob_fn)
  }

  oryx$MultivariateNormalDiag <- function(loc, scale_diag = NULL, scale = NULL) {
    sd <- scale_diag %||% scale
    if (is.null(sd)) {
      stop("Provide 'scale_diag' (preferred) or 'scale'.", call. = FALSE)
    }
    if (!is_py_object(loc)) {
      loc <- jnp$array(loc)
    }
    if (!is_py_object(sd)) {
      sd <- jnp$array(sd)
    }
    sd <- jnp$maximum(jnp$abs(sd), scale_floor)

    k <- as.integer(utils::tail(shape_of(loc), 1L))

    sample_fn <- function(sample_shape = NULL, seed = NULL) {
      key <- ensure_key(seed)
      shp <- c(as_shape(sample_shape), shape_of(loc))
      eps <- jax$random$normal(key, shape = shp)
      eps * sd + loc
    }

    log_prob_fn <- function(x) {
      diff <- (x - loc) / sd
      const <- 0.5 * as.numeric(k) * jnp$log(2 * pi)
      -0.5 * jnp$sum(jnp$square(diff) + 2 * jnp$log(sd), axis = -1L) - const
    }

    make_dist(
      "MultivariateNormalDiag",
      list(loc = loc, scale_diag = sd),
      sample_fn,
      log_prob_fn
    )
  }

  oryx$MultivariateNormalTriL <- function(loc, scale_tril) {
    if (!is_py_object(loc)) {
      loc <- jnp$array(loc)
    }
    if (!is_py_object(scale_tril)) {
      scale_tril <- jnp$array(scale_tril)
    }

    scale_tril <- safe_scale_tril(scale_tril)

    k <- as.integer(utils::tail(shape_of(loc), 1L))

    sample_fn <- function(sample_shape = NULL, seed = NULL) {
      key <- ensure_key(seed)
      shp <- c(as_shape(sample_shape), k)
      eps <- jax$random$normal(key, shape = shp)
      jnp$matmul(
        jnp$expand_dims(eps, axis = -2L),
        jnp$swapaxes(scale_tril, -1L, -2L)
      )$squeeze(axis = -2L) + loc
    }

    log_prob_fn <- function(x) {
      diff <- x - loc
      y <- jax$scipy$linalg$solve_triangular(
        scale_tril,
        jnp$expand_dims(diff, axis = -1L),
        lower = TRUE,
        check_finite = FALSE
      )$squeeze(axis = -1L)
      logdet <- jnp$sum(
        jnp$log(jnp$diagonal(scale_tril, axis1 = -2L, axis2 = -1L)),
        axis = -1L
      )
      const <- 0.5 * as.numeric(k) * jnp$log(2 * pi)
      -0.5 * jnp$sum(jnp$square(y), axis = -1L) - logdet - const
    }

    make_dist(
      "MultivariateNormalTriL",
      list(loc = loc, scale_tril = scale_tril),
      sample_fn,
      log_prob_fn
    )
  }

  oryx$MultivariateNormalFullCovariance <- function(loc, covariance_matrix) {
    if (!is_py_object(loc)) {
      loc <- jnp$array(loc)
    }
    if (!is_py_object(covariance_matrix)) {
      covariance_matrix <- jnp$array(covariance_matrix)
    }

    k <- as.integer(utils::tail(shape_of(loc), 1L))
    covariance_matrix <- 0.5 * (
      covariance_matrix + jnp$swapaxes(covariance_matrix, -1L, -2L)
    ) + covariance_jitter * jnp$eye(k, dtype = covariance_matrix$dtype)
    chol <- jnp$linalg$cholesky(covariance_matrix)
    out <- oryx$MultivariateNormalTriL(loc = loc, scale_tril = chol)
    out$name <- "MultivariateNormalFullCovariance"
    out$parameters$covariance_matrix <- covariance_matrix
    out
  }

  oryx$math <- list()
  oryx$math$fill_triangular <- function(vec) {
    if (!is_py_object(vec)) {
      vec <- jnp$array(vec)
    }
    n_elem <- as.integer(vec$size)
    m <- as.integer(floor((sqrt(8 * n_elem + 1) - 1) / 2))
    if (m * (m + 1) / 2 != n_elem) {
      stop("fill_triangular: length does not match any triangular number", call. = FALSE)
    }
    out <- jnp$zeros(list(m, m), dtype = vec$dtype)
    idx <- jnp$tril_indices(m)
    out$at[idx[[1]], idx[[2]]]$set(vec)
  }

  mvn_scale_tril <- function(dist) {
    if (!is.null(dist$parameters$scale_diag)) {
      return(diagonal_matrix(jnp$maximum(
        jnp$abs(dist$parameters$scale_diag),
        scale_floor
      )))
    }
    if (!is.null(dist$parameters$scale_tril)) {
      return(safe_scale_tril(dist$parameters$scale_tril))
    }
    NULL
  }

  oryx$kl_divergence <- function(q, p) {
    if (q$name == "Normal" && p$name == "Normal") {
      mu_q <- q$parameters$loc
      sd_q <- jnp$maximum(jnp$abs(q$parameters$scale), scale_floor)
      mu_p <- p$parameters$loc
      sd_p <- jnp$maximum(jnp$abs(p$parameters$scale), scale_floor)
      return(
        jnp$log(sd_p / sd_q) +
          (jnp$square(sd_q) + jnp$square(mu_q - mu_p)) /
            (2 * jnp$square(sd_p)) -
          0.5
      )
    }

    if (grepl("^MultivariateNormal", q$name) &&
        grepl("^MultivariateNormal", p$name)) {
      q_factor <- mvn_scale_tril(q)
      p_factor <- mvn_scale_tril(p)
      if (is.null(q_factor) || is.null(p_factor)) {
        stop(
          "KL divergence requires a diagonal, triangular, or full-covariance Gaussian factor.",
          call. = FALSE
        )
      }

      broadcast_factors <- jnp$broadcast_arrays(q_factor, p_factor)
      q_factor <- broadcast_factors[[1L]]
      p_factor <- broadcast_factors[[2L]]
      broadcast_locations <- jnp$broadcast_arrays(
        q$parameters$loc,
        p$parameters$loc
      )
      q_location <- broadcast_locations[[1L]]
      p_location <- broadcast_locations[[2L]]

      whitened_q_factor <- jax$scipy$linalg$solve_triangular(
        p_factor,
        q_factor,
        lower = TRUE,
        check_finite = FALSE
      )
      mean_difference <- p_location - q_location
      whitened_mean_difference <- jax$scipy$linalg$solve_triangular(
        p_factor,
        jnp$expand_dims(mean_difference, axis = -1L),
        lower = TRUE,
        check_finite = FALSE
      )$squeeze(axis = -1L)

      dimension <- as.numeric(utils::tail(shape_of(q$parameters$loc), 1L))
      trace_term <- jnp$sum(
        jnp$sum(jnp$square(whitened_q_factor), axis = -1L),
        axis = -1L
      )
      mahalanobis_term <- jnp$sum(
        jnp$square(whitened_mean_difference),
        axis = -1L
      )
      logdet_q <- 2 * jnp$sum(
        jnp$log(jnp$diagonal(q_factor, axis1 = -2L, axis2 = -1L)),
        axis = -1L
      )
      logdet_p <- 2 * jnp$sum(
        jnp$log(jnp$diagonal(p_factor, axis1 = -2L, axis2 = -1L)),
        axis = -1L
      )

      return(0.5 * (
        trace_term + mahalanobis_term - dimension + logdet_p - logdet_q
      ))
    }

    stop(
      sprintf("KL divergence is not implemented for %s || %s.", q$name, p$name),
      call. = FALSE
    )
  }

  oryx
}
