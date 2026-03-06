ndm_make_oryx_shim <- function(jax, jnp, np) {
  stopifnot(!is.null(jax), !is.null(jnp), !is.null(np))

  is_py_object <- function(x) {
    "python.builtin.object" %in% class(x)
  }

  ensure_key <- function(seed) {
    if (is.null(seed)) {
      stop("seed must be provided", call. = FALSE)
    }

    try({
      shp <- seed$shape
      if (!is.null(shp) && length(shp) == 1L && as.integer(shp[[1]]) == 2L) {
        return(seed)
      }
    }, silent = TRUE)

    if (!is.null(seed$shape) && length(seed$shape) == 0L) {
      return(jax$random$PRNGKey(as.integer(np$asarray(seed))))
    }

    if (is.numeric(seed)) {
      return(jax$random$PRNGKey(as.integer(seed)))
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

  oryx <- list()

  oryx$Normal <- function(loc, scale) {
    if (!is_py_object(loc)) {
      loc <- jnp$array(loc)
    }
    if (!is_py_object(scale)) {
      scale <- jnp$array(scale)
    }

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

    k <- as.integer(utils::tail(shape_of(loc), 1L))

    sample_fn <- function(sample_shape = NULL, seed = NULL) {
      key <- ensure_key(seed)
      shp <- c(as_shape(sample_shape), k)
      eps <- jax$random$normal(key, shape = shp)
      jnp$dot(eps, jnp$transpose(scale_tril)) + loc
    }

    log_prob_fn <- function(x) {
      diff <- x - loc
      y <- jax$scipy$linalg$solve_triangular(
        scale_tril,
        jnp$transpose(diff),
        lower = TRUE,
        check_finite = FALSE
      )
      y <- jnp$transpose(y)
      logdet <- jnp$sum(jnp$log(jnp$diag(scale_tril)))
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

    chol <- jnp$linalg$cholesky(covariance_matrix)
    out <- oryx$MultivariateNormalTriL(loc = loc, scale_tril = chol)
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

  oryx$kl_divergence <- function(p, q) {
    if (p$name == "Normal" && q$name == "Normal") {
      mu1 <- p$parameters$loc
      sd1 <- p$parameters$scale
      mu2 <- q$parameters$loc
      sd2 <- q$parameters$scale
      return(
        jnp$log(sd2 / sd1) +
          (jnp$square(sd1) + jnp$square(mu1 - mu2)) / (2 * jnp$square(sd2)) -
          0.5
      )
    }

    if (grepl("MultivariateNormal", p$name) &&
        grepl("MultivariateNormal", q$name) &&
        !is.null(p$parameters$scale_diag) &&
        !is.null(q$parameters$scale_diag)) {
      s1 <- p$parameters$scale_diag
      s2 <- q$parameters$scale_diag
      m1 <- p$parameters$loc
      m2 <- q$parameters$loc
      k <- as.numeric(utils::tail(shape_of(m1), 1L))
      term2 <- jnp$sum(jnp$square(s1) / jnp$square(s2))
      term3 <- jnp$sum(jnp$square(m2 - m1) / jnp$square(s2))
      return(0.5 * (term2 + term3 - k))
    }

    jnp$zeros(list())
  }

  oryx
}
