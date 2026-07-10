.ndm_student_t_masked_nll <- function(jax,
                                      jnp,
                                      y,
                                      location,
                                      scale,
                                      mask,
                                      df = 4,
                                      scale_floor = 1e-3) {
  df <- jnp$array(as.numeric(df), dtype = location$dtype)
  scale <- jnp$maximum(
    jnp$abs(scale),
    jnp$array(as.numeric(scale_floor), dtype = location$dtype)
  )
  mask <- mask$astype(location$dtype)
  safe_y <- jnp$where(mask > 0, y, location)
  absolute_residual <- jnp$abs(safe_y - location)
  log_squared_standardized_residual_over_df <- (
    2 * jnp$log(
      absolute_residual + jnp$array(1e-30, dtype = absolute_residual$dtype)
    ) -
      2 * jnp$log(scale) -
      jnp$log(df)
  )
  element_nll <- (
    jnp$log(scale) +
      0.5 * jnp$log(df * jnp$array(pi, dtype = scale$dtype)) +
      jax$scipy$special$gammaln(df / 2) -
      jax$scipy$special$gammaln((df + 1) / 2) +
      ((df + 1) / 2) * jnp$logaddexp(
        jnp$array(0, dtype = scale$dtype),
        log_squared_standardized_residual_over_df
      )
  )
  masked_nll <- jnp$where(mask > 0, element_nll, jnp$zeros_like(element_nll))
  nll_by_example <- jnp$sum(
    jnp$reshape(masked_nll, list(masked_nll$shape[[1]], -1L)),
    axis = 1L
  )
  list(
    loss = jnp$mean(nll_by_example),
    nll_by_example = nll_by_example,
    element_nll = element_nll
  )
}
