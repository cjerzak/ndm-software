.ndm_checkpoint_save_steps <- function(n_steps, n_checkpoints) {
  n_steps <- max(1L, as.integer(n_steps))
  n_checkpoints <- max(0L, as.integer(n_checkpoints))
  if (n_checkpoints == 0L) {
    return(integer())
  }
  if (n_checkpoints == 1L || n_steps == 1L) {
    return(n_steps)
  }
  first_step <- min(10L, n_steps)
  intermediate <- round(10^seq(
    log(first_step, base = 10),
    log(n_steps, base = 10),
    length.out = n_checkpoints
  ))
  sort(unique(c(as.integer(intermediate), n_steps)))
}
