# A checkpoint is evaluated once, including the runner's terminal fallback.
ndm_evaluate_checkpoint <- function(env) {
  previous <- get0("ndm_last_evaluation_file", envir = env, inherits = FALSE)
  if (identical(get0("ndm_last_evaluated_step", envir = env, inherits = FALSE), as.integer(env$i)) &&
      !is.null(previous) && file.exists(previous) && file.exists(paste0(previous, ".evaluation.rds"))) {
    receipt <- readRDS(paste0(previous, ".evaluation.rds"))
    if (identical(receipt$result_sha256, digest::digest(file = previous, algo = "sha256"))) return(invisible(NULL))
  }
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) old_seed <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    if (had_seed) assign(".Random.seed", old_seed, envir = .GlobalEnv)
    else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) rm(".Random.seed", envir = .GlobalEnv)
  }, add = TRUE)
  env$ndm_source_extracted("ResultsGet/SuperLModel_GetAnalytics_Real.R", env_target = env)
  env$ndm_source_extracted("ResultsGet/SuperLModel_GetAnalytics_Sim.R", env_target = env)
  env$ndm_last_evaluated_step <- as.integer(env$i)
  invisible(NULL)
}
ndm_evaluate_checkpoint(environment())
