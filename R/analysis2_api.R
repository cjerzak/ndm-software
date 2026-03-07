#' Deprecated Analysis2 package runners
#'
#' `ndm` no longer owns runnable Analysis2 orchestration. Use the local
#' `Analysis2/SuperLModel_MasterReal.R` and `Analysis2/SuperLModel_MasterSim.R`
#' entrypoints in the project checkout instead.
#'
#' @param args Character vector of command-line-style arguments. Retained only
#'   for compatibility with the previous function signatures.
#'
#' @returns These functions do not return; they always error with migration
#'   guidance.
#'
#' @examples
#' \dontrun{
#' ndm_run_analysis2_real()
#' }
#'
#' @export
ndm_run_analysis2_real <- function(args = commandArgs(TRUE)) {
  stop(
    "`ndm_run_analysis2_real()` is deprecated. Run the project-local ",
    "`Analysis2/SuperLModel_MasterReal.R` entrypoint instead.",
    call. = FALSE
  )
}

#' @rdname ndm_run_analysis2_real
#' @export
ndm_run_analysis2_sim <- function(args = commandArgs(TRUE)) {
  stop(
    "`ndm_run_analysis2_sim()` is deprecated. Run the project-local ",
    "`Analysis2/SuperLModel_MasterSim.R` entrypoint instead.",
    call. = FALSE
  )
}
