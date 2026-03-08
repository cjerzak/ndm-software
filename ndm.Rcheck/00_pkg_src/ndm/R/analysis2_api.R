#' Legacy Analysis2-compatible package runners
#'
#' These wrappers preserve the old `ndm_run_analysis2_*()` entrypoints while
#' routing them to the package-owned runtime bundle. They accept the same
#' command-line-style argument vectors as before, but no longer depend on an
#' external `Analysis2` checkout.
#'
#' @param args Character vector of command-line-style arguments.
#'
#' @returns The underlying workflow result, including dry-run previews.
#'
#' @examples
#' \dontrun{
#' ndm_run_analysis2_real("--help")
#' }
#'
#' @export
ndm_run_analysis2_real <- function(args = commandArgs(TRUE)) {
  .ndm_invoke_legacy_analysis2_runner("real", args = args)
}

#' @rdname ndm_run_analysis2_real
#' @export
ndm_run_analysis2_sim <- function(args = commandArgs(TRUE)) {
  .ndm_invoke_legacy_analysis2_runner("sim", args = args)
}
