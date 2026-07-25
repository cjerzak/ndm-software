.ndm_solver_failure_condition <- function(iteration = NA_integer_,
                                          example_indices,
                                          stages,
                                          diagnostics = NULL,
                                          context = list()) {
  example_indices <- as.integer(example_indices)
  stages <- as.character(stages)
  if (length(example_indices) == 0L) {
    stop("`example_indices` must identify at least one failed example.", call. = FALSE)
  }
  if (length(stages) == 1L && length(example_indices) > 1L) {
    stages <- rep(stages, length(example_indices))
  }
  if (length(stages) != length(example_indices)) {
    stop("`stages` must have length one or match `example_indices`.", call. = FALSE)
  }
  if (!is.list(context)) {
    stop("`context` must be a list.", call. = FALSE)
  }

  iteration_label <- if (
    length(iteration) == 1L && !is.na(suppressWarnings(as.integer(iteration)))
  ) {
    sprintf(" at iteration %s", as.integer(iteration))
  } else {
    ""
  }
  failure_labels <- paste0(example_indices, " [", stages, "]")
  message <- sprintf(
    "ODE solver failure%s for example(s) %s; no candidate state was committed.",
    iteration_label,
    paste(failure_labels, collapse = ", ")
  )

  structure(
    list(
      message = message,
      call = NULL,
      iteration = suppressWarnings(as.integer(iteration)),
      example_indices = example_indices,
      stage = stages,
      diagnostics = diagnostics,
      context = context
    ),
    class = c("ndm_solver_failure", "error", "condition")
  )
}

.ndm_write_solver_failure_report <- function(report,
                                             holder_folder,
                                             outer_iteration = NA_integer_,
                                             iteration = NA_integer_) {
  if (!is.list(report)) {
    stop("`report` must be a list.", call. = FALSE)
  }
  if (!is.character(holder_folder) ||
      length(holder_folder) != 1L ||
      is.na(holder_folder) ||
      !nzchar(holder_folder)) {
    stop("`holder_folder` must be one non-empty path.", call. = FALSE)
  }
  dir.create(holder_folder, recursive = TRUE, showWarnings = FALSE)
  if (!dir.exists(holder_folder)) {
    stop("Could not create solver failure report directory.", call. = FALSE)
  }
  scalar_label <- function(value) {
    value <- suppressWarnings(as.integer(value))
    if (length(value) != 1L || is.na(value)) "NA" else as.character(value)
  }
  report_path <- file.path(
    holder_folder,
    sprintf(
      "solver_failure_outer%s_i%s_pid%s_%s.rds",
      scalar_label(outer_iteration),
      scalar_label(iteration),
      Sys.getpid(),
      format(Sys.time(), "%Y%m%dT%H%M%OS6")
    )
  )
  temporary <- tempfile(
    "solver-failure-",
    tmpdir = holder_folder,
    fileext = ".rds"
  )
  on.exit(unlink(temporary, force = TRUE), add = TRUE)
  saveRDS(report, temporary)
  if (!file.rename(temporary, report_path)) {
    stop("Could not atomically publish solver failure report.", call. = FALSE)
  }
  normalizePath(report_path, winslash = "/", mustWork = TRUE)
}
