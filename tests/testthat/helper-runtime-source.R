ndm_test_runtime_source_text <- function(relative_path) {
  source_root <- try(ndm:::.ndm_runtime_source_root(), silent = TRUE)
  if (!inherits(source_root, "try-error") &&
      !is.null(source_root) &&
      dir.exists(source_root)) {
    source_path <- file.path(source_root, relative_path)
    if (file.exists(source_path)) {
      return(paste(
        readLines(source_path, warn = FALSE, encoding = "UTF-8"),
        collapse = "\n"
      ))
    }
  }

  source_text <- try(
    ndm:::.ndm_embedded_runtime_source(relative_path),
    silent = TRUE
  )
  if (inherits(source_text, "try-error") ||
      length(source_text) != 1L ||
      is.na(source_text)) {
    stop(
      "Package-managed runtime source is unavailable for `",
      relative_path,
      "`.",
      call. = FALSE
    )
  }
  source_text
}

ndm_test_runtime_source_expressions <- function(relative_path) {
  parse(
    text = ndm_test_runtime_source_text(relative_path),
    keep.source = FALSE
  )
}
