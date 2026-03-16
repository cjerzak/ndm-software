.ndm_embedded_match_key <- function(path, keys, label) {
  if (is.null(path) || !nzchar(path)) {
    stop("Missing ", label, " path.", call. = FALSE)
  }

  normalized <- gsub("\\\\", "/", as.character(path))
  if (normalized %in% keys) {
    return(normalized)
  }

  matches <- keys[endsWith(normalized, keys)]
  if (length(matches) == 1L) {
    return(matches[[1L]])
  }

  stop(
    "Unknown embedded ",
    label,
    " path: ",
    normalized,
    call. = FALSE
  )
}

.ndm_embedded_model_spec_key <- function(path) {
  .ndm_embedded_match_key(
    path = path,
    keys = names(.ndm_embedded_model_spec_sources),
    label = "model spec"
  )
}

.ndm_embedded_model_spec_text <- function(path) {
  .ndm_embedded_model_spec_sources[[.ndm_embedded_model_spec_key(path)]]
}

.ndm_write_embedded_tree <- function(root, sources) {
  stopifnot(is.character(root), length(root) == 1L, is.list(sources))

  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  for (relative_path in names(sources)) {
    target <- file.path(root, relative_path)
    dir.create(dirname(target), recursive = TRUE, showWarnings = FALSE)
    writeLines(sources[[relative_path]], target, useBytes = TRUE)
  }

  invisible(.ndm_normalize_path(root, must_work = TRUE))
}

.ndm_embedded_model_spec_dir <- function(refresh = FALSE) {
  cached <- ndm_env$embedded_model_spec_dir %||% NULL
  if (!isTRUE(refresh) && !is.null(cached) && dir.exists(cached)) {
    return(.ndm_normalize_path(cached, must_work = TRUE))
  }

  root <- file.path(tempdir(), "ndm_embedded_model_specs")
  .ndm_write_embedded_tree(root, .ndm_embedded_model_spec_sources)
  ndm_env$embedded_model_spec_dir <- root
  .ndm_normalize_path(root, must_work = TRUE)
}

.ndm_embedded_model_spec_path <- function(path) {
  file.path(.ndm_embedded_model_spec_dir(), .ndm_embedded_model_spec_key(path))
}
