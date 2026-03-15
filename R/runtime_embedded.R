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

.ndm_embedded_runtime_key <- function(path) {
  .ndm_embedded_match_key(
    path = path,
    keys = names(.ndm_embedded_runtime_sources),
    label = "runtime"
  )
}

.ndm_embedded_runtime_source <- function(path) {
  .ndm_embedded_runtime_sources[[.ndm_embedded_runtime_key(path)]]
}

.ndm_embedded_runtime_lines <- function(path) {
  strsplit(.ndm_embedded_runtime_source(path), "\n", fixed = TRUE)[[1L]]
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

.ndm_embedded_analysis_root <- function(refresh = FALSE) {
  cached <- ndm_env$embedded_analysis_root %||% NULL
  if (!isTRUE(refresh) && !is.null(cached) && dir.exists(cached)) {
    return(.ndm_normalize_path(cached, must_work = TRUE))
  }

  root <- file.path(tempdir(), "ndm_embedded_analysis_runtime", "Analysis2")
  .ndm_write_embedded_tree(root, .ndm_embedded_runtime_sources)
  ndm_env$embedded_analysis_root <- root
  .ndm_normalize_path(root, must_work = TRUE)
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

.ndm_eval_embedded_runtime <- function(path, env) {
  stopifnot(is.environment(env))

  expr <- parse(text = .ndm_embedded_runtime_source(path), keep.source = FALSE)
  eval(expr, envir = env)
  invisible(env)
}

.ndm_normalize_runtime_relative <- function(path) {
  path <- gsub("\\\\", "/", as.character(path))
  path <- sub("^\\./", "", path)
  path <- sub("^.*/Analysis2/", "", path)
  path
}

.ndm_has_embedded_runtime_relative <- function(relative_path) {
  rel <- .ndm_normalize_runtime_relative(relative_path)
  !inherits(try(.ndm_embedded_runtime_key(rel), silent = TRUE), "try-error")
}

.ndm_eval_embedded_runtime_relative <- function(relative_path, env) {
  .ndm_eval_embedded_runtime(.ndm_normalize_runtime_relative(relative_path), env)
}

.ndm_eval_runtime_path <- function(path, env, analysis_root = NULL) {
  rel <- .ndm_normalize_runtime_relative(path)
  if (.ndm_has_embedded_runtime_relative(rel)) {
    return(.ndm_eval_embedded_runtime_relative(rel, env))
  }

  if (!is.null(analysis_root) && nzchar(analysis_root)) {
    candidate <- file.path(analysis_root, rel)
    if (file.exists(candidate)) {
      sys.source(candidate, envir = env, keep.source = FALSE)
      return(invisible(env))
    }
  }

  if (file.exists(path)) {
    sys.source(path, envir = env, keep.source = FALSE)
    return(invisible(env))
  }

  stop(
    "Runtime component could not be resolved from the embedded registry or filesystem: ",
    path,
    call. = FALSE
  )
}
