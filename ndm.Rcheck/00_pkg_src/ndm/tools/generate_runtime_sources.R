#!/usr/bin/env Rscript

resolve_script_path <- function() {
  source_frame <- tryCatch(sys.frame(1L)$ofile, error = function(...) NULL)
  if (!is.null(source_frame) && nzchar(source_frame)) {
    return(normalizePath(source_frame, winslash = "/", mustWork = TRUE))
  }

  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0L) {
    return(normalizePath(sub("^--file=", "", file_arg[[1L]]), winslash = "/", mustWork = TRUE))
  }

  stop("Unable to resolve the path to tools/generate_runtime_sources.R.", call. = FALSE)
}

escape_r_string <- function(x) {
  x <- gsub("\\\\", "\\\\\\\\", x)
  x <- gsub("\"", "\\\\\"", x)
  x <- gsub("\n", "\\\\n", x, fixed = TRUE)
  x
}

repo_root <- normalizePath(file.path(dirname(resolve_script_path()), ".."), winslash = "/", mustWork = TRUE)
runtime_root <- file.path(repo_root, "tools", "runtime_source", "ndm_runtime")
output_path <- file.path(repo_root, "R", "runtime_sources.R")

runtime_files <- list.files(runtime_root, recursive = TRUE, full.names = FALSE)
runtime_files <- sort(runtime_files[file.info(file.path(runtime_root, runtime_files))$isdir %in% FALSE])
runtime_text <- lapply(runtime_files, function(path) {
  paste(readLines(file.path(runtime_root, path), warn = FALSE), collapse = "\n")
})
names(runtime_text) <- runtime_files

con <- file(output_path, open = "w", encoding = "UTF-8")
on.exit(close(con), add = TRUE)

writeLines("# Generated package-owned runtime sources.", con = con)
writeLines("# This file replaces the old installed inst/extdata runtime snapshot as the runtime source of truth.", con = con)
writeLines("", con = con)
writeLines(".ndm_embedded_runtime_sources <-", con = con)
writeLines("list(", con = con)

for (i in seq_along(runtime_files)) {
  comma <- if (i < length(runtime_files)) "," else ""
  writeLines(
    sprintf(
      "  `%s` = \"%s\"%s",
      runtime_files[[i]],
      escape_r_string(runtime_text[[i]]),
      comma
    ),
    con = con
  )
}

writeLines(")", con = con)
