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
  codepoints <- utf8ToInt(enc2utf8(x))
  pieces <- vapply(
    codepoints,
    function(codepoint) {
      if (codepoint == 0x5CL) {
        return("\\\\")
      }
      if (codepoint == 0x22L) {
        return("\\\"")
      }
      if (codepoint == 0x0AL) {
        return("\\n")
      }
      if (codepoint == 0x0DL) {
        return("\\r")
      }
      if (codepoint == 0x09L) {
        return("\\t")
      }
      if (codepoint >= 0x20L && codepoint <= 0x7EL) {
        return(intToUtf8(codepoint))
      }
      if (codepoint <= 0xFFFFL) {
        return(sprintf("\\u%04X", codepoint))
      }
      sprintf("\\U%08X", codepoint)
    },
    character(1L),
    USE.NAMES = FALSE
  )
  paste0(pieces, collapse = "")
}

repo_root <- normalizePath(file.path(dirname(resolve_script_path()), ".."), winslash = "/", mustWork = TRUE)
runtime_root <- file.path(repo_root, "tools", "runtime_source", "ndm_runtime")
output_path <- file.path(repo_root, "R", "runtime_sources.R")

runtime_files <- list.files(runtime_root, recursive = TRUE, full.names = FALSE)
runtime_files <- sort(runtime_files[file.info(file.path(runtime_root, runtime_files))$isdir %in% FALSE])
runtime_text <- lapply(runtime_files, function(path) {
  paste(
    readLines(file.path(runtime_root, path), warn = FALSE, encoding = "UTF-8"),
    collapse = "\n"
  )
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
