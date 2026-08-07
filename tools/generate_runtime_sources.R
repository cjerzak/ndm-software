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

escape_r_codepoint <- function(codepoint) {
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
  if (codepoint > 0x7FL) {
    if (codepoint <= 0xFFFFL) {
      return(sprintf("\\u%04X", codepoint))
    }
    return(sprintf("\\U%08X", codepoint))
  }
  intToUtf8(codepoint)
}

escape_r_string <- function(x) {
  pieces <- vapply(
    utf8ToInt(enc2utf8(x)),
    escape_r_codepoint,
    character(1L),
    USE.NAMES = FALSE
  )
  paste0(pieces, collapse = "")
}

escape_r_string_chunks <- function(x, max_chars = 6000L) {
  pieces <- vapply(
    utf8ToInt(enc2utf8(x)),
    escape_r_codepoint,
    character(1L),
    USE.NAMES = FALSE
  )
  if (length(pieces) == 0L) {
    return("")
  }

  chunks <- character()
  current <- character()
  current_chars <- 0L
  for (piece in pieces) {
    piece_chars <- nchar(piece, type = "chars", allowNA = FALSE, keepNA = FALSE)
    if (current_chars > 0L && current_chars + piece_chars > max_chars) {
      chunks <- c(chunks, paste0(current, collapse = ""))
      current <- character()
      current_chars <- 0L
    }
    current <- c(current, piece)
    current_chars <- current_chars + piece_chars
  }
  if (length(current) > 0L) {
    chunks <- c(chunks, paste0(current, collapse = ""))
  }
  chunks
}

repo_root <- normalizePath(file.path(dirname(resolve_script_path()), ".."), winslash = "/", mustWork = TRUE)
runtime_root <- file.path(repo_root, "tools", "runtime_source", "ndm_runtime")
output_path <- file.path(repo_root, "R", "runtime_sources.R")
temporary_output_path <- tempfile(
  pattern = ".runtime_sources-",
  tmpdir = dirname(output_path),
  fileext = ".R"
)
on.exit(unlink(temporary_output_path), add = TRUE)

runtime_files <- list.files(runtime_root, recursive = TRUE, full.names = FALSE)
runtime_files <- sort(runtime_files[file.info(file.path(runtime_root, runtime_files))$isdir %in% FALSE])
runtime_text <- lapply(runtime_files, function(path) {
  paste(
    readLines(file.path(runtime_root, path), warn = FALSE, encoding = "UTF-8"),
    collapse = "\n"
  )
})
names(runtime_text) <- runtime_files

con <- file(temporary_output_path, open = "w", encoding = "UTF-8")
on.exit(if (isOpen(con)) close(con), add = TRUE)

writeLines("# Generated package-owned runtime sources.", con = con)
writeLines("# This file replaces the old installed inst/extdata runtime snapshot as the runtime source of truth.", con = con)
writeLines("", con = con)
writeLines(".ndm_embedded_runtime_sources <-", con = con)
writeLines("list(", con = con)

for (i in seq_along(runtime_files)) {
  comma <- if (i < length(runtime_files)) "," else ""
  chunks <- escape_r_string_chunks(runtime_text[[i]])
  if (length(chunks) == 1L) {
    writeLines(
      sprintf(
        "  `%s` = \"%s\"%s",
        runtime_files[[i]],
        chunks[[1L]],
        comma
      ),
      con = con
    )
    next
  }

  writeLines(sprintf("  `%s` = paste0(", runtime_files[[i]]), con = con)
  for (chunk_index in seq_along(chunks)) {
    chunk_comma <- if (chunk_index < length(chunks)) "," else ""
    writeLines(sprintf("    \"%s\"%s", chunks[[chunk_index]], chunk_comma), con = con)
  }
  writeLines(sprintf("  )%s", comma), con = con)
}

writeLines(")", con = con)
close(con)

# Fail at generation time if the serialized UTF-8 payload is not valid R code.
invisible(parse(file = temporary_output_path))

generated_bytes <- readBin(
  temporary_output_path,
  what = "raw",
  n = file.info(temporary_output_path)$size
)
if (any(as.integer(generated_bytes) > 0x7F)) {
  stop("Generated runtime source file must contain only ASCII bytes.", call. = FALSE)
}
if (!file.rename(temporary_output_path, output_path)) {
  stop("Could not atomically publish generated runtime sources.", call. = FALSE)
}
