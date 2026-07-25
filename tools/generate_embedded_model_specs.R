#!/usr/bin/env Rscript

resolve_script_path <- function() {
  source_frame <- tryCatch(sys.frame(1L)$ofile, error = function(...) NULL)
  if (!is.null(source_frame) && nzchar(source_frame)) {
    return(normalizePath(source_frame, winslash = "/", mustWork = TRUE))
  }

  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0L) {
    return(normalizePath(
      sub("^--file=", "", file_arg[[1L]]),
      winslash = "/",
      mustWork = TRUE
    ))
  }

  stop(
    "Unable to resolve the path to tools/generate_embedded_model_specs.R.",
    call. = FALSE
  )
}

repo_root <- normalizePath(
  file.path(dirname(resolve_script_path()), ".."),
  winslash = "/",
  mustWork = TRUE
)
model_spec_root <- file.path(repo_root, "inst", "extdata", "model_specs")
runtime_model_spec_root <- file.path(
  repo_root,
  "tools",
  "runtime_source",
  "ndm_runtime",
  "ModelStructureTex"
)
output_path <- file.path(repo_root, "R", "sysdata.rda")

model_spec_files <- sort(list.files(
  model_spec_root,
  pattern = "[.]tex$",
  full.names = FALSE
))
if (length(model_spec_files) == 0L) {
  stop("No packaged model specifications were found.", call. = FALSE)
}

read_model_spec <- function(root, name) {
  paste(
    readLines(
      file.path(root, name),
      warn = FALSE,
      encoding = "UTF-8"
    ),
    collapse = "\n"
  )
}

.ndm_embedded_model_spec_sources <- lapply(
  model_spec_files,
  function(name) read_model_spec(model_spec_root, name)
)
names(.ndm_embedded_model_spec_sources) <- model_spec_files

for (name in model_spec_files) {
  runtime_path <- file.path(runtime_model_spec_root, name)
  if (!file.exists(runtime_path)) {
    stop("Missing runtime model specification: ", name, call. = FALSE)
  }
  if (!identical(
    .ndm_embedded_model_spec_sources[[name]],
    read_model_spec(runtime_model_spec_root, name)
  )) {
    stop(
      "Packaged and runtime model specifications differ: ",
      name,
      call. = FALSE
    )
  }
}

save(
  .ndm_embedded_model_spec_sources,
  file = output_path,
  version = 2L,
  compress = "xz"
)
