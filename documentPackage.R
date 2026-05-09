{
  options(error = NULL)

  resolve_script_path <- function() {
    source_frame <- tryCatch(sys.frame(1L)$ofile, error = function(...) NULL)
    if (!is.null(source_frame)) {
      return(normalizePath(source_frame, winslash = "/", mustWork = TRUE))
    }

    args <- commandArgs(trailingOnly = FALSE)
    script_arg <- grep("^--file=", args, value = TRUE)
    if (length(script_arg) > 0L) {
      return(normalizePath(sub("^--file=", "", script_arg[[1L]]), winslash = "/", mustWork = TRUE))
    }

    fallback <- file.path(getwd(), "documentPackage.R")
    if (file.exists(fallback)) {
      return(normalizePath(fallback, winslash = "/", mustWork = TRUE))
    }

    stop("Unable to determine the path to documentPackage.R.")
  }

  run_system_step <- function(args) {
    output <- system2(r_bin, args, stdout = TRUE, stderr = TRUE)
    status <- attr(output, "status")

    if (length(output) > 0L) {
      cat(paste(output, collapse = "\n"), "\n")
    }

    if (!is.null(status) && status != 0L) {
      stop(sprintf("Command failed: %s", paste(c(r_bin, args), collapse = " ")))
    }

    invisible(output)
  }

  run_rscript_step <- function(args) {
    output <- system2(rscript_bin, args, stdout = TRUE, stderr = TRUE)
    status <- attr(output, "status")

    if (length(output) > 0L) {
      cat(paste(output, collapse = "\n"), "\n")
    }

    if (!is.null(status) && status != 0L) {
      stop(sprintf("Command failed: %s", paste(c(rscript_bin, args), collapse = " ")))
    }

    invisible(output)
  }

  cleanup_rdpdf_dirs <- function(path) {
    rd2pdf_dirs <- list.files(path, pattern = "^\\.Rd2pdf", all.files = TRUE, full.names = TRUE)
    rd2pdf_dirs <- rd2pdf_dirs[dir.exists(rd2pdf_dirs)]

    if (length(rd2pdf_dirs) > 0L) {
      unlink(rd2pdf_dirs, recursive = TRUE, force = TRUE)
    }
  }

  cleanup_build_artifacts <- function() {
    if (file.exists(tarball_path)) {
      file.remove(tarball_path)
    }

    if (dir.exists(check_path)) {
      unlink(check_path, recursive = TRUE, force = TRUE)
    }
  }

  script_path <- resolve_script_path()
  package_path <- dirname(script_path)
  setwd(package_path)

  desc <- read.dcf(file.path(package_path, "DESCRIPTION"))[1L, ]
  package_name <- unname(desc[["Package"]])
  version_number <- unname(desc[["Version"]])
  pdf_path <- file.path(package_path, sprintf("%s.pdf", package_name))
  tarball_path <- file.path(package_path, sprintf("%s_%s.tar.gz", package_name, version_number))
  check_path <- file.path(package_path, sprintf("%s.Rcheck", package_name))
  r_bin <- file.path(R.home("bin"), "R")
  rscript_bin <- file.path(R.home("bin"), "Rscript")
  runtime_registry_generator <- file.path(package_path, "tools", "generate_runtime_registry.R")
  runtime_sources_generator <- file.path(package_path, "tools", "generate_runtime_sources.R")

  refresh_generated_runtime_artifact <- function(path, label) {
    if (!file.exists(path)) {
      stop(sprintf("%s generator not found: %s", label, path))
    }
    run_rscript_step(path)
  }

  refresh_runtime_artifacts <- function() {
    refresh_generated_runtime_artifact(runtime_registry_generator, "Runtime registry")
    refresh_generated_runtime_artifact(runtime_sources_generator, "Runtime sources")
  }

  refresh_runtime_artifacts()
  install.packages(package_path, repos = NULL, type = "source", force = FALSE)
  #ndm::ndm_build_backend()

  if (dir.exists(file.path(package_path, "data"))) {
    tools::add_datalist(package_path, force = TRUE, small.size = 1L)
  }

  if (dir.exists(file.path(package_path, "vignettes"))) {
    devtools::build_vignettes(package_path, install = FALSE)
  }

  refresh_runtime_artifacts()
  devtools::document(package_path)

  if (file.exists(pdf_path)) {
    file.remove(pdf_path)
  }

  cleanup_rdpdf_dirs(package_path)
  tryCatch(
    run_system_step(c("CMD", "Rd2pdf", sprintf("--output=%s", basename(pdf_path)), package_path)),
    finally = cleanup_rdpdf_dirs(package_path)
  )

  refresh_runtime_artifacts()
  devtools::check(package_path)

  refresh_runtime_artifacts()
  cleanup_build_artifacts()
  run_system_step(c("CMD", "build", "--resave-data", package_path))

  cleanup_build_artifacts()
  run_system_step(c("CMD", "check", "--as-cran", tarball_path))

  refresh_runtime_artifacts()
  install.packages(package_path, repos = NULL, type = "source", force = FALSE)
  # install.packages("~/Documents/ndm-software", repos = NULL, type = "source", force = FALSE)
}
