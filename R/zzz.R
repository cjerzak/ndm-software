ndm_env <- new.env(parent = emptyenv())
ndm_env$backend <- NULL

`%||%` <- function(x, y) {
  if (is.null(x)) {
    return(y)
  }
  x
}

#' Print a timestamped diagnostic message
#'
#' `ndm_print()` is a small logging helper used throughout the package-facing
#' orchestration code.
#'
#' @param text Text to print.
#' @param quiet Logical scalar indicating whether output should be suppressed.
#'
#' @returns The input `text`, invisibly.
#'
#' @examples
#' ndm_print("starting")
#'
#' @export
ndm_print <- function(text, quiet = FALSE) {
  if (!quiet) {
    message(sprintf("[%s] %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), text))
  }
  invisible(text)
}

.ndm_default_analysis_root <- function() {
  option_root <- getOption("ndm.analysis_root")
  env_root <- Sys.getenv("NDM_ANALYSIS_ROOT", unset = "")
  if (nzchar(env_root)) {
    return(env_root)
  }
  option_root
}

.ndm_normalize_path <- function(path, must_work = TRUE) {
  normalizePath(path, winslash = "/", mustWork = must_work)
}

.ndm_package_root <- function() {
  pkg_root <- system.file(package = "ndm")
  if (nzchar(pkg_root)) {
    return(pkg_root)
  }

  candidates <- unique(c(
    getwd(),
    normalizePath(file.path(getwd(), ".."), winslash = "/", mustWork = FALSE),
    normalizePath(file.path(getwd(), "..", ".."), winslash = "/", mustWork = FALSE)
  ))

  for (candidate in candidates) {
    if (!nzchar(candidate)) {
      next
    }
    if (file.exists(file.path(candidate, "DESCRIPTION")) &&
        dir.exists(file.path(candidate, "R"))) {
      return(.ndm_normalize_path(candidate, must_work = TRUE))
    }
  }

  stop("Could not determine the ndm package root.", call. = FALSE)
}

.ndm_inst_path <- function(...) {
  pkg_root <- system.file(package = "ndm")
  if (nzchar(pkg_root)) {
    return(file.path(pkg_root, ...))
  }
  file.path(.ndm_package_root(), "inst", ...)
}

.ndm_resolve_analysis_root <- function(analysis_root = .ndm_default_analysis_root(),
                                       must_work = TRUE) {
  if (is.null(analysis_root) || !nzchar(analysis_root)) {
    stop(
      "`analysis_root` must point to a local Analysis or Analysis2 directory. ",
      "Set it explicitly or via `options(ndm.analysis_root=...)` / `NDM_ANALYSIS_ROOT`.",
      call. = FALSE
    )
  }

  .ndm_normalize_path(analysis_root, must_work = must_work)
}

.ndm_make_classed_list <- function(x, class_name) {
  stopifnot(is.list(x))
  structure(x, class = c(class_name, "list"))
}

#' Create runtime configuration objects
#'
#' Configuration objects collect the runtime options that are threaded through
#' the higher-level orchestration helpers such as `ndm_prepare_runtime()` and
#' `ndm_fit()`.
#'
#' @param model_type Model family metadata. Use `"DecoderOnly"` or
#'   `"NeuralODE"`.
#' @param backbone Backbone family. Phase 1 supports `"transformer"` only.
#' @param analysis_root Optional analysis root recorded on the configuration
#'   object for the local runtime interface.
#' @param float_type Floating point precision used when initializing the backend.
#'   Use `"32"` or `"64"`.
#' @param force_to_gpu Logical scalar indicating whether the runtime should try
#'   to place arrays on GPU when available.
#' @param resave_tfrecords Logical scalar preserved for compatibility with the
#'   legacy runtime.
#' @param gpu_mem_frac Optional GPU memory fraction forwarded into the runtime
#'   bootstrap code.
#' @param ... Additional named values appended to the configuration object.
#'
#' @returns `ndm_create_config()` returns an object of class `ndm_config`.
#'
#' @examples
#' cfg <- ndm_create_config()
#' print(cfg)
#'
#' @export
ndm_create_config <- function(model_type = c("DecoderOnly", "NeuralODE"),
                              backbone = "transformer",
                              analysis_root = .ndm_default_analysis_root(),
                              float_type = c("32", "64"),
                              force_to_gpu = TRUE,
                              resave_tfrecords = FALSE,
                              gpu_mem_frac = NULL,
                              ...) {
  model_type <- match.arg(model_type)
  float_type <- match.arg(float_type)
  if (!identical(backbone, "transformer")) {
    stop("Phase 1 only supports backbone = 'transformer'.", call. = FALSE)
  }

  if (!is.null(analysis_root) && nzchar(analysis_root)) {
    analysis_root <- .ndm_resolve_analysis_root(analysis_root, must_work = FALSE)
  }

  extras <- list(...)
  config <- list(
    model_type = model_type,
    backbone = backbone,
    analysis_root = analysis_root,
    float_type = float_type,
    force_to_gpu = isTRUE(force_to_gpu),
    resave_tfrecords = isTRUE(resave_tfrecords),
    gpu_mem_frac = gpu_mem_frac
  )

  config <- c(config, extras)
  .ndm_make_classed_list(config, "ndm_config")
}

#' @rdname ndm_create_config
#'
#' @returns `print.ndm_config()` prints a brief summary of an `ndm_config`
#'   object and invisibly returns `x`.
#'
#' @param x An `ndm_config` object.
#' @param ... Unused; included for S3 method compatibility.
#'
#' @export
print.ndm_config <- function(x, ...) {
  cat("<ndm_config>\n")
  cat(sprintf("  model_type: %s\n", x$model_type %||% NA_character_))
  cat(sprintf("  backbone: %s\n", x$backbone %||% NA_character_))
  cat(sprintf("  float_type: %s\n", x$float_type %||% NA_character_))
  if (!is.null(x$analysis_root)) {
    cat(sprintf("  analysis_root: %s\n", x$analysis_root))
  }
  invisible(x)
}

#' @rdname ndm_model_spec_presets
#'
#' @returns `print.ndm_model_spec()` prints a brief summary of an
#'   `ndm_model_spec` object and invisibly returns `x`.
#'
#' @param x An `ndm_model_spec` object.
#' @param ... Unused; included for S3 method compatibility.
#'
#' @export
print.ndm_model_spec <- function(x, ...) {
  cat("<ndm_model_spec>\n")
  cat(sprintf("  preset: %s\n", x$preset %||% "custom"))
  cat(sprintf("  model_type: %s\n", x$model_type %||% NA_character_))
  cat(sprintf("  compartments: %s\n", x$compartments %||% NA_character_))
  cat(sprintf("  dynamic_beta: %s\n", x$dynamic_beta %||% NA))
  cat(sprintf("  dynamic_global: %s\n", x$dynamic_global %||% NA))
  cat(sprintf("  multi_outcome: %s\n", x$multi_outcome %||% NA))
  if (!is.null(x$source_path)) {
    cat(sprintf("  source_path: %s\n", x$source_path))
  }
  invisible(x)
}

.onUnload <- function(libpath) {
  ndm_env$backend <- NULL
  invisible(libpath)
}
