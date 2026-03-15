.ndm_builtin_spec_registry <- function() {
  list(
    seir_fixed = list(
      file = "bayes_ode_SEIRS_FixedBeta_FixedGlobal.tex",
      compartments = "seir",
      dynamic_beta = FALSE,
      dynamic_global = FALSE,
      multi_outcome = FALSE,
      description = "Classical fixed-beta, fixed-global SEIR baseline."
    ),
    seirs_dynamic_beta = list(
      file = "bayes_ode_SEIRS_DynamicBeta_FixedGlobal.tex",
      compartments = "seirs",
      dynamic_beta = TRUE,
      dynamic_global = FALSE,
      multi_outcome = FALSE,
      description = "SEIRS with dynamic local transmission and fixed global rates."
    ),
    seirs_dynamic_beta_multi_outcome = list(
      file = "bayes_ode_SEIRS_DynamicBeta_FixedGlobal_MultiOutcome.tex",
      compartments = "seirs",
      dynamic_beta = TRUE,
      dynamic_global = FALSE,
      multi_outcome = TRUE,
      description = "SEIRS with dynamic local transmission and multi-outcome observation model."
    ),
    seirs_dynamic_beta_dynamic_global = list(
      file = "bayes_ode_SEIRS_DynamicBeta_DynamicGlobal.tex",
      compartments = "seirs",
      dynamic_beta = TRUE,
      dynamic_global = TRUE,
      multi_outcome = FALSE,
      description = "SEIRS with dynamic local transmission and dynamic global rates."
    ),
    seirs_dynamic_beta_dynamic_global_multi_outcome = list(
      file = "bayes_ode_SEIRS_DynamicBeta_DynamicGlobal_MultiOutcome.tex",
      compartments = "seirs",
      dynamic_beta = TRUE,
      dynamic_global = TRUE,
      multi_outcome = TRUE,
      description = "SEIRS with dynamic local transmission, dynamic global rates, and multi-outcome observation model."
    )
  )
}

.ndm_spec_with_path <- function(entry) {
  entry$source_path <- .ndm_embedded_model_spec_path(entry$file)
  entry
}

.ndm_resolve_builtin_preset <- function(compartments,
                                        dynamic_beta,
                                        dynamic_global,
                                        multi_outcome) {
  if (identical(compartments, "seir") &&
      identical(dynamic_beta, FALSE) &&
      identical(dynamic_global, FALSE) &&
      identical(multi_outcome, FALSE)) {
    return("seir_fixed")
  }

  if (identical(compartments, "seirs") &&
      identical(dynamic_beta, TRUE) &&
      identical(dynamic_global, FALSE) &&
      identical(multi_outcome, FALSE)) {
    return("seirs_dynamic_beta")
  }

  if (identical(compartments, "seirs") &&
      identical(dynamic_beta, TRUE) &&
      identical(dynamic_global, FALSE) &&
      identical(multi_outcome, TRUE)) {
    return("seirs_dynamic_beta_multi_outcome")
  }

  if (identical(compartments, "seirs") &&
      identical(dynamic_beta, TRUE) &&
      identical(dynamic_global, TRUE) &&
      identical(multi_outcome, FALSE)) {
    return("seirs_dynamic_beta_dynamic_global")
  }

  if (identical(compartments, "seirs") &&
      identical(dynamic_beta, TRUE) &&
      identical(dynamic_global, TRUE) &&
      identical(multi_outcome, TRUE)) {
    return("seirs_dynamic_beta_dynamic_global_multi_outcome")
  }

  stop(
    "Unsupported preset combination. Use one of ndm_model_spec_presets() or import a custom .tex file with ndm_model_spec_from_tex().",
    call. = FALSE
  )
}

#' Inspect and create epidemic model specifications
#'
#' The package ships a small registry of built-in compartmental model
#' specifications and also supports importing custom `.tex` model definitions
#' from disk. The structured `ndm_model_spec` object is the canonical
#' representation used by the runtime wrappers.
#'
#' @returns `ndm_model_spec_presets()` returns a data frame describing the
#'   built-in presets bundled with the package.
#'
#' @examples
#' ndm_model_spec_presets()
#'
#' @export
ndm_model_spec_presets <- function() {
  registry <- .ndm_builtin_spec_registry()
  rows <- lapply(names(registry), function(name) {
    entry <- registry[[name]]
    data.frame(
      preset = name,
      compartments = entry$compartments,
      dynamic_beta = entry$dynamic_beta,
      dynamic_global = entry$dynamic_global,
      multi_outcome = entry$multi_outcome,
      description = entry$description,
      stringsAsFactors = FALSE
    )
  })

  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

#' @rdname ndm_model_spec_presets
#'
#' @param preset Optional preset name from `ndm_model_spec_presets()`. When
#'   omitted, the remaining arguments are used to infer a built-in preset.
#' @param compartments Compartment family used when inferring a built-in preset.
#' @param dynamic_beta Logical scalar controlling whether the local transmission
#'   rate is time-varying for built-in presets.
#' @param dynamic_global Logical scalar controlling whether the global rates are
#'   time-varying for built-in presets.
#' @param multi_outcome Logical scalar indicating whether the observation model
#'   should include multiple outcomes.
#' @param model_type Model family metadata attached to the returned
#'   specification. Either `"DecoderOnly"` or `"NeuralODE"`.
#' @param time_varying_terms Optional character vector overriding the default
#'   time-varying term metadata stored on the returned spec.
#' @param endogenous_terms Optional character vector overriding the default
#'   endogenous term metadata stored on the returned spec.
#' @param tex_path Optional path to a custom `.tex` file. When supplied,
#'   `ndm_model_spec()` delegates to `ndm_model_spec_from_tex()`.
#'
#' @returns `ndm_model_spec()` returns an object of class `ndm_model_spec`.
#'   Built-in presets include metadata such as the packaged source path and the
#'   normalized TeX contents.
#'
#' @examples
#' spec <- ndm_model_spec()
#' print(spec)
#'
#' @export
ndm_model_spec <- function(preset = NULL,
                           compartments = c("seir", "seirs"),
                           dynamic_beta = NULL,
                           dynamic_global = FALSE,
                           multi_outcome = FALSE,
                           model_type = c("DecoderOnly", "NeuralODE"),
                           time_varying_terms = NULL,
                           endogenous_terms = NULL,
                           tex_path = NULL) {
  model_type <- match.arg(model_type)

  if (!is.null(tex_path)) {
    return(ndm_model_spec_from_tex(tex_path, model_type = model_type))
  }

  registry <- .ndm_builtin_spec_registry()
  if (is.null(preset)) {
    compartments <- match.arg(compartments)
    if (is.null(dynamic_beta)) {
      dynamic_beta <- identical(compartments, "seirs")
    }
    preset <- .ndm_resolve_builtin_preset(
      compartments = compartments,
      dynamic_beta = isTRUE(dynamic_beta),
      dynamic_global = isTRUE(dynamic_global),
      multi_outcome = isTRUE(multi_outcome)
    )
  }

  if (!preset %in% names(registry)) {
    stop("Unknown preset '", preset, "'.", call. = FALSE)
  }

  entry <- .ndm_spec_with_path(registry[[preset]])
  tex_text <- paste(readLines(entry$source_path, warn = FALSE), collapse = "\n")

  if (is.null(time_varying_terms)) {
    time_varying_terms <- switch(
      preset,
      seir_fixed = character(0),
      seirs_dynamic_beta = c("beta_l", "p_l"),
      seirs_dynamic_beta_multi_outcome = c("beta_l", "p_l"),
      seirs_dynamic_beta_dynamic_global = c("beta_l", "p_l", "sigma", "gamma", "xi", "delta"),
      seirs_dynamic_beta_dynamic_global_multi_outcome = c("beta_l", "p_l", "sigma", "gamma", "xi", "delta"),
      character(0)
    )
  }

  if (is.null(endogenous_terms)) {
    endogenous_terms <- switch(
      preset,
      seir_fixed = character(0),
      seirs_dynamic_beta = c("beta_l", "p_l"),
      seirs_dynamic_beta_multi_outcome = c("beta_l", "p_l"),
      seirs_dynamic_beta_dynamic_global = c("beta_l", "p_l", "sigma", "gamma", "xi", "delta"),
      seirs_dynamic_beta_dynamic_global_multi_outcome = c("beta_l", "p_l", "sigma", "gamma", "xi", "delta"),
      character(0)
    )
  }

  .ndm_make_classed_list(
    list(
      preset = preset,
      model_type = model_type,
      compartments = entry$compartments,
      dynamic_beta = entry$dynamic_beta,
      dynamic_global = entry$dynamic_global,
      multi_outcome = entry$multi_outcome,
      description = entry$description,
      time_varying_terms = unique(time_varying_terms),
      endogenous_terms = unique(endogenous_terms),
      source_path = entry$source_path,
      tex_text = tex_text
    ),
    "ndm_model_spec"
  )
}

#' @rdname ndm_model_spec_presets
#'
#' @param path Path to a `.tex` model specification file on disk.
#'
#' @returns `ndm_model_spec_from_tex()` returns an `ndm_model_spec` object whose
#'   `source_path` points at the supplied file. When the basename matches a
#'   built-in preset, the preset metadata is reused.
#'
#' @examples
#' spec <- ndm_model_spec()
#' tex_path <- tempfile(fileext = ".tex")
#' ndm_model_spec_to_tex(spec, tex_path)
#' ndm_model_spec_from_tex(tex_path)
#'
#' @export
ndm_model_spec_from_tex <- function(path,
                                    model_type = c("DecoderOnly", "NeuralODE")) {
  model_type <- match.arg(model_type)
  source_path <- .ndm_normalize_path(path, must_work = TRUE)
  registry <- .ndm_builtin_spec_registry()
  preset_name <- names(registry)[vapply(
    registry,
    function(entry) identical(basename(source_path), entry$file),
    logical(1)
  )]

  tex_text <- paste(readLines(source_path, warn = FALSE), collapse = "\n")

  if (length(preset_name) == 1L) {
    spec <- ndm_model_spec(preset = preset_name, model_type = model_type)
    spec$source_path <- source_path
    spec$tex_text <- tex_text
    return(spec)
  }

  .ndm_make_classed_list(
    list(
      preset = "custom",
      model_type = model_type,
      compartments = NA_character_,
      dynamic_beta = NA,
      dynamic_global = NA,
      multi_outcome = NA,
      description = "Custom TeX model specification imported from disk.",
      time_varying_terms = character(0),
      endogenous_terms = character(0),
      source_path = source_path,
      tex_text = tex_text
    ),
    "ndm_model_spec"
  )
}

#' @rdname ndm_model_spec_presets
#'
#' @param spec An `ndm_model_spec` object.
#'
#' @returns `ndm_model_spec_to_tex()` returns the TeX text as a single character
#'   string when `path` is `NULL`. When `path` is supplied, it writes the
#'   specification to disk and invisibly returns the normalized output path.
#'
#' @examples
#' spec <- ndm_model_spec()
#' tex <- ndm_model_spec_to_tex(spec)
#' nchar(tex) > 0
#'
#' @export
ndm_model_spec_to_tex <- function(spec, path = NULL) {
  if (!inherits(spec, "ndm_model_spec")) {
    stop("`spec` must inherit from class 'ndm_model_spec'.", call. = FALSE)
  }

  if (is.null(path)) {
    return(spec$tex_text)
  }

  writeLines(spec$tex_text, con = path, useBytes = TRUE)
  invisible(.ndm_normalize_path(path, must_work = TRUE))
}

.ndm_materialize_model_spec <- function(spec) {
  if (!inherits(spec, "ndm_model_spec")) {
    stop("`spec` must inherit from class 'ndm_model_spec'.", call. = FALSE)
  }

  if (!is.null(spec$source_path) && file.exists(spec$source_path)) {
    return(list(path = spec$source_path, cleanup = FALSE))
  }

  tmp <- tempfile(pattern = "ndm_model_spec_", fileext = ".tex")
  ndm_model_spec_to_tex(spec, path = tmp)
  list(path = tmp, cleanup = TRUE)
}
