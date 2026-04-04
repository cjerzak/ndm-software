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

.ndm_execution_spec_compartments <- function(execution_spec) {
  stopifnot(is.list(execution_spec))

  family <- execution_spec$family %||% NULL
  if (!is.null(family) && grepl("^tb_", family)) {
    return("tb")
  }

  state_terms <- execution_spec$state_terms %||% character(0)
  if (length(state_terms) > 0L &&
      all(c("s_l", "e_l", "i_l") %in% state_terms)) {
    if ("r_l" %in% state_terms) {
      return("seirs")
    }
    return("seir")
  }

  "custom"
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
#' For a maintainer-oriented walkthrough of preset selection, structured
#' families, TeX round-tripping, and safe modification workflows, see
#' \code{vignette("ode-structures", package = "ndm")}.
#'
#' @returns `ndm_model_spec_presets()` returns a data frame describing the
#'   built-in presets bundled with the package.
#'
#' @examples
#' ndm_model_spec_presets()
#'
#' @export
ndm_model_spec_presets <- function() {
  builtin_registry <- .ndm_builtin_spec_registry()
  builtin_rows <- lapply(names(builtin_registry), function(name) {
    entry <- builtin_registry[[name]]
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

  structured_registry <- .ndm_structured_spec_registry()
  structured_rows <- lapply(names(structured_registry), function(name) {
    entry <- structured_registry[[name]]
    data.frame(
      preset = name,
      compartments = entry$compartments,
      dynamic_beta = FALSE,
      dynamic_global = FALSE,
      multi_outcome = FALSE,
      description = entry$description,
      stringsAsFactors = FALSE
    )
  })

  out <- do.call(rbind, c(builtin_rows, structured_rows))
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
#' @param family_args Optional named list used to expand parameterized model
#'   families such as chained latent-stage specifications.
#'
#' @returns `ndm_model_spec()` returns an object of class `ndm_model_spec`.
#'   Built-in presets include metadata such as the packaged source path and the
#'   normalized TeX contents.
#'
#' @examples
#' spec <- ndm_model_spec()
#' print(spec)
#'
#' tb_spec <- ndm_model_spec(
#'   preset = "tb_b",
#'   model_type = "NeuralODE",
#'   family_args = list(n = 4L)
#' )
#' tb_spec$family_args
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
                           tex_path = NULL,
                           family_args = NULL) {
  model_type <- match.arg(model_type)

  if (!is.null(tex_path)) {
    return(ndm_model_spec_from_tex(tex_path, model_type = model_type))
  }

  builtin_registry <- .ndm_builtin_spec_registry()
  structured_registry <- .ndm_structured_spec_registry()
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

  if (preset %in% names(structured_registry)) {
    return(.ndm_structured_spec_from_preset(
      preset = preset,
      model_type = model_type,
      family_args = family_args,
      time_varying_terms = time_varying_terms,
      endogenous_terms = endogenous_terms
    ))
  }

  if (!preset %in% names(builtin_registry)) {
    stop("Unknown preset '", preset, "'.", call. = FALSE)
  }

  entry <- .ndm_spec_with_path(builtin_registry[[preset]])
  tex_text <- paste(readLines(entry$source_path, warn = FALSE), collapse = "\n")
  execution_spec <- .ndm_parse_tex_execution_spec(tex_text)

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
      source_format = "tex",
      family = NULL,
      family_args = list(),
      time_varying_terms = unique(time_varying_terms),
      endogenous_terms = unique(endogenous_terms),
      state_terms = execution_spec$state_terms %||% character(0),
      init_state_terms = execution_spec$init_state_terms %||% character(0),
      parameter_terms = execution_spec$parameter_terms %||% character(0),
      auxiliary_terms = execution_spec$auxiliary_terms %||% character(0),
      local_dynamic_terms = execution_spec$local_dynamic_terms %||% character(0),
      global_dynamic_terms = execution_spec$global_dynamic_terms %||% character(0),
      source_path = entry$source_path,
      tex_text = tex_text,
      execution_spec = execution_spec
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
#' tb_spec <- ndm_model_spec(
#'   preset = "tb_k",
#'   model_type = "NeuralODE"
#' )
#' tb_tex_path <- tempfile(fileext = ".tex")
#' ndm_model_spec_to_tex(tb_spec, tb_tex_path)
#' ndm_model_spec_from_tex(tb_tex_path, model_type = "NeuralODE")
#'
#' @export
ndm_model_spec_from_tex <- function(path,
                                    model_type = c("DecoderOnly", "NeuralODE")) {
  model_type <- match.arg(model_type)
  source_path <- .ndm_normalize_path(path, must_work = TRUE)
  builtin_registry <- .ndm_builtin_spec_registry()
  preset_name <- names(builtin_registry)[vapply(
    builtin_registry,
    function(entry) identical(basename(source_path), entry$file),
    logical(1)
  )]

  tex_text <- paste(readLines(source_path, warn = FALSE), collapse = "\n")
  execution_spec <- .ndm_parse_tex_execution_spec(tex_text)

  if (length(preset_name) == 1L) {
    spec <- ndm_model_spec(preset = preset_name, model_type = model_type)
    spec$source_path <- source_path
    spec$tex_text <- tex_text
    spec$execution_spec <- execution_spec
    return(spec)
  }

  preset <- execution_spec$preset %||% "custom"
  description <- if (!is.null(execution_spec$family) && grepl("^tb_", execution_spec$family)) {
    registry_entry <- .ndm_structured_spec_registry()[[execution_spec$family]]
    registry_entry$description %||% "Structured model specification imported from disk."
  } else {
    "Custom TeX model specification imported from disk."
  }

  .ndm_make_classed_list(
    list(
      preset = preset,
      model_type = model_type,
      compartments = .ndm_execution_spec_compartments(execution_spec),
      dynamic_beta = NA,
      dynamic_global = NA,
      multi_outcome = NA,
      description = description,
      source_format = execution_spec$source_format %||% "tex",
      family = execution_spec$family %||% NULL,
      family_args = execution_spec$family_args %||% list(),
      time_varying_terms = execution_spec$time_varying_terms %||% character(0),
      endogenous_terms = execution_spec$endogenous_terms %||% character(0),
      state_terms = execution_spec$state_terms %||% character(0),
      init_state_terms = execution_spec$init_state_terms %||% character(0),
      parameter_terms = execution_spec$parameter_terms %||% character(0),
      auxiliary_terms = execution_spec$auxiliary_terms %||% character(0),
      local_dynamic_terms = execution_spec$local_dynamic_terms %||% character(0),
      global_dynamic_terms = execution_spec$global_dynamic_terms %||% character(0),
      source_path = source_path,
      tex_text = tex_text,
      execution_spec = execution_spec
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
