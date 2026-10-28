.ndm_default_initial_state_prior <- function(terms, infectious_center = NULL) {
  reference <- if ("s_l" %in% terms) "s_l" else terms[[1L]]
  other <- setdiff(terms, reference)
  center <- stats::setNames(rep(exp(-11) / (1 + length(other) * exp(-11)), length(terms)), terms)
  center[[reference]] <- 1 - sum(center[other])
  if (!is.null(infectious_center)) {
    if (!"i_l" %in% other || length(infectious_center) != 1L || !is.finite(infectious_center) || infectious_center <= 0) {
      stop("initial_incidence_center must induce a positive infectious composition center.", call. = FALSE)
    }
    center[["i_l"]] <- infectious_center
    center[[reference]] <- 1 - sum(center[other])
    if (center[[reference]] <= 0) stop("Initial composition centers exhaust the population.", call. = FALSE)
  }
  covariance <- diag(length(other)) + matrix(1, length(other), length(other))
  dimnames(covariance) <- list(other, other)
  list(type = "logistic_normal", reference_state = reference,
       logratio_mean = log(center[other] / center[[reference]]), logratio_cov = covariance)
}

.ndm_validate_initial_state_prior <- function(prior, terms) {
  if (!length(terms) || anyDuplicated(terms) || anyNA(terms) || any(!nzchar(terms))) {
    stop("init_state_terms must contain unique state names.", call. = FALSE)
  }
  if (!is.list(prior) || !identical(prior$type, "logistic_normal")) {
    stop("Declare initial_state_prior with type = 'logistic_normal'; old compartment Gaussian priors require migration.", call. = FALSE)
  }
  if (length(prior$reference_state) != 1L || !prior$reference_state %in% terms) stop("Invalid initial-state reference_state.", call. = FALSE)
  other <- setdiff(terms, prior$reference_state)
  mu <- prior$logratio_mean; covariance <- prior$logratio_cov
  if (!is.numeric(mu) || length(mu) != length(other) || any(!is.finite(mu)) ||
      (length(other) && (anyDuplicated(names(mu)) || !setequal(names(mu), other)))) {
    stop("logratio_mean must name exactly the non-reference states and contain finite values.", call. = FALSE)
  }
  if (!is.matrix(covariance) || !is.numeric(covariance) ||
      !identical(dim(covariance), rep(length(other), 2L)) || any(!is.finite(covariance))) {
    stop("logratio_cov must be a finite covariance matrix of dimension K - 1.", call. = FALSE)
  }
  if (length(other)) {
    if (!setequal(rownames(covariance), other) || !setequal(colnames(covariance), other) ||
        anyDuplicated(rownames(covariance)) || anyDuplicated(colnames(covariance))) {
      stop("logratio_cov must have the non-reference state names on both axes.", call. = FALSE)
    }
    covariance <- covariance[other, other, drop = FALSE]
    if (!isTRUE(all.equal(covariance, t(covariance), tolerance = 1e-12)) ||
        inherits(try(chol(covariance), silent = TRUE), "try-error")) {
      stop("logratio_cov must be symmetric positive definite.", call. = FALSE)
    }
  }
  list(type = "logistic_normal", reference_state = prior$reference_state,
       logratio_mean = mu[other], logratio_cov = covariance)
}

.ndm_initial_state_metadata <- function(spec) {
  prior <- .ndm_validate_initial_state_prior(spec$initial_state_prior, spec$init_state_terms)
  numeric_text <- function(x) paste(format(as.numeric(x), digits = 17L, scientific = TRUE, trim = TRUE), collapse = ",")
  c("% model_semantics_version: 2",
    "% initial_state_prior_type: logistic_normal",
    paste0("% initial_state_reference: ", prior$reference_state),
    paste0("% initial_state_mean: ", numeric_text(prior$logratio_mean)),
    paste0("% initial_state_cov: ", numeric_text(t(prior$logratio_cov))),
    paste0("% initial_state_fixed_names: ", paste(names(spec$initial_state_values), collapse = ",")),
    paste0("% initial_state_fixed_values: ", numeric_text(spec$initial_state_values)))
}

.ndm_initial_state_tex <- function(spec) {
  prior <- .ndm_validate_initial_state_prior(spec$initial_state_prior, spec$init_state_terms)
  number <- function(x) format(x, digits = 8L, scientific = FALSE, trim = TRUE)
  mu <- paste(number(prior$logratio_mean), collapse = ", ")
  covariance <- if (length(prior$logratio_mean)) paste(apply(prior$logratio_cov, 1L,
    function(row) paste(number(row), collapse = " & ")), collapse = " \\\\ ") else ""
  c("% NDM_COMPOSITION_BEGIN",
    "\\paragraph{Initial population composition.}",
    paste0("Reference state: \\texttt{\\detokenize{", prior$reference_state, "}}. ",
           "Non-reference coordinates, in order: \\texttt{\\detokenize{", paste(names(prior$logratio_mean), collapse = ", "), "}}."),
    if (length(prior$logratio_mean)) c(
      "The joint prior is $z \\sim \\mathcal{N}(\\mu,\\Sigma)$; the reference logit is zero.",
      paste0("\\[ \\mu = (", mu, ")^\\top, \\qquad \\Sigma = \\begin{pmatrix}", covariance, "\\end{pmatrix}. \\]"),
      "Initial counts are $N\\,\\mathrm{softmax}(z,0)$, reordered to the declared state order.") else
        "The sole population state starts deterministically at $N$.",
    "These are log-ratio parameters; transformed centers are not arithmetic means of counts.",
    "% NDM_COMPOSITION_END")
}

.ndm_read_initial_state_metadata <- function(values) {
  if (is.null(values$initial_state_prior_type)) return(list(initial_state_prior = NULL, initial_state_values = numeric()))
  terms <- .ndm_parse_csv_field(values$init_state_terms)
  other <- setdiff(terms, values$initial_state_reference)
  numbers <- function(x) as.numeric(.ndm_parse_csv_field(x))
  covariance_values <- numbers(values$initial_state_cov)
  if (length(covariance_values) != length(other)^2) stop("Invalid initial-state covariance metadata.", call. = FALSE)
  prior <- list(type = values$initial_state_prior_type, reference_state = values$initial_state_reference,
                logratio_mean = stats::setNames(numbers(values$initial_state_mean), other),
                logratio_cov = matrix(covariance_values, length(other), length(other), byrow = TRUE, dimnames = list(other, other)))
  fixed_names <- .ndm_parse_csv_field(values$initial_state_fixed_names)
  fixed_values <- numbers(values$initial_state_fixed_values)
  if (length(fixed_names) != length(fixed_values)) stop("Invalid fixed initial-state metadata.", call. = FALSE)
  list(initial_state_prior = .ndm_validate_initial_state_prior(prior, terms),
       initial_state_values = stats::setNames(fixed_values, fixed_names), model_semantics_version = 2L)
}

.ndm_initial_state_draw <- function(z, prior, terms, population, jax, jnp) {
  other <- setdiff(terms, prior$reference_state)
  logits <- lapply(terms, function(name) {
    if (name == prior$reference_state) jnp$array(0., dtype = population$dtype) else
      jnp$take(z, as.integer(match(name, other) - 1L))$astype(population$dtype)
  })
  jax$nn$softmax(jnp$stack(logits)) * population
}

.ndm_override_initial_state_prior <- function(spec, prior) {
  if (is.null(prior)) return(spec)
  if ("initial_incidence_center" %in% names(spec$family_args)) stop("Supply either initial_incidence_center or initial_state_prior.", call. = FALSE)
  spec$execution_spec$initial_state_prior <- .ndm_validate_initial_state_prior(prior, spec$init_state_terms)
  lines <- strsplit(spec$tex_text, "\n", fixed = TRUE)[[1L]]
  visible_start <- grep("^% NDM_COMPOSITION_BEGIN$", lines)
  visible_end <- grep("^% NDM_COMPOSITION_END$", lines)
  if (length(visible_start) && length(visible_end)) lines <- lines[-seq.int(visible_start[[1L]], visible_end[[1L]])]
  first <- grep("^%%%NDM_SPEC_META_BEGIN", lines); last <- grep("^%%%NDM_SPEC_META_END", lines)
  metadata <- .ndm_emit_execution_spec_metadata(spec$execution_spec)
  if (length(first) && length(last)) {
    lines <- c(utils::head(lines, first[[1L]] - 1L), metadata, tail(lines, length(lines) - last[[1L]]))
  } else lines <- c(metadata, lines)
  start <- grep("^%%%START BAYES ODE", lines)[[1L]]
  lines <- append(lines, .ndm_initial_state_tex(spec$execution_spec), after = start - 1L)
  spec$tex_text <- paste(lines, collapse = "\n")
  spec
}
