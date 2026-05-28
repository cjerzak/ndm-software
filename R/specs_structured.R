.ndm_param_spec <- function(transformation = c("Identity", "InvSoftPlus", "InvSigmoid"),
                            prior_mean,
                            prior_sd) {
  transformation <- match.arg(transformation)
  list(
    transformation = transformation,
    prior_mean = prior_mean,
    prior_sd = prior_sd
  )
}

.ndm_state_init_prior_spec <- function(prior_mean = 0, prior_sd = 0.1) {
  list(
    prior_mean = prior_mean,
    prior_sd = prior_sd
  )
}

.ndm_coerce_family_args <- function(args) {
  if (is.null(args)) {
    return(list())
  }
  if (!is.list(args)) {
    stop("`family_args` must be NULL or a named list.", call. = FALSE)
  }
  args
}

.ndm_as_named_numeric_list <- function(x, arg) {
  if (length(x) == 0L) {
    return(numeric())
  }
  if (is.null(names(x)) || any(!nzchar(names(x)))) {
    stop("`", arg, "` must be a named numeric vector.", call. = FALSE)
  }
  x_names <- names(x)
  x <- suppressWarnings(as.numeric(x))
  names(x) <- x_names
  if (anyNA(x)) {
    stop("`", arg, "` must contain only finite numeric values.", call. = FALSE)
  }
  x
}

.ndm_compact_text <- function(x) {
  x <- gsub("[[:space:]]+", " ", x)
  gsub("^\\s+|\\s+$", "", x)
}

.ndm_csv_field <- function(x) {
  x <- as.character(x %||% character(0))
  x <- x[nzchar(x)]
  if (length(x) == 0L) {
    return("")
  }
  paste(x, collapse = ",")
}

.ndm_parse_csv_field <- function(x) {
  x <- .ndm_compact_text(x %||% "")
  if (!nzchar(x)) {
    return(character(0))
  }
  vals <- trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
  vals[nzchar(vals)]
}

.ndm_parse_family_args_field <- function(x) {
  x <- .ndm_compact_text(x %||% "")
  if (!nzchar(x)) {
    return(list())
  }
  parts <- trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
  out <- vector("list", length(parts))
  names(out) <- rep("", length(parts))
  for (i in seq_along(parts)) {
    kv <- trimws(strsplit(parts[[i]], "=", fixed = TRUE)[[1L]])
    if (length(kv) != 2L || !nzchar(kv[[1L]])) {
      next
    }
    value <- utils::type.convert(kv[[2L]], as.is = TRUE)
    out[[i]] <- value
    names(out)[[i]] <- kv[[1L]]
  }
  out <- out[nzchar(names(out))]
  out
}

.ndm_format_family_args_field <- function(x) {
  if (!is.list(x) || length(x) == 0L) {
    return("")
  }
  paste(
    vapply(
      names(x),
      function(name) sprintf("%s=%s", name, as.character(x[[name]])),
      character(1L)
    ),
    collapse = ","
  )
}

.ndm_state_init_symbol <- function(state_term) {
  if (grepl("_l$", state_term)) {
    return(sub("_l$", "_{l0}", state_term))
  }
  paste0(state_term, "_0")
}

.ndm_transformed_prior_mean <- function(param) {
  if (identical(param$transformation, "Identity")) {
    return(as.character(param$prior_mean))
  }
  sprintf("gst(%s)", as.character(param$prior_mean))
}

.ndm_tex_parameter_symbol <- function(name) {
  name
}

.ndm_tex_prior_lhs <- function(name, transformation) {
  symbol <- .ndm_tex_parameter_symbol(name)
  if (identical(transformation, "Identity")) {
    return(symbol)
  }
  sprintf("\\%s(%s)", transformation, symbol)
}

.ndm_structured_template_text <- function() {
  path <- .ndm_embedded_model_spec_path("bayes_ode_SEIRS_DynamicBeta_DynamicGlobal.tex")
  paste(readLines(path, warn = FALSE), collapse = "\n")
}

.ndm_extract_bayes_ode_template <- function() {
  lines <- strsplit(.ndm_structured_template_text(), "\n", fixed = TRUE)[[1L]]
  start_idx <- grep("^%%%START BAYES ODE%%%\\s*$", lines)
  end_idx <- grep("^%%%END BAYES ODE%%%\\s*$", lines)
  if (length(start_idx) == 0L || length(end_idx) == 0L) {
    stop("Could not locate the embedded Bayes ODE template markers.", call. = FALSE)
  }
  list(
    prefix = lines[seq_len(start_idx[[1L]] - 1L)],
    suffix = lines[(end_idx[[1L]] + 1L):length(lines)]
  )
}

.ndm_emit_execution_spec_metadata <- function(spec) {
  meta_lines <- c(
    "%%%NDM_SPEC_META_BEGIN%%%",
    sprintf("%% source_format: %s", spec$source_format %||% "structured"),
    sprintf("%% preset: %s", spec$preset %||% ""),
    sprintf("%% family: %s", spec$family %||% ""),
    sprintf("%% family_args: %s", .ndm_format_family_args_field(spec$family_args)),
    sprintf("%% state_terms: %s", .ndm_csv_field(spec$state_terms)),
    sprintf("%% init_state_terms: %s", .ndm_csv_field(spec$init_state_terms)),
    sprintf("%% parameter_terms: %s", .ndm_csv_field(spec$parameter_terms)),
    sprintf("%% auxiliary_terms: %s", .ndm_csv_field(names(spec$auxiliary))),
    sprintf("%% local_dynamic_terms: %s", .ndm_csv_field(spec$local_dynamic_terms)),
    sprintf("%% global_dynamic_terms: %s", .ndm_csv_field(spec$global_dynamic_terms)),
    sprintf("%% time_varying_terms: %s", .ndm_csv_field(spec$time_varying_terms)),
    sprintf("%% endogenous_terms: %s", .ndm_csv_field(spec$endogenous_terms)),
    sprintf("%% local_inputs: %s", .ndm_csv_field(spec$neural$local_inputs %||% character(0))),
    "%%%NDM_SPEC_META_END%%%"
  )
  meta_lines
}

.ndm_parse_execution_spec_metadata <- function(tex_text) {
  lines <- strsplit(tex_text, "\n", fixed = TRUE)[[1L]]
  begin_idx <- grep("^%%%NDM_SPEC_META_BEGIN%%%\\s*$", lines)
  end_idx <- grep("^%%%NDM_SPEC_META_END%%%\\s*$", lines)
  if (length(begin_idx) == 0L || length(end_idx) == 0L) {
    return(NULL)
  }
  meta_lines <- lines[(begin_idx[[1L]] + 1L):(end_idx[[1L]] - 1L)]
  values <- list()
  for (line in meta_lines) {
    cleaned <- trimws(sub("^%+", "", line))
    parts <- strsplit(cleaned, ":", fixed = TRUE)[[1L]]
    if (length(parts) < 2L) {
      next
    }
    key <- trimws(parts[[1L]])
    value <- trimws(paste(parts[-1L], collapse = ":"))
    values[[key]] <- value
  }
  if (length(values) == 0L) {
    return(NULL)
  }
  list(
    source_format = values$source_format %||% "structured",
    preset = values$preset %||% "custom",
    family = values$family %||% NULL,
    family_args = .ndm_parse_family_args_field(values$family_args %||% ""),
    state_terms = .ndm_parse_csv_field(values$state_terms %||% ""),
    init_state_terms = .ndm_parse_csv_field(values$init_state_terms %||% ""),
    parameter_terms = .ndm_parse_csv_field(values$parameter_terms %||% ""),
    auxiliary_terms = .ndm_parse_csv_field(values$auxiliary_terms %||% ""),
    local_dynamic_terms = .ndm_parse_csv_field(values$local_dynamic_terms %||% ""),
    global_dynamic_terms = .ndm_parse_csv_field(values$global_dynamic_terms %||% ""),
    time_varying_terms = .ndm_parse_csv_field(values$time_varying_terms %||% ""),
    endogenous_terms = .ndm_parse_csv_field(values$endogenous_terms %||% ""),
    neural = list(
      local_inputs = .ndm_parse_csv_field(values$local_inputs %||% "")
    )
  )
}

.ndm_inline_auxiliary_expr <- function(expr, auxiliary) {
  expr <- .ndm_compact_text(expr)
  if (length(auxiliary) == 0L) {
    return(expr)
  }
  for (name in names(auxiliary)) {
    pattern <- sprintf("(?<![[:alnum:]_])%s(?![[:alnum:]_])", name)
    expr <- gsub(pattern, sprintf("(%s)", auxiliary[[name]]), expr, perl = TRUE)
  }
  expr
}

.ndm_normalize_prior_lhs <- function(lhs, init_state = FALSE) {
  lhs <- trimws(lhs)
  lhs <- sub("^\\\\[A-Za-z0-9]+\\(", "", lhs)
  lhs <- sub("\\)$", "", lhs)
  lhs <- gsub("\\s+", "", lhs)
  lhs <- gsub("[{}\\\\]", "", lhs)
  if (isTRUE(init_state)) {
    lhs <- sub("0$", "", lhs)
    lhs <- sub("_$", "", lhs)
  }
  lhs
}

.ndm_prior_lhs_is_init_state <- function(lhs) {
  lhs <- trimws(lhs)
  lhs <- sub("^\\\\[A-Za-z0-9]+\\(", "", lhs)
  lhs <- sub("\\)$", "", lhs)
  lhs <- gsub("\\s+", "", lhs)
  grepl("(_\\{[^{}]*0\\}|_0)$", lhs, perl = TRUE)
}

.ndm_parse_bayes_ode_text <- function(tex_text) {
  lines <- strsplit(tex_text, "\n", fixed = TRUE)[[1L]]
  start_idx <- grep("^%%%START BAYES ODE%%%\\s*$", lines)
  end_idx <- grep("^%%%END BAYES ODE%%%\\s*$", lines)
  if (length(start_idx) == 0L || length(end_idx) == 0L) {
    return(list(
      constants = numeric(),
      equations = character(),
      observations = character(),
      init_state_terms = character(),
      parameter_terms = character(),
      prior_terms = character()
    ))
  }
  lines <- lines[(start_idx[[1L]] + 1L):(end_idx[[1L]] - 1L)]
  lines <- gsub("%.*$", "", lines)
  lines <- trimws(lines)
  lines <- lines[nzchar(lines)]
  lines <- lines[grepl("^\\\\item\\s*\\$", lines)]
  lines <- sub("^\\\\item\\s*\\$", "", lines)
  lines <- sub("\\$$", "", lines)

  constant_lines <- lines[grepl("\\\\leftarrow", lines)]
  constants <- if (length(constant_lines) == 0L) {
    numeric()
  } else {
    vals <- vapply(
      constant_lines,
      function(line) {
        parts <- strsplit(line, "\\\\leftarrow")[[1L]]
        suppressWarnings(as.numeric(trimws(parts[[2L]])))
      },
      numeric(1L)
    )
    names(vals) <- vapply(
      constant_lines,
      function(line) trimws(strsplit(line, "\\\\leftarrow")[[1L]][[1L]]),
      character(1L)
    )
    vals
  }

  equation_lines <- lines[grepl("^\\\\Evolve\\{", lines)]
  equations <- if (length(equation_lines) == 0L) {
    character()
  } else {
    vals <- vapply(
      equation_lines,
      function(line) {
        parts <- strsplit(line, "=")[[1L]]
        .ndm_compact_text(parts[[2L]])
      },
      character(1L)
    )
    names(vals) <- vapply(
      equation_lines,
      function(line) {
        lhs <- sub("^\\\\Evolve\\{", "", strsplit(line, "=")[[1L]][[1L]])
        lhs <- sub("\\}$", "", lhs)
        gsub("[{}\\\\ ]", "", lhs)
      },
      character(1L)
    )
    vals
  }

  observations <- vapply(
    lines[grepl("^\\\\Observe\\s*=", lines)],
    function(line) .ndm_compact_text(strsplit(line, "=")[[1L]][[2L]]),
    character(1L)
  )

  prior_lines <- lines[grepl("\\\\sim", lines)]
  if (length(prior_lines) == 0L) {
    init_state_terms <- character()
    parameter_terms <- character()
  } else {
    prior_lhs <- vapply(
      prior_lines,
      function(line) trimws(strsplit(line, "\\\\sim")[[1L]][[1L]]),
      character(1L)
    )
    init_flags <- vapply(prior_lhs, .ndm_prior_lhs_is_init_state, logical(1L))
    init_state_terms <- unname(vapply(
      prior_lhs[init_flags],
      .ndm_normalize_prior_lhs,
      character(1L),
      init_state = TRUE
    ))
    parameter_terms <- unname(vapply(
      prior_lhs[!init_flags],
      .ndm_normalize_prior_lhs,
      character(1L),
      init_state = FALSE
    ))
  }
  prior_terms <- c(init_state_terms, parameter_terms)

  list(
    constants = constants,
    equations = equations,
    observations = observations,
    init_state_terms = init_state_terms,
    parameter_terms = parameter_terms,
    prior_terms = prior_terms
  )
}

.ndm_validate_execution_spec <- function(spec) {
  stopifnot(is.list(spec))
  if (length(spec$state_terms %||% character(0)) == 0L) {
    stop("Structured model specs must define at least one state term.", call. = FALSE)
  }
  if (length(spec$init_state_terms %||% character(0)) == 0L) {
    stop("Structured model specs must define at least one init-state term.", call. = FALSE)
  }
  if (!all(spec$init_state_terms %in% spec$state_terms)) {
    stop("`init_state_terms` must be a subset of `state_terms`.", call. = FALSE)
  }
  if (length(spec$equations %||% character()) != length(spec$state_terms)) {
    stop("Structured model specs must define one equation per state term.", call. = FALSE)
  }
  if (!setequal(names(spec$equations), spec$state_terms)) {
    stop("Structured model spec equation names must match `state_terms` exactly.", call. = FALSE)
  }
  if (length(spec$observations %||% character()) == 0L) {
    stop("Structured model specs must define at least one observation expression.", call. = FALSE)
  }
  invisible(spec)
}

.ndm_build_tex_from_execution_spec <- function(spec) {
  .ndm_validate_execution_spec(spec)
  template <- .ndm_extract_bayes_ode_template()
  bayes_lines <- c(
    "%%%START BAYES ODE%%%",
    .ndm_emit_execution_spec_metadata(spec),
    "",
    "%%%Priors ",
    "\\begin{itemize}"
  )

  if (length(spec$constants) > 0L) {
    for (name in names(spec$constants)) {
      bayes_lines <- c(
        bayes_lines,
        sprintf("\\item $%s \\leftarrow %s$", name, format(spec$constants[[name]], scientific = FALSE, trim = TRUE))
      )
    }
  }

  for (name in names(spec$parameters)) {
    param <- spec$parameters[[name]]
    bayes_lines <- c(
      bayes_lines,
      sprintf(
        "\\item $%s \\sim \\Dist{Normal}(%s,%s)$",
        .ndm_tex_prior_lhs(name, param$transformation),
        .ndm_transformed_prior_mean(param),
        as.character(param$prior_sd)
      )
    )
  }

  for (state_term in spec$init_state_terms) {
    prior <- spec$state_init_priors[[state_term]]
    bayes_lines <- c(
      bayes_lines,
      sprintf(
        "\\item $%s \\sim \\Dist{Normal}(%s, %s)$",
        .ndm_state_init_symbol(state_term),
        as.character(prior$prior_mean),
        as.character(prior$prior_sd)
      )
    )
  }

  inlined_equations <- vapply(
    spec$equations,
    .ndm_inline_auxiliary_expr,
    character(1L),
    auxiliary = spec$auxiliary
  )
  for (name in names(inlined_equations)) {
    bayes_lines <- c(
      bayes_lines,
      sprintf("\\item $\\Evolve{%s} = %s$", name, inlined_equations[[name]])
    )
  }

  for (obs in spec$observations) {
    bayes_lines <- c(
      bayes_lines,
      sprintf("\\item $\\Observe = %s$", .ndm_inline_auxiliary_expr(obs, spec$auxiliary))
    )
  }

  bayes_lines <- c(
    bayes_lines,
    "\\end{itemize}",
    "%%%END BAYES ODE%%%"
  )

  paste(c(template$prefix, bayes_lines, template$suffix), collapse = "\n")
}

.ndm_make_execution_spec <- function(preset = "custom",
                                     family = NULL,
                                     family_args = list(),
                                     description,
                                     states,
                                     parameters,
                                     equations,
                                     observations,
                                     constants = c(N = 1),
                                     auxiliary = character(),
                                     local_dynamic_terms = character(),
                                     global_dynamic_terms = character(),
                                     time_varying_terms = character(),
                                     endogenous_terms = character(),
                                     source_format = "structured",
                                     init_state_terms = states,
                                     state_init_priors = NULL,
                                     neural = list(local_inputs = states)) {
  if (is.null(state_init_priors)) {
    state_init_priors <- stats::setNames(
      replicate(length(init_state_terms), .ndm_state_init_prior_spec(), simplify = FALSE),
      init_state_terms
    )
  }
  spec <- list(
    preset = preset,
    family = family,
    family_args = .ndm_coerce_family_args(family_args),
    description = description,
    source_format = source_format,
    state_terms = as.character(states),
    init_state_terms = as.character(init_state_terms),
    parameter_terms = names(parameters),
    constant_terms = names(constants),
    constants = .ndm_as_named_numeric_list(constants, arg = "constants"),
    parameters = parameters,
    equations = stats::setNames(as.character(equations), names(equations)),
    observations = as.character(observations),
    auxiliary = stats::setNames(as.character(auxiliary), names(auxiliary)),
    auxiliary_terms = names(auxiliary),
    local_dynamic_terms = as.character(local_dynamic_terms),
    global_dynamic_terms = as.character(global_dynamic_terms),
    time_varying_terms = as.character(time_varying_terms),
    endogenous_terms = as.character(endogenous_terms),
    neural = neural,
    state_init_priors = state_init_priors
  )
  .ndm_validate_execution_spec(spec)
  spec
}

.ndm_tb_positive_rate <- function(mean, sd = 0.25) {
  .ndm_param_spec("InvSoftPlus", prior_mean = mean, prior_sd = sd)
}

.ndm_tb_fraction <- function(mean, sd = 0.25) {
  .ndm_param_spec("InvSigmoid", prior_mean = mean, prior_sd = sd)
}

.ndm_tb_auxiliary_time_varying <- function(kind) {
  switch(
    kind,
    k = "x1 * (max(1.0, t) ^ x2)",
    l = "x1 * (x2 + exp((-1 * x3 * t)))",
    stop("Unknown TB time-varying auxiliary kind.", call. = FALSE)
  )
}

.ndm_tb_observation_target <- function(family_args) {
  target <- family_args$observation_target %||% "incidence"
  target <- as.character(target)
  if (length(target) != 1L || is.na(target) || !target %in% c("incidence", "cumulative")) {
    stop(
      "`family_args$observation_target` must be either \"incidence\" or \"cumulative\".",
      call. = FALSE
    )
  }
  target
}

.ndm_tb_dynamics_variant <- function(family_args) {
  variant <- family_args$dynamics_variant %||% "balanced_incidence"
  variant <- as.character(variant)
  if (length(variant) != 1L || is.na(variant) || !variant %in% c("progression", "balanced_incidence")) {
    stop(
      "`family_args$dynamics_variant` must be either \"progression\" or \"balanced_incidence\".",
      call. = FALSE
    )
  }
  variant
}

.ndm_tb_family_numeric_arg <- function(family_args, name, default, positive = TRUE) {
  value <- family_args[[name]] %||% default
  value <- suppressWarnings(as.numeric(value))
  if (length(value) != 1L || !is.finite(value) || (isTRUE(positive) && value <= 0)) {
    stop("`family_args$", name, "` must be a finite", if (isTRUE(positive)) " positive" else "", " numeric scalar.", call. = FALSE)
  }
  value
}

.ndm_tb_balanced_parameters <- function(parameters, family_args) {
  if (any(c("mu", "gamma_i") %in% names(parameters))) {
    stop("Balanced TB specs reserve parameter names `mu` and `gamma_i`.", call. = FALSE)
  }
  c(
    parameters,
    list(
      mu = .ndm_tb_positive_rate(
        .ndm_tb_family_numeric_arg(family_args, "balanced_mu_mean", 0.02),
        .ndm_tb_family_numeric_arg(family_args, "balanced_mu_sd", 0.05)
      ),
      gamma_i = .ndm_tb_positive_rate(
        .ndm_tb_family_numeric_arg(family_args, "balanced_gamma_i_mean", 1.0),
        .ndm_tb_family_numeric_arg(family_args, "balanced_gamma_i_sd", 0.35)
      )
    )
  )
}

.ndm_tb_balanced_state_init_priors <- function(states, family_args) {
  priors <- stats::setNames(
    replicate(length(states), .ndm_state_init_prior_spec(), simplify = FALSE),
    states
  )
  initial_incidence <- .ndm_tb_family_numeric_arg(family_args, "balanced_initial_incidence_mean", 0.08, positive = FALSE)
  mu <- .ndm_tb_family_numeric_arg(family_args, "balanced_mu_mean", 0.02)
  gamma_i <- .ndm_tb_family_numeric_arg(family_args, "balanced_gamma_i_mean", 1.0)
  i0_sd <- .ndm_tb_family_numeric_arg(family_args, "balanced_i0_prior_sd", 0.1)
  if ("i_l" %in% names(priors)) {
    priors[["i_l"]] <- .ndm_state_init_prior_spec(
      prior_mean = initial_incidence / (gamma_i + mu),
      prior_sd = i0_sd
    )
  }
  priors
}

.ndm_tb_susceptible_infection_outflow <- function(equations) {
  s_equation <- .ndm_compact_text(equations[["s_l"]] %||% "")
  if (!nzchar(s_equation)) {
    stop("TB structures must define an `s_l` susceptible equation.", call. = FALSE)
  }
  outflow <- sub("^-\\s*", "", s_equation)
  if (identical(outflow, s_equation)) {
    stop(
      "Balanced TB specs require the original `s_l` equation to be a susceptible outflow.",
      call. = FALSE
    )
  }
  .ndm_compact_text(outflow)
}

.ndm_tb_balance_equations <- function(states, parameters, equations, family_args) {
  incidence_expr <- .ndm_compact_text(equations[["i_l"]] %||% "")
  if (!nzchar(incidence_expr)) {
    stop("TB structures must define an `i_l` progression equation.", call. = FALSE)
  }
  susceptible_infection_outflow <- .ndm_tb_susceptible_infection_outflow(equations)
  latent_states <- setdiff(states, c("s_l", "i_l"))
  balanced_equations <- equations
  balanced_equations[["s_l"]] <- sprintf(
    "mu * N + gamma_i * i_l - (%s) - mu * s_l",
    susceptible_infection_outflow
  )
  for (state in latent_states) {
    balanced_equations[[state]] <- sprintf("(%s) - mu * %s", equations[[state]], state)
  }
  balanced_equations[["i_l"]] <- sprintf("(%s) - (gamma_i + mu) * i_l", incidence_expr)
  list(
    parameters = .ndm_tb_balanced_parameters(parameters, family_args),
    equations = balanced_equations,
    incidence_expr = incidence_expr,
    state_init_priors = .ndm_tb_balanced_state_init_priors(states, family_args)
  )
}

.ndm_tb_observations_from_equations <- function(equations, observation_target, incidence_expr = NULL) {
  if (identical(observation_target, "cumulative")) {
    return("i_l")
  }
  i_l_equation <- incidence_expr %||% equations[["i_l"]]
  if (is.null(i_l_equation) || !nzchar(i_l_equation)) {
    stop("TB structures must define an `i_l` progression equation.", call. = FALSE)
  }
  unname(i_l_equation)
}

.ndm_make_tb_execution_spec <- function(observation_target, ...) {
  args <- list(...)
  family_args <- args$family_args %||% list()
  dynamics_variant <- .ndm_tb_dynamics_variant(family_args)
  incidence_expr <- NULL
  if (identical(dynamics_variant, "balanced_incidence")) {
    if (identical(observation_target, "cumulative")) {
      stop("Balanced TB specs require `observation_target = \"incidence\"`.", call. = FALSE)
    }
    balanced <- .ndm_tb_balance_equations(
      states = args$states,
      parameters = args$parameters,
      equations = args$equations,
      family_args = family_args
    )
    args$parameters <- balanced$parameters
    args$equations <- balanced$equations
    args$state_init_priors <- args$state_init_priors %||% balanced$state_init_priors
    incidence_expr <- balanced$incidence_expr
  }
  args$observations <- .ndm_tb_observations_from_equations(
    equations = args$equations,
    observation_target = observation_target,
    incidence_expr = incidence_expr
  )
  do.call(.ndm_make_execution_spec, args)
}

.ndm_tb_structure_spec <- function(family, family_args = list()) {
  family_args <- .ndm_coerce_family_args(family_args)
  n <- if (!is.null(family_args$n)) as.integer(family_args$n) else 3L
  if (!is.na(n) && n < 2L) {
    stop("TB chain-family specs require `family_args$n >= 2`.", call. = FALSE)
  }
  observation_target <- .ndm_tb_observation_target(family_args)

  switch(
    family,
    tb_a = .ndm_make_tb_execution_spec(
      observation_target = observation_target,
      preset = "tb_a",
      family = "tb_a",
      family_args = family_args,
      description = "TB structure A: S -> L -> I.",
      states = c("s_l", "l_l", "i_l"),
      parameters = list(
        lambda = .ndm_tb_positive_rate(0.08),
        c = .ndm_tb_positive_rate(0.15)
      ),
      equations = c(
        s_l = "- lambda * s_l",
        l_l = "lambda * s_l - c * l_l",
        i_l = "c * l_l"
      )
    ),
    tb_b = {
      fast_states <- sprintf("lf%s_l", seq_len(n))
      states <- c("s_l", fast_states, "ls_l", "i_l")
      parameters <- c(
        list(lambda = .ndm_tb_positive_rate(0.08), c = .ndm_tb_positive_rate(0.05)),
        stats::setNames(lapply(seq_len(n), function(...) .ndm_tb_positive_rate(0.18)), sprintf("k%s", seq_len(n))),
        stats::setNames(lapply(seq_len(n), function(...) .ndm_tb_positive_rate(0.12)), sprintf("d%s", seq_len(n)))
      )
      equations <- c()
      equations[["s_l"]] <- "- lambda * s_l"
      equations[[fast_states[[1L]]]] <- "- (k1 + d1) * lf1_l + lambda * s_l"
      if (n > 1L) {
        for (i in 2:n) {
          state_name <- fast_states[[i]]
          equations[[state_name]] <- sprintf("k%s * %s - (k%s + d%s) * %s", i - 1L, fast_states[[i - 1L]], i, i, state_name)
        }
      }
      equations[["ls_l"]] <- sprintf("k%s * %s - c * ls_l", n, fast_states[[n]])
      infectious_terms <- c(
        vapply(seq_len(n), function(i) sprintf("d%s * %s", i, fast_states[[i]]), character(1L)),
        "c * ls_l"
      )
      equations[["i_l"]] <- paste(infectious_terms, collapse = " + ")
      .ndm_make_tb_execution_spec(
        observation_target = observation_target,
        preset = "tb_b",
        family = "tb_b",
        family_args = utils::modifyList(list(n = n), family_args),
        description = "TB structure B: fast latent chain into slow latent and infectious.",
        states = states,
        parameters = parameters,
        equations = equations
      )
    },
    tb_c = .ndm_make_tb_execution_spec(
      observation_target = observation_target,
      preset = "tb_c",
      family = "tb_c",
      family_args = family_args,
      description = "TB structure C: split fast/slow latent progression.",
      states = c("s_l", "lf_l", "ls_l", "i_l"),
      parameters = list(
        lambda = .ndm_tb_positive_rate(0.08),
        d = .ndm_tb_positive_rate(0.14),
        e = .ndm_tb_positive_rate(0.06),
        c = .ndm_tb_positive_rate(0.05)
      ),
      equations = c(
        s_l = "- lambda * s_l",
        lf_l = "lambda * s_l - (d + e) * lf_l",
        ls_l = "e * lf_l - c * ls_l",
        i_l = "d * lf_l + c * ls_l"
      )
    ),
    tb_d = .ndm_make_tb_execution_spec(
      observation_target = observation_target,
      preset = "tb_d",
      family = "tb_d",
      family_args = family_args,
      description = "TB structure D: direct S -> I progression.",
      states = c("s_l", "i_l"),
      parameters = list(
        lambda = .ndm_tb_positive_rate(0.08)
      ),
      equations = c(
        s_l = "- lambda * s_l",
        i_l = "lambda * s_l"
      )
    ),
    tb_e = .ndm_make_tb_execution_spec(
      observation_target = observation_target,
      preset = "tb_e",
      family = "tb_e",
      family_args = family_args,
      description = "TB structure E: direct progression fraction plus latent path.",
      states = c("s_l", "l_l", "i_l"),
      parameters = list(
        lambda = .ndm_tb_positive_rate(0.08),
        a = .ndm_tb_fraction(0.25),
        c = .ndm_tb_positive_rate(0.08)
      ),
      equations = c(
        s_l = "- lambda * s_l",
        l_l = "lambda * (1 - a) * s_l - c * l_l",
        i_l = "lambda * a * s_l + c * l_l"
      )
    ),
    tb_f = .ndm_make_tb_execution_spec(
      observation_target = observation_target,
      preset = "tb_f",
      family = "tb_f",
      family_args = family_args,
      description = "TB structure F: split incident cases into slow and fast latent tracks.",
      states = c("s_l", "ls_l", "lf_l", "i_l"),
      parameters = list(
        lambda = .ndm_tb_positive_rate(0.08),
        b = .ndm_tb_fraction(0.35),
        c = .ndm_tb_positive_rate(0.05),
        d = .ndm_tb_positive_rate(0.14)
      ),
      equations = c(
        s_l = "- lambda * s_l",
        ls_l = "lambda * (1 - b) * s_l - c * ls_l",
        lf_l = "lambda * b * s_l - d * lf_l",
        i_l = "c * ls_l + d * lf_l"
      )
    ),
    tb_g = .ndm_make_tb_execution_spec(
      observation_target = observation_target,
      preset = "tb_g",
      family = "tb_g",
      family_args = family_args,
      description = "TB structure G: direct progression plus fast latent decay to slow latent.",
      states = c("s_l", "lf_l", "ls_l", "i_l"),
      parameters = list(
        lambda = .ndm_tb_positive_rate(0.08),
        a = .ndm_tb_fraction(0.2),
        d = .ndm_tb_positive_rate(0.14),
        e = .ndm_tb_positive_rate(0.06),
        c = .ndm_tb_positive_rate(0.05)
      ),
      equations = c(
        s_l = "- lambda * s_l",
        lf_l = "lambda * (1 - a) * s_l - (d + e) * lf_l",
        ls_l = "e * lf_l - c * ls_l",
        i_l = "lambda * a * s_l + d * lf_l + c * ls_l"
      )
    ),
    tb_h = .ndm_make_tb_execution_spec(
      observation_target = observation_target,
      preset = "tb_h",
      family = "tb_h",
      family_args = family_args,
      description = "TB structure H: slow latent reactivation into fast latent.",
      states = c("s_l", "ls_l", "lf_l", "i_l"),
      parameters = list(
        lambda = .ndm_tb_positive_rate(0.08),
        b = .ndm_tb_fraction(0.35),
        f = .ndm_tb_positive_rate(0.04),
        d = .ndm_tb_positive_rate(0.14)
      ),
      equations = c(
        s_l = "- lambda * s_l",
        ls_l = "lambda * (1 - b) * s_l - f * ls_l",
        lf_l = "lambda * b * s_l + f * ls_l - d * lf_l",
        i_l = "d * lf_l"
      )
    ),
    tb_i = .ndm_make_tb_execution_spec(
      observation_target = observation_target,
      preset = "tb_i",
      family = "tb_i",
      family_args = family_args,
      description = "TB structure I: fast/slow latent exchange with infectious progression from fast latent only.",
      states = c("s_l", "lf_l", "ls_l", "i_l"),
      parameters = list(
        lambda = .ndm_tb_positive_rate(0.08),
        f = .ndm_tb_positive_rate(0.04),
        d = .ndm_tb_positive_rate(0.14),
        e = .ndm_tb_positive_rate(0.06)
      ),
      equations = c(
        s_l = "- lambda * s_l",
        lf_l = "lambda * s_l + f * ls_l - (d + e) * lf_l",
        ls_l = "e * lf_l - f * ls_l",
        i_l = "d * lf_l"
      )
    ),
    tb_j = {
      chain_states <- sprintf("l%s_l", seq_len(n))
      states <- c("s_l", chain_states, "i_l")
      parameters <- c(
        list(lambda = .ndm_tb_positive_rate(0.08), c = .ndm_tb_positive_rate(0.05)),
        stats::setNames(lapply(seq_len(n - 1L), function(...) .ndm_tb_positive_rate(0.12)), sprintf("f%s", seq_len(n - 1L)))
      )
      equations <- c()
      equations[["s_l"]] <- "- lambda * s_l"
      equations[[chain_states[[1L]]]] <- "lambda * s_l - f1 * l1_l"
      if (n > 2L) {
        for (i in 2:(n - 1L)) {
          state_name <- chain_states[[i]]
          equations[[state_name]] <- sprintf("f%s * %s - f%s * %s", i - 1L, chain_states[[i - 1L]], i, state_name)
        }
      }
      equations[[chain_states[[n]]]] <- sprintf("f%s * %s - c * %s", n - 1L, chain_states[[n - 1L]], chain_states[[n]])
      equations[["i_l"]] <- sprintf("c * %s", chain_states[[n]])
      .ndm_make_tb_execution_spec(
        observation_target = observation_target,
        preset = "tb_j",
        family = "tb_j",
        family_args = utils::modifyList(list(n = n), family_args),
        description = "TB structure J: Erlang-style latent chain.",
        states = states,
        parameters = parameters,
        equations = equations
      )
    },
    tb_k = .ndm_make_tb_execution_spec(
      observation_target = observation_target,
      preset = "tb_k",
      family = "tb_k",
      family_args = family_args,
      description = "TB structure K: time-varying progression c_t with power law.",
      states = c("s_l", "l_l", "i_l"),
      parameters = list(
        lambda = .ndm_tb_positive_rate(0.08),
        x1 = .ndm_tb_positive_rate(0.06),
        x2 = .ndm_tb_positive_rate(0.15)
      ),
      auxiliary = c(c_t = .ndm_tb_auxiliary_time_varying("k")),
      equations = c(
        s_l = "- lambda * s_l",
        l_l = "lambda * s_l - c_t * l_l",
        i_l = "c_t * l_l"
      ),
      time_varying_terms = "c_t"
    ),
    tb_l = .ndm_make_tb_execution_spec(
      observation_target = observation_target,
      preset = "tb_l",
      family = "tb_l",
      family_args = family_args,
      description = "TB structure L: time-varying progression c_t with exponential decay.",
      states = c("s_l", "l_l", "i_l"),
      parameters = list(
        lambda = .ndm_tb_positive_rate(0.08),
        x1 = .ndm_tb_positive_rate(0.06),
        x2 = .ndm_tb_positive_rate(0.3),
        x3 = .ndm_tb_positive_rate(0.08)
      ),
      auxiliary = c(c_t = .ndm_tb_auxiliary_time_varying("l")),
      equations = c(
        s_l = "- lambda * s_l",
        l_l = "lambda * s_l - c_t * l_l",
        i_l = "c_t * l_l"
      ),
      time_varying_terms = "c_t"
    ),
    stop("Unknown structured model family `", family, "`.", call. = FALSE)
  )
}

.ndm_structured_spec_registry <- function() {
  list(
    tb_a = list(
      family = "tb_a",
      compartments = "tb",
      description = "TB structure A (S -> L -> I)."
    ),
    tb_b = list(
      family = "tb_b",
      compartments = "tb",
      description = "TB structure B (parameterized fast-latent chain into slow latent)."
    ),
    tb_c = list(
      family = "tb_c",
      compartments = "tb",
      description = "TB structure C (fast latent split to slow latent or infectious)."
    ),
    tb_d = list(
      family = "tb_d",
      compartments = "tb",
      description = "TB structure D (direct S -> I)."
    ),
    tb_e = list(
      family = "tb_e",
      compartments = "tb",
      description = "TB structure E (direct-progression fraction plus latent path)."
    ),
    tb_f = list(
      family = "tb_f",
      compartments = "tb",
      description = "TB structure F (fast/slow latent split)."
    ),
    tb_g = list(
      family = "tb_g",
      compartments = "tb",
      description = "TB structure G (direct progression plus fast-to-slow latent decay)."
    ),
    tb_h = list(
      family = "tb_h",
      compartments = "tb",
      description = "TB structure H (slow latent reactivation)."
    ),
    tb_i = list(
      family = "tb_i",
      compartments = "tb",
      description = "TB structure I (fast/slow latent exchange)."
    ),
    tb_j = list(
      family = "tb_j",
      compartments = "tb",
      description = "TB structure J (parameterized Erlang-style latent chain)."
    ),
    tb_k = list(
      family = "tb_k",
      compartments = "tb",
      description = "TB structure K (power-law time-varying progression)."
    ),
    tb_l = list(
      family = "tb_l",
      compartments = "tb",
      description = "TB structure L (exponential-decay time-varying progression)."
    )
  )
}

.ndm_model_spec_from_execution_spec <- function(execution_spec,
                                                model_type = c("DecoderOnly", "NeuralODE"),
                                                preset = execution_spec$preset %||% "custom",
                                                description = execution_spec$description %||% "Structured model specification.") {
  model_type <- match.arg(model_type)
  execution_spec <- .ndm_validate_execution_spec(execution_spec)
  tex_text <- .ndm_build_tex_from_execution_spec(execution_spec)
  .ndm_make_classed_list(
    list(
      preset = preset,
      model_type = model_type,
      compartments = .ndm_execution_spec_compartments(execution_spec),
      dynamic_beta = FALSE,
      dynamic_global = length(execution_spec$global_dynamic_terms) > 0L,
      multi_outcome = length(execution_spec$observations) > 1L,
      description = description,
      source_format = execution_spec$source_format,
      family = execution_spec$family %||% NULL,
      family_args = execution_spec$family_args %||% list(),
      time_varying_terms = unique(execution_spec$time_varying_terms %||% character(0)),
      endogenous_terms = unique(execution_spec$endogenous_terms %||% character(0)),
      state_terms = execution_spec$state_terms,
      init_state_terms = execution_spec$init_state_terms,
      parameter_terms = execution_spec$parameter_terms,
      auxiliary_terms = execution_spec$auxiliary_terms %||% character(0),
      local_dynamic_terms = execution_spec$local_dynamic_terms %||% character(0),
      global_dynamic_terms = execution_spec$global_dynamic_terms %||% character(0),
      source_path = NULL,
      tex_text = tex_text,
      execution_spec = execution_spec
    ),
    "ndm_model_spec"
  )
}

.ndm_structured_spec_from_preset <- function(preset,
                                             model_type = c("DecoderOnly", "NeuralODE"),
                                             family_args = NULL,
                                             time_varying_terms = NULL,
                                             endogenous_terms = NULL) {
  model_type <- match.arg(model_type)
  registry <- .ndm_structured_spec_registry()
  if (!preset %in% names(registry)) {
    stop("Unknown structured preset `", preset, "`.", call. = FALSE)
  }
  info <- registry[[preset]]
  execution_spec <- .ndm_tb_structure_spec(info$family, family_args = family_args)
  if (!is.null(time_varying_terms)) {
    execution_spec$time_varying_terms <- unique(as.character(time_varying_terms))
  }
  if (!is.null(endogenous_terms)) {
    execution_spec$endogenous_terms <- unique(as.character(endogenous_terms))
  }
  .ndm_model_spec_from_execution_spec(
    execution_spec = execution_spec,
    model_type = model_type,
    preset = preset,
    description = info$description
  )
}

.ndm_parse_tex_execution_spec <- function(tex_text) {
  metadata <- .ndm_parse_execution_spec_metadata(tex_text)
  bayes <- .ndm_parse_bayes_ode_text(tex_text)
  if (is.null(metadata)) {
    state_terms <- names(bayes$equations)
    return(list(
      source_format = "tex",
      preset = "custom",
      family = NULL,
      family_args = list(),
      state_terms = state_terms,
      init_state_terms = bayes$init_state_terms,
      parameter_terms = bayes$parameter_terms,
      auxiliary_terms = character(0),
      local_dynamic_terms = character(0),
      global_dynamic_terms = character(0),
      time_varying_terms = character(0),
      endogenous_terms = character(0),
      neural = list(local_inputs = state_terms)
    ))
  }

  metadata$constants <- bayes$constants
  metadata$equations <- bayes$equations
  metadata$observations <- bayes$observations
  metadata$auxiliary <- stats::setNames(rep("", length(metadata$auxiliary_terms)), metadata$auxiliary_terms)
  metadata
}

#' Inspect and create epidemic model specifications from structured declarations
#'
#' `ndm_model_spec_from_structure()` builds an `ndm_model_spec` from a
#' structured declaration. The declaration may either describe the full model
#' directly or identify a supported built-in family plus optional `family_args`
#' that expand parameterized structures into explicit executable equations.
#'
#' For a maintainer-oriented guide to the required fields, parameter-spec shape,
#' TeX contract, and extension workflow, see
#' \code{vignette("ode-structures", package = "ndm")}.
#'
#' @param structure A structured model declaration.
#' @param model_type Model family metadata attached to the returned
#'   specification. Either `"DecoderOnly"` or `"NeuralODE"`.
#' @param family_args Optional named list used to expand parameterized family
#'   declarations. For structured TB families, `observation_target` chooses
#'   `"incidence"` (default flow into `i_l`) or `"cumulative"` (`i_l`).
#'   TB families use `dynamics_variant = "balanced_incidence"` by default,
#'   preserving the original active-progression flow while adding normalized
#'   at-risk renewal (`N = 1`), active-disease exit back to susceptibility,
#'   latent mortality, and incidence-flow observations. Set
#'   `dynamics_variant = "progression"` for legacy one-way depletion dynamics.
#'   Balanced TB specs require incidence observations.
#'
#' @returns An object of class `ndm_model_spec`.
#'
#' @examples
#' family_spec <- ndm_model_spec_from_structure(
#'   list(
#'     family = "tb_b",
#'     family_args = list(n = 4L)
#'   ),
#'   model_type = "NeuralODE"
#' )
#' family_spec$state_terms
#'
#' toy_spec <- ndm_model_spec_from_structure(
#'   list(
#'     preset = "toy_slqi",
#'     description = "Toy four-state latent structure with a holding compartment.",
#'     states = c("s_l", "l_l", "q_l", "i_l"),
#'     parameters = list(
#'       lambda = list(
#'         transformation = "InvSoftPlus",
#'         prior_mean = 0.08,
#'         prior_sd = 0.25
#'       ),
#'       c = list(
#'         transformation = "InvSoftPlus",
#'         prior_mean = 0.05,
#'         prior_sd = 0.25
#'       ),
#'       h = list(
#'         transformation = "InvSoftPlus",
#'         prior_mean = 0.03,
#'         prior_sd = 0.25
#'       )
#'     ),
#'     equations = c(
#'       s_l = "- lambda * s_l",
#'       l_l = "lambda * s_l - (c + h) * l_l",
#'       q_l = "h * l_l",
#'       i_l = "c * l_l"
#'     ),
#'     observations = "i_l"
#'   ),
#'   model_type = "NeuralODE"
#' )
#' toy_spec$parameter_terms
#'
#' @export
ndm_model_spec_from_structure <- function(structure,
                                          model_type = c("DecoderOnly", "NeuralODE"),
                                          family_args = NULL) {
  model_type <- match.arg(model_type)

  if (!is.list(structure)) {
    stop("`structure` must be a named list.", call. = FALSE)
  }

  family <- structure$family %||% NULL
  if (!is.null(family)) {
    spec <- .ndm_tb_structure_spec(as.character(family), family_args = utils::modifyList(structure$family_args %||% list(), family_args %||% list()))
    if (!is.null(structure$description)) {
      spec$description <- structure$description
    }
    return(.ndm_model_spec_from_execution_spec(spec, model_type = model_type))
  }

  required_fields <- c("states", "parameters", "equations", "observations")
  missing <- required_fields[!required_fields %in% names(structure)]
  if (length(missing) > 0L) {
    stop(
      "Structured model declarations must define: ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }

  execution_spec <- .ndm_make_execution_spec(
    preset = structure$preset %||% "custom",
    family = NULL,
    family_args = family_args %||% list(),
    description = structure$description %||% "Custom structured model specification.",
    states = structure$states,
    parameters = structure$parameters,
    equations = structure$equations,
    observations = structure$observations,
    constants = structure$constants %||% c(N = 1),
    auxiliary = structure$auxiliary %||% character(),
    local_dynamic_terms = structure$local_dynamic_terms %||% character(),
    global_dynamic_terms = structure$global_dynamic_terms %||% character(),
    time_varying_terms = structure$time_varying_terms %||% character(),
    endogenous_terms = structure$endogenous_terms %||% character(),
    init_state_terms = structure$init_state_terms %||% structure$states,
    state_init_priors = structure$state_init_priors %||% NULL,
    neural = structure$neural %||% list(local_inputs = structure$states)
  )

  .ndm_model_spec_from_execution_spec(
    execution_spec = execution_spec,
    model_type = model_type
  )
}
