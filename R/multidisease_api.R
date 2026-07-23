.ndm_runtime_get0 <- function(env, name, ifnotfound = NULL) {
  get0(name, envir = env, inherits = FALSE, ifnotfound = ifnotfound)
}

.ndm_with_working_directory <- function(path, fn) {
  stopifnot(is.function(fn))

  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(path)
  fn()
}

.ndm_multidisease_required_runtime_globals <- function() {
  c(
    "ContextLength",
    "evaluationTime",
    "dataInputs",
    "initialTransform",
    "initialNormType",
    "paddingMethod",
    "OSSType",
    "nSamplesTrain"
  )
}

.ndm_normalize_disease_label <- function(x) {
  x <- trimws(tolower(as.character(x)))
  gsub("[^a-z0-9]+", "", x)
}

.ndm_format_unique <- function(x) {
  paste(unique(as.character(x)), collapse = ", ")
}

.ndm_multidisease_fill_bidirectional <- function(x) {
  if (length(x) == 0L) {
    return(x)
  }

  for (i in seq_along(x)) {
    if (is.na(x[[i]]) && i > 1L) {
      x[[i]] <- x[[i - 1L]]
    }
  }
  for (i in rev(seq_along(x))) {
    if (is.na(x[[i]]) && i < length(x)) {
      x[[i]] <- x[[i + 1L]]
    }
  }

  x
}

.ndm_multidisease_require_paths <- function(paths, description) {
  existing <- paths[file.exists(paths)]
  if (length(existing) > 0L) {
    return(.ndm_normalize_path(existing[[1]], must_work = TRUE))
  }

  stop(
    description,
    " not found. Checked: ",
    paste(paths, collapse = ", "),
    call. = FALSE
  )
}

.ndm_multidisease_require_dir <- function(paths, description) {
  existing <- paths[dir.exists(paths)]
  if (length(existing) > 0L) {
    return(.ndm_normalize_path(existing[[1]], must_work = TRUE))
  }

  stop(
    description,
    " directory not found. Checked: ",
    paste(paths, collapse = ", "),
    call. = FALSE
  )
}

.ndm_multidisease_resolve_diseases <- function(disease_names,
                                               data_format,
                                               available = NULL) {
  requested <- unique(as.character(disease_names))
  requested <- requested[nzchar(trimws(requested))]
  if (length(requested) == 0L) {
    stop("`disease_names` must contain at least one non-empty value.", call. = FALSE)
  }

  requested_norm <- .ndm_normalize_disease_label(requested)
  format_key <- toupper(as.character(data_format))

  if (format_key %in% c("WHO", "TYCHO")) {
    supported_norm <- c("tb", "tuberculosis", "allformsoftuberculosis", "tuberculosisallforms")
    bad <- requested[!requested_norm %in% supported_norm]
    if (length(bad) > 0L) {
      stop(
        "Requested disease names are not supported for ",
        data_format,
        ": ",
        .ndm_format_unique(bad),
        ". Supported values map to TB only.",
        call. = FALSE
      )
    }
    return("TB")
  }

  if (!identical(format_key, "IHME")) {
    stop("Unsupported multidisease `data_format`: ", data_format, call. = FALSE)
  }

  available <- unique(as.character(available))
  available <- available[nzchar(trimws(available))]
  if (length(available) == 0L) {
    stop("IHME data does not expose any `cause_name` values.", call. = FALSE)
  }

  available_norm <- .ndm_normalize_disease_label(available)
  alias_map <- c(
    hiv = "HIV/AIDS and sexually transmitted infections",
    hivaids = "HIV/AIDS and sexually transmitted infections",
    hivaidsandsexuallytransmittedinfections = "HIV/AIDS and sexually transmitted infections",
    sti = "HIV/AIDS and sexually transmitted infections",
    std = "HIV/AIDS and sexually transmitted infections",
    sexuallytransmittedinfections = "HIV/AIDS and sexually transmitted infections",
    all = "All causes",
    allcauses = "All causes"
  )

  resolved <- character()
  unresolved <- character()
  ambiguous <- character()

  for (i in seq_along(requested)) {
    request <- requested[[i]]
    request_norm <- requested_norm[[i]]
    candidates <- character()

    if (request_norm %in% names(alias_map)) {
      candidates <- c(candidates, alias_map[[request_norm]])
    }

    exact <- available[available_norm == request_norm]
    candidates <- unique(c(candidates, exact))

    if (length(candidates) == 0L) {
      fuzzy <- available[grepl(request_norm, available_norm, fixed = TRUE)]
      if (length(fuzzy) == 1L) {
        candidates <- fuzzy
      } else if (length(fuzzy) > 1L) {
        ambiguous <- c(ambiguous, request)
      }
    }

    candidates <- unique(candidates[candidates %in% available])
    if (length(candidates) == 1L) {
      resolved <- c(resolved, candidates)
    } else if (length(candidates) == 0L && !request %in% ambiguous) {
      unresolved <- c(unresolved, request)
    } else if (length(candidates) > 1L) {
      ambiguous <- c(ambiguous, request)
    }
  }

  if (length(ambiguous) > 0L) {
    stop(
      "Requested disease names are ambiguous for IHME: ",
      .ndm_format_unique(ambiguous),
      ". Available causes: ",
      .ndm_format_unique(available),
      call. = FALSE
    )
  }
  if (length(unresolved) > 0L) {
    stop(
      "Requested disease names are not available for IHME: ",
      .ndm_format_unique(unresolved),
      ". Available causes: ",
      .ndm_format_unique(available),
      call. = FALSE
    )
  }

  unique(resolved)
}

.ndm_multidisease_make_bundle <- function(data_format,
                                          resolved_diseases,
                                          truth_df_red,
                                          input_df_red,
                                          data_inputs_past,
                                          data_inputs_future = character(),
                                          true_value_names = "CountValue",
                                          outcome_metric = "CountValue") {
  list(
    data_format = data_format,
    resolved_diseases = resolved_diseases,
    truth_df_red = as.data.frame(truth_df_red, stringsAsFactors = FALSE),
    input_df_red = as.data.frame(input_df_red, stringsAsFactors = FALSE),
    dataInputs_colnames_past = unique(as.character(data_inputs_past)),
    dataInputs_colnames_future = unique(as.character(data_inputs_future)),
    true_value_names = unique(as.character(true_value_names)),
    all_true_value_names = unique(as.character(true_value_names)),
    outcome_metric = as.character(outcome_metric),
    nOutcomes = length(unique(as.character(true_value_names))),
    nPlaces = length(unique(as.character(truth_df_red$location_id)))
  )
}

.ndm_multidisease_load_tycho <- function(project_root,
                                         disease_names,
                                         outcome_metric = "CountValue") {
  Admin1ISO <- Admin1Name <- CountValue <- POP_population <- PeriodStartDate <- NULL
  date_start <- location_id <- location_id_numeric <- location_name <- NULL
  population <- state_abbr <- targetTime_id <- time_id <- year <- NULL

  resolved_diseases <- .ndm_multidisease_resolve_diseases(
    disease_names = disease_names,
    data_format = "Tycho"
  )

  tycho_dir <- .ndm_multidisease_require_dir(
    c(file.path(project_root, "Data", "MultiDiseaseRuns", "TB-USA-2")),
    "Tycho data"
  )
  csv_files <- list.files(tycho_dir, pattern = "\\.csv$", full.names = TRUE)
  if (length(csv_files) == 0L) {
    stop("Tycho data directory does not contain any CSV files: ", tycho_dir, call. = FALSE)
  }

  raw_input <- data.table::rbindlist(
    lapply(csv_files, function(path) data.table::fread(path, showProgress = FALSE)),
    fill = TRUE
  )
  raw_input[, date_start := as.Date(PeriodStartDate)]
  min_date <- min(raw_input$date_start, na.rm = TRUE)
  raw_input[, `:=`(
    location_id = as.character(Admin1ISO),
    location_name = as.character(Admin1Name),
    year = as.integer(format(date_start, "%Y")),
    time_id = as.integer(difftime(date_start, min_date, units = "weeks")),
    targetTime_id = as.integer(difftime(date_start, min_date, units = "weeks")),
    CountValue = as.numeric(CountValue)
  )]
  raw_input <- raw_input[!is.na(location_id) & !is.na(time_id)]

  pop_file <- .ndm_multidisease_require_paths(
    c(file.path(project_root, "Data", "MultiDiseaseRuns", "population_state_year.csv")),
    "Tycho population data"
  )
  population_dt <- data.table::fread(pop_file, header = FALSE, showProgress = FALSE)
  data.table::setnames(population_dt, c("state_abbr", "year", "population"))
  population_dt[, `:=`(
    state_abbr = as.character(state_abbr),
    year = as.integer(year),
    population = as.numeric(population)
  )]

  raw_input[, state_abbr := sub("^US-", "", location_id)]
  raw_input <- merge(
    raw_input,
    population_dt,
    by = c("state_abbr", "year"),
    all.x = TRUE,
    sort = FALSE
  )
  raw_input[, population := .ndm_multidisease_fill_bidirectional(population), by = state_abbr]
  raw_input[, POP_population := population]
  raw_input <- raw_input[!is.na(population) & population > 0]
  raw_input[, CountValue := CountValue / population]
  raw_input[, location_id_numeric := as.integer(factor(location_id, levels = sort(unique(location_id)))) - 1L]

  truth_df_red <- raw_input[
    ,
    list(
      CountValue = sum(CountValue, na.rm = TRUE),
      POP_population = max(POP_population, na.rm = TRUE)
    ),
    by = list(location_id, location_id_numeric, location_name, time_id, targetTime_id)
  ]
  if (!identical(outcome_metric, "CountValue")) {
    data.table::setnames(truth_df_red, "CountValue", outcome_metric)
  }

  .ndm_multidisease_make_bundle(
    data_format = "Tycho",
    resolved_diseases = resolved_diseases,
    truth_df_red = truth_df_red,
    input_df_red = truth_df_red,
    data_inputs_past = outcome_metric,
    true_value_names = outcome_metric,
    outcome_metric = outcome_metric
  )
}

.ndm_multidisease_load_who <- function(project_root,
                                       disease_names,
                                       outcome_metric = "CountValue") {
  CasesPerPop <- CountValue <- Covariate1 <- location_id <- location_id_numeric <- NULL
  location_name <- targetTime_id <- time_id <- year <- NULL

  resolved_diseases <- .ndm_multidisease_resolve_diseases(
    disease_names = disease_names,
    data_format = "WHO"
  )

  who_file <- .ndm_multidisease_require_paths(
    c(file.path(project_root, "Data", "MultiDiseaseRuns", "DiseasePool", "incidence-of-tuberculosis-sdgs.csv")),
    "WHO tuberculosis data"
  )
  who_dt <- data.table::fread(who_file, skip = 1L, showProgress = FALSE)
  data.table::setnames(
    who_dt,
    old = c("Entity", "Code", "Year", "Estimated incidence of all forms of tuberculosis"),
    new = c("location_name", "location_id", "year", "CasesPerPop"),
    skip_absent = TRUE
  )
  who_dt[, `:=`(
    location_name = as.character(location_name),
    location_id = as.character(location_id),
    year = as.integer(year),
    CasesPerPop = as.numeric(CasesPerPop)
  )]
  who_dt[is.na(location_id) | !nzchar(trimws(location_id)), location_id := location_name]
  min_year <- min(who_dt$year, na.rm = TRUE)
  who_dt[, `:=`(
    time_id = year - min_year,
    targetTime_id = year - min_year,
    CountValue = CasesPerPop / 1e3
  )]
  who_dt[, location_id_numeric := as.integer(factor(location_id, levels = sort(unique(location_id)))) - 1L]

  truth_df_red <- who_dt[
    !is.na(CountValue),
    list(CountValue = sum(CountValue, na.rm = TRUE)),
    by = list(location_id, location_id_numeric, location_name, time_id, targetTime_id)
  ]
  if (!identical(outcome_metric, "CountValue")) {
    data.table::setnames(truth_df_red, "CountValue", outcome_metric)
  }
  truth_df_red$Covariate1 <- truth_df_red[[outcome_metric]]
  input_df_red <- data.table::copy(truth_df_red)

  .ndm_multidisease_make_bundle(
    data_format = "WHO",
    resolved_diseases = resolved_diseases,
    truth_df_red = truth_df_red,
    input_df_red = input_df_red,
    data_inputs_past = c(outcome_metric, "Covariate1"),
    true_value_names = outcome_metric,
    outcome_metric = outcome_metric
  )
}

.ndm_multidisease_load_ihme <- function(project_root,
                                        disease_names,
                                        outcome_metric = "CountValue",
                                        desired_measure = NULL) {
  age_name <- cause_name <- CountFraction_tmp <- CountValue <- Covariate1 <- NULL
  location_id <- location_id_numeric <- location_name <- measure_name <- metric_name <- NULL
  metric_rank <- sex_name <- targetTime_id <- time_id <- val <- year <- NULL

  ihme_file <- .ndm_multidisease_require_paths(
    c(
      file.path(project_root, "Data", "MultiDiseaseRuns", "IHMEData", "IHME-GBD_2021_DATA-ea2ad67b-1", "IHME-GBD_2021_DATA-ea2ad67b-1.csv"),
      file.path(project_root, "Data", "MultiDiseaseRuns", "DiseasePool", "IHMEData", "IHME-GBD_2021_DATA-ea2ad67b-1", "IHME-GBD_2021_DATA-ea2ad67b-1.csv")
    ),
    "IHME data"
  )
  ihme_dt <- data.table::fread(ihme_file, showProgress = FALSE)
  ihme_dt <- ihme_dt[sex_name == "Both" & age_name == "All ages"]
  resolved_diseases <- .ndm_multidisease_resolve_diseases(
    disease_names = disease_names,
    data_format = "IHME",
    available = unique(ihme_dt$cause_name)
  )
  ihme_dt <- ihme_dt[cause_name %in% resolved_diseases]

  if (!is.null(desired_measure) && nzchar(desired_measure)) {
    ihme_dt <- ihme_dt[measure_name == desired_measure]
  } else {
    preferred_measure <- "Prevalence"
    available_measures <- unique(as.character(ihme_dt$measure_name))
    chosen_measure <- if (preferred_measure %in% available_measures) {
      preferred_measure
    } else {
      available_measures[[1]]
    }
    ihme_dt <- ihme_dt[measure_name == chosen_measure]
  }

  ihme_dt[, CountFraction_tmp := data.table::fifelse(
    metric_name == "Rate",
    as.numeric(val) / 1e5,
    data.table::fifelse(
      metric_name == "Percent",
      data.table::fifelse(as.numeric(val) > 1, as.numeric(val) / 100, as.numeric(val)),
      NA_real_
    )
  )]
  ihme_dt[, metric_rank := data.table::fifelse(
    metric_name == "Rate",
    1L,
    data.table::fifelse(metric_name == "Percent", 2L, 99L)
  )]
  ihme_dt <- ihme_dt[!is.na(CountFraction_tmp)]
  data.table::setorderv(ihme_dt, c("location_id", "year", "metric_rank"))
  ihme_dt <- ihme_dt[, .SD[1L], by = list(location_id, location_name, cause_name, year)]
  ihme_dt <- ihme_dt[
    ,
    list(CountValue = sum(CountFraction_tmp, na.rm = TRUE)),
    by = list(location_id, location_name, year)
  ]
  min_year <- min(ihme_dt$year, na.rm = TRUE)
  ihme_dt[, `:=`(
    time_id = as.integer(year - min_year),
    targetTime_id = as.integer(year - min_year),
    location_id = as.character(location_id)
  )]
  ihme_dt[, location_id_numeric := as.integer(factor(location_id, levels = sort(unique(location_id)))) - 1L]

  truth_df_red <- ihme_dt[
    !is.na(CountValue),
    list(CountValue = sum(CountValue, na.rm = TRUE)),
    by = list(location_id, location_id_numeric, location_name, time_id, targetTime_id)
  ]
  if (!identical(outcome_metric, "CountValue")) {
    data.table::setnames(truth_df_red, "CountValue", outcome_metric)
  }
  truth_df_red$Covariate1 <- truth_df_red[[outcome_metric]]
  input_df_red <- data.table::copy(truth_df_red)

  .ndm_multidisease_make_bundle(
    data_format = "IHME",
    resolved_diseases = resolved_diseases,
    truth_df_red = truth_df_red,
    input_df_red = input_df_red,
    data_inputs_past = c(outcome_metric, "Covariate1"),
    true_value_names = outcome_metric,
    outcome_metric = outcome_metric
  )
}

.ndm_load_multidisease_bundle <- function(project_root,
                                          data_format = "IHME",
                                          disease_names,
                                          outcome_metric = "CountValue",
                                          desired_measure = NULL) {
  format_key <- toupper(as.character(data_format))
  switch(
    format_key,
    IHME = .ndm_multidisease_load_ihme(
      project_root = project_root,
      disease_names = disease_names,
      outcome_metric = outcome_metric,
      desired_measure = desired_measure
    ),
    WHO = .ndm_multidisease_load_who(
      project_root = project_root,
      disease_names = disease_names,
      outcome_metric = outcome_metric
    ),
    TYCHO = .ndm_multidisease_load_tycho(
      project_root = project_root,
      disease_names = disease_names,
      outcome_metric = outcome_metric
    ),
    stop("Unsupported multidisease `data_format`: ", data_format, call. = FALSE)
  )
}

.ndm_multidisease_subset_bundle <- function(bundle, data_subset = "all") {
  location_id <- location_id_numeric <- location_name <- NULL
  data_subset <- as.character(data_subset)
  if (length(data_subset) != 1L || is.na(data_subset) || !nzchar(data_subset)) {
    stop("`data_subset` must be one non-empty value.", call. = FALSE)
  }
  truth <- data.table::as.data.table(bundle$truth_df_red)
  input <- data.table::as.data.table(bundle$input_df_red)
  if (identical(data_subset, "all")) {
    keep_place_ids <- unique(truth$location_id)
  } else if (identical(data_subset, "high_income")) {
    if (!"LOC2_region_name" %in% names(truth)) {
      warning("high_income subset requested but LOC2_region_name not available; using all locations")
      keep_place_ids <- unique(truth$location_id)
    } else {
      keep_place_ids <- unique(truth$location_id[
        truth$LOC2_region_name %in% c(
          "Western Europe",
          "High-income North America",
          "Central Europe",
          "High-income Asia Pacific"
        )
      ])
    }
  } else {
    stop("Unsupported multidisease `data_subset`: ", data_subset, call. = FALSE)
  }

  truth <- truth[location_id %in% keep_place_ids]
  input <- input[location_id %in% keep_place_ids]
  if ("location_name" %in% names(truth)) {
    truth <- truth[!is.na(location_name)]
  }
  if ("location_name" %in% names(input)) {
    input <- input[!is.na(location_name)]
  }
  if (nrow(truth) == 0L) {
    stop("The multidisease subset does not contain any observations.", call. = FALSE)
  }
  location_levels <- sort(unique(as.character(truth$location_id)))
  truth[, location_id_numeric := match(as.character(location_id), location_levels) - 1L]
  input[, location_id_numeric := match(as.character(location_id), location_levels) - 1L]
  if (!any(as.integer(truth$time_id) == 0L)) {
    stop("time_id seems to be non-zero indexed!", call. = FALSE)
  }

  bundle$truth_df_red <- as.data.frame(truth, stringsAsFactors = FALSE)
  bundle$input_df_red <- as.data.frame(input, stringsAsFactors = FALSE)
  bundle$nPlaces <- length(location_levels)
  bundle$data_subset <- data_subset
  bundle
}

.ndm_prepare_multidisease_bundle <- function(project_root,
                                             data_format,
                                             disease_names,
                                             outcome_metric,
                                             data_subset,
                                             desired_measure = NULL) {
  bundle <- .ndm_load_multidisease_bundle(
    project_root = project_root,
    data_format = data_format,
    disease_names = disease_names,
    outcome_metric = outcome_metric,
    desired_measure = desired_measure
  )
  .ndm_multidisease_subset_bundle(bundle, data_subset = data_subset)
}

.ndm_multidisease_resolve_inputs <- function(bundle, data_inputs = "all") {
  data_inputs <- as.character(data_inputs)
  if (length(data_inputs) == 0L || all(is.na(data_inputs)) ||
      (length(data_inputs) == 1L &&
       (!nzchar(data_inputs) || identical(data_inputs, "all")))) {
    selected <- setdiff(bundle$dataInputs_colnames_past, bundle$outcome_metric)
    if (length(selected) == 0L) {
      selected <- bundle$dataInputs_colnames_past
    }
  } else if (length(data_inputs) == 1L) {
    selected <- unique(strsplit(data_inputs, split = "__", fixed = TRUE)[[1L]])
  } else {
    selected <- unique(data_inputs)
  }
  selected <- selected[nzchar(selected)]
  missing_inputs <- selected[!selected %in% names(bundle$truth_df_red)]
  if (length(missing_inputs) > 0L) {
    stop(
      "Requested multidisease `dataInputs` are not present in the loaded data: ",
      paste(missing_inputs, collapse = ", "),
      call. = FALSE
    )
  }
  selected
}

.ndm_multidisease_table_bundle <- function(bundle,
                                           data_inputs = "all",
                                           dataset_call = .ndm_canonical_dataset_call) {
  selected_inputs <- .ndm_multidisease_resolve_inputs(bundle, data_inputs)
  table_bundle <- dataset_call(
    "ndm_real_table_bundle",
    truth_df = bundle$truth_df_red,
    data_inputs = selected_inputs,
    outcomes = bundle$true_value_names,
    disease = paste(bundle$resolved_diseases, collapse = "__"),
    outcome_metric = bundle$outcome_metric,
    per_capita_scaling_factor = 1,
    roll_window = 1L
  )
  list(
    bundle = bundle,
    table_bundle = table_bundle,
    data_inputs = selected_inputs,
    source_sha256 = dataset_call("ndm_real_source_sha256", table_bundle)
  )
}

.ndm_multidisease_row_value <- function(row_values,
                                        names,
                                        default = NULL) {
  if (is.data.frame(row_values)) {
    if (nrow(row_values) != 1L) {
      stop("A multidisease grid row must contain exactly one row.", call. = FALSE)
    }
    row_values <- as.list(row_values[1L, , drop = FALSE])
  }
  for (name in names) {
    value <- row_values[[name]] %||% NULL
    if (.ndm_runtime_value_is_present(value)) {
      return(value[[1L]])
    }
  }
  default
}

.ndm_multidisease_row_integer <- function(row_values,
                                          names,
                                          default = NULL,
                                          required = FALSE) {
  value <- .ndm_multidisease_row_value(row_values, names, default = default)
  if (is.null(value) && !isTRUE(required)) {
    return(NULL)
  }
  numeric_value <- suppressWarnings(as.numeric(as.character(value)))
  if (length(numeric_value) != 1L || !is.finite(numeric_value) ||
      numeric_value != floor(numeric_value)) {
    stop(
      "Multidisease grid field `", names[[1L]], "` must be one integer.",
      call. = FALSE
    )
  }
  as.integer(numeric_value)
}

.ndm_multidisease_dataset_spec <- function(row_values,
                                           bundle,
                                           data_subset = "all",
                                           lookahead = 12L,
                                           min_anchoring_time = 4L,
                                           dataset_call = .ndm_canonical_dataset_call) {
  selected_inputs <- .ndm_multidisease_resolve_inputs(
    bundle,
    .ndm_multidisease_row_value(row_values, "dataInputs", "all")
  )
  evaluation_horizon <- .ndm_multidisease_row_integer(
    row_values,
    "evaluationHorizon",
    4L
  )
  if (evaluation_horizon < 1L) {
    stop("Multidisease `evaluationHorizon` must be positive.", call. = FALSE)
  }
  dataset_call(
    "ndm_datasets_dataset_spec",
    kind = "real",
    disease = paste(bundle$resolved_diseases, collapse = "__"),
    context_length = .ndm_multidisease_row_integer(
      row_values,
      "ContextLength",
      required = TRUE
    ),
    lookahead = as.integer(lookahead),
    evaluation_time = .ndm_multidisease_row_integer(
      row_values,
      "evaluationTime",
      4L
    ),
    evaluation_sequence = seq_len(evaluation_horizon),
    evaluation_origin_time_id = .ndm_multidisease_row_integer(
      row_values,
      "evaluationOriginTimeID"
    ),
    evaluation_horizon = evaluation_horizon,
    inference_sampling = as.character(.ndm_multidisease_row_value(
      row_values,
      "inferenceSampling",
      "random"
    )),
    initial_transform = as.character(.ndm_multidisease_row_value(
      row_values,
      "initialTransform",
      "none"
    )),
    initial_norm_type = as.character(.ndm_multidisease_row_value(
      row_values,
      "initialNormType",
      "all"
    )),
    padding_method = as.character(.ndm_multidisease_row_value(
      row_values,
      "paddingMethod",
      "left"
    )),
    split_type = as.character(.ndm_multidisease_row_value(
      row_values,
      "OSSType",
      "OutOfTime"
    )),
    data_subset = as.character(data_subset),
    data_inputs = selected_inputs,
    outcomes = bundle$true_value_names,
    per_capita_scaling_factor = 1,
    roll_window = 1L,
    min_anchoring_time = as.integer(min_anchoring_time),
    train_location_fraction = 0.8,
    n_inference_samples = .ndm_multidisease_row_integer(
      row_values,
      "nObsInference",
      required = TRUE
    ),
    base_id = .ndm_multidisease_row_integer(
      row_values,
      "BaseID",
      required = TRUE
    ),
    outcome_metric = bundle$outcome_metric
  )
}

.ndm_multidisease_data_seed <- function(row_values) {
  seed <- .ndm_multidisease_row_integer(row_values, "dataSeed", 0L)
  if (seed < 0L) {
    stop("Multidisease `dataSeed` must be non-negative.", call. = FALSE)
  }
  seed
}

.ndm_multidisease_artifact_contract <- function(row_values,
                                                bundle,
                                                data_subset = "all",
                                                lookahead = 12L,
                                                min_anchoring_time = 4L,
                                                dataset_call = .ndm_canonical_dataset_call) {
  table_contract <- .ndm_multidisease_table_bundle(
    bundle,
    data_inputs = .ndm_multidisease_row_value(row_values, "dataInputs", "all"),
    dataset_call = dataset_call
  )
  dataset_spec <- .ndm_multidisease_dataset_spec(
    row_values = row_values,
    bundle = bundle,
    data_subset = data_subset,
    lookahead = lookahead,
    min_anchoring_time = min_anchoring_time,
    dataset_call = dataset_call
  )
  prepared <- dataset_call(
    "ndm_real_prepare_tables",
    table_bundle = table_contract$table_bundle,
    dataset_spec = dataset_spec
  )
  c(
    table_contract,
    list(
      dataset_spec = dataset_spec,
      data_seed = .ndm_multidisease_data_seed(row_values),
      n_inference = as.integer(dataset_spec$n_inference_samples),
      inference_support = prepared$inference_support %||% NULL
    )
  )
}

.ndm_multidisease_expected_producer <- function(runtime_env) {
  producer <- .ndm_runtime_get0(
    runtime_env,
    "ndm_tfrecord_producer",
    ifnotfound = NULL
  )
  if (!is.null(producer)) {
    return(producer)
  }

  contract <- trimws(Sys.getenv(
    "NDM_TFRECORD_PRODUCER_CONTRACT",
    unset = ""
  ))
  if (!nzchar(contract)) {
    stop(
      "Canonical multidisease training requires expected producer metadata. ",
      "Set `ndm_tfrecord_producer` or ",
      "`NDM_TFRECORD_PRODUCER_CONTRACT` to match the bootstrap producer.",
      call. = FALSE
    )
  }
  list(contract = contract)
}

.ndm_preflight_multidisease_tfrecords <- function(
    runtime_env,
    row_values,
    bundle,
    data_subset = "all",
    lookahead = 12L,
    min_anchoring_time = 4L,
    paths = NULL,
    verify_checksum = TRUE) {
  contract <- .ndm_multidisease_artifact_contract(
    row_values = row_values,
    bundle = bundle,
    data_subset = data_subset,
    lookahead = lookahead,
    min_anchoring_time = min_anchoring_time
  )
  ndm_set_runtime_globals(
    runtime_env,
    list(
      ndm_real_table_bundle = contract$table_bundle,
      ndm_real_source_sha256 = contract$source_sha256,
      ndm_inference_support = contract$inference_support,
      ndm_canonical_verify_checksum = isTRUE(verify_checksum)
    )
  )
  .ndm_preflight_canonical_real_runtime(
    runtime_env = runtime_env,
    paths = paths,
    dataset_spec = contract$dataset_spec,
    base_id = .ndm_multidisease_row_integer(
      row_values,
      "BaseID",
      required = TRUE
    ),
    n_train = .ndm_multidisease_row_integer(
      row_values,
      "nSamplesTrain",
      required = TRUE
    ),
    n_inference = contract$n_inference,
    source_sha256 = contract$source_sha256,
    expected_seed = contract$data_seed,
    expected_producer = .ndm_multidisease_expected_producer(runtime_env),
    expected_inference_support = contract$inference_support,
    verify_checksum = isTRUE(verify_checksum),
    bootstrap = "ndm_bootstrap_multidisease_tfrecords()"
  )
  invisible(contract)
}

.ndm_multidisease_plan <- function(grid,
                                   base_ids = NULL,
                                   tfrecord_dir) {
  if (!is.data.frame(grid)) {
    stop("`grid` must be a data.frame.", call. = FALSE)
  }
  required <- c(
    "BaseID", "ContextLength", "evaluationTime", "initialTransform",
    "initialNormType", "paddingMethod", "OSSType", "dataInputs",
    "nSamplesTrain", "nObsInference"
  )
  missing <- setdiff(required, names(grid))
  if (length(missing) > 0L) {
    stop(
      "Multidisease grid is missing required field(s): ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }
  grid_base_ids <- suppressWarnings(as.integer(grid$BaseID))
  if (length(grid_base_ids) != nrow(grid) || anyNA(grid_base_ids)) {
    stop("Every multidisease grid row must contain an integer `BaseID`.", call. = FALSE)
  }
  ordered_base_ids <- unique(grid_base_ids)
  if (!is.null(base_ids)) {
    requested <- unique(suppressWarnings(as.integer(base_ids)))
    if (length(requested) == 0L || anyNA(requested)) {
      stop("`base_ids` must contain at least one integer BaseID.", call. = FALSE)
    }
    missing_ids <- setdiff(requested, ordered_base_ids)
    if (length(missing_ids) > 0L) {
      stop(
        "Requested BaseID(s) are missing from the grid: ",
        paste(missing_ids, collapse = ", "),
        ".",
        call. = FALSE
      )
    }
    ordered_base_ids <- requested
  }
  defining_fields <- intersect(
    c(
      "ContextLength", "evaluationTime", "evaluationOriginTimeID",
      "evaluationHorizon", "inferenceSampling", "dataSeed",
      "initialTransform", "initialNormType", "paddingMethod", "OSSType",
      "dataInputs", "nObsInference", "DiseaseName"
    ),
    names(grid)
  )
  rows <- lapply(ordered_base_ids, function(base_id) {
    row_indices <- which(grid_base_ids == base_id)
    group <- grid[row_indices, , drop = FALSE]
    inconsistent <- defining_fields[vapply(
      group[defining_fields],
      function(value) length(unique(as.character(value))) != 1L,
      logical(1L)
    )]
    if (length(inconsistent) > 0L) {
      stop(
        "Rows sharing BaseID ", base_id,
        " disagree on dataset-defining field(s): ",
        paste(inconsistent, collapse = ", "),
        ".",
        call. = FALSE
      )
    }
    capacity <- suppressWarnings(as.numeric(group$nSamplesTrain))
    if (any(!is.finite(capacity)) || any(capacity < 1) ||
        any(capacity != floor(capacity))) {
      stop(
        "Rows sharing BaseID ", base_id,
        " contain invalid `nSamplesTrain` values.",
        call. = FALSE
      )
    }
    max_capacity <- max(capacity)
    canonical_row <- min(row_indices[capacity == max_capacity])
    paths <- .ndm_canonical_tfrecord_paths(tfrecord_dir, base_id)
    data.frame(
      BaseID = as.integer(base_id),
      selected_rows = paste(row_indices, collapse = ","),
      canonical_row = as.integer(canonical_row),
      artifact_n_samples_train = as.integer(max_capacity),
      train_file = paths$train_file,
      inference_file = paths$inference_file,
      stringsAsFactors = FALSE
    )
  })
  if (length(rows) == 0L) {
    return(data.frame(
      BaseID = integer(),
      selected_rows = character(),
      canonical_row = integer(),
      artifact_n_samples_train = integer(),
      train_file = character(),
      inference_file = character(),
      stringsAsFactors = FALSE
    ))
  }
  do.call(rbind, rows)
}

#' Bootstrap canonical multidisease TFRecords by BaseID
#'
#' Builds one serial, canonical real-schema TFRecord pair per multidisease
#' `BaseID`. The bootstrap and training paths share the same raw-data loader,
#' table adapter, dataset specification, source fingerprint, data seed, and
#' complete-location support contract.
#'
#' @param project_root Project root containing multidisease data and grids.
#' @param analysis_name Analysis label used for default paths.
#' @param grid Optional in-memory multidisease grid.
#' @param grid_file Optional multidisease grid CSV path.
#' @param base_ids Optional BaseIDs to build, in requested order.
#' @param tfrecord_dir Canonical TFRecord output directory.
#' @param disease_names Non-empty disease names passed to the multidisease
#'   loader.
#' @param data_format Multidisease source format: `"IHME"`, `"WHO"`, or
#'   `"Tycho"`.
#' @param outcome_metric Outcome column name.
#' @param data_subset Multidisease location subset.
#' @param lookahead Tensor lookahead. This remains distinct from a grid row's
#'   shorter `evaluationHorizon`.
#' @param min_anchoring_time Minimum training anchor time.
#' @param producer Non-empty named producer metadata. Required to publish.
#'   For environment-bound training, use `list(contract = "<contract>")`.
#' @param overwrite Whether to replace an existing complete pair.
#' @param dry_run Whether to return the plan without loading data or writing.
#'
#' @returns A data frame with one row per BaseID. `status` is `"planned"` for
#'   dry runs, `"written"` after publication, or `"skipped_existing"` only
#'   after an existing pair passes full contract and readability validation.
#' @export
ndm_bootstrap_multidisease_tfrecords <- function(
    project_root = getwd(),
    analysis_name = "RealLatest",
    grid = NULL,
    grid_file = file.path(
      "Data", "RunGrids", "RealGrids",
      sprintf("RealGrid_%s.csv", analysis_name)
    ),
    base_ids = NULL,
    tfrecord_dir = file.path(
      "Data", "RunTFRecords", "RealTFRecords", analysis_name
    ),
    disease_names,
    data_format = "IHME",
    outcome_metric = "CountValue",
    data_subset = "all",
    lookahead = 12L,
    min_anchoring_time = 4L,
    producer = NULL,
    overwrite = FALSE,
    dry_run = FALSE) {
  project_root <- .ndm_normalize_path(project_root, must_work = TRUE)
  if (missing(disease_names)) {
    stop("`disease_names` must be supplied.", call. = FALSE)
  }
  disease_names <- unique(as.character(disease_names))
  disease_names <- disease_names[nzchar(trimws(disease_names))]
  if (length(disease_names) == 0L) {
    stop("`disease_names` must contain at least one non-empty value.", call. = FALSE)
  }
  lookahead <- .ndm_canonical_optional_count(lookahead, "lookahead")
  min_anchoring_time <- .ndm_canonical_optional_count(
    min_anchoring_time,
    "min_anchoring_time"
  )
  if (is.null(lookahead) || lookahead < 1L ||
      is.null(min_anchoring_time) || min_anchoring_time < 0L) {
    stop("Lookahead must be positive and minimum anchoring time non-negative.", call. = FALSE)
  }
  if (!isTRUE(dry_run) && is.null(producer)) {
    stop("`producer` is required when publishing canonical TFRecords.", call. = FALSE)
  }
  if (!is.null(grid) && !is.data.frame(grid)) {
    stop("`grid` must be a data.frame when supplied.", call. = FALSE)
  }
  if (!is.null(grid)) {
    grid_file <- NULL
  } else {
    grid_file <- .ndm_normalize_path(
      .ndm_path_join_if_relative(project_root, grid_file),
      must_work = TRUE
    )
    grid <- as.data.frame(
      data.table::fread(grid_file),
      stringsAsFactors = FALSE
    )
  }
  tfrecord_dir <- .ndm_normalize_path(
    .ndm_path_join_if_relative(project_root, tfrecord_dir),
    must_work = FALSE
  )
  plan <- .ndm_multidisease_plan(
    grid = grid,
    base_ids = base_ids,
    tfrecord_dir = tfrecord_dir
  )
  if (nrow(plan) == 0L) {
    plan$status <- character()
    return(plan)
  }
  if (isTRUE(dry_run)) {
    plan$status <- vapply(seq_len(nrow(plan)), function(i) {
      files <- c(
        plan$train_file[[i]],
        plan$inference_file[[i]],
        .ndm_canonical_manifest_path(plan$train_file[[i]]),
        .ndm_canonical_manifest_path(plan$inference_file[[i]])
      )
      present <- file.exists(files)
      if (!isTRUE(overwrite) && any(present) && !all(present)) {
        stop(
          "Found incomplete TFRecord artifacts for BaseID ",
          plan$BaseID[[i]], ". Set `overwrite = TRUE` to replace them.",
          call. = FALSE
        )
      }
      "planned"
    }, character(1L))
    return(plan)
  }

  dir.create(tfrecord_dir, recursive = TRUE, showWarnings = FALSE)
  bundle <- .ndm_prepare_multidisease_bundle(
    project_root = project_root,
    data_format = data_format,
    disease_names = disease_names,
    outcome_metric = outcome_metric,
    data_subset = data_subset
  )
  tensorflow <- NULL
  statuses <- character(nrow(plan))
  for (i in seq_len(nrow(plan))) {
    row_values <- grid[plan$canonical_row[[i]], , drop = FALSE]
    contract <- .ndm_multidisease_artifact_contract(
      row_values = row_values,
      bundle = bundle,
      data_subset = data_subset,
      lookahead = lookahead,
      min_anchoring_time = min_anchoring_time
    )
    training_spec <- .ndm_canonical_dataset_call(
      "ndm_datasets_training_spec",
      n_samples_train = plan$artifact_n_samples_train[[i]]
    )
    if (is.null(tensorflow)) {
      tensorflow <- .ndm_resolve_tensorflow()
    }
    result <- .ndm_canonical_dataset_call(
      "ndm_real_bootstrap_tfrecords",
      table_bundle = contract$table_bundle,
      dataset_spec = contract$dataset_spec,
      output_dir = tfrecord_dir,
      training_spec = training_spec,
      producer = producer,
      batch_size = 64L,
      seed = contract$data_seed,
      overwrite = isTRUE(overwrite),
      verify_readable = TRUE,
      tensorflow = tensorflow,
      quiet = TRUE
    )
    statuses[[i]] <- as.character(result$status %||% "written")
  }
  plan$status <- statuses
  plan
}

.ndm_multidisease_required_globals_missing <- function(env) {
  required <- .ndm_multidisease_required_runtime_globals()
  required[!vapply(required, exists, logical(1), envir = env, inherits = FALSE)]
}

.ndm_multidisease_reject_retired_tfrecord_regeneration <- function(runtime_env) {
  if (isTRUE(.ndm_runtime_get0(runtime_env, "resave_tfrecords", ifnotfound = FALSE)) ||
      isTRUE(.ndm_runtime_get0(runtime_env, "ReSaveTfRecords", ifnotfound = FALSE))) {
    stop(
      "`resave_tfrecords = TRUE` is no longer supported for multidisease workflows. ",
      "Use `ndm_bootstrap_multidisease_tfrecords()` before training.",
      call. = FALSE
    )
  }

  invisible(runtime_env)
}

.ndm_multidisease_set_default_globals <- function(runtime_env, bundle) {
  project_root <- .ndm_normalize_path(
    .ndm_runtime_get0(runtime_env, "project_root", ifnotfound = getwd()),
    must_work = TRUE
  )
  analysis_name <- as.character(.ndm_runtime_get0(runtime_env, "AnalysisName", ifnotfound = "RealLatest"))
  tfrecord_dir <- .ndm_runtime_get0(
    runtime_env,
    "TfRecordDir",
    ifnotfound = file.path(project_root, "Data", "RunTFRecords", "RealTFRecords", analysis_name)
  )
  holder_folder <- .ndm_runtime_get0(
    runtime_env,
    "HolderFolder",
    ifnotfound = file.path(project_root, "SavedResults", "Real", sprintf("Results_%s", analysis_name))
  )

  dir.create(tfrecord_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(holder_folder, recursive = TRUE, showWarnings = FALSE)

  max_times_past <- as.integer(.ndm_runtime_get0(runtime_env, "ContextLength"))
  n_times_lookahead <- as.integer(.ndm_runtime_get0(runtime_env, "nTimesLookahead", ifnotfound = 12L))
  n_batch <- as.integer(.ndm_runtime_get0(runtime_env, "nBatch", ifnotfound = 32L))
  n_checkpoints_default <- as.integer(.ndm_runtime_get0(runtime_env, "nCheckpoints", ifnotfound = 10L))
  n_epoches_max <- as.integer(.ndm_runtime_get0(runtime_env, "nEpochesMax", ifnotfound = 9L))
  n_samples_train <- as.integer(.ndm_runtime_get0(runtime_env, "nSamplesTrain"))
  n_time_steps_sim <- as.integer((n_times_lookahead + abs(max_times_past)) * 2L)
  nsgd_calibration <- .ndm_resolve_nsgd_calibration(
    mode = "multidisease",
    project_root = project_root,
    analysis_name = analysis_name,
    n_epoches_max = n_epoches_max,
    grid = .ndm_runtime_get0(runtime_env, "grid", ifnotfound = NULL),
    grid_file = .ndm_runtime_get0(runtime_env, "grid_file", ifnotfound = NULL),
    fallback_n_samples_train = .ndm_runtime_get0(
      runtime_env,
      "nSamples_max",
      ifnotfound = n_samples_train
    )
  )
  n_sgd <- as.integer(nsgd_calibration$resolved_n_sgd)
  n_samples_max <- as.integer(nsgd_calibration$anchor_max_n_samples_train)
  n_checkpoints <- .ndm_small_run_n_checkpoints(
    n_samples_max,
    n_sgd,
    n_checkpoints_default
  )
  max_time_index <- max(bundle$truth_df_red$time_id, na.rm = TRUE)
  jnp <- runtime_env$jnp
  diffrax <- runtime_env$diffrax
  runtime_array <- function(x) {
    if (identical(runtime_env$backend$default_backend, "cpu")) {
      return(jnp$array(x))
    }
    if (isTRUE(.ndm_runtime_get0(runtime_env, "force2GPU", ifnotfound = FALSE)) ||
        isTRUE(.ndm_runtime_get0(runtime_env, "force_to_gpu", ifnotfound = FALSE))) {
      return(runtime_env$send2gpu(x))
    }
    runtime_env$send2cpu(x)
  }
  decoder_in_neural_ode <- isTRUE(.ndm_runtime_get0(runtime_env, "DecoderInNeuralODE", ifnotfound = FALSE))

  selected_inputs <- .ndm_multidisease_resolve_inputs(
    bundle,
    .ndm_runtime_get0(runtime_env, "dataInputs", ifnotfound = "all")
  )

  base_id <- as.integer(.ndm_runtime_get0(runtime_env, "BaseID", ifnotfound = 1L))
  n_obs_inference <- as.integer(.ndm_runtime_get0(
    runtime_env,
    "nObsInference",
    ifnotfound = 32L
  ))
  real_entry <- data.frame(
    ContextLength = max_times_past,
    evaluationMethod = as.character(.ndm_runtime_get0(runtime_env, "evaluationMethod", ifnotfound = "prospective")),
    initialTransform = as.character(.ndm_runtime_get0(runtime_env, "initialTransform")),
    initialNormType = as.character(.ndm_runtime_get0(runtime_env, "initialNormType")),
    paddingMethod = as.character(.ndm_runtime_get0(runtime_env, "paddingMethod")),
    floatType = as.character(.ndm_runtime_get0(runtime_env, "floatType", ifnotfound = runtime_env$float_type %||% "32")),
    evaluationTime = as.integer(.ndm_runtime_get0(runtime_env, "evaluationTime")),
    evaluationOriginTimeID = as.integer(.ndm_runtime_get0(
      runtime_env,
      "evaluationOriginTimeID",
      ifnotfound = NA_integer_
    )),
    evaluationHorizon = as.integer(.ndm_runtime_get0(
      runtime_env,
      "evaluationHorizon",
      ifnotfound = NA_integer_
    )),
    inferenceSampling = as.character(.ndm_runtime_get0(
      runtime_env,
      "inferenceSampling",
      ifnotfound = "random"
    )),
    dataInputs = paste(selected_inputs, collapse = "__"),
    OSSType = as.character(.ndm_runtime_get0(runtime_env, "OSSType")),
    simplexType = as.integer(.ndm_runtime_get0(runtime_env, "simplexType", ifnotfound = 1L)),
    BaseID = base_id,
    DiseaseName = paste(bundle$resolved_diseases, collapse = "__"),
    ModelType = as.character(.ndm_runtime_get0(runtime_env, "ModelType", ifnotfound = runtime_env$model_type %||% "DecoderOnly")),
    ModelDepth = as.integer(.ndm_runtime_get0(runtime_env, "ModelDepth", ifnotfound = NA_integer_)),
    ModelDims = as.integer(.ndm_runtime_get0(runtime_env, "ModelDims", ifnotfound = NA_integer_)),
    nSamplesTrain = n_samples_train,
    nObsInference = n_obs_inference,
    dataSeed = as.integer(.ndm_runtime_get0(runtime_env, "dataSeed", ifnotfound = 0L)),
    ResaveThisTFRecord = 0L,
    stringsAsFactors = FALSE
  )

  ndm_set_runtime_globals(
    runtime_env,
    c(
      list(
      project_root = project_root,
      AnalysisName = analysis_name,
      AnalysisDate = Sys.Date(),
      DiseaseNameVec = bundle$resolved_diseases,
      COMMAND_ARG_INPUT = as.character(.ndm_runtime_get0(runtime_env, "COMMAND_ARG_INPUT", ifnotfound = "ndm")),
      ReSaveTfRecords = FALSE,
      force2GPU = isTRUE(runtime_env$force_to_gpu),
      GPU_MEM_FRAC = runtime_env$gpu_mem_frac,
      UseShortOutcomes = TRUE,
      nTimesLookahead = n_times_lookahead,
      nTimesLookValidationInference = n_times_lookahead,
      lookahead = n_times_lookahead,
      OverDoDataFrac = 0.90,
      DecoderInNeuralODE = decoder_in_neural_ode,
      endAppend = FALSE,
      useLSTM = FALSE,
      doGrid = TRUE,
      SimMode = FALSE,
      nBatch = n_batch,
      nCheckpoints = n_checkpoints,
      nEpochesMax = n_epoches_max,
      nSamples_max = n_samples_max,
      nSGD_pretrain = 0L,
      nSGD_DefiningLRSeq = n_sgd,
      nSGD_model = n_sgd,
      nSGD_posttrain = n_sgd,
      PreTrain = FALSE,
      specificOptState = TRUE,
      SharedListNames = c("TS"),
      LEARNING_RATE_MAX = as.numeric(.ndm_runtime_get0(runtime_env, "LEARNING_RATE_MAX", ifnotfound = 0.0002)),
      LEARNING_RATE_MAX_model = as.numeric(.ndm_runtime_get0(runtime_env, "LEARNING_RATE_MAX_model", ifnotfound = 0.0002)),
      LEARNING_RATE_MAX_pretrain = as.numeric(.ndm_runtime_get0(runtime_env, "LEARNING_RATE_MAX_pretrain", ifnotfound = 1e-5)),
      HolderFolder = holder_folder,
      TfRecordDir = tfrecord_dir,
      maxTimesPast = max_times_past,
      nTimesTotal = max_times_past + n_times_lookahead,
      VI_TotalTimesInLikelihood = n_times_lookahead,
      minAnchoringTimeID = as.integer(.ndm_runtime_get0(runtime_env, "minAnchoringTimeID", ifnotfound = 4L)),
      min_anchoring_time = as.integer(.ndm_runtime_get0(runtime_env, "minAnchoringTimeID", ifnotfound = 4L)),
      MIN_NA_ACCEPT_FRAC = 4 / max_times_past,
      NTimeSteps_SIM = n_time_steps_sim,
      MaxSteps = as.integer(.ndm_runtime_get0(runtime_env, "MaxSteps", ifnotfound = 10^6)),
      nOutcomes = bundle$nOutcomes,
      nPlaces = bundle$nPlaces,
      MaxTimeIndex = max_time_index,
      AVERAGE_TRUTH = mean(bundle$truth_df_red[[bundle$outcome_metric]], na.rm = TRUE),
      VI_SaveAt_ODE = diffrax$SaveAt(ts = runtime_array(seq_len(n_times_lookahead))),
      diff_eq_solver = diffrax$Dopri8(),
      VI_diff_eq_solver = diffrax$Dopri8(),
      stepsize_controller = diffrax$PIDController(rtol = 1e-7, atol = 1e-9),
      diffraxInterpolator = diffrax$LinearInterpolation,
      VI_SaveAt_ODE_sim = diffrax$SaveAt(ts = runtime_array(0L:(n_time_steps_sim - 1L))),
      VI_SaveAt_ODE_optim = diffrax$SaveAt(ts = runtime_array(0L:(n_times_lookahead - 1L))),
      VI_diff_eq_solver_dgp = diffrax$Tsit5(),
      dt0_init = 1e-1,
      dt0_init_dgp = 1e-3,
      stepsize_controller_dgp = diffrax$PIDController(rtol = 1e-6, atol = 1e-7),
      evaluation_seq = as.integer(.ndm_runtime_get0(runtime_env, "evaluation_seq", ifnotfound = c(1L, 2L, 3L, 4L))),
      all_true_value_names = bundle$all_true_value_names,
      true_value_names = bundle$true_value_names,
      outcome_metric = bundle$outcome_metric,
      disease = paste(bundle$resolved_diseases, collapse = "__"),
      outcomes = bundle$true_value_names,
      per_capita_scaling_factor = 1,
      roll_window = 1L,
      nObsInference = n_obs_inference,
      dataSeed = as.integer(real_entry$dataSeed),
      evaluationOriginTimeID = real_entry$evaluationOriginTimeID,
      evaluationHorizon = real_entry$evaluationHorizon,
      inferenceSampling = real_entry$inferenceSampling,
      dataInputs_pool_orig = bundle$dataInputs_colnames_past,
      dataInputs_pool = selected_inputs,
      dataInputs_colnames = selected_inputs,
      dataInputs_colnames_past = bundle$dataInputs_colnames_past,
      dataInputs_colnames_future = bundle$dataInputs_colnames_future,
      RealEntry = real_entry,
      RealGrid = real_entry,
      ndm_multidisease_resolved_diseases = bundle$resolved_diseases,
      ndm_data_generator = "multidisease"
      ),
      .ndm_nsgd_calibration_globals(
        nsgd_calibration
      )
    )
  )
  if (!exists("neuralode_optim_controller", envir = runtime_env, inherits = FALSE) &&
      decoder_in_neural_ode) {
    assign("neuralode_optim_controller", "constant", envir = runtime_env)
  }
  .ndm_materialize_neuralode_runtime_config(runtime_env)

  invisible(runtime_env)
}

.ndm_prepare_multidisease_data <- function(runtime_env) {
  LOC2_region_name <- NULL

  .ndm_multidisease_reject_retired_tfrecord_regeneration(runtime_env)
  missing <- .ndm_multidisease_required_globals_missing(runtime_env)
  if (length(missing) > 0L) {
    stop(
      "Missing required multidisease globals: ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }

  project_root <- .ndm_normalize_path(
    .ndm_runtime_get0(runtime_env, "project_root", ifnotfound = getwd()),
    must_work = TRUE
  )
  disease_names <- .ndm_runtime_get0(runtime_env, "disease_names", ifnotfound = character())
  disease_names <- unique(as.character(disease_names))
  disease_names <- disease_names[nzchar(trimws(disease_names))]
  if (length(disease_names) == 0L) {
    stop("`disease_names` must contain at least one non-empty value.", call. = FALSE)
  }
  data_format <- as.character(.ndm_runtime_get0(runtime_env, "data_format", ifnotfound = "IHME"))
  data_subset <- as.character(.ndm_runtime_get0(runtime_env, "data_subset", ifnotfound = "all"))
  outcome_metric <- as.character(.ndm_runtime_get0(runtime_env, "outcome_metric", ifnotfound = "CountValue"))
  desired_measure <- .ndm_runtime_get0(runtime_env, "desired_measure", ifnotfound = NULL)

  bundle <- .ndm_prepare_multidisease_bundle(
    project_root = project_root,
    data_format = data_format,
    disease_names = disease_names,
    outcome_metric = outcome_metric,
    data_subset = data_subset,
    desired_measure = desired_measure
  )

  truth_df_red <- data.table::as.data.table(bundle$truth_df_red)
  if ("POP_population" %in% names(truth_df_red)) {
    truth_df_red$Pop <- truth_df_red$POP_population
  } else {
    truth_df_red$Pop <- NA_real_
    warning("POP_population column not available; Pop set to NA")
  }
  evaluation_seq <- as.integer(.ndm_runtime_get0(runtime_env, "evaluation_seq", ifnotfound = c(1L, 2L, 3L, 4L)))
  evaluation_time <- as.integer(.ndm_runtime_get0(runtime_env, "evaluationTime"))
  evaluation_origin_time_id <- .ndm_runtime_get0(
    runtime_env,
    "evaluationOriginTimeID",
    ifnotfound = NULL
  )
  time_split <- .ndm_multidisease_time_split(
    time_ids = truth_df_red$time_id,
    evaluation_time = evaluation_time,
    evaluation_seq = evaluation_seq,
    evaluation_origin_time_id = evaluation_origin_time_id
  )
  times_in <- time_split$times_in
  times_out <- time_split$times_out

  ndm_set_runtime_globals(
    runtime_env,
    list(
      truth_df_red = as.data.frame(truth_df_red, stringsAsFactors = FALSE),
      input_df_red = as.data.frame(truth_df_red, stringsAsFactors = FALSE),
      times_in = times_in,
      times_out = times_out,
      data_subset = data_subset,
      data_format = data_format
    )
  )
  .ndm_multidisease_set_default_globals(runtime_env, bundle = bundle)
  skip_tfrecords <- isTRUE(.ndm_runtime_get0(
    runtime_env,
    "SkipTfRecords",
    ifnotfound = FALSE
  ))
  if (!skip_tfrecords) {
    real_entry <- .ndm_runtime_get0(runtime_env, "RealEntry")
    .ndm_preflight_multidisease_tfrecords(
      runtime_env = runtime_env,
      row_values = real_entry,
      bundle = bundle,
      data_subset = data_subset,
      lookahead = .ndm_runtime_get0(
        runtime_env,
        "nTimesLookahead",
        ifnotfound = 12L
      ),
      min_anchoring_time = .ndm_runtime_get0(
        runtime_env,
        "minAnchoringTimeID",
        ifnotfound = 4L
      ),
      verify_checksum = isTRUE(.ndm_runtime_get0(
        runtime_env,
        "ndm_canonical_verify_checksum",
        ifnotfound = TRUE
      ))
    )
  }
  ndm_source_runtime_data(
    env = runtime_env,
    generator = "real"
  )

  invisible(runtime_env)
}

.ndm_multidisease_time_split <- function(time_ids,
                                         evaluation_time,
                                         evaluation_seq,
                                         evaluation_origin_time_id = NULL) {
  available_times <- sort(unique(as.integer(time_ids)))
  if (length(available_times) == 0L || anyNA(available_times)) {
    stop("`time_ids` must contain at least one finite integer value.", call. = FALSE)
  }

  if (!is.null(evaluation_origin_time_id)) {
    evaluation_origin_time_id <- as.integer(evaluation_origin_time_id)
    if (length(evaluation_origin_time_id) != 1L ||
        is.na(evaluation_origin_time_id) ||
        evaluation_origin_time_id < 1L ||
        !evaluation_origin_time_id %in% available_times) {
      stop(
        "`evaluation_origin_time_id` must be one available non-initial time ID.",
        call. = FALSE
      )
    }
    in_out_cutpoint <- evaluation_origin_time_id - 1L
  } else {
    evaluation_time <- as.integer(evaluation_time)
    evaluation_seq <- as.integer(evaluation_seq)
    if (length(evaluation_time) != 1L || is.na(evaluation_time) ||
        length(evaluation_seq) == 0L || anyNA(evaluation_seq)) {
      stop("Legacy evaluation fields must contain finite integer values.", call. = FALSE)
    }
    in_out_cutpoint <- round(stats::quantile(
      available_times,
      probs = evaluation_time / (max(evaluation_seq) + 1L),
      names = FALSE
    ))
  }

  times_out <- as.integer(in_out_cutpoint + 1L)
  times_out <- times_out[times_out <= max(available_times)]
  if (length(times_out) != 1L) {
    stop("The requested evaluation split has no available forecast origin.", call. = FALSE)
  }

  list(
    in_out_cutpoint = as.integer(in_out_cutpoint),
    times_in = seq.int(0L, as.integer(in_out_cutpoint)),
    times_out = times_out,
    explicit_origin = !is.null(evaluation_origin_time_id)
  )
}

.ndm_runtime_project_root <- function(env) {
  project_root <- .ndm_runtime_get0(env, "project_root", ifnotfound = NULL)
  if (is.null(project_root) || !nzchar(project_root)) {
    return(NULL)
  }

  .ndm_normalize_path(project_root, must_work = TRUE)
}

.ndm_prepare_train_environment <- function(runtime_env) {
  if (identical(.ndm_runtime_get0(runtime_env, "ndm_data_generator", ifnotfound = NULL), "multidisease") &&
      !exists("batch_l_cal", envir = runtime_env, inherits = FALSE)) {
    project_root <- .ndm_runtime_project_root(runtime_env)
    run_calibration <- function() {
      ndm_source_runtime_calibration(
        env = runtime_env
      )
    }
    if (is.null(project_root)) {
      run_calibration()
    } else {
      .ndm_with_working_directory(project_root, run_calibration)
    }
  }

  invisible(runtime_env)
}
