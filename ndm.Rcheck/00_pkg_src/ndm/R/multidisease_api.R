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
    .(
      CountValue = sum(CountValue, na.rm = TRUE),
      POP_population = max(POP_population, na.rm = TRUE)
    ),
    by = .(location_id, location_id_numeric, location_name, time_id, targetTime_id)
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
    .(CountValue = sum(CountValue, na.rm = TRUE)),
    by = .(location_id, location_id_numeric, location_name, time_id, targetTime_id)
  ]
  if (!identical(outcome_metric, "CountValue")) {
    data.table::setnames(truth_df_red, "CountValue", outcome_metric)
  }
  truth_df_red[, Covariate1 := get(outcome_metric)]
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
  ihme_dt <- ihme_dt[, .SD[1L], by = .(location_id, location_name, cause_name, year)]
  ihme_dt <- ihme_dt[
    ,
    .(CountValue = sum(CountFraction_tmp, na.rm = TRUE)),
    by = .(location_id, location_name, year)
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
    .(CountValue = sum(CountValue, na.rm = TRUE)),
    by = .(location_id, location_id_numeric, location_name, time_id, targetTime_id)
  ]
  if (!identical(outcome_metric, "CountValue")) {
    data.table::setnames(truth_df_red, "CountValue", outcome_metric)
  }
  truth_df_red[, Covariate1 := get(outcome_metric)]
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

.ndm_multidisease_required_globals_missing <- function(env) {
  required <- .ndm_multidisease_required_runtime_globals()
  required[!vapply(required, exists, logical(1), envir = env, inherits = FALSE)]
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
  n_checkpoints <- as.integer(.ndm_runtime_get0(runtime_env, "nCheckpoints", ifnotfound = 10L))
  n_epoches_max <- as.integer(.ndm_runtime_get0(runtime_env, "nEpochesMax", ifnotfound = 9L))
  n_samples_train <- as.integer(.ndm_runtime_get0(runtime_env, "nSamplesTrain"))
  n_time_steps_sim <- as.integer((n_times_lookahead + abs(max_times_past)) * 2L)
  n_sgd <- as.integer(round(n_epoches_max * (n_samples_train / n_batch)))
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

  data_inputs <- .ndm_runtime_get0(runtime_env, "dataInputs", ifnotfound = NULL)
  if (is.null(data_inputs) || !nzchar(as.character(data_inputs)) || identical(as.character(data_inputs), "all")) {
    selected_inputs <- setdiff(bundle$dataInputs_colnames_past, bundle$outcome_metric)
    if (length(selected_inputs) == 0L) {
      selected_inputs <- bundle$dataInputs_colnames_past
    }
  } else {
    selected_inputs <- unique(strsplit(as.character(data_inputs), split = "__", fixed = TRUE)[[1L]])
  }
  missing_inputs <- selected_inputs[!selected_inputs %in% names(bundle$truth_df_red)]
  if (length(missing_inputs) > 0L) {
    stop(
      "Requested multidisease `dataInputs` are not present in the loaded data: ",
      paste(missing_inputs, collapse = ", "),
      call. = FALSE
    )
  }

  base_id <- as.integer(.ndm_runtime_get0(runtime_env, "BaseID", ifnotfound = 1L))
  real_entry <- data.frame(
    ContextLength = max_times_past,
    evaluationMethod = as.character(.ndm_runtime_get0(runtime_env, "evaluationMethod", ifnotfound = "prospective")),
    initialTransform = as.character(.ndm_runtime_get0(runtime_env, "initialTransform")),
    initialNormType = as.character(.ndm_runtime_get0(runtime_env, "initialNormType")),
    paddingMethod = as.character(.ndm_runtime_get0(runtime_env, "paddingMethod")),
    floatType = as.character(.ndm_runtime_get0(runtime_env, "floatType", ifnotfound = runtime_env$float_type %||% "32")),
    evaluationTime = as.integer(.ndm_runtime_get0(runtime_env, "evaluationTime")),
    dataInputs = paste(selected_inputs, collapse = "__"),
    OSSType = as.character(.ndm_runtime_get0(runtime_env, "OSSType")),
    simplexType = as.integer(.ndm_runtime_get0(runtime_env, "simplexType", ifnotfound = 1L)),
    BaseID = base_id,
    DiseaseName = paste(bundle$resolved_diseases, collapse = "__"),
    ModelType = as.character(.ndm_runtime_get0(runtime_env, "ModelType", ifnotfound = runtime_env$model_type %||% "DecoderOnly")),
    ModelDepth = as.integer(.ndm_runtime_get0(runtime_env, "ModelDepth", ifnotfound = NA_integer_)),
    ModelDims = as.integer(.ndm_runtime_get0(runtime_env, "ModelDims", ifnotfound = NA_integer_)),
    nSamplesTrain = n_samples_train,
    ResaveThisTFRecord = as.integer(.ndm_runtime_get0(runtime_env, "ResaveThisTFRecord", ifnotfound = 1L)),
    stringsAsFactors = FALSE
  )

  ndm_set_runtime_globals(
    runtime_env,
    list(
      project_root = project_root,
      AnalysisName = analysis_name,
      AnalysisDate = Sys.Date(),
      DiseaseNameVec = bundle$resolved_diseases,
      COMMAND_ARG_INPUT = as.character(.ndm_runtime_get0(runtime_env, "COMMAND_ARG_INPUT", ifnotfound = "ndm")),
      ReSaveTfRecords = isTRUE(runtime_env$resave_tfrecords),
      force2GPU = isTRUE(runtime_env$force_to_gpu),
      GPU_MEM_FRAC = runtime_env$gpu_mem_frac,
      UseShortOutcomes = TRUE,
      nTimesLookahead = n_times_lookahead,
      nTimesLookValidationInference = n_times_lookahead,
      OverDoDataFrac = 0.90,
      DecoderInNeuralODE = decoder_in_neural_ode,
      endAppend = FALSE,
      useLSTM = FALSE,
      doGrid = TRUE,
      SimMode = FALSE,
      nBatch = n_batch,
      nCheckpoints = n_checkpoints,
      nEpochesMax = n_epoches_max,
      nSamples_max = as.integer(.ndm_runtime_get0(runtime_env, "nSamples_max", ifnotfound = 20000L)),
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
      VI_diff_eq_solver_optim = diffrax$Tsit5(),
      VI_diff_eq_solver_dgp = diffrax$Tsit5(),
      dt0_init = 1e-1,
      dt0_init_dgp = 1e-3,
      dt0_init_optim = 1e-3,
      stepsize_controller_dgp = diffrax$PIDController(rtol = 1e-6, atol = 1e-7),
      stepsize_controller_optim = if (decoder_in_neural_ode) {
        diffrax$ConstantStepSize()
      } else {
        diffrax$PIDController(rtol = 1e-5, atol = 1e-7)
      },
      evaluation_seq = as.integer(.ndm_runtime_get0(runtime_env, "evaluation_seq", ifnotfound = c(1L, 2L, 3L, 4L))),
      all_true_value_names = bundle$all_true_value_names,
      true_value_names = bundle$true_value_names,
      outcome_metric = bundle$outcome_metric,
      dataInputs_pool_orig = bundle$dataInputs_colnames_past,
      dataInputs_pool = selected_inputs,
      dataInputs_colnames = selected_inputs,
      dataInputs_colnames_past = bundle$dataInputs_colnames_past,
      dataInputs_colnames_future = bundle$dataInputs_colnames_future,
      RealEntry = real_entry,
      RealGrid = real_entry,
      ndm_multidisease_resolved_diseases = bundle$resolved_diseases,
      ndm_data_generator = "multidisease"
    )
  )

  invisible(runtime_env)
}

.ndm_prepare_multidisease_data <- function(runtime_env) {
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

  bundle <- .ndm_load_multidisease_bundle(
    project_root = project_root,
    data_format = data_format,
    disease_names = disease_names,
    outcome_metric = outcome_metric,
    desired_measure = desired_measure
  )

  truth_df_red <- data.table::as.data.table(bundle$truth_df_red)
  if (identical(data_subset, "all")) {
    keep_place_ids <- unique(truth_df_red$location_id)
  } else if (identical(data_subset, "high_income")) {
    if (!"LOC2_region_name" %in% names(truth_df_red)) {
      warning("high_income subset requested but LOC2_region_name not available; using all locations")
      keep_place_ids <- unique(truth_df_red$location_id)
    } else {
      keep_place_ids <- unique(truth_df_red[
        LOC2_region_name %in% c(
          "Western Europe",
          "High-income North America",
          "Central Europe",
          "High-income Asia Pacific"
        )
      ]$location_id)
    }
  } else {
    stop("Unsupported multidisease `data_subset`: ", data_subset, call. = FALSE)
  }

  truth_df_red <- truth_df_red[location_id %in% keep_place_ids]
  if ("POP_population" %in% names(truth_df_red)) {
    truth_df_red[, Pop := POP_population]
  } else {
    truth_df_red[, Pop := NA_real_]
    warning("POP_population column not available; Pop set to NA")
  }
  truth_df_red <- truth_df_red[!is.na(location_name)]
  truth_df_red[, location_id_numeric := as.integer(factor(location_id, levels = sort(unique(location_id)))) - 1L]
  if (!any(truth_df_red$time_id == 0L)) {
    stop("time_id seems to be non-zero indexed!", call. = FALSE)
  }

  evaluation_seq <- as.integer(.ndm_runtime_get0(runtime_env, "evaluation_seq", ifnotfound = c(1L, 2L, 3L, 4L)))
  evaluation_time <- as.integer(.ndm_runtime_get0(runtime_env, "evaluationTime"))
  in_out_cutpoint <- round(stats::quantile(
    sort(unique(truth_df_red$time_id)),
    probs = evaluation_time / (max(evaluation_seq) + 1L),
    names = FALSE
  ))
  times_out <- in_out_cutpoint + 1L
  times_out <- times_out[times_out <= max(truth_df_red$time_id)]
  times_in <- 0L:in_out_cutpoint

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
  ndm_source_runtime_data(
    analysis_root = .ndm_internal_analysis_root(),
    env = runtime_env,
    generator = "real"
  )

  invisible(runtime_env)
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
        analysis_root = .ndm_internal_analysis_root(),
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
