.ndm_canonical_dataset_call <- function(name, ...) {
  fun <- getExportedValue("ndmdatasets", name)
  do.call(fun, list(...))
}

.ndm_canonical_tfrecord_paths <- function(tfrecord_dir, base_id) {
  if (length(tfrecord_dir) != 1L || is.na(tfrecord_dir) || !nzchar(tfrecord_dir)) {
    stop("`tfrecord_dir` must be one non-empty path.", call. = FALSE)
  }
  base_id <- suppressWarnings(as.integer(base_id))
  if (length(base_id) != 1L || is.na(base_id)) {
    stop("`base_id` must be one integer value.", call. = FALSE)
  }

  list(
    train_file = file.path(tfrecord_dir, sprintf("train_%s.tfrecord", base_id)),
    inference_file = file.path(tfrecord_dir, sprintf("inference_%s.tfrecord", base_id))
  )
}

.ndm_canonical_manifest_path <- function(file) {
  paste0(file, ".manifest.rds")
}

.ndm_canonical_optional_count <- function(x, name) {
  if (is.null(x)) {
    return(NULL)
  }
  value <- suppressWarnings(as.numeric(x))
  if (length(value) != 1L || !is.finite(value) || value < 0 ||
      value > .Machine$integer.max || value != floor(value)) {
    stop("`", name, "` must be NULL or one non-negative integer.", call. = FALSE)
  }
  as.integer(value)
}

.ndm_canonical_manifest_value <- function(x) {
  if (is.list(x)) {
    out <- lapply(x, .ndm_canonical_manifest_value)
    names(out) <- names(x)
    return(out)
  }
  if (is.factor(x)) {
    return(as.character(x))
  }
  x
}

.ndm_validate_canonical_tfrecord_pair <- function(paths,
                                                   schema_kind = c("real", "sim"),
                                                   dataset_spec,
                                                   n_train = NULL,
                                                   n_inference = NULL,
                                                   source_sha256 = NULL,
                                                   expected_seed = 0L,
                                                   expected_producer = NULL,
                                                   expected_inference_support = NULL,
                                                   verify_checksum = TRUE,
                                                   dataset_call = .ndm_canonical_dataset_call) {
  schema_kind <- match.arg(schema_kind)
  if (!is.list(paths) ||
      !all(c("train_file", "inference_file") %in% names(paths))) {
    stop("`paths` must contain `train_file` and `inference_file`.", call. = FALSE)
  }
  if (!is.function(dataset_call)) {
    stop("`dataset_call` must be a function.", call. = FALSE)
  }
  n_train <- .ndm_canonical_optional_count(n_train, "n_train")
  n_inference <- .ndm_canonical_optional_count(n_inference, "n_inference")

  train_args <- list(
    file = paths$train_file,
    schema = schema_kind,
    expected_split = "train",
    expected_seed = expected_seed,
    verify_checksum = isTRUE(verify_checksum),
    verify_readable = FALSE
  )
  if (!is.null(dataset_spec)) {
    train_args$expected_dataset_spec <- dataset_spec
  }
  if (!is.null(expected_producer)) {
    train_args$expected_producer <- expected_producer
  }
  train_manifest <- do.call(
    dataset_call,
    c(list("ndm_datasets_validate_tfrecord_artifact"), train_args)
  )
  if (!is.null(n_train) && as.numeric(train_manifest$n_examples) < n_train) {
    stop(
      "Canonical training TFRecord has ", train_manifest$n_examples,
      " examples; the selected run requires at least ", n_train, ".",
      call. = FALSE
    )
  }

  pair_dataset_spec <- dataset_spec %||% train_manifest$dataset_spec %||% NULL
  if (is.null(pair_dataset_spec)) {
    stop("Canonical training TFRecord manifest does not contain a dataset specification.", call. = FALSE)
  }
  inference_args <- list(
    file = paths$inference_file,
    schema = schema_kind,
    expected_dataset_spec = pair_dataset_spec,
    expected_split = "inference",
    expected_seed = expected_seed,
    verify_checksum = isTRUE(verify_checksum),
    verify_readable = FALSE
  )
  if (!is.null(n_inference)) {
    inference_args$expected_n_examples <- n_inference
  }
  if (!is.null(expected_producer)) {
    inference_args$expected_producer <- expected_producer
  }
  inference_manifest <- do.call(
    dataset_call,
    c(list("ndm_datasets_validate_tfrecord_artifact"), inference_args)
  )
  if (!is.null(n_inference) &&
      !identical(as.numeric(inference_manifest$n_examples), as.numeric(n_inference))) {
    stop(
      "Canonical inference TFRecord has ", inference_manifest$n_examples,
      " examples; expected ", n_inference, ".",
      call. = FALSE
    )
  }

  if (!identical(train_manifest$scaler_sha256, inference_manifest$scaler_sha256)) {
    stop("Canonical train and inference TFRecords were built with different scalers.", call. = FALSE)
  }
  if (!identical(
    train_manifest$schema$example_shape_map,
    inference_manifest$schema$example_shape_map
  )) {
    stop("Canonical train and inference TFRecords have different example shapes.", call. = FALSE)
  }
  if (!identical(train_manifest$producer_sha256, inference_manifest$producer_sha256)) {
    stop("Canonical train and inference TFRecords were built by different producers.", call. = FALSE)
  }
  train_source <- (train_manifest$metadata %||% list())$source_sha256 %||% NULL
  inference_source <- (inference_manifest$metadata %||% list())$source_sha256 %||% NULL
  if (!identical(train_source, inference_source)) {
    stop("Canonical train and inference TFRecords were built from different source tables.", call. = FALSE)
  }
  if (!is.null(source_sha256)) {
    if (!identical(train_source, source_sha256) ||
        !identical(inference_source, source_sha256)) {
      stop("Canonical real TFRecords were built from different source tables.", call. = FALSE)
    }
  }
  train_metadata <- train_manifest$metadata %||% list()
  inference_metadata <- inference_manifest$metadata %||% list()
  train_support <- train_metadata$inference_support %||% NULL
  inference_support <- inference_metadata$inference_support %||% NULL
  if (!identical(train_support, inference_support)) {
    stop("Canonical train and inference TFRecords have different inference support.", call. = FALSE)
  }
  if (!identical(
    train_metadata$inference_support_sha256 %||% NULL,
    inference_metadata$inference_support_sha256 %||% NULL
  )) {
    stop("Canonical train and inference TFRecords have different inference-support checksums.", call. = FALSE)
  }
  if (!is.null(expected_inference_support) &&
      !identical(
        train_support,
        .ndm_canonical_manifest_value(expected_inference_support)
      )) {
    stop("Canonical real TFRecords have different inference support.", call. = FALSE)
  }

  invisible(list(
    paths = paths,
    dataset_spec = pair_dataset_spec,
    train = train_manifest,
    inference = inference_manifest,
    inference_support = train_support,
    verify_checksum = isTRUE(verify_checksum)
  ))
}

.ndm_attach_canonical_tfrecords <- function(runtime_env,
                                             train_file,
                                             inference_file,
                                             schema_kind = c("real", "sim"),
                                             batch_size,
                                             shuffle_train = FALSE,
                                             reshuffle_train = shuffle_train,
                                             run_seed = NULL,
                                             verify_checksum = TRUE,
                                             max_train_examples = NULL,
                                             max_inference_examples = NULL,
                                             tensorflow = NULL,
                                             dataset_call = .ndm_canonical_dataset_call,
                                             iterator_factory = reticulate::as_iterator) {
  if (!is.environment(runtime_env)) {
    stop("`runtime_env` must be an environment.", call. = FALSE)
  }
  schema_kind <- match.arg(schema_kind)
  batch_size <- .ndm_canonical_optional_count(batch_size, "batch_size")
  if (is.null(batch_size) || batch_size < 1L) {
    stop("`batch_size` must be one positive integer.", call. = FALSE)
  }
  if (!is.function(dataset_call) || !is.function(iterator_factory)) {
    stop("Dataset and iterator factories must be functions.", call. = FALSE)
  }
  if (!is.null(run_seed)) {
    run_seed <- .ndm_canonical_optional_count(run_seed, "run_seed")
  }

  max_train_examples <- max_train_examples %||%
    get0("nSamplesTrain", envir = runtime_env, inherits = TRUE, ifnotfound = NULL)
  max_inference_examples <- max_inference_examples %||%
    get0("nObsInference", envir = runtime_env, inherits = TRUE, ifnotfound = NULL)
  max_train_examples <- .ndm_canonical_optional_count(max_train_examples, "max_train_examples")
  max_inference_examples <- .ndm_canonical_optional_count(
    max_inference_examples,
    "max_inference_examples"
  )
  tensorflow <- tensorflow %||%
    get0("tf", envir = runtime_env, inherits = TRUE, ifnotfound = NULL)
  if (is.null(tensorflow)) {
    tensorflow <- .ndm_resolve_tensorflow()
  }

  read_dataset <- function(file,
                           max_examples,
                           shuffle,
                           shuffle_seed,
                           reshuffle_each_iteration) {
    dataset_call(
      "ndm_datasets_read_tfrecord_dataset",
      file = file,
      batch_size = batch_size,
      schema = schema_kind,
      max_examples = max_examples,
      shuffle = isTRUE(shuffle),
      shuffle_seed = shuffle_seed,
      reshuffle_each_iteration = isTRUE(reshuffle_each_iteration),
      verify_checksum = isTRUE(verify_checksum),
      tensorflow = tensorflow
    )
  }

  train_dataset <- read_dataset(
    train_file,
    max_examples = max_train_examples,
    shuffle = shuffle_train,
    shuffle_seed = run_seed,
    reshuffle_each_iteration = reshuffle_train
  )
  inference_dataset <- read_dataset(
    inference_file,
    max_examples = max_inference_examples,
    shuffle = FALSE,
    shuffle_seed = NULL,
    reshuffle_each_iteration = FALSE
  )

  assign("TFDataset_train", train_dataset, envir = runtime_env)
  assign("TFDatasetIterator_train", iterator_factory(train_dataset), envir = runtime_env)
  assign("TFDataset_inference", inference_dataset, envir = runtime_env)
  assign("TFDatasetIterator_inference", iterator_factory(inference_dataset), envir = runtime_env)
  assign("ndm_canonical_tfrecord_schema", schema_kind, envir = runtime_env)

  invisible(runtime_env)
}

.ndm_runtime_tensor_shape <- function(x) {
  r_dim <- dim(x)
  if (!is.null(r_dim)) {
    return(as.integer(r_dim))
  }
  shape <- tryCatch(x$shape, error = function(e) NULL)
  if (is.null(shape)) {
    return(integer())
  }
  shape <- tryCatch(reticulate::py_to_r(shape), error = function(e) shape)
  suppressWarnings(as.integer(unlist(shape, recursive = TRUE, use.names = FALSE)))
}

.ndm_select_model_targets <- function(y,
                                      y_mask,
                                      n_outcomes,
                                      jnp = NULL) {
  n_outcomes <- suppressWarnings(as.numeric(n_outcomes))
  if (length(n_outcomes) != 1L || !is.finite(n_outcomes) ||
      n_outcomes < 1 || n_outcomes > .Machine$integer.max ||
      n_outcomes != floor(n_outcomes)) {
    stop("`n_outcomes` must be one positive integer.", call. = FALSE)
  }
  n_outcomes <- as.integer(n_outcomes)

  y_shape <- .ndm_runtime_tensor_shape(y)
  mask_shape <- .ndm_runtime_tensor_shape(y_mask)
  if (length(y_shape) < 1L) {
    stop("`y` must have at least one dimension.", call. = FALSE)
  }
  if (length(mask_shape) < 1L) {
    stop("`y_mask` must have at least one dimension.", call. = FALSE)
  }
  if (length(y_shape) != length(mask_shape)) {
    stop("`y` and `y_mask` must have the same rank.", call. = FALSE)
  }
  if (anyNA(y_shape) || anyNA(mask_shape)) {
    stop("`y` and `y_mask` must have statically known shapes.", call. = FALSE)
  }
  if (!identical(y_shape[-length(y_shape)], mask_shape[-length(mask_shape)])) {
    stop("`y` and `y_mask` must agree on all non-channel dimensions.", call. = FALSE)
  }

  y_channels <- y_shape[[length(y_shape)]]
  mask_channels <- mask_shape[[length(mask_shape)]]
  if (y_channels < n_outcomes) {
    stop(
      "`y` has ", y_channels, " outcome channel(s), but `n_outcomes` is ",
      n_outcomes, ".",
      call. = FALSE
    )
  }
  if (mask_channels < n_outcomes) {
    stop(
      "`y_mask` has ", mask_channels,
      " outcome channel(s), but `n_outcomes` is ", n_outcomes, ".",
      call. = FALSE
    )
  }

  if (is.null(jnp)) {
    if (inherits(y, "python.builtin.object") ||
        inherits(y_mask, "python.builtin.object")) {
      stop(
        "`jnp` is required when selecting channels from Python tensors.",
        call. = FALSE
      )
    }
    select_r_channels <- function(x) {
      selectors <- rep(list(TRUE), length(dim(x)))
      selectors[[length(selectors)]] <- seq_len(n_outcomes)
      do.call(`[`, c(list(x), selectors, list(drop = FALSE)))
    }
    return(list(
      y = select_r_channels(y),
      y_mask = select_r_channels(y_mask)
    ))
  }

  backend_arange <- tryCatch(jnp$arange, error = function(e) NULL)
  backend_take <- tryCatch(jnp$take, error = function(e) NULL)
  if (is.null(backend_arange) || is.null(backend_take)) {
    stop("`jnp` must provide `arange()` and `take()`.", call. = FALSE)
  }

  channel_indices <- jnp$arange(n_outcomes)
  list(
    y = jnp$take(y, channel_indices, axis = -1L),
    y_mask = jnp$take(y_mask, channel_indices, axis = -1L)
  )
}

.ndm_seed_runtime_batch <- function(runtime_env, batch_l) {
  if (!is.environment(runtime_env)) {
    stop("`runtime_env` must be an environment.", call. = FALSE)
  }
  if (is.null(batch_l) || !all(c("XPred", "YTrue") %in% names(batch_l))) {
    stop("`batch_l` must contain `XPred` and `YTrue`.", call. = FALSE)
  }

  assign("batch_l", batch_l, envir = runtime_env)
  assign("batch_l_cal", batch_l, envir = runtime_env)
  assign("jax_batchx", batch_l$XPred, envir = runtime_env)
  assign("jax_batchy", batch_l$YTrue, envir = runtime_env)
  x_shape <- .ndm_runtime_tensor_shape(batch_l$XPred)
  if (length(x_shape) >= 3L && !is.na(x_shape[[3L]])) {
    assign("nCovars_real", x_shape[[3L]], envir = runtime_env)
  }

  invisible(runtime_env)
}

.ndm_batch_to_runtime_jax <- function(x, runtime_env) {
  if (is.list(x)) {
    return(lapply(x, .ndm_batch_to_runtime_jax, runtime_env = runtime_env))
  }
  if (inherits(x, "python.builtin.object")) {
    x <- reticulate::py_to_r(x)
  }
  runtime_env$jnp$array(x)
}

.ndm_sim_outcome_sd <- function(dataset_spec,
                                scaler,
                                n_batch = 32L,
                                n_outer = 4L,
                                dataset_call = .ndm_canonical_dataset_call) {
  n_batch <- .ndm_canonical_optional_count(n_batch, "n_batch")
  n_outer <- .ndm_canonical_optional_count(n_outer, "n_outer")
  if (is.null(n_batch) || n_batch < 1L || is.null(n_outer) || n_outer < 1L) {
    stop("Simulation outcome scaling counts must be positive integers.", call. = FALSE)
  }

  values <- numeric()
  counter <- 0L
  for (i in seq_len(n_outer)) {
    for (j in seq_len(n_batch)) {
      counter <- counter + 1L
      example <- dataset_call(
        "ndm_sim_make_example",
        dataset_spec = dataset_spec,
        scaler = scaler,
        seed = 6000L + counter
      )
      values <- c(values, as.numeric(example$YTrue[, 1L]))
    }
  }
  stats::sd(values, na.rm = TRUE)
}

.ndm_assert_sim_covariates_available_at_issue <- function(dataset_spec) {
  shifts <- vapply(
    c("forward_shift_h", "forward_shift_c"),
    function(name) {
      value <- suppressWarnings(as.numeric(dataset_spec[[name]] %||% 0))
      if (length(value) != 1L || !is.finite(value)) {
        stop(
          "Simulation `", name, "` must be one finite value.",
          call. = FALSE
        )
      }
      value
    },
    numeric(1L)
  )
  if (any(shifts != 0)) {
    stop(
      paste(
        "Simulation context covariates must be available at forecast issue time;",
        "non-zero `forward_shift_h` or `forward_shift_c` values are not supported",
        "without an explicit causal lag convention."
      ),
      call. = FALSE
    )
  }
  invisible(dataset_spec)
}

.ndm_sim_get_batch_factory <- function(dataset_spec,
                                       scaler,
                                       runtime_env,
                                       dataset_call = .ndm_canonical_dataset_call) {
  .ndm_assert_sim_covariates_available_at_issue(dataset_spec)
  force(dataset_spec)
  force(scaler)
  force(runtime_env)
  force(dataset_call)

  function(nBatch = NULL,
           training = TRUE,
           openBrowser = FALSE,
           INPUT_REF_DAT = NULL,
           finalGenLocID = NULL,
           finalGenTimeID = NULL,
           nTimes = dataset_spec$n_time_steps %||%
             ((dataset_spec$context_length + dataset_spec$lookahead) * 4L),
           nTimesLook = dataset_spec$lookahead,
           PolicyList = list(ComputeScenario = FALSE, Scenario = NULL)) {
    if (isTRUE(openBrowser)) {
      browser()
    }
    nBatch <- .ndm_canonical_optional_count(nBatch, "nBatch")
    if (is.null(nBatch) || nBatch < 1L) {
      stop("`nBatch` must be a positive integer.", call. = FALSE)
    }

    local_spec <- dataset_spec
    local_spec$n_time_steps <- as.integer(nTimes)
    local_spec$lookahead <- as.integer(nTimesLook)
    compute_scenario <- isTRUE(PolicyList$ComputeScenario)
    policy_scenario <- PolicyList$Scenario %||% numeric(local_spec$n_time_steps)

    examples <- lapply(seq_len(nBatch), function(i) {
      dataset_call(
        "ndm_sim_make_example",
        dataset_spec = local_spec,
        scaler = scaler,
        seed = 5000L + i,
        compute_scenario = compute_scenario,
        policy_scenario = policy_scenario
      )
    })
    stacked <- dataset_call("ndm_datasets_stack_batches", examples = examples)
    .ndm_batch_to_runtime_jax(stacked, runtime_env = runtime_env)
  }
}

.ndm_runtime_value_is_present <- function(value) {
  if (is.null(value) || length(value) == 0L) {
    return(FALSE)
  }
  if (is.atomic(value)) {
    return(!all(is.na(value)))
  }
  TRUE
}

.ndm_runtime_value <- function(runtime_env, names, default = NULL) {
  for (name in names) {
    value <- get0(name, envir = runtime_env, inherits = TRUE, ifnotfound = NULL)
    if (.ndm_runtime_value_is_present(value)) {
      return(value)
    }
  }
  default
}

.ndm_sim_entry_value <- function(entry, runtime_env, names, default = NULL) {
  for (name in names) {
    value <- entry[[name]] %||% NULL
    if (.ndm_runtime_value_is_present(value)) {
      return(value)
    }
  }
  .ndm_runtime_value(runtime_env, names, default = default)
}

.ndm_sim_dataset_spec_from_runtime <- function(runtime_env,
                                               sim_entry = NULL,
                                               dataset_call = .ndm_canonical_dataset_call) {
  if (!is.environment(runtime_env)) {
    stop("`runtime_env` must be an environment.", call. = FALSE)
  }
  sim_entry <- sim_entry %||%
    get0("SimEntry", envir = runtime_env, inherits = TRUE, ifnotfound = NULL)
  if (is.data.frame(sim_entry)) {
    if (nrow(sim_entry) != 1L) {
      stop("`SimEntry` must contain exactly one row.", call. = FALSE)
    }
    sim_entry <- as.list(sim_entry[1L, , drop = FALSE])
  }
  if (!is.list(sim_entry)) {
    stop("A list-like `SimEntry` is required to construct the simulation dataset specification.", call. = FALSE)
  }
  value <- function(names, default = NULL) {
    .ndm_sim_entry_value(sim_entry, runtime_env, names, default = default)
  }
  number <- function(names, default = NULL) {
    out <- value(names, default)
    if (is.null(out)) NULL else as.numeric(as.character(out[[1L]]))
  }
  integer_value <- function(names, default = NULL) {
    out <- number(names, default)
    if (is.null(out)) NULL else as.integer(round(out))
  }

  dataset_call(
    "ndm_datasets_dataset_spec",
    kind = "sim",
    context_length = integer_value(c("ContextLength", "nTimesPast"), 8L),
    lookahead = integer_value(c("lookahead", "nTimesLookahead"), 12L),
    n_time_steps = integer_value(c("n_time_steps", "NTimeSteps_SIM")),
    global_population = number(c("global_population", "GLOBAL_ODE_NPOP"), 10000),
    gamma = number("gamma", 1),
    sigma = number("sigma", 1 / 0.76),
    xi = number("xi", 0.01),
    i0_a = number("i0_a", log(10)),
    i0_b = number("i0_b", 0.2),
    r0_a = number("r0_a", log(100)),
    r0_b = number("r0_b", 0.2),
    betat_init = number("betat_init", 2),
    invbetat_sd = number("invbetat_sd", 0.05),
    beta_restore_rate = number("beta_restore_rate", 0.25),
    c_endogeneous = number("c_endogeneous", 0.05),
    policy_responsiveness = number("policy_responsiveness", 0),
    policy_effectiveness = number("policy_effectiveness", 0),
    policy_decay = number("policy_decay", 0),
    covariate_type = as.character(value(c("covariate_type", "covariateType"), "sqrt")),
    roll_window = integer_value(c("roll_window", "rollCompute_window"), 52L),
    measurement_noise = number("measurement_noise", 0),
    hosp_rate = number("hosp_rate", 0.1),
    death_rate = number("death_rate", 0.01),
    forward_shift_h = 0L,
    forward_shift_c = 0L,
    n_inference_batches = integer_value("n_inference_batches", 8L),
    scaling_batches = integer_value("scaling_batches", 8L),
    base_id = integer_value(c("BaseID", "base_id"))
  )
}

.ndm_real_dataset_spec_from_runtime <- function(runtime_env,
                                                real_entry = NULL,
                                                dataset_call = .ndm_canonical_dataset_call) {
  if (!is.environment(runtime_env)) {
    stop("`runtime_env` must be an environment.", call. = FALSE)
  }
  real_entry <- real_entry %||%
    get0("RealEntry", envir = runtime_env, inherits = TRUE, ifnotfound = NULL)
  if (is.data.frame(real_entry)) {
    if (nrow(real_entry) != 1L) {
      stop("`RealEntry` must contain exactly one row.", call. = FALSE)
    }
    real_entry <- as.list(real_entry[1L, , drop = FALSE])
  }
  if (!is.list(real_entry)) {
    stop("A list-like `RealEntry` is required to construct the real dataset specification.", call. = FALSE)
  }
  value <- function(names, default = NULL) {
    for (name in names) {
      entry_value <- real_entry[[name]] %||% NULL
      if (.ndm_runtime_value_is_present(entry_value)) {
        return(entry_value)
      }
    }
    .ndm_runtime_value(runtime_env, names, default = default)
  }
  scalar_character <- function(names, default = NULL) {
    out <- value(names, default)
    if (is.null(out)) NULL else as.character(out[[1L]])
  }
  scalar_number <- function(names, default = NULL) {
    out <- value(names, default)
    if (is.null(out)) NULL else as.numeric(as.character(out[[1L]]))
  }
  scalar_integer <- function(names, default = NULL) {
    out <- scalar_number(names, default)
    if (is.null(out)) NULL else as.integer(round(out))
  }

  dataset_call(
    "ndm_datasets_dataset_spec",
    kind = "real",
    disease = scalar_character(c("disease", "DiseaseName"), "Covid"),
    context_length = scalar_integer(c("ContextLength", "maxTimesPast"), 8L),
    lookahead = scalar_integer(c("lookahead", "nTimesLookahead"), 12L),
    evaluation_time = scalar_integer("evaluationTime", 4L),
    evaluation_sequence = as.integer(value("evaluation_sequence", 1L:4L)),
    evaluation_origin_time_id = scalar_integer(
      c("evaluationOriginTimeID", "evaluation_origin_time_id")
    ),
    evaluation_horizon = scalar_integer(
      c("evaluationHorizon", "evaluation_horizon")
    ),
    inference_sampling = scalar_character(
      c("inferenceSampling", "inference_sampling"),
      "random"
    ),
    initial_transform = scalar_character(c("initialTransform", "initial_transform"), "none"),
    initial_norm_type = scalar_character(c("initialNormType", "initial_norm_type"), "at_t"),
    padding_method = scalar_character(c("paddingMethod", "padding_method"), "left"),
    split_type = scalar_character(c("OSSType", "split_type"), "OutOfTime"),
    data_subset = scalar_character("data_subset", "high_income"),
    data_inputs = value(c("dataInputs", "data_inputs"), "all"),
    outcomes = value(c("outcomes", "true_value_names"), "ihme_true_value_per_capita"),
    per_capita_scaling_factor = scalar_number("per_capita_scaling_factor", 10000),
    roll_window = scalar_integer(c("roll_window", "rollCompute_window"), 26L),
    min_anchoring_time = scalar_integer("min_anchoring_time", 4L),
    train_location_fraction = scalar_number("train_location_fraction", 0.8),
    n_inference_samples = scalar_integer("n_inference_samples", 1024L),
    base_id = scalar_integer(c("BaseID", "base_id")),
    outcome_metric = scalar_character("outcome_metric", "inc_death")
  )
}

.ndm_check_sim_manifest_runtime_contract <- function(dataset_spec,
                                                     runtime_env,
                                                     base_id = NULL) {
  expected <- list(
    base_id = base_id,
    context_length = .ndm_runtime_value(runtime_env, c("ContextLength", "nTimesPast")),
    lookahead = .ndm_runtime_value(runtime_env, c("lookahead", "nTimesLookahead")),
    n_time_steps = .ndm_runtime_value(runtime_env, c("n_time_steps", "NTimeSteps_SIM"))
  )
  mismatches <- character()
  for (name in names(expected)) {
    expected_value <- expected[[name]]
    manifest_value <- dataset_spec[[name]] %||% NULL
    if (!is.null(expected_value) && !is.null(manifest_value) &&
        !identical(as.numeric(manifest_value), as.numeric(expected_value))) {
      mismatches <- c(
        mismatches,
        sprintf("%s (runtime %s, artifact %s)", name, expected_value, manifest_value)
      )
    }
  }
  if (length(mismatches) > 0L) {
    stop(
      "Canonical simulation TFRecord does not match the active runtime: ",
      paste(mismatches, collapse = "; "),
      ".",
      call. = FALSE
    )
  }
  invisible(dataset_spec)
}

.ndm_canonical_preflight_error <- function(error,
                                           schema_kind,
                                           base_id = NULL,
                                           bootstrap = NULL) {
  bootstrap <- bootstrap %||% if (identical(schema_kind, "sim")) {
    "ndm_bootstrap_sim_tfrecords()"
  } else {
    "ndm_bootstrap_real_tfrecords()"
  }
  identity <- if (is.null(base_id)) "" else paste0(" for BaseID ", base_id)
  rlang::abort(
    paste0(
      "Canonical ", schema_kind, " TFRecord preparation failed", identity,
      ": ", conditionMessage(error), ". Run `", bootstrap,
      "` before training to create or refresh the artifact pair."
    ),
    class = "ndm_tfrecord_preflight_error",
    parent = error
  )
}

.ndm_prepare_canonical_tfrecord_runtime <- function(runtime_env,
                                                    paths,
                                                    schema_kind = c("real", "sim"),
                                                    dataset_spec,
                                                    batch_size,
                                                    n_train = NULL,
                                                    n_inference = NULL,
                                                    source_sha256 = NULL,
                                                    shuffle_train = TRUE,
                                                    reshuffle_train = shuffle_train,
                                                    run_seed = NULL,
                                                    verify_checksum = TRUE,
                                                    preview_batch = NULL,
                                                    dataset_call = .ndm_canonical_dataset_call,
                                                    iterator_factory = reticulate::as_iterator) {
  schema_kind <- match.arg(schema_kind)
  validated_pair <- tryCatch(
    .ndm_validate_canonical_tfrecord_pair(
      paths = paths,
      schema_kind = schema_kind,
      dataset_spec = dataset_spec,
      n_train = n_train,
      n_inference = n_inference,
      source_sha256 = source_sha256,
      verify_checksum = verify_checksum,
      dataset_call = dataset_call
    ),
    error = function(e) .ndm_canonical_preflight_error(e, schema_kind)
  )
  if (!is.null(preview_batch)) {
    .ndm_seed_runtime_batch(runtime_env, preview_batch)
  }
  .ndm_attach_canonical_tfrecords(
    runtime_env = runtime_env,
    train_file = paths$train_file,
    inference_file = paths$inference_file,
    schema_kind = schema_kind,
    batch_size = batch_size,
    shuffle_train = shuffle_train,
    reshuffle_train = reshuffle_train,
    run_seed = run_seed,
    verify_checksum = verify_checksum,
    max_train_examples = n_train,
    max_inference_examples = n_inference,
    dataset_call = dataset_call,
    iterator_factory = iterator_factory
  )

  invisible(list(
    runtime_env = runtime_env,
    validated_pair = validated_pair,
    preview_batch = preview_batch
  ))
}

.ndm_preflight_canonical_real_runtime <- function(
    runtime_env,
    dataset_spec = NULL,
    tfrecord_dir = NULL,
    base_id = NULL,
    paths = NULL,
    n_train = NULL,
    n_inference = NULL,
    source_sha256 = NULL,
    expected_seed = 0L,
    expected_producer = NULL,
    expected_inference_support = NULL,
    verify_checksum = TRUE,
    skip_tfrecords = NULL,
    bootstrap = NULL,
    dataset_call = .ndm_canonical_dataset_call) {
  if (!is.environment(runtime_env)) {
    stop("`runtime_env` must be an environment.", call. = FALSE)
  }
  skip_tfrecords <- skip_tfrecords %||%
    isTRUE(get0("SkipTfRecords", envir = runtime_env, inherits = TRUE, ifnotfound = FALSE))
  if (isTRUE(skip_tfrecords)) {
    return(invisible(list(runtime_env = runtime_env, skipped = TRUE)))
  }

  real_entry <- get0("RealEntry", envir = runtime_env, inherits = TRUE, ifnotfound = list())
  if (is.data.frame(real_entry) && nrow(real_entry) == 1L) {
    real_entry <- as.list(real_entry[1L, , drop = FALSE])
  }
  if (!is.list(real_entry)) {
    real_entry <- list()
  }
  tfrecord_dir <- tfrecord_dir %||%
    get0("TfRecordDir", envir = runtime_env, inherits = TRUE, ifnotfound = NULL)
  base_id <- base_id %||% (real_entry$BaseID %||% NULL) %||%
    get0("BaseID", envir = runtime_env, inherits = TRUE, ifnotfound = NULL)
  n_train <- n_train %||%
    get0("nSamplesTrain", envir = runtime_env, inherits = TRUE, ifnotfound = NULL) %||%
    (real_entry$nSamplesTrain %||% NULL)

  validated_pair <- tryCatch({
    paths <- paths %||% .ndm_canonical_tfrecord_paths(tfrecord_dir, base_id)
    dataset_spec <- dataset_spec %||% .ndm_real_dataset_spec_from_runtime(
      runtime_env = runtime_env,
      real_entry = real_entry,
      dataset_call = dataset_call
    )
    n_inference <- n_inference %||% dataset_spec$n_inference_samples %||% NULL
    source_sha256 <- source_sha256 %||% .ndm_runtime_value(
      runtime_env,
      c("ndm_real_source_sha256", "real_source_sha256", "source_sha256"),
      default = NULL
    )
    if (is.null(source_sha256)) {
      real_bundle <- .ndm_runtime_value(
        runtime_env,
        c("ndm_real_table_bundle", "analysis2_real_bundle", "real_bundle"),
        default = NULL
      )
      if (!is.null(real_bundle)) {
        source_sha256 <- dataset_call("ndm_real_source_sha256", real_bundle)
      }
    }
    pair <- .ndm_validate_canonical_tfrecord_pair(
      paths = paths,
      schema_kind = "real",
      dataset_spec = dataset_spec,
      n_train = n_train,
      n_inference = n_inference,
      source_sha256 = source_sha256,
      expected_seed = expected_seed,
      expected_producer = expected_producer,
      expected_inference_support = expected_inference_support,
      verify_checksum = verify_checksum,
      dataset_call = dataset_call
    )
    manifest_base_ids <- suppressWarnings(as.integer(c(
      pair$train$base_id,
      pair$inference$base_id
    )))
    if (!is.null(base_id) &&
        (anyNA(manifest_base_ids) || any(manifest_base_ids != as.integer(base_id)))) {
      stop("Canonical real TFRecord manifests do not match the active BaseID.", call. = FALSE)
    }
    pair
  }, error = function(e) {
    .ndm_canonical_preflight_error(
      e,
      "real",
      base_id,
      bootstrap = bootstrap
    )
  })

  assign("ndm_canonical_dataset_spec", validated_pair$dataset_spec, envir = runtime_env)
  assign("ndm_canonical_tfrecord_pair", validated_pair, envir = runtime_env)
  invisible(list(
    runtime_env = runtime_env,
    skipped = FALSE,
    validated_pair = validated_pair
  ))
}

.ndm_prepare_canonical_sim_runtime <- function(runtime_env,
                                               dataset_spec = NULL,
                                               paths = NULL,
                                               tfrecord_dir = NULL,
                                               base_id = NULL,
                                               batch_size = NULL,
                                               n_train = NULL,
                                               n_inference = NULL,
                                               shuffle_train = TRUE,
                                               reshuffle_train = shuffle_train,
                                               run_seed = NULL,
                                               verify_checksum = TRUE,
                                               sim_outcome_sd = NULL,
                                               outcome_sd_n_batch = 32L,
                                               outcome_sd_n_outer = 4L,
                                               preview_batch = NULL,
                                               skip_tfrecords = NULL,
                                               dataset_call = .ndm_canonical_dataset_call,
                                               iterator_factory = reticulate::as_iterator) {
  if (!is.environment(runtime_env)) {
    stop("`runtime_env` must be an environment.", call. = FALSE)
  }
  skip_tfrecords <- skip_tfrecords %||%
    isTRUE(get0("SkipTfRecords", envir = runtime_env, inherits = TRUE, ifnotfound = FALSE))
  if (isTRUE(skip_tfrecords)) {
    return(invisible(list(runtime_env = runtime_env, skipped = TRUE)))
  }

  sim_entry <- get0("SimEntry", envir = runtime_env, inherits = TRUE, ifnotfound = list())
  if (is.data.frame(sim_entry) && nrow(sim_entry) == 1L) {
    sim_entry <- as.list(sim_entry[1L, , drop = FALSE])
  }
  if (!is.list(sim_entry)) {
    sim_entry <- list()
  }
  tfrecord_dir <- tfrecord_dir %||%
    get0("TfRecordDir", envir = runtime_env, inherits = TRUE, ifnotfound = NULL)
  base_id <- base_id %||% (sim_entry$BaseID %||% NULL) %||%
    get0("BaseID", envir = runtime_env, inherits = TRUE, ifnotfound = NULL)
  batch_size <- batch_size %||%
    get0("nBatch", envir = runtime_env, inherits = TRUE, ifnotfound = NULL)
  n_train <- n_train %||%
    get0("nSamplesTrain", envir = runtime_env, inherits = TRUE, ifnotfound = NULL) %||%
    (sim_entry$nSamplesTrain %||% NULL)
  run_seed <- run_seed %||%
    get0("SEED_", envir = runtime_env, inherits = TRUE, ifnotfound = NULL)

  preflight <- tryCatch({
    paths <- paths %||% .ndm_canonical_tfrecord_paths(tfrecord_dir, base_id)
    expected_dataset_spec <- dataset_spec %||% .ndm_sim_dataset_spec_from_runtime(
      runtime_env = runtime_env,
      sim_entry = sim_entry,
      dataset_call = dataset_call
    )
    .ndm_assert_sim_covariates_available_at_issue(expected_dataset_spec)
    n_inference <- n_inference %||% if (!is.null(expected_dataset_spec$n_inference_batches)) {
      as.integer(expected_dataset_spec$n_inference_batches) * 128L
    } else {
      NULL
    }
    validated_pair <- .ndm_validate_canonical_tfrecord_pair(
      paths = paths,
      schema_kind = "sim",
      dataset_spec = expected_dataset_spec,
      n_train = n_train,
      n_inference = n_inference,
      verify_checksum = verify_checksum,
      dataset_call = dataset_call
    )
    scaler <- validated_pair$train$scaler %||% NULL
    if (is.null(scaler) || is.null(scaler$mean) || is.null(scaler$sd)) {
      stop(
        "Validated simulation TFRecord manifests did not provide a complete scaler.",
        call. = FALSE
      )
    }
    list(
      paths = paths,
      dataset_spec = validated_pair$dataset_spec,
      n_inference = n_inference,
      validated_pair = validated_pair,
      scaler = scaler
    )
  }, error = function(e) .ndm_canonical_preflight_error(e, "sim", base_id))
  paths <- preflight$paths
  dataset_spec <- preflight$dataset_spec
  n_inference <- preflight$n_inference
  validated_pair <- preflight$validated_pair
  scaler <- preflight$scaler
  sim_outcome_sd <- sim_outcome_sd %||% get0(
    "SIM_GLOBAL_OUTCOME_SD",
    envir = runtime_env,
    inherits = TRUE,
    ifnotfound = NULL
  ) %||% .ndm_sim_outcome_sd(
    dataset_spec = dataset_spec,
    scaler = scaler,
    n_batch = outcome_sd_n_batch,
    n_outer = outcome_sd_n_outer,
    dataset_call = dataset_call
  )
  sim_outcome_sd <- suppressWarnings(as.numeric(sim_outcome_sd))
  if (length(sim_outcome_sd) != 1L || !is.finite(sim_outcome_sd) || sim_outcome_sd <= 0) {
    stop("`SIM_GLOBAL_OUTCOME_SD` must be one finite positive value.", call. = FALSE)
  }
  get_batch <- .ndm_sim_get_batch_factory(
    dataset_spec = dataset_spec,
    scaler = scaler,
    runtime_env = runtime_env,
    dataset_call = dataset_call
  )
  n_batch_sim_grid <- .ndm_runtime_value(runtime_env, "nBatch_SimGridGen", 256L)
  n_batch_sim_grid <- .ndm_canonical_optional_count(n_batch_sim_grid, "nBatch_SimGridGen")
  if (is.null(n_batch_sim_grid) || n_batch_sim_grid < 1L) {
    stop("`nBatch_SimGridGen` must be one positive integer.", call. = FALSE)
  }
  n_monte_eval <- .ndm_runtime_value(
    runtime_env,
    "nMonteEval",
    ceiling(1000 / n_batch_sim_grid)
  )
  n_monte_eval <- .ndm_canonical_optional_count(n_monte_eval, "nMonteEval")
  if (is.null(n_monte_eval) || n_monte_eval < 1L) {
    stop("`nMonteEval` must be one positive integer.", call. = FALSE)
  }
  n_times_look_validation <- .ndm_runtime_value(
    runtime_env,
    c("nTimesLookValidation", "nTimesLookValidationInference"),
    dataset_spec$lookahead
  )
  n_times_look_validation <- .ndm_canonical_optional_count(
    n_times_look_validation,
    "nTimesLookValidation"
  )
  if (is.null(n_times_look_validation) || n_times_look_validation < 1L) {
    stop("`nTimesLookValidation` must be one positive integer.", call. = FALSE)
  }

  runtime_values <- list(
    SIM_GLOBAL_SCALE_MEAN = scaler$mean,
    SIM_GLOBAL_SCALE_SD = scaler$sd,
    SIM_GLOBAL_OUTCOME_SD = sim_outcome_sd,
    analysis2_sim_get_batch = get_batch,
    GetBatch = get_batch,
    GetBatch_sim = get_batch,
    GetBatch_base = get_batch,
    input_df_red_in = data.frame(dummy = 0),
    input_df_red_out = data.frame(dummy = 0),
    input_df_red_full = data.frame(dummy = 0),
    nBatch_SimGridGen = n_batch_sim_grid,
    nMonteEval = n_monte_eval,
    nTimesLookValidation = n_times_look_validation,
    ndm_canonical_dataset_spec = dataset_spec,
    ndm_canonical_tfrecord_pair = validated_pair
  )
  list2env(runtime_values, envir = runtime_env)

  preview_batch <- preview_batch %||% get_batch(nBatch = batch_size)
  .ndm_seed_runtime_batch(runtime_env, preview_batch)
  tryCatch(
    .ndm_attach_canonical_tfrecords(
      runtime_env = runtime_env,
      train_file = paths$train_file,
      inference_file = paths$inference_file,
      schema_kind = "sim",
      batch_size = batch_size,
      shuffle_train = shuffle_train,
      reshuffle_train = reshuffle_train,
      run_seed = run_seed,
      verify_checksum = verify_checksum,
      max_train_examples = n_train,
      max_inference_examples = n_inference,
      dataset_call = dataset_call,
      iterator_factory = iterator_factory
    ),
    error = function(e) .ndm_canonical_preflight_error(e, "sim", base_id)
  )

  invisible(list(
    runtime_env = runtime_env,
    skipped = FALSE,
    dataset_spec = dataset_spec,
    validated_pair = validated_pair,
    scaler = scaler,
    outcome_sd = sim_outcome_sd,
    get_batch = get_batch,
    preview_batch = preview_batch
  ))
}
