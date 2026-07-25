.ndm_requests_tfrecord_regeneration <- function(runtime_env, value_sources = list()) {
  flags <- c("resave_tfrecords", "ReSaveTfRecords")
  is_invalid <- function(value) !is.null(value) && !identical(value, FALSE)
  any(vapply(flags, function(flag) {
    is_invalid(get0(flag, envir = runtime_env, inherits = TRUE)) ||
      any(vapply(value_sources, function(values) is_invalid(values[[flag]]), logical(1)))
  }, logical(1)))
}

#' Prepare a local runtime for execution
#'
#' These wrappers load package-owned runtime components into an isolated
#' environment and prepare the selected data workflow.
#'
#' @details
#' Before building, training, predicting, or restoring artifacts from a runtime
#' environment, initialize the backend for the target conda environment with
#' `ndm_initialize_backend()`. `ndm_prepare_runtime()` requires the
#' `fastmatch` package. Canonical real and simulation preparation requires a
#' completed `train_<BaseID>.tfrecord` / `inference_<BaseID>.tfrecord` pair and
#' their ndmdatasets manifests. Set `SkipTfRecords = TRUE` only for explicit
#' in-memory preparation; the simulation compatibility path then also requires
#' `zoo`. The compatibility fields
#' `resave_tfrecords` and `ReSaveTfRecords` must remain `FALSE` in configs,
#' runtime environments, and runtime globals.
#'
#' @param config An object of class `ndm_config`, usually created by
#'   `ndm_create_config()`.
#' @param runtime_env Runtime environment that should receive the sourced
#'   objects.
#' @param runtime_globals Named list of additional bindings to assign before the
#'   relevant runtime or data-preparation stage.
#'
#' @returns `ndm_prepare_runtime()` invisibly returns `runtime_env` after
#'   loading the package-owned helper and backend code.
#'
#' @examples
#' env <- ndm_new_runtime_env()
#' cfg <- ndm_create_config()
#' \dontrun{
#' ndm_prepare_runtime(cfg, env)
#' }
#'
#' @export
ndm_prepare_runtime <- function(config = ndm_create_config(),
                                runtime_env = ndm_new_runtime_env(),
                                runtime_globals = list()) {
  if (!inherits(config, "ndm_config")) {
    stop("`config` must inherit from class 'ndm_config'.", call. = FALSE)
  }
  .ndm_validate_resave_tfrecords(
    "generic",
    .ndm_requests_tfrecord_regeneration(runtime_env, list(config, runtime_globals))
  )

  ndm_set_runtime_globals(runtime_env, list(ndm_config = config))
  ndm_set_runtime_globals(runtime_env, as.list(config))
  ndm_set_runtime_globals(
    runtime_env,
    list(
      EnableKVCaching = config$enable_kv_cache %||% FALSE,
      InferenceMCDraws = config$inference_mc_draws %||% 5L,
      ObservationScaleFloor = config$observation_scale_floor %||% 1e-5,
      InitialObservationScale = config$initial_observation_scale %||% 1.0,
      neuralode_variational = config$neuralode_variational %||%
        identical(config$model_type, "NeuralODE")
    )
  )
  ndm_set_runtime_globals(runtime_env, runtime_globals)
  ndm_load_runtime(
    env = runtime_env,
    float_type = config$float_type,
    force_to_gpu = config$force_to_gpu,
    gpu_mem_frac = config$gpu_mem_frac,
    resave_tfrecords = config$resave_tfrecords
  )
}

#' @rdname ndm_prepare_runtime
#'
#' @param generator Which data workflow to prepare in `runtime_env`.
#'
#' @returns `ndm_prepare_data()` invisibly returns `runtime_env` after preparing
#'   the requested data workflow.
#'
#' @examples
#' env <- ndm_new_runtime_env()
#' ndm_set_runtime_globals(env, list(example_value = 1L))
#'
#' @export
ndm_prepare_data <- function(runtime_env,
                             generator = c("sim", "real", "multidisease"),
                             runtime_globals = list()) {
  if (!is.environment(runtime_env)) {
    stop("`runtime_env` must be an environment.", call. = FALSE)
  }

  generator <- match.arg(generator)
  .ndm_validate_resave_tfrecords(
    generator,
    .ndm_requests_tfrecord_regeneration(runtime_env, list(runtime_globals))
  )
  ndm_set_runtime_globals(runtime_env, runtime_globals)
  skip_tfrecords <- isTRUE(get0(
    "SkipTfRecords",
    envir = runtime_env,
    inherits = TRUE,
    ifnotfound = FALSE
  ))
  ndm_set_runtime_globals(
    runtime_env,
    list(SkipTfRecords = skip_tfrecords)
  )

  if (identical(generator, "sim") && isTRUE(skip_tfrecords)) {
    .ndm_require_namespaces(
      "zoo",
      context = "ndm_prepare_data(generator = 'sim', SkipTfRecords = TRUE)"
    )
  }
  if (identical(generator, "multidisease")) {
    .ndm_prepare_multidisease_data(runtime_env)
  } else if (identical(generator, "sim") && !isTRUE(skip_tfrecords)) {
    .ndm_prepare_canonical_sim_runtime(runtime_env)
  } else {
    if (identical(generator, "real") && !isTRUE(skip_tfrecords)) {
      .ndm_preflight_canonical_real_runtime(runtime_env)
    }
    ndm_source_runtime_data(
      env = runtime_env,
      generator = generator
    )
  }
  ndm_set_runtime_globals(runtime_env, list(ndm_data_generator = generator))
}

.ndm_runtime_env_from_object <- function(x, arg = "x") {
  if (is.environment(x)) {
    return(x)
  }

  if (inherits(x, "ndm_model") || inherits(x, "ndm_trained_model")) {
    if (!is.environment(x$env)) {
      stop("`", arg, "` does not contain a valid runtime environment.", call. = FALSE)
    }
    return(x$env)
  }

  stop("`", arg, "` must be an `ndm_model`, an `ndm_trained_model`, or a prepared runtime environment.", call. = FALSE)
}

.ndm_require_runtime_bindings <- function(env, names, context) {
  missing <- names[!vapply(names, exists, logical(1), envir = env, inherits = FALSE)]
  if (length(missing) > 0L) {
    stop(
      "Runtime environment is missing required objects for ",
      context,
      ": ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }

  invisible(env)
}

.ndm_is_packaged_prediction_batch <- function(batch) {
  if (!is.list(batch) || length(batch) != 4L) {
    return(FALSE)
  }

  expected_lengths <- c(2L, 2L, 1L, 1L)
  if (!all(vapply(batch, is.list, logical(1)))) {
    return(FALSE)
  }

  if (!identical(vapply(batch, length, integer(1)), expected_lengths)) {
    return(FALSE)
  }

  TRUE
}

.ndm_batch_size_from_object <- function(x) {
  if (is.null(x)) {
    stop("Could not infer batch size from a NULL object.", call. = FALSE)
  }

  if (is.list(x) && !is.null(names(x)) && "XPred" %in% names(x)) {
    return(.ndm_batch_size_from_object(x$XPred))
  }

  if (is.list(x) && length(x) > 0L) {
    return(.ndm_batch_size_from_object(x[[1]]))
  }

  dims <- NULL
  if ("python.builtin.object" %in% class(x)) {
    dims <- try(reticulate::py_to_r(x$shape), silent = TRUE)
  } else {
    dims <- dim(x)
  }

  if (inherits(dims, "try-error") || is.null(dims) || length(dims) == 0L) {
    stop("Could not infer batch size from the supplied batch object.", call. = FALSE)
  }

  as.integer(dims[[1]])
}

.ndm_as_runtime_jax <- function(x, env) {
  if (is.list(x)) {
    return(lapply(x, .ndm_as_runtime_jax, env = env))
  }
  if ("python.builtin.object" %in% class(x)) {
    if (exists("ndm_runtime_data_to_device", envir = env, inherits = FALSE)) {
      return(get("ndm_runtime_data_to_device", envir = env, inherits = FALSE)(x))
    }
    return(env$jnp$array(reticulate::py_to_r(x)))
  }
  if (exists("jnp", envir = env, inherits = FALSE)) {
    x <- get("jnp", envir = env, inherits = FALSE)$array(x)
    if (exists("ndm_runtime_data_to_device", envir = env, inherits = FALSE)) {
      return(get("ndm_runtime_data_to_device", envir = env, inherits = FALSE)(x))
    }
    return(x)
  }
  x
}

.ndm_runtime_device_helper <- function(env, helper_name, value) {
  if (!exists(helper_name, envir = env, inherits = FALSE)) {
    return(value)
  }
  get(helper_name, envir = env, inherits = FALSE)(value)
}

.ndm_runtime_getpred_saveat_info <- function(env, value) {
  value <- .ndm_runtime_device_helper(
    env,
    "ndm_runtime_normalize_getpred_saveat_info",
    value
  )
  .ndm_runtime_device_helper(env, "ndm_runtime_replicate_tree", value)
}

.ndm_prepare_prediction_batch <- function(batch, env) {
  if (is.null(batch)) {
    if (!exists("batch_l_cal", envir = env, inherits = FALSE)) {
      stop("No `batch` supplied and `batch_l_cal` is not available in the model environment.", call. = FALSE)
    }
    batch <- get("batch_l_cal", envir = env, inherits = FALSE)
  }

  if (is.list(batch) && !is.null(names(batch)) && "XPred" %in% names(batch)) {
    batch <- .ndm_as_runtime_jax(batch, env = env)
    packaged <- ndm_batch_to_model_inputs(batch)
    packaged <- .ndm_runtime_device_helper(env, "ndm_runtime_data_to_device", packaged)
    return(list(raw = batch, packaged = packaged))
  }

  if (.ndm_is_packaged_prediction_batch(batch)) {
    packaged <- .ndm_runtime_device_helper(env, "ndm_runtime_data_to_device", batch)
    return(list(raw = batch, packaged = packaged))
  }

  stop(
    "`batch` must be either a named TFRecord-style batch list or a packaged model-input list that follows ndm_batch_to_model_inputs().",
    call. = FALSE
  )
}

#' Build and train models with package-managed runtime stages
#'
#' These wrappers layer a small R API over package-managed Phase 1 model build
#' and training stages.
#'
#' @details
#' These functions assume the backend has already been initialized with
#' `ndm_initialize_backend()` and that `runtime_env` already contains the
#' upstream runtime and data globals required by the selected workflow.
#' `ndm_train()` requires `rrapply`; it also requires `zip` when checkpointing
#' is enabled and `zoo` for multidisease training. Simulation preparation only
#' requires `zoo` when `SkipTfRecords = TRUE` selects the in-memory compatibility
#' path documented on [ndm_prepare_runtime()].
#'
#' @param config An object of class `ndm_config`, usually created by
#'   `ndm_create_config()`.
#' @param runtime_env Runtime environment containing loaded runtime helpers and
#'   data globals.
#' @param model_type Model family to build. Either `"DecoderOnly"` or
#'   `"NeuralODE"`.
#' @param model_spec Optional `ndm_model_spec` object used to override the model
#'   TeX specification supplied to the runtime model builder.
#' @param backbone Backbone family. Phase 1 supports `"transformer"` only.
#' @param runtime_globals Named list of additional globals assigned during the
#'   relevant runtime stage. `ndm_build_model()` uses them before building the
#'   model, while `ndm_fit()` uses them before sourcing the runtime.
#' @param x Either an `ndm_model`, an `ndm_trained_model`, or a prepared runtime
#'   environment, depending on the function being called.
#' @param run_define Logical scalar indicating whether the training definition
#'   script should be sourced.
#' @param run_loop Logical scalar indicating whether the training loop script
#'   should be sourced.
#' @param data_generator Which data workflow to prepare before calling
#'   `ndm_build_model()` inside `ndm_fit()`.
#' @param data_globals Named list of globals assigned before data preparation in
#'   `ndm_fit()`.
#' @param build_globals Named list of globals assigned before building the model
#'   in `ndm_fit()`.
#' @param train_globals Named list of globals assigned immediately before
#'   training in `ndm_fit()`.
#' @param ... Unused; included for S3 method compatibility.
#'
#' @returns `ndm_build_model()` returns an object of class `ndm_model`.
#'   `ndm_train()` and `ndm_fit()` return an object of class
#'   `ndm_trained_model`.
#'
#' @examples
#' cfg <- ndm_create_config()
#' spec <- ndm_model_spec()
#' \dontrun{
#' runtime_env <- ndm_prepare_runtime(cfg)
#' ndm_prepare_data(runtime_env, generator = "sim")
#' model <- ndm_build_model(runtime_env, model_spec = spec)
#' trained <- ndm_train(model)
#' }
#'
#' @export
ndm_build_model <- function(runtime_env,
                            model_type = c("DecoderOnly", "NeuralODE"),
                            model_spec = NULL,
                            backbone = "transformer",
                            runtime_globals = list()) {
  runtime_env <- .ndm_runtime_env_from_object(runtime_env, arg = "runtime_env")

  model_type <- match.arg(model_type)
  if (!identical(backbone, "transformer")) {
    stop("Phase 1 only supports backbone = 'transformer'.", call. = FALSE)
  }

  .ndm_install_runtime_helpers(runtime_env)
  paths <- ndm_runtime_paths()
  ndm_source_runtime_backend(
    env = runtime_env,
    float_type = runtime_env$float_type %||% "32",
    force_to_gpu = isTRUE(runtime_env$force_to_gpu),
    gpu_mem_frac = runtime_env$gpu_mem_frac,
    resave_tfrecords = isTRUE(runtime_env$resave_tfrecords)
  )
  backend_modules <- ndm_backend_modules()
  ndm_set_runtime_globals(
    runtime_env,
    list(
      backend = backend_modules,
      jax = backend_modules$jax,
      jnp = backend_modules$jnp,
      np = backend_modules$np,
      optax = backend_modules$optax,
      eq = backend_modules$eq,
      diffrax = backend_modules$diffrax,
      flash_mha = backend_modules$flash_mha,
      py_gc = backend_modules$py_gc,
      tf = backend_modules$tf,
      jaxFloatType = backend_modules$jaxFloatType,
      send2cpu = backend_modules$send2cpu,
      send2gpu = backend_modules$send2gpu,
      oryx = backend_modules$oryx,
      SoftPlus = backend_modules$SoftPlus,
      Sigmoid = backend_modules$Sigmoid,
      InvSoftPlus = backend_modules$InvSoftPlus,
      switch_filter_jit = backend_modules$eq$filter_jit,
      DefaultDtypeTf = if (identical(backend_modules$float_type, "64")) "tf$float64" else "tf$float32"
    )
  )
  ndm_set_runtime_globals(runtime_env, runtime_globals)
  ndm_set_runtime_globals(
    runtime_env,
    list(
      ModelType = model_type,
      BackboneType = backbone,
      UseLatentAttention = FALSE,
      ndm_model_type = model_type,
      ndm_backbone = backbone
    )
  )
  .ndm_materialize_neuralode_runtime_config(runtime_env)

  cleanup <- FALSE
  if (!is.null(model_spec)) {
    materialized <- .ndm_materialize_model_spec(model_spec)
    cleanup <- isTRUE(materialized$cleanup)
    ndm_set_runtime_globals(
      runtime_env,
      list(
        model_tex_loc = materialized$path,
        ndm_model_spec = model_spec
      )
    )
    if (cleanup) {
      on.exit(unlink(materialized$path), add = TRUE)
    }
  }

  tryCatch(
      .ndm_source_runtime_file(paths$build_model, runtime_env),
      error = function(e) {
        stop(
          "Failed to load the package-owned model builder. This assumes the upstream setup and data globals are already present in `runtime_env`.\n",
          "Original error: ", conditionMessage(e),
          call. = FALSE
        )
    }
  )

  .ndm_require_runtime_bindings(
    runtime_env,
    c(
      "ModelList",
      "state",
      "PriorList",
      "PolicyList",
      "GetPredSaveAtInfo_default",
      "GetPred_inference",
      "GetPred_train_jit",
      "getLoss_train"
    ),
    context = "prediction and loss after model build"
  )

  ndm_set_runtime_globals(
    runtime_env,
    list(
      ndm_model_spec = model_spec
    )
  )

  structure(
    list(
      env = runtime_env,
      model_type = model_type,
      backbone = backbone,
      config = .ndm_runtime_get0(runtime_env, "ndm_config", ifnotfound = NULL),
      model_spec = model_spec,
      data_generator = .ndm_runtime_get0(runtime_env, "ndm_data_generator", ifnotfound = NULL),
      model = if (exists("ModelList", envir = runtime_env, inherits = FALSE)) {
        get("ModelList", envir = runtime_env, inherits = FALSE)
      } else {
        NULL
      },
      state = if (exists("state", envir = runtime_env, inherits = FALSE)) {
        get("state", envir = runtime_env, inherits = FALSE)
      } else {
        NULL
      }
    ),
    class = "ndm_model"
  )
}

#' @rdname ndm_build_model
#'
#' @returns `print.ndm_model()` prints a brief summary of an `ndm_model` object
#'   and invisibly returns `x`.
#'
#' @export
print.ndm_model <- function(x, ...) {
  cat("<ndm_model>\n")
  cat(sprintf("  model_type: %s\n", x$model_type %||% NA_character_))
  cat(sprintf("  backbone: %s\n", x$backbone %||% NA_character_))
  cat(sprintf("  has_model: %s\n", !is.null(x$model)))
  invisible(x)
}

#' Generate predictions or evaluate the training loss
#'
#' These helpers call the runtime prediction and loss functions stored in a
#' prepared runtime environment.
#'
#' @param x Either an `ndm_model`, an `ndm_trained_model`, or a prepared runtime
#'   environment that already contains the required runtime objects.
#' @param batch Optional batch object. Supply either a named TFRecord-style list
#'   or the four-element packaged list returned by `ndm_batch_to_model_inputs()`.
#'   When `NULL`, the runtime must already contain `batch_l_cal`.
#' @param inference Logical scalar indicating whether `ndm_predict()` should use
#'   the inference prediction path or the training prediction path.
#' @param seed Integer seed used to derive the per-example JAX RNG keys.
#' @param update_state Logical scalar indicating whether the runtime `state`
#'   object should be updated with the returned state.
#' @param y Optional observed outcome tensor passed to `ndm_loss()`. When
#'   omitted, `YTrue_out` is inferred from `batch`.
#' @param y_mask Optional outcome mask tensor passed to `ndm_loss()`. When
#'   omitted, `YTrue_out_mask` is inferred from `batch`.
#' @param iteration Training iteration number forwarded to the runtime loss
#'   function.
#'
#' @returns `ndm_predict()` returns the prediction object produced by the
#'   runtime. `ndm_loss()` returns the loss object produced by the runtime.
#'
#' @examples
#' \dontrun{
#' preds <- ndm_predict(model, batch = batch_l_cal)
#' loss <- ndm_loss(model, batch = batch_l_cal)
#' }
#'
#' @export
ndm_predict <- function(x,
                        batch = NULL,
                        inference = TRUE,
                        seed = 1L,
                        update_state = TRUE) {
  runtime_env <- .ndm_runtime_env_from_object(x)

  pred_fun_name <- if (isTRUE(inference)) {
    if (exists("predict_step_compiled", envir = runtime_env, inherits = FALSE)) {
      "predict_step_compiled"
    } else {
      "GetPred_inference"
    }
  } else {
    if (exists("predict_train_step_compiled", envir = runtime_env, inherits = FALSE)) {
      "predict_train_step_compiled"
    } else {
      "GetPred_train_jit"
    }
  }
  .ndm_require_runtime_bindings(
    runtime_env,
    c(
      "ModelList",
      "state",
      "PriorList",
      "PolicyList",
      "GetPredSaveAtInfo_default",
      "jax",
      "JaxKey",
      pred_fun_name
    ),
    context = "prediction"
  )

  prepared_batch <- .ndm_prepare_prediction_batch(batch, env = runtime_env)
  batch_size <- .ndm_batch_size_from_object(prepared_batch$raw)
  seed_matrix <- runtime_env$jax$random$split(runtime_env$JaxKey(as.integer(seed)), as.integer(batch_size))
  seed_matrix <- .ndm_runtime_device_helper(runtime_env, "ndm_runtime_data_to_device", seed_matrix)
  getpred_saveat_info <- .ndm_runtime_getpred_saveat_info(
    runtime_env,
    runtime_env$GetPredSaveAtInfo_default
  )

  pred_fun <- get(pred_fun_name, envir = runtime_env, inherits = FALSE)
  result <- pred_fun(
    .ndm_runtime_device_helper(runtime_env, "ndm_runtime_replicate_tree", runtime_env$ModelList),
    prepared_batch$packaged,
    .ndm_runtime_device_helper(runtime_env, "ndm_runtime_replicate_tree", runtime_env$state),
    .ndm_runtime_device_helper(runtime_env, "ndm_runtime_replicate_tree", runtime_env$PriorList),
    .ndm_runtime_device_helper(runtime_env, "ndm_runtime_replicate_tree", runtime_env$PolicyList),
    getpred_saveat_info,
    seed_matrix
  )

  if (isTRUE(update_state)) {
    assign("state", result[[2]], envir = runtime_env)
  }

  result[[1]]
}

#' @rdname ndm_predict
#' @export
ndm_loss <- function(x,
                     batch = NULL,
                     y = NULL,
                     y_mask = NULL,
                     iteration = 1L,
                     seed = 1L,
                     update_state = TRUE) {
  runtime_env <- .ndm_runtime_env_from_object(x)
  .ndm_require_runtime_bindings(
    runtime_env,
    c(
      "ModelList",
      "state",
      "PriorList",
      "PolicyList",
      "GetPredSaveAtInfo_default",
      "getLoss_train",
      "jax",
      "JaxKey",
      "jnp"
    ),
    context = "loss evaluation"
  )

  prepared_batch <- .ndm_prepare_prediction_batch(batch, env = runtime_env)
  raw_batch <- prepared_batch$raw
  if (is.null(y)) {
    if (!is.list(raw_batch) || is.null(names(raw_batch)) || !"YTrue_out" %in% names(raw_batch)) {
      stop("`y` was not supplied and could not be inferred from the batch.", call. = FALSE)
    }
    y <- raw_batch$YTrue_out
  }
  if (is.null(y_mask)) {
    if (!is.list(raw_batch) || is.null(names(raw_batch)) || !"YTrue_out_mask" %in% names(raw_batch)) {
      stop("`y_mask` was not supplied and could not be inferred from the batch.", call. = FALSE)
    }
    y_mask <- raw_batch$YTrue_out_mask
  }

  y <- .ndm_as_runtime_jax(y, env = runtime_env)
  y_mask <- .ndm_as_runtime_jax(y_mask, env = runtime_env)
  batch_size <- .ndm_batch_size_from_object(raw_batch)
  seed_matrix <- runtime_env$jax$random$split(runtime_env$JaxKey(as.integer(seed)), as.integer(batch_size))
  seed_matrix <- .ndm_runtime_device_helper(runtime_env, "ndm_runtime_data_to_device", seed_matrix)
  getpred_saveat_info <- .ndm_runtime_getpred_saveat_info(
    runtime_env,
    runtime_env$GetPredSaveAtInfo_default
  )
  loss_fun <- if (exists("loss_step_compiled", envir = runtime_env, inherits = FALSE)) {
    get("loss_step_compiled", envir = runtime_env, inherits = FALSE)
  } else {
    runtime_env$getLoss_train
  }

  result <- loss_fun(
    .ndm_runtime_device_helper(runtime_env, "ndm_runtime_replicate_tree", runtime_env$ModelList),
    prepared_batch$packaged,
    y,
    y_mask,
    runtime_env$jnp$array(as.numeric(iteration)),
    .ndm_runtime_device_helper(runtime_env, "ndm_runtime_replicate_tree", runtime_env$state),
    .ndm_runtime_device_helper(runtime_env, "ndm_runtime_replicate_tree", runtime_env$PriorList),
    .ndm_runtime_device_helper(runtime_env, "ndm_runtime_replicate_tree", runtime_env$PolicyList),
    getpred_saveat_info,
    seed_matrix
  )

  if (isTRUE(update_state)) {
    assign("state", result[[2]], envir = runtime_env)
  }

  result[[1]]
}

.ndm_prediction_value <- function(pred, name) {
  if (is.null(pred)) {
    return(NULL)
  }
  if (is.list(pred) && !is.null(names(pred)) && name %in% names(pred)) {
    return(pred[[name]])
  }
  NULL
}

.ndm_prediction_ode_value <- function(pred, name) {
  params <- .ndm_prediction_value(pred, "ODEParamsSampList")
  if (is.null(params)) {
    return(NULL)
  }
  if (is.list(params) && !is.null(names(params)) && name %in% names(params)) {
    return(params[[name]])
  }
  parts <- strsplit(name, ".", fixed = TRUE)[[1L]]
  if (length(parts) == 2L && is.list(params)) {
    nested <- params[[parts[[1L]]]]
    if (is.list(nested) && !is.null(names(nested)) && parts[[2L]] %in% names(nested)) {
      return(nested[[parts[[2L]]]])
    }
  }
  NULL
}

.ndm_array_to_r <- function(x, env = NULL) {
  if (is.null(x)) {
    return(NULL)
  }
  if ("python.builtin.object" %in% class(x)) {
    if (is.environment(env) && exists("np", envir = env, inherits = FALSE)) {
      arr <- try(get("np", envir = env, inherits = FALSE)$asanyarray(x), silent = TRUE)
      if (!inherits(arr, "try-error")) {
        return(reticulate::py_to_r(arr))
      }
    }
    return(reticulate::py_to_r(x))
  }
  x
}

.ndm_runtime_int <- function(env, name, ifnotfound = NA_integer_) {
  value <- .ndm_runtime_get0(env, name, ifnotfound = ifnotfound)
  value <- suppressWarnings(as.integer(value))
  if (length(value) != 1L || is.na(value)) {
    return(ifnotfound)
  }
  value
}

.ndm_runtime_character <- function(env, name, ifnotfound = character(0)) {
  value <- .ndm_runtime_get0(env, name, ifnotfound = ifnotfound)
  if (is.null(value)) {
    return(character(0))
  }
  value <- as.character(value)
  value[nzchar(value)]
}

.ndm_prediction_model_type <- function(x, env) {
  if ((inherits(x, "ndm_model") || inherits(x, "ndm_trained_model")) && !is.null(x$model_type)) {
    return(as.character(x$model_type))
  }
  as.character(.ndm_runtime_get0(
    env,
    "ModelType",
    ifnotfound = .ndm_runtime_get0(env, "ndm_model_type", ifnotfound = NA_character_)
  ))
}

.ndm_prediction_n_outcomes <- function(env, pred) {
  n_outcomes <- .ndm_runtime_int(env, "nOutcomes", ifnotfound = NA_integer_)
  if (!is.na(n_outcomes)) {
    return(n_outcomes)
  }
  y_mu <- .ndm_array_to_r(.ndm_prediction_value(pred, "y_mu"), env = env)
  dims <- dim(y_mu)
  if (!is.null(dims) && length(dims) > 0L) {
    return(as.integer(tail(dims, 1L)))
  }
  0L
}

.ndm_spec_parameter_transforms <- function(spec) {
  transforms <- spec$parameter_transforms %||% spec$execution_spec$parameter_transforms %||% character(0)
  if (length(transforms) == 0L) {
    return(character(0))
  }
  transforms <- as.character(transforms)
  if (is.null(names(transforms)) || any(!nzchar(names(transforms)))) {
    names(transforms) <- spec$parameter_terms[seq_along(transforms)]
  }
  transforms
}

.ndm_transform_output_name <- function(transform) {
  switch(
    as.character(transform %||% "Identity"),
    InvSoftPlus = "SoftPlus",
    InvSigmoid = "Sigmoid",
    Identity = "Identity",
    as.character(transform)
  )
}

.ndm_temporal_feature_map <- function(env, pred = NULL) {
  spec <- .ndm_runtime_get0(env, "ndm_model_spec", ifnotfound = NULL)
  model_type <- as.character(.ndm_runtime_get0(
    env,
    "ModelType",
    ifnotfound = .ndm_runtime_get0(env, "ndm_model_type", ifnotfound = NA_character_)
  ))
  has_ode_output <- !is.null(.ndm_prediction_ode_value(pred, "diff_eq_sol_ys.Neural1")) ||
    !is.null(.ndm_prediction_ode_value(pred, "diff_eq_sol_ys.Neural2"))
  has_decoder_output <- {
    temporal <- .ndm_prediction_value(pred, "TemporalLatents")
    is.list(temporal) && "decoder_head_input" %in% names(temporal)
  }
  is_ode <- identical(model_type, "NeuralODE") || has_ode_output
  is_decoder <- identical(model_type, "DecoderOnly") || has_decoder_output

  local_dim <- if (is_ode) {
    .ndm_runtime_int(env, "LocalNeuralEmbedDim", ifnotfound = 0L)
  } else {
    0L
  }
  global_dim <- if (is_ode) {
    .ndm_runtime_int(env, "GlobalNeuralEmbedDim", ifnotfound = 0L)
  } else {
    0L
  }
  n_outcomes <- .ndm_prediction_n_outcomes(env, pred)

  local_terms <- if (is_ode) {
    .ndm_runtime_character(env, "uq_encneural_vec")
  } else {
    character(0)
  }
  global_terms <- if (is_ode) {
    .ndm_runtime_character(env, "uq_globalneural_vec")
  } else {
    character(0)
  }
  if (is_ode && inherits(spec, "ndm_model_spec")) {
    if (length(local_terms) == 0L) {
      local_terms <- spec$local_dynamic_terms %||% character(0)
    }
    if (length(global_terms) == 0L) {
      global_terms <- spec$global_dynamic_terms %||% character(0)
    }
    if (length(local_terms) == 0L && length(global_terms) == 0L) {
      local_terms <- grep("_l$", spec$time_varying_terms %||% character(0), value = TRUE)
      global_terms <- setdiff(spec$time_varying_terms %||% character(0), local_terms)
    }
  }

  transforms <- if (inherits(spec, "ndm_model_spec")) {
    .ndm_spec_parameter_transforms(spec)
  } else {
    character(0)
  }

  make_rows <- function(source, role, feature, index, transform = "Identity") {
    feature <- unname(as.character(feature))
    index <- unname(as.integer(index))
    transform <- unname(as.character(transform))
    data.frame(
      source = rep(source, length(feature)),
      role = rep(role, length(feature)),
      feature = feature,
      index = index,
      transform = transform,
      output = unname(vapply(transform, .ndm_transform_output_name, character(1L))),
      stringsAsFactors = FALSE
    )
  }

  local_rows <- data.frame()
  if (local_dim > 0L) {
    local_rows <- rbind(
      local_rows,
      make_rows(
        source = "Neural1",
        role = "hidden",
        feature = paste0("local_hidden_", seq_len(local_dim)),
        index = seq_len(local_dim)
      )
    )
  }
  if (length(local_terms) > 0L) {
    term_transforms <- transforms[local_terms]
    term_transforms[is.na(term_transforms)] <- "Identity"
    local_rows <- rbind(
      local_rows,
      make_rows(
        source = "Neural1",
        role = "interpretable",
        feature = local_terms,
        index = local_dim + seq_along(local_terms),
        transform = term_transforms
      )
    )
  }
  if (is_ode && n_outcomes > 0L) {
    local_rows <- rbind(
      local_rows,
      make_rows(
        source = "Neural1",
        role = "observation_scale",
        feature = paste0("observation_scale_", seq_len(n_outcomes)),
        index = local_dim + length(local_terms) + seq_len(n_outcomes),
        transform = rep("InvSoftPlus", n_outcomes)
      )
    )
  }
  rownames(local_rows) <- NULL

  global_rows <- data.frame()
  if (global_dim > 0L) {
    global_rows <- rbind(
      global_rows,
      make_rows(
        source = "Neural2",
        role = "hidden",
        feature = paste0("global_hidden_", seq_len(global_dim)),
        index = seq_len(global_dim)
      )
    )
  }
  if (length(global_terms) > 0L) {
    term_transforms <- transforms[global_terms]
    term_transforms[is.na(term_transforms)] <- "Identity"
    global_rows <- rbind(
      global_rows,
      make_rows(
        source = "Neural2",
        role = "interpretable",
        feature = global_terms,
        index = global_dim + seq_along(global_terms),
        transform = term_transforms
      )
    )
  }
  rownames(global_rows) <- NULL

  list(
    decoder = list(
      head_input = {
        decoder_dim <- if (is_decoder) {
          max(0L, .ndm_runtime_int(env, "ModelDims", 0L))
        } else {
          0L
        }
        data.frame(
          source = rep("DecoderOnly", decoder_dim),
          role = rep("head_input", decoder_dim),
          feature = if (decoder_dim > 0L) {
            paste0("decoder_head_input_", seq_len(decoder_dim))
          } else {
            character(0)
          },
          index = seq_len(decoder_dim),
          transform = rep("Identity", decoder_dim),
          output = rep("Identity", decoder_dim),
          stringsAsFactors = FALSE
        )
      }
    ),
    ode = list(
      local = local_rows,
      global = global_rows,
      states = if (is_ode && inherits(spec, "ndm_model_spec")) {
        spec$state_terms %||% character(0)
      } else {
        character(0)
      }
    )
  )
}

.ndm_take_last_dim <- function(x, index, env = NULL, as_r = TRUE) {
  if (is.null(x) || length(index) == 0L) {
    return(NULL)
  }
  if (!isTRUE(as_r) && "python.builtin.object" %in% class(x)) {
    shape <- try(reticulate::py_to_r(x$shape), silent = TRUE)
    if (!inherits(shape, "try-error") && length(shape) > 0L &&
        is.environment(env) && exists("jnp", envir = env, inherits = FALSE)) {
      jnp <- get("jnp", envir = env, inherits = FALSE)
      return(jnp$take(x, jnp$array(as.integer(index) - 1L), axis = as.integer(length(shape) - 1L)))
    }
  }

  x <- .ndm_array_to_r(x, env = env)
  dims <- dim(x)
  if (is.null(dims)) {
    return(x)
  }
  if (length(dims) == 1L) {
    return(x[index])
  }
  if (length(dims) == 2L) {
    return(x[, index, drop = FALSE])
  }
  index_expr <- rep(list(TRUE), length(dims))
  index_expr[[length(dims)]] <- index
  do.call("[", c(list(x), index_expr, list(drop = FALSE)))
}

.ndm_softplus_r <- function(x) {
  log1p(exp(-abs(x))) + pmax(x, 0)
}

.ndm_sigmoid_r <- function(x) {
  1 / (1 + exp(-x))
}

.ndm_apply_feature_transform <- function(x, transform, env = NULL, as_r = TRUE) {
  transform <- as.character(transform %||% "Identity")
  if (identical(transform, "Identity") || is.null(x)) {
    return(x)
  }
  if (!isTRUE(as_r) && "python.builtin.object" %in% class(x)) {
    if (identical(transform, "InvSoftPlus") && is.environment(env) && exists("SoftPlus", envir = env, inherits = FALSE)) {
      return(get("SoftPlus", envir = env, inherits = FALSE)(x))
    }
    if (identical(transform, "InvSigmoid") && is.environment(env) && exists("Sigmoid", envir = env, inherits = FALSE)) {
      return(get("Sigmoid", envir = env, inherits = FALSE)(x))
    }
    return(x)
  }
  x <- .ndm_array_to_r(x, env = env)
  switch(
    transform,
    InvSoftPlus = .ndm_softplus_r(x),
    InvSigmoid = .ndm_sigmoid_r(x),
    x
  )
}

.ndm_drop_last_singleton <- function(x) {
  dims <- dim(x)
  if (is.null(dims) || length(dims) < 2L || tail(dims, 1L) != 1L) {
    return(x)
  }
  dn <- dimnames(x)
  dim(x) <- dims[-length(dims)]
  if (!is.null(dn)) {
    dimnames(x) <- dn[-length(dn)]
  }
  x
}

.ndm_named_feature_arrays <- function(raw, rows, role, env, as_r) {
  rows <- rows[rows$role == role, , drop = FALSE]
  if (nrow(rows) == 0L || is.null(raw)) {
    return(list())
  }
  out <- lapply(seq_len(nrow(rows)), function(i) {
    .ndm_drop_last_singleton(.ndm_apply_feature_transform(
      .ndm_take_last_dim(raw, rows$index[[i]], env = env, as_r = as_r),
      rows$transform[[i]],
      env = env,
      as_r = as_r
    ))
  })
  names(out) <- rows$feature
  out
}

.ndm_set_feature_names <- function(x, feature_names) {
  if (is.null(x) || "python.builtin.object" %in% class(x)) {
    return(x)
  }
  dims <- dim(x)
  if (is.null(dims) || length(dims) == 0L || tail(dims, 1L) != length(feature_names)) {
    return(x)
  }
  dn <- dimnames(x)
  if (is.null(dn)) {
    dn <- vector("list", length(dims))
  }
  dn[[length(dims)]] <- feature_names
  dimnames(x) <- dn
  x
}

.ndm_extract_ode_latents <- function(pred, feature_map, env, as_r = TRUE) {
  local_raw <- .ndm_prediction_ode_value(pred, "diff_eq_sol_ys.Neural1")
  global_raw <- .ndm_prediction_ode_value(pred, "diff_eq_sol_ys.Neural2")
  local_rows <- feature_map$ode$local
  global_rows <- feature_map$ode$global

  if (isTRUE(as_r)) {
    local_raw <- .ndm_array_to_r(local_raw, env = env)
    global_raw <- .ndm_array_to_r(global_raw, env = env)
  }

  local_hidden_rows <- local_rows[local_rows$role == "hidden", , drop = FALSE]
  global_hidden_rows <- global_rows[global_rows$role == "hidden", , drop = FALSE]
  state_terms <- feature_map$ode$states %||% character(0)
  states <- lapply(state_terms, function(term) {
    value <- .ndm_prediction_ode_value(pred, paste0("diff_eq_sol_ys.", term))
    if (isTRUE(as_r)) {
      value <- .ndm_array_to_r(value, env = env)
    }
    value
  })
  names(states) <- state_terms
  states <- states[!vapply(states, is.null, logical(1L))]

  list(
    local_state = .ndm_set_feature_names(local_raw, local_rows$feature),
    global_state = .ndm_set_feature_names(global_raw, global_rows$feature),
    local_hidden = .ndm_set_feature_names(
      .ndm_take_last_dim(local_raw, local_hidden_rows$index, env = env, as_r = as_r),
      local_hidden_rows$feature
    ),
    global_hidden = .ndm_set_feature_names(
      .ndm_take_last_dim(global_raw, global_hidden_rows$index, env = env, as_r = as_r),
      global_hidden_rows$feature
    ),
    features = list(
      local = .ndm_named_feature_arrays(local_raw, local_rows, "interpretable", env = env, as_r = as_r),
      global = .ndm_named_feature_arrays(global_raw, global_rows, "interpretable", env = env, as_r = as_r),
      observation_scale = .ndm_named_feature_arrays(local_raw, local_rows, "observation_scale", env = env, as_r = as_r)
    ),
    states = states
  )
}

#' Extract temporal latent states for interpretability analyses
#'
#' `ndm_extract_temporal_latents()` runs the existing prediction path and
#' returns aligned temporal representations for decoder-only and NeuralODE
#' models. `ndm_latent_matrix()` flattens one extracted representation to a
#' two-dimensional matrix suitable for external analyses such as CCA.
#'
#' @param x Either an `ndm_model`, an `ndm_trained_model`, or a prepared runtime
#'   environment that already contains the required runtime objects.
#' @param batch Optional prediction batch passed through to [ndm_predict()].
#' @param seed Integer seed used to derive the per-example JAX RNG keys.
#' @param inference Logical scalar indicating whether the inference prediction
#'   path or training prediction path should be used.
#' @param update_state Logical scalar indicating whether the runtime `state`
#'   object should be updated with the returned state.
#' @param as_r Logical scalar. When `TRUE`, JAX arrays are copied to R arrays.
#' @param latents An object returned by `ndm_extract_temporal_latents()`.
#' @param source Representation to flatten. Supported values include
#'   `"decoder_head_input"`, `"ode_local_hidden"`, `"ode_global_hidden"`,
#'   `"ode_local_features"`, `"ode_global_features"`, and `"ode_states"`.
#' @param features Optional feature names or integer feature positions.
#'
#' @returns `ndm_extract_temporal_latents()` returns an
#'   `ndm_temporal_latents` list. `ndm_latent_matrix()` returns a numeric matrix
#'   with rows ordered by batch within time.
#'
#' @examples
#' \dontrun{
#' latents <- ndm_extract_temporal_latents(trained, batch = batch_l_cal)
#' decoder_x <- ndm_latent_matrix(latents, "decoder_head_input")
#' ode_x <- ndm_latent_matrix(latents, "ode_local_hidden")
#' }
#'
#' @export
ndm_extract_temporal_latents <- function(x,
                                         batch = NULL,
                                         seed = 1L,
                                         inference = TRUE,
                                         update_state = FALSE,
                                         as_r = TRUE) {
  runtime_env <- .ndm_runtime_env_from_object(x)
  pred <- ndm_predict(
    x,
    batch = batch,
    inference = inference,
    seed = seed,
    update_state = update_state
  )
  feature_map <- .ndm_temporal_feature_map(runtime_env, pred = pred)
  model_type <- .ndm_prediction_model_type(x, runtime_env)

  temporal_latents <- .ndm_prediction_value(pred, "TemporalLatents")
  decoder_head_input <- NULL
  if (is.list(temporal_latents) && "decoder_head_input" %in% names(temporal_latents)) {
    decoder_head_input <- temporal_latents$decoder_head_input
    if (isTRUE(as_r)) {
      decoder_head_input <- .ndm_array_to_r(decoder_head_input, env = runtime_env)
    }
    decoder_features <- feature_map$decoder$head_input$feature
    if (!is.null(dim(decoder_head_input)) && length(decoder_features) == tail(dim(decoder_head_input), 1L)) {
      decoder_head_input <- .ndm_set_feature_names(decoder_head_input, decoder_features)
    }
  }

  ode <- .ndm_extract_ode_latents(pred, feature_map = feature_map, env = runtime_env, as_r = as_r)

  time <- .ndm_prediction_ode_value(pred, "diff_eq_sol_ts")
  if (isTRUE(as_r)) {
    time <- .ndm_array_to_r(time, env = runtime_env)
  }
  if (is.null(time) && !is.null(decoder_head_input)) {
    dims <- dim(decoder_head_input)
    if (!is.null(dims) && length(dims) >= 2L) {
      time <- seq_len(dims[[2L]]) - 1L
    }
  }

  out <- list(
    model_type = model_type,
    time = time,
    decoder = list(head_input = decoder_head_input),
    ode = ode,
    feature_map = feature_map,
    prediction = list(
      y_mu = if (isTRUE(as_r)) .ndm_array_to_r(.ndm_prediction_value(pred, "y_mu"), env = runtime_env) else .ndm_prediction_value(pred, "y_mu"),
      y_sigma = if (isTRUE(as_r)) .ndm_array_to_r(.ndm_prediction_value(pred, "y_sigma"), env = runtime_env) else .ndm_prediction_value(pred, "y_sigma")
    )
  )
  class(out) <- c("ndm_temporal_latents", "list")
  out
}

.ndm_latent_source_object <- function(latents, source) {
  source <- match.arg(
    source,
    c(
      "decoder_head_input",
      "decoder",
      "ode_local_hidden",
      "local_hidden",
      "ode_global_hidden",
      "global_hidden",
      "ode_local_features",
      "local_features",
      "ode_global_features",
      "global_features",
      "ode_observation_scale",
      "observation_scale",
      "ode_states",
      "states"
    )
  )
  switch(
    source,
    decoder_head_input = latents$decoder$head_input,
    decoder = latents$decoder$head_input,
    ode_local_hidden = latents$ode$local_hidden,
    local_hidden = latents$ode$local_hidden,
    ode_global_hidden = latents$ode$global_hidden,
    global_hidden = latents$ode$global_hidden,
    ode_local_features = latents$ode$features$local,
    local_features = latents$ode$features$local,
    ode_global_features = latents$ode$features$global,
    global_features = latents$ode$features$global,
    ode_observation_scale = latents$ode$features$observation_scale,
    observation_scale = latents$ode$features$observation_scale,
    ode_states = latents$ode$states,
    states = latents$ode$states
  )
}

.ndm_feature_list_to_array <- function(x, features = NULL) {
  if (length(x) == 0L) {
    stop("No features are available for this latent source.", call. = FALSE)
  }
  if (!is.null(features)) {
    if (is.character(features)) {
      missing <- setdiff(features, names(x))
      if (length(missing) > 0L) {
        stop("Unknown feature(s): ", paste(missing, collapse = ", "), call. = FALSE)
      }
      x <- x[features]
    } else {
      x <- x[features]
    }
  }
  dims <- dim(x[[1L]])
  if (is.null(dims)) {
    dims <- length(x[[1L]])
  }
  out <- array(NA_real_, dim = c(dims, length(x)))
  for (i in seq_along(x)) {
    out_slice <- x[[i]]
    dim(out_slice) <- dims
    selector <- c(rep(list(TRUE), length(dims)), list(i))
    out <- do.call("[<-", c(list(out), selector, list(value = out_slice)))
  }
  dn <- vector("list", length(dim(out)))
  dn[[length(dim(out))]] <- names(x)
  dimnames(out) <- dn
  out
}

.ndm_array_select_features <- function(x, features = NULL) {
  if (is.null(features)) {
    return(x)
  }
  dims <- dim(x)
  if (is.null(dims) || length(dims) < 2L) {
    stop("Selected latent source is not feature-indexed.", call. = FALSE)
  }
  if (is.character(features)) {
    feature_names <- dimnames(x)[[length(dims)]]
    if (is.null(feature_names)) {
      stop("This latent source does not have feature names; use integer `features`.", call. = FALSE)
    }
    features <- match(features, feature_names)
    if (anyNA(features)) {
      stop("Unknown feature name in `features`.", call. = FALSE)
    }
  }
  index_expr <- rep(list(TRUE), length(dims))
  index_expr[[length(dims)]] <- features
  do.call("[", c(list(x), index_expr, list(drop = FALSE)))
}

.ndm_flatten_latent_array <- function(x) {
  if ("python.builtin.object" %in% class(x)) {
    stop("`ndm_latent_matrix()` requires R arrays. Re-run extraction with `as_r = TRUE`.", call. = FALSE)
  }
  dims <- dim(x)
  if (is.null(dims)) {
    return(matrix(as.numeric(x), ncol = 1L))
  }
  if (length(dims) == 1L) {
    return(matrix(as.numeric(x), ncol = 1L))
  }
  if (length(dims) == 2L) {
    mat <- matrix(as.numeric(x), nrow = dims[[1L]], ncol = dims[[2L]])
    colnames(mat) <- dimnames(x)[[2L]]
    return(mat)
  }
  feature_names <- dimnames(x)[[length(dims)]]
  feature_dim <- dims[[length(dims)]]
  mat <- matrix(as.numeric(x), nrow = prod(dims[-length(dims)]), ncol = feature_dim)
  if (!is.null(feature_names) && length(feature_names) == feature_dim) {
    colnames(mat) <- feature_names
  }
  mat
}

#' @rdname ndm_extract_temporal_latents
#' @export
ndm_latent_matrix <- function(latents,
                              source = "decoder_head_input",
                              features = NULL) {
  if (!inherits(latents, "ndm_temporal_latents")) {
    stop("`latents` must be returned by ndm_extract_temporal_latents().", call. = FALSE)
  }
  x <- .ndm_latent_source_object(latents, source)
  if (is.null(x)) {
    stop("The requested latent source is not available.", call. = FALSE)
  }
  if (is.list(x) && !is.data.frame(x)) {
    x <- .ndm_feature_list_to_array(x, features = features)
  } else {
    x <- .ndm_array_select_features(x, features = features)
  }
  .ndm_flatten_latent_array(x)
}

#' @rdname ndm_build_model
#' @export
ndm_train <- function(x,
                      run_define = TRUE,
                      run_loop = TRUE) {
  runtime_env <- .ndm_runtime_env_from_object(x)
  required_packages <- "rrapply"
  if (isTRUE(.ndm_runtime_get0(runtime_env, "nCheckpoints", ifnotfound = 0L) > 0L)) {
    required_packages <- c(required_packages, "zip")
  }
  if (identical(.ndm_runtime_get0(runtime_env, "ndm_data_generator", ifnotfound = NULL), "multidisease")) {
    required_packages <- c(required_packages, "zoo")
  }
  .ndm_require_namespaces(required_packages, context = "ndm_train()")

  .ndm_install_runtime_helpers(runtime_env)
  paths <- ndm_runtime_paths()

  .ndm_require_runtime_bindings(
    runtime_env,
    c("ModelList"),
    context = "training setup"
  )

  project_root <- .ndm_runtime_project_root(runtime_env)
  run_training <- function() {
    .ndm_prepare_train_environment(runtime_env)

    if (isTRUE(run_define)) {
      .ndm_source_runtime_file(paths$train_define, runtime_env)
    }

    if (isTRUE(run_loop)) {
      .ndm_source_runtime_file(paths$train_do, runtime_env)
    }
  }
  if (is.null(project_root)) {
    run_training()
  } else {
    .ndm_with_working_directory(project_root, run_training)
  }

  if (isTRUE(run_loop)) {
    .ndm_require_runtime_bindings(
      runtime_env,
      c("state", "PriorList", "PolicyList", "GetPredSaveAtInfo_default", "gradLoss_jax"),
      context = "training loop"
    )
  }

  structure(
    list(
      env = runtime_env,
      config = .ndm_runtime_get0(runtime_env, "ndm_config", ifnotfound = NULL),
      model_spec = .ndm_runtime_get0(runtime_env, "ndm_model_spec", ifnotfound = NULL),
      model_type = .ndm_runtime_get0(runtime_env, "ndm_model_type", ifnotfound = NULL),
      backbone = .ndm_runtime_get0(runtime_env, "ndm_backbone", ifnotfound = NULL),
      data_generator = .ndm_runtime_get0(runtime_env, "ndm_data_generator", ifnotfound = NULL),
      model = if (exists("ModelList", envir = runtime_env, inherits = FALSE)) {
        get("ModelList", envir = runtime_env, inherits = FALSE)
      } else {
        NULL
      },
      state = if (exists("state", envir = runtime_env, inherits = FALSE)) {
        get("state", envir = runtime_env, inherits = FALSE)
      } else {
        NULL
      },
      opt_state = if (exists("opt_state", envir = runtime_env, inherits = FALSE)) {
        get("opt_state", envir = runtime_env, inherits = FALSE)
      } else {
        NULL
      }
    ),
    class = "ndm_trained_model"
  )
}

#' @rdname ndm_build_model
#'
#' @returns `print.ndm_trained_model()` prints a brief summary of an
#'   `ndm_trained_model` object and invisibly returns `x`.
#'
#' @export
print.ndm_trained_model <- function(x, ...) {
  cat("<ndm_trained_model>\n")
  cat(sprintf("  has_model: %s\n", !is.null(x$model)))
  cat(sprintf("  has_state: %s\n", !is.null(x$state)))
  cat(sprintf("  has_opt_state: %s\n", !is.null(x$opt_state)))
  invisible(x)
}

#' @rdname ndm_build_model
#' @export
ndm_fit <- function(config = ndm_create_config(),
                    model_spec = NULL,
                    data_generator = c("sim", "real", "multidisease"),
                    runtime_globals = list(),
                    data_globals = list(),
                    build_globals = list(),
                    train_globals = list(),
                    run_define = TRUE,
                    run_loop = TRUE) {
  if (!inherits(config, "ndm_config")) {
    stop("`config` must inherit from class 'ndm_config'.", call. = FALSE)
  }

  data_generator <- match.arg(data_generator)
  runtime_env <- ndm_prepare_runtime(
    config = config,
    runtime_globals = runtime_globals
  )
  if (!is.null(model_spec)) {
    ndm_set_runtime_globals(
      runtime_env,
      list(
        ndm_model_spec = model_spec
      )
    )
  }
  ndm_prepare_data(
    runtime_env = runtime_env,
    generator = data_generator,
    runtime_globals = data_globals
  )
  model <- ndm_build_model(
    runtime_env = runtime_env,
    model_type = config$model_type,
    model_spec = model_spec,
    backbone = config$backbone,
    runtime_globals = build_globals
  )
  if (length(train_globals) > 0L) {
    ndm_set_runtime_globals(model$env, train_globals)
  }
  ndm_train(
    model,
    run_define = run_define,
    run_loop = run_loop
  )
}
