#' Prepare the vendored runtime for execution
#'
#' These wrappers load the extracted runtime scripts into an isolated
#' environment and then source the selected data generator.
#'
#' @param config An object of class `ndm_config`, usually created by
#'   `ndm_create_config()`.
#' @param runtime_env Runtime environment that should receive the sourced
#'   objects.
#' @param runtime_globals Named list of additional bindings to assign before the
#'   runtime code is sourced.
#'
#' @returns `ndm_prepare_runtime()` invisibly returns `runtime_env` after
#'   loading the vendored helper and backend code.
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

  ndm_set_runtime_globals(runtime_env, as.list(config))
  ndm_set_runtime_globals(runtime_env, runtime_globals)
  ndm_load_runtime(
    analysis_root = config$analysis_root,
    env = runtime_env,
    float_type = config$float_type,
    force_to_gpu = config$force_to_gpu,
    gpu_mem_frac = config$gpu_mem_frac,
    resave_tfrecords = config$resave_tfrecords
  )
}

#' @rdname ndm_prepare_runtime
#'
#' @param analysis_root Root directory for the legacy analysis runtime. The
#'   current implementation resolves runtime files from the vendored package
#'   bundle.
#' @param generator Which extracted data generator to source into `runtime_env`.
#'
#' @returns `ndm_prepare_data()` invisibly returns `runtime_env` after sourcing
#'   the requested data generator.
#'
#' @examples
#' env <- ndm_new_runtime_env()
#' ndm_set_runtime_globals(env, list(example_value = 1L))
#'
#' @export
ndm_prepare_data <- function(runtime_env,
                             analysis_root = .ndm_default_analysis_root(),
                             generator = c("sim", "real"),
                             runtime_globals = list()) {
  if (!is.environment(runtime_env)) {
    stop("`runtime_env` must be an environment.", call. = FALSE)
  }

  generator <- match.arg(generator)
  ndm_set_runtime_globals(runtime_env, runtime_globals)
  ndm_source_runtime_data(
    analysis_root = analysis_root,
    env = runtime_env,
    generator = generator
  )
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
    return(env$jnp$array(reticulate::py_to_r(x)))
  }
  x
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
    return(list(raw = batch, packaged = env$batch2package(batch)))
  }

  if (is.list(batch) && length(batch) == 4L && is.list(batch[[1]])) {
    return(list(raw = batch, packaged = batch))
  }

  stop(
    "`batch` must be either a named TFRecord-style batch list or a packaged model-input list from ndm_batch_to_model_inputs().",
    call. = FALSE
  )
}

#' Build and train models with the vendored runtime
#'
#' These wrappers layer a small R API over the extracted Phase 1 model build and
#' training scripts.
#'
#' @param config An object of class `ndm_config`, usually created by
#'   `ndm_create_config()`.
#' @param runtime_env Runtime environment containing the sourced legacy helper
#'   code and data globals.
#' @param analysis_root Root directory for the legacy analysis runtime. The
#'   current implementation resolves runtime files from the vendored package
#'   bundle.
#' @param model_type Model family to build. Either `"DecoderOnly"` or
#'   `"NeuralODE"`.
#' @param model_spec Optional `ndm_model_spec` object used to override the model
#'   TeX specification supplied to the vendored builder.
#' @param backbone Backbone family. Phase 1 supports `"transformer"` only.
#' @param runtime_globals Named list of additional globals assigned during the
#'   relevant runtime stage. `ndm_build_model()` uses them before building the
#'   model, while `ndm_fit()` uses them before sourcing the vendored runtime.
#' @param x Either an `ndm_model`, an `ndm_trained_model`, or a prepared runtime
#'   environment, depending on the function being called.
#' @param run_define Logical scalar indicating whether the training definition
#'   script should be sourced.
#' @param run_loop Logical scalar indicating whether the training loop script
#'   should be sourced.
#' @param data_generator Which extracted data generator to source before calling
#'   `ndm_build_model()` inside `ndm_fit()`.
#' @param data_globals Named list of globals assigned before sourcing the data
#'   generator in `ndm_fit()`.
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
                            analysis_root = .ndm_default_analysis_root(),
                            model_type = c("DecoderOnly", "NeuralODE"),
                            model_spec = NULL,
                            backbone = "transformer",
                            runtime_globals = list()) {
  if (!is.environment(runtime_env)) {
    stop("`runtime_env` must be an environment returned by ndm_load_runtime() or prepared manually.", call. = FALSE)
  }

  .ndm_install_runtime_helpers(runtime_env)
  model_type <- match.arg(model_type)
  if (!identical(backbone, "transformer")) {
    stop("Phase 1 only supports backbone = 'transformer'.", call. = FALSE)
  }

  paths <- ndm_runtime_paths(analysis_root)
  ndm_set_runtime_globals(runtime_env, runtime_globals)
  ndm_set_runtime_globals(
    runtime_env,
    list(
      ModelType = model_type,
      BackboneType = backbone,
      UseLatentAttention = FALSE
    )
  )

  cleanup <- FALSE
  if (!is.null(model_spec)) {
    materialized <- .ndm_materialize_model_spec(model_spec)
    cleanup <- isTRUE(materialized$cleanup)
    runtime_env$model_tex_loc <- materialized$path
    if (cleanup) {
      on.exit(unlink(materialized$path), add = TRUE)
    }
  }

  tryCatch(
    .ndm_source_runtime_file(paths$build_model, runtime_env),
    error = function(e) {
      stop(
        "Failed to source the packaged model builder. This assumes the upstream setup and data globals are already present in `runtime_env`.\n",
        "Original error: ", conditionMessage(e),
        call. = FALSE
      )
    }
  )

  structure(
    list(
      env = runtime_env,
      analysis_root = paths$analysis_root,
      model_type = model_type,
      backbone = backbone,
      model_spec = model_spec,
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
#' These helpers call the vendored prediction and loss functions stored in a
#' prepared runtime environment.
#'
#' @param x Either an `ndm_model`, an `ndm_trained_model`, or a prepared runtime
#'   environment that already contains the required vendored objects.
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
#' @param iteration Training iteration number forwarded to the vendored loss
#'   function.
#'
#' @returns `ndm_predict()` returns the prediction object produced by the
#'   vendored runtime. `ndm_loss()` returns the loss object produced by the
#'   vendored runtime.
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
  runtime_env <- if (inherits(x, "ndm_model")) x$env else x
  if (!is.environment(runtime_env)) {
    stop("`x` must be an `ndm_model` or a prepared runtime environment.", call. = FALSE)
  }
  if (!exists("ModelList", envir = runtime_env, inherits = FALSE)) {
    stop("`ModelList` is not available in the supplied environment. Build the model first.", call. = FALSE)
  }

  pred_fun_name <- if (isTRUE(inference)) "GetPred_inference" else "GetPred_train_jit"
  if (!exists(pred_fun_name, envir = runtime_env, inherits = FALSE)) {
    stop("Prediction function `", pred_fun_name, "` is not available. Build the model first.", call. = FALSE)
  }

  prepared_batch <- .ndm_prepare_prediction_batch(batch, env = runtime_env)
  batch_size <- .ndm_batch_size_from_object(prepared_batch$raw)
  seed_matrix <- runtime_env$jax$random$split(runtime_env$JaxKey(as.integer(seed)), as.integer(batch_size))

  pred_fun <- get(pred_fun_name, envir = runtime_env, inherits = FALSE)
  result <- pred_fun(
    runtime_env$ModelList,
    prepared_batch$packaged,
    runtime_env$state,
    runtime_env$PriorList,
    runtime_env$PolicyList,
    runtime_env$GetPredSaveAtInfo_default,
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
  runtime_env <- if (inherits(x, "ndm_model")) x$env else x
  if (!is.environment(runtime_env)) {
    stop("`x` must be an `ndm_model` or a prepared runtime environment.", call. = FALSE)
  }
  if (!exists("getLoss_train", envir = runtime_env, inherits = FALSE)) {
    stop("Loss function `getLoss_train` is not available. Build the model first.", call. = FALSE)
  }

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

  result <- runtime_env$getLoss_train(
    runtime_env$ModelList,
    prepared_batch$packaged,
    y,
    y_mask,
    runtime_env$jnp$array(as.numeric(iteration)),
    runtime_env$state,
    runtime_env$PriorList,
    runtime_env$PolicyList,
    runtime_env$GetPredSaveAtInfo_default,
    seed_matrix
  )

  if (isTRUE(update_state)) {
    assign("state", result[[2]], envir = runtime_env)
  }

  result[[1]]
}

#' @rdname ndm_build_model
#' @export
ndm_train <- function(x,
                      analysis_root = NULL,
                      run_define = TRUE,
                      run_loop = TRUE) {
  runtime_env <- if (inherits(x, "ndm_model")) x$env else x
  if (!is.environment(runtime_env)) {
    stop("`x` must be an `ndm_model` or a prepared runtime environment.", call. = FALSE)
  }

  .ndm_install_runtime_helpers(runtime_env)
  analysis_root <- analysis_root %||% if (inherits(x, "ndm_model")) x$analysis_root else .ndm_default_analysis_root()
  paths <- ndm_runtime_paths(analysis_root)

  if (isTRUE(run_define)) {
    tryCatch(
      .ndm_source_runtime_file(paths$train_define, runtime_env),
      error = function(e) {
        stop("Failed while sourcing packaged training definition code: ", conditionMessage(e), call. = FALSE)
      }
    )
  }

  if (isTRUE(run_loop)) {
    tryCatch(
      .ndm_source_runtime_file(paths$train_do, runtime_env),
      error = function(e) {
        stop("Failed while sourcing packaged training loop code: ", conditionMessage(e), call. = FALSE)
      }
    )
  }

  structure(
    list(
      env = runtime_env,
      analysis_root = paths$analysis_root,
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
                    data_generator = c("sim", "real"),
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
  ndm_prepare_data(
    runtime_env = runtime_env,
    analysis_root = config$analysis_root,
    generator = data_generator,
    runtime_globals = data_globals
  )
  model <- ndm_build_model(
    runtime_env = runtime_env,
    analysis_root = config$analysis_root,
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
    analysis_root = config$analysis_root,
    run_define = run_define,
    run_loop = run_loop
  )
}
