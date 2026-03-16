#' Prepare a local runtime for execution
#'
#' These wrappers load package-owned runtime components into an isolated
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

  ndm_set_runtime_globals(runtime_env, as.list(config))
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
#' @param generator Which data generator to source into `runtime_env`.
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
                             generator = c("sim", "real", "multidisease"),
                             runtime_globals = list()) {
  if (!is.environment(runtime_env)) {
    stop("`runtime_env` must be an environment.", call. = FALSE)
  }

  generator <- match.arg(generator)
  ndm_set_runtime_globals(runtime_env, runtime_globals)
  if (identical(generator, "sim")) {
    .ndm_require_namespaces(
      c("progress", "zoo"),
      context = "ndm_prepare_data(generator = 'sim')"
    )
  }
  if (identical(generator, "multidisease")) {
    .ndm_prepare_multidisease_data(runtime_env)
  } else {
    ndm_source_runtime_data(
      env = runtime_env,
      generator = generator
    )
  }
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
    return(list(raw = batch, packaged = ndm_batch_to_model_inputs(batch)))
  }

  if (.ndm_is_packaged_prediction_batch(batch)) {
    return(list(raw = batch, packaged = batch))
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
#' @param data_generator Which local data generator to source before calling
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

  structure(
    list(
      env = runtime_env,
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

  pred_fun_name <- if (isTRUE(inference)) "GetPred_inference" else "GetPred_train_jit"
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
      tryCatch(
        .ndm_source_runtime_file(paths$train_define, runtime_env),
        error = function(e) {
          stop("Failed while loading package-owned training definition code: ", conditionMessage(e), call. = FALSE)
        }
      )
    }

    if (isTRUE(run_loop)) {
      tryCatch(
        .ndm_source_runtime_file(paths$train_do, runtime_env),
        error = function(e) {
          stop("Failed while loading package-owned training loop code: ", conditionMessage(e), call. = FALSE)
        }
      )
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
