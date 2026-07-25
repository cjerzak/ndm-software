ndm_env <- new.env(parent = emptyenv())
ndm_env$backend <- NULL

`%||%` <- function(x, y) {
  if (is.null(x)) {
    return(y)
  }
  x
}

#' Print a timestamped diagnostic message
#'
#' `ndm_print()` is a small logging helper used throughout the package-facing
#' orchestration code.
#'
#' @param text Text to print.
#' @param quiet Logical scalar indicating whether output should be suppressed.
#'
#' @returns The input `text`, invisibly.
#'
#' @examples
#' ndm_print("starting")
#'
#' @importFrom bestNormalize yeojohnson
#' @export
ndm_print <- function(text, quiet = FALSE) {
  if (!quiet) {
    message(sprintf("[%s] %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), text))
  }
  invisible(text)
}

.ndm_default_analysis_root <- function() {
  .ndm_internal_analysis_root()
}

.ndm_normalize_path <- function(path, must_work = TRUE) {
  normalizePath(path, winslash = "/", mustWork = must_work)
}

.ndm_is_absolute_path <- function(path) {
  if (length(path) != 1L || is.na(path) || !nzchar(path)) {
    return(FALSE)
  }
  grepl("^(/|[A-Za-z]:[/\\\\]|[/\\\\]{2})", path)
}

.ndm_path_join_if_relative <- function(root, path) {
  if (.ndm_is_absolute_path(path) || startsWith(path, "~")) {
    return(path)
  }
  file.path(root, path)
}

.ndm_package_root <- function() {
  pkg_root <- system.file(package = "ndm")
  if (nzchar(pkg_root)) {
    return(pkg_root)
  }

  candidates <- unique(c(
    getwd(),
    normalizePath(file.path(getwd(), ".."), winslash = "/", mustWork = FALSE),
    normalizePath(file.path(getwd(), "..", ".."), winslash = "/", mustWork = FALSE)
  ))

  for (candidate in candidates) {
    if (!nzchar(candidate)) {
      next
    }
    if (file.exists(file.path(candidate, "DESCRIPTION")) &&
        dir.exists(file.path(candidate, "R"))) {
      return(.ndm_normalize_path(candidate, must_work = TRUE))
    }
  }

  stop("Could not determine the ndm package root.", call. = FALSE)
}

.ndm_inst_path <- function(...) {
  pkg_root <- system.file(package = "ndm")
  if (nzchar(pkg_root)) {
    return(file.path(pkg_root, ...))
  }
  file.path(.ndm_package_root(), "inst", ...)
}

.ndm_resolve_analysis_root <- function(analysis_root = .ndm_default_analysis_root(),
                                       must_work = TRUE) {
  if (is.null(analysis_root) || !nzchar(analysis_root)) {
    return(.ndm_default_analysis_root())
  }

  .ndm_normalize_path(analysis_root, must_work = must_work)
}

.ndm_make_classed_list <- function(x, class_name) {
  stopifnot(is.list(x))
  structure(x, class = c(class_name, "list"))
}

.ndm_materialize_neuralode_runtime_config <- function(runtime_env) {
  stopifnot(is.environment(runtime_env))
  if (!exists("diffrax", envir = runtime_env, inherits = FALSE)) {
    return(invisible(runtime_env))
  }

  runtime_get0 <- function(name, ifnotfound = NULL) {
    get0(name, envir = runtime_env, inherits = FALSE, ifnotfound = ifnotfound)
  }

  if (!exists("VI_diff_eq_solver_optim", envir = runtime_env, inherits = FALSE)) {
    solver_name <- tolower(as.character(runtime_get0("neuralode_optim_solver", ifnotfound = "tsit5")))
    solver <- switch(
      solver_name,
      tsit5 = runtime_env$diffrax$Tsit5(),
      dopri8 = runtime_env$diffrax$Dopri8(),
      stop(
        "`neuralode_optim_solver` must be one of 'tsit5' or 'dopri8'.",
        call. = FALSE
      )
    )
    assign("VI_diff_eq_solver_optim", solver, envir = runtime_env)
  }

  if (!exists("dt0_init_optim", envir = runtime_env, inherits = FALSE)) {
    dt0_init_optim <- suppressWarnings(as.numeric(runtime_get0("neuralode_optim_dt0", ifnotfound = 1e-3)))
    if (length(dt0_init_optim) != 1L || !is.finite(dt0_init_optim)) {
      stop("`neuralode_optim_dt0` must be a finite numeric scalar.", call. = FALSE)
    }
    assign("dt0_init_optim", dt0_init_optim[[1L]], envir = runtime_env)
  }

  if (!exists("stepsize_controller_optim", envir = runtime_env, inherits = FALSE)) {
    controller_name <- tolower(
      as.character(
        runtime_get0(
          "neuralode_optim_controller",
          ifnotfound = "pid"
        )
      )
    )
    if (length(controller_name) != 1L || !nzchar(controller_name)) {
      stop("`neuralode_optim_controller` must be a non-empty scalar.", call. = FALSE)
    }

    controller <- switch(
      controller_name,
      pid = {
        rtol <- suppressWarnings(as.numeric(runtime_get0("neuralode_optim_rtol", ifnotfound = 1e-5)))
        atol <- suppressWarnings(as.numeric(runtime_get0("neuralode_optim_atol", ifnotfound = 1e-7)))
        if (length(rtol) != 1L || !is.finite(rtol) || length(atol) != 1L || !is.finite(atol)) {
          stop("`neuralode_optim_rtol` and `neuralode_optim_atol` must be finite numeric scalars.", call. = FALSE)
        }
        runtime_env$diffrax$PIDController(rtol = rtol[[1L]], atol = atol[[1L]])
      },
      constant = runtime_env$diffrax$ConstantStepSize(),
      stop(
        "`neuralode_optim_controller` must be one of 'pid' or 'constant'.",
        call. = FALSE
      )
    )
    assign("stepsize_controller_optim", controller, envir = runtime_env)
  }

  if (!exists("InitStateLogitOffset", envir = runtime_env, inherits = FALSE)) {
    init_state_logit_offset <- runtime_get0("neuralode_init_state_logit_offset", ifnotfound = NULL)
    if (!is.null(init_state_logit_offset)) {
      assign("InitStateLogitOffset", init_state_logit_offset, envir = runtime_env)
    }
  }

  if (!exists("InitStateLogitScaleMax", envir = runtime_env, inherits = FALSE)) {
    init_state_logit_scale_max <- runtime_get0("neuralode_init_state_logit_scale_max", ifnotfound = Inf)
    assign("InitStateLogitScaleMax", init_state_logit_scale_max, envir = runtime_env)
  }

  invisible(runtime_env)
}

#' Create runtime configuration objects
#'
#' Configuration objects collect the runtime options that are threaded through
#' the higher-level orchestration helpers such as `ndm_prepare_runtime()` and
#' `ndm_fit()`.
#'
#' @param model_type Model family metadata. Use `"DecoderOnly"` or
#'   `"NeuralODE"`.
#' @param backbone Backbone family. Phase 1 supports `"transformer"` only.
#' @param float_type Floating point precision used when initializing the backend.
#'   Use `"32"` or `"64"`.
#' @param force_to_gpu Logical scalar requiring a CUDA-capable JAX GPU when
#'   `TRUE`.
#' @param resave_tfrecords Logical scalar preserved for compatibility. Public
#'   training APIs require `FALSE`; prepare canonical inputs before training.
#' @param gpu_mem_frac Optional finite GPU memory fraction in `(0, 1]`.
#' @param neuralode_local_latent_dim Optional local NeuralODE latent width. When
#'   `NULL`, runtime code resolves this to `ModelDims`.
#' @param neuralode_global_latent_dim Optional global NeuralODE latent width.
#'   When `NULL`, runtime code resolves this to `ModelDims`.
#' @param neuralode_init_state_logit_offset Optional init-state logit offset
#'   vector for the NeuralODE state simplex initializer.
#' @param neuralode_init_state_logit_scale_max Scalar cap applied to the
#'   NeuralODE init-state logit scale.
#' @param neuralode_optim_solver NeuralODE optimization solver name. Use
#'   `"tsit5"` or `"dopri8"`.
#' @param neuralode_optim_dt0 Initial step size used by the NeuralODE
#'   optimization solve.
#' @param neuralode_optim_controller NeuralODE optimization stepsize controller.
#'   Use `"pid"` or `"constant"`.
#' @param neuralode_optim_rtol Relative tolerance used when
#'   `neuralode_optim_controller = "pid"`.
#' @param neuralode_optim_atol Absolute tolerance used when
#'   `neuralode_optim_controller = "pid"`.
#' @param enable_kv_cache Logical scalar controlling DecoderOnly KV caching.
#'   The corrected default is `FALSE`; opt in only when the cached and uncached
#'   paths have been verified equivalent for the active padding layout.
#' @param inference_mc_draws Positive integer number of posterior draws
#'   requested for inference. Deterministic NeuralODE and DecoderOnly inference
#'   reduce this to one effective draw downstream.
#' @param observation_scale_floor Finite positive lower bound added to learned
#'   observation scales.
#' @param initial_observation_scale Finite positive initial observation-scale
#'   value on the outcome scale. The default is `1`, matching the unit-scale
#'   initialization used before this control became configurable. It must be
#'   greater than `observation_scale_floor`.
#' @param neuralode_variational Logical scalar. When `FALSE`, NeuralODE
#'   training uses deterministic posterior means instead of stochastic
#'   variational samples.
#' @param neuralode_kl_weight Non-negative scalar KL multiplier used only when
#'   `neuralode_variational = TRUE`.
#' @param neuralode_mean_loss_weight Non-negative multiplier for an auxiliary
#'   masked mean-squared-error term applied to NeuralODE forecasts alongside
#'   the Student-t likelihood.
#' @param ... Additional named values appended to the configuration object.
#'
#' @returns `ndm_create_config()` returns an object of class `ndm_config`.
#'
#' @examples
#' cfg <- ndm_create_config()
#' print(cfg)
#'
#' @export
ndm_create_config <- function(model_type = c("DecoderOnly", "NeuralODE"),
                              backbone = "transformer",
                              float_type = c("32", "64"),
                              force_to_gpu = TRUE,
                              resave_tfrecords = FALSE,
                              gpu_mem_frac = NULL,
                              neuralode_local_latent_dim = NULL,
                              neuralode_global_latent_dim = NULL,
                              neuralode_init_state_logit_offset = NULL,
                              neuralode_init_state_logit_scale_max = Inf,
                              neuralode_optim_solver = c("tsit5", "dopri8"),
                              neuralode_optim_dt0 = 1e-3,
                              neuralode_optim_controller = c("pid", "constant"),
                              neuralode_optim_rtol = 1e-5,
                              neuralode_optim_atol = 1e-7,
                              enable_kv_cache = FALSE,
                              inference_mc_draws = 5L,
                              observation_scale_floor = 1e-5,
                              initial_observation_scale = 1.0,
                              neuralode_variational = TRUE,
                              neuralode_kl_weight = 1.0,
                              neuralode_mean_loss_weight = 0.0,
                              ...) {
  model_type <- match.arg(model_type)
  float_type <- match.arg(float_type)
  neuralode_optim_solver <- match.arg(neuralode_optim_solver)
  neuralode_optim_controller <- match.arg(neuralode_optim_controller)
  if (!identical(backbone, "transformer")) {
    stop("Phase 1 only supports backbone = 'transformer'.", call. = FALSE)
  }
  if (!is.logical(force_to_gpu) || length(force_to_gpu) != 1L || is.na(force_to_gpu)) {
    stop("`force_to_gpu` must be one non-missing logical value.", call. = FALSE)
  }
  if (!is.logical(enable_kv_cache) || length(enable_kv_cache) != 1L || is.na(enable_kv_cache)) {
    stop("`enable_kv_cache` must be one non-missing logical value.", call. = FALSE)
  }
  if (!is.logical(neuralode_variational) ||
      length(neuralode_variational) != 1L ||
      is.na(neuralode_variational)) {
    stop("`neuralode_variational` must be one non-missing logical value.", call. = FALSE)
  }
  .ndm_validate_resave_tfrecords("generic", resave_tfrecords)
  if (!is.null(gpu_mem_frac)) {
    gpu_mem_frac <- suppressWarnings(as.numeric(gpu_mem_frac))
    if (length(gpu_mem_frac) != 1L || !is.finite(gpu_mem_frac) ||
        gpu_mem_frac <= 0 || gpu_mem_frac > 1) {
      stop("`gpu_mem_frac` must be NULL or one finite value in (0, 1].", call. = FALSE)
    }
  }
  inference_mc_draws <- suppressWarnings(as.numeric(inference_mc_draws))
  if (length(inference_mc_draws) != 1L || !is.finite(inference_mc_draws) ||
      inference_mc_draws < 1 || inference_mc_draws > .Machine$integer.max ||
      inference_mc_draws != floor(inference_mc_draws)) {
    stop("`inference_mc_draws` must be one positive integer.", call. = FALSE)
  }
  inference_mc_draws <- as.integer(inference_mc_draws)
  observation_scale_floor <- suppressWarnings(as.numeric(observation_scale_floor))
  if (length(observation_scale_floor) != 1L ||
      !is.finite(observation_scale_floor) || observation_scale_floor <= 0) {
    stop("`observation_scale_floor` must be one finite positive numeric scalar.", call. = FALSE)
  }
  initial_observation_scale <- suppressWarnings(as.numeric(initial_observation_scale))
  if (length(initial_observation_scale) != 1L ||
      !is.finite(initial_observation_scale) || initial_observation_scale <= 0) {
    stop("`initial_observation_scale` must be one finite positive numeric scalar.", call. = FALSE)
  }
  if (initial_observation_scale <= observation_scale_floor) {
    stop(
      "`initial_observation_scale` must be greater than `observation_scale_floor`.",
      call. = FALSE
    )
  }
  neuralode_kl_weight <- suppressWarnings(as.numeric(neuralode_kl_weight))
  if (length(neuralode_kl_weight) != 1L || !is.finite(neuralode_kl_weight) || neuralode_kl_weight < 0) {
    stop("`neuralode_kl_weight` must be one finite non-negative numeric scalar.", call. = FALSE)
  }
  neuralode_mean_loss_weight <- suppressWarnings(as.numeric(neuralode_mean_loss_weight))
  if (length(neuralode_mean_loss_weight) != 1L ||
      !is.finite(neuralode_mean_loss_weight) || neuralode_mean_loss_weight < 0) {
    stop("`neuralode_mean_loss_weight` must be one finite non-negative numeric scalar.", call. = FALSE)
  }

  extras <- list(...)
  config <- list(
    model_type = model_type,
    backbone = backbone,
    float_type = float_type,
    force_to_gpu = isTRUE(force_to_gpu),
    resave_tfrecords = isTRUE(resave_tfrecords),
    gpu_mem_frac = gpu_mem_frac,
    neuralode_local_latent_dim = neuralode_local_latent_dim,
    neuralode_global_latent_dim = neuralode_global_latent_dim,
    neuralode_init_state_logit_offset = neuralode_init_state_logit_offset,
    neuralode_init_state_logit_scale_max = neuralode_init_state_logit_scale_max,
    neuralode_optim_solver = neuralode_optim_solver,
    neuralode_optim_dt0 = neuralode_optim_dt0,
    neuralode_optim_controller = neuralode_optim_controller,
    neuralode_optim_rtol = neuralode_optim_rtol,
    neuralode_optim_atol = neuralode_optim_atol,
    enable_kv_cache = enable_kv_cache,
    inference_mc_draws = inference_mc_draws,
    observation_scale_floor = observation_scale_floor,
    initial_observation_scale = initial_observation_scale,
    neuralode_variational = neuralode_variational,
    neuralode_kl_weight = neuralode_kl_weight,
    neuralode_mean_loss_weight = neuralode_mean_loss_weight
  )

  config <- c(config, extras)
  .ndm_make_classed_list(config, "ndm_config")
}

#' @rdname ndm_create_config
#'
#' @returns `print.ndm_config()` prints a brief summary of an `ndm_config`
#'   object and invisibly returns `x`.
#'
#' @param x An `ndm_config` object.
#' @param ... Unused; included for S3 method compatibility.
#'
#' @export
print.ndm_config <- function(x, ...) {
  cat("<ndm_config>\n")
  cat(sprintf("  model_type: %s\n", x$model_type %||% NA_character_))
  cat(sprintf("  backbone: %s\n", x$backbone %||% NA_character_))
  cat(sprintf("  float_type: %s\n", x$float_type %||% NA_character_))
  invisible(x)
}

#' @rdname ndm_model_spec_presets
#'
#' @returns `print.ndm_model_spec()` prints a brief summary of an
#'   `ndm_model_spec` object and invisibly returns `x`.
#'
#' @param x An `ndm_model_spec` object.
#' @param ... Unused; included for S3 method compatibility.
#'
#' @export
print.ndm_model_spec <- function(x, ...) {
  cat("<ndm_model_spec>\n")
  cat(sprintf("  preset: %s\n", x$preset %||% "custom"))
  cat(sprintf("  model_type: %s\n", x$model_type %||% NA_character_))
  cat(sprintf("  source_format: %s\n", x$source_format %||% "tex"))
  cat(sprintf("  compartments: %s\n", x$compartments %||% NA_character_))
  cat(sprintf("  dynamic_beta: %s\n", x$dynamic_beta %||% NA))
  cat(sprintf("  dynamic_global: %s\n", x$dynamic_global %||% NA))
  cat(sprintf("  multi_outcome: %s\n", x$multi_outcome %||% NA))
  if (!is.null(x$family)) {
    cat(sprintf("  family: %s\n", x$family))
  }
  if (length(x$state_terms %||% character(0)) > 0L) {
    cat(sprintf("  state_terms: %s\n", paste(x$state_terms, collapse = ", ")))
  }
  if (!is.null(x$source_path)) {
    cat(sprintf("  source_path: %s\n", x$source_path))
  }
  invisible(x)
}

.onUnload <- function(libpath) {
  ndm_env$backend <- NULL
  invisible(libpath)
}
