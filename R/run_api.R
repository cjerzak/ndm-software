#' Create package-native real-data run configurations
#'
#' These helpers capture the maintained package-only orchestration surface for
#' real, simulation, and multidisease runs. They replace the old expectation
#' that callers execute an external runtime checkout.
#'
#' @details
#' `dry_run = TRUE` resolves paths and selected rows without executing the
#' underlying workflow. Dry runs can use lightweight in-memory grids such as the
#' `BaseID` / `ModelType` preview shown in `README.md`.
#'
#' Non-dry execution expects Analysis2-compatible grids. The tested simulation
#' grids in this package are produced by `ndmdatasets::ndm_sim_build_grid()` and
#' then augmented with fields such as `paddingMethod`, `lookahead`,
#' `n_time_steps`, `n_inference_batches`, `scaling_batches`, and either
#' `model_spec_name` or `model_tex_loc`. The tested real and multidisease grids
#' include `BaseID`, `ContextLength`, `evaluationTime`, `initialTransform`,
#' `initialNormType`, `paddingMethod`, `OSSType`, `dataInputs`, `ModelType`,
#' `ModelDepth`, `ModelDims`, `nSamplesTrain`, `nObsInference`, `floatType`,
#' and either `model_spec_name` or `model_tex_loc`.
#'
#' Non-dry runner workflows also require the full execution stack: the
#' `ndmdatasets` package, the helper packages used by the runtime, and a backend
#' environment aligned with `NDM_SOFTWARE_CONDA_ENV`, `NDM_CONDA_ENV`, or the
#' platform-specific fallback used by the runner layer.
#'
#' Distributed simulation TFRecord bootstrap is handled separately via
#' [ndm_bootstrap_sim_tfrecords()]. Unlike `ndm_run_sim(..., outer = ...)`, the
#' bootstrap entrypoint schedules work by canonical `BaseID` in raw grid order
#' and is the supported surface for parallel TFRecord materialization.
#'
#' @param project_root Working project directory used for grids, TFRecords, and
#'   outputs.
#' @param analysis_name Analysis label threaded through output paths.
#' @param grid Optional in-memory grid data frame. When supplied, `grid_file`
#'   must be `NULL`.
#' @param grid_file Optional CSV path for the run grid.
#' @param outer Integer vector of outer-iteration rows to execute.
#' @param run_seed Optional integer scalar used to seed model initialization,
#'   training, and inference. When `NULL`, legacy outer-row-derived seeding is
#'   retained for backward compatibility.
#' @param force_to_gpu Deprecated logical alias for `compute_backend`. `TRUE`
#'   maps to `"gpu"` and `FALSE` maps to `"cpu"`.
#' @param compute_backend Compute device policy: `"auto"` selects a supported
#'   JAX GPU when available and otherwise CPU, while `"cpu"` and `"gpu"`
#'   require the named backend.
#' @param gpu_mem_frac Optional finite GPU memory fraction in `(0, 1]`. It is
#'   invalid when `compute_backend = "cpu"` and rejected at initialization if
#'   `"auto"` resolves to CPU.
#' @param enable_kv_cache Logical scalar controlling DecoderOnly KV caching.
#'   The default enables caching for inference.
#' @param enable_kv_cache_training Logical scalar controlling caching during
#'   DecoderOnly gradient training. It defaults to `TRUE`: cached training is
#'   gradient-equivalent to the un-cached rollout and several times faster per
#'   step. Set `FALSE` to reproduce historical un-cached training exactly.
#' @param inference_mc_draws Positive integer number of posterior draws
#'   requested for inference. Deterministic NeuralODE and DecoderOnly paths use
#'   one effective draw downstream.
#' @param observation_scale_floor Finite positive lower bound added to learned
#'   observation scales.
#' @param initial_observation_scale Finite positive initial observation-scale
#'   value on the outcome scale. The default is `1`, matching the unit-scale
#'   initialization used before this control became configurable. It must be
#'   greater than `observation_scale_floor`.
#' @param neuralode_variational Logical scalar controlling variational
#'   NeuralODE training and inference. It has no effect on DecoderOnly models.
#' @param neuralode_kl_weight Finite non-negative multiplier for the NeuralODE
#'   KL contribution. A value of zero disables KL evaluation without disabling
#'   variational sampling.
#' @param neuralode_mean_loss_weight Finite non-negative multiplier for the
#'   NeuralODE auxiliary masked mean-squared-error term.
#' @param training_objective Training loss for observed outcomes. The default
#'   `"student_t_nll"` preserves the current likelihood; `"scaled_mse"` is the
#'   deterministic scale-normalized squared-error objective.
#' @param outcome_loss_scale Optional finite positive scalar or outcome-length
#'   vector used by `training_objective = "scaled_mse"`. It is required for
#'   scaled MSE and otherwise must be `NULL`.
#' @param n_checkpoints Positive integer number of checkpoint/analytics states
#'   to retain, including the terminal post-update state. The production
#'   default is one final checkpoint.
#' @param max_sgd_steps Optional positive integer cap used only by explicitly
#'   labeled pilot runs.
#' @param prior_sd_multiplier Finite positive multiplier applied to ODE prior
#'   standard deviations; `1` is the core-grid default.
#' @param solver_profile ODE optimization solver profile: `"default"`,
#'   `"loose"`, `"tight"`, or `"alternative"`.
#' @param model_type Optional model family override.
#' @param respect_grid_model_type Logical scalar indicating whether grid rows
#'   should be allowed to override the requested model type.
#' @param resave_tfrecords Logical scalar retained for compatibility. Training
#'   workflows require `FALSE`. For real and simulation runs, build canonical
#'   artifacts separately with [ndm_bootstrap_real_tfrecords()] or
#'   [ndm_bootstrap_sim_tfrecords()]. For multidisease runs, use
#'   [ndm_bootstrap_multidisease_tfrecords()].
#' @param tfrecord_dir Optional TFRecord output directory.
#' @param raw_data_dir Real-data input directory.
#' @param outcome_metric Outcome metric name for real-data or multidisease runs.
#' @param data_subset Optional real-data or multidisease subset name.
#' @param disease_names Character vector of multidisease names.
#' @param data_format Multidisease data format.
#' @param covariate_panel_file Optional WHO annual covariate-panel CSV, keyed
#'   by `location_id` and `year`. Supply it together with
#'   `covariate_manifest_file`.
#' @param covariate_manifest_file Optional CSV whose ordered `feature_name`
#'   column defines the covariate-panel feature contract and whose remaining
#'   columns carry provenance metadata.
#' @param dry_run Logical scalar indicating whether the run should stop after
#'   resolving paths and grid rows.
#'
#' @returns `ndm_create_*_run_config()` returns a classed list describing the
#'   requested workflow. Each `ndm_run_*()` function returns a dry-run preview
#'   when `config$dry_run` is `TRUE` and otherwise the underlying workflow
#'   result.
#'
#' @examples
#' \dontrun{
#' preview_grid <- data.frame(
#'   BaseID = c(1L, 2L),
#'   ModelType = c("DecoderOnly", "NeuralODE"),
#'   stringsAsFactors = FALSE
#' )
#'
#' cfg <- ndm_create_sim_run_config(
#'   project_root = tempdir(),
#'   grid = preview_grid,
#'   outer = 1:2,
#'   dry_run = TRUE
#' )
#'
#' ndm_run_sim(cfg)
#' }
#'
#' @export
ndm_create_real_run_config <- function(project_root = getwd(),
                                       analysis_name = "RealApril15",
                                       grid = NULL,
                                       grid_file = file.path("Data", "RunGrids", "RealGrids", sprintf("RealGrid_%s.csv", analysis_name)),
                                       outer = 3L,
                                       run_seed = NULL,
                                       force_to_gpu = NULL,
                                       gpu_mem_frac = NULL,
                                       enable_kv_cache = TRUE,
                                       enable_kv_cache_training = TRUE,
                                       inference_mc_draws = 5L,
                                       observation_scale_floor = 1e-5,
                                       initial_observation_scale = 1.0,
                                       neuralode_variational = TRUE,
                                       neuralode_kl_weight = 1.0,
                                       neuralode_mean_loss_weight = 0.0,
                                       training_objective = c("student_t_nll", "scaled_mse"),
                                       outcome_loss_scale = NULL,
                                       n_checkpoints = 1L,
                                       max_sgd_steps = NULL,
                                       prior_sd_multiplier = 1.0,
                                       solver_profile = "default",
                                       model_type = NULL,
                                       respect_grid_model_type = TRUE,
                                       resave_tfrecords = FALSE,
                                       tfrecord_dir = file.path("Data", "RunTFRecords", "RealTFRecords", analysis_name),
                                       raw_data_dir = file.path("Data", "MainData"),
                                       outcome_metric = "inc_death",
                                       data_subset = "high_income",
                                       dry_run = FALSE,
                                       compute_backend = c("auto", "cpu", "gpu")) {
  compute_backend_supplied <- !missing(compute_backend)
  if (!is.null(grid)) {
    grid_file <- NULL
  }
  .ndm_make_run_config(
    mode = "real",
    project_root = project_root,
    analysis_name = analysis_name,
    grid = grid,
    grid_file = grid_file,
    outer = outer,
    run_seed = run_seed,
    force_to_gpu = force_to_gpu,
    compute_backend = compute_backend,
    compute_backend_supplied = compute_backend_supplied,
    gpu_mem_frac = gpu_mem_frac,
    enable_kv_cache = enable_kv_cache,
    enable_kv_cache_training = enable_kv_cache_training,
    inference_mc_draws = inference_mc_draws,
    observation_scale_floor = observation_scale_floor,
    initial_observation_scale = initial_observation_scale,
    neuralode_variational = neuralode_variational,
    neuralode_kl_weight = neuralode_kl_weight,
    neuralode_mean_loss_weight = neuralode_mean_loss_weight,
    training_objective = training_objective,
    outcome_loss_scale = outcome_loss_scale,
    n_checkpoints = n_checkpoints,
    max_sgd_steps = max_sgd_steps,
    prior_sd_multiplier = prior_sd_multiplier,
    solver_profile = solver_profile,
    model_type = model_type,
    respect_grid_model_type = respect_grid_model_type,
    resave_tfrecords = resave_tfrecords,
    tfrecord_dir = tfrecord_dir,
    raw_data_dir = raw_data_dir,
    outcome_metric = outcome_metric,
    data_subset = data_subset,
    dry_run = dry_run
  )
}

#' @rdname ndm_create_real_run_config
#' @export
ndm_create_sim_run_config <- function(project_root = getwd(),
                                      analysis_name = "BigSimsLatest",
                                      grid = NULL,
                                      grid_file = file.path("Data", "RunGrids", "SimGrids", sprintf("SimGrid_%s.csv", analysis_name)),
                                      outer = 1L,
                                      run_seed = NULL,
                                      force_to_gpu = NULL,
                                      gpu_mem_frac = NULL,
                                      enable_kv_cache = TRUE,
                                      enable_kv_cache_training = TRUE,
                                      inference_mc_draws = 5L,
                                      observation_scale_floor = 1e-5,
                                      initial_observation_scale = 1.0,
                                      neuralode_variational = TRUE,
                                      neuralode_kl_weight = 1.0,
                                      neuralode_mean_loss_weight = 0.0,
                                      training_objective = c("student_t_nll", "scaled_mse"),
                                      outcome_loss_scale = NULL,
                                      n_checkpoints = 1L,
                                      max_sgd_steps = NULL,
                                      prior_sd_multiplier = 1.0,
                                      solver_profile = "default",
                                      model_type = NULL,
                                      respect_grid_model_type = TRUE,
                                      resave_tfrecords = FALSE,
                                      tfrecord_dir = file.path("Data", "RunTFRecords", "SimTFRecords", analysis_name),
                                      dry_run = FALSE,
                                      compute_backend = c("auto", "cpu", "gpu")) {
  compute_backend_supplied <- !missing(compute_backend)
  if (!is.null(grid)) {
    grid_file <- NULL
  }
  .ndm_make_run_config(
    mode = "sim",
    project_root = project_root,
    analysis_name = analysis_name,
    grid = grid,
    grid_file = grid_file,
    outer = outer,
    run_seed = run_seed,
    force_to_gpu = force_to_gpu,
    compute_backend = compute_backend,
    compute_backend_supplied = compute_backend_supplied,
    gpu_mem_frac = gpu_mem_frac,
    enable_kv_cache = enable_kv_cache,
    enable_kv_cache_training = enable_kv_cache_training,
    inference_mc_draws = inference_mc_draws,
    observation_scale_floor = observation_scale_floor,
    initial_observation_scale = initial_observation_scale,
    neuralode_variational = neuralode_variational,
    neuralode_kl_weight = neuralode_kl_weight,
    neuralode_mean_loss_weight = neuralode_mean_loss_weight,
    training_objective = training_objective,
    outcome_loss_scale = outcome_loss_scale,
    n_checkpoints = n_checkpoints,
    max_sgd_steps = max_sgd_steps,
    prior_sd_multiplier = prior_sd_multiplier,
    solver_profile = solver_profile,
    model_type = model_type,
    respect_grid_model_type = respect_grid_model_type,
    resave_tfrecords = resave_tfrecords,
    tfrecord_dir = tfrecord_dir,
    dry_run = dry_run
  )
}

#' @rdname ndm_create_real_run_config
#' @export
ndm_create_multidisease_run_config <- function(project_root = getwd(),
                                               analysis_name = "RealLatest",
                                               grid = NULL,
                                               grid_file = file.path("Data", "RunGrids", "RealGrids", sprintf("RealGrid_%s.csv", analysis_name)),
                                               outer = 1L,
                                               run_seed = NULL,
                                               force_to_gpu = NULL,
                                               gpu_mem_frac = NULL,
                                               enable_kv_cache = TRUE,
                                               enable_kv_cache_training = TRUE,
                                               inference_mc_draws = 5L,
                                               observation_scale_floor = 1e-5,
                                               initial_observation_scale = 1.0,
                                               neuralode_variational = TRUE,
                                               neuralode_kl_weight = 1.0,
                                               neuralode_mean_loss_weight = 0.0,
                                               training_objective = c("student_t_nll", "scaled_mse"),
                                               outcome_loss_scale = NULL,
                                               n_checkpoints = 1L,
                                               max_sgd_steps = NULL,
                                               prior_sd_multiplier = 1.0,
                                               solver_profile = "default",
                                               model_type = NULL,
                                               respect_grid_model_type = TRUE,
                                               resave_tfrecords = FALSE,
                                               tfrecord_dir = file.path("Data", "RunTFRecords", "RealTFRecords", analysis_name),
                                               outcome_metric = "CountValue",
                                               data_subset = "all",
                                               disease_names = c("Covid", "Flu"),
                                               data_format = "IHME",
                                               dry_run = FALSE,
                                               covariate_panel_file = NULL,
                                               covariate_manifest_file = NULL,
                                               compute_backend = c("auto", "cpu", "gpu")) {
  compute_backend_supplied <- !missing(compute_backend)
  if (!is.null(grid)) {
    grid_file <- NULL
  }
  .ndm_make_run_config(
    mode = "multidisease",
    project_root = project_root,
    analysis_name = analysis_name,
    grid = grid,
    grid_file = grid_file,
    outer = outer,
    run_seed = run_seed,
    force_to_gpu = force_to_gpu,
    compute_backend = compute_backend,
    compute_backend_supplied = compute_backend_supplied,
    gpu_mem_frac = gpu_mem_frac,
    enable_kv_cache = enable_kv_cache,
    enable_kv_cache_training = enable_kv_cache_training,
    inference_mc_draws = inference_mc_draws,
    observation_scale_floor = observation_scale_floor,
    initial_observation_scale = initial_observation_scale,
    neuralode_variational = neuralode_variational,
    neuralode_kl_weight = neuralode_kl_weight,
    neuralode_mean_loss_weight = neuralode_mean_loss_weight,
    training_objective = training_objective,
    outcome_loss_scale = outcome_loss_scale,
    n_checkpoints = n_checkpoints,
    max_sgd_steps = max_sgd_steps,
    prior_sd_multiplier = prior_sd_multiplier,
    solver_profile = solver_profile,
    model_type = model_type,
    respect_grid_model_type = respect_grid_model_type,
    resave_tfrecords = resave_tfrecords,
    tfrecord_dir = tfrecord_dir,
    outcome_metric = outcome_metric,
    data_subset = data_subset,
    disease_names = disease_names,
    data_format = data_format,
    covariate_panel_file = covariate_panel_file,
    covariate_manifest_file = covariate_manifest_file,
    dry_run = dry_run
  )
}

#' Bootstrap canonical simulation TFRecords by BaseID
#'
#' This helper materializes canonical simulation TFRecords without going through
#' the legacy `outer` scheduling path. It reads the simulation grid in raw CSV
#' or data-frame order, builds one canonical write plan row per `BaseID`, and
#' optionally writes those TFRecords with a per-`BaseID` cross-process lock.
#'
#' @param project_root Working project directory used to resolve relative grid
#'   and TFRecord paths.
#' @param analysis_name Analysis label used to derive default grid and TFRecord
#'   paths.
#' @param grid Optional in-memory simulation grid. When supplied, `grid_file`
#'   must be `NULL`.
#' @param grid_file Optional CSV path for the simulation grid.
#' @param base_ids Optional integer vector of canonical `BaseID` values to
#'   materialize. When `NULL`, bootstrap all `BaseID`s in raw grid order of
#'   first appearance.
#' @param tfrecord_dir Output directory for canonical TFRecords.
#' @param overwrite Logical scalar controlling whether existing canonical
#'   TFRecords should be regenerated.
#' @param producer Nonempty named producer metadata stored in every artifact
#'   manifest. Required for publication; dry runs may omit it.
#' @param parallel_workers Positive integer number of forked BaseID workers on
#'   Linux. Inputs are prepared before forking. macOS and Windows use the
#'   deterministic serial path because TensorFlow/JAX runtimes are not
#'   fork-safe there.
#' @param dry_run Logical scalar indicating whether to return the canonical
#'   write plan without writing TFRecords.
#'
#' @returns A data frame with one row per canonical `BaseID` and columns
#'   `BaseID`, `selected_rows`, `canonical_row`,
#'   `artifact_n_samples_train`, `train_file`, `inference_file`, and `status`.
#'
#' @export
ndm_bootstrap_sim_tfrecords <- function(project_root = getwd(),
                                        analysis_name = "BigSimsLatest",
                                        grid = NULL,
                                        grid_file = file.path("Data", "RunGrids", "SimGrids", sprintf("SimGrid_%s.csv", analysis_name)),
                                        base_ids = NULL,
                                        tfrecord_dir = file.path("Data", "RunTFRecords", "SimTFRecords", analysis_name),
                                        producer = NULL,
                                        parallel_workers = 1L,
                                        overwrite = FALSE,
                                        dry_run = FALSE) {
  project_root <- .ndm_normalize_path(project_root, must_work = TRUE)
  if (!isTRUE(dry_run) && is.null(producer)) {
    stop("`producer` is required when publishing canonical TFRecords.", call. = FALSE)
  }

  if (!is.null(grid)) {
    grid_file <- NULL
  }

  if (!is.null(grid) && !is.null(grid_file)) {
    stop("Supply either `grid` or `grid_file`, not both.", call. = FALSE)
  }
  if (!is.null(grid) && !is.data.frame(grid)) {
    stop("`grid` must be a data.frame when supplied.", call. = FALSE)
  }

  if (!is.null(grid)) {
    grid_file <- NULL
    sim_grid <- as.data.frame(grid, stringsAsFactors = FALSE)
  } else {
    grid_file <- .ndm_path_join_if_relative(project_root, grid_file)
    grid_file <- .ndm_normalize_path(grid_file, must_work = TRUE)
    sim_grid <- as.data.frame(data.table::fread(grid_file), stringsAsFactors = FALSE)
  }

  tfrecord_dir <- .ndm_path_join_if_relative(project_root, tfrecord_dir)
  tfrecord_dir <- .ndm_normalize_path(tfrecord_dir, must_work = FALSE)

  api_env <- .ndm_legacy_run_env()
  bootstrap_fun <- get("analysis2_bootstrap_sim_tfrecords", envir = api_env, inherits = FALSE)
  bootstrap_fun(
    project_root = project_root,
    analysis_name = as.character(analysis_name),
    grid = sim_grid,
    base_ids = base_ids,
    tfrecord_dir = tfrecord_dir,
    producer = producer,
    parallel_workers = as.integer(parallel_workers),
    overwrite = isTRUE(overwrite),
    dry_run = isTRUE(dry_run)
  )
}

#' Bootstrap canonical real-data TFRecords by BaseID
#'
#' Materializes one canonical real-data train/inference artifact pair per
#' `BaseID`, with the same cross-process lock and canonical-row selection used
#' by [ndm_bootstrap_sim_tfrecords()].
#'
#' @inheritParams ndm_bootstrap_sim_tfrecords
#' @param analysis_name Analysis label used to derive default real-data paths.
#' @param grid Optional in-memory real-data grid.
#' @param grid_file Optional CSV path for the real-data grid.
#' @param tfrecord_dir Output directory for canonical real-data TFRecords.
#' @param raw_data_dir Directory containing the real-data source tables.
#' @param outcome_metric Outcome metric passed to `ndmdatasets`.
#' @param data_subset Real-data subset passed to `ndmdatasets`.
#'
#' @returns A canonical per-`BaseID` write-plan data frame with a `status`
#'   column.
#'
#' @export
ndm_bootstrap_real_tfrecords <- function(project_root = getwd(),
                                         analysis_name = "RealApril15",
                                         grid = NULL,
                                         grid_file = file.path("Data", "RunGrids", "RealGrids", sprintf("RealGrid_%s.csv", analysis_name)),
                                         base_ids = NULL,
                                         tfrecord_dir = file.path("Data", "RunTFRecords", "RealTFRecords", analysis_name),
                                         raw_data_dir = file.path("Data", "MainData"),
                                         outcome_metric = "inc_death",
                                         data_subset = "high_income",
                                         producer = NULL,
                                         parallel_workers = 1L,
                                         overwrite = FALSE,
                                         dry_run = FALSE) {
  project_root <- .ndm_normalize_path(project_root, must_work = TRUE)
  if (!isTRUE(dry_run) && is.null(producer)) {
    stop("`producer` is required when publishing canonical TFRecords.", call. = FALSE)
  }
  if (!is.null(grid)) {
    grid_file <- NULL
  }
  if (!is.null(grid) && !is.data.frame(grid)) {
    stop("`grid` must be a data.frame when supplied.", call. = FALSE)
  }
  if (is.null(grid)) {
    grid_file <- .ndm_path_join_if_relative(project_root, grid_file)
    grid_file <- .ndm_normalize_path(grid_file, must_work = TRUE)
    grid <- as.data.frame(data.table::fread(grid_file), stringsAsFactors = FALSE)
  }
  tfrecord_dir <- .ndm_normalize_path(
    .ndm_path_join_if_relative(project_root, tfrecord_dir),
    must_work = FALSE
  )
  raw_data_dir <- .ndm_normalize_path(
    .ndm_path_join_if_relative(project_root, raw_data_dir),
    must_work = !isTRUE(dry_run)
  )

  api_env <- .ndm_legacy_run_env()
  bootstrap_fun <- get("analysis2_bootstrap_real_tfrecords", envir = api_env, inherits = FALSE)
  bootstrap_fun(
    project_root = project_root,
    analysis_name = as.character(analysis_name),
    grid = grid,
    base_ids = base_ids,
    tfrecord_dir = tfrecord_dir,
    raw_data_dir = raw_data_dir,
    outcome_metric = as.character(outcome_metric),
    data_subset = as.character(data_subset),
    producer = producer,
    parallel_workers = as.integer(parallel_workers),
    overwrite = isTRUE(overwrite),
    dry_run = isTRUE(dry_run)
  )
}

.ndm_validate_resave_tfrecords <- function(mode, resave_tfrecords) {
  if (is.null(resave_tfrecords) || identical(resave_tfrecords, FALSE)) {
    return(invisible(TRUE))
  }

  guidance <- switch(
    mode,
    real = "Use `ndm_bootstrap_real_tfrecords()` before training.",
    sim = "Use `ndm_bootstrap_sim_tfrecords()` before training.",
    multidisease = "Use `ndm_bootstrap_multidisease_tfrecords()` before training.",
    paste(
      "Use `ndm_bootstrap_real_tfrecords()` or `ndm_bootstrap_sim_tfrecords()` before training.",
      "Use `ndm_bootstrap_multidisease_tfrecords()` for multidisease inputs."
    )
  )
  stop(
    "`resave_tfrecords = TRUE` is no longer supported by training workflows. ",
    guidance,
    call. = FALSE
  )
}

.ndm_run_env_get <- function(env, name) {
  get(name, envir = env, inherits = FALSE)
}

.ndm_validate_run_flag <- function(value, name) {
  if (!is.logical(value) || length(value) != 1L || is.na(value)) {
    stop("`", name, "` must be one non-missing logical value.", call. = FALSE)
  }
  value
}

.ndm_make_run_config <- function(mode,
                                 project_root,
                                 analysis_name,
                                 grid = NULL,
                                 grid_file = NULL,
                                 outer,
                                 run_seed = NULL,
                                 force_to_gpu = NULL,
                                 gpu_mem_frac = NULL,
                                 enable_kv_cache = TRUE,
                                 enable_kv_cache_training = TRUE,
                                 inference_mc_draws = 5L,
                                 observation_scale_floor = 1e-5,
                                 initial_observation_scale = 1.0,
                                 neuralode_variational = TRUE,
                                 neuralode_kl_weight = 1.0,
                                 neuralode_mean_loss_weight = 0.0,
                                 training_objective = c("student_t_nll", "scaled_mse"),
                                 outcome_loss_scale = NULL,
                                 n_checkpoints = 1L,
                                 max_sgd_steps = NULL,
                                 prior_sd_multiplier = 1.0,
                                 solver_profile = "default",
                                 model_type = NULL,
                                 respect_grid_model_type = TRUE,
                                 resave_tfrecords = FALSE,
                                 tfrecord_dir = NULL,
                                 raw_data_dir = NULL,
                                 outcome_metric = NULL,
                                 data_subset = NULL,
                                 disease_names = NULL,
                                 data_format = NULL,
                                 covariate_panel_file = NULL,
                                 covariate_manifest_file = NULL,
                                 dry_run = FALSE,
                                 compute_backend = c("auto", "cpu", "gpu"),
                                 compute_backend_supplied = !missing(compute_backend)) {
  project_root <- .ndm_normalize_path(project_root, must_work = TRUE)

  if (!is.null(grid) && !is.null(grid_file)) {
    stop("Supply either `grid` or `grid_file`, not both.", call. = FALSE)
  }
  if (!is.null(grid) && !is.data.frame(grid)) {
    stop("`grid` must be a data.frame when supplied.", call. = FALSE)
  }

  outer <- as.integer(outer)
  if (length(outer) == 0L || anyNA(outer)) {
    stop("`outer` must contain at least one integer row index.", call. = FALSE)
  }
  if (!is.null(run_seed)) {
    run_seed_numeric <- suppressWarnings(as.numeric(run_seed))
    if (length(run_seed_numeric) != 1L || !is.finite(run_seed_numeric) ||
        run_seed_numeric < 0 || run_seed_numeric > .Machine$integer.max ||
        run_seed_numeric != floor(run_seed_numeric)) {
      stop("`run_seed` must be NULL or one non-negative integer.", call. = FALSE)
    }
    run_seed <- as.integer(run_seed_numeric)
  }
  compute_backend <- .ndm_resolve_compute_backend(
    compute_backend = compute_backend,
    force_to_gpu = force_to_gpu,
    compute_backend_supplied = compute_backend_supplied
  )
  force_to_gpu_compat <- switch(
    compute_backend,
    auto = NULL,
    cpu = FALSE,
    gpu = TRUE
  )
  enable_kv_cache <- .ndm_validate_run_flag(enable_kv_cache, "enable_kv_cache")
  enable_kv_cache_training <- .ndm_validate_run_flag(
    enable_kv_cache_training,
    "enable_kv_cache_training"
  )
  if (isTRUE(enable_kv_cache_training) && !isTRUE(enable_kv_cache)) {
    stop(
      "`enable_kv_cache_training = TRUE` requires `enable_kv_cache = TRUE`.",
      call. = FALSE
    )
  }
  neuralode_variational <- .ndm_validate_run_flag(
    neuralode_variational,
    "neuralode_variational"
  )
  training_objective <- match.arg(training_objective)
  respect_grid_model_type <- .ndm_validate_run_flag(
    respect_grid_model_type,
    "respect_grid_model_type"
  )
  dry_run <- .ndm_validate_run_flag(dry_run, "dry_run")
  gpu_mem_frac <- .ndm_validate_gpu_mem_frac(
    gpu_mem_frac,
    if (identical(compute_backend, "cpu")) "cpu" else NULL
  )
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
    stop("`observation_scale_floor` must be one finite positive value.", call. = FALSE)
  }
  initial_observation_scale <- suppressWarnings(as.numeric(initial_observation_scale))
  if (length(initial_observation_scale) != 1L ||
      !is.finite(initial_observation_scale) || initial_observation_scale <= 0) {
    stop("`initial_observation_scale` must be one finite positive value.", call. = FALSE)
  }
  if (initial_observation_scale <= observation_scale_floor) {
    stop(
      "`initial_observation_scale` must be greater than `observation_scale_floor`.",
      call. = FALSE
    )
  }
  neuralode_kl_weight <- suppressWarnings(as.numeric(neuralode_kl_weight))
  if (length(neuralode_kl_weight) != 1L || !is.finite(neuralode_kl_weight) ||
      neuralode_kl_weight < 0) {
    stop("`neuralode_kl_weight` must be one finite non-negative value.", call. = FALSE)
  }
  neuralode_mean_loss_weight <- suppressWarnings(as.numeric(neuralode_mean_loss_weight))
  if (length(neuralode_mean_loss_weight) != 1L ||
      !is.finite(neuralode_mean_loss_weight) || neuralode_mean_loss_weight < 0) {
    stop("`neuralode_mean_loss_weight` must be one finite non-negative value.", call. = FALSE)
  }
  if (!is.null(outcome_loss_scale)) {
    if (is.list(outcome_loss_scale)) {
      outcome_loss_scale <- unlist(outcome_loss_scale, recursive = TRUE, use.names = FALSE)
    }
    outcome_loss_scale <- suppressWarnings(as.numeric(outcome_loss_scale))
    if (!length(outcome_loss_scale) || any(!is.finite(outcome_loss_scale)) ||
        any(outcome_loss_scale <= 0)) {
      stop("`outcome_loss_scale` must be NULL or a finite positive numeric vector.", call. = FALSE)
    }
  }
  if (identical(training_objective, "scaled_mse")) {
    if (is.null(outcome_loss_scale)) {
      stop("`outcome_loss_scale` is required for `training_objective = \"scaled_mse\"`.", call. = FALSE)
    }
    if (any(outcome_loss_scale < 1e-8)) {
      stop("`outcome_loss_scale` values must be at least 1e-8 for scaled MSE.", call. = FALSE)
    }
    if (isTRUE(neuralode_variational) || neuralode_kl_weight != 0 ||
        neuralode_mean_loss_weight != 0 || inference_mc_draws != 1L) {
      stop(
        paste(
          "`training_objective = \"scaled_mse\"` requires",
          "`neuralode_variational = FALSE`, `neuralode_kl_weight = 0`,",
          "`neuralode_mean_loss_weight = 0`, and `inference_mc_draws = 1`."
        ),
        call. = FALSE
      )
    }
  } else if (!is.null(outcome_loss_scale)) {
    stop("`outcome_loss_scale` is only valid with `training_objective = \"scaled_mse\"`.", call. = FALSE)
  }
  n_checkpoints <- suppressWarnings(as.numeric(n_checkpoints))
  if (length(n_checkpoints) != 1L || !is.finite(n_checkpoints) ||
      n_checkpoints < 1 || n_checkpoints > .Machine$integer.max ||
      n_checkpoints != floor(n_checkpoints)) {
    stop("`n_checkpoints` must be one positive integer.", call. = FALSE)
  }
  n_checkpoints <- as.integer(n_checkpoints)
  if (!is.null(max_sgd_steps)) {
    max_sgd_steps <- suppressWarnings(as.numeric(max_sgd_steps))
    if (length(max_sgd_steps) != 1L || !is.finite(max_sgd_steps) ||
        max_sgd_steps < 1 || max_sgd_steps > .Machine$integer.max ||
        max_sgd_steps != floor(max_sgd_steps)) {
      stop("`max_sgd_steps` must be NULL or one positive integer.", call. = FALSE)
    }
    max_sgd_steps <- as.integer(max_sgd_steps)
  }
  prior_sd_multiplier <- suppressWarnings(as.numeric(prior_sd_multiplier))
  if (length(prior_sd_multiplier) != 1L || !is.finite(prior_sd_multiplier) ||
      prior_sd_multiplier <= 0) {
    stop("`prior_sd_multiplier` must be one finite positive value.", call. = FALSE)
  }
  solver_profile <- tolower(as.character(solver_profile))
  if (length(solver_profile) != 1L ||
      !solver_profile %in% c("default", "loose", "tight", "alternative")) {
    stop("`solver_profile` must be default, loose, tight, or alternative.", call. = FALSE)
  }
  .ndm_validate_resave_tfrecords(mode, resave_tfrecords)
  if (!identical(mode, "multidisease") &&
      (!is.null(covariate_panel_file) || !is.null(covariate_manifest_file))) {
    stop(
      "Covariate panel files are supported only for multidisease runs.",
      call. = FALSE
    )
  }
  covariate_files <- .ndm_multidisease_optional_file_pair(
    project_root = project_root,
    covariate_panel_file = covariate_panel_file,
    covariate_manifest_file = covariate_manifest_file,
    must_work = FALSE
  )
  if (!is.null(covariate_files$panel) &&
      !identical(toupper(as.character(data_format)), "WHO")) {
    stop(
      "Covariate panel files are supported only when `data_format = \"WHO\"`.",
      call. = FALSE
    )
  }

  class_name <- switch(
    mode,
    real = "ndm_real_run_config",
    sim = "ndm_sim_run_config",
    multidisease = "ndm_multidisease_run_config",
    stop("Unsupported run mode: ", mode, call. = FALSE)
  )

  .ndm_make_classed_list(
    list(
      mode = mode,
      project_root = project_root,
      analysis_name = as.character(analysis_name),
      grid = grid,
      grid_file = grid_file,
      outer = outer,
      run_seed = run_seed,
      force_to_gpu = force_to_gpu_compat,
      compute_backend = compute_backend,
      gpu_mem_frac = gpu_mem_frac,
      enable_kv_cache = enable_kv_cache,
      enable_kv_cache_training = enable_kv_cache_training,
      inference_mc_draws = inference_mc_draws,
      observation_scale_floor = observation_scale_floor,
      initial_observation_scale = initial_observation_scale,
      neuralode_variational = neuralode_variational,
      neuralode_kl_weight = neuralode_kl_weight,
      neuralode_mean_loss_weight = neuralode_mean_loss_weight,
      training_objective = training_objective,
      outcome_loss_scale = outcome_loss_scale,
      n_checkpoints = n_checkpoints,
      max_sgd_steps = max_sgd_steps,
      prior_sd_multiplier = prior_sd_multiplier,
      solver_profile = solver_profile,
      model_type = model_type,
      respect_grid_model_type = respect_grid_model_type,
      resave_tfrecords = isTRUE(resave_tfrecords),
      tfrecord_dir = tfrecord_dir,
      raw_data_dir = raw_data_dir,
      outcome_metric = outcome_metric,
      data_subset = data_subset,
      disease_names = disease_names,
      data_format = data_format,
      covariate_panel_file = covariate_files$panel,
      covariate_manifest_file = covariate_files$manifest,
      dry_run = dry_run
    ),
    class_name
  )
}

.ndm_run_mode_fun_name <- function(mode) {
  switch(
    mode,
    real = "analysis2_run_real",
    sim = "analysis2_run_sim",
    multidisease = "analysis2_run_real_multidisease",
    stop("Unsupported run mode: ", mode, call. = FALSE)
  )
}

.ndm_run_config_to_args <- function(config) {
  stopifnot(inherits(config, "list"))

  .ndm_validate_resave_tfrecords(config$mode, config$resave_tfrecords)
  config$resave_tfrecords <- FALSE
  compute_backend <- .ndm_resolve_compute_backend(
    compute_backend = config$compute_backend %||% "auto",
    force_to_gpu = config$force_to_gpu,
    compute_backend_supplied = !is.null(config$compute_backend),
    warn_deprecated = FALSE
  )
  config$compute_backend <- compute_backend
  config$force_to_gpu <- switch(
    compute_backend,
    auto = NULL,
    cpu = FALSE,
    gpu = TRUE
  )
  config$gpu_mem_frac <- .ndm_validate_gpu_mem_frac(
    config$gpu_mem_frac,
    if (identical(compute_backend, "cpu")) "cpu" else NULL
  )
  for (field in c(
    "enable_kv_cache",
    "enable_kv_cache_training",
    "neuralode_variational",
    "respect_grid_model_type",
    "dry_run"
  )) {
    config[[field]] <- .ndm_validate_run_flag(config[[field]], field)
  }

  grid_file <- config$grid_file
  if (!is.null(config$grid)) {
    grid_file <- tempfile(sprintf("ndm-%s-grid-", config$mode), fileext = ".csv")
    utils::write.csv(config$grid, file = grid_file, row.names = FALSE)
  }

  args <- c(
    sprintf("--project_root=%s", config$project_root),
    sprintf("--analysis_name=%s", config$analysis_name),
    sprintf("--outer=%s", paste(config$outer, collapse = ",")),
    sprintf("--compute_backend=%s", config$compute_backend),
    sprintf("--enable_kv_cache=%s", toupper(as.character(config$enable_kv_cache))),
    sprintf(
      "--enable_kv_cache_training=%s",
      toupper(as.character(config$enable_kv_cache_training))
    ),
    sprintf("--inference_mc_draws=%s", as.integer(config$inference_mc_draws)),
    sprintf("--observation_scale_floor=%s", format(config$observation_scale_floor, scientific = TRUE, trim = TRUE)),
    sprintf("--initial_observation_scale=%s", format(config$initial_observation_scale, scientific = FALSE, trim = TRUE)),
    sprintf("--neuralode_variational=%s", toupper(as.character(config$neuralode_variational))),
    sprintf("--neuralode_kl_weight=%s", format(config$neuralode_kl_weight, scientific = FALSE, trim = TRUE)),
    sprintf("--neuralode_mean_loss_weight=%s", format(config$neuralode_mean_loss_weight, scientific = FALSE, trim = TRUE)),
    sprintf("--training_objective=%s", config$training_objective),
    sprintf("--n_checkpoints=%s", as.integer(config$n_checkpoints)),
    sprintf("--prior_sd_multiplier=%s", format(config$prior_sd_multiplier, scientific = FALSE, trim = TRUE)),
    sprintf("--solver_profile=%s", config$solver_profile),
    sprintf("--respect_grid_model_type=%s", toupper(as.character(config$respect_grid_model_type))),
    sprintf("--resave_tfrecords=%s", toupper(as.character(config$resave_tfrecords))),
    "--run_figures=FALSE",
    sprintf("--dry_run=%s", toupper(as.character(config$dry_run)))
  )

  if (!is.null(config$run_seed)) {
    args <- c(args, sprintf("--run_seed=%s", as.integer(config$run_seed)))
  }
  if (!is.null(config$max_sgd_steps)) {
    args <- c(args, sprintf("--max_sgd_steps=%s", as.integer(config$max_sgd_steps)))
  }
  if (!is.null(config$gpu_mem_frac)) {
    args <- c(args, sprintf("--gpu_mem_frac=%s", format(config$gpu_mem_frac, scientific = FALSE, trim = TRUE)))
  }
  if (!is.null(config$outcome_loss_scale)) {
    args <- c(
      args,
      sprintf(
        "--outcome_loss_scale=%s",
        paste(sprintf("%.17g", config$outcome_loss_scale), collapse = ",")
      )
    )
  }

  if (!is.null(grid_file)) {
    grid_file <- .ndm_path_join_if_relative(config$project_root, grid_file)
    args <- c(args, sprintf("--grid_file=%s", grid_file))
  }
  if (!is.null(config$model_type) && nzchar(config$model_type)) {
    args <- c(args, sprintf("--model_type=%s", config$model_type))
  }
  if (!is.null(config$tfrecord_dir) && nzchar(config$tfrecord_dir)) {
    tfrecord_dir <- config$tfrecord_dir
    tfrecord_dir <- .ndm_path_join_if_relative(config$project_root, tfrecord_dir)
    args <- c(args, sprintf("--tfrecord_dir=%s", tfrecord_dir))
  }
  if (!is.null(config$raw_data_dir) && nzchar(config$raw_data_dir)) {
    raw_data_dir <- config$raw_data_dir
    raw_data_dir <- .ndm_path_join_if_relative(config$project_root, raw_data_dir)
    args <- c(args, sprintf("--raw_data_dir=%s", raw_data_dir))
  }
  if (!is.null(config$outcome_metric) && nzchar(config$outcome_metric)) {
    args <- c(args, sprintf("--outcome_metric=%s", config$outcome_metric))
  }
  if (!is.null(config$data_subset) && nzchar(config$data_subset)) {
    args <- c(args, sprintf("--data_subset=%s", config$data_subset))
  }
  if (!is.null(config$data_format) && nzchar(config$data_format)) {
    args <- c(args, sprintf("--data_format=%s", config$data_format))
  }
  if (!is.null(config$disease_names) && length(config$disease_names) > 0L) {
    args <- c(args, sprintf("--disease_names=%s", paste(config$disease_names, collapse = ",")))
  }
  for (field in c("covariate_panel_file", "covariate_manifest_file")) {
    value <- config[[field]]
    if (!is.null(value) && nzchar(value)) {
      value <- .ndm_path_join_if_relative(config$project_root, value)
      args <- c(args, sprintf("--%s=%s", field, value))
    }
  }

  args
}

.ndm_disable_legacy_run_manifests <- function(env) {
  stopifnot(is.environment(env))

  # The maintained `ndm_run_*()` entrypoints consume structured config objects
  # directly and no longer require the legacy YAML manifest layer.
  require_yaml <- function() invisible("yaml")
  assign("analysis2_require_yaml", require_yaml, envir = env)

  resolve_config_file <- function(mode, opts, analysis_root, project_root) {
    normalize_string <- .ndm_run_env_get(env, "analysis2_normalize_string")
    path_from_project <- .ndm_run_env_get(env, "analysis2_path_from_project")
    explicit_path <- normalize_string(opts$config)
    if (is.null(explicit_path)) {
      return(NULL)
    }
    path_from_project(explicit_path, project_root = project_root, must_work = FALSE)
  }
  assign("analysis2_resolve_config_file", resolve_config_file, envir = env)

  invisible(env)
}

.ndm_filter_formal_args <- function(fun, args) {
  formal_names <- names(formals(fun))
  if (is.null(formal_names) || "..." %in% formal_names) {
    return(args)
  }

  arg_names <- names(args)
  if (is.null(arg_names)) {
    arg_names <- rep("", length(args))
  }

  keep <- !nzchar(arg_names) | arg_names %in% formal_names
  args[keep]
}

.ndm_override_legacy_model_tex_loc <- function(env) {
  resolve_model_tex_loc <- function(row_values) {
    normalize_string <- .ndm_run_env_get(env, "analysis2_normalize_string")
    model_spec_name <- .ndm_run_env_get(env, "analysis2_model_spec_name")

    legacy_path <- normalize_string(row_values$model_tex_loc)
    if (!is.null(legacy_path)) {
      return(legacy_path)
    }

    spec_name <- model_spec_name(
      model_spec_name = row_values$model_spec_name,
      model_tex_loc = row_values$model_tex_loc
    )
    if (is.null(spec_name)) {
      return(NULL)
    }

    packaged_tex_path <- system.file(
      "extdata",
      "model_specs",
      sprintf("%s.tex", spec_name),
      package = "ndm"
    )
    if (nzchar(packaged_tex_path) && file.exists(packaged_tex_path)) {
      return(packaged_tex_path)
    }

    file.path("./Analysis2/ModelStructureTex", sprintf("%s.tex", spec_name))
  }
  assign("analysis2_resolve_model_tex_loc", resolve_model_tex_loc, envir = env)

  invisible(env)
}

.ndm_override_legacy_multidisease_runner <- function(env) {
  run_real_multidisease <- function(args = commandArgs(TRUE)) {
    analysis2_log <- .ndm_run_env_get(env, "analysis2_log")
    analysis2_build_run_spec <- .ndm_run_env_get(env, "analysis2_build_run_spec")
    analysis2_print_usage <- .ndm_run_env_get(env, "analysis2_print_usage")
    analysis2_order_grid <- .ndm_run_env_get(env, "analysis2_order_grid")
    analysis2_validate_outer_iterations <- .ndm_run_env_get(env, "analysis2_validate_outer_iterations")
    analysis2_dry_run_result <- .ndm_run_env_get(env, "analysis2_dry_run_result")
    analysis2_prepare_output_roots <- .ndm_run_env_get(env, "analysis2_prepare_output_roots")
    analysis2_dir_create <- .ndm_run_env_get(env, "analysis2_dir_create")
    analysis2_as_int <- .ndm_run_env_get(env, "analysis2_as_int")
    analysis2_small_run_n_checkpoints <- .ndm_run_env_get(env, "analysis2_small_run_n_checkpoints")
    analysis2_multidisease_structured_control_globals <- get0(
      "analysis2_multidisease_structured_control_globals",
      envir = env,
      inherits = FALSE,
      ifnotfound = NULL
    )
    analysis2_small_run_n_obs_inference <- .ndm_run_env_get(env, "analysis2_small_run_n_obs_inference")
    analysis2_resolve_nsgd_calibration <- .ndm_run_env_get(env, "analysis2_resolve_nsgd_calibration")
    analysis2_log_nsgd_calibration <- .ndm_run_env_get(env, "analysis2_log_nsgd_calibration")
    analysis2_solver_profile <- .ndm_run_env_get(env, "analysis2_solver_profile")
    analysis2_model_type <- .ndm_run_env_get(env, "analysis2_model_type")

    old_error <- getOption("error")
    old_wd <- getwd()
    on.exit(options(error = old_error), add = TRUE)
    on.exit(setwd(old_wd), add = TRUE)
    options(error = NULL)
    analysis2_log("Starting package-native multidisease runner")
    spec <- analysis2_build_run_spec("multidisease", args)
    if (isTRUE(spec$help)) {
      return(analysis2_print_usage("multidisease", paths = spec$paths))
    }

    paths <- spec$paths
    grid_file <- normalizePath(spec$grid_file, winslash = "/", mustWork = TRUE)
    real_grid_raw <- as.data.frame(data.table::fread(grid_file), stringsAsFactors = FALSE)
    nsgd_calibration <- analysis2_resolve_nsgd_calibration(
      mode = "multidisease",
      spec = spec,
      grid = real_grid_raw,
      n_epoches_max = 9L
    )
    real_grid <- analysis2_order_grid(real_grid_raw, spec$outer)
    analysis2_validate_outer_iterations(real_grid, spec$outer, grid_file)

    if (isTRUE(spec$dry_run)) {
      return(analysis2_dry_run_result(spec, real_grid, nsgd_calibration = nsgd_calibration))
    }

    setwd(paths$project_root)
    analysis2_prepare_output_roots(paths$project_root, sim_mode = FALSE)
    analysis2_log_nsgd_calibration("multidisease", nsgd_calibration)
    holder_folder <- file.path(paths$project_root, "SavedResults", "Real", sprintf("Results_%s", spec$analysis_name))
    analysis2_dir_create(holder_folder)

    driver_env <- new.env(parent = globalenv())
    .ndm_install_runtime_helpers(
      driver_env,
      analysis_root = paths$analysis_root
    )
    driver_env$analysis2_as_int <- analysis2_as_int
    driver_env$analysis2_small_run_n_checkpoints <- analysis2_small_run_n_checkpoints
    if (!is.null(analysis2_multidisease_structured_control_globals)) {
      driver_env$analysis2_multidisease_structured_control_globals <-
        analysis2_multidisease_structured_control_globals
    }
    driver_env$analysis2_small_run_n_obs_inference <- analysis2_small_run_n_obs_inference
    driver_env$analysis2_resolve_nsgd_calibration <- analysis2_resolve_nsgd_calibration
    driver_env$analysis2_solver_profile <- analysis2_solver_profile
    driver_env$analysis2_model_type <- analysis2_model_type
    driver_env$analysis2_trusted_artifact_index <- get0(
      "analysis2_trusted_artifact_index",
      envir = env,
      inherits = FALSE,
      ifnotfound = function(...) NULL
    )
    driver_env$analysis2_trusted_index_covers_pair <- get0(
      "analysis2_trusted_index_covers_pair",
      envir = env,
      inherits = FALSE,
      ifnotfound = function(...) FALSE
    )
    driver_env$analysis2_multidisease_spec <- spec
    driver_env$analysis2_multidisease_grid <- real_grid
    driver_env$analysis2_nsgd_calibration <- nsgd_calibration
    source(
      file.path(paths$analysis_root, "SetupEnv", "Analysis2_legacy_multidisease_driver.R"),
      local = driver_env,
      chdir = FALSE
    )

    if (!exists("analysis2_multidisease_result", envir = driver_env, inherits = FALSE)) {
      stop("Legacy multidisease driver completed without assigning `analysis2_multidisease_result`.", call. = FALSE)
    }
    result <- get("analysis2_multidisease_result", envir = driver_env, inherits = FALSE)
    if (!isTRUE(result)) {
      stop("Legacy multidisease driver did not report successful completion.", call. = FALSE)
    }

    invisible(result)
  }
  assign("analysis2_run_real_multidisease", run_real_multidisease, envir = env)

  invisible(env)
}

.ndm_configure_legacy_run_env <- function(env) {
  stopifnot(is.environment(env))

  assign("analysis2_internal_analysis_root", function() .ndm_internal_analysis_root(), envir = env)
  assign(
    "analysis2_call",
    function(pkg, name, ...) {
      fun <- getExportedValue(pkg, name)
      args <- list(...)
      do.call(fun, .ndm_filter_formal_args(fun, args))
    },
    envir = env
  )

  .ndm_disable_legacy_run_manifests(env)
  .ndm_override_legacy_model_tex_loc(env)
  .ndm_override_legacy_multidisease_runner(env)
  env
}

.ndm_legacy_run_env <- local({
  cache <- NULL

  function(refresh = FALSE) {
    if (isTRUE(refresh) || is.null(cache)) {
      cache <<- .ndm_configure_legacy_run_env(.ndm_new_run_impl_env())
    }
    cache
  }
})

.ndm_call_analysis2_runner <- function(mode, config) {
  .ndm_validate_resave_tfrecords(mode, config$resave_tfrecords)
  api_env <- .ndm_legacy_run_env()
  run_fun <- get(.ndm_run_mode_fun_name(mode), envir = api_env, inherits = FALSE)
  run_fun(.ndm_run_config_to_args(config))
}

.ndm_invoke_legacy_analysis2_runner <- function(mode, args = commandArgs(TRUE)) {
  api_env <- .ndm_legacy_run_env()
  run_fun <- get(.ndm_run_mode_fun_name(mode), envir = api_env, inherits = FALSE)
  run_fun(args)
}

#' @param config A package-native run configuration created by the matching
#'   `ndm_create_*_run_config()` helper.
#'
#' @rdname ndm_create_real_run_config
#' @export
ndm_run_real <- function(config = ndm_create_real_run_config()) {
  if (!inherits(config, "ndm_real_run_config")) {
    stop("`config` must inherit from class 'ndm_real_run_config'.", call. = FALSE)
  }
  .ndm_call_analysis2_runner("real", config)
}

#' @rdname ndm_create_real_run_config
#' @export
ndm_run_sim <- function(config = ndm_create_sim_run_config()) {
  if (!inherits(config, "ndm_sim_run_config")) {
    stop("`config` must inherit from class 'ndm_sim_run_config'.", call. = FALSE)
  }
  .ndm_call_analysis2_runner("sim", config)
}

#' @rdname ndm_create_real_run_config
#' @export
ndm_run_multidisease <- function(config = ndm_create_multidisease_run_config()) {
  if (!inherits(config, "ndm_multidisease_run_config")) {
    stop("`config` must inherit from class 'ndm_multidisease_run_config'.", call. = FALSE)
  }
  .ndm_call_analysis2_runner("multidisease", config)
}
