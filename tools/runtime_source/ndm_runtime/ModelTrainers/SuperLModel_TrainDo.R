# Content of SuperLModel_TrainDo.R
print("Sarting SuperLModel_TrainDo.R")
SavedModelDir <- get0(
  "SavedModelDir",
  ifnotfound = file.path(
    "./SavedModels",
    ifelse(isTRUE(SimMode), "FromSim", "FromReal"),
    sprintf(
      "Model_%s_%s",
      AnalysisName,
      get0("RUN_ID", inherits = TRUE, ifnotfound = paste0("legacy_outer", OUTER_ITERATION))
    )
  )
)
checkpoint_file <- function(prefix, step){
  file.path(SavedModelDir, sprintf("%s_i%s_%s_%s.eqx", prefix, step, AnalysisName, AnalysisDate))
}
reset_train_iterator <- function(){
  TFDatasetIterator_train <<- reticulate::as_iterator(TFDataset_train)
  TFDatasetIterator_train
}
batch_has_expected_shape <- function(dat_, expected_batch_size){
  if(is.null(dat_) || length(dat_) == 0L){
    return(FALSE)
  }
  batch_dims <- try(vapply(dat_, function(l_){ as.integer(np$array(l_$shape)[[1]]) }, integer(1)), TRUE)
  if("try-error" %in% class(batch_dims) || length(batch_dims) == 0L || any(is.na(batch_dims))){
    return(FALSE)
  }
  batch_size <- batch_dims[[1]]
  batch_size > 0L && batch_size == expected_batch_size && all(batch_dims == batch_size)
}
next_train_batch <- function(max_attempts = 100L){
  for(attempt_i in seq_len(max_attempts)){
    dat_ <- reticulate::iter_next(TFDatasetIterator_train)
    if(batch_has_expected_shape(dat_, nBatch)){
      return(TFConst2JAXArray(dat_))
    }
    if(attempt_i < max_attempts){
      print2(sprintf("Resetting train iterator in TrainDo.R [attempt %s]", attempt_i))
      reset_train_iterator()
    }
  }
  stop(
    sprintf(
      "Could not obtain a full training batch of %s examples after %s attempts (%s retries) in TrainDo.R.",
      nBatch,
      max_attempts,
      max(0L, max_attempts - 1L)
    ),
    call. = FALSE
  )
}
save_eqx_enabled <- isTRUE(get0("SaveEqx", ifnotfound = TRUE))
recover_checkpoint_at <- get0("RecoverCheckpointAt", ifnotfound = NULL)
if(isTRUE(recover_checkpoint_at)){
  recover_checkpoint_at <- "last"
}
checkpoint_has_scale_state <- function(){
  exists("SIM_GLOBAL_SCALE_MEAN", inherits = TRUE) &&
    exists("SIM_GLOBAL_SCALE_SD", inherits = TRUE)
}
checkpoint_scale_state <- function(){
  if(!checkpoint_has_scale_state()){
    return(NULL)
  }
  jnp$array(list(SIM_GLOBAL_SCALE_MEAN, SIM_GLOBAL_SCALE_SD))
}
checkpoint_model_payload <- function(){
  payload <- list(ModelList, state, opt_state)
  scale_state <- checkpoint_scale_state()
  if(!is.null(scale_state)){
    payload[[4]] <- scale_state
  }
  payload
}
restore_checkpoint_scale_state <- function(recovered_payload){
  if(length(recovered_payload) < 4L || is.null(recovered_payload[[4]])){
    return(invisible(NULL))
  }
  SIM_GLOBAL_SCALE_MEAN <<- jnp$take(recovered_payload[[4]], 0L, axis = 0L)$tolist()
  SIM_GLOBAL_SCALE_SD <<- jnp$take(recovered_payload[[4]], 1L, axis = 0L)$tolist()
  invisible(NULL)
}
ndm_condition_message <- utils::getFromNamespace(".ndm_condition_message", "ndm")
ndm_numeric_summary <- utils::getFromNamespace(".ndm_numeric_summary", "ndm")
ndm_first_nonfinite_name <- utils::getFromNamespace(".ndm_first_nonfinite_name", "ndm")
ndm_write_nonfinite_report <- utils::getFromNamespace(".ndm_write_nonfinite_report", "ndm")
ndm_solver_failure_condition <- utils::getFromNamespace(
  ".ndm_solver_failure_condition",
  "ndm"
)
ndm_write_solver_failure_report <- utils::getFromNamespace(
  ".ndm_write_solver_failure_report",
  "ndm"
)
solver_runtime_scalar <- function(value, default = NA_real_) {
  if (is.null(value)) {
    return(default)
  }
  host_value <- try(
    as.vector(as.array(np$array(value))),
    silent = TRUE
  )
  if (inherits(host_value, "try-error") || length(host_value) < 1L) {
    return(default)
  }
  suppressWarnings(as.numeric(host_value[[1L]]))
}
solver_runtime_attribute <- function(object, name) {
  tryCatch(
    reticulate::py_get_attr(object, name),
    error = function(...) NULL
  )
}
solver_runtime_label <- function(object) {
  tryCatch(
    as.character(reticulate::py_str(object)),
    error = function(...) paste(class(object), collapse = "/")
  )
}
solver_runtime_configuration <- function() {
  list(
    solver = solver_runtime_label(VI_diff_eq_solver_optim),
    stepsize_controller = solver_runtime_label(stepsize_controller_optim),
    rtol = solver_runtime_scalar(
      solver_runtime_attribute(stepsize_controller_optim, "rtol")
    ),
    atol = solver_runtime_scalar(
      solver_runtime_attribute(stepsize_controller_optim, "atol")
    ),
    dt0 = solver_runtime_scalar(dt0_init_optim),
    max_steps = suppressWarnings(as.integer(MaxSteps))
  )
}
solver_batch_identifier <- function(value, name, expected_length) {
  host_value <- try(
    as.vector(as.array(np$array(value))),
    silent = TRUE
  )
  if (inherits(host_value, "try-error")) {
    stop("Could not transfer `", name, "` for solver diagnostics.", call. = FALSE)
  }
  host_value <- suppressWarnings(as.integer(host_value))
  if (length(host_value) != expected_length || anyNA(host_value)) {
    stop(
      "`", name, "` is not aligned with the solver diagnostic batch.",
      call. = FALSE
    )
  }
  host_value
}
solver_diagnostics_to_host <- function(diagnostics) {
  required_fields <- c(
    "success",
    "failure_stage_code",
    "prediction_finite",
    "global_attempted",
    "global_result_success",
    "global_state_finite",
    "global_result_code",
    "global_num_steps",
    "global_num_accepted_steps",
    "global_num_rejected_steps",
    "global_max_steps",
    "local_attempted",
    "local_result_success",
    "local_state_finite",
    "local_result_code",
    "local_num_steps",
    "local_num_accepted_steps",
    "local_num_rejected_steps",
    "local_max_steps"
  )
  missing_fields <- setdiff(required_fields, names(diagnostics))
  if (length(missing_fields) > 0L) {
    stop(
      sprintf(
        "Training prediction omitted required solver diagnostics: %s.",
        paste(missing_fields, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  boolean_fields <- c(
    "success",
    "prediction_finite",
    "global_attempted",
    "global_result_success",
    "global_state_finite",
    "local_attempted",
    "local_result_success",
    "local_state_finite"
  )
  host_fields <- lapply(required_fields, function(field_name) {
    value <- as.vector(as.array(np$array(diagnostics[[field_name]])))
    if (field_name %in% boolean_fields) {
      return(as.logical(value))
    }
    as.integer(value)
  })
  names(host_fields) <- required_fields
  field_lengths <- vapply(host_fields, length, integer(1L))
  if (length(unique(field_lengths)) != 1L || field_lengths[[1L]] < 1L) {
    stop("Solver diagnostic fields do not share one non-empty batch dimension.", call. = FALSE)
  }
  data.frame(
    example_index = seq_len(field_lengths[[1L]]),
    host_fields,
    check.names = FALSE
  )
}
solver_failure_stage <- function(stage_code) {
  stage_names <- c(
    "success",
    "global_ode",
    "local_ode",
    "prediction"
  )
  stage_code <- as.integer(stage_code)
  valid <- !is.na(stage_code) & stage_code >= 0L & stage_code < length(stage_names)
  out <- rep("unknown", length(stage_code))
  out[valid] <- stage_names[stage_code[valid] + 1L]
  out
}
solver_iteration_telemetry <- function(diagnostics) {
  global_rows <- which(diagnostics$global_attempted %in% TRUE)
  local_rows <- which(diagnostics$local_attempted %in% TRUE)
  steps <- c(
    diagnostics$global_num_steps[global_rows],
    diagnostics$local_num_steps[local_rows]
  )
  rejected_steps <- c(
    diagnostics$global_num_rejected_steps[global_rows],
    diagnostics$local_num_rejected_steps[local_rows]
  )
  steps <- steps[is.finite(steps)]
  rejected_steps <- rejected_steps[is.finite(rejected_steps)]
  list(
    solver_num_steps_max = if (length(steps) == 0L) 0L else max(steps),
    solver_num_rejected_steps_max = if (length(rejected_steps) == 0L) {
      0L
    } else {
      max(rejected_steps)
    },
    solver_num_rejected_steps_total = if (length(rejected_steps) == 0L) {
      0L
    } else {
      sum(rejected_steps)
    }
  )
}
nonfinite_empty_summary <- function(error = NA_character_){
  list(
    error = error,
    dims = integer(),
    length = 0L,
    finite_fraction = NA_real_,
    n_nonfinite = NA_integer_,
    n_nan = NA_integer_,
    n_inf = NA_integer_,
    min = NA_real_,
    max = NA_real_,
    mean = NA_real_,
    first_nonfinite_index = integer()
  )
}
nonfinite_capture_summary <- function(expr){
  value <- try(eval.parent(substitute(expr)), silent = TRUE)
  if(inherits(value, "try-error")){
    return(nonfinite_empty_summary(ndm_condition_message(value)))
  }
  ndm_numeric_summary(value, np = np)
}
capture_nonfinite_learning_rate <- function(iteration){
  if(!exists("LR_schedule_vec", inherits = TRUE) || length(LR_schedule_vec) < iteration){
    return(NA_real_)
  }
  suppressWarnings(as.numeric(LR_schedule_vec[[iteration]])[[1L]])
}
capture_nonfinite_base_id <- function(){
  if(!exists("SimEntry", inherits = TRUE) || is.null(SimEntry) || is.null(SimEntry$BaseID)){
    return(NA_integer_)
  }
  suppressWarnings(as.integer(as.numeric(SimEntry$BaseID)[[1L]]))
}
capture_nonfinite_report <- function(batch_l,
                                     loss_value,
                                     grad_norm_value,
                                     iteration,
                                     keys_mat){
  GetPredSaveAtInfo_runtime <- if (exists("ndm_runtime_normalize_getpred_saveat_info", inherits = TRUE)) {
    ndm_runtime_normalize_getpred_saveat_info(GetPredSaveAtInfo_default)
  } else {
    GetPredSaveAtInfo_default
  }
  prediction <- try(
    GetPred_train_jit(
      ModelList,
      batch2package(batch_l),
      state,
      PriorList,
      PolicyList,
      GetPredSaveAtInfo_runtime,
      keys_mat
    ),
    silent = TRUE
  )

  tensor_summaries <- list(
    batch_YTrue_out = nonfinite_capture_summary(batch_l$YTrue_out),
    batch_YTrue_out_mask = nonfinite_capture_summary(batch_l$YTrue_out_mask),
    batch_YTrue = nonfinite_capture_summary(batch_l$YTrue),
    batch_YTrue_mask = nonfinite_capture_summary(batch_l$YTrue_mask),
    batch_XPred = nonfinite_capture_summary(batch_l$XPred),
    batch_XPred_mask = nonfinite_capture_summary(batch_l$XPred_mask)
  )

  if(inherits(prediction, "try-error")){
    tensor_summaries$prediction_forward_pass <- nonfinite_empty_summary(ndm_condition_message(prediction))
  } else {
    pred_obj <- prediction[[1]]
    tensor_summaries$prediction_y_mu <- nonfinite_capture_summary(pred_obj$y_mu)
    tensor_summaries$prediction_y_sigma <- nonfinite_capture_summary(pred_obj$y_sigma)
    tensor_summaries$prediction_center_param <- nonfinite_capture_summary(pred_obj$ODEParamsSampList$center_param)
    tensor_summaries$prediction_scale_param <- nonfinite_capture_summary(pred_obj$ODEParamsSampList$scale_param)
    tensor_summaries$prediction_s_l <- nonfinite_capture_summary(pred_obj$ODEParamsSampList$s_l_samp)
    tensor_summaries$prediction_e_l <- nonfinite_capture_summary(pred_obj$ODEParamsSampList$e_l_samp)
    tensor_summaries$prediction_i_l <- nonfinite_capture_summary(pred_obj$ODEParamsSampList$i_l_samp)
    tensor_summaries$prediction_r_l <- nonfinite_capture_summary(pred_obj$ODEParamsSampList$r_l_samp)
    tensor_summaries$prediction_raw_beta_t <- nonfinite_capture_summary(pred_obj$ODEParamsSampList$diff_eq_sol_ys$raw_beta_t)
    tensor_summaries$prediction_raw_policy_t <- nonfinite_capture_summary(pred_obj$ODEParamsSampList$diff_eq_sol_ys$raw_policy_t)
    tensor_summaries$prediction_neural1 <- nonfinite_capture_summary(pred_obj$ODEParamsSampList$diff_eq_sol_ys$Neural1)
  }

  outer_iteration <- get0("OUTER_ITERATION", ifnotfound = NA_integer_)
  report <- list(
    analysis_name = get0("AnalysisName", ifnotfound = NA_character_),
    outer_iteration = outer_iteration,
    iteration = as.integer(iteration),
    base_id = capture_nonfinite_base_id(),
    model_type = get0("ModelType", ifnotfound = NA_character_),
    n_samples_train = suppressWarnings(as.integer(get0("nSamplesTrain", ifnotfound = NA_integer_))),
    model_depth = suppressWarnings(as.integer(get0("ModelDepth", ifnotfound = NA_integer_))),
    model_dims = suppressWarnings(as.integer(get0("ModelDims", ifnotfound = NA_integer_))),
    context_length = suppressWarnings(as.integer(get0("nTimesPast", ifnotfound = NA_integer_))),
    learning_rate = capture_nonfinite_learning_rate(iteration),
    loss = loss_value,
    grad_norm = grad_norm_value,
    sim_entry = if(exists("SimEntry", inherits = TRUE)) SimEntry else NULL,
    tensor_summaries = tensor_summaries,
    first_bad_tensor = ndm_first_nonfinite_name(tensor_summaries)
  )
  report_path <- ndm_write_nonfinite_report(
    report = report,
    holder_folder = HolderFolder,
    outer_iteration = outer_iteration,
    iteration = iteration
  )
  list(report = report, report_path = report_path)
}
if (exists("ndm_runtime_normalize_getpred_saveat_info", inherits = TRUE)) {
  GetPredSaveAtInfo_default <- ndm_runtime_normalize_getpred_saveat_info(GetPredSaveAtInfo_default)
}
checkpoint_save_steps <- function(n_steps, n_checkpoints) {
  n_steps <- max(1L, as.integer(n_steps))
  n_checkpoints <- max(0L, as.integer(n_checkpoints))
  if (n_checkpoints == 0L) {
    return(integer())
  }
  if (n_checkpoints == 1L || n_steps == 1L) {
    return(n_steps)
  }
  first_step <- min(10L, n_steps)
  intermediate <- round(10^seq(
    log(first_step, base = 10),
    log(n_steps, base = 10),
    length.out = n_checkpoints
  ))
  sort(unique(c(as.integer(intermediate), n_steps)))
}
CheckPointSaveAt <- checkpoint_save_steps(nSGD_MASTER, nCheckpoints)
KVCacheTrainingExercised <- isTRUE(get0(
  "KVCacheTrainingExercised", inherits = TRUE, ifnotfound = FALSE
))
KVCacheInferenceExercised <- isTRUE(get0(
  "KVCacheInferenceExercised", inherits = TRUE, ifnotfound = FALSE
))
print2(sprintf(
  paste(
    "KV cache runtime: requested=%s; training_requested=%s;",
    "inference_effective=%s; training_effective=%s"
  ),
  isTRUE(get0("EnableKVCachingRequested", inherits = TRUE, ifnotfound = FALSE)),
  isTRUE(get0("EnableKVCachingTrainingRequested", inherits = TRUE, ifnotfound = FALSE)),
  isTRUE(get0("EnableKVCachingInferenceEffective", inherits = TRUE, ifnotfound = FALSE)),
  isTRUE(get0("EnableKVCachingTrainingEffective", inherits = TRUE, ifnotfound = FALSE))
))
training_telemetry <- data.frame(
  iteration = integer(),
  training_objective = character(),
  outcome_loss_scale = character(),
  kv_cache_requested = logical(),
  kv_cache_training_requested = logical(),
  kv_cache_inference_effective = logical(),
  kv_cache_training_effective = logical(),
  kv_cache_inference_exercised = logical(),
  kv_cache_training_exercised = logical(),
  elapsed_seconds = numeric(),
  gradient_seconds = numeric(),
  loss = numeric(),
  objective_data_loss = numeric(),
  student_t_nll = numeric(),
  raw_mse = numeric(),
  scaled_mse = numeric(),
  kl_local = numeric(),
  kl_global = numeric(),
  kl_place = numeric(),
  kl_unweighted = numeric(),
  kl_weighted = numeric(),
  auxiliary_mean_loss = numeric(),
  prediction_abs_mean = numeric(),
  truth_abs_mean = numeric(),
  gradient_norm = numeric(),
  trainable_parameters = numeric(),
  solver_num_steps_max = integer(),
  solver_num_rejected_steps_max = integer(),
  solver_num_rejected_steps_total = integer(),
  stringsAsFactors = FALSE
)
loss_component_names <- c(
  "objective_data_loss", "student_t_nll", "raw_mse", "scaled_mse",
  "kl_local", "kl_global", "kl_place", "kl_unweighted", "kl_weighted",
  "auxiliary_mean_loss", "prediction_abs_mean", "truth_abs_mean"
)
loss_components_to_host <- function(components) {
  values <- stats::setNames(rep(NA_real_, length(loss_component_names)), loss_component_names)
  if (is.null(components)) {
    return(values)
  }
  for (component_name in intersect(loss_component_names, names(components))) {
    value <- suppressWarnings(as.numeric(np$array(components[[component_name]])))
    if (length(value)) {
      values[[component_name]] <- value[[1L]]
    }
  }
  values
}
write_training_telemetry <- function() {
  if (nrow(training_telemetry) == 0L) {
    return(invisible(NULL))
  }
  telemetry_file <- file.path(HolderFolder, "training_telemetry.csv")
  dir.create(dirname(telemetry_file), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile("training-telemetry-", tmpdir = dirname(telemetry_file), fileext = ".csv")
  data.table::fwrite(training_telemetry, temporary)
  if (!file.rename(temporary, telemetry_file)) {
    unlink(temporary, force = TRUE)
    stop("Could not atomically write training telemetry to ", telemetry_file, ".", call. = FALSE)
  }
  invisible(telemetry_file)
}
ndm_training_iteration_sequence <- function(resume_iteration, final_iteration) {
  bounds <- suppressWarnings(as.numeric(c(resume_iteration, final_iteration)))
  if (length(bounds) != 2L ||
      any(!is.finite(bounds)) ||
      any(bounds < 1) ||
      any(bounds > .Machine$integer.max) ||
      any(bounds != floor(bounds))) {
    stop(
      "Training iteration bounds must be positive integers.",
      call. = FALSE
    )
  }
  resume_iteration <- as.integer(bounds[[1L]])
  final_iteration <- as.integer(bounds[[2L]])
  if (resume_iteration > final_iteration) {
    return(integer())
  }
  seq.int(resume_iteration, final_iteration)
}
ndm_training_iteration_key <- function(iteration) {
  iteration <- suppressWarnings(as.numeric(iteration))
  if (length(iteration) != 1L ||
      !is.finite(iteration) ||
      iteration < 1 ||
      iteration > .Machine$integer.max ||
      iteration != floor(iteration)) {
    stop("Training iteration must be one positive integer.", call. = FALSE)
  }
  jax$random$fold_in(
    ndm_runtime_seed_key(104729L),
    as.integer(iteration)
  )
}
for(i in ndm_training_iteration_sequence(i_, nSGD_model)){
  i_ <- i; fulliter_timer <- Sys.time()
  if(i %% 10 == 0){  print(i); gc(); py_gc$collect() }

  # get batch
  if(nSGD_pretrain > 0L | nSGD_posttrain > 0){ 
  dat_ <- next_train_batch()

  observation_mask_host <- as.array(np$array(dat_$YTrue_out_mask))
  if (length(dim(observation_mask_host)) == 0L) {
    observation_mask_host <- array(
      observation_mask_host,
      dim = c(length(observation_mask_host), 1L)
    )
  }
  observation_mask_matrix <- matrix(
    as.numeric(observation_mask_host),
    nrow = dim(observation_mask_host)[[1L]]
  )
  invalid_mask_examples <- which(
    !apply(is.finite(observation_mask_matrix), 1L, all) |
      rowSums(observation_mask_matrix != 0) == 0
  )
  if (length(invalid_mask_examples) > 0L) {
    stop(
      sprintf(
        "Training batch contains non-finite or all-zero observation masks for example(s): %s.",
        paste(invalid_mask_examples, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  # update step
  {
    gd_timer <- Sys.time()
    if( i == 1 ){ print2( "At first gradLoss_jax()" ) }
    if (isTRUE(get0(
      "EnableKVCachingTrainingEffective", inherits = TRUE, ifnotfound = FALSE
    ))) {
      KVCacheTrainingExercised <- TRUE
    }
    iteration_key <- ndm_training_iteration_key(i)
    keys_mat <- ndm_runtime_data_to_device(
      jax$random$split(iteration_key, nBatch)
    )
    batch_pkg <- batch2package(dat_)
    GetPredSaveAtInfo_runtime <- if (exists("ndm_runtime_normalize_getpred_saveat_info", inherits = TRUE)) {
      ndm_runtime_normalize_getpred_saveat_info(GetPredSaveAtInfo_default)
    } else {
      GetPredSaveAtInfo_default
    }
    train_step_result <- train_step_compiled(
      ModelList,
      batch_pkg,
      dat_$YTrue_out,
      dat_$YTrue_out_mask,
      jnp$array(as.numeric(i)),
      state,
      PriorList,
      PolicyList,
      GetPredSaveAtInfo_runtime,
      keys_mat,
      opt_state
    )
    Loss_i <- in_loss_vec[i] <- suppressWarnings(as.numeric(np$array(train_step_result$loss))[[1L]])
    GradNorm_i <- grad_norm_vec[i] <- suppressWarnings(as.numeric(np$array(train_step_result$grad_norm))[[1L]])
    loss_components_i <- loss_components_to_host(train_step_result$loss_components)

    solver_diagnostics_host <- solver_diagnostics_to_host(
      train_step_result$solver_diagnostics
    )
    solver_diagnostics_host$location_id_numeric <- solver_batch_identifier(
      dat_$location_id_numeric,
      "location_id_numeric",
      nrow(solver_diagnostics_host)
    )
    solver_diagnostics_host$time_id_numeric <- solver_batch_identifier(
      dat_$time_id_numeric,
      "time_id_numeric",
      nrow(solver_diagnostics_host)
    )
    failed_solver_examples <- which(
      is.na(solver_diagnostics_host$success) |
        !solver_diagnostics_host$success
    )
    if (length(failed_solver_examples) > 0L) {
      failed_diagnostics <- solver_diagnostics_host[
        failed_solver_examples,
        ,
        drop = FALSE
      ]
      failed_diagnostics$failure_stage <- solver_failure_stage(
        failed_diagnostics$failure_stage_code
      )
      solver_configuration <- solver_runtime_configuration()
      failure_report <- list(
        schema_version = 1L,
        event = "ndm_solver_failure",
        created_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
        analysis_name = get0("AnalysisName", ifnotfound = "Analysis2"),
        outer_iteration = get0("OUTER_ITERATION", ifnotfound = NA_integer_),
        iteration = as.integer(i),
        base_id = capture_nonfinite_base_id(),
        model_type = get0("ModelType", ifnotfound = NA_character_),
        loss = Loss_i,
        gradient_norm = GradNorm_i,
        candidate_update_applied = FALSE,
        solver_configuration = solver_configuration,
        failed_examples = failed_diagnostics
      )
      failure_report_path <- ndm_write_solver_failure_report(
        report = failure_report,
        holder_folder = HolderFolder,
        outer_iteration = get0(
          "OUTER_ITERATION",
          ifnotfound = NA_integer_
        ),
        iteration = i
      )
      stop(ndm_solver_failure_condition(
        iteration = i,
        example_indices = failed_diagnostics$example_index,
        stages = failed_diagnostics$failure_stage,
        diagnostics = failed_diagnostics,
        context = list(
          phase = "training",
          analysis_name = get0("AnalysisName", ifnotfound = "Analysis2"),
          outer_iteration = get0("OUTER_ITERATION", ifnotfound = NA_integer_),
          base_id = capture_nonfinite_base_id(),
          model_type = get0("ModelType", ifnotfound = NA_character_),
          candidate_update_applied = FALSE,
          solver_configuration = solver_configuration,
          report_path = failure_report_path
        )
      ))
    }
    solver_telemetry_i <- solver_iteration_telemetry(
      solver_diagnostics_host
    )

    UpdateParametersCond <- is.finite(Loss_i) & is.finite(GradNorm_i)
    if(! UpdateParametersCond ){
      nonfinite_capture <- capture_nonfinite_report(
        batch_l = dat_,
        loss_value = Loss_i,
        grad_norm_value = GradNorm_i,
        iteration = i,
        keys_mat = keys_mat
      )
      stop(
        sprintf(
          paste(
            "Non-finite training state in %s",
            "[outer=%s, BaseID=%s, ModelType=%s, iter=%s, loss=%s, grad_norm=%s].",
            "Saved debug report to %s.",
            sep = " "
          ),
          get0("AnalysisName", ifnotfound = "Analysis2"),
          get0("OUTER_ITERATION", ifnotfound = NA_integer_),
          capture_nonfinite_base_id(),
          get0("ModelType", ifnotfound = NA_character_),
          i,
          format(Loss_i, scientific = TRUE),
          format(GradNorm_i, scientific = TRUE),
          nonfinite_capture$report_path
        )
      )
    }
    if( UpdateParametersCond){
      ExecuteUpdateCounter <- ExecuteUpdateCounter + 1
      if (exists("append_block_update_log", inherits = TRUE)) {
        append_block_update_log(i, train_step_result$block_update_metrics)
      }
      ModelList <- train_step_result$model
      state <- train_step_result$state
      opt_state <- train_step_result$opt_state
      rm(batch_pkg, dat_)
    }

      # plotting sequence (do this after all parameter + state updates complete)
      {
      if( (i == 1 | i %% 10 == 0)  & is.na(COMMAND_ARG_INPUT) ){
          plottingSeq_counter <- plottingSeq_counter + 1
          if(plottingSeq_counter == 1){ print2( "First plotting seq analysis"  ) }
          par(mfrow = c(1,2))
          for(reri in 1:2){
            # tmp2 <- GetPred_train_jit( ModelList, jax_batchx, state, PriorList, PolicyList, jax$random$split(JaxKey(ai(runif(1,1,100000))), nBatch))
            # plot( np$array( tmp2[[1]]$y_mu )[,,1] ) ; plot( c(np$array( tmp2[[1]]$y_sigma[,,1] ) ))
            ( {tmp555 <- np$array(GetPred_train_jit(
              ModelList, batch2package(batch_l_cal),
              state, PriorList, PolicyList,
              GetPredSaveAtInfo_runtime,
              ndm_runtime_data_to_device(jax$random$split(JaxKey(ai(runif(1,1,100000))), nBatch)))[[1]]$y_mu)[,,1] })
            if(reri == 1){ tmp_ <- tmp555 }
            if(reri == 1){ try(plot(tmp555[1,],ylim = c(0,max(tmp555)),type = 'l',main="Y-hat"),T) }
            for(i3 in 2:nrow(tmp555)){ points(tmp555[i3,],type = 'l',col = ifelse(reri == 1, yes = "black", no = "gray"),
                                           lwd = ifelse(reri == 1, yes = 2, no = 2),
                                           lty = ifelse(reri == 1, yes = 1, no = 2)) }
          }

          # plot compare against truth
          tmp_true <- np$array(batch_l_cal$YTrue_out)[,,1]
          try(plot(tmp_true[1,],ylim = c(0,max(tmp_true)),type = 'l',main="Y-true"),T)
          for(i3 in 2:nrow(tmp555)){ points(tmp_true[i3,],type = 'l',col = "black",
                                         lwd = 2,
                                         lty = 1) }
          print(sprintf("Current cor with truth: %.3f", cor(c(tmp555),c(tmp_true))))
          
          plot(c(tmp555),c(tmp_true),
               xlim = summary(c(tmp555,tmp_true))[c(1,6)],
               ylim = summary(c(tmp555,tmp_true))[c(1,6)]); abline(a=0,b=1); 
          plot(c(tmp555),c(tmp_true),
               xlim = summary(c(tmp555,tmp_true))[c(1,6)],
               ylim = summary(c(tmp555,tmp_true))[c(1,6)]);abline(a=0,b=1)

          print2(sprintf("Within iteration mean correlation: %.10f", mean( diag( cor(t(tmp_), t(tmp555))) ) ))
          if(plottingSeq_counter == 1){ sharedXSamp <- sample(1:length(tmp555),length(tmp555)/10) }
          if(plottingSeq_counter > 1){
            crossIterCor_vec <- c(crossIterCor_vec, cor(tmp555[sharedXSamp], tmpSamp_minus1))
            print2( sprintf("Across iteration correlation: %.10f",crossIterCor_vec[ length(crossIterCor_vec) ]) )
          }
          if(i == 1){ tmpSamp_t0 <- tmp555[sharedXSamp] }
          tmpSamp_minus1 <- tmp555[sharedXSamp] # update minus 1 value
          #rm(tmp555,tmp_,tmp_true)
          rm(tmp_,tmp_true)
      }
      if( (i > 10 & i %% 10 == 5) & is.na(COMMAND_ARG_INPUT) ){
        par(mfrow=c(1,2)); try({plot(rank(na.omit(in_loss_vec)),main="Loss"); points(smooth.spline(rank(na.omit(in_loss_vec))),type = "l",lwd=3)},T)
        try({plot(grad_norm_vec[1:i],log = 'y',main="Gradients")},T)
      }
      }

      # print results
      if (i == 1 || (is.na(COMMAND_ARG_INPUT) && i %% 4 == 0L) ||
            (!is.na(COMMAND_ARG_INPUT) && i %% 20 == 0L)) {
          te_total <- as.numeric(difftime(Sys.time(), fulliter_timer, units = "secs"))
          te_grads <- as.numeric(difftime(Sys.time(), gd_timer, units = "secs"))
          print2( sprintf("Iter: %s of %s -- Loss: %.4f (%.3f%%) -- Last OutCor %.2f: -- Last OutSkill: %.2f -- Total (s): %s -- Grad & updates (s): %.3f",
                            i, nSGD_model, 
                            Loss_i,
                            100*(1-rank(in_loss_vec[1:i])[i]/i),
                            LastOutCor, Skill8SanityCheck,
                            round(te_total, 2L),
                            round(te_grads, 3L) )   )
          training_telemetry <- rbind(
            training_telemetry,
            data.frame(
              iteration = as.integer(i),
              training_objective = as.character(get0(
                "training_objective", inherits = TRUE, ifnotfound = "student_t_nll"
              )),
              outcome_loss_scale = paste(
                format(
                  as.numeric(get0(
                    "outcome_loss_scale",
                    inherits = TRUE,
                    ifnotfound = NA_real_
                  )),
                  digits = 17L,
                  scientific = TRUE,
                  trim = TRUE
                ),
                collapse = ";"
              ),
              kv_cache_requested = isTRUE(get0(
                "EnableKVCachingRequested", inherits = TRUE, ifnotfound = FALSE
              )),
              kv_cache_training_requested = isTRUE(get0(
                "EnableKVCachingTrainingRequested", inherits = TRUE, ifnotfound = FALSE
              )),
              kv_cache_inference_effective = isTRUE(get0(
                "EnableKVCachingInferenceEffective", inherits = TRUE, ifnotfound = FALSE
              )),
              kv_cache_training_effective = isTRUE(get0(
                "EnableKVCachingTrainingEffective", inherits = TRUE, ifnotfound = FALSE
              )),
              kv_cache_inference_exercised = KVCacheInferenceExercised,
              kv_cache_training_exercised = KVCacheTrainingExercised,
              elapsed_seconds = te_total,
              gradient_seconds = te_grads,
              loss = Loss_i,
              objective_data_loss = loss_components_i[["objective_data_loss"]],
              student_t_nll = loss_components_i[["student_t_nll"]],
              raw_mse = loss_components_i[["raw_mse"]],
              scaled_mse = loss_components_i[["scaled_mse"]],
              kl_local = loss_components_i[["kl_local"]],
              kl_global = loss_components_i[["kl_global"]],
              kl_place = loss_components_i[["kl_place"]],
              kl_unweighted = loss_components_i[["kl_unweighted"]],
              kl_weighted = loss_components_i[["kl_weighted"]],
              auxiliary_mean_loss = loss_components_i[["auxiliary_mean_loss"]],
              prediction_abs_mean = loss_components_i[["prediction_abs_mean"]],
              truth_abs_mean = loss_components_i[["truth_abs_mean"]],
              gradient_norm = GradNorm_i,
              trainable_parameters = as.numeric(nParams),
              solver_num_steps_max = solver_telemetry_i$solver_num_steps_max,
              solver_num_rejected_steps_max =
                solver_telemetry_i$solver_num_rejected_steps_max,
              solver_num_rejected_steps_total =
                solver_telemetry_i$solver_num_rejected_steps_total,
              stringsAsFactors = FALSE
            )
          )
      }

      # Checkpoints and analytics are generated after the optimizer update, so
      # a final-only policy records the actual terminal model state.
      if (i %in% CheckPointSaveAt) {
        if (!(i %in% training_telemetry$iteration)) {
          checkpoint_total_seconds <- as.numeric(difftime(Sys.time(), fulliter_timer, units = "secs"))
          checkpoint_gradient_seconds <- as.numeric(difftime(Sys.time(), gd_timer, units = "secs"))
          training_telemetry <- rbind(
            training_telemetry,
            data.frame(
              iteration = as.integer(i),
              training_objective = as.character(get0(
                "training_objective", inherits = TRUE, ifnotfound = "student_t_nll"
              )),
              outcome_loss_scale = paste(
                format(
                  as.numeric(get0(
                    "outcome_loss_scale",
                    inherits = TRUE,
                    ifnotfound = NA_real_
                  )),
                  digits = 17L,
                  scientific = TRUE,
                  trim = TRUE
                ),
                collapse = ";"
              ),
              kv_cache_requested = isTRUE(get0(
                "EnableKVCachingRequested", inherits = TRUE, ifnotfound = FALSE
              )),
              kv_cache_training_requested = isTRUE(get0(
                "EnableKVCachingTrainingRequested", inherits = TRUE, ifnotfound = FALSE
              )),
              kv_cache_inference_effective = isTRUE(get0(
                "EnableKVCachingInferenceEffective", inherits = TRUE, ifnotfound = FALSE
              )),
              kv_cache_training_effective = isTRUE(get0(
                "EnableKVCachingTrainingEffective", inherits = TRUE, ifnotfound = FALSE
              )),
              kv_cache_inference_exercised = KVCacheInferenceExercised,
              kv_cache_training_exercised = KVCacheTrainingExercised,
              elapsed_seconds = checkpoint_total_seconds,
              gradient_seconds = checkpoint_gradient_seconds,
              loss = Loss_i,
              objective_data_loss = loss_components_i[["objective_data_loss"]],
              student_t_nll = loss_components_i[["student_t_nll"]],
              raw_mse = loss_components_i[["raw_mse"]],
              scaled_mse = loss_components_i[["scaled_mse"]],
              kl_local = loss_components_i[["kl_local"]],
              kl_global = loss_components_i[["kl_global"]],
              kl_place = loss_components_i[["kl_place"]],
              kl_unweighted = loss_components_i[["kl_unweighted"]],
              kl_weighted = loss_components_i[["kl_weighted"]],
              auxiliary_mean_loss = loss_components_i[["auxiliary_mean_loss"]],
              prediction_abs_mean = loss_components_i[["prediction_abs_mean"]],
              truth_abs_mean = loss_components_i[["truth_abs_mean"]],
              gradient_norm = GradNorm_i,
              trainable_parameters = as.numeric(nParams),
              solver_num_steps_max = solver_telemetry_i$solver_num_steps_max,
              solver_num_rejected_steps_max =
                solver_telemetry_i$solver_num_rejected_steps_max,
              solver_num_rejected_steps_total =
                solver_telemetry_i$solver_num_rejected_steps_total,
              stringsAsFactors = FALSE
            )
          )
        }
        print2( "Saving checkpoint (Optimizer, Model, Model-generating Code)..." )
        saveCheckpointCounter <- saveCheckpointCounter + 1

        if(saveCheckpointCounter == 1){
          zip::zip(zipfile = file.path(SavedModelDir, sprintf("AnalysisR_%s_%s.zip", AnalysisName, AnalysisDate)),
                   files = NDM_INTERNAL_ANALYSIS_DIR)
        }

        save_i <- i
        if(save_eqx_enabled){
          eq$tree_serialise_leaves(checkpoint_file("ModelList", save_i),
                                   checkpoint_model_payload())
          if("ModelList_notshared_set" %in% ls()){
            eq$tree_serialise_leaves(checkpoint_file("ModelList_notshared_set", save_i),
                                     list(ModelList_notshared_set,
                                          lapply(sim_dat_norm_list,jnp$array)))
          }
        }

        if(save_eqx_enabled && !is.null(recover_checkpoint_at)){
          RecoverAt <- recover_checkpoint_at
          ModelList_recovered <- eq$tree_deserialise_leaves(
            checkpoint_file("ModelList", RecoverAt),
            checkpoint_model_payload())
          if("ModelList_notshared_set" %in% ls()){
            ModelList_notshared_set_recovered <- eq$tree_deserialise_leaves(
              checkpoint_file("ModelList_notshared_set", RecoverAt),
              list(ModelList_notshared_set, lapply(sim_dat_norm_list,jnp$array))
            )
            sim_dat_norm_list <- lapply(ModelList_notshared_set_recovered[[2]],function(l_){
              SIM_GLOBAL_SCALE_MEAN_ <- jnp$take(l_,0L,axis=0L)$tolist()
              SIM_GLOBAL_SCALE_SD_ <- jnp$take(l_,1L,axis=0L)$tolist()
              list(SIM_GLOBAL_SCALE_MEAN_, SIM_GLOBAL_SCALE_SD_)
            })
            restore_checkpoint_scale_state(ModelList_recovered)
            ModelList_notshared_set <- ModelList_notshared_set_recovered[[1]]
          }

          restore_checkpoint_scale_state(ModelList_recovered)
          opt_state <- ModelList_recovered[[3]]
          state <- ModelList_recovered[[2]]
          ModelList <- ModelList_recovered[[1]]
          if (exists("ndm_runtime_replicate_tree", inherits = TRUE)) {
            opt_state <- ndm_runtime_replicate_tree(opt_state)
            state <- ndm_runtime_replicate_tree(state)
            ModelList <- ndm_runtime_replicate_tree(ModelList)
          }
          i_ <- np$array( opt_state[[4]][[2]]$count )
          rm( ModelList_recovered )
        }

        print2( sprintf("Starting GetAnalytics.R at %s of %s", i, nSGD_model) )
        outSampCounter <- outSampCounter+1
        ndm_source_extracted("ResultsGet/SuperLModel_GetAnalytics.R")
        if (isTRUE(get0(
          "EnableKVCachingInferenceEffective", inherits = TRUE, ifnotfound = FALSE
        ))) {
          KVCacheInferenceExercised <- TRUE
          training_telemetry$kv_cache_inference_exercised <- TRUE
        }
        write_training_telemetry()
      }
    }
  }
}
write_training_telemetry()
print("Done with SuperLModel_TrainDo.R")
