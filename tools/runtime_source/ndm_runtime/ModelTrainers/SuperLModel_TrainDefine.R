# Content of SuperLModel_TrainDefine.R
ndm_training_lr_schedule <- function(n_steps, peak_value) {
  # Optax's decay_steps includes warmup. Reserve at least one post-warmup
  # update for short runs; a single-update run uses the requested peak rate.
  if (n_steps == 1L) {
    return(optax$constant_schedule(peak_value))
  }
  warmup_steps <- as.integer(min(n_steps - 1L, max(min(100L, n_steps), 0.1 * n_steps)))
  optax$warmup_cosine_decay_schedule(
    warmup_steps = warmup_steps,
    decay_steps = as.integer(n_steps),
    init_value = peak_value / 100,
    peak_value = peak_value,
    end_value = peak_value / 100
  )
}

ndm_training_clip_mask <- function(params) {
  mask <- jax$tree_util$tree_map(function(leaf) TRUE, params)
  # Pseudo-queries must start at zero for uniform residual mixing. Relative
  # clipping would suppress their gradients precisely while they learn to
  # leave zero; retain clipping for all other parameter leaves.
  backbone <- mask$TSList$TSBackbone
  for (layer_name in grep("^d[0-9]+$", names(backbone), value = TRUE)) {
    for (residual_name in c("AttnRes1", "AttnRes2")) {
      if (!is.null(backbone[[layer_name]][[residual_name]])) {
        backbone[[layer_name]][[residual_name]]$PseudoQuery <- FALSE
      }
    }
  }
  if (!is.null(backbone$AttnResOutput)) {
    backbone$AttnResOutput$PseudoQuery <- FALSE
  }
  mask$TSList$TSBackbone <- backbone
  mask
}

ndm_training_optimizer <- function(learning_rate) {
  optax$chain(
    optax$masked(optax$adaptive_grad_clip(0.1, eps = 0.0001), ndm_training_clip_mask),
    optax$adabelief(learning_rate = learning_rate, eps = 1e-6, eps_root = 1e-6)
  )
}

ndm_training_tree_finite <- function(tree) {
  finite <- jnp$array(TRUE)
  for (leaf in jax$tree_util$tree_leaves(tree)) {
    if (eq$is_inexact_array(leaf)) finite <- jnp$logical_and(finite, jnp$all(jnp$isfinite(leaf)))
  }
  finite
}

# Field orders shared with the host-side unpackers in SuperLModel_TrainDo.R.
ndm_training_solver_diagnostic_fields <- c(
  "success", "failure_stage_code", "prediction_finite",
  "global_attempted", "global_result_success", "global_state_finite", "global_result_code",
  "global_num_steps", "global_num_accepted_steps", "global_num_rejected_steps", "global_max_steps",
  "local_attempted", "local_result_success", "local_state_finite", "local_result_code",
  "local_num_steps", "local_num_accepted_steps", "local_num_rejected_steps", "local_max_steps"
)
ndm_training_loss_component_fields <- c(
  "objective_data_loss", "student_t_nll", "raw_mse", "scaled_mse",
  "kl_local", "kl_global", "kl_place", "kl_unweighted", "kl_weighted",
  "auxiliary_mean_loss", "prediction_abs_mean", "truth_abs_mean"
)

# Pack the per-example solver diagnostics into one [batch, field] int32 array so
# the host fetches them in a single transfer instead of one per field. Returns
# NULL when any field is absent, which keeps reduced test doubles on the
# unpacked path.
ndm_training_pack_diagnostics <- function(diagnostics) {
  if (!all(ndm_training_solver_diagnostic_fields %in% names(diagnostics))) return(NULL)
  columns <- lapply(ndm_training_solver_diagnostic_fields, function(field) {
    jnp$reshape(diagnostics[[field]]$astype(jnp$int32), list(-1L))
  })
  jnp$stack(columns, axis = 1L)
}

ndm_training_pack_scalars <- function(values, fields, dtype) {
  if (!all(fields %in% names(values))) return(NULL)
  jnp$stack(lapply(fields, function(field) jnp$reshape(values[[field]]$astype(dtype), list())), axis = 0L)
}

# TRUE when every example's observation mask is finite and has at least one
# observed entry; scalar masks (test doubles) are treated as valid.
ndm_training_observation_mask_valid <- function(mask) {
  if (length(mask$shape) < 2L) return(jnp$array(TRUE))
  rows <- jnp$reshape(mask$astype(jnp$float32), list(mask$shape[[1]], -1L))
  finite <- jnp$all(jnp$isfinite(rows), axis = 1L)
  nonzero <- jnp$any(jnp$not_equal(rows, 0), axis = 1L)
  jnp$all(jnp$logical_and(finite, nonzero))
}


{
  print("Sarting SuperLModel_TrainDefine.R")
  # sort( sapply(ls(), function(zr){ object.size(eval(parse(text = zr))) }) )
  # if(!SimMode){ save(input_df_red_full, file = "./tmp_input_df_red_full.Rdata"); rm ( input_df_red_full ) }
  saveCheckpointCounter <- outSampCounter <- 0;
  nRestarts <- 1L
  LR_schedule <- ndm_training_lr_schedule(nSGD_DefiningLRSeq, LEARNING_RATE_MAX)
  if(nRestarts %in% c(2,3)){ stop("Case not implemented in TrainDefine.R") } 
  if(nRestarts > 3){
    LR_schedule <- c(replicate(nRestarts-2L,
                               optax$cosine_onecycle_schedule(transition_steps = jnp$array(ai(ceiling(nSGD_DefiningLRSeq/(nRestarts) ))),
                                                              peak_value = jnp$array(LEARNING_RATE_MAX) )
                              ), optax$cosine_decay_schedule( init_value = LEARNING_RATE_MAX, 
                                          decay_steps = ai(ceiling(nSGD_DefiningLRSeq/(nRestarts-3) ) )))
    LR_schedule <- optax$join_schedules(LR_schedule,
                                        boundaries = jnp$array(ai(ceiling(nSGD_DefiningLRSeq / nRestarts * 1:(nRestarts-1) ))))
  }
  nSGD_MASTER <- nSGD_DefiningLRSeq
  #LR_schedule_vec <- np$array(  LR_schedule(jnp$array(1L:as.integer(nSGD_DefiningLRSeq) ) ))
  # optax schedules are vectorized, so evaluate the whole table in one call.
  # The per-step sapply cost 0.33 ms/step through reticulate (about 33 s per
  # 100k steps) for values only used in telemetry.
  LR_schedule_vec <- as.numeric(np$array(LR_schedule(jnp$arange(as.integer(nSGD_MASTER)))))

  if(T == T){ 
  optax_optimizer <- ndm_training_optimizer(LR_schedule)
  }
  if(T == F){ 
    optax_shampoo <- import("optax_shampoo")
    optax_optimizer = optax_shampoo$distributed_shampoo$distributed_shampoo(
      learning_rate=LR_schedule,
      block_size=128L,
      #beta1=0.9,
      #beta2=0.999,
      #diagonal_epsilon=1e-10,
      #matrix_epsilon=1e-6,
      #weight_decay=0.0,
      #start_preconditioning_step=1000,
      #preconditioning_compute_steps=1,
      #statistics_compute_steps=1,
      #best_effort_shape_interpretation=TRUE,
      #graft_type=distributed_shampoo.GraftingType.ADAGRAD,
      nesterov=TRUE,
      exponent_override=0
    )
  }

  # optimizer setup
  opt_state <- optax_optimizer$init(  eq$partition(ModelList, eq$is_array)[[1]]  )
  if (exists("ndm_runtime_replicate_tree", inherits = TRUE)) {
    opt_state <- ndm_runtime_replicate_tree(opt_state)
  }
  TrackBlockUpdateNorms <- isTRUE(get0("TrackBlockUpdateNorms", inherits = TRUE, ifnotfound = FALSE))
  PersistBlockUpdateNorms <- isTRUE(get0("PersistBlockUpdateNorms", inherits = TRUE, ifnotfound = FALSE))
  BlockUpdateTrackNames <- intersect(
    c("InitProcessList", "LocalNeural", "GlobalNeural", "ScaleList", "TSList", "BNList"),
    names(ModelList)
  )
  train_define_env <- environment()
  block_metric_norm <- function(tree) {
    leaves <- jax$tree_util$tree_leaves(tree)
    if (length(leaves) == 0L) {
      return(jnp$array(0.))
    }
    optax$global_norm(leaves)
  }
  block_update_log <- data.frame(
    iter = integer(0L),
    block = character(0L),
    param_norm = numeric(0L),
    grad_norm = numeric(0L),
    update_norm = numeric(0L),
    rel_update = numeric(0L),
    stringsAsFactors = FALSE
  )
  append_block_update_log <- function(iteration, block_metrics) {
    if (!TrackBlockUpdateNorms || is.null(block_metrics) || length(block_metrics) == 0L) {
      return(invisible(NULL))
    }
    py_scalar_num <- function(x) {
      value <- suppressWarnings(as.numeric(np$array(x)))
      if (length(value) == 0L) {
        return(NA_real_)
      }
      value[[1L]]
    }
    rows <- do.call(
      rbind,
      lapply(names(block_metrics), function(block_name) {
        metric <- block_metrics[[block_name]]
        data.frame(
          iter = as.integer(iteration),
          block = block_name,
          param_norm = py_scalar_num(metric$param_norm),
          grad_norm = py_scalar_num(metric$grad_norm),
          update_norm = py_scalar_num(metric$update_norm),
          rel_update = py_scalar_num(metric$rel_update),
          stringsAsFactors = FALSE
        )
      })
    )
    current_log <- get("block_update_log", envir = train_define_env, inherits = FALSE)
    updated_log <- rbind(current_log, rows)
    assign("block_update_log", updated_log, envir = train_define_env)
    if (PersistBlockUpdateNorms) {
      data.table::fwrite(
        updated_log,
        file.path(HolderFolder, sprintf("block_updates_i%s.csv", as.integer(iteration)))
      )
    }
    invisible(rows)
  }
  jit_apply_updates <- eq$filter_jit(optax$apply_updates)
  jit_get_update <- eq$filter_jit(optax_optimizer$update)
  train_step_impl <- function(ModelList,
                                                    batch_pkg,
                                                    y_true,
                                                    y_mask,
                                                    iteration,
                                                    state,
                                                    PriorList,
                                                    PolicyList,
                                                    GetPredSaveAtInfo,
                                                    seed,
                                                    opt_state) {
    loss_and_grads <- eq$filter_value_and_grad(getLoss_train, has_aux = T)(
      ModelList,
      batch_pkg,
      y_true,
      y_mask,
      iteration,
      state,
      PriorList,
      PolicyList,
      GetPredSaveAtInfo,
      seed
    )
    loss_and_state <- loss_and_grads[[1]]
    loss_aux <- loss_and_state[[2]]
    grads <- loss_and_grads[[2]]
    model_arrays <- eq$partition(ModelList, eq$is_array)
    model_array_tree <- model_arrays[[1]]
    grad_arrays <- eq$partition(grads, eq$is_inexact_array)[[1]]
    updates_and_state <- optax_optimizer$update(
      grad_arrays,
      opt_state,
      model_array_tree
    )
    block_update_metrics <- if (TrackBlockUpdateNorms && length(BlockUpdateTrackNames) > 0L) {
      metrics <- lapply(BlockUpdateTrackNames, function(block_name) {
        param_norm <- block_metric_norm(model_array_tree[[block_name]])
        grad_norm <- block_metric_norm(grad_arrays[[block_name]])
        update_norm <- block_metric_norm(updates_and_state[[1]][[block_name]])
        rel_update <- update_norm / jnp$maximum(
          param_norm,
          jnp$array(1e-12)$astype(param_norm$dtype)
        )
        list(
          "param_norm" = param_norm,
          "grad_norm" = grad_norm,
          "update_norm" = update_norm,
          "rel_update" = rel_update
        )
      })
      names(metrics) <- BlockUpdateTrackNames
      metrics
    } else {
      NULL
    }
    candidate_arrays <- optax$apply_updates(model_array_tree, updates_and_state[[1]])
    grad_norm <- optax$global_norm(jax$tree_util$tree_leaves(grad_arrays))
    accepted <- jnp$logical_and(jnp$isfinite(loss_and_state[[1]]), jnp$isfinite(grad_norm))
    accepted <- jnp$logical_and(accepted, jnp$all(loss_aux$solver_diagnostics$success))
    accepted <- jnp$logical_and(accepted, ndm_training_tree_finite(
      list(candidate_arrays, updates_and_state[[2]], loss_aux$model_state)
    ))
    # Observation masks are validated on the device. A batch with a non-finite
    # or all-zero mask is rejected in-graph, and the host raises the detailed
    # error only then, instead of copying the mask back every iteration.
    observation_mask_valid <- ndm_training_observation_mask_valid(y_mask)
    accepted <- jnp$logical_and(accepted, observation_mask_valid)
    # Donation consumes the input handles even on rejection. Return unchanged
    # values through the compiled boundary so R can still diagnose failures.
    chosen <- jax$lax$cond(
      accepted, function(pair) pair[[1L]], function(pair) pair[[2L]],
      list(list(candidate_arrays, updates_and_state[[2]]), list(model_array_tree, opt_state))
    )
    updated_model <- eq$combine(chosen[[1L]], model_arrays[[2]])
    list(
      "loss" = loss_and_state[[1]],
      "state" = loss_aux$model_state,
      "solver_diagnostics" = loss_aux$solver_diagnostics,
      "loss_components" = loss_aux$loss_components,
      "grad_norm" = grad_norm,
      "update_accepted" = accepted,
      "model" = updated_model,
      "opt_state" = chosen[[2L]],
      "block_update_metrics" = block_update_metrics,
      # Packed copies of the per-step scalars and diagnostics: the host reads
      # these in three transfers rather than one per field.
      "solver_diagnostics_packed" = ndm_training_pack_diagnostics(loss_aux$solver_diagnostics),
      "loss_components_packed" = ndm_training_pack_scalars(
        loss_aux$loss_components, ndm_training_loss_component_fields, loss_and_state[[1]]$dtype
      ),
      "scalars_packed" = jnp$stack(list(
        loss_and_state[[1]]$astype(jnp$float32),
        grad_norm$astype(jnp$float32),
        accepted$astype(jnp$float32),
        observation_mask_valid$astype(jnp$float32)
      ), axis = 0L)
    )
  }
  DonateTrainingState <- get0("DonateTrainingState", inherits = TRUE, ifnotfound = TRUE)
  if (!is.logical(DonateTrainingState) || length(DonateTrainingState) != 1L || is.na(DonateTrainingState)) {
    stop("DonateTrainingState must be one non-missing logical value.", call. = FALSE)
  }
  train_step_owned_compiled <- switch_filter_jit(function(fixed, owned) {
    do.call(train_step_impl, c(list(ModelList = owned$model), fixed, list(opt_state = owned$opt_state)))
  }, donate = if (DonateTrainingState) "all-except-first" else "none")
  train_step_compiled <- function(ModelList, batch_pkg, y_true, y_mask, iteration,
                                  state, PriorList, PolicyList, GetPredSaveAtInfo, seed, opt_state) {
    train_step_owned_compiled(
      list(batch_pkg = batch_pkg, y_true = y_true, y_mask = y_mask, iteration = iteration,
           state = state, PriorList = PriorList, PolicyList = PolicyList,
           GetPredSaveAtInfo = GetPredSaveAtInfo, seed = seed),
      list(model = ModelList, opt_state = opt_state)
    )
  }
}

# perform main training sequence
NA20 <- function(zer){zer[is.na(zer)] <- 0;zer[is.infinite(zer)] <- 0;zer}
grad_norm_vec <- out_loss_vec <- in_loss_vec <- rep(NA, times = nSGD_DefiningLRSeq );i_<-1
grad_norm_mat <- c()
LastOutCor <- Skill8SanityCheck <- NA
te_total <- te_grads <- 0
st0 <- Sys.time()
plottingSeq_counter <- ExecuteUpdateCounter <- 0; GradNorm_jit <- jax$jit(optax$global_norm)
crossIterCor_vec <- c()

# create save directory for results
saved_model_run_id <- get0(
  "RUN_ID",
  inherits = TRUE,
  ifnotfound = paste0("legacy_outer", get0("OUTER_ITERATION", inherits = TRUE, ifnotfound = "unknown"))
)
SavedModelDir <- file.path(
  "./SavedModels",
  ifelse(isTRUE(SimMode), "FromSim", "FromReal"),
  sprintf("Model_%s_%s", AnalysisName, saved_model_run_id)
)
if(dir.exists(SavedModelDir)){
  SavedModelDir <- sprintf(
    "%s_retry_%s_%s",
    SavedModelDir,
    format(Sys.time(), "%Y%m%dT%H%M%S"),
    Sys.getpid()
  )
}
dir.create(SavedModelDir, recursive = TRUE, showWarnings = FALSE)

# calculate total parameter number
print2(sprintf("Total trainable parameter count: %s",
               nParams <- sum(unlist(lapply(jax$tree$leaves( eq$partition(ModelList, eq$is_array)[[1]] ), function(zer){zer$size})))))
print("Done with SuperLModel_TrainDefine.R")
