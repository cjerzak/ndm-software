# Content of SuperLModel_TrainDefine.R
{
  print("Sarting SuperLModel_TrainDefine.R")
  # sort( sapply(ls(), function(zr){ object.size(eval(parse(text = zr))) }) )
  # if(!SimMode){ save(input_df_red_full, file = "./tmp_input_df_red_full.Rdata"); rm ( input_df_red_full ) }
  saveCheckpointCounter <- outSampCounter <- 0;
  nRestarts <- 1L; LR_schedule <- optax$warmup_cosine_decay_schedule(warmup_steps = 
                                                                      (nWarmup <- max(c(min(c(100L,
                                                                                              nSGD_DefiningLRSeq)),
                                                                                              0.1*nSGD_DefiningLRSeq))),
                                                                     decay_steps = max(c(101L,nSGD_DefiningLRSeq-nWarmup)),
                                                                     init_value = LEARNING_RATE_MAX/100, 
                                                                     peak_value = LEARNING_RATE_MAX, 
                                                                     end_value =  LEARNING_RATE_MAX/100)
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
  LR_schedule_vec <- sapply(1:nSGD_MASTER, function(x_){ np$array(  LR_schedule(jnp$array(x_) ))})

  if(T == T){ 
  optax_optimizer <-  optax$chain(
    #optax$clip(1),
    optax$adaptive_grad_clip( 0.1, eps = 0.0001 ),
    optax$adabelief( learning_rate = LR_schedule, eps = 1e-6, eps_root = 1e-6 ) 
    #optax$adamw( learning_rate = LR_schedule  ) 
  )
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
  train_step_compiled <- switch_filter_jit(function(ModelList,
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
    updated_model <- eq$combine(
      optax$apply_updates(model_array_tree, updates_and_state[[1]]),
      model_arrays[[2]]
    )
    list(
      "loss" = loss_and_state[[1]],
      "state" = loss_aux$model_state,
      "solver_diagnostics" = loss_aux$solver_diagnostics,
      "grad_norm" = optax$global_norm(jax$tree_util$tree_leaves(grad_arrays)),
      "model" = updated_model,
      "opt_state" = updates_and_state[[2]],
      "block_update_metrics" = block_update_metrics
    )
  })
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
