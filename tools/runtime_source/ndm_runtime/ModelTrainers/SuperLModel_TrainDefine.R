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
  plot( LR_schedule_vec )

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
    grads <- loss_and_grads[[2]]
    model_arrays <- eq$partition(ModelList, eq$is_array)
    grad_arrays <- eq$partition(grads, eq$is_inexact_array)[[1]]
    updates_and_state <- optax_optimizer$update(
      grad_arrays,
      opt_state,
      model_arrays[[1]]
    )
    updated_model <- eq$combine(
      optax$apply_updates(model_arrays[[1]], updates_and_state[[1]]),
      model_arrays[[2]]
    )
    list(
      "loss" = loss_and_state[[1]],
      "state" = loss_and_state[[2]],
      "grad_norm" = optax$global_norm(jax$tree_util$tree_leaves(grad_arrays)),
      "model" = updated_model,
      "opt_state" = updates_and_state[[2]]
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
SavedModelDir <- sprintf("./SavedModels/%s/Model_%s_%s",
                         ifelse(grepl(tolower(AnalysisName), pattern = "sim"),
                                yes = "FromSim",no="FromReal"),
                         AnalysisName,
                         AnalysisDate)
if(dir.exists(SavedModelDir)){
  SavedModelDir <- sprintf("%s_%s", SavedModelDir, format(Sys.time(), "%H-%M-%S"))
}
dir.create(SavedModelDir, recursive = TRUE, showWarnings = FALSE)

# calculate total parameter number
print2(sprintf("Total trainable parameter count: %s",
               nParams <- sum(unlist(lapply(jax$tree$leaves( eq$partition(ModelList, eq$is_array)[[1]] ), function(zer){zer$size})))))
print("Done with SuperLModel_TrainDefine.R")
