# Content of SuperLModel_GetAnalytics_Real.R
ndm_inference_draw_plan <- function(model_type,
                                    neuralode_variational,
                                    requested_draws = 5L,
                                    test_without_sampling = FALSE) {
  requested_draws <- suppressWarnings(as.numeric(requested_draws))
  if (length(requested_draws) != 1L ||
      !is.finite(requested_draws) ||
      requested_draws < 1 ||
      requested_draws > .Machine$integer.max ||
      requested_draws != floor(requested_draws)) {
    stop("`InferenceMCDraws` must be one positive integer.", call. = FALSE)
  }
  requested_draws <- as.integer(requested_draws)
  stochastic <- identical(as.character(model_type), "NeuralODE") &&
    isTRUE(neuralode_variational) &&
    !isTRUE(test_without_sampling)
  list(
    requested = requested_draws,
    effective = if (stochastic) requested_draws else 1L,
    stochastic = stochastic,
    key_scheme = "jax-fold-in-v1"
  )
}

ndm_inference_key_lineage <- function(run_seed,
                                      outer_iteration,
                                      checkpoint_step,
                                      draw_id,
                                      location_ids,
                                      time_anchor_ids) {
  scalar_id <- function(value, name) {
    value <- suppressWarnings(as.numeric(value))
    if (length(value) != 1L ||
        !is.finite(value) ||
        value < 0 ||
        value > .Machine$integer.max ||
        value != floor(value)) {
      stop(sprintf("`%s` must be one non-negative integer.", name), call. = FALSE)
    }
    as.integer(value)
  }
  location_ids <- suppressWarnings(as.numeric(location_ids))
  time_anchor_ids <- suppressWarnings(as.numeric(time_anchor_ids))
  if (length(location_ids) != length(time_anchor_ids) ||
      any(!is.finite(location_ids)) ||
      any(!is.finite(time_anchor_ids)) ||
      any(location_ids < 0) ||
      any(time_anchor_ids < 0) ||
      any(location_ids > .Machine$integer.max) ||
      any(time_anchor_ids > .Machine$integer.max) ||
      any(location_ids != floor(location_ids)) ||
      any(time_anchor_ids != floor(time_anchor_ids))) {
    stop(
      "Inference location and anchor identifiers must be aligned non-negative integers.",
      call. = FALSE
    )
  }
  shared <- c(
    104729L,
    scalar_id(outer_iteration, "outer_iteration"),
    scalar_id(checkpoint_step, "checkpoint_step"),
    scalar_id(draw_id, "draw_id")
  )
  list(
    run_seed = scalar_id(run_seed, "run_seed"),
    global = c(shared, 130363L),
    local = lapply(
      seq_along(location_ids),
      function(row_id) {
        c(
          shared,
          155921L,
          as.integer(location_ids[[row_id]]),
          173867L,
          as.integer(time_anchor_ids[[row_id]])
        )
      }
    )
  )
}

ndm_inference_scoped_keys <- function(jax,
                                      jnp,
                                      JaxKey,
                                      run_seed,
                                      outer_iteration,
                                      checkpoint_step,
                                      draw_id,
                                      location_ids,
                                      time_anchor_ids) {
  lineage <- ndm_inference_key_lineage(
    run_seed = run_seed,
    outer_iteration = outer_iteration,
    checkpoint_step = checkpoint_step,
    draw_id = draw_id,
    location_ids = location_ids,
    time_anchor_ids = time_anchor_ids
  )
  fold_lineage <- function(values) {
    key <- JaxKey(lineage$run_seed)
    for (value in values) {
      key <- jax$random$fold_in(key, as.integer(value))
    }
    key
  }
  global_key <- fold_lineage(lineage$global)
  row_keys <- lapply(
    lineage$local,
    function(local_lineage) {
      jnp$stack(
        list(
          fold_lineage(local_lineage),
          global_key
        ),
        axis = 0L
      )
    }
  )
  jnp$stack(row_keys, axis = 0L)
}

ndm_stop_incomplete_inference_draws <- function(draw_results,
                                                requested,
                                                checkpoint_step) {
  failures <- which(!vapply(draw_results, function(result) isTRUE(result$ok), logical(1)))
  if (length(failures) == 0L) {
    return(invisible(TRUE))
  }
  first_error <- draw_results[[failures[[1]]]]$error
  if (inherits(first_error, "ndm_solver_failure")) {
    stop(first_error)
  }
  message <- sprintf(
    "Inference checkpoint %s completed %s of %s required posterior draws.",
    checkpoint_step,
    length(draw_results) - length(failures),
    requested
  )
  condition <- structure(
    list(
      message = message,
      call = NULL,
      failed_draws = as.integer(failures),
      requested_draws = as.integer(requested),
      checkpoint_step = as.integer(checkpoint_step),
      parent = first_error
    ),
    class = c("ndm_inference_draw_failure", "error", "condition")
  )
  stop(condition)
}

ndm_mean_inference_draws <- function(draws, extract = identity) {
  if (length(draws) == 0L) {
    stop("At least one successful inference draw is required.", call. = FALSE)
  }
  Reduce(`+`, lapply(draws, extract)) / length(draws)
}

ndm_inference_host_array <- function(value, np) {
  as.vector(as.array(np$array(value)))
}

ndm_validate_inference_solver_diagnostics <- function(prediction,
                                                      draw_id,
                                                      location_ids,
                                                      time_anchor_ids,
                                                      host_array = as.vector,
                                                      require_diagnostics = TRUE) {
  diagnostics <- prediction$solver_diagnostics
  if (is.null(diagnostics)) {
    if (isTRUE(require_diagnostics)) {
      stop(
        "Current-runtime inference did not return required solver diagnostics.",
        call. = FALSE
      )
    }
    return(invisible(FALSE))
  }
  host_field <- function(name, default = NA) {
    value <- diagnostics[[name]]
    if (is.null(value)) {
      return(rep(default, length(location_ids)))
    }
    value <- host_array(value)
    if (length(value) == 1L && length(location_ids) > 1L) {
      value <- rep(value, length(location_ids))
    }
    value
  }
  success <- as.logical(host_field("success", FALSE))
  prediction_finite <- as.logical(host_field("prediction_finite", FALSE))
  if (length(success) != length(location_ids) ||
      length(prediction_finite) != length(location_ids)) {
    stop(
      "Solver diagnostics are not aligned with the inference batch.",
      call. = FALSE
    )
  }
  failed <- which(is.na(success) | !success |
                    is.na(prediction_finite) | !prediction_finite)
  if (length(failed) == 0L) {
    return(invisible(TRUE))
  }

  report <- data.frame(
    batch_row = as.integer(failed),
    location_id = as.integer(location_ids[failed]),
    time_anchor_id = as.integer(time_anchor_ids[failed]),
    failure_stage_code = as.integer(host_field("failure_stage_code")[failed]),
    prediction_finite = prediction_finite[failed],
    global_attempted = as.logical(host_field("global_attempted", FALSE)[failed]),
    global_result_success = as.logical(host_field("global_result_success", FALSE)[failed]),
    global_state_finite = as.logical(host_field("global_state_finite", FALSE)[failed]),
    global_result_code = as.integer(host_field("global_result_code")[failed]),
    global_num_steps = as.integer(host_field("global_num_steps")[failed]),
    global_num_accepted_steps = as.integer(
      host_field("global_num_accepted_steps")[failed]
    ),
    global_num_rejected_steps = as.integer(
      host_field("global_num_rejected_steps")[failed]
    ),
    global_max_steps = as.integer(host_field("global_max_steps")[failed]),
    local_attempted = as.logical(host_field("local_attempted", FALSE)[failed]),
    local_result_success = as.logical(host_field("local_result_success", FALSE)[failed]),
    local_state_finite = as.logical(host_field("local_state_finite", FALSE)[failed]),
    local_result_code = as.integer(host_field("local_result_code")[failed]),
    local_num_steps = as.integer(host_field("local_num_steps")[failed]),
    local_num_accepted_steps = as.integer(
      host_field("local_num_accepted_steps")[failed]
    ),
    local_num_rejected_steps = as.integer(
      host_field("local_num_rejected_steps")[failed]
    ),
    local_max_steps = as.integer(host_field("local_max_steps")[failed]),
    stringsAsFactors = FALSE
  )
  stage_names <- c(
    "0" = "none",
    "1" = "global_ode",
    "2" = "local_ode",
    "3" = "prediction"
  )
  report$failure_stage <- unname(stage_names[as.character(report$failure_stage_code)])
  report$failure_stage[is.na(report$failure_stage)] <- "unknown"

  message <- sprintf(
    "Inference posterior draw %s failed solver validation for %s batch row(s).",
    draw_id,
    nrow(report)
  )
  condition <- structure(
    list(
      message = message,
      call = NULL,
      stage = "inference",
      draw_id = as.integer(draw_id),
      solver_report = report
    ),
    class = c("ndm_solver_failure", "error", "condition")
  )
  stop(condition)
}

if( !SimMode ){
    print("Starting SuperLModel_GetAnalytics_Real.R")
    ndm_plot_log_series_safe <- utils::getFromNamespace(".ndm_plot_log_series_safe", "ndm")
    inference_draw_plan <- ndm_inference_draw_plan(
      model_type = ModelType,
      neuralode_variational = get0(
        "neuralode_variational",
        inherits = TRUE,
        ifnotfound = identical(ModelType, "NeuralODE")
      ),
      requested_draws = get0("InferenceMCDraws", inherits = TRUE, ifnotfound = 5L),
      test_without_sampling = get0(
        "testWithoutSampling",
        inherits = TRUE,
        ifnotfound = FALSE
      )
    )
    inference_mc_draws_requested <- inference_draw_plan$requested
    inference_mc_draws_effective <- inference_draw_plan$effective
    inference_key_scheme <- paste0(
      inference_draw_plan$key_scheme,
      ":run>outer>checkpoint>draw;",
      "local>location>anchor;global>shared-draw"
    )
    allow_legacy_missing_solver_diagnostics <- isTRUE(get0(
      "AllowLegacyMissingSolverDiagnostics",
      inherits = TRUE,
      ifnotfound = FALSE
    ))
    # get out-of-sample results
    #try(plot(rank(na.omit(in_loss_vec))), T) ; 
    #try(points(smooth.spline(rank(na.omit(in_loss_vec))),type = "l",lwd=3), T) 
  
    # reset iterator and do analysis 
    gc(); py_gc$collect()
    TFDatasetIterator_inference <- reticulate::as_iterator( TFDataset_inference )
    next_inference_tfrecord_batch <- utils::getFromNamespace(
      ".ndm_tfrecord_iter_next",
      "ndm"
    )
    inference_batch_dims <- function(batch_l){
      if(is.null(batch_l) || length(batch_l) == 0L){
        return(integer())
      }
      vapply(batch_l, function(l_){ as.integer(np$array(l_$shape)[[1]]) }, integer(1))
    }
    sl_dat <- c();ok_<-F; ok_counter_ <- 0; while(!ok_){ # !ok_ means do a batch
          ok_counter_ <- ok_counter_ + 1
          next_batch <- next_inference_tfrecord_batch(TFDatasetIterator_inference)
          batch_l <- next_batch$batch
          
          # reasons not to go on 
          if(isTRUE(next_batch$completed)){ ok_ <- T }
          if( !ok_ ){
            batch_dims <- inference_batch_dims(batch_l)
            if(length(batch_dims) == 0L || batch_dims[[1]] == 0L){
              stop("Malformed empty inference batch in GetAnalytics_Real.R")
            }
            if(!ok_ && any(batch_dims != batch_dims[[1]])){
              stop("Inconsistent field dimensions in inference batch in GetAnalytics_Real.R")
            }
            if(!ok_){
              batch_l <- TFConst2JAXArray( batch_l )
            }
          }
          
          # if going in do this
          if(!ok_){
          print2(sprintf("GetAnalytics_Real [@ sgd iter: %s, analytics iter: %s]", i, ok_counter_))
          #plot(as.matrix(batch_l$XPred_Mask[1,,])[,100],as.matrix(batch_l$XPred[1,,])[,100])
          #input_df_red_full[input_df_red_full$location_id %in% loc_id & input_df_red_full$time_id %in% time_id,]
          if(!("try-error" %in% class(batch_l))){
            #if(!is.null(batch_l) & !all(batch_l$is_null_indicator) & (length(batch_l_prior)>0)){
            #if(!is.null(batch_l)  & (length(batch_l_prior)>0)){
            #if(place_counter__ == 1 & time_counter__ == 1 & outSampCounter == 1){
            {
                # lapply(unlist(batch2package(batch_l)), function(l_){paste0(l_$shape)})
                 inference_location_ids <- as.integer(c(np$array(batch_l$location_id_numeric)))
                 inference_time_anchor_ids <- as.integer(c(np$array(batch_l$time_id_numeric)))
                 draw_results <- lapply(
                   seq_len(inference_mc_draws_effective),
                   function(draw_id) {
                     tryCatch(
                       {
                         scoped_keys <- ndm_inference_scoped_keys(
                           jax = jax,
                           jnp = jnp,
                           JaxKey = JaxKey,
                           run_seed = SEED_,
                           outer_iteration = OUTER_ITERATION,
                           checkpoint_step = i,
                           draw_id = draw_id,
                           location_ids = inference_location_ids,
                           time_anchor_ids = inference_time_anchor_ids
                         )
                         scoped_keys <- ndm_runtime_data_to_device(scoped_keys)
                         prediction_raw <- GetPred_inference(
                           ModelList,
                           batch2package(batch_l),
                           state,
                           PriorList,
                           PolicyList,
                           GetPredSaveAtInfo_default,
                           scoped_keys
                         )
                         prediction <- prediction_raw[[1]]
                         ndm_validate_inference_solver_diagnostics(
                           prediction = prediction,
                           draw_id = draw_id,
                           location_ids = inference_location_ids,
                           time_anchor_ids = inference_time_anchor_ids,
                           host_array = function(value) {
                             ndm_inference_host_array(value, np)
                           },
                           require_diagnostics =
                             !allow_legacy_missing_solver_diagnostics
                         )
                         list(ok = TRUE, value = prediction, error = NULL)
                       },
                       error = function(error) {
                         list(ok = FALSE, value = NULL, error = error)
                       }
                     )
                   }
                 )
                 ndm_stop_incomplete_inference_draws(
                   draw_results,
                   requested = inference_mc_draws_effective,
                   checkpoint_step = i
                 )
                 add_pred_all <- lapply(draw_results, `[[`, "value")
                 plot( np$array(add_pred_all[[1]]$y_mu)[,,1], 
                       np$array(batch_l$YTrue_out)[,,1],
                       main = sprintf("Just1 at %s [Cor: %.3f]",ok_counter_,
                               cor(c(np$array(add_pred_all[[1]]$y_mu)[,,1]),
                                   c(np$array(batch_l$YTrue_out)[,,1]))));abline(a=0,b=1)
                 # plot( np$array(add_pred_all[[1]]$y_mu)[,,1], np$array(batch_l_train$YTrue_out)[,,1]);abline(a=0,b=1)
                 # add_pred_all[[1]]$y_mu$shape
                 # batch_l$XPred$shape
                 # batch_l$XPred_mask
                 # batch_l$XPred$mean(axis = 0L:1L)
  
                # analyze time dependent parameters
                at_t <- try(colMeans(do.call(rbind,lapply(add_pred_all,function(zer){
                                        c(ExtractBetaDraw(zer$ODEParamsSampList)$TSDraw) }))),T)
                beta_hat_at_t <- try(colMeans(do.call(rbind,lapply(add_pred_all,function(zer){c(ExtractBetaDraw(zer$ODEParamsSampList)$BetaDraw)}))),T)
                s_average <- try(colMeans(do.call(rbind,lapply(add_pred_all,function(zer){c(np$asanyarray(zer$ODEParamsSampList$s_l_samp) )}))),T)
                e_average <- try(colMeans(do.call(rbind,lapply(add_pred_all,function(zer){c(np$asanyarray(zer$ODEParamsSampList$e_l_samp) )}))),T)
                i_average <- try(colMeans(do.call(rbind,lapply(add_pred_all,function(zer){c(np$asanyarray(zer$ODEParamsSampList$i_l_samp) )}))),T)
                r_average <- try(colMeans(do.call(rbind,lapply(add_pred_all,function(zer){c(np$asanyarray(zer$ODEParamsSampList$r_l_samp) )}))),T)
  
                add_sd_all <- colMeans(do.call(rbind,lapply(add_pred_all,function(zer){np$asanyarray(zer$y_sigma) })))
                #if(mean(abs(add_sd_all)) < 0.01){browser() }
                # plot(  sort(  input_df_red_full[input_df_red_full$location_id %in% loc_id,]$time_id ) )
                # plot(  sort(  input_df_red_full[input_df_red_full$location_id %in% place_pool,]$time_id ) )
                
                #mean(input_df_red_in$time_id %in% input_df_red_full$time_id)
                if(!UseShortOutcomes){ 
                  add_sd_past <- t(add_sd_all[c(1:abs(maxTimesPast))])
                  add_sd_out <- t(add_sd_all[-c(1:abs(maxTimesPast))])
                }
                if(UseShortOutcomes){ 
                  add_sd_past <- t(add_sd_all); add_sd_out <- t(add_sd_all)
                }
  
                # obtain posterior predicted mean
                add_pred_all_reduced <- ndm_mean_inference_draws(
                  add_pred_all,
                  extract = function(l_){np$asanyarray(l_$y_mu)[,,1]}
                )
                add_pred_all_reduced[add_pred_all_reduced<0] <- 0
  
                if(UseShortOutcomes){ 
                  add_pred_past <- (add_pred_all_reduced)
                  add_pred_out <- (add_pred_all_reduced)
                }
                if(!UseShortOutcomes){ 
                  add_pred_past <- (add_pred_all_reduced[,c(1:abs(maxTimesPast))])
                  add_pred_out <- (add_pred_all_reduced[,-c(1:abs(maxTimesPast))])
                }
  
                # subset truth (no need to change subsetting here for short outcomes [using YTrue])
                add_true_prior <- (np$asanyarray(batch_l$YTrue)[,c(1:abs(maxTimesPast)),])
                add_true_prior_mask <- (np$asanyarray(batch_l$YTrue_mask)[,c(1:abs(maxTimesPast)),])
                
                add_true_out <- (np$asanyarray(batch_l$YTrue)[,-c(1:abs(maxTimesPast)),])
                # max(abs(add_true_out - np$asanyarray(batch_l$YTrue_out)[,,1])) # sanity 0 value 
                add_true_out_mask <- (np$asanyarray(batch_l$YTrue_mask)[,-c(1:abs(maxTimesPast)),])
                add_true_prior[add_true_prior_mask == 0] <- NA
                add_true_out[add_true_out_mask == 0] <- NA
                
                if(length(dim(add_true_out)) > 2 & tail(dim(add_true_out),1)>1){ add_true_out <- add_true_out[,,1] }
                if(length(dim(add_true_prior)) > 2 & tail(dim(add_true_prior),1)>1){ add_true_prior <- add_true_prior[,,1] }
                # View(add_true_out[,,1])
                # View(add_true_out_mask[,,1])
  
                # barplot
                #par(mfrow = c(1,1))
                #seir_ <- c(s_average,e_average,i_average,r_average)
                #names(seir_) <- c("S[0]","E[0]","I[0]","R[0]")
                #barplot(seir_ , add = F, density = 0.3, horiz  = F)
  
                # beta plot
                #try(plot(at_t , beta_hat_at_t,type = "b",xlab="Time (t) in Times"),T)
              }
              if("try-error" %in% class(add_true_prior)){
                add_true_prior <- add_true_out
                add_true_prior[] <- NA
              }
              # par(mfrow=c(1,1));plot(x_<-c(add_true_out), y_<-c(add_pred_out));abline(a=0,b=1); print(cor(x_,y_, use = "pairwise.complete"))
              colnames(add_pred_out) <- c(paste("Pred_l",1:ncol(add_pred_out),sep=""))
              colnames(add_true_out) <- c(paste("Truth_l",1:ncol(add_true_out),sep=""))
              colnames(add_true_prior) <- c(paste("PreviousSequenceTruth_l",1:ncol(add_true_prior),sep=""))
              
              # Persistence baseline: repeat each sequence's most recent finite
              # observed context value across every forecast horizon.
              add_true_prior_matrix <- as.matrix(add_true_prior)
              last_observed_context <- apply(
                add_true_prior_matrix,
                1L,
                function(values) {
                  values <- values[is.finite(values)]
                  if (length(values)) tail(values, 1L) else NA_real_
                }
              )
              add_pred_out_baseline <- matrix(
                rep(last_observed_context, times = ncol(add_pred_out)),
                nrow = nrow(add_pred_out),
                ncol = ncol(add_pred_out)
              )
              colnames(add_pred_out_baseline) <- colnames(add_pred_out)
              colnames(add_pred_out_baseline) <- gsub(colnames(add_pred_out_baseline),
                                                   pattern = "Pred", replace = "PredBase")
            
              # jnp$mean(jnp$array(batch_l$XPred),axis = 0L:1L)
              # jnp$std(jnp$array(batch_l$XPred),axis = 0L:1L)
              plot(add_true_out, add_pred_out, main=sprintf("%s [Cor: %.3f]",ok_counter_, LastOutCor <- cor(c(add_true_out),c(add_pred_out)))); abline(a=0,b=1)
              
              # combine results 
              sl_dat <- rbind(sl_dat, cbind("location_id" = np$array(batch_l$location_id_numeric),
                                         "time_anchor_id" = np$array(batch_l$time_id_numeric),#[!batch_l$is_null_indicator],
                                         cbind(add_true_out,
                                               add_pred_out,
                                               add_pred_out_baseline,
                                               add_true_prior) ) )
          configured_horizon <- suppressWarnings(as.integer(
            if (exists("RealEntry", inherits = TRUE) &&
                "evaluationHorizon" %in% names(RealEntry)) {
              RealEntry$evaluationHorizon[[1L]]
            } else {
              ncol(add_pred_out)
            }
          ))
          if (!is.finite(configured_horizon) || configured_horizon < 1L) {
            configured_horizon <- ncol(add_pred_out)
          }
          available_horizons <- seq_len(min(
            configured_horizon,
            ncol(add_pred_out)
          ))
          truth_skill <- unlist(lapply(
            paste0("Truth_l", available_horizons),
            function(column) f2n(sl_dat[, column])
          ), use.names = FALSE)
          pred_skill <- unlist(lapply(
            paste0("Pred_l", available_horizons),
            function(column) f2n(sl_dat[, column])
          ), use.names = FALSE)
          baseline_skill <- unlist(lapply(
            paste0("PredBase_l", available_horizons),
            function(column) f2n(sl_dat[, column])
          ), use.names = FALSE)
          skill_cells <- is.finite(truth_skill) & is.finite(pred_skill) &
            is.finite(baseline_skill)
          ForecastSkillSanityCheck <- if (any(skill_cells)) {
            1 - (sum((truth_skill[skill_cells] - pred_skill[skill_cells])^2) + 0.01) /
              (sum((truth_skill[skill_cells] - baseline_skill[skill_cells])^2) + 0.01)
          } else {
            NA_real_
          }
          # Retain the legacy output name while making its calculation honor
          # the configured horizon instead of hard-coding lead eight.
          Skill8SanityCheck <- ForecastSkillSanityCheck
          print2(sprintf(
            "Persistence skill sanity across %s horizon(s): %.3f",
            length(available_horizons),
            ForecastSkillSanityCheck
          ))
          }
        }
    }
  
    sl_dat <- eval(parse(text = sprintf('cbind(sl_dat,
                    %s,
                    "modelingStrategy_name" = modelingStrategyNameKey,
                    "nSGD" = nSGD_model,
                    "nSGDPolicy" = get0("nSGDPolicy", inherits = TRUE, ifnotfound = NA_character_),
                    "nSGDAnchorMaxSamplesTrain" = get0("nSGDAnchorMaxSamplesTrain", inherits = TRUE, ifnotfound = NA_integer_),
                    "nSGDAnchorScope" = get0("nSGDAnchorScope", inherits = TRUE, ifnotfound = NA_character_),
                    "nSGDAnchorBatch" = get0("nSGDAnchorBatch", inherits = TRUE, ifnotfound = NA_integer_),
                    "inference_mc_draws_requested" = inference_mc_draws_requested,
                    "inference_mc_draws_effective" = inference_mc_draws_effective,
                    "inference_key_scheme" = inference_key_scheme,
                    "nBatch" = nBatch,
                    "maxTimesPast" = maxTimesPast, # past context
                    "evaluationTime" = evaluationTime,
                    "evaluationMethod" = evaluationMethod,
                    "OUTER_ITERATION" = OUTER_ITERATION,
                    "i_in_sgd" = i,
                    "te_total" = te_total, 
                    "te_grads" = te_grads, 
                    "nTrainingSamplesSeen" = i*nBatch, 
                    "Skill8SanityCheck" = Skill8SanityCheck, 
                    "ForecastSkillSanityCheck" = get0("ForecastSkillSanityCheck", inherits = TRUE, ifnotfound = NA_real_),
                    "atEpoch" = i*nBatch/nSamplesTrain, 
                    "nParamsModel" = nParamsModel, 
                    "maxInSampleTime_id" = max( input_df_red_in$time_id ),
                    "model_id" = rlang::hash(modelingStrategyNameKey) )',
                      paste(paste("'",names(unlist(RealEntry)), "'='", unlist(RealEntry),"'", sep=""),collapse=",")
                      ) ))
    sl_dat <- as.data.frame( sl_dat )
    sl_dat$time_anchor_id <- unlist( sl_dat$time_anchor_id )
    #plot(abs(f2n(sl_dat$PredBase_l6) - f2n(sl_dat$Truth_l6)), abs(f2n(sl_dat$Pred_l6) - f2n(sl_dat$Truth_l6))); abline(a=0,b=1)
    #plot(abs(f2n(sl_dat$PredBase_l12) - f2n(sl_dat$Truth_l12)), abs(f2n(sl_dat$Pred_l12) - f2n(sl_dat$Truth_l12))); abline(a=0,b=1)

    #summary(abs(f2n(sl_dat$PredBase_l4) - f2n(sl_dat$Truth_l4)))
    #summary(abs(f2n(sl_dat$Pred_l4) - f2n(sl_dat$Truth_l4)))

    #summary(abs(f2n(sl_dat$PredBase_l6) - f2n(sl_dat$Truth_l6)))
    #summary(abs(f2n(sl_dat$Pred_l6) - f2n(sl_dat$Truth_l6)))

    #summary(abs(f2n(sl_dat$PredBase_l12) - f2n(sl_dat$Truth_l12)))
    #summary(abs(f2n(sl_dat$Pred_l12) - f2n(sl_dat$Truth_l12)))

    #np$array(ModelList[[1]][[3]][[1]])
    #np$array(ModelList[[1]][[3]][[2]])
    #plot( predict(prcomp(np$array(ModelList[[1]][[3]][[1]])))[,1:2])
    #plot( predict(prcomp(np$array(ModelList[[1]][[3]][[2]])))[,1:2])
    #tapply(truth_df_red$location_id_numeric,truth_df_red$location_id_numeric,unique)

    #  write loss fig for debugging (note: files are re-written)
    pdf(file.path(
      HolderFolder,
      sprintf("diagnostics_%s_EvalTime%s_IsPretraining%s_OuterKey%s.pdf",
              rlang::hash(modelingStrategyNameKey),
              evaluationTime,
              IsPretraining,
              OUTER_ITERATION)
    ), height = 10, width = 5)
    {
    par(mfrow=c(2,1 + is.na(COMMAND_ARG_INPUT) ))
    loss_obs <- na.omit(in_loss_vec)
    grad_obs <- na.omit(grad_norm_vec)
    ndm_plot_log_series_safe(loss_obs, main = "Loss")
    if(length(loss_obs) > 0L){ plot(rank(loss_obs), log = "") } else { plot.new() }
    ndm_plot_log_series_safe(grad_obs, main = "Gradients")
    if(is.na(COMMAND_ARG_INPUT)){
      tmp555 <- as.matrix(tmp555);try(plot(tmp555[1,],ylim = c(0,max(c(tmp555[1:10,]))),type = 'l',main="Y-hat"),T)
      for(i3 in 2:10){ try(points(tmp555[i3,],type = 'l',col = "black", lwd = 2, lty = 1),T) }
    }
    }
    dev.off(); par(mfrow = c(1,1))
    
    # save results 
    data.table::fwrite(
      as.matrix(sl_dat),
      file = as.character(file.path(
        HolderFolder,
        sprintf("predicted_df_out_%s_EvalTime%s_IsPretraining%s_TrainIter%sof%s_OuterIter%s.csv",
                rlang::hash(modelingStrategyNameKey),
                evaluationTime,
                IsPretraining,
                i,
                nSGD_model,
                OUTER_ITERATION)
      ))
    )
    print("Done with SuperLModel_GetAnalytics_Real.R")
}

# load(file = "./tmp_input_df_red_full.Rdata");
# full sample
#place_pool <- sample(unique(input_df_red_out$location_id[input_df_red_out$location_id %in% input_df_red_in$location_id]))

# partial sample
# place_pool <- sample(unique(input_df_red_out$location_id[input_df_red_out$location_id %in% input_df_red_in$location_id]), 50)
# place_counter__ <- 0; for(loc_id in place_pool){
# out of sample targets - how to  gen the input data for the base learners?
#for(time_iter in time_iter_pool <- times_out){
