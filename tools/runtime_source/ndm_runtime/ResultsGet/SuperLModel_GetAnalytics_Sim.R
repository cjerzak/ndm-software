# Evaluate every canonical example at every checkpoint. Batch boundaries never
# determine the panel, posterior draw keys, or aggregation of nonlinear metrics.
if (isTRUE(SimMode)) {
  ndm_plot_log_series_safe <- utils::getFromNamespace(".ndm_plot_log_series_safe", "ndm")
  sim_base_id <- get0("BaseID", ifnotfound = SimEntry$BaseID %||% 1L)
  inference_draw_plan <- ndm_inference_draw_plan(
    ModelType, get0("neuralode_variational", ifnotfound = identical(ModelType, "NeuralODE")),
    get0("InferenceMCDraws", ifnotfound = 5L), get0("testWithoutSampling", ifnotfound = FALSE))
  GetPredSaveAtInfo_inference <- list(nTimesLookValidationInference,
    diffrax$SaveAt(ts = jnp$arange(start = 0L, stop = as.integer(nTimesLookValidationInference), dtype = jnp$int32)))
  sim_matrix <- function(value, n, width, name, finite = TRUE) {
    value <- np$asanyarray(value)
    dims <- dim(value)
    if (length(dims) == 3L && dims[[3L]] >= 1L && dims[[2L]] == width) value <- value[,,1L, drop = FALSE]
    if (length(value) != n * width) stop("Unexpected shape for ", name)
    value <- matrix(as.numeric(value), nrow = n, ncol = width)
    if (finite && any(!is.finite(value))) stop("Nonfinite canonical inference values in ", name)
    value
  }
  sim_predict_draws <- function(batch, ids, policy = PolicyList) {
    results <- lapply(seq_len(inference_draw_plan$effective), function(draw_id) {
      tryCatch({
        keys <- ndm_inference_scoped_keys(jax, jnp, JaxKey, SEED_, sim_base_id, i, draw_id, ids, rep(0L, length(ids)))
        pred <- GetPred_inference(ModelList, batch2package(batch), state, PriorList, policy,
          GetPredSaveAtInfo_inference, ndm_runtime_data_to_device(keys))[[1L]]
        ndm_validate_inference_solver_diagnostics(pred, draw_id, ids, rep(0L, length(ids)),
          host_array = function(x) ndm_inference_host_array(x, np),
          require_diagnostics = !isTRUE(get0("AllowLegacyMissingSolverDiagnostics", ifnotfound = FALSE)))
        # Reject each invalid draw before reduction; failed draws cannot disappear.
        sim_matrix(pred$y_mu, length(ids), nTimesLookValidationInference, "posterior prediction")
        sim_matrix(pred$y_sigma, length(ids), nTimesLookValidationInference, "posterior scale")
        list(ok = TRUE, value = pred)
      }, error = function(error) list(ok = FALSE, error = error))
    })
    ndm_stop_incomplete_inference_draws(results, inference_draw_plan$effective, i)
    lapply(results, `[[`, "value")
  }
  inference_iterator <- reticulate::as_iterator(TFDataset_inference)
  next_batch <- utils::getFromNamespace(".ndm_tfrecord_iter_next", "ndm")
  panel <- utils::getFromNamespace(".ndm_collect_inference_panel", "ndm")(
    get0("ndm_inference_expected_count", ifnotfound = NA_integer_),
    function() next_batch(inference_iterator),
    function(batch) vapply(batch, function(x) as.integer(np$array(x$shape)[[1L]]), integer(1L)),
    function(batch, ids) {
      batch <- TFConst2JAXArray(batch)
      n <- length(ids); h <- nTimesLookValidationInference
      draws <- sim_predict_draws(batch, ids)
      mean_field <- function(field, width = 1L, structural = FALSE) {
        Reduce(`+`, lapply(draws, function(x) sim_matrix(
          if (structural) x$ODEParamsSampList[[field]] else x[[field]], n, width, field))) / length(draws)
      }
      out <- list(truth = sim_matrix(batch$YTrue_out, n, h, "YTrue_out"),
        past_future = sim_matrix(batch$YTrue, n, nTimesTotal, "YTrue"),
        prediction = mean_field("y_mu", h), scale = mean_field("y_sigma", h))
      if (ModelType != "DecoderOnly") {
        out$init_true <- sim_matrix(batch$init_true, n, 4L, "init_true")
        out$gamma_true <- sim_matrix(batch$gamma_true, n, 1L, "gamma_true")
        out$sigma_true <- sim_matrix(batch$sigma_true, n, 1L, "sigma_true")
        out$init_hat <- do.call(cbind, lapply(c("s_l_samp", "e_l_samp", "i_l_samp", "r_l_samp"), mean_field, structural = TRUE))
        out$gamma_hat <- mean_field("gamma_samp", structural = TRUE)
        out$sigma_hat <- mean_field("sigma_samp", structural = TRUE)
      }
      out
    })
  l_true <- panel$truth; pred_l_mean <- panel$prediction; pred_l_sd <- panel$scale
  pred_l_baselineVal <- panel$past_future[, nTimesPast]
  pred_l_baselineVal_mat <- matrix(pred_l_baselineVal, nrow(l_true), ncol(l_true))
  n_units <- nrow(l_true); n_times <- ncol(panel$past_future)
  ndm_viz_trajectory_rows <- list(data.table::data.table(
    monte_eval = 1L, unit = rep(seq_len(n_units), times = n_times),
    t = rep(seq_len(n_times), each = n_units),
    is_future = rep(seq_len(n_times) > nTimesPast, each = n_units),
    y_true = as.vector(panel$past_future),
    y_model = as.vector(cbind(matrix(NA_real_, n_units, nTimesPast), pred_l_mean)),
    y_sd = as.vector(cbind(matrix(NA_real_, n_units, nTimesPast), pred_l_sd)),
    y_persist = as.vector(cbind(matrix(NA_real_, n_units, nTimesPast), pred_l_baselineVal_mat)),
    sim_index = af, i_in_sgd = i, nSGD = nSGD_model, nParamsModel = nParamsModel,
    ModelType = ModelType, model_spec_name = as.character(SimEntry$model_spec_name %||% NA_character_),
    analysis_name = AnalysisName))
  ##############################
  # R-squared analysis
  ##############################
  {
  ### with predicted values
  RawR2 <- function(yhat,ytrue){ 1 - sum( (yhat-ytrue)^2) / sum((ytrue-mean(ytrue))^2) }
  MedianWithinR2_Raw <- median( sapply(1:nrow(l_true),function(zer){
    WithinR2 <- RawR2(c(pred_l_mean[zer,]),l_true[zer,])
  }) )
  MedianWithinR2_Fit <- median( sapply(1:nrow(l_true),function(zer){
    WithinR2 <- lm(c(l_true[zer,])~c(pred_l_mean[zer,]))
    WithinR2 <- summary(WithinR2)$adj.r.squared
  }) )
  GlobalR2_Raw <- RawR2(c(pred_l_mean),c(l_true))
  GlobalR2_Fit <- summary(lm(c(l_true)~c(pred_l_mean)))$adj.r.squared

  ### compare against baseline
  MedianWithinR2_Raw_baseline <- median( sapply(1:nrow(l_true),function(zer){
    WithinR2 <- RawR2(
      rep(pred_l_baselineVal[zer],times = ncol(l_true)),
      l_true[zer,])
  }) )
  MedianWithinR2_Fit_baseline <- median( sapply(1:nrow(l_true),function(zer){
    WithinR2 <- lm(c(l_true[zer,])~1)
    WithinR2 <- summary(WithinR2)$adj.r.squared
  }) )
  GlobalR2_Raw_baseline <- RawR2(c(pred_l_baselineVal_mat),c(l_true))
  GlobalR2_Fit_baseline <- summary(lm(c(l_true)~c(pred_l_baselineVal_mat)))$adj.r.squared

  FNorm_Raw <- mean(sqrt((c(pred_l_mean) - c(l_true))^2))
  FNorm_Rel <- mean(sqrt((c(pred_l_mean) - c(l_true))^2)) / mean(sqrt(l_true^2))
  FNorm_Raw_baseline <- mean(sqrt((c(pred_l_baselineVal_mat) - c(l_true))^2))
  FNorm_Rel_baseline <- FNorm_Raw_baseline / mean(sqrt(l_true^2))

  # agg norm
  AggNorm_Raw <- mean(sqrt((rowSums(pred_l_mean) - rowSums(l_true))^2))
  AggNorm_Rel <- mean(sqrt((rowSums(pred_l_mean) - rowSums(l_true))^2)) / 
                                    mean(sqrt(rowSums(l_true)^2))
  AggNorm_Raw_baseline <- mean(sqrt((rowSums(pred_l_baselineVal_mat) - rowSums(l_true))^2))
  AggNorm_Rel_baseline <- AggNorm_Raw_baseline / mean(sqrt(rowSums(l_true)^2))

  # Skill measure
  #plot(l_true[rer<-sample(1:10,1),],ylim=c(0,max(l_true))); points(pred_l_mean[rer,],pch="^")
  #plot(l_true[1,]);plot(pred_l_mean[1,])
  skill_metrics <- vapply(seq_len(ncol(l_true)), function(horizon) {
    ndm_paired_rmse_skill(
      prediction = pred_l_mean[,horizon],
      baseline = pred_l_baselineVal_mat[,horizon],
      truth = l_true[,horizon]
    )
  }, numeric(3L))
  skill_vec <- skill_metrics["skill",]
  RSS_pred <- skill_metrics["rss_pred",]
  RSS_baseline <- skill_metrics["rss_baseline",]
  names(skill_vec) <- paste("SkillTime",1:length(skill_vec), sep = "")
  names(RSS_pred) <- paste("RSSPredTime",1:length(skill_vec), sep = "")
  names(RSS_baseline) <- paste("RSSBaselineTime",1:length(skill_vec), sep = "")

  sd_outcome <- median( apply(l_true,1,sd) )
  }

  # The structural summaries use all examples, preserving population units.
  AbsDiff_init <- KLDiv_init <- AbsDiff_gamma <- RelAbsDiff_gamma <- AbsDiff_sigma <- RelAbsDiff_sigma <- NA_real_
  KLDiv_init_status <- "not_applicable"; KLDiv_init_infinite_count <- 0L
  beta_MedianWithinR2_Fit <- beta_MedianWithinR2_Raw <- beta_GlobalR2_Raw <- beta_GlobalR2_Fit <-
    beta_FNorm_Rel <- beta_FNorm_Raw <- beta_AggNorm_Rel <- beta_AggNorm_Raw <- NA_real_
  if (ModelType != "DecoderOnly") {
    init_hat <- panel$init_hat; init_true_mat <- panel$init_true
    AbsDiff_init <- median(rowSums(abs(init_hat - init_true_mat)))
    kl <- utils::getFromNamespace(".ndm_population_weighted_kl", "ndm")(init_true_mat, init_hat)
    KLDiv_init <- kl$value; KLDiv_init_status <- kl$status; KLDiv_init_infinite_count <- kl$infinite_count
    AbsDiff_gamma <- median(abs(panel$gamma_true - panel$gamma_hat))
    RelAbsDiff_gamma <- median(abs(panel$gamma_true - panel$gamma_hat) / panel$gamma_true)
    AbsDiff_sigma <- median(abs(panel$sigma_true - panel$sigma_hat))
    RelAbsDiff_sigma <- median(abs(panel$sigma_true - panel$sigma_hat) / panel$sigma_true)
  }
  ##############################
  # policy eval metrics
  ##############################
  {
  # data gen part  is indexed to: t0 = 0, t1 = NTimeSteps_SIM,
  # model part  is indexed to: t0 = 0, t1 = nTimesTotal,
  InterpolatorBasis_ <- rep(0.1, times = NTimeSteps_SIM+1)
  #ShiftFactor <- min( np$array( NTimeSteps_SIM - nTimesPast )  )
  ShiftFactor <- min( np$array( 0 + nTimesPast )  )

  startIntervention <- max( np$array( ShiftFactor )  ) + 2L
  IndicatorBasis_un <- rep(0, NTimeSteps_SIM+1)
  IndicatorBasis_un[-c(1:startIntervention)] <- 1
  # IndicatorBasis_un[-c(1:ShiftFactor)][1:nTimesTotal]
  # PolicyList_GetBatch_unnatural[[1]]$evaluate(jnp$array((startIntervention-1):startIntervention))
  PolicyList_GetBatch_natural <- list(
    "ComputeScenario" = F,

    # scenario (defines the smooth shift trajectory in the policy from the previous state)
    "Scenario" = as.numeric(InterpolatorBasis_[seq_len(NTimeSteps_SIM)])
  )

  PolicyList_GetBatch_unnatural <- list(
    "ComputeScenario" = T,

    # scenario (defines the smooth shift trajectory in the policy from the previous state)
    "Scenario" = as.numeric(InterpolatorBasis_[seq_len(NTimeSteps_SIM)])
    )

  model_policy_ts <- jnp$array(as.numeric(0:(nTimesTotal-1L)))
  model_policy_scenario <- jnp$array(InterpolatorBasis_[-c(1:ShiftFactor)][1:(nTimesTotal+0)])
  PolicyList_Model_natural <- list(
    # policy indicator
    diffrax$LinearInterpolation(
      model_policy_ts,
      jnp$array(rep(0, nTimesTotal + 0))
    ),

    # scenario (defines the smooth shift trajectory in the policy from the previous state)
    diffrax$LinearInterpolation(
      model_policy_ts,
      model_policy_scenario
    )
  )

  PolicyList_Model_unnatural <- list(
    # policy indicator
    diffrax$LinearInterpolation(
      model_policy_ts,
      jnp$array(IndicatorBasis_un[-c(1:ShiftFactor)][1:(nTimesTotal+0)])
    ),

    # scenario (defines the smooth shift trajectory in the policy from the previous state)
    diffrax$LinearInterpolation(
      model_policy_ts,
      model_policy_scenario
    )
  )
  }

  
  # get factual and counterfactual data
  PolicyScenarioSkillRes <- PolicyScenarioSkillBaselineRes <- NULL; nCounterfactuals <- 0; policy_eval_failed <- FALSE; while( nCounterfactuals < 2 ){
    nCounterfactuals <- nCounterfactuals + 1
    counterf_ <- as.integer(nCounterfactuals)
    policy_count <- 32L
    policy_ids <- 1000000L + (counterf_ - 1L) * policy_count + seq_len(policy_count)
    policy_seeds <- as.integer((as.double(SEED_) * 104729 + as.double(sim_base_id) * 130363 + policy_ids) %% 2147483646)
    ok_<-F; ok_counter_ <- 0; while(!ok_){
      ok_counter_ <- ok_counter_ + 1
      batch_l_natural <- try(GetBatch(
                                      nBatch = policy_count, example_seeds = policy_seeds,
                                      INPUT_REF_DAT = input_df_red_in,
                                      PolicyList = PolicyList_GetBatch_natural,
                                      nTimes = NTimeSteps_SIM,
                                      nTimesLook = nTimesLookValidation), T)
      batch_l_unnatural <- try(GetBatch(nBatch = policy_count, example_seeds = policy_seeds,
                                      INPUT_REF_DAT = input_df_red_in,
                                      nTimes = NTimeSteps_SIM,
                                      nTimesLook = nTimesLookValidation,
                                      PolicyList = PolicyList_GetBatch_unnatural), T) 
                                      #PolicyList = PolicyList_GetBatch_natural),T) # for debugging
      if(ok_counter_ >= 1 && (inherits(batch_l_natural, "try-error") || inherits(batch_l_unnatural, "try-error"))){
        warning("Skipping policy scenario analytics because natural/counterfactual batches could not be generated")
        policy_eval_failed <- TRUE
        break
      }
      if(!"try-error" %in% class(batch_l_natural) & !"try-error" %in% class(batch_l_unnatural) ){ ok_ <- T }
    }
    if(policy_eval_failed){ break }
    {
      # get factual and counterfactual predictions
      # note: t is differnt in meaning for getting data and for modeling
      policy_predict <- function(batch, policy) {
        draws <- sim_predict_draws(batch, policy_ids, policy)
        Reduce(`+`, lapply(draws, function(x) sim_matrix(x$y_mu, policy_count, nTimesLookValidationInference, "policy prediction"))) / length(draws)
      }
      hat_y_natural <- policy_predict(batch_l_natural, PolicyList_Model_natural)
      hat_y_unnatural <- policy_predict(batch_l_unnatural, PolicyList_Model_unnatural)

      # skill baseline -predicting unnatural outcomes with natural outcomes
      NUM <- (0.001+colMeans(abs(np$array( batch_l_unnatural$YTrue_out )[,,1] -
                            np$array( batch_l_natural$YTrue_out )[,,1])))
      DENOM <- (0.001+colMeans(abs( np$array( batch_l_unnatural$YTrue_out )[,,1] -
                            matrix(np$array( batch_l_natural$YTrue_out )[,1,1],
                                   ncol = ncol(np$array( batch_l_natural$YTrue_out )),
                                   nrow = nrow(np$array( batch_l_natural$YTrue_out )),byrow=F)   )) )
      PolicyScenarioSkill_baseline <- NUM / DENOM

      # skill - predicting unnatural outcomes using natural predictions
      NUM <- (0.001+colMeans(abs(np$array( batch_l_unnatural$YTrue_out )[,,1] -#out_y_indices
                            hat_y_unnatural )))
                            #np$array( hat_y_unnatural$y_mu )[,out_y_indices,1] )))
      DENOM <- (0.001+colMeans(abs(np$array( batch_l_unnatural$YTrue_out )[,,1] -
                                hat_y_natural)) )
                                     #np$array( hat_y_natural$y_mu )[,out_y_indices,1])) )
      PolicyScenarioSkill <- NUM  / DENOM

      names(PolicyScenarioSkill) <- paste("PolicySkill", 1:length(PolicyScenarioSkill), sep="")
      names(PolicyScenarioSkill_baseline) <- paste("PolicySkillBaseline", 1:length(PolicyScenarioSkill_baseline), sep="")
      PolicyScenarioSkillRes <- rbind(PolicyScenarioSkillRes,PolicyScenarioSkill)
      PolicyScenarioSkillBaselineRes <- rbind(PolicyScenarioSkillBaselineRes,PolicyScenarioSkill_baseline)


    }
  }
  if(is.null(PolicyScenarioSkillRes) || is.null(PolicyScenarioSkillBaselineRes)){
    PolicyScenarioSkillRes <- rep(NA_real_, nTimesLookValidationInference)
    names(PolicyScenarioSkillRes) <- paste("PolicySkill", seq_len(nTimesLookValidationInference), sep="")
    PolicyScenarioSkillBaselineRes <- rep(NA_real_, nTimesLookValidationInference)
    names(PolicyScenarioSkillBaselineRes) <- paste("PolicySkillBaseline", seq_len(nTimesLookValidationInference), sep="")
  } else {
    PolicyScenarioSkillRes <- colMeans(PolicyScenarioSkillRes)
    PolicyScenarioSkillBaselineRes <- colMeans(PolicyScenarioSkillBaselineRes)
  }
  rm_list_ <- c("batch_l_unnatural","batch_l_natural","hat_y_natural","hat_y_unnatural")
  rm_list_ <- rm_list_[vapply(rm_list_, exists, logical(1), inherits = FALSE)]
  if(length(rm_list_) > 0L){ rm(list = rm_list_) }

  ##############################
  # setup save variable names
  ##############################
  {
    SaveQuantityNames <- c(outcome2 <- c("MedianWithinR2_Raw","MedianWithinR2_Fit",
                                         "GlobalR2_Raw","GlobalR2_Fit",
                                         "FNorm_Raw","FNorm_Rel",
                                         "AggNorm_Raw","AggNorm_Rel"),
                           paste(outcome2, "_baseline",sep = ""))
    SaveQuantityNames <- c(SaveQuantityNames,"beta_MedianWithinR2_Raw",
                  "beta_MedianWithinR2_Fit",
                  "beta_AggNorm_Raw",
                  "beta_AggNorm_Rel",
                  "beta_FNorm_Raw",
                  "sd_outcome",
                  "beta_FNorm_Rel",
                  "beta_GlobalR2_Raw",
                  "beta_GlobalR2_Fit")
    SaveQuantityNames <- c(SaveQuantityNames,
                           "AbsDiff_init","KLDiv_init",
                           "AbsDiff_gamma","RelAbsDiff_gamma",
                           "AbsDiff_sigma","RelAbsDiff_sigma")
  }

  tmp <- sapply(SaveQuantityNames,function(ze){
    list(eval(parse(text = sprintf("c('%s' = eval(parse(text = %s)))",ze,ze))))
  })
  names(tmp) <- NULL
  panel_metrics <- c(do.call(c,tmp), skill_vec, RSS_pred, RSS_baseline, PolicyScenarioSkillRes, PolicyScenarioSkillBaselineRes)
  
  try({
    ndm_viz_trajectory_rows <- ndm_viz_trajectory_rows[!vapply(ndm_viz_trajectory_rows, inherits, logical(1), "try-error")]
    if(length(ndm_viz_trajectory_rows) > 0L){
      data.table::fwrite(
        data.table::rbindlist(ndm_viz_trajectory_rows),
        file = file.path(HolderFolder, sprintf("trajectories_sim%s_i%s.csv", af, i))
      )
    }
  }, silent = TRUE)

  #  write loss fig for debugging 
  pdf(file.path(HolderFolder, sprintf("diagnostics%s.pdf", af)), height = 10, width = 5)
  {
  par(mfrow=c(2,2))
  loss_obs <- na.omit(in_loss_vec)
  grad_obs <- na.omit(grad_norm_vec)
  ndm_plot_log_series_safe(loss_obs, main = "Loss")
  if(length(loss_obs) > 0L){ plot(rank(loss_obs), log = "") } else { plot.new() }
  ndm_plot_log_series_safe(grad_obs, main = "Gradients")
  dev.off()
  }
  par(mfrow = c(1,1))
  
  
  # --- Skill sanity check (aggregate RMSE vs. persistence baseline) ---
  {
    skill_horizon <- max(1L, min(8L, ncol(pred_l_mean)))
    sanity_metrics <- ndm_paired_rmse_skill(
      prediction = pred_l_mean[,skill_horizon],
      baseline = pred_l_baselineVal_mat[,skill_horizon],
      truth = l_true[,skill_horizon]
    )
    Skill8SanityCheck <- unname(sanity_metrics[["skill"]])
    print2(sprintf("Skill sanity: %.3f", Skill8SanityCheck))
  }
  
  # save results to disk 
  res_vec <- t(c("sim_index" = af,
                 "i_in_sgd" = i,
                 "nTrainingSamplesSeen" = i*nBatch,
                 "atEpoch" = i*nBatch/nSamplesTrain, 
                 "nSGD" = nSGD_model,
                 "nSGDPolicy" = get0("nSGDPolicy", inherits = TRUE, ifnotfound = NA_character_),
                 "nSGDAnchorMaxSamplesTrain" = get0("nSGDAnchorMaxSamplesTrain", inherits = TRUE, ifnotfound = NA_integer_),
                 "nSGDAnchorScope" = get0("nSGDAnchorScope", inherits = TRUE, ifnotfound = NA_character_),
                 "nSGDAnchorBatch" = get0("nSGDAnchorBatch", inherits = TRUE, ifnotfound = NA_integer_),
                 "nParamsModel" = nParamsModel,
                 "te_total" = te_total, 
                 "te_grads" = te_grads, 
                 unlist(SimEntry),
                 panel_metrics))
  res_vec <- ndm_result_metadata(as.data.frame(res_vec), environment(), nTimesLookValidationInference)
  print(sprintf("WRITING af of %s", af))
  res_vec$KLDiv_init_status <- KLDiv_init_status
  res_vec$KLDiv_init_infinite_count <- KLDiv_init_infinite_count
  res_vec$inference_mc_draws_requested <- inference_draw_plan$requested
  res_vec$inference_mc_draws_effective <- inference_draw_plan$effective
  res_vec$inference_key_scheme <- "jax-fold-in-v2:run>BaseID>checkpoint>draw;local>canonical_ordinal;global>shared-draw"
  res_vec$inference_examples <- nrow(l_true)
  res_vec$policy_examples <- 64L
  res_vec$policy_status <- if (policy_eval_failed) "failed" else "ok"
  result_file <- file.path(HolderFolder, sprintf("res%s_i%s.csv", af, i))
  utils::getFromNamespace(".ndm_atomic_result_csv", "ndm")(
    res_vec, result_file, list(
      run_id = as.character(res_vec$RunID[[1L]]), checkpoint_step = as.integer(i), n_sgd = as.integer(nSGD_model),
      inference_examples = nrow(l_true), evaluation_horizon = ncol(l_true),
      manifest_sha256 = get0("ndm_inference_manifest_sha256", ifnotfound = NA_character_),
      draws_requested = inference_draw_plan$requested, draws_effective = inference_draw_plan$effective,
      key_scheme = res_vec$inference_key_scheme,
      canonical_ids_sha256 = digest::digest(seq_len(nrow(l_true)), algo = "sha256")))
  ndm_last_evaluation_file <- result_file
  print("Done with SuperLModel_GetAnalytics_Sim.R")
}
