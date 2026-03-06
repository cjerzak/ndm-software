ndm_test_copy_env <- function(src, dest = globalenv()) {
  src_names <- ls(src, all.names = TRUE)
  overwritten <- intersect(src_names, ls(dest, all.names = TRUE))
  overwritten_values <- if (length(overwritten) > 0L) {
    mget(overwritten, envir = dest, inherits = FALSE)
  } else {
    list()
  }

  list2env(mget(src_names, envir = src, inherits = FALSE), envir = dest)

  list(
    all_names = src_names,
    overwritten = overwritten,
    overwritten_values = overwritten_values
  )
}

ndm_test_restore_env <- function(snapshot, dest = globalenv()) {
  created <- setdiff(snapshot$all_names, snapshot$overwritten)
  if (length(created) > 0L) {
    rm(list = created, envir = dest)
  }
  if (length(snapshot$overwritten) > 0L) {
    list2env(snapshot$overwritten_values, envir = dest)
  }
  invisible(dest)
}

ndm_test_fit_sim_case <- function(model_type,
                                  endogeneity,
                                  n_sgd = NULL) {
  if (is.null(n_sgd)) {
    n_sgd <- if (identical(model_type, "NeuralODE")) 3L else 2L
  }
  learning_rate_max <- if (identical(model_type, "NeuralODE")) 3e-3 else 2e-3
  case_seed <- 1000L +
    (match(model_type, c("DecoderOnly", "NeuralODE")) - 1L) * 100L +
    as.integer(round(endogeneity * 100))
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  old_seed <- if (had_seed) get(".Random.seed", envir = .GlobalEnv, inherits = FALSE) else NULL
  set.seed(case_seed)
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(list = ".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)

  work_dir <- tempfile("ndm-sim-fit-")
  dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)
  old_wd <- setwd(work_dir)
  on.exit(setwd(old_wd), add = TRUE)

  dir.create(file.path(work_dir, "SavedModels", "FromSim"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(work_dir, "SavedResults", "Sim"), recursive = TRUE, showWarnings = FALSE)

  sim_entry <- list(
    BaseID = 1L,
    betat_init = 0.35,
    invbetat_sd = 0.02,
    i0_a = 5,
    i0_b = 0.2,
    r0_a = 2,
    r0_b = 0.2,
    gamma = 0.25,
    sigma = 0.2,
    xi = 0.02,
    c_endogeneous = endogeneity,
    policy_effectiveness = 0.1,
    policy_responsiveness = 1.0,
    policy_decay = 0.05,
    beta_baseline = 0.35,
    beta_restore_rate = 0.1,
    measurement_noise = 0.05,
    nSamplesTrain = 4L,
    ResaveThisTFRecord = 1L
  )
  sim_grid <- do.call(
    rbind,
    list(
      as.data.frame(sim_entry, stringsAsFactors = FALSE),
      as.data.frame(
        utils::modifyList(
          sim_entry,
          list(BaseID = 2L, c_endogeneous = endogeneity + 0.1)
        ),
        stringsAsFactors = FALSE
      )
    )
  )

  config <- ndm_create_config(
    model_type = model_type,
    backbone = "transformer",
    analysis_root = ndm_runtime_paths()$analysis_root,
    float_type = "32",
    force_to_gpu = FALSE,
    resave_tfrecords = TRUE
  )

  runtime_env <- ndm_prepare_runtime(
    config = config,
    runtime_env = ndm_new_runtime_env(parent = globalenv()),
    runtime_globals = list(
      SimMode = TRUE,
      GLOBAL_ODE_NPOP = 10000L,
      nBatch = 4L,
      covariateType = "sqrt",
      simCovariates = c(
        XPred_c_sqrt = "inc_case_per_capita_sqrt",
        XPred_h_sqrt = "inc_hosp_per_capita_sqrt",
        XPred_d_sqrt = "inc_death_per_capita_sqrt"
      ),
      rollCompute_window = 52L,
      AnalysisName = sprintf("TestPandemicSim_%s", model_type),
      AnalysisDate = Sys.Date(),
      COMMAND_ARG_INPUT = "test",
      TfRecordDir = file.path(work_dir, "tfrecords"),
      nSamples_max = 4L,
      nSamplesTrain = 4L,
      SimEntry = sim_entry,
      SimGrid = sim_grid,
      SEED_ = 101L,
      LEARNING_RATE_MAX = learning_rate_max,
      nSGD_DefiningLRSeq = n_sgd,
      nSGD_model = n_sgd,
      nSGD_pretrain = 0L,
      nSGD_posttrain = n_sgd,
      nCheckpoints = 0L,
      ModelDims = 32L,
      ModelDepth = 1L,
      nOutcomes = 1L,
      nPlaces = 1L,
      af = 1L,
      HolderFolder = file.path(work_dir, "results"),
      endAppend = TRUE,
      paddingMethod = "left",
      nBatch_SimGridGen = 8L,
      nMonteEval = 1L,
      SimScalingOuterLoops = 1L,
      SimScalingInnerLoops = 2L
    )
  )

  n_times_lookahead <- 4L
  n_times_past <- 8L
  n_times_total <- n_times_past + n_times_lookahead
  n_time_steps_sim <- (n_times_past + n_times_lookahead) * 2L
  ndm_set_runtime_globals(
    runtime_env,
    list(
      nTimesPast = n_times_past,
      nTimesLookahead = n_times_lookahead,
      nTimesTotal = n_times_total,
      VI_TotalTimesInLikelihood = n_times_lookahead,
      nTimesInLikelihood = n_times_lookahead,
      NTimeSteps_SIM = n_time_steps_sim,
      nTimesLookValidationInference = 4L,
      MaxSteps = as.integer(1e4),
      VI_SaveAt_ODE_sim = runtime_env$diffrax$SaveAt(
        ts = runtime_env$jnp$array(0L:(n_time_steps_sim - 1L))
      ),
      VI_SaveAt_ODE_optim = runtime_env$diffrax$SaveAt(
        ts = runtime_env$jnp$array(0L:(n_times_lookahead - 1L))
      ),
      VI_diff_eq_solver_optim = runtime_env$diffrax$Tsit5(),
      VI_diff_eq_solver_dgp = runtime_env$diffrax$Tsit5(),
      dt0_init_dgp = 1e-3,
      stepsize_controller_dgp = runtime_env$diffrax$PIDController(
        rtol = 1e-5,
        atol = 1e-6
      ),
      dt0_init_optim = 1e0,
      stepsize_controller_optim = runtime_env$diffrax$ConstantStepSize(),
      diffraxInterpolator = runtime_env$diffrax$LinearInterpolation
    )
  )

  suppressWarnings(
    ndm_prepare_data(
      runtime_env = runtime_env,
      analysis_root = config$analysis_root,
      generator = "sim"
    )
  )

  snapshot <- ndm_test_copy_env(runtime_env)
  on.exit(ndm_test_restore_env(snapshot), add = TRUE)

  model_spec <- ndm_model_spec(
    preset = "seirs_dynamic_beta",
    model_type = model_type
  )
  model <- suppressWarnings(
    ndm_build_model(
      runtime_env = runtime_env,
      analysis_root = config$analysis_root,
      model_type = model_type,
      model_spec = model_spec,
      backbone = "transformer"
    )
  )
  trained <- suppressWarnings(
    ndm_train(
      model,
      analysis_root = config$analysis_root,
      run_define = TRUE,
      run_loop = TRUE
    )
  )

  losses <- as.numeric(trained$env$in_loss_vec[seq_len(n_sgd)])
  data.frame(
    model_type = model_type,
    spec_preset = model_spec$preset,
    endogeneity = endogeneity,
    first_loss = losses[[1]],
    last_loss = losses[[length(losses)]],
    loss_delta = losses[[1]] - losses[[length(losses)]],
    stringsAsFactors = FALSE
  )
}

test_that("simulated pandemic fits improve across model families and endogeneity levels", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip_if_not_installed("fastmatch")
  skip_if_not_installed("rrapply")
  skip_if_not_installed("progress")
  skip_if_not_installed("zip")
  skip_if_not_installed("zoo")

  backend_ready <- ndm_check_backend(
    conda_env = "jax_cpu",
    modules = c("jax", "numpy", "optax", "equinox", "diffrax", "tensorflow")
  )
  skip_if(is.null(backend_ready), "jax_cpu with JAX/TensorFlow is required for the simulation fit test.")

  cases <- data.frame(
    model_type = c("DecoderOnly", "DecoderOnly", "NeuralODE", "NeuralODE"),
    endogeneity = c(0.0, 0.6, 0.0, 0.6),
    stringsAsFactors = FALSE
  )

  results <- do.call(
    rbind,
    lapply(seq_len(nrow(cases)), function(i) {
      ndm_test_fit_sim_case(
        model_type = cases$model_type[[i]],
        endogeneity = cases$endogeneity[[i]]
      )
    })
  )
  results_info <- paste(capture.output(print(results)), collapse = "\n")
  neural_ode_results <- results[results$model_type == "NeuralODE", , drop = FALSE]

  expect_true(all(is.finite(results$first_loss)))
  expect_true(all(is.finite(results$last_loss)))
  expect_true(all(results$spec_preset == "seirs_dynamic_beta"), info = results_info)
  expect_true(all(results$loss_delta >= 0), info = results_info)
  expect_true(mean(results$loss_delta) > 1e-3, info = results_info)
  expect_true(min(neural_ode_results$loss_delta) > 1e-3, info = results_info)
})
