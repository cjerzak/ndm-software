{
  # setup infrastructure
  {
  ReplicateFxn <- function(ComputeScenario,nTimes, nTimesLook){
    invbeta_true_init <- InvSoftPlus(  SimEntry$betat_init )
    inv_beta_noise <- (sapply(0:(nTimes-1L),function(zer){ rnorm(1, mean = 0, sd = SimEntry$invbetat_sd )}))
    # plot( cumsum(inv_beta_noise) ) # look at implied random walk

    # SEIRS
    S_log <- log(10000)
    I_log <- runif(1,
                   min = SimEntry$i0_a - SimEntry$i0_a*abs(SimEntry$i0_b),
                   max = SimEntry$i0_a + SimEntry$i0_a*abs(SimEntry$i0_b))
    E_log <- log(10) + I_log
    R_log <- runif(1, 
                   min = SimEntry$r0_a - SimEntry$r0_a*abs(SimEntry$r0_b),
                   max = SimEntry$r0_a + SimEntry$r0_a*abs(SimEntry$r0_b))

    # define initial values from log scale
    init_true <- jnp$array( c(S_log, E_log, I_log, R_log) )
    init_true <- init_true - jax$nn$logsumexp(jnp$array(init_true) )
    init_true <- jnp$exp( init_true + log(GLOBAL_ODE_NPOP) )
    init_true <- np$array( init_true )
    #plot( init_true )

    # DGP shifts
    forward_shift_h <- 4 # sample(ai(c(2:9)),1)
    forward_shift_c <- forward_shift_h + 3 # sample(ai(c(2,9)),1)

    # take initial sample shift
    MIN_SHIFT <- 0L; MAX_SHIFT <- nTimes - (nTimesPast + nTimesLook + forward_shift_c)
    if(MAX_SHIFT < MIN_SHIFT) {
      stop(sprintf("Invalid time configuration: MAX_SHIFT=%d < MIN_SHIFT=%d (nTimes=%d, nTimesPast=%d, nTimesLook=%d, forward_shift_c=%d)",
                   MAX_SHIFT, MIN_SHIFT, nTimes, nTimesPast, nTimesLook, forward_shift_c))
    }
    initial_shift <- if(MIN_SHIFT == MAX_SHIFT) MIN_SHIFT else sample(MIN_SHIFT:MAX_SHIFT, 1L)

    if( !ComputeScenario ){
      PolicyScenarioSimBasis <- rep(0,times=nTimes)
    }
    if( ComputeScenario ){
      PolicyScenarioSimBasis <- rep(0,times=nTimes)
      PolicyScenarioSimBasis[ -c(1L:(1L+initial_shift + nTimesPast)) ] <- 1
    }
    return(list(
      "invbeta_true_init" = jnp$array(invbeta_true_init),
      "inv_beta_noise" = jnp$array(inv_beta_noise),
      "init_true" = jnp$array(init_true),
      "PolicyScenarioSimBasis" = jnp$array( PolicyScenarioSimBasis),
      "initial_shift" = jnp$array(initial_shift),
      "forward_shift_h" = jnp$array(forward_shift_h),
      "forward_shift_c" = jnp$array(forward_shift_c) ))
  }
  GetDataFromODE_sim <- eq$filter_jit( function(y0, args, nTimes, nTimes_arange){ # test jit compile - trying to speed up runtime 
    VI_ODE_term_SynthOutcomeGenerator <- diffrax$ODETerm(
      f_SIM <- ( function(t, y, args){
        beta_noise_evaluated <- diffrax$LinearInterpolation(
                                      nTimes_arange, 
                                      args$inv_beta_noise)$evaluate( t )
        policy_scenario_evaluated <- diffrax$LinearInterpolation(
                                  nTimes_arange,
                                  args$policy_scenario)$evaluate(t)
        policy_scenario_indicator_evaluated <- diffrax$LinearInterpolation(
                                  nTimes_arange,
                                  args$policy_scenario_indicator)$evaluate(t)
        #policy_scenario_evaluated <- args$policy_scenario$evaluate( t )
        #policy_scenario_indicator_evaluated <- args$policy_scenario_indicator$evaluate( t )
        y <- list(
          "s"=jnp$add(jnp$multiply(jnp$multiply(jnp$negative( SoftPlus(y$raw_beta_t) ),y$i),
                                   jnp$divide(y$s, GLOBAL_ODE_NPOP_jax)),
                      jnp$multiply(args$xi,y$r)),
          "e"=jnp$add(jnp$multiply(jnp$negative(args$sigma),y$e),
                      jnp$multiply(  SoftPlus( y$raw_beta_t ),
                                     jnp$multiply(y$i, jnp$divide(y$s,GLOBAL_ODE_NPOP_jax )))),
          "i"=jnp$add(jnp$multiply(jnp$negative(args$gamma),y$i),
                      jnp$multiply(args$sigma,y$e)),
          "r"=jnp$subtract(jnp$multiply(args$gamma, y$i), jnp$multiply(args$xi,y$r)),
          # --- BEHAVIORAL DYNAMICS ---
          # raw_beta_t (Transmission Rate)
          'raw_beta_t' =
            # Restoration Term: Drifts back to 'betat_init' when suppression forces are zero
            (args$beta_restore_rate * ( InvSoftPlus(args$betat_init) - y$raw_beta_t) -
            # [Endogenous Fear]: Controlled by c_endogeneous - reduces Beta based on prevalence
            args$c_endogeneous * ((10/GLOBAL_ODE_NPOP_jax) * y$i) * SoftPlus(y$raw_beta_t) -
            # [Policy Impact]: Mechanical effect of policy on transmission (works regardless of c_endogeneous)
            Sigmoid(y$raw_policy_t) * args$policy_effectiveness * SoftPlus(y$raw_beta_t) +
            beta_noise_evaluated),
          # raw_policy_t (Government Response)
          'raw_policy_t' =
            ((1 - policy_scenario_indicator_evaluated) * (
              # [Endogenous Policy Generation]: Public Channel of endogeneity
              # c_endogeneous scales BOTH Fear (above) and Policy (here)
              #(args$c_endogeneous * ((10/GLOBAL_ODE_NPOP_jax) * y$i) * args$policy_responsiveness) -
              # Policy not affected by fear 
              (0.1 * ((10/GLOBAL_ODE_NPOP_jax) * y$i) * args$policy_responsiveness) - 
              # Natural Decay of policy when cases drop
              Sigmoid(y$raw_policy_t) * args$policy_decay
            ) +
            policy_scenario_indicator_evaluated * (
              # Target-Seeking Scenario: drift towards target instead of adding rate
              args$policy_responsiveness * (policy_scenario_evaluated - y$raw_policy_t)
            ))
        )
        return(   y    )
      } )  )
    diff_eq_sol <- try(diffrax$diffeqsolve(terms = VI_ODE_term_SynthOutcomeGenerator,
                                         solver = VI_diff_eq_solver_dgp,
                                         saveat = diffrax$SaveAt( ts = nTimes_arange ),
                                         args = args,
                                         max_steps = MaxSteps,
                                         y0 = y0,
                                         t0 = 0,
                                         t1 = nTimes, 
                                         dt0 = dt0_init_dgp,
                                         stepsize_controller = stepsize_controller_dgp),T)
    if('try-error' %in% class(diff_eq_sol)){
      print(diff_eq_sol)
      stop("ODE solver failed - cannot continue with corrupted solution")
    }
    return( diff_eq_sol )
  } )
  GetDataFromODE_sim_batch <- eq$filter_jit( jax$vmap(function(
    init_true,
    invbeta_true_init,
    inv_beta_noise,
    PolicyScenarioSimBasis,
    
    PolicyScenario,
    initial_shift,
    TheSimEntry,
    nTimes, 
    nTimes_arange
    ){
    diff_eq_sol <- GetDataFromODE_sim(
        y0 = { list("s"=jnp$array(jnp$take(init_true,0L)),
                   "e"=jnp$array(jnp$take(init_true,1L)),
                   "i"=jnp$array(jnp$take(init_true,2L)),
                   "r"=jnp$array(jnp$take(init_true,3L)),
                   "raw_beta_t" = invbeta_true_init, 
                   "raw_policy_t" = jnp$array(-1.)) },
        args = { list(
          "gamma" = TheSimEntry$gamma, # recovery rate
          "sigma" = TheSimEntry$sigma, # incubation rate
          "xi" = TheSimEntry$xi, # return to s from r rate
          "betat_init" = TheSimEntry$betat_init,
          "inv_beta_noise" = inv_beta_noise, # beta's layer to betat's
          "c_endogeneous" = TheSimEntry$c_endogeneous, # level of endogeneity
          "policy_effectiveness" = TheSimEntry$policy_effectiveness,  # level of policy effectiveness
          "policy_responsiveness" = TheSimEntry$policy_responsiveness, # level of policy responsiveness
          "policy_decay" = TheSimEntry$policy_decay, # level of policy decay
          "beta_baseline" = TheSimEntry$beta_baseline, # target "normal" value for raw_beta_t (corresponding to R_0)
          "beta_restore_rate" = TheSimEntry$beta_restore_rate, # speed at which society returns to normal (e.g., 0.1 = ~10 weeks)
          "policy_scenario_indicator" = PolicyScenarioSimBasis, # policy scenario on?
          "policy_scenario" = PolicyScenario # policy scenario
        ) },
        nTimes = nTimes, nTimes_arange = nTimes_arange)
    return( diff_eq_sol )
  },
  in_axes = list(0L,0L,0L,0L,
                 NULL,0L,NULL,NULL,NULL)) )

  # set initial scalings in global environment (to be re-written upon calibration)
  SIM_GLOBAL_SCALE_MEAN <- 0; SIM_GLOBAL_SCALE_SD <- 1
  hosp_rate_true <- 0.1; death_rate_true <- 0.01
  GLOBAL_ODE_NPOP_jax <- jnp$array(GLOBAL_ODE_NPOP)
  
  # initialize a policy scenario
  policy_scenario_ <- rnorm(NTimeSteps_SIM)
  policy_scenario_indicator_ <- rep(0, NTimeSteps_SIM)
  #policy_scenario_ <- diffrax$LinearInterpolation(jnp$array(as.numeric(0:(NTimeSteps_SIM-1))),jnp$array(policy_scenario_))
  #policy_scenario_indicator_ <- diffrax$LinearInterpolation(jnp$array(as.numeric(0:NTimeSteps_SIM)),jnp$array(policy_scenario_indicator_))
  policy_scenario_ <- jnp$array(policy_scenario_)
  policy_scenario_indicator_ <- jnp$array(policy_scenario_indicator_)
  
  #policy_scenario_ <- diffrax$LinearInterpolation(jnp$array(as.numeric(0:(nTimes-1))),
                                                  #jnp$array(Scenario))
  #policy_scenario_indicator_ <- diffrax$LinearInterpolation(jnp$zeros(nTimes$astype(jnp$int32)),
                                                            #jnp$array(Scenario))
  
  # main function create 
  GetBatch_sim <- GetBatch <- function(nBatch=NULL, training = T, openBrowser=F,
                                       INPUT_REF_DAT = NULL, finalGenLocID = NULL, finalGenTimeID = NULL,
                                       nTimes = NTimeSteps_SIM,
                                       nTimesLook = nTimesLookahead,
                                       PolicyList = list( "ComputeScenario"= F, 
                                                          "Scenario"=policy_scenario_)
                                       ){
    {
      tmp <- replicate(nBatch, ReplicateFxn(PolicyList$ComputeScenario,
                                            nTimes = nTimes, nTimesLook=nTimesLook))
      for(name_ in row.names(tmp)){ eval(parse(text=sprintf("%s <- jnp$stack(tmp['%s',],0L)", name_, name_))) }; rm(tmp)
      
      # get a batch of results
      SimEntry_numeric <- SimEntry[!is.na(f2n(SimEntry))]
      TheSimEntry <- sapply(1:length(SimEntry_numeric), function(re_){
        eval(parse(text = sprintf("list('%s'=jnp$array(%s))", 
                                  names(SimEntry_numeric)[re_],SimEntry_numeric[re_] ))) })

      # perform ODE analysis on CPU
      diff_eq_sol <- GetDataFromODE_sim_batch(
                                      ndm_runtime_data_to_device(init_true),
                                      ndm_runtime_data_to_device(invbeta_true_init),
                                      ndm_runtime_data_to_device(inv_beta_noise),
                                      ndm_runtime_data_to_device(PolicyScenarioSimBasis),
                                      ndm_runtime_replicate_tree(PolicyList$Scenario),
                                      ndm_runtime_data_to_device(initial_shift),
                                      ndm_runtime_replicate_tree(TheSimEntry),
                                      ndm_runtime_replicate_tree(jnp$array(nTimes)),
                                      ndm_runtime_replicate_tree(jnp$arange(nTimes)))

      # process output
      {
      inv_beta_true <- try(diff_eq_sol$ys$raw_beta_t,T)
      noise_fraction <- jnp$array(SimEntry$measurement_noise)
      obs_i_true <- diff_eq_sol$ys$i

      # noise
      #noise_d <- oryx$distributions$Uniform(low = jnp$negative(noise_fraction), high = noise_fraction)
      create_uniform_dist <- function(low, high) {
        list(sample = function(shape, seed) {
            jax$random$uniform(
              key = seed,
              shape = shape,
              minval = low, maxval = high,
              dtype = jnp$float32
            )})
      }
      noise_d <- create_uniform_dist(
        low = jnp$negative(noise_fraction), 
        high = noise_fraction
      )
      
      # policy
      obs_policy <- Sigmoid( diff_eq_sol$ys$raw_policy_t )

      # cases
      obs_c_pure <- obs_i_true
      noise_obs_c <- noise_d$sample(obs_i_true$shape, seed = JaxKey(jnp$array(as.integer(runif(1,0,100000)))))
      obs_c <- jnp$add(obs_c_pure, jnp$multiply(obs_c_pure, noise_obs_c))

      # hosp
      obs_h_pure <- jnp$multiply(obs_i_true, hosp_rate_true)
      noise_obs_h <- noise_d$sample(obs_h_pure$shape,
                                    seed = JaxKey(jnp$array(as.integer(runif(1,0,100000)))))
      obs_h <- jnp$add(obs_h_pure, jnp$multiply(obs_h_pure, noise_obs_h))

      # death
      obs_d_pure <- jnp$multiply(obs_i_true, death_rate_true)
      noise_obs_d <- noise_d$sample(obs_d_pure$shape,
                                    seed = JaxKey(jnp$array(as.integer(runif(1,0,100000)))))
      obs_d <- jnp$add(obs_d_pure, jnp$multiply(obs_d_pure, noise_obs_d))

      # grab data observed data
      i_dpast <- t(sapply(np$array(initial_shift),function(init_){
                                        init_:(init_ + nTimesPast-1L) }))
      XPred_d <- jnp$maximum(0., jnp$take_along_axis(obs_d,
                      select_xd <- jnp$array(i_dpast), axis = 1L))
      XPred_h <- jnp$maximum(0., jnp$take_along_axis(obs_h,
                      select_xh <- jnp$array(i_dpast+c(np$array(forward_shift_h)))$astype(jnp$int32), axis = 1L))
      XPred_c <- jnp$maximum(0., jnp$take_along_axis(obs_c,
                      select_xc <- jnp$array(i_dpast+c(np$array(forward_shift_c)))$astype(jnp$int32), axis = 1L))

      # process outcome
      max_i <- apply(i_dpast, 1, max)
      min_i <- apply(i_dpast, 1, min)
      select_xd_out <- jnp$array( t(sapply(max_i,function(max_i_){ (max_i_+1):(max_i_+nTimesLook)}) ) )
      select_xd_all <- jnp$array( t(sapply(1:length(min_i),function(i_){  min_i[i_]:(max_i[i_]+nTimesLook) }) ))
      YTrue_d_out <- jnp$take_along_axis(obs_d, select_xd_out, axis = 1L) # future Y only i
      YTrue_d <- jnp$maximum(0., jnp$take_along_axis(obs_d, select_xd_all, axis = 1L) )# all Y (past and future)

      # select policies
      PolicyDat_seen <- jnp$take_along_axis(obs_policy,
                                  select_policy_seen<-select_xc, axis = 1L)
      select_policy_unseen <- t(apply(np$array(select_xc),1,function(zer){(max(zer)+1L):(max(zer)+nTimesLook)}))
      select_policy_all <- jnp$array( cbind( np$array(select_policy_seen),
                                             np$array(select_policy_unseen)) )
      PolicyDat_all <- jnp$take_along_axis(obs_policy, select_policy_all, axis = 1L)
      PolicyDat_out <- jnp$take_along_axis(obs_policy, select_policy_unseen, axis = 1L)

      # transformations
      if( grepl(covariateType, pattern = "sqrt") ){ 
        XPred_d_sqrt <- jnp$sqrt( XPred_d )
        XPred_h_sqrt <- jnp$sqrt( XPred_h )
        XPred_c_sqrt <- jnp$sqrt( XPred_c )
      } 
      logConst <- 1/GLOBAL_ODE_NPOP
      if( grepl(covariateType, pattern = "log") ){ 
        XPred_d_log <- jnp$log( logConst + XPred_d )
        XPred_h_log <- jnp$log( logConst + XPred_h )
        XPred_c_log <- jnp$log( logConst + XPred_c )
      } 

      if( any(grepl(simCovariates,pattern = "roll")) ){
        # check correctness of roll apply: 
        # plot(zoo::rollapply(c(rep(0,times=10),1:20), width = 5, partial = T, FUN = function(x){ mean(x,na.rm=T) }, align="right",fill = NA))

        # roll means
        obs_rc <- jnp$array( zoo::rollapply(np$asanyarray(obs_c),
                                            width = rollCompute_window, partial = T,
                                            FUN = function(x){ mean(x,na.rm=T) }, align="right",fill = NA)  )
        obs_rh <- jnp$array( zoo::rollapply(np$asanyarray(obs_h),
                                            width = rollCompute_window, partial = T,
                                            FUN = function(x){ mean(x,na.rm=T) }, align="right",fill = NA)  )
        obs_rd <- jnp$array( zoo::rollapply(np$asanyarray(obs_d),
                                            width = rollCompute_window, partial = T,
                                            FUN = function(x){ mean(x,na.rm=T) }, align="right",fill = NA) )
        XPred_rmc <- jnp$take_along_axis(obs_rc, select_xc, axis = 1L)
        XPred_rmh <- jnp$take_along_axis(obs_rh, select_xh, axis = 1L)
        XPred_rmd <- jnp$take_along_axis(obs_rd, select_xd, axis = 1L)
        if( grepl(covariateType, pattern = "log") ){ 
          XPred_rmc_log <- jnp$log( logConst + XPred_rmc ) 
          XPred_rmh_log <- jnp$log( logConst + XPred_rmh  )
          XPred_rmd_log <- jnp$log( logConst + XPred_rmd ) 
        }

        # roll stds
        obs_rsc <- ( zoo::rollapply(np$asanyarray(obs_c),
                                    width = rollCompute_window, partial = T,
                                    FUN = function(x){ sd(x,na.rm=T) }, align="right",fill = NA)  )
        obs_rsc[1,] <- obs_rsc[2,]; obs_rsc <- jnp$array( obs_rsc )
        obs_rsh <- ( zoo::rollapply(np$asanyarray(obs_h),
                                    width = rollCompute_window, partial = T,
                                    FUN = function(x){ sd(x,na.rm=T) }, align="right",fill = NA)  )
        obs_rsh[1,] <- obs_rsh[2,]; obs_rsh <- jnp$array( obs_rsh )
        obs_rsd <- ( zoo::rollapply(np$asanyarray(obs_d),
                                    width = rollCompute_window, partial = T,
                                    FUN = function(x){ sd(x,na.rm=T) }, align="right",fill = NA) )
        obs_rsd[1,] <- obs_rsd[2,]; obs_rsd <- jnp$array( obs_rsd )
        XPred_rsc <- jnp$take_along_axis(obs_rsc, select_xc, axis = 1L)
        XPred_rsh <- jnp$take_along_axis(obs_rsh, select_xh, axis = 1L)
        XPred_rsd <- jnp$take_along_axis(obs_rsd, select_xd, axis = 1L)
        if( grepl(covariateType, pattern = "log") ){ 
          XPred_rsc_log <- jnp$log(  logConst + XPred_rsc ) 
          XPred_rsh_log <- jnp$log( logConst + XPred_rsh ) 
          XPred_rsd_log <- jnp$log( logConst + XPred_rsd ) 
        }

        # roll maxs
        obs_rxh <- jnp$array( zoo::rollapply(np$asanyarray(obs_h),
                                             width = rollCompute_window, partial = T,
                                             FUN = function(x){ median(x,na.rm=T) }, align="right",fill = NA)  )
        obs_rxc <- jnp$array( zoo::rollapply(np$asanyarray(obs_c),
                                             width = rollCompute_window, partial = T,
                                             FUN = function(x){ median(x,na.rm=T) }, align="right",fill = NA)  )
        obs_rxd <- jnp$array( zoo::rollapply(np$asanyarray(obs_d),
                                             width = rollCompute_window, partial = T,
                                             FUN = function(x){ median(x,na.rm=T) }, align="right",fill = NA) )
        XPred_rxc <- jnp$take_along_axis(obs_rxc, select_xc, axis = 1L)
        XPred_rxh <- jnp$take_along_axis(obs_rxh, select_xh, axis = 1L)
        XPred_rxd <- jnp$take_along_axis(obs_rxd, select_xd, axis = 1L)
        if( grepl(covariateType, pattern = "log" ) ){ 
          XPred_rxc_log <- jnp$log(  logConst  + XPred_rxc ) 
          XPred_rxh_log <- jnp$log( logConst +  XPred_rxh ) 
          XPred_rxd_log <- jnp$log(logConst + XPred_rxd ) 
        }
      }

      # generate noise (if pre-training)
      for(noise_ in simCovariates[grepl(simCovariates, pattern="noise")]){
        eval(parse(text = sprintf( "%s <- jnp$array(rnorm(XPred_c$shape[[1]]))", noise_)))
      }

      # generate zeros (if pre-training)
      for(zero_ in simCovariates[grepl(simCovariates,pattern="zeros")]){
        eval(parse(text = sprintf( "%s <- jnp$array(rep(0,times = XPred_c$shape[[1]]))", zero_)))
      }

      # specify observed data
      XPred <- eval(parse( text = sprintf("jnp$stack(list(%s),2L)",
                                  paste( c(names(simCovariates), "PolicyDat_seen"), collapse = ",") )  ))
      XPred <- ( XPred - jnp$expand_dims(jnp$array(t(SIM_GLOBAL_SCALE_MEAN)),0L)) /
                              jnp$expand_dims(jnp$array(t(0.001+SIM_GLOBAL_SCALE_SD)), 0L)

      # pure versions
      XPred_d_pure <- XPred_d #jnp$take(obs_d_pure, jnp$array(i_dpast <- 0L:(nTimesPast-1L)))
      YTrue_d_out_pure <- YTrue_d_out #jnp$take(obs_d_pure,jnp$array((max(i_dpast)+1):(max(i_dpast)+nTimesLook))) # future Y only i
      if(nTimesTotal > nTimesInLikelihood){ 
        #YTrue_d <- YTrue_d_pure <- jnp$take(YTrue_d, jnp$array(  (1:nTimesTotal)[-c(1:(nTimesTotal-nTimesInLikelihood))] - 1L ), axis = 1L) 
        #PolicyDat_all <- jnp$take(PolicyDat_all, jnp$array(  (1:nTimesTotal)[-c(1:(nTimesTotal-nTimesInLikelihood))] - 1L ), axis = 1L)
        
        YTrue_d <- YTrue_d_pure <- jnp$take(YTrue_d, jnp$array(  (1:nTimesTotal) - 1L ), axis = 1L) 
        PolicyDat_all <- jnp$take(PolicyDat_all, jnp$array(  (1:nTimesTotal) - 1L ), axis = 1L)
      }
      if(nTimesTotal <= nTimesInLikelihood){ 
        YTrue_d_pure <- YTrue_d  # take all 
      }
      XPred_h_pure <- XPred_h #jnp$take(obs_h_pure,jnp$array(i_dpast+forward_shift_h))
      XPred_c_pure <- XPred_c #jnp$take(obs_c_pure,jnp$array(i_dpast+forward_shift_c))
      XPred_pure <- XPred #jnp$stack(list(XPred_d_pure, XPred_h_pure, XPred_c_pure),1L)
      
      c2a <- function(in_){jnp$expand_dims(jnp$array(rep(in_, times = nBatch)), 1L)}
      ret_list <- list("XPred" = XPred,
                       "XPred_mask" = jnp$ones(c(XPred$shape[1:2],1L))$astype(XPred$dtype),
                       "Context" = c2a(0.),  # unused 
                       "Context_mask" = c2a(0.),  # unused 
                       "location_id_numeric" = jnp$expand_dims(jnp$array(rep(0., times = nBatch)), 1L),
                       "time_id_numeric" = jnp$expand_dims(jnp$array(rep(0., times = nBatch)), 1L),
                       "YTrue" = (YTrue_ <- jnp$expand_dims(YTrue_d,2L)),
                       "YTrue_mask" = jnp$ones_like(YTrue_),
                       "YTrue_out" = (YTrue_out_ <- jnp$expand_dims(YTrue_d_out,2L)), #4
                       "YTrue_out_mask" = jnp$ones_like(YTrue_out_),
                       "inv_beta_true" = inv_beta_true, #5
                       "gamma_true" = c2a(SimEntry$gamma), #6
                       "sigma_true" = c2a(SimEntry$sigma), #7
                       "xi_true" = c2a(SimEntry$xi), #8
                       "init_true" = init_true, #9
                       "XPred_pure" =  XPred_pure, #10
                       "YTrue_pure" = YTrue_d_pure, #11
                       "YTrue_out_pure" = YTrue_d_out_pure, #12
                       "PolicyDat_all" = PolicyDat_all, #13
                       "initial_shift" = initial_shift$astype(XPred$dtype)
      )
      }
    }
    
    # push data to target hardware and return
    return(   ret_list  ) 
  }
  
  # SimEntry$policy_effectiveness <- 0 # -> when effectiveness too big then infections crushed to 0
  SIM_GLOBAL_SCALE_MEAN_l <- SIM_GLOBAL_SCALE_SD_l <- SIM_GLOBAL_OUTCOME_SD_l <- list()
  scaling_outer_loops <- if(exists("SimScalingOuterLoops", inherits = FALSE)) ai(SimScalingOuterLoops) else 2L
  scaling_inner_loops <- if(exists("SimScalingInnerLoops", inherits = FALSE)) ai(SimScalingInnerLoops) else 10L
  ct <- 0; for(rr_ in 1:scaling_outer_loops){
    for(l_ in seq_len(scaling_inner_loops)){
      batch_l <- try(GetBatch_sim( nBatch ), T)
      # batch_l <- GetBatch_sim( nBatch, nTimes = 250L,SaveAt_VI_ODE_SIM = diffrax$SaveAt(ts = jnp$array(0L:(250L-1L))),nTimesLook = 50L )
      #batch_l <- try(GetBatch_sim( nBatch, nTimes = NTimeSteps_SIM, nTimesLook = nTimesLookahead ), T)
      # plot( c( np$array(batch_l$YTrue)[,,1] )); plot( c( np$array(batch_l$YTrue_out[[1]])[,1] ))
      # plot( c( np$array(batch_l$YTrue)[,,1] )+0.001,log="y");
      if("try-error" %in% class(batch_l)){ print(batch_l); stop("Problem in initial GetBatch_sim() call") }
      if(rr_ == 1){
        ct <- ct + 1
        print(" add masks here in application? ")
        SIM_GLOBAL_SCALE_MEAN_l[[ct]] <- apply(np$asanyarray( batch_l$XPred),3,mean)
        SIM_GLOBAL_SCALE_SD_l[[ct]] <- apply(np$asanyarray( batch_l$XPred),3,sd)
        SIM_GLOBAL_OUTCOME_SD_l[[ct]] <- sd(np$asanyarray( batch_l$YTrue )[,,1])
      }
    }
    if(rr_ == 2){
      plot(np$asanyarray( batch_l$YTrue ) [1,,1],type="l",col="gray",ylim=c(0,summary(c(np$asanyarray( batch_l$YTrue )))[c(6)]))
      for(i in 1:50){
        points(np$asanyarray( batch_l$YTrue ) [sample(batch_l$YTrue$shape[[1]],1),,1],type="l",col="black")
      }
      jax_batchx <-  batch_l$XPred; jax_batchy <- batch_l$YTrue
    }
  }
  # SimEntry
  # SimEntry$i0_a <- 3
  # SimEntry$xi <- 0.1
  # SimEntry$gamma <- 1
  suppressWarnings(sort((f2n(unlist(SimEntry)) - colMeans(apply(SimGrid,2,f2n),na.rm=T) ) / apply(apply(SimGrid,2,f2n),2,sd)) )
  # xi (Re-susceptibility rate).
  # It implies that the average duration of immunity is only 1/xi time steps (xi).
  
  # compute global mean/sd params from batch samples (LoTV)
  SIM_GLOBAL_SCALE_MEAN <- colMeans(SIM_GLOBAL_SCALE_MEAN_ <- do.call(rbind,SIM_GLOBAL_SCALE_MEAN_l))
  SIM_GLOBAL_SCALE_VAR <- (do.call(rbind,SIM_GLOBAL_SCALE_SD_l))^2
  SIM_GLOBAL_SCALE_SD <- sqrt( colMeans(SIM_GLOBAL_SCALE_VAR + t(SIM_GLOBAL_SCALE_MEAN-t(SIM_GLOBAL_SCALE_MEAN_))^2))
  SIM_GLOBAL_OUTCOME_SD <- colMeans(do.call(rbind,SIM_GLOBAL_OUTCOME_SD_l))
  rm(SIM_GLOBAL_SCALE_MEAN_l,SIM_GLOBAL_SCALE_SD_l,SIM_GLOBAL_OUTCOME_SD_l, SIM_GLOBAL_SCALE_VAR)
  }
  
  # read canonical tfrecords
  { 
    skip_tfrecords <- exists("SkipTfRecords", inherits = FALSE) && isTRUE(SkipTfRecords)

    if(skip_tfrecords){
      print2("Skipping TFRecord setup; simulation batches will stay in-memory.")
    }

    if(!skip_tfrecords){
    # warning("Forcing small sample size in tfrecord for testing!"); nSamplesTrain <- sample(c(10,100,1000),1);Sys.sleep(10)
    tf <- reticulate::import("tensorflow")
    RESHUFFLE_EACH_ITERATION <- isTRUE(get0("ReshuffleEachIteration", inherits = TRUE, ifnotfound = FALSE))
    
    parse_single_example_fxn <- function(example_proto){
      # Define the features to be extracted.
      parseText <- paste(sapply(names(batch_l),function(nl){
        sprintf("
        '%s' = tf$io$FixedLenFeature(list(), tf$string),
        '%s_shape' = tf$io$FixedLenFeature(list(), tf$string)
        ",nl,nl) }),collapse = ",")
      eval(parse(text = sprintf("feature_description <- list(
                                %s )", parseText )))
      
      # Parse the input `tf.train.Example` proto using the dictionary above.
      parseText <- paste(sapply(names(batch_l),function(nl){
        sprintf("
        '%s' = tf$cast(tf$reshape(tf$io$parse_tensor(example[['%s']], out_type = tf$float32), 
                                  tf$io$parse_tensor(example[['%s_shape']], out_type = tf$int32)), 
                                  dtype = %s)
        ",nl, nl, nl, DefaultDtypeTf) }),collapse = ",")
      example <- tf$io$parse_single_example(example_proto, feature_description)
      eval(parse(text = sprintf("data <- list(%s)",parseText)))
      return(data)
    }
    read_from_tfrecord <- function(file, batchSize, TfRecords_BufferScaler = 10L, nTake = NULL){
      raw_dataset <- tf$data$TFRecordDataset( file )
      parsed_dataset <- raw_dataset$map( parse_single_example_fxn )
      if(!is.null(nTake)){ parsed_dataset <- parsed_dataset$take(nTake) } # take only first nTake elements 
      parsed_dataset <- parsed_dataset$shuffle(buffer_size = tf$constant(as.integer(TfRecords_BufferScaler*batchSize), 
                                                                         dtype=tf$int64),
                                               reshuffle_each_iteration = RESHUFFLE_EACH_ITERATION)
      parsed_dataset <- parsed_dataset$batch( batchSize )
      return( parsed_dataset )
    }
  
    nBatch_SimGridGen <- if(exists("nBatch_SimGridGen", inherits = FALSE)) ai(nBatch_SimGridGen) else 256L
    tfrecord_file_train <- sprintf('%s/%s_%s.tfrecord', TfRecordDir, "train", SimEntry$BaseID) 
    tfrecord_file_inference <- sprintf('%s/%s_%s.tfrecord', TfRecordDir, "inference", SimEntry$BaseID) 
    nMonteEval <- if(exists("nMonteEval", inherits = FALSE)) ai(nMonteEval) else ceiling(1000/nBatch_SimGridGen)
    nTimesLookValidation <- nTimesLookValidationInference
    
    TFDatasetIterator_train <- reticulate::as_iterator(
          TFDataset_train <- read_from_tfrecord(file = tfrecord_file_train, 
                                                batchSize = nBatch, 
                                                nTake = ai(SimEntry$nSamplesTrain) ) )
    TFDatasetIterator_inference <- reticulate::as_iterator( 
      TFDataset_inference <- read_from_tfrecord(file = tfrecord_file_inference, 
                                                batchSize = nBatch)
    )
    }
  }
}
