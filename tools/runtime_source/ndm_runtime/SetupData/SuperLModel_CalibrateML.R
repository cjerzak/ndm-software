# Content of SuperLModel_CalibrateML.R
print2("Starting a round of SuperLModel_CalibrateML.R")
# initial calibration
{
  #for(cal in 1:2){ # 1 recalibrates s0, 2 recalibrates sd
  for(cal in 2){ print2("WARNING: ONLY SD CALIBRATION") # 1 recalibrates s0, 2 recalibrates sd
  #if(T == F){ # do no calibration
  print2(sprintf("Cal %s", cal))
  okc <- 0; ok <- F; while(!ok){
    if( (okc <- okc + 1) > 100){stop("Problem with GetBatch()")}
    # hist(np$array( batch_l$XPred[[1]][[1]] )) ;  np$array( batch_l$XPred[[1]][[1]] )[1,,]
    # SimEntry$betat_init <- 2
    # SimEntry$invbetat_sd <- 0.3
    # SimEntry$s0_sdprop <- 10^(-0.5)
    # SimEntry$s0_meanprop <- -2
    okhr_ <- F; while(okhr_ == F){ 
      batch_l_cal <- batch_l <- try(GetBatch(nBatch = nBatch, 
                                             INPUT_REF_DAT = input_df_red_in ),T)
      if(!"try-error" %in% class(batch_l)){okhr_ <- T}
    }
    # plot( c(np$array(batch_l$YTrue)[1,,,1]), cex = 0.25, log="")

    print2("Out Means & SDs:")
    try(print(apply(input_df_red_out[ input_df_red_full$time_id %in% times_out[1]:(max(times_out)+nTimesLookahead) , dataInputs_colnames_past],2,function(zer){mean(zer,na.rm=T)})), T)
    try(print( apply(input_df_red_out[ input_df_red_full$time_id %in% times_out[1]:(max(times_out)+nTimesLookahead) , dataInputs_colnames_past],2,function(zer){sd(zer,na.rm=T)}) ), T)

    if("try-error" %in% class(batch_l) ){ print2("Failed GetBatch()"); print( batch_l ) }
    if(!"try-error" %in% class(batch_l) ){
      # plot(  np$array(  batch_l$XPred[[1]][[1]]) [1,,1] )
      # hist(np$asarray(jax_batchx[[1]][[1]]))
      # plot( np$asarray(jax_batchx[[1]][[1]])[sample(1:10,1),,6] )
      # cbind(simCovariates,dataInputs_colnames_past)

      print2("Getting trial predictions...")
      # SIM_GLOBAL_SCALE_MEAN <- sim_dat_norm_list[[ FocalDisease ]][[1]]; SIM_GLOBAL_SCALE_SD <- sim_dat_norm_list[[ FocalDisease ]][[2]]
      # SimEntry <- SimGrid[FocalDisease,]; set.seed(999); batch_l_cal <- batch_l <- try(GetBatch(nBatch = nBatch, INPUT_REF_DAT = input_df_red_in ),T)
      
      # batch_l_cal <- batch_l <- try(GetBatch(nBatch = nBatch, INPUT_REF_DAT = input_df_red_in ),T)
      jax_batchy <- batch_l$YTrue_out; 
      jax_batchymask <- batch_l$YTrue_out_mask;
      if(is.null(batch_l$YTrue_mask)){ jax_batchymask <- jnp$ones(list(jax_batchy$shape[[1]],jax_batchy$shape[[2]],1L)) }
      hat_y <- GetPred_train(
        ModelList, (jax_batchx <- batch2package(batch_l)),
        state, PriorList, PolicyList,
        GetPredSaveAtInfo_default,
        jax$random$split(JaxKey(999L), nBatch))[[1]]
      # batch_l_cal$XPred[[1]][[1]]
      
      #plot(np$array(batch_l$XPred[[1]][[1]])[1,,1])
      #points(np$array(batch_l$XPred[[1]][[1]])[1,,2],col="gray")
      #points(np$array(batch_l$XPred[[1]][[1]])[1,,3],col="red")
      #hist(np$array(batch_l$XPred[[1]][[1]])[,,1]) 
      #hist(np$array(batch_l$XPred[[1]][[1]])[,,2])
      #hist(np$array(batch_l$XPred[[1]][[1]])[,,3])
      #plot(np$array(batch_l$XPred[[1]][[1]])[1,,4])

      print2("Getting trial loss...")
      loss <- getLoss_train(   ModelList, jax_batchx, jax_batchy, jax_batchymask, 
                               jnp$array(4), state, PriorList, PolicyList, 
                               GetPredSaveAtInfo_default, jax$random$split(JaxKey(9L), nBatch) )
      if(is.na(loss[[1]]$tolist())){ stop("NA in loss in CalibrateML") }
      if(T == F){ # deeper performance checks
        plot(   c(np$array( hat_y$y_mu )[,,1])  )
        plot(   c(np$array( hat_y$y_mu )[,,2])  )
        plot(   c(np$array( hat_y$y_sigma )[,,1])  )
        plot(   c(np$array( hat_y$y_sigma )[,,2])  )
        hist( np$array(hat_y$ODEParamsSampList$diff_eq_sol_ys.Neural1) )
        # jnp$mean(jnp$array(batch_l$XPred),axis = 0L:1L)
        # jnp$std(jnp$array(batch_l$XPred),axis = 0L:1L)
        par(mfrow=c(1,1));plot( x_<-c(np$array( jax_batchy )[,,1]), y_<-c(np$array( hat_y$y_mu )[,,1]),
                               col = c({pch_<- (np$array( hat_y$y_mu )[,,1]); pch_[] <- 1:nrow(pch_); pch_}), 
                               main="Outcome");abline(a=0,b=1); cor(x_,y_)
        plot(c(np$array( hat_y$y_mu )[,,2]),np$array( jax_batchy )[1,,2],main="Policy");abline(a=0,b=1)

        summary(lm(c(np$array( hat_y$y_mu )[,,1])~c(np$array( jax_batchy )[1,,,1])))
        summary(lm(c(np$array( hat_y$y_mu )[,,2])~c(np$array( jax_batchy )[1,,,2])))

        # save one disease
        if(T == F){
          write.csv( np$array(jax_batchx[[1]][[1]])[,,1], file = "~/Downloads/in_x1_OneDisease.csv" )
          write.csv( np$array(jax_batchx[[1]][[1]])[,,2], file = "~/Downloads/in_x2_OneDisease.csv" )
          write.csv( np$array(jax_batchx[[1]][[1]])[,,3], file = "~/Downloads/in_x3_OneDisease.csv" )
          write.csv( np$array( hat_y$y_mu )[,,1], file = "~/Downloads/hat_y_OneDisease.csv" )
        }

        # save multi disease
        if(T == F){
          write.csv(np$array(jax_batchx[[1]][[1]])[,,1], file = "~/Downloads/in_x1_MultiDisease.csv" )
          write.csv(np$array(jax_batchx[[1]][[1]])[,,2], file = "~/Downloads/in_x2_MultiDisease.csv" )
          write.csv(np$array(jax_batchx[[1]][[1]])[,,3], file = "~/Downloads/in_x3_MultiDisease.csv" )
          write.csv( np$array( hat_y$y_mu )[,,1], file = "~/Downloads/hat_y_MultiDisease.csv" )
        }

        if(T == F){
         yOne <- read.csv( file = "~/Downloads/hat_y_OneDisease.csv" )[,-1]
         xOne1 <- read.csv( file = "~/Downloads/in_x1_OneDisease.csv" )[,-1]
         xOne2 <- read.csv( file = "~/Downloads/in_x2_OneDisease.csv" )[,-1]
         xOne3 <- read.csv( file = "~/Downloads/in_x3_OneDisease.csv" )[,-1]

         yMulti <- read.csv( file = "~/Downloads/hat_y_MultiDisease.csv" )[,-1]
         xMulti1 <- read.csv( file = "~/Downloads/in_x1_MultiDisease.csv" )[,-1]
         xMulti2 <- read.csv( file = "~/Downloads/in_x2_MultiDisease.csv" )[,-1]
         xMulti3 <- read.csv( file = "~/Downloads/in_x3_MultiDisease.csv" )[,-1]
      }

         plot(unlist(yOne),unlist(yMulti))
         summary(lm(unlist(yOne)~unlist(yMulti)))
         plot(unlist(xOne1),unlist(xMulti1)); abline(a=0,b=1)
         plot(unlist(xOne2),unlist(xMulti2)); abline(a=0,b=1)
         plot(unlist(xOne3),unlist(xMulti3)); abline(a=0,b=1)
        # plot(unlist(xOne1),unlist(xOne2)); abline(a=0,b=1)
      }

      if('try-error' %in% class(hat_y)){print2("Calibration try-error"); print( hat_y )}
      if(!'try-error' %in% class( hat_y )){
      if(ModelType != "DecoderOnly"){
        # approximate targets
        gl__ <- c(VariationalInitGlobalMean,
                  VariationalInitNonEncLocalMean,
                  VariationalInitTemporalGlobalMean[!duplicated(VariationalInitTemporalGlobalMean)])
        gl_ <- np$asanyarray(SoftPlus(jnp$array(gl__)))
        names(gl_) <- names(gl__)
        l_ <- np$asanyarray(SoftPlus(jnp$array(VariationalInitEncLocalMean)))
        names(l_) <- names(VariationalInitEncLocalMean)

        # compare against sampled values
        samp_param <- hat_y$ODEParamsSampList
        initial_calibrations <- rbind( initial_calibrations <- c(),
                                      try2(c(summary(c(ExtractBetaDraw(samp_param)$BetaDraw)),
                                             sd(c(ExtractBetaDraw(samp_param)$BetaDraw)),
                                             "beta", l_[grep(names(l_),pattern="beta")[1]] )) )
        initial_calibrations <- rbind(initial_calibrations,
                                      try2(c(summary(c( np$asanyarray(SoftPlus( samp_param$diff_eq_sol_ys.sd_l ))[,,NeuralEmbedDim+1])),
                                             sd(c(np$asanyarray(SoftPlus( samp_param$diff_eq_sol_ys.sd_l ))[,,NeuralEmbedDim+1])),
                                             "sd_l", NA )))
        # diagnostics about temporal evolution - should always go up OR down
        beta_draw <- ExtractBetaDraw(samp_param)$BetaDraw
        if(!is.null(dim(beta_draw)) && nrow(beta_draw) > 0L){
          plot(beta_draw[sample(seq_len(nrow(beta_draw)), 1L),])
          #plot(np$asanyarray(SoftPlus( samp_param$diff_eq_sol_ys.sd_l ))[,,NeuralEmbedDim+1][sample(seq_len(dim(beta_draw)[1]),1L),])
          hist(apply(beta_draw,1,diff));abline(v=0,lwd = 2)
        }
        #hist(apply(np$asanyarray(SoftPlus( samp_param$diff_eq_sol_ys.sd_l ))[,,NeuralEmbedDim+1],1,diff));abline(v=0,lwd = 5)

        initial_calibrations <- rbind(initial_calibrations,
                                      try2(c(summary( tmp <- c(np$asanyarray(samp_param[[grep(names(samp_param),pattern="delta")]]) )),
                                             sd(tmp),
                                             "delta", na.omit(c(l_[grep(names(l_),pattern="delta")[1]],
                                                     gl_[grep(names(gl_),pattern="delta")[1]] )))))
        #plot(np$array( samp_param$diff_eq_sol_ys.MeanNeural2 )[1,,NeuralEmbedDim+1])
        #plot(np$array( samp_param$diff_eq_sol_ys.MeanNeural2 )[1,,NeuralEmbedDim+2])
        #plot(np$array( samp_param$diff_eq_sol_ys.MeanNeural2 )[1,,NeuralEmbedDim+3])
        #plot(np$array( samp_param$diff_eq_sol_ys.MeanNeural2 )[1,,NeuralEmbedDim+4])
        initial_calibrations <- rbind(initial_calibrations,
                                      try2(c(summary( tmp <- c(np$asanyarray( samp_param[[grep(names(samp_param),pattern="sigma")]]) )),
                                             sd(tmp ),
                                             "sigma", gl_[grep(names(gl_),pattern="sigma")[1]])))
        initial_calibrations <- rbind(initial_calibrations,
                                      try2(c(summary(  tmp <- c(np$asanyarray(samp_param[[grep(names(samp_param),pattern="gamma")]]) )),
                                             sd(tmp),
                                             "gamma",gl_[grep(names(gl_),pattern="gamma")[1]])))
        initial_calibrations <- rbind(initial_calibrations,
                                      try2(c(summary( tmp <- c(np$asanyarray( samp_param[[grep(names(samp_param),pattern="xi")]]) )),
                                             sd( tmp  ),
                                             "xi",gl_[grep(names(gl_),pattern="xi")[1]])))
        l_[c("s_l","e_l","i_l","r_l")] <- prop.table(l_[c("s_l","e_l","i_l","r_l")])
        # sum( l_[c("s_l","e_l","i_l","r_l")])
        initial_calibrations <- rbind(initial_calibrations,
                                      try2(c(summary(tmp<-np$asanyarray(samp_param$s_l_samp)) ,
                                             sd(tmp),
                                             "s_l",l_["s_l"]*CONST_N_r)))
        initial_calibrations <- rbind(initial_calibrations,
                                      try2(c(summary(tmp<-np$asanyarray( samp_param$e_l_samp)),
                                             sd(tmp), "e_l",
                                             l_["e_l"]*CONST_N_r)))
        initial_calibrations <- rbind(initial_calibrations,
                                      try2(c(summary(np$asanyarray( samp_param$i_l_samp) ),
                                             sd(np$asanyarray( samp_param$i_l_samp) ),"i_l",
                                             l_["i_l"]*CONST_N_r)))
        initial_calibrations <- rbind(initial_calibrations,
                                      try2(c(summary(np$asanyarray( samp_param$r_l_samp)  ),
                                             sd( np$asanyarray( samp_param$r_l_samp) ),"r_l",
                                             l_["r_l"]*CONST_N_r )))
        initial_calibrations <- as.data.frame(initial_calibrations,stringsAsFactors=F)
        colnames(initial_calibrations)[(ncol(initial_calibrations)-2):ncol(initial_calibrations)] <- c("SD", "ParamName","TargetVal")
        initial_calibrations$RelDeviation <- abs(f2n(initial_calibrations[,c("Mean")]) - f2n(initial_calibrations[,ncol(initial_calibrations)])) /
          f2n(initial_calibrations[,ncol(initial_calibrations)])
        # round(f2n(initial_calibrations[,c("Mean")]),2L)
        plot(  f2n( initial_calibrations[["RelDeviation"]] ),
               main = "Initialization Checks", cex=0 , ylab = "Rel Deviation",log="y")
        text(  f2n( initial_calibrations[["RelDeviation"]] ),
               labels = substr(initial_calibrations$ParamName,start=0,stop=3),cex=0.5 )
        if(grepl(version$os, pattern="darwin")){
         print( dplyr::as_tibble( initial_calibrations[,c("ParamName","Mean","SD","TargetVal","RelDeviation")] ))
        }
        R0_target <- f2n(initial_calibrations[initial_calibrations$ParamName=="beta_lt",]$TargetVal) / f2n(initial_calibrations[initial_calibrations$ParamName=="gamma",]$TargetVal)
        R0_mean <- f2n(initial_calibrations[initial_calibrations$ParamName=="beta_lt",]$Mean) / f2n(initial_calibrations[initial_calibrations$ParamName=="gamma",]$Mean)
        print2(sprintf("Basic repro target: %.3f, Basic repro mean: %.3f, Implied initial death rate: %.3f",
                      R0_target, R0_mean,
                      f2n(initial_calibrations$Mean[ initial_calibrations$ParamName=="delta" ]) *(
                        f2n(initial_calibrations$Mean[ initial_calibrations$ParamName=="sigma" ])*f2n(initial_calibrations$Mean[ initial_calibrations$ParamName=="e_l" ]) -
                          f2n(initial_calibrations$Mean[ initial_calibrations$ParamName=="gamma" ])*f2n(initial_calibrations$Mean[ initial_calibrations$ParamName=="i_l" ])
                      )))
        }
        hat_y <- hat_y$y_mu
        if("try-error" %in% class(hat_y)){ print2("Failed GetPred_train()") }
        if(!"try-error" %in% class(hat_y)){ ok <- T }

        # do calibration
        { # WARNING: RE-CALIBRATING MAY CAUSE ERRORS!
          # calibrate sd
          # if(T == F){ print2("Experimentally not calibrating s0")
          if(T == T){print2("Experimentally calibrating s0")
          if(cal == 2){
            print2("Calibrating sd0...")
            tmppp <- np$asanyarray( hat_y - batch_l$YTrue_out )
            #tmppp <- apply(tmppp[,1,],2,sd)
            #tmppp <- apply(apply(tmppp,2:3,sd), 2, median)
            tmppp <- apply(apply(tmppp,2:3,sd), 2, mean)
            tmppp <- ifelse(tmppp > 100, tmppp, InvSoftPlus_r(tmppp))[1:nOutcomes]
            ModelList$ScaleList$ScaleBayes$VarInit <-  jnp$array( tmppp )
            #ModelList[[ListIndices['ScaleList']]][[1]][[1]] <- jnp$array( tmppp ); #rm( tmp )   # calibrates ScaleList[[1]][[1]] i.e.
          }
          }

          # calibrate s0
          if(cal == 1){
            print2("Calibrating s0...")
            # question: can the softmax calibration allow for improved calibration?
            # if alpha inits get too big, hard to change. (they are scaled by 100 anyway. )
            # deterministic approach

            print2(sprintf("nDimOutput_dense: %i",nDimOutput_dense))
            #InvFxn <- function(x)(np$array(InvSoftPlus(x)))
            #InvFxn <- function(x)(np$array(log(x)))
            #InvFxn <- function(x)(np$array((x)))
            #diff_ <- sapply(up_seq <- seq(1,10000,by=0.1),function(z){ InvFxn(min_frac*z) + InvFxn(1*z) })
            #up_ <- up_seq[  which.min(abs(diff_)) ]

            #InvFxn <- function(x)(np$array(InvSoftPlus(x))); up_ <- 100; min_frac <- 0.0001; min_ <- np$array(InvFxn(min_frac*up_))[1]; max_ <- np$array(InvFxn(up_))[1]
            #InvFxn <- function(x)(np$array(InvSoftPlus(x))); min_ <- np$array(InvFxn( (tar_ <- .00001)^(1/DirPower)) ); max_ <-  np$array(InvSoftPlusLargeInputApprox(np$array((tar_*100000)^(1/DirPower)))) # max_ * 1/0.0001 # if not standardizing
            InvFxn <- function(x)(np$array((x))); min_ <- -2; max_ <- 2  #
            print(c(min_, max_))
            # jnp$power(SoftPlus(min_),DirPower); jnp$power(SoftPlus(max_),DirPower)
            #InvFxn <- function(x)(np$array(InvSoftPlus(x))); min_ <- exp(0.5)-1; max_ <- (exp(0.5)-1)*10000# max_ * 1/0.0001
            s_cal_vec <- seq(from = min_, to = max_, length.out = 4)
            #s_cal_vec <- seq(from = np$array(log(min_frac*up_))[1],to = np$array(log(up_))[1], length.out = 4)
            #up_ <- (10000); s_cal_vec <- seq(from = np$array(InvSoftPlus(0.00001*up_))[1],to = np$array(up_)[1], length.out = 4)
            s_cal_vec <- expand.grid(s_cal_vec,s_cal_vec,s_cal_vec,s_cal_vec)
            if(!all(apply(s_cal_vec,1,sd) == 0)){
              s_cal_vec <- t(s_cal_vec[apply(s_cal_vec,1,sd) > 0,])
            }

            # random sampling approach
            ss_vec <- rep(NA,times = ncol(s_cal_vec))
            for(fri in 1:ncol(s_cal_vec)){
              print2( round(fri/ncol(s_cal_vec), 3) )
              InitMean_vec[c("s_l","e_l","i_l","r_l")] <- s_cal_vec[,fri]
              ModelList[[ListIndices['TSList']]][[5]][[2]] <- jnp$array( InitMean_vec )
              yhat_cal <- try(GetPred_train_jit(
                ModelList, jax_batchx,
                state, PriorList, PolicyList, 
                GetPredSaveAtInfo_default, jax$random$split(JaxKey(6L),nBatch))[[1]],T)

              ss_vec[fri] <- mean(apply(abs( (np$asanyarray( yhat_cal$y_mu  )[,,1] -
                                            np$asanyarray( batch_l$YTrue )[,,1] )) *
                                          np$array(batch_l$YTrue[[2]])[,,1],1,max) )
              # plot( np$array(yhat_cal$y_mu) )
              # hist( np$array(yhat_cal$ODEParamsSampList$diff_eq_sol_ys.Neural1)[,,NeuralEmbedDim+1] )
              # hist( np$array(yhat_cal$ODEParamsSampList$diff_eq_sol_ys.Neural1)[,,NeuralEmbedDim+2] )
            }
            #plot( ss_vec )
            #head(np$array(ModelList[[4]][[3]][[2]][[2]]))
            #plot(np$array(ModelList[[4]][[3]][[2]][[2]]))
            InitMean_vec[c("s_l","e_l","i_l","r_l")] <- s_cal_vec[,which.min(ss_vec)[1]]
            # SoftPlus(np$array(s_cal_vec[,which.min(ss_vec)[1]]))
            ModelList[[ListIndices['TSList']]][[5]][[2]] <- jnp$array( InitMean_vec )

            # check updates
            # plot( np$array( InitMean_vec ), np$array(ModelList[[ListIndices['BNList']]][ length( ModelList[[ListIndices['BNList']]] )][[1]][[2]][[2]])); abline(a=0,b=1)
            # plot( np$array( InitMean_vec ) - np$array(ModelList[[ListIndices['BNList']]][ length( ModelList[[ListIndices['BNList']]] )][[1]][[2]][[2]]) )
          }
        }
        #ModelList[[2]][[ListIndices['ScaleList']]][[1]][[1]] <- jnp$array(InvSoftPlus(10*sd(hat_y$to_py() - batch_l$YTrue$to_py())))  # calibrates ScaleList[[1]][[1]] i.e.
        hat_y <- np$asanyarray( hat_y )
        #hist( hat_y )

        #for(ifa in (samp_<-  1:10)){
        par(mfrow=c(1,2))
        try(plot( np$asanyarray(jax_batchy)[1,,1],
                    main = "Examples of\n YTrue",
                    type = "l",lwd = 2,col=1,
                    ylim = c(0,1*max(np$array(jax_batchy)[,,1]))),T)
        for(ifa in (samp_<-1:jax_batchy$shape[[1]])){
          try(points(np$asanyarray(jax_batchy)[ifa,,1],
                     type = "l",lwd = 2,pch=""),T)
        }

        plot(hat_y[1,,1], main = "Examples of Y-hat",
             type = "l",lwd = 0.1, col=1,ylim = c(0,max(hat_y[samp_,,1])))
        for(ifa in (samp_<-1:jax_batchy$shape[[1]])){
          points(hat_y[ifa,,1],type = "l",lwd = 0.2,pch = "^")
        }
        par(mfrow=c(1,1))

        Sys.sleep(2L)
        # hist( np$asanyarray(  jax_batchx[[1]][[1]] )  )
        # hist( np$asanyarray(  jax_batchy )  )
        # hist( hat_y )
      }
    }
  }
  }
  print2("Done with a round of SuperLModel_CalibrateML.R")
}
