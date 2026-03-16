# Content of SuperLModel_helperFxns.R
ai <- as.integer

mean2 <- function(x){mean(x,na.rm=T)}

summary2 <- function(x){summary(x,na.rm=T)}

clipAt <- function(zr, qval = 0.975, na.rm=T){
  zr[ zr > quantile(zr,probs = qval, na.rm=na.rm) ] <- quantile(zr, probs = qval, na.rm=na.rm) ;
  zr[ zr < quantile(zr,probs = 1-qval, na.rm=na.rm) ] <- quantile(zr, probs = 1-qval, na.rm=na.rm) ;
  return( zr ) 
}

f2n <- function(.){as.numeric(as.character(.))}

print2 <- function(text){ print( sprintf("[%s] %s" ,format(Sys.time(), "%Y-%m-%d %H:%M:%S"),text) ) }

TrimPercentiles <- function(x,trim.percentile=0.975,na.rm = T){
  x[x > quantile(x,probs = trim.percentile, na.rm = na.rm)] <- quantile(x,na.rm = na.rm, probs = trim.percentile) # upper trim
  x[x < quantile(x,1-trim.percentile)] <- quantile(x, probs =  1-trim.percentile) # lower trim
  return(x)
}

GlobalPartition <- function(zer, eq_fxn){
  yes_branches <- rrapply::rrapply(zer,f=function(zerr){
    unlist(ifelse(eq_fxn(zerr),yes = list(zerr), no = list(NULL))[[1]])
  },how="list")
  no_branches <- rrapply::rrapply(zer,f=function(zerrr){
    unlist(ifelse(eq_fxn(zerrr),yes = list(NULL), no = list(zerrr))[[1]])
  },how="list")
  list(yes_branches,no_branches)
}
PartFxn <- function(zerz){ !"first_time_index" %in% names(zerz)}

DisappList <- function(.){ret_<-.;if(is.list(.)){if(length(unlist(.))==0){ret_ <- NULL}};return(ret_)}

neuralODE_lowRank <- F; lowRankDims <- 7L
residCon_neuralODE <- T
nODE_indices <- function(nDimInput_nODE, nDepth_nODE,
                         nWidth_nODE, nDimOutput_nODE){
  if(!neuralODE_lowRank){ wts_resid <- residCon_neuralODE * ( nDimInput_nODE * nDimOutput_nODE) }
  if(neuralODE_lowRank){ # nDimInput_nODE by nDimOutput_nODE
    wts_resid <- (nDimInput_nODE*lowRankDims) + (lowRankDims) + (lowRankDims*nDimOutput_nODE)
  }
  wts1 <- nDimInput_nODE * nWidth_nODE
  if(neuralODE_lowRank){
    wts1 <-  (nDimInput_nODE*lowRankDims) + (lowRankDims) + (lowRankDims*nWidth_nODE)
  }
  if(ODE_NORM_TYPE == "none"){  ln1_center <-  ln1_scale <-  0L }
  if(ODE_NORM_TYPE == "pre-activation1"){  ln1_center <-  0L; ln1_scale <-  nDimInput_nODE }
  if(ODE_NORM_TYPE == "pre-activation2"){  ln1_scale <- ln1_center <-  nWidth_nODE }
  if(ODE_NORM_TYPE == "post-activation"){  ln1_center <-  ln1_scale <-  nWidth_nODE }


  bias1 <- nWidth_nODE
  if(nDepth_nODE > 1){
    for(dr_ in 2:nDepth_nODE){
      if(ODE_NORM_TYPE == "none"){
        eval(parse(text = sprintf("ln%s_center <-  0L",dr_)))
        eval(parse(text = sprintf("ln%s_scale <-  0L",dr_)))
      }
      if(ODE_NORM_TYPE == "pre-activation1"){
        eval(parse(text = sprintf("ln%s_center <-  0L",dr_)))
        eval(parse(text = sprintf("ln%s_scale <-  nDimInput_nODE",dr_)))
      }
      if(ODE_NORM_TYPE == "pre-activation2"){
        eval(parse(text = sprintf("ln%s_center <-  nWidth_nODE",dr_)))
        eval(parse(text = sprintf("ln%s_scale <-  nWidth_nODE",dr_)))
      }
      if(ODE_NORM_TYPE == "post-activation"){
        eval(parse(text = sprintf("ln%s_center <-  nDimOutput_nODE",dr_)))
        eval(parse(text = sprintf("ln%s_scale <-  nDimOutput_nODE",dr_)))
      }
      if(neuralODE_lowRank){
        eval(parse(text = sprintf("wts%s <-  (nWidth_nODE*lowRankDims) + (lowRankDims) + (lowRankDims*nWidth_nODE)",dr_)))
      }
      if(!neuralODE_lowRank){ eval(parse(text = sprintf("wts%s <-  nWidth_nODE*nWidth_nODE",dr_))) }
      eval(parse(text = sprintf("bias%s <-  nWidth_nODE",dr_)))
    }
  }
  if(!neuralODE_lowRank){ eval(parse(text = sprintf("wts%s <-  nWidth_nODE*nDimOutput_nODE",(nDepth_nODE+1) ))) }
  if(neuralODE_lowRank){ # nDimInput_nODE by nDimOutput_nODE
    eval(parse(text = sprintf("wts%s <-  (nWidth_nODE*lowRankDims) + (lowRankDims) + (lowRankDims*nDimOutput_nODE)",nDepth_nODE+1)))
  }
  eval(parse(text = sprintf("bias%s <-  nDimOutput_nODE",nDepth_nODE+1)))
  eval(parse(text = sprintf("ln%s_scale <- 0L",(nDepth_nODE+1) )))
  eval(parse(text = sprintf("ln%s_center <- 0L",(nDepth_nODE+1) )))

  vec_ <- eval(parse(text = paste("c(",
    ifelse(residCon_neuralODE,no="",yes = "'wts_resid'=wts_resid,"),
    paste(sapply(1:(nDepth_nODE+1),function(d_r)sprintf(
    "
    'ln%s_scale' = ln%s_scale,
    'ln%s_center' = ln%s_center,
    'wts%s' = wts%s,
    'bias%s' = bias%s
    ", d_r,d_r,
       d_r,d_r,
       d_r,d_r,
       d_r,d_r)),
    collapse = ","),")",collapse="")))
  vec_ <- vec_[ vec_ != 0 ]
  vec_ <- cumsum(vec_)
  indices_ <- sapply(1:length(vec_),function(zer){
    if(zer == 1){in_ <- 1:vec_[zer]}
    if(zer > 1){ in_ <- (vec_[zer-1]+1):vec_[zer] }
    return( jnp$array( in_ - 1L ) ) })
  names(indices_) <- names(vec_)
  return(  indices_ )
}

if(T == F){
f_neuralODE <- diffrax$ODETerm(  f_neuralODE_R <- function(t, y, args){
  # y is the evolving state of the system and is given by a neural network

  # add dimension for matmul
  y_orig <- (y <- jnp$expand_dims(y, 0L))

  # residual connection
  y_resid <- jnp$matmul( y, args$nODE_wt_resid ) # output dim (assumed)

  # first hidden layer
  if(ODE_NORM_TYPE == "none" | ODE_NORM_TYPE == "post-activation" | ODE_NORM_TYPE == "pre-activation2"){ y_in <- y }
  if(ODE_NORM_TYPE == "pre-activation1"){
    # LN
    # y <- jnp$multiply(jax$nn$standardize(y, axis = 1L, epsilon = ep_LN), args$nODE_ln1_scale)

    # RMSE norms
    y <- jnp$multiply(jnp$divide(y, jnp$sqrt(jnp$add(ep_LN, jnp$mean(jnp$square(y),keepdims=T)))), args$nODE_ln1_scale)
  }
  y <- jnp$matmul( y, args$nODE_wt1) # + args$nODE_bias1
  if(ODE_NORM_TYPE == "pre-activation2"){
    # y has dimensions 1 by nHiddenDim
    y <- jnp$add( jnp$multiply(
      #jax$nn$standardize(y, axis = 1L, epsilon = ep_LN),
      jnp$divide(y, jnp$sqrt(jnp$add(ep_LN, jnp$mean(jnp$square(y),keepdims=T)))),
      args$nODE_ln1_scale),
      args$nODE_ln1_center ) # normalize, scale, and shift
  }
  y <- jax$nn$swish( y ) # non-linearity (be mindful of residual connection and scale [>0])

  # additional hidden layers
  if(nDepth_nODE > 1){
    for(da_ in 2:nDepth_nODE){
      #eval(parse(text = sprintf("wts_da <- jnp$reshape(args$nODE_wt%s,list(nWidthODEHidden_ts_local, nWidthODEHidden_ts_local))",da_)))
      eval(parse(text = sprintf("wts_da <- args$nODE_wt%s",da_)))
      eval(parse(text = sprintf("bias_da <- args$nODE_bias%s",da_)))
      #if(nODE_weightNorm == T){wts_da <- jnp$divide(wts_da,jnp$add(jnp$array(0.01),jnp$linalg$norm(wts_da,axis = 0L,keepdims=T))); wts_da <- jnp$multiply(wts_da,args$nODE_scale[[da_]]) }
      y_resid <- y
      y <- jnp$add(jnp$matmul(y, wts_da), bias_da)
      y <- jax$nn$swish( y ) # non-linearity
      y <- jnp$add( y_resid, y ) # residual connection
    }
  }

  eval(parse(text = sprintf("wts_da <-  args$nODE_wt%s", nDepth_nODE+1)))
  eval(parse(text = sprintf("bias_da <- args$nODE_bias%s", nDepth_nODE+1)))

  if(ODE_NORM_TYPE == "post-activation"){
    y <- jnp$add( jnp$multiply(
      #jax$nn$standardize(y , axis = 1L,  epsilon = ep_LN),
      jnp$divide(y, jnp$sqrt(jnp$add(ep_LN, jnp$mean(jnp$square(y),keepdims=T)))),
      args$nODE_ln1_scale),
      args$nODE_ln1_center) # normalize, scale, shift
  }

  # no output bias
  #y <- jnp$matmul(  y,  jnp$reshape(wts_da, list(-1L,bias_da$shape[[1]]))  )
  y <- jnp$matmul(  y,  wts_da  )

  # residual connection to baseline
  y <-  y_resid + y * jnp$array(1/sqrt(nTimesTotal))

  # scale output
  y <-  y * args$nODE_out_scale[[1]]

  # squeeze extra dimension created for matmul
  y <- jnp$squeeze(y, 0L)

  # clip output
  y <- jnp$clip(y, jnp$array(-10.), jnp$array(10.))

  # return
  return(   y    )
}   )
}

try2 <- function(expr){
  ret_ <- tmp_<-try(expr,T);
  if('try-error' %in% class(tmp_)){ ret_ <- NULL }
  return( ret_  ) }

ExtractBetaDraw <- function(PARAM_DRAW){
  if(temporalModelType == "linearInterpolation"){
    BetaDraw <- np$asanyarray( PARAM_DRAW$Neural1_samp$ys )
    TSDraw <- np$asanyarray( PARAM_DRAW$Neural1_samp$ts )
  }
  if(temporalModelType == "neuralODE"){
    BetaDraw <- np$asanyarray( SoftPlus( PARAM_DRAW$diff_eq_sol_ys.Neural1)) [,, LocalNeuralEmbedDim+1 ]
    TSDraw <- np$asanyarray( PARAM_DRAW$diff_eq_sol_ts )
  }
  return(list("TSDraw" = TSDraw,
              "BetaDraw" = BetaDraw)  )
}

TFConst2JAXArray <- function(l_){ lapply(l_,function(l_){ jnp$array(l_) })}

batch2package <- function(l_){
  return( list(list(l_$XPred, l_$XPred_mask),
               list(l_$Context, l_$Context_mask),
               list(l_$location_id_numeric),
               list(l_$time_id_numeric) ) )
}

# gen fig functions 
heatMap <- function(x, y, z,
                    main = "",
                    N,yaxt = NULL,
                    xlab = "",
                    ylab ="",horizontal = NULL,
                    useLog = "", legend.width = 1,
                    ylim = NULL, xlim = NULL, zlim = NULL,
                    add.legend = T,legend.only=F,
                    vline = NULL, col_vline = "black",
                    hline = NULL, col_hline = "black",
                    cex.lab = 2,cex.main = 2, myCol = NULL,
                    includeMarginals = F,
                    marginalJitterSD_x = 0.01,
                    marginalJitterSD_y = 0.01,
                    openBrowser = F){
  if(openBrowser){browser()}
  library(fields); library(akima)
  s_ <- interp(x=x, y=y, z=z,
               xo=seq(min(x),max(x),length=N),
               yo=seq(min(y),max(y),length=N),duplicate="mean")
  if(is.null(xlim)){xlim = c(summary( s_$x ))[c(1,6)]}
  if(is.null(ylim)){ylim = c(summary( s_$y ))[c(1,6)]}
  imageFxn <- image.plot
  if(add.legend == F){imageFxn <- image}
  if(!grepl(useLog,pattern="z")){
    imageFxn(s_, xlab = xlab, ylab = ylab, log = useLog, cex.lab = cex.lab , main = main, cex.main = cex.main,
             col = myCol,  xlim = xlim, ylim = ylim,
             legend.width = legend.width,
             horizontal = horizontal, yaxt = yaxt,
             zlim = zlim, legend.only = legend.only)
  }
  if(grepl(useLog,pattern="z")){
    useLog <- gsub(useLog,pattern="z",replace ="")
    zTicks <- summary( c(s_$z ))
    ep_ <- 0.001
    zTicks[zTicks<ep_] <- ep_
    zTicks <- exp(seq(log(min(zTicks)),log(max(zTicks)), length.out = 10))
    zTicks <- round(zTicks, abs(min(log(zTicks,base=10))))
    s_$z[s_$z<ep_] <- ep_
    imageFxn(s_$x,s_$y,log(s_$z),yaxt = yaxt,
             axis.args = list(at=log(zTicks),labels = zTicks), main = main, cex.main = cex.main,
             xlab = xlab, ylab = ylab,log = useLog, cex.lab = cex.lab,
             xlim = xlim,
             ylim = ylim,,horizontal=horizontal,
             col = myCol,legend.width=legend.width,
             zlim = zlim,legend.only = legend.only)
  }
  if(!is.null(vline)){ abline(v=vline,lwd=10,col=col_vline) }
  if(!is.null(hline)){ abline(h=hline,lwd=10,col=col_hline) }
  
  if(includeMarginals == T){
    points(x+rnorm(length(y),sd=marginalJitterSD_x*sd(x)), rep(ylim[1]*1.1,length(y)),pch="|",col="darkgray")
    points(rep(xlim[1]*1.1,length(x)), y+rnorm(length(y),sd=sd(y)*marginalJitterSD_y),pch="-",col="darkgray")
  }
}
colSummmary <- function(x){ 
    apply(x,2,function(x_){
      x_f2n <- f2n(x_)
      if(!all(is.na(x_f2n))){ x_ <- mean(x_f2n, na.rm=T) }
      if(all(is.na(x_f2n))){ x_ <- names(table(x_)[which.max(table(x_))[1]]) }
      return( x_ ) })
}
cols2numeric <- function(x){ 
  apply(x,2,function(x_){
    x_f2n <- f2n(x_)
    if(!all(is.na(x_f2n))){ x_ <- x_f2n }
    if(all(is.na(x_f2n))){ x_ <- x_ }
    return( x_ ) })
}
MakeHeatMap <- function(factor1, factor2, outcome, dat, lm_obj, pdf_path, 
                        extrap_factor1 = 1, extrap_factor2 = 1, 
                        useLog = "", OUTCOME_SCALER = 1, 
                        OutcomeTransformFxn = function(x){x},
                        openBrowser = F){ 
  if(openBrowser){browser()}
  eval(parse(text = sprintf('
    dat_new <- expand.grid("%s" = seq((1/extrap_factor1)*min(f2n(dat[,factor1])),
                                      extrap_factor1*max(f2n(dat[,factor1])), 
                                      length.out = 50L),
                           "%s" = seq((1/extrap_factor2)*min(f2n(dat[,factor2])),
                                      extrap_factor2*max(f2n(dat[,factor2])), 
                                      length.out = 50L))
                           ', factor1, factor2) ) ) 
  dat_new <- cbind(dat_new,
                   as.data.frame(t(colSummmary(dat[,!colnames(dat) %in% c(factor1,factor2,outcome)])))
                   )
  dat_new$Yhat <- OutcomeTransformFxn( c(predict(lm_obj, newdata = dat_new)) ) 
  dat_new$Yhat <- dat_new$Yhat * OUTCOME_SCALER
  #hist(dat_new$Yhat)
  
  pdf( pdf_path )
  {
    par(mar=c(5,5,3,1), mfrow = c(1,1))
    xlim__ <- c(summary(dat_new[,factor1])[c(1,6)])
    ylim__ <- c(summary(dat_new[,factor2])[c(1,6)])
    zlim__ <- c(summary(dat_new$Yhat)[c(1,6)])
    myColScheme <- viridis::plasma(15,alpha = 0.9)
    heatMap(x = dat_new[,factor1],
            y = dat_new[,factor2],
            z = dat_new$Yhat,
            N = 50, myCol  = myColScheme,
            xlim = xlim__,  ylim = ylim__, zlim = zlim__, 
            main = "", 
            add.legend = T, horizontal = F,legend.width = 1,
            openBrowser = F,
            xlab = names(factor1), ylab = names(factor2),
            #yaxt ="n", useLog="xyz",
            useLog = useLog,
            includeMarginals = F)
    # mtext(text = expression("Entropy: Concentrated"%<->%"Dispersed"), side = 2,line = 3,cex=2)
  }
  dev.off()
}

se <- function(zer){ sqrt( 1/length(zer) * var(zer) ) }

clippedMean <- function(zer,pct = 0.05,directional = "lower", na.rm = T){
  if(directional %in% c("lower","both")){ zer[zer < quantile(zer,pct/2,na.rm=na.rm)] <- quantile(zer,pct/2,na.rm=na.rm) }
  if(directional %in% c("upper","both")){ zer[zer > quantile(zer,1-pct/2,na.rm=na.rm)] <- quantile(zer,1-pct/2,na.rm=na.rm) }
  return( mean(zer) )
}

robust_cut <- function(x, n_bins = 2L){

  # extract the non-NA uniques
  uniq_vals <- sort(unique(x[!is.na(x)]))
  
  # 3. fallback if too few unique values
  if (length(uniq_vals) <= n_bins) {
    warning(
      sprintf(
        "Only %d unique non NA values but %d bins requested, returning rank(x)",
        length(uniq_vals), n_bins
      )
    )
    out <- rep(NA_integer_, length(x))
    out[!is.na(x)] <- rank(x[!is.na(x)], ties.method = "min")
    return(out)
  }
  
  # 4. compute pretty breaks and enforce full coverage
  #brks <- pretty(uniq_vals, n = n_bins + 0L)
  #brks[1]                 <- min(uniq_vals)
  #brks[length(brks)]      <- max(uniq_vals)
  
  # finally cut
  cut(x,
      breaks         = n_bins,
      include.lowest = TRUE,
      labels         = FALSE
  )
}

# create oryx helpers 
{
  # ---------------------------------------------------------------------
  # Minimal Oryx stand-ins (JAX via reticulate) 
  # ---------------------------------------------------------------------
  oryx <- list()
  
  # ---------- helpers ----------
  .ensure_key <- function(seed) {
    if (is.null(seed)) stop("seed must be provided")
    # If looks like a PRNGKey (shape (2,))
    try({
      shp <- seed$shape
      if (!is.null(shp) && length(shp) == 1 && as.integer(shp[[1]]) == 2) return(seed)
    }, silent = TRUE)
    # If it's a JAX scalar or R numeric, convert to PRNGKey
    if (!is.null(seed$shape) && length(seed$shape) == 0) {
      return(jax$random$PRNGKey(as.integer(np$asarray(seed))))
    }
    if (is.numeric(seed)) return(jax$random$PRNGKey(as.integer(seed)))
    # last resort
    jax$random$PRNGKey(as.integer(reticulate::py_to_r(seed)))
  }
  
  .shape_of <- function(x) {
    tryCatch({ as.integer(unlist(x$shape)) }, error = function(e) integer(0L))
  }
  .as_shape <- function(s) {
    if (is.null(s)) return(integer(0L))
    if (is.list(s)) return(as.integer(unlist(s)))
    if ("python.builtin.object" %in% class(s)) {
      shp <- tryCatch(as.integer(unlist(s$shape)), error = function(e) NULL)
      if (!is.null(shp)) return(shp)
    }
    as.integer(s)
  }
  .make_dist <- function(name, params, sample_fn, log_prob_fn = NULL) {
    list(
      name = name,
      parameters = params,
      sample = sample_fn,
      log_prob = if (is.null(log_prob_fn))
        function(...) stop("log_prob not implemented for ", name) else log_prob_fn
    )
  }
  
  # ---------- distributions ----------
  oryx$Normal <- function(loc, scale) {
    if (!"python.builtin.object" %in% class(loc))   loc   <- jnp$array(loc)
    if (!"python.builtin.object" %in% class(scale)) scale <- jnp$array(scale)
    sample_fn <- function(sample_shape = NULL, seed = NULL) {
      key <- .ensure_key(seed)
      shp <- c(.as_shape(sample_shape), .shape_of(loc))
      eps <- jax$random$normal(key, shape = shp)
      eps * scale + loc
    }
    logp_fn <- function(x) {
      -0.5 * jnp$square((x - loc) / scale) - jnp$log(scale) - 0.5 * jnp$log(2 * pi)
    }
    .make_dist("Normal", list(loc = loc, scale = scale), sample_fn, logp_fn)
  }
  
  oryx$Uniform <- function(low = 0., high = 1.) {
    if (!"python.builtin.object" %in% class(low))  low  <- jnp$array(low)
    if (!"python.builtin.object" %in% class(high)) high <- jnp$array(high)
    sample_fn <- function(sample_shape = NULL, seed = NULL) {
      key <- .ensure_key(seed)
      jax$random$uniform(key, shape = .as_shape(sample_shape), minval = low, maxval = high)
    }
    logp_fn <- function(x) {
      jnp$where((x >= low) & (x <= high), -jnp$log(high - low), -jnp$inf)
    }
    .make_dist("Uniform", list(low = low, high = high), sample_fn, logp_fn)
  }
  
  # Accepts either scale_diag= or scale=
  oryx$MultivariateNormalDiag <- function(loc, scale_diag = NULL, scale = NULL) {
    sd <- if (!is.null(scale_diag)) scale_diag else scale
    if (is.null(sd)) stop("Provide 'scale_diag' (preferred) or 'scale'.")
    if (!"python.builtin.object" %in% class(loc)) loc <- jnp$array(loc)
    if (!"python.builtin.object" %in% class(sd))  sd  <- jnp$array(sd)
    k <- as.integer(tail(.shape_of(loc), 1L))
    sample_fn <- function(sample_shape = NULL, seed = NULL) {
      key <- .ensure_key(seed)
      shp <- c(.as_shape(sample_shape), .shape_of(loc))
      eps <- jax$random$normal(key, shape = shp)
      eps * sd + loc
    }
    logp_fn <- function(x) {
      diff  <- (x - loc) / sd
      const <- 0.5 * as.numeric(k) * jnp$log(2 * pi)
      -0.5 * jnp$sum(jnp$square(diff) + 2 * jnp$log(sd), axis = -1L) - const
    }
    .make_dist("MultivariateNormalDiag", list(loc = loc, scale_diag = sd), sample_fn, logp_fn)
  }
  
  oryx$MultivariateNormalTriL <- function(loc, scale_tril) {
    if (!"python.builtin.object" %in% class(loc))        loc        <- jnp$array(loc)
    if (!"python.builtin.object" %in% class(scale_tril)) scale_tril <- jnp$array(scale_tril)
    k <- as.integer(tail(.shape_of(loc), 1L))
    sample_fn <- function(sample_shape = NULL, seed = NULL) {
      key <- .ensure_key(seed)
      shp <- c(.as_shape(sample_shape), k)
      eps <- jax$random$normal(key, shape = shp)
      jnp$dot(eps, jnp$transpose(scale_tril)) + loc
    }
    logp_fn <- function(x) {
      diff  <- x - loc
      y     <- jax$scipy$linalg$solve_triangular(scale_tril, jnp$transpose(diff),
                                                 lower = TRUE, check_finite = FALSE)
      y     <- jnp$transpose(y)
      logdet <- jnp$sum(jnp$log(jnp$diag(scale_tril)))
      const  <- 0.5 * as.numeric(k) * jnp$log(2 * pi)
      -0.5 * jnp$sum(jnp$square(y), axis = -1L) - logdet - const
    }
    .make_dist("MultivariateNormalTriL", list(loc = loc, scale_tril = scale_tril), sample_fn, logp_fn)
  }
  
  oryx$MultivariateNormalFullCovariance <- function(loc, covariance_matrix) {
    if (!"python.builtin.object" %in% class(loc))               loc <- jnp$array(loc)
    if (!"python.builtin.object" %in% class(covariance_matrix)) covariance_matrix <- jnp$array(covariance_matrix)
    L <- jnp$linalg$cholesky(covariance_matrix)
    d <- oryx$MultivariateNormalTriL(loc = loc, scale_tril = L)
    d$parameters$covariance_matrix <- covariance_matrix
    d
  }
  
  # ---------- math helpers ----------
  oryx$math <- list()
  oryx$math$fill_triangular <- function(vec) {
    if (!"python.builtin.object" %in% class(vec)) vec <- jnp$array(vec)
    n_elem <- as.integer(vec$size)
    m <- as.integer(floor((sqrt(8 * n_elem + 1) - 1) / 2))
    if (m * (m + 1) / 2 != n_elem) stop("fill_triangular: length does not match any triangular number")
    L   <- jnp$zeros(list(m, m), dtype = vec$dtype)
    idx <- jnp$tril_indices(m)
    L$at[idx[[1]], idx[[2]]]$set(vec)
  }
  
  # ---------- (optional) KL divergence ----------
  oryx$kl_divergence <- function(p, q) {
    if (p$name == "Normal" && q$name == "Normal") {
      mu1 <- p$parameters$loc;  sd1 <- p$parameters$scale
      mu2 <- q$parameters$loc;  sd2 <- q$parameters$scale
      return(jnp$log(sd2 / sd1) + (jnp$square(sd1) + jnp$square(mu1 - mu2)) / (2 * jnp$square(sd2)) - 0.5)
    }
    if (grepl("MultivariateNormal", p$name) && grepl("MultivariateNormal", q$name)) {
      if (!is.null(p$parameters$scale_diag) && !is.null(q$parameters$scale_diag)) {
        s1 <- p$parameters$scale_diag; s2 <- q$parameters$scale_diag
        m1 <- p$parameters$loc;       m2 <- q$parameters$loc
        k  <- as.numeric(tail(.shape_of(m1), 1L))
        term2 <- jnp$sum(jnp$square(s1) / jnp$square(s2))
        term3 <- jnp$sum(jnp$square(m2 - m1) / jnp$square(s2))
        return(0.5 * (term2 + term3 - k))
      }
    }
    jnp$zeros(list())
  }
}

print2("Done loading helper functions...")



