# Content of SuperLModel_MasterImports.R
# jax + related imports
# GPU_MEM_FRAC <- 1; ReSaveTfRecords <- FALSE;  force2GPU <- function(x){x}; floatType <- "32"
{
  # r packages;
  # conda create -n jax_gpu python=3.13
  # fastmatch reticulate gtools qdapRegex dplyr 
  # conda install -c conda-forge r zip r-fastmatch r-reticulate r-gtools r-bestnormalize r-qdapregex r-tidyverse r-rrapply r-zip
  # install.packages(c("rrapply"))
  # conda install conda-forge::parallel
  # sudo apt update; sudo apt install parallel
  
  # python packages: 
  # pip install -U "jax[cuda12]" uv
  # uv pip install --upgrade optax equinox diffrax tensorflow tensorflow_datasets nvitop
  # uv pip install --upgrade optax equinox diffrax tensorflow tensorflow_probability tensorflow_datasets nvitop

  # may require on GPU JAX + CUDA 12/13 (installs the matching jaxlib wheel automatically)
  # and numpy 1.
  # pip install "jax[cuda12]==0.5" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

  # setup tf and python
  #https://developer.apple.com/metal/jax/
  # use newest jax / jaxlib version as found here: https://github.com/google/jax
  # https://developer.apple.com/forums/thread/731071
  # note: with metal, 0 dim matrices kill
  # broken for now

  # see https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
  Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "0")
  if(!grepl(version$os,pattern="darwin")){
    if( length(list.files('/ihme')) == 0 & 
              !grepl(Sys.info()["nodename"],pattern="utexas") ){  # personal GPU 
      # Sys.setenv(RETICULATE_PYTHON = "/home/cjerzak/miniconda3/")
      # reticulate::use_python("/home/cjerzak/miniconda3/envs/tensorflow_m1/bin/python3") if needing to init a python too
      Sys.setenv_text <- '
            Sys.setenv(XLA_PYTHON_CLIENT_PREALLOCATE = "true");
            Sys.setenv(PATH = "/home/cjerzak/miniconda3/bin:/home/cjerzak/miniconda3/condabin:/usr/local/cuda-12.4/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin$");
            Sys.setenv(LD_LIBRARY_PATH = "/usr/local/cuda-12.4/lib64");
            Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "0");
            Sys.setenv(SHELL = "/usr/bin/bash")'
      eval(parse(text = Sys.setenv_text))
      if(!is.null(GPU_MEM_FRAC)){ Sys.setenv(XLA_PYTHON_CLIENT_MEM_FRACTION = sprintf("%s",GPU_MEM_FRAC)) } 
      if(ReSaveTfRecords){
        Sys.setenv(CUDA_VISIBLE_DEVICES = ""); Sys.setenv(XLA_PYTHON_CLIENT_PREALLOCATE = "false")
        reticulate::use_condaenv(conda_env<-"jax_cpu", required = T); print("Using CPU")
      }
      if(!ReSaveTfRecords){
        reticulate::use_condaenv(conda_env<-"jax_gpu", required = T); print("Using GPU") 
      }
    }
    if( length(list.files('/ihme')) == 0 & 
               grepl(Sys.info()["nodename"],pattern="utexas") ){ #  texas cluster 
      if(!is.null(GPU_MEM_FRAC)){ Sys.setenv(XLA_PYTHON_CLIENT_MEM_FRACTION = sprintf("%s",GPU_MEM_FRAC)) }
      if(ReSaveTfRecords){
        Sys.setenv(CUDA_VISIBLE_DEVICES = ""); Sys.setenv(XLA_PYTHON_CLIENT_PREALLOCATE = "false")
        reticulate::use_condaenv(conda_env<-"jax_cpu", required = T); print("Using CPU")
      }
      if(!ReSaveTfRecords){
        reticulate::use_condaenv(conda_env<-"jax_gpu", required = T); print("Using GPU") 
      }
    }
    if(length(list.files('/ihme')) != 0){ 
      if(!is.null(GPU_MEM_FRAC)){ Sys.setenv(XLA_PYTHON_CLIENT_MEM_FRACTION = sprintf("%s",GPU_MEM_FRAC)) }
      Sys.setenv(R_ZIPCMD = "/ihme/homes/cjerzak/miniconda3/envs/jax_gpu/bin/zip")
      # to log in in terminal: cjerzak@gen-nvidia-gpu-d01.cluster.ihme.washington.edu
      # . ~/.bashrc; conda activate jax_gpu
      # run a bash job: sh runParallelGPUSims.sh
      ssh_connection <- ssh::ssh_connect("cjerzak@gen-nvidia-gpu-d01.cluster.ihme.washington.edu", verbose = 2)
      reticulate::use_python("/ihme/homes/cjerzak/miniconda3/envs/jax_gpu/bin/python", required = T)
      # https://jrkwon.com/2022/11/22/cuda-and-cudnn-inside-a-conda-env/
    }
  }
  if(grepl(version$os,pattern="darwin")){
    reticulate::use_condaenv(conda_env <- "jax_cpu", required = T);print("Using CPU backend")
    #reticulate::use_condaenv("jax_gpu", required = T) # experimental 
  }

  # see also
  # CONDA_SUBDIR=osx-arm64 conda create -n tensorflow_m1 python==3.9
  # conda create -n jax_cpu python=3.13
  # conda activate jax_cpu
  # pip install numpy jax jaxlib optax tensorflow equinox diffrax tensorflow-datasets
  # also need to install R packages 
  # reticulate::use_condaenv("jax_gpu")
  
  # https://github.com/google/jax/issues/8074
  jnp <- (jax <- reticulate::import("jax"))$numpy
  flash_mha <- try(reticulate::import("flash_attn_jax")$flash_mha,TRUE)
  np <- reticulate::import("numpy") # version? 
  optax <- reticulate::import("optax")
  eq <- reticulate::import("equinox")
  diffrax <- reticulate::import("diffrax")
  #oryx <- reticulate::import("tensorflow_probability.substrates.jax.distributions") # depreciated
  #oryx <- reticulate::import("distrax")
  py_gc <- reticulate::import("gc")
  # tensorflow_datasets make sure is installed also

  # load in libraries 
  library(fastmatch); library(reticulate)
  
  # enable 64 bit computations
  if(tolower(jax$default_backend()) == "cpu"){
    send2cpu <- send2gpu <- function(x){ x }
    #jax$config$update("jax_enable_x64", TRUE); jaxFloatType <- jnp$float64; print("***Using float64***");DefaultDtypeTf <- "tf$float64"
    if(floatType == "32"){jax$config$update("jax_enable_x64", FALSE); jaxFloatType <- jnp$float32;  print("***Using float32***"); DefaultDtypeTf <- "tf$float32"}
    if(floatType == "64"){jax$config$update("jax_enable_x64", TRUE); jaxFloatType <- jnp$float64;  print("***Forcing float64***"); DefaultDtypeTf <- "tf$float64"}
  }
  if(tolower(jax$default_backend()) != "cpu"){
    send2cpu <- function(x){ jax$device_put(x, jax$devices("cpu")[[1]]) }
    send2gpu <- function(x){ jax$device_put(x, jax$devices( ifelse(Sys.info()[["machine"]]=="arm64", yes = "METAL", no = "gpu"))[[1]])}
    if(force2GPU){ send2cpu <- send2gpu <- function(x){ x } }
    
    #jnp$array(34)$devices(); send2cpu(jnp$array(42))$devices();send2gpu(send2cpu(jnp$array(42)))$devices()
    if(floatType == "32"){jax$config$update("jax_enable_x64", FALSE); jaxFloatType <- jnp$float32;  print("***Using float32***"); DefaultDtypeTf <- "tf$float32"}
    if(floatType == "64"){jax$config$update("jax_enable_x64", TRUE); jaxFloatType <- jnp$float64;  print("***Forcing float64***"); DefaultDtypeTf <- "tf$float64"}
  }
  #jax$config$update("jax_enable_x64", FALSE); jaxFloatType <- jnp$float32;  print("***Forcing float32***")
  #jax$config$update("jax_enable_x64", TRUE); jaxFloatType <- jnp$float64;  print("***Forcing float64***")
  

  # helper fxns
  JaxKey <- function(int_){ jax$random$PRNGKey(int_)}
  SoftPlus_r <- function(x){
    x[x<100] <- log(exp(x) + 1.)[x<100]
    x[x>=100] <- x[x>=100]; return( x )
  }
  InvSoftPlus_r <- function(x){
    x[x<100] <- log(exp(x) - 1.)[x<100]
    x[x>=100] <- x[x>=100]; return( x )
  }

  if(T == F){
    # jax gpu checks
    # reticulate::use_condaenv("tensorflow_m1", required = T);
    reticulate::use_condaenv("tensorflow_m1_jaxgpu", required = T);
    Sys.setenv(EQX_ON_ERROR = "nan")
    jax <- reticulate::import("jax")
    jnp <- reticulate::import("jax.numpy")
    eq <- reticulate::import("equinox")
    diffrax <- reticulate::import("diffrax")
    x <- jnp$array(1:10.,dtype = jnp$float32)
    jnp$sqrt( x )

    #jax$config$update("jax_enable_x64", TRUE)
    #Sys.setenv(jax_enable_x64 = 'True')
    #jnp$array(1.) # -> fails
    # jnp$array(1.,dtype = jnp$float64) # succeeds
    #jnp$array(1.,dtype = jnp$float64) # succeeds
    jax$lib$xla_bridge$get_backend()$platform
    jax$lib$xla_bridge$default_backend()
    #jax$config$update('jax_platform_name', 'gpu')
    jax$devices()
    jax$config$jax_platforms

    ff <- function(y){   return(jnp$negative(y))  }
    ff_j <- jax$jit(ff)
    tmp <- jnp$array(-4.,dtype = jnp$float64)
    jnp$square(tmp)
    system.time(replicate(1000,ff(tmp)));
    system.time(replicate(1000,ff_j(tmp)));
    system.time(replicate(1000,-(-4)))

    f <- function(t, y, args){ return(  jnp$negative(y)   )}
    term = diffrax$ODETerm( f )
    solver <- diffrax$Tsit5()
    #solver <- diffrax$Dopri8()
    y0 = jnp$array(t(c(2.,34.)),dtype = jnp$float32)
    y0 = jnp$array(as.matrix(c(2.,34.)),dtype = jnp$float32)
    #y0 = jnp$array((c(2.,34.)),dtype = jnp$float32)
    y0 <- jnp$array(as.matrix(c(2.,2)),dtype = jnp$float32)
    #y0 <- jnp$expand_dims(y0,0L)
    #stepsize_controller = diffrax$PIDController(rtol=1e-5, atol=1e-5)
    solution <- diffrax$diffeqsolve(terms = term,
                                   solver = solver,
                                   throw = F,
                                   #t0 = 0., t1 = 1., dt0 = 0.01,
                                   t0 = jnp$array(0.), t1 = jnp$array(10.), dt0 = jnp$array(0.01),
                                   #saveat = diffrax$SaveAt(ts= jnp$array(c(0.1, 0.5)) ),
                                   saveat = diffrax$SaveAt(ts = jnp$array((c(1,2)) ), dense = TRUE), #solver_state = T, controller_state = T, made_jump = T),
                                   y0 = y0)
                                   #t0 = jnp$array(0), t1 = jnp$array(1), dt0 = jnp$array(0.01),
                                   #throw = T,

    ## zz
    # Global flag to set a specific platform, must be used at startup.
    jax$config$update('jax_platform_name', 'cpu')
    x = jnp$square(jnp$array(2,dtype=jnp$float32))
    x$device_buffer$device()  # CpuDevice(id=0)

    #with(device){
    #f_cpu = jax$jit(diffrax$diffeqsolve, backend='cpu')
    #solution = f_cpu(term, solver, t0=0, t1=1, dt0=0.1, y0=y0,
    if(T == F){
      vector_field <- function(t, y, args){ jnp$negative( y ) }
      term = diffrax$ODETerm(vector_field)
      solver = diffrax$Dopri5()
      saveat = diffrax$SaveAt(ts=c(0., 1., 2., 3.))
      stepsize_controller = diffrax$PIDController(rtol=1e-5, atol=1e-5)

      sol = diffrax$diffeqsolve(term, solver,
                                #t0 = 0., t1 = 3., dt0 = 0.01,
                                t0 = jnp$array(0),
                                t1 = jnp$array(5.),
                                dt0 = jnp$array(0.01),
                                y0 = jnp$array(1),
                                saveat = saveat,
                                stepsize_controller = stepsize_controller)

    }
  }

  # package versions compatable at one point
  # jax==0.3.21 tensorflow-probability==0.18.0
  # tensorflow==? jaxlib==0.3.20 oryx==0.2.4 optax==0.1.4 chex==0.1.5

  #jnp$abs(jnp$array(1.,dtype=jnp$float64))
  # jnp$array(1.)
  #jnp$stack(list(jnp$array(0.), jnp$array(0.)),0L)

  # setup sharding
  #https://docs.kidger.site/equinox/examples/parallelism/
  num_devices <- 1
  if(T == F){
    num_devices <- length(  jax$devices() )

    # setup devices
    devices1d <-  reticulate::r_to_py( jax$experimental$mesh_utils$create_device_mesh( num_devices ) )
    devices2d <-  reticulate::r_to_py( jax$experimental$mesh_utils$create_device_mesh( list(num_devices,1L) ) )
    devices3d <-  reticulate::r_to_py( jax$experimental$mesh_utils$create_device_mesh( list(num_devices,1L,1L) ) )

    # setup sharding
    shard1d <- jax$sharding$PositionalSharding(  devices1d   )

    shard2d <- jax$sharding$PositionalSharding(  list(devices2d)   )
    shard2d <- shard2d$transpose()

    shard3d <- jax$sharding$PositionalSharding(  list(list(devices3d))   )
    shard3d <- shard3d$transpose()
  }

  # define various functions
  InvSoftPlus <- (function(z){jnp$log(jnp$subtract(jnp$exp(z),jnp$array(1.)))})
  InvSoftPlusLargeInputApprox <- (function(z){
    jnp$subtract(jnp$logaddexp(z, jnp$array(0.)),jnp$array(0.5))
  })
  SoftPlus <- jax$nn$softplus

  #
  SoftMax <- jax$nn$softmax
  InvSoftMax <- (function(z){jnp$add(jnp$log(z), jnp$array(jnp$log(1)))})

  InvSigmoid <- (function(z){jnp$log(jnp$divide(z,jnp$subtract(1.,z)))})
  Sigmoid <- jax$nn$sigmoid

  Identity <- (function(z){z})

  LinearizeNestedList <- function (NList, LinearizeDataFrames = FALSE, NameSep = "/",
            ForceNames = FALSE){
    stopifnot(is.character(NameSep), length(NameSep) == 1)
    stopifnot(is.logical(LinearizeDataFrames), length(LinearizeDataFrames) ==
                1)
    stopifnot(is.logical(ForceNames), length(ForceNames) == 1)
    if (!is.list(NList))
      return(NList)
    if (is.null(names(NList)) | ForceNames == TRUE)
      names(NList) <- as.character(1:length(NList))
    if (is.data.frame(NList) & LinearizeDataFrames == FALSE)
      return(NList)
    if (is.data.frame(NList) & LinearizeDataFrames == TRUE)
      return(as.list(NList))
    A <- 1
    B <- length(NList)
    while (A <= B) {
      Element <- NList[[A]]
      EName <- names(NList)[A]
      if (is.list(Element)) {
        Before <- if (A == 1)
          NULL
        else NList[1:(A - 1)]
        After <- if (A == B)
          NULL
        else NList[(A + 1):B]
        if (is.data.frame(Element)) {
          if (LinearizeDataFrames == TRUE) {
            Jump <- length(Element)
            NList[[A]] <- NULL
            if (is.null(names(Element)) | ForceNames ==
                TRUE)
              names(Element) <- as.character(1:length(Element))
            Element <- as.list(Element)
            names(Element) <- paste(EName, names(Element),
                                    sep = NameSep)
            NList <- c(Before, Element, After)
          }
          Jump <- 1
        }
        else {
          NList[[A]] <- NULL
          if (is.null(names(Element)) | ForceNames == TRUE)
            names(Element) <- as.character(1:length(Element))
          Element <- LinearizeNestedList(Element, LinearizeDataFrames,
                                         NameSep, ForceNames)
          names(Element) <- paste(EName, names(Element),
                                  sep = NameSep)
          Jump <- length(Element)
          NList <- c(Before, Element, After)
        }
      }
      else {
        Jump <- 1
      }
      A <- A + Jump
      B <- length(NList)
    }
    return(NList)
  }
  
  # print default device 
  print(sprintf("Default device: %s", jnp$array(1.)$devices()))
  
  # jit function (for debugging)
  switch_filter_jit <- eq$filter_jit  # for production implementation
  #switch_filter_jit <- function(x,...){  x  }; warning("Turning off jit for debugging" )
}

