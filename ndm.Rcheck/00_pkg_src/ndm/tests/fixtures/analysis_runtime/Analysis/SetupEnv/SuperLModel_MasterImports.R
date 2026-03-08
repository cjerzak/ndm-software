# Package-owned backend bootstrap for the bundled Analysis2 runtime.
{
  Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "0")
  if (!is.null(GPU_MEM_FRAC)) {
    Sys.setenv(XLA_PYTHON_CLIENT_MEM_FRACTION = sprintf("%s", GPU_MEM_FRAC))
  }
  if (isTRUE(ReSaveTfRecords)) {
    Sys.setenv(CUDA_VISIBLE_DEVICES = "")
    Sys.setenv(XLA_PYTHON_CLIENT_PREALLOCATE = "false")
  }

  backend_conda_env <- get0(
    "conda_env",
    inherits = TRUE,
    ifnotfound = Sys.getenv(
      "NDM_SOFTWARE_CONDA_ENV",
      unset = Sys.getenv(
        "NDM_CONDA_ENV",
        unset = if (grepl(version$os, pattern = "darwin")) "jax_cpu" else "ndm_software_env"
      )
    )
  )

  backend <- ndm::ndm_initialize_backend(
    conda_env = backend_conda_env,
    float_type = as.character(floatType),
    import_tensorflow = TRUE
  )

  jax <- backend$jax
  jnp <- backend$jnp
  np <- backend$np
  optax <- backend$optax
  eq <- backend$eq
  diffrax <- backend$diffrax
  flash_mha <- backend$flash_mha
  py_gc <- backend$py_gc
  tf <- backend$tf
  jaxFloatType <- backend$jaxFloatType
  send2cpu <- backend$send2cpu
  send2gpu <- backend$send2gpu
  num_devices <- length(jax$devices())
  if (isTRUE(force2GPU) && !identical(backend$default_backend, "cpu")) {
    send2cpu <- send2gpu <- function(x) {
      x
    }
  }
  DefaultDtypeTf <- if (identical(as.character(floatType), "64")) "tf$float64" else "tf$float32"
  oryx <- backend$oryx
  SoftPlus <- backend$SoftPlus
  Sigmoid <- backend$Sigmoid
  InvSoftPlus <- backend$InvSoftPlus
  InvSoftPlusLargeInputApprox <- function(z) {
    jnp$subtract(jnp$logaddexp(z, jnp$array(0.)), jnp$array(0.5))
  }
  SoftMax <- jax$nn$softmax
  InvSoftMax <- function(z) {
    jnp$add(jnp$log(z), jnp$array(jnp$log(1)))
  }
  InvSigmoid <- function(z) {
    jnp$log(jnp$divide(z, jnp$subtract(1., z)))
  }
  switch_filter_jit <- eq$filter_jit

  library(fastmatch)
  library(reticulate)

  JaxKey <- function(int_) {
    if (is.environment(int_) || "python.builtin.object" %in% class(int_)) {
      return(jax$random$PRNGKey(int_))
    }
    jax$random$PRNGKey(as.integer(int_))
  }

  SoftPlus_r <- function(x) {
    x[x < 100] <- log(exp(x) + 1.)[x < 100]
    x[x >= 100] <- x[x >= 100]
    x
  }

  InvSoftPlus_r <- function(x) {
    x[x < 100] <- log(exp(x) - 1.)[x < 100]
    x[x >= 100] <- x[x >= 100]
    x
  }
}
