# Exercise the runtime functions directly without generating a TFRecord corpus.
ndm_test_architecture_function <- function(path, name, env) {
  find <- function(expr) {
    if (missing(expr) || !is.call(expr)) return(NULL)
    if (identical(expr[[1L]], as.name("<-")) &&
        identical(expr[[2L]], as.name(name))) return(expr)
    for (i in seq_along(expr)[-1L]) {
      found <- find(expr[[i]])
      if (!is.null(found)) return(found)
    }
    NULL
  }
  for (expr in ndm_test_runtime_source_expressions(path)) {
    found <- find(expr)
    if (!is.null(found)) return(eval(found, env))
  }
  stop("Missing runtime function: ", name)
}

ndm_test_architecture_env <- function(backbone = FALSE, compute_dtype = "native", checkpointing = FALSE, depth = 2L) {
  conda_env <- ndm_require_backend_test_stack("neural architecture regressions")
  backend <- ndm_initialize_backend(
    conda_env = conda_env, float_type = "32", import_tensorflow = FALSE
  )
  env <- new.env(parent = baseenv())
  for (name in c("jax", "jnp", "np", "eq", "optax", "oryx", "jaxFloatType", "InvSoftPlus")) {
    env[[name]] <- backend[[name]]
  }
  env$ai <- as.integer
  if (backbone) {
    list2env(list(
      backbonePath = "initialize", ModelType = "DecoderOnly", ModelDims = 16L,
      ModelDepth = depth, TransformerComputeDtype = compute_dtype,
      TransformerActivationCheckpointing = checkpointing, TransformerHeads = 2L, TransformerKVHeads = 1L,
      TransformerHeadDim = 8L, TransformerKVGroupSize = 2L,
      TransformerList = replicate(depth, list(), simplify = FALSE), UseLatentAttention = FALSE,
      WideMultiplicationFactor = 2L, nOutcomes = 1L,
      key = env$jax$random$PRNGKey(123L)
    ), env)
    eval(ndm_test_runtime_source_expressions(
      "ModelDefiners/SuperLModel_BackboneTransformer.R"
    ), env)
    ndm_test_architecture_function(
      "ModelDefiners/SuperLModel_BuildML.R", "RMSNorm", env
    )
    env$NormFxn <- env$RMSNorm
    env$ffmap <- env$jax$vmap(function(layer, x) layer(x), in_axes = list(NULL, 0L))
  }
  env
}

ndm_test_architecture_array <- function(x, env) as.array(env$np$array(x))

