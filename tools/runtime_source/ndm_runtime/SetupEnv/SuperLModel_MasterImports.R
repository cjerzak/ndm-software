# Package-owned backend bootstrap for the bundled Analysis2 runtime.
{
  Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "0")

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

  backend_compute_request <- get0(
    "computeBackend",
    inherits = TRUE,
    ifnotfound = get0("compute_backend", inherits = TRUE, ifnotfound = NULL)
  )
  if (is.null(backend_compute_request)) {
    legacy_force_to_gpu <- get0(
      "force2GPU",
      inherits = TRUE,
      ifnotfound = NULL
    )
    backend_compute_request <- if (is.null(legacy_force_to_gpu)) {
      "auto"
    } else if (isTRUE(legacy_force_to_gpu)) {
      "gpu"
    } else {
      "cpu"
    }
  }

  backend <- ndm::ndm_initialize_backend(
    conda_env = backend_conda_env,
    float_type = as.character(floatType),
    import_tensorflow = TRUE,
    compute_backend = backend_compute_request,
    gpu_mem_frac = GPU_MEM_FRAC
  )

  jax <- backend$jax
  jnp <- backend$jnp
  np <- backend$np
  optax <- backend$optax
  eq <- backend$eq
  diffrax <- backend$diffrax
  compute_backend_requested <- backend$compute_backend_requested
  compute_backend_resolved <- backend$compute_backend_resolved
  if (is.null(compute_backend_resolved)) {
    compute_backend_resolved <- backend$compute_backend
  }
  compute_backend <- compute_backend_resolved
  computeBackend <- compute_backend
  force2GPU <- identical(compute_backend, "gpu")
  selected_device <- backend$selected_device
  device_platform <- backend$device_platform
  device_kind <- backend$device_kind
  device_vendor <- backend$device_vendor
  device_provenance <- backend$device_provenance
  accelerator_runtime <- backend$accelerator_runtime
  is_cuda <- isTRUE(backend$is_cuda) && identical(accelerator_runtime, "cuda")
  NDM_CUDA_ATTENTION_AVAILABLE <- force2GPU && is_cuda
  flash_mha <- if (isTRUE(NDM_CUDA_ATTENTION_AVAILABLE)) backend$flash_mha else NULL
  py_gc <- backend$py_gc
  tf <- backend$tf
  if (!is.null(tf)) {
    # TensorFlow is used only for CPU-side TFRecord input. Hide its GPU devices
    # through the TensorFlow API without changing JAX accelerator visibility.
    try(tf$config$set_visible_devices(list(), "GPU"), silent = TRUE)
  }
  jaxFloatType <- backend$jaxFloatType
  send2device <- backend$send2device
  send2cpu <- backend$send2cpu
  send2gpu <- backend$send2gpu
  selected_devices <- tryCatch(
    jax$devices(device_platform),
    error = function(...) list(selected_device)
  )
  if (length(selected_devices) == 0L) {
    selected_devices <- list(selected_device)
  }
  device_count <- backend$device_count
  if (is.null(device_count)) {
    device_count <- backend$device_provenance$device_count
  }
  if (is.null(device_count)) {
    device_count <- length(selected_devices)
  }
  device_count <- as.integer(device_count)
  num_devices <- min(length(selected_devices), device_count)
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

  jax_sharding <- tryCatch(reticulate::import("jax.sharding"), error = function(...) NULL)

  JaxKey <- function(int_) {
    if (is.environment(int_) || "python.builtin.object" %in% class(int_)) {
      return(jax$random$PRNGKey(int_))
    }
    jax$random$PRNGKey(as.integer(int_))
  }

  ndm_runtime_is_python_array <- function(x) {
    if (!("python.builtin.object" %in% class(x))) {
      return(FALSE)
    }
    shape <- try(x$shape, silent = TRUE)
    !inherits(shape, "try-error") && !is.null(shape)
  }

  ndm_runtime_array_rank <- function(x) {
    if (!ndm_runtime_is_python_array(x)) {
      dims <- dim(x)
      return(if (is.null(dims)) 0L else length(dims))
    }
    dims <- try(reticulate::py_to_r(x$shape), silent = TRUE)
    if (inherits(dims, "try-error") || is.null(dims)) {
      return(0L)
    }
    length(dims)
  }

  ndm_runtime_make_mesh <- function() {
    if (is.null(jax_sharding) || num_devices <= 1L) {
      return(NULL)
    }
    tryCatch(
      {
        if (!is.null(jax$make_mesh)) {
          mesh <- tryCatch(
            jax$make_mesh(
              list(as.integer(num_devices)),
              list("data"),
              devices = selected_devices
            ),
            error = function(...) NULL
          )
          if (!is.null(mesh)) {
            return(mesh)
          }
        }
        jax_sharding$Mesh(selected_devices, list("data"))
      },
      error = function(...) NULL
    )
  }

  ndm_mesh <- ndm_runtime_make_mesh()
  ndm_sharding_enabled <- !is.null(ndm_mesh)

  ndm_runtime_partition_spec <- function(rank, sharded = FALSE) {
    if (is.null(jax_sharding)) {
      return(NULL)
    }
    rank <- as.integer(rank)
    entries <- rep(list(NULL), max(rank, 0L))
    if (isTRUE(sharded) && rank > 0L) {
      entries[[1L]] <- "data"
    }
    do.call(jax_sharding$PartitionSpec, entries)
  }

  ndm_runtime_named_sharding <- function(x, sharded = FALSE) {
    if (!ndm_sharding_enabled || !ndm_runtime_is_python_array(x)) {
      return(NULL)
    }
    jax_sharding$NamedSharding(
      ndm_mesh,
      ndm_runtime_partition_spec(ndm_runtime_array_rank(x), sharded = sharded)
    )
  }

  ndm_runtime_shard_tree <- function(x, sharded = FALSE) {
    if (!ndm_sharding_enabled || is.null(eq$filter_shard)) {
      return(NULL)
    }
    shardings <- tryCatch(
      jax$tree_util$tree_map(
        function(leaf) {
          if (!ndm_runtime_is_python_array(leaf)) {
            return(NULL)
          }
          ndm_runtime_named_sharding(leaf, sharded = sharded)
        },
        x
      ),
      error = function(...) NULL
    )
    if (is.null(shardings)) {
      return(NULL)
    }
    tryCatch(eq$filter_shard(x, shardings), error = function(...) NULL)
  }

  ndm_runtime_data_to_device <- function(x) {
    if (is.list(x)) {
      return(lapply(x, ndm_runtime_data_to_device))
    }
    if (!ndm_runtime_is_python_array(x)) {
      return(x)
    }
    if (!ndm_sharding_enabled) {
      return(tryCatch(send2device(x), error = function(...) x))
    }
    sharding <- ndm_runtime_named_sharding(
      x,
      sharded = ndm_runtime_array_rank(x) > 0L
    )
    if (is.null(sharding)) {
      return(x)
    }
    tryCatch(jax$device_put(x, sharding), error = function(...) x)
  }

  ndm_runtime_replicate_tree <- function(x) {
    if (is.list(x)) {
      return(lapply(x, ndm_runtime_replicate_tree))
    }
    if (!ndm_runtime_is_python_array(x)) {
      return(x)
    }
    if (!ndm_sharding_enabled) {
      return(tryCatch(send2device(x), error = function(...) x))
    }
    sharded_tree <- ndm_runtime_shard_tree(x, sharded = FALSE)
    if (!is.null(sharded_tree)) {
      return(sharded_tree)
    }
    sharding <- ndm_runtime_named_sharding(x, sharded = FALSE)
    if (is.null(sharding)) {
      return(x)
    }
    tryCatch(jax$device_put(x, sharding), error = function(...) x)
  }

  ndm_runtime_host_integer <- function(x) {
    if (is.null(x)) {
      return(NULL)
    }
    if ("python.builtin.object" %in% class(x)) {
      value <- try(reticulate::py_to_r(x), silent = TRUE)
      if (!inherits(value, "try-error") && !is.null(value)) {
        return(as.integer(value)[[1L]])
      }
      value <- try(np$array(x), silent = TRUE)
      if (!inherits(value, "try-error") && !is.null(value)) {
        return(as.integer(value)[[1L]])
      }
    }
    as.integer(x)[[1L]]
  }

  ndm_runtime_normalize_getpred_saveat_info <- function(info) {
    if (!is.list(info) || length(info) == 0L) {
      return(info)
    }
    info[[1L]] <- ndm_runtime_host_integer(info[[1L]])
    info
  }

  ndm_runtime_sync_value <- function(x) {
    if (is.list(x)) {
      invisible(lapply(x, ndm_runtime_sync_value))
      return(invisible(x))
    }
    if (!("python.builtin.object" %in% class(x))) {
      return(invisible(x))
    }
    blocker <- try(x$block_until_ready, silent = TRUE)
    if (!inherits(blocker, "try-error") && !is.null(blocker)) {
      try(blocker(), silent = TRUE)
    }
    invisible(x)
  }

  ndm_runtime_benchmark <- function(fn, ..., warmup = 1L, runs = 5L) {
    args <- list(...)
    if (warmup > 0L) {
      for (warm_i in seq_len(warmup)) {
        ndm_runtime_sync_value(do.call(fn, args))
      }
    }
    elapsed <- numeric(runs)
    for (run_i in seq_len(runs)) {
      started <- proc.time()[["elapsed"]]
      ndm_runtime_sync_value(do.call(fn, args))
      elapsed[[run_i]] <- proc.time()[["elapsed"]] - started
    }
    list(
      seconds = elapsed,
      mean_seconds = mean(elapsed),
      runs = as.integer(runs),
      warmup = as.integer(warmup)
    )
  }

  ndm_shardings <- list(
    enabled = ndm_sharding_enabled,
    mesh = ndm_mesh,
    data = function(x) ndm_runtime_named_sharding(x, sharded = TRUE),
    replicate = function(x) ndm_runtime_named_sharding(x, sharded = FALSE)
  )
  ndm_benchmark_helpers <- list(
    sync_value = ndm_runtime_sync_value,
    benchmark = ndm_runtime_benchmark
  )

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
