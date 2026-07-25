ndm_test_eval_transformer_backbone_source <- function(envir) {
  source_path <- file.path(
    testthat::test_path("..", ".."),
    "tools",
    "runtime_source",
    "ndm_runtime",
    "ModelDefiners",
    "SuperLModel_BackboneTransformer.R"
  )
  if (file.exists(source_path)) {
    sys.source(
      normalizePath(source_path, winslash = "/", mustWork = TRUE),
      envir = envir
    )
    return(invisible(envir))
  }
  source_text <- ndm:::.ndm_embedded_runtime_source(
    "ModelDefiners/SuperLModel_BackboneTransformer.R"
  )
  eval(parse(text = source_text, keep.source = FALSE), envir = envir)
  invisible(envir)
}

ndm_test_jax_key_value <- function(key, np) {
  as.numeric(reticulate::py_to_r(np$asarray(key)$astype("uint64")))
}

test_that("transformer initialization assigns five distinct JAX split keys", {
  ndm_require_backend_test_stack(
    "transformer initialization key tests",
    modules = c("jax", "numpy"),
    packages = "reticulate"
  )

  jax <- reticulate::import("jax", convert = FALSE)
  np <- reticulate::import("numpy", convert = FALSE)
  source_env <- new.env(parent = baseenv())
  source_env$backbonePath <- "key-test"
  source_env$jax <- jax
  ndm_test_eval_transformer_backbone_source(source_env)

  base_key <- jax$random$PRNGKey(123L)
  actual <- source_env$ndm_transformer_initialization_keys(base_key)
  expected <- jax$random$split(base_key, 5L)
  actual_values <- do.call(
    rbind,
    lapply(actual, ndm_test_jax_key_value, np = np)
  )
  expected_values <- do.call(
    rbind,
    lapply(0L:4L, function(index) {
      ndm_test_jax_key_value(expected[index], np = np)
    })
  )

  expect_named(actual, c("W_q", "W_k", "W_v", "W_o", "next_layer"))
  expect_equal(unname(actual_values), unname(expected_values))
  expect_equal(nrow(unique(as.data.frame(actual_values))), 5L)

  initialized <- lapply(actual[c("W_q", "W_k", "W_v", "W_o")], function(key) {
    reticulate::py_to_r(
      np$asarray(jax$random$normal(key, shape = reticulate::tuple(16L, 16L)))
    )
  })
  for (pair in combn(names(initialized), 2L, simplify = FALSE)) {
    expect_false(
      isTRUE(all.equal(initialized[[pair[[1L]]]], initialized[[pair[[2L]]]])),
      label = paste(pair, collapse = " versus ")
    )
  }
})

test_that("same-shaped transformer projections initialize independently", {
  conda_env <- ndm_require_backend_test_stack(
    "transformer projection initialization tests",
    packages = "reticulate"
  )
  old_backend <- ndm:::ndm_env$backend
  on.exit(assign("backend", old_backend, envir = ndm:::ndm_env), add = TRUE)
  backend <- ndm_initialize_backend(
    conda_env = conda_env,
    float_type = "32",
    import_tensorflow = FALSE
  )

  source_env <- new.env(parent = baseenv())
  base_key <- backend$jax$random$PRNGKey(123L)
  list2env(
    list(
      backbonePath = "initialize",
      jax = backend$jax,
      jnp = backend$jnp,
      np = backend$np,
      eq = backend$eq,
      oryx = backend$oryx,
      jaxFloatType = backend$jaxFloatType,
      InvSoftPlus = backend$InvSoftPlus,
      ai = as.integer,
      ModelType = "DecoderOnly",
      ModelDims = 16L,
      ModelDepth = 2L,
      TransformerHeads = 1L,
      TransformerKVHeads = 1L,
      TransformerHeadDim = 16L,
      TransformerKVGroupSize = 1L,
      TransformerList = list(list(), list()),
      UseLatentAttention = FALSE,
      WideMultiplicationFactor = 2L,
      nOutcomes = 1L,
      key = base_key
    ),
    envir = source_env
  )
  ndm_test_eval_transformer_backbone_source(source_env)

  multihead <- source_env$TransformerList$d1$Multihead
  weights <- lapply(
    multihead[c("W_q", "W_k", "W_v", "W_o")],
    function(weight) reticulate::py_to_r(backend$np$asarray(weight))
  )
  expected_keys <- source_env$ndm_transformer_initialization_keys(base_key)
  init_std <- sqrt(2 / (16 + 16))
  initialize_expected <- function(seed_key) {
    reticulate::py_to_r(backend$np$asarray(
      backend$oryx$Normal(
        loc = 0,
        scale = backend$jnp$array(init_std)
      )$sample(list(16L, 16L), seed = seed_key)$astype(backend$jaxFloatType)
    ))
  }
  expected_weights <- lapply(
    expected_keys[c("W_q", "W_k", "W_v", "W_o")],
    initialize_expected
  )
  layer_two_keys <- source_env$ndm_transformer_initialization_keys(
    expected_keys$next_layer
  )
  layer_two_wq <- reticulate::py_to_r(
    backend$np$asarray(source_env$TransformerList$d2$Multihead$W_q)
  )

  expect_named(weights, c("W_q", "W_k", "W_v", "W_o"))
  expect_equal(
    unname(lapply(weights, dim)),
    rep(list(c(16L, 16L)), 4L)
  )
  expect_equal(weights, expected_weights, tolerance = 0)
  expect_equal(layer_two_wq, initialize_expected(layer_two_keys$W_q), tolerance = 0)
  expect_false(isTRUE(all.equal(weights$W_q, layer_two_wq)))
  for (pair in combn(names(weights), 2L, simplify = FALSE)) {
    expect_false(
      isTRUE(all.equal(
        weights[[pair[[1L]]]],
        weights[[pair[[2L]]]]
      )),
      label = paste(pair, collapse = " versus ")
    )
  }
})
