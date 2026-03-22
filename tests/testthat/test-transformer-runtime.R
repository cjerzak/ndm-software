# Shared transformer regression tests that are narrower than the heavier
# end-to-end sim-fit suite.

ndm_test_transformer_source_root <- function() {
  source_root <- try(ndm:::.ndm_runtime_source_root(), silent = TRUE)
  if (inherits(source_root, "try-error") || is.null(source_root) || !dir.exists(source_root)) {
    start_paths <- unique(c(
      getwd(),
      tryCatch(testthat::test_path("..", ".."), error = function(...) ""),
      system.file(package = "ndm")
    ))
    candidate_roots <- unique(unlist(lapply(start_paths[nzchar(start_paths)], function(path) {
      path <- normalizePath(path, winslash = "/", mustWork = FALSE)
      parents <- character(7L)
      parents[[1L]] <- path
      for (i in 1:6) {
        parents[[i + 1L]] <- normalizePath(
          file.path(parents[[i]], ".."),
          winslash = "/",
          mustWork = FALSE
        )
      }
      unique(parents[nzchar(parents)])
    })))

    for (root in candidate_roots) {
      candidate <- file.path(root, "tools", "runtime_source", "ndm_runtime")
      if (dir.exists(candidate)) {
        source_root <- candidate
        break
      }
    }
  }
  if (inherits(source_root, "try-error") || is.null(source_root) || !dir.exists(source_root)) {
    skip("runtime source tree is not available in this installed-package context")
  }
  normalizePath(source_root, winslash = "/", mustWork = TRUE)
}

ndm_test_py_numeric <- function(x) {
  as.numeric(reticulate::py_to_r(x))
}

test_that("transformer runtime preserves residual scaling and backbone output contracts in jax_cpu", {
  ndm_skip_if_no_sim_backend()

  captured <- new.env(parent = emptyenv())
  details <- ndm_test_fit_sim_case(
    model_type = "DecoderOnly",
    endogeneity = 0.0,
    n_sgd = 1L,
    return_details = TRUE,
    before_train = function(runtime_env, model) {
      layer <- model$env$ModelList$TSList$TSBackbone$d1
      capture_pair <- function(pair) {
        list(
          skip = ndm_test_py_numeric(
            runtime_env$np$asanyarray(runtime_env$jax$nn$softplus(pair$WtSkipPath))
          ),
          resid = ndm_test_py_numeric(
            runtime_env$np$asanyarray(runtime_env$jax$nn$softplus(pair$WtResidPath))
          )
        )
      }

      captured$model_depth <- as.integer(runtime_env$ModelDepth)
      captured$num_heads <- as.integer(runtime_env$TransformerHeads)
      captured$num_kv_heads <- as.integer(runtime_env$TransformerKVHeads)
      captured$resid_con_1 <- capture_pair(layer$ResidCon1)
      captured$resid_con_2 <- capture_pair(layer$ResidCon2)
      captured$q_norm_shape <- as.integer(unlist(reticulate::py_to_r(layer$Multihead$QNormScale$shape)))
      captured$k_norm_shape <- as.integer(unlist(reticulate::py_to_r(layer$Multihead$KNormScale$shape)))
      captured$q_norm <- ndm_test_py_numeric(runtime_env$np$asanyarray(layer$Multihead$QNormScale))
      captured$k_norm <- ndm_test_py_numeric(runtime_env$np$asanyarray(layer$Multihead$KNormScale))
    }
  )

  expected_resid <- sqrt(1 / (1 + 0.01^2 * captured$model_depth))
  expected_skip <- sqrt(1 - expected_resid^2)

  for (pair in list(captured$resid_con_1, captured$resid_con_2)) {
    expect_true(all(is.finite(pair$skip)))
    expect_true(all(is.finite(pair$resid)))
    expect_gt(mean(pair$resid), mean(pair$skip))
    expect_lt(max(abs(pair$skip - expected_skip)), 1e-4)
    expect_lt(max(abs(pair$resid - expected_resid)), 1e-4)
  }

  expect_equal(captured$q_norm_shape, c(captured$num_heads, 1L))
  expect_equal(captured$k_norm_shape, c(captured$num_kv_heads, 1L))
  expect_equal(captured$q_norm, rep(1, captured$num_heads))
  expect_equal(captured$k_norm, rep(1, captured$num_kv_heads))

  hidden_state <- details$runtime_env$jnp$zeros(list(as.integer(details$runtime_env$ModelDims)))
  bridge_out <- details$runtime_env$DecoderBackboneToOutput(
    TSList = details$model$env$ModelList$TSList,
    hidden_state = hidden_state
  )
  bridge_shape <- as.integer(unlist(reticulate::py_to_r(bridge_out$shape)))
  proj_out <- details$model$env$ModelList$TSList$TSBackbone$DecoderProj(bridge_out)
  proj_shape <- as.integer(unlist(reticulate::py_to_r(proj_out$shape)))

  expect_equal(bridge_shape, as.integer(details$runtime_env$ModelDims))
  expect_equal(proj_shape, as.integer(details$runtime_env$nOutcomes))
})

test_that("maintained transformer sources keep cache and rotary guardrails", {
  source_root <- ndm_test_transformer_source_root()
  buildml_source <- paste(
    readLines(
      file.path(source_root, "ModelDefiners", "SuperLModel_BuildML.R"),
      warn = FALSE
    ),
    collapse = "\n"
  )
  backbone_source <- paste(
    readLines(
      file.path(source_root, "ModelDefiners", "SuperLModel_BackboneTransformer.R"),
      warn = FALSE
    ),
    collapse = "\n"
  )

  expect_match(buildml_source, "DecoderBackboneToOutput <- function\\(TSList, hidden_state\\)")
  expect_match(buildml_source, "jnp\\$expand_dims\\(xt_last, 0L\\)")
  expect_match(buildml_source, "jnp\\$expand_dims\\(embed_out, 0L\\)")
  expect_match(buildml_source, "DecoderProj\\(embed_out\\)")
  expect_match(backbone_source, "qk_normalize_heads <- function")
  expect_match(backbone_source, "QNormScale")
  expect_match(backbone_source, "KNormScale")
  expect_false(grepl("RotaryPositionalEmbedding", paste(buildml_source, backbone_source), fixed = TRUE))
})
