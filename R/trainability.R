# This role mask is shared by counting, differentiation and the optimizer.
# Zeros do not imply frozen parameters: attention pseudo-queries are trainable.
.ndm_trainability_mask <- function(model, env) {
  map <- env$jax$tree_util$tree_map
  mask <- map(env$eq$is_inexact_array, model)
  freeze <- function(x) map(function(leaf) FALSE, x)
  disable <- function(path) {
    tree <- mask
    descend <- function(node, depth) {
      if (!is.list(node)) return(node)
      name <- path[[depth]]
      if (is.null(node[[name]])) return(node)
      node[[name]] <- if (depth == length(path)) freeze(node[[name]]) else descend(node[[name]], depth + 1L)
      node
    }
    mask <<- descend(tree, 1L)
  }
  disable(c("InitProcessList", "TimeEmbeds"))
  if (!isTRUE(env$AppendTimeEmbeds)) disable(c("InitProcessList", "TimeEmbeds_Proj"))
  if (isTRUE(model$InitProcessList$PlaceEmbedsFixed) || !isTRUE(env$AppendPlaceEmbeds)) {
    disable(c("InitProcessList", "PlaceEmbeds"))
  }
  if (!isTRUE(env$AppendPlaceEmbeds)) disable(c("InitProcessList", "PlaceEmbeds_Proj"))
  disable(c("TSList", "OutputProcess", "Proj2"))
  disable("BNList")
  for (name in c("nODE_scale", "nODE_out_scale", "GlobalNorm")) disable(c("ScaleList", name))
  for (name in c("DirichletScale1", "DirichletScale2")) disable(c("ScaleList", "ScaleBayes", name))
  if (isTRUE(env$UseFullAttentionResiduals)) {
    for (layer in grep("^d[0-9]+$", names(model$TSList$TSBackbone), value = TRUE)) {
      for (name in c("ResidCon1", "ResidCon2")) disable(c("TSList", "TSBackbone", layer, name))
    }
  }
  if (identical(env$ModelType, "DecoderOnly")) {
    for (name in c("FixedLocalTerms", "GlobalTerms", "GlobalNeural", "LocalNeural")) disable(name)
    disable(c("TSList", "InitialCLS"))
    disable(c("TSList", "OutputProcess", "ManualBias"))
    for (name in setdiff(names(model$ScaleList$ScaleBayes), "DecoderObservationScale")) {
      disable(c("ScaleList", "ScaleBayes", name))
    }
    if (identical(env$training_objective, "scaled_mse")) disable(c("ScaleList", "ScaleBayes", "DecoderObservationScale"))
  } else {
    disable(c("TSList", "TSBackbone", "DecoderProj"))
    disable(c("ScaleList", "ScaleBayes", "DecoderObservationScale"))
    if (!length(env$uq_globalneural_vec)) disable("GlobalNeural")
    disable(c("GlobalNeural", "InitConScaler"))
    if (is.null(env$nFixedLocal) || env$nFixedLocal == 0L) disable("FixedLocalTerms")
    if (is.null(env$nGlobalParams) || env$nGlobalParams == 0L) {
      disable("GlobalTerms")
      for (name in c("globalLmatDiagScaler", "globalLmatGlobalScaler")) disable(c("ScaleList", "ScaleBayes", name))
    }
    if (!is.null(env$nGlobalParams) && env$nGlobalParams <= 1L) {
      for (name in c("globalLmatDiagScaler", "globalLmatGlobalScaler")) disable(c("ScaleList", "ScaleBayes", name))
    }
    if (!is.null(env$nODEParams_base) && env$nODEParams_base == 0L) {
      for (name in c("LmatLocalDiagScaler", "LmatGlobalScaler")) disable(c("ScaleList", "ScaleBayes", name))
    }
    if (isTRUE(env$UseDiagonalLMatVCov)) disable(c("ScaleList", "ScaleBayes", "LmatGlobalScaler"))
    for (block in c("LocalNeural", "GlobalNeural")) {
      if (identical(env$ODE_NORM_TYPE, "none")) disable(c(block, "LNScaler"))
      if (!identical(env$ODE_NORM_TYPE, "pre-activation1")) disable(c(block, "MLP", "WideProj2"))
    }
  }
  mask
}

.ndm_parameter_counts <- function(model, mask, eq, jax) {
  count <- function(tree) sum(vapply(jax$tree_util$tree_leaves(tree), function(x) as.numeric(x$size), numeric(1L)))
  total <- count(eq$filter(model, eq$is_array))
  trainable <- count(eq$filter(model, mask))
  list(total = total, trainable = trainable, nontrainable = total - trainable)
}
