# ndm

`ndm` is an R package for neural disease modeling with an R-first interface to
a `reticulate`-managed JAX runtime. Model build, train, and predict flows are
package-native and no longer require callers to supply an executable
external runtime checkout.

It provides:

- backend setup helpers for the Python/JAX environment
- built-in compartment-model specifications
- TFRecord utilities
- package-managed runtime/model helpers
- package-native real, sim, and multidisease run APIs

## Installation

Install from GitHub with `remotes::install_github()`. `ndm` imports
`ndmdatasets`, so install that package before installing `ndm`. The helper
below uses `GITHUB_PAT` when it is already set; otherwise, if the terminal has
been authenticated with `gh auth login`, it asks the GitHub CLI for the active
token. Public repositories can install without a token, while private
repositories require the token/account to have access to both repositories.

```r
install.packages(c("remotes", "reticulate"))

github_token <- function() {
  token <- Sys.getenv("GITHUB_PAT")
  if (nzchar(token)) {
    return(token)
  }

  if (!nzchar(Sys.which("gh"))) {
    return(NULL)
  }

  token <- tryCatch(
    system2("gh", c("auth", "token"), stdout = TRUE, stderr = FALSE),
    warning = function(w) character(),
    error = function(e) character()
  )

  if (!length(token) || !nzchar(token[[1L]])) {
    return(NULL)
  }

  token <- token[[1L]]
  Sys.setenv(GITHUB_PAT = token)
  token
}

token <- github_token()

remotes::install_github(
  "cjerzak/ndm-datasets",
  auth_token = token,
  upgrade = "never"
)
remotes::install_github(
  "cjerzak/ndm-software",
  auth_token = token,
  upgrade = "never"
)
```

If you are developing from local checkouts instead, use the local paths:

```r
install.packages(c("remotes", "reticulate"))
remotes::install_local("/path/to/ndm-datasets")
remotes::install_local("/path/to/ndm-software")
```

For full execution workflows rather than dry-run previews, also install the
helper packages used by the runtime, training, and artifact paths:

```r
install.packages(c("fastmatch", "progress", "rrapply", "zip", "zoo"))
```

If you need the Python/JAX backend for model execution, TFRecord parsing,
artifacts, or `ndm_run_*()` workflows, point both backend helpers and runners
at the same conda environment:

```r
Sys.setenv(NDM_SOFTWARE_CONDA_ENV = "ndm_software_env")

library(ndm)

conda_env <- Sys.getenv("NDM_SOFTWARE_CONDA_ENV")

ndm_build_backend(conda_env = conda_env, include_tensorflow = TRUE)
ndm_check_backend(conda_env = conda_env)
ndm_initialize_backend(conda_env = conda_env, import_tensorflow = TRUE)
```

When `NDM_SOFTWARE_CONDA_ENV` is unset, the Analysis2-backed runner helpers
next consult `NDM_CONDA_ENV` and otherwise fall back to `jax_cpu` on macOS and
`ndm_software_env` elsewhere.

## Quick Tutorial

The code blocks below are part of the automated test suite. Blocks marked with
`# readme-test: tutorial` are executed directly from `README.md`, so this
tutorial is expected to run as written.

The first step is to create a model config and pick a built-in model
specification:

```r
# readme-test: tutorial
library(ndm)

cfg <- ndm_create_config(
  model_type = "DecoderOnly",
  float_type = "32",
  force_to_gpu = FALSE
)
specs <- ndm_model_spec_presets()
spec <- ndm_model_spec(
  preset = "seirs_dynamic_beta",
  model_type = cfg$model_type
)

stopifnot(
  inherits(cfg, "ndm_config"),
  nrow(specs) >= 1L,
  inherits(spec, "ndm_model_spec"),
  identical(spec$model_type, cfg$model_type)
)
```

For a maintainer-oriented guide to the ODE structure system, including the
built-in preset taxonomy, structured TB families, custom structured
declarations, and TeX round-tripping, see
`vignette("ode-structures", package = "ndm")`.

The maintained orchestration surface is package-native. You can preview real,
simulation, and multidisease runs with an in-memory grid and `dry_run = TRUE`.
This minimal grid is for preview-only and is intentionally lighter than the
grids required for executable non-dry runs:

```r
# readme-test: tutorial
grid <- data.frame(
  BaseID = c(1L, 2L),
  ModelType = c("DecoderOnly", "NeuralODE"),
  stringsAsFactors = FALSE
)

sim_preview <- ndm_run_sim(
  ndm_create_sim_run_config(
    project_root = tempdir(),
    grid = grid,
    outer = 1:2,
    dry_run = TRUE
  )
)

real_preview <- ndm_run_real(
  ndm_create_real_run_config(
    project_root = tempdir(),
    grid = grid,
    outer = 1:2,
    dry_run = TRUE
  )
)

multidisease_preview <- ndm_run_multidisease(
  ndm_create_multidisease_run_config(
    project_root = tempdir(),
    grid = grid,
    outer = 1:2,
    dry_run = TRUE
  )
)

stopifnot(
  identical(sim_preview$run_spec$mode, "sim"),
  identical(real_preview$run_spec$mode, "real"),
  identical(multidisease_preview$run_spec$mode, "multidisease"),
  identical(sim_preview$grid_rows, 2L),
  identical(real_preview$grid_rows, 2L),
  identical(multidisease_preview$grid_rows, 2L)
)

sim_preview$grid_preview
```

For non-dry `ndm_run_*()` workflows, use Analysis2-compatible grids rather than
the lightweight preview grid above:

- `sim`: start from `ndmdatasets::ndm_sim_build_grid()` and keep fields such as
  `BaseID`, `ContextLength`, `ModelType`, `ModelDepth`, `ModelDims`,
  `nSamplesTrain`, `floatType`, `paddingMethod`, `lookahead`, `n_time_steps`,
  `n_inference_batches`, `scaling_batches`, and either `model_spec_name` or
  `model_tex_loc`.
- `real` and `multidisease`: use grids that include `BaseID`, `ContextLength`,
  `evaluationTime`, `initialTransform`, `initialNormType`, `paddingMethod`,
  `OSSType`, `dataInputs`, `ModelType`, `ModelDepth`, `ModelDims`,
  `nSamplesTrain`, `nObsInference`, `floatType`, and either `model_spec_name`
  or `model_tex_loc`.

For full execution rather than dry runs, the package also exposes
`ndm_prepare_runtime()`, `ndm_prepare_data()`, `ndm_build_model()`,
`ndm_train()`, `ndm_predict()`, and `ndm_fit()`. Those workflows need backend
setup plus runtime globals and data inputs beyond the lightweight tutorial
above. `ndm_prepare_runtime()` requires `fastmatch`; simulation data helpers
also require `progress` and `zoo`; training and artifact restore paths use
`rrapply`, checkpointed training uses `zip`, and multidisease training also
uses `zoo`.

Transformer backbones now use Full Attention Residuals by default. Set the
runtime global `UseFullAttentionResiduals = FALSE` if you need the legacy
additive residual path for compatibility or comparison runs.

Trained low-level models can also be persisted as versioned artifacts and
restored later. These helpers assume that `ndm_initialize_backend()` has
already run for the active conda environment and that `trained_model` came from
`ndm_build_model()` plus `ndm_train()` or from `ndm_fit()`, so its runtime
environment still contains objects such as `ModelList` and `state`. Exact
resume also needs optimizer state; if you omit `bundle` in
`ndm_resume_training()`, the artifact metadata must already contain a recorded
TFRecord bundle reference:

```r
# After ndm_initialize_backend() and ndm_train() / ndm_fit():
artifact_dir <- ndm_save_model(trained_model, "artifacts/run-001")
restored_model <- ndm_load_model(artifact_dir, bundle = tf_bundle)
resumed_model <- ndm_resume_training(artifact_dir, bundle = tf_bundle)
```

Equivalent run helpers exist for each orchestration mode:

- `ndm_create_real_run_config()` + `ndm_run_real()`
- `ndm_create_sim_run_config()` + `ndm_run_sim()`
- `ndm_create_multidisease_run_config()` + `ndm_run_multidisease()`
