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

`ndm` imports `ndmdatasets`, so install that package before installing
`ndm`:

```r
install.packages(c("remotes", "reticulate"))
remotes::install_local("/path/to/ndm-datasets")
remotes::install_local("/path/to/ndm-software")
```

If you need the Python/JAX backend for model execution:

```r
library(ndm)

ndm_build_backend()
ndm_check_backend()
```

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

The maintained orchestration surface is package-native. You can preview real,
simulation, and multidisease runs with an in-memory grid and `dry_run = TRUE`:

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

For full execution rather than dry runs, the package also exposes
`ndm_prepare_runtime()`, `ndm_prepare_data()`, `ndm_build_model()`,
`ndm_train()`, `ndm_predict()`, and `ndm_fit()`. Those workflows need backend
setup plus runtime globals and data inputs beyond the lightweight tutorial
above.

Trained low-level models can also be persisted as versioned artifacts and
restored later:

```r
artifact_dir <- ndm_save_model(trained_model, "artifacts/run-001")
restored_model <- ndm_load_model(artifact_dir)
resumed_model <- ndm_resume_training(artifact_dir, bundle = tf_bundle)
```

Equivalent run helpers exist for each orchestration mode:

- `ndm_create_real_run_config()` + `ndm_run_real()`
- `ndm_create_sim_run_config()` + `ndm_run_sim()`
- `ndm_create_multidisease_run_config()` + `ndm_run_multidisease()`
