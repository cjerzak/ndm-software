# ndm

`ndm` is an R package for neural disease modeling with an R-first interface to
a `reticulate`-managed JAX runtime. The repository now ships its own bundled
analysis runtime, so model build/train/predict flows no longer depend on a
separate `Analysis` or `Analysis2` checkout.

It provides:

- backend setup helpers for the Python/JAX environment
- built-in compartment-model specifications
- TFRecord utilities
- self-contained runtime/model helpers
- package-native real, sim, and multidisease run APIs

## Installation

```r
install.packages(c("remotes", "reticulate"))
remotes::install_local("~/Documents/ndm-datasets")
remotes::install_local("~/Documents/ndm-software")
```

If you need the Python/JAX backend for execution:

```r
library(ndm)

ndm_build_backend()
ndm_check_backend()
```

## Model Workflow

The default configuration now uses the bundled internal runtime:

```r
library(ndm)

cfg <- ndm_create_config(
  model_type = "DecoderOnly",
  float_type = "32",
  force_to_gpu = FALSE
)

trained <- ndm_fit(
  config = cfg,
  model_spec = ndm_model_spec(
    preset = "seirs_dynamic_beta",
    model_type = cfg$model_type
  ),
  data_generator = "sim"
)
```

If you need explicit control, the same path is available via
`ndm_prepare_runtime()`, `ndm_prepare_data()`, `ndm_build_model()`, and
`ndm_train()`.

## Package-Native Run APIs

The maintained orchestration surface is now package-only:

```r
library(ndm)

sim_cfg <- ndm_create_sim_run_config(
  project_root = getwd(),
  grid_file = "Data/RunGrids/SimGrids/SimGrid_BigSimsLatest.csv",
  outer = 1L,
  dry_run = TRUE
)

preview <- ndm_run_sim(sim_cfg)
preview$grid_preview
```

Equivalent helpers exist for real-data and multidisease workflows:

- `ndm_create_real_run_config()` + `ndm_run_real()`
- `ndm_create_sim_run_config()` + `ndm_run_sim()`
- `ndm_create_multidisease_run_config()` + `ndm_run_multidisease()`
