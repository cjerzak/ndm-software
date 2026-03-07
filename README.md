# ndm

`ndm` is an R package for neural disease modeling with a `reticulate`-managed
JAX backend. It provides backend/bootstrap helpers, built-in epidemic model
specifications, TFRecord readers, and runtime/model wrappers that operate on a
caller-supplied local `Analysis` or `Analysis2` tree.

`ndm` no longer ships a runnable analysis checkout and no longer owns
`Analysis2` orchestration. The full analysis suite should stay in the project
repo; reusable modeling and data primitives live in `ndm` and `ndmdatasets`.

## Installation

Install both packages from local checkouts:

```r
install.packages(c("remotes", "reticulate"))
remotes::install_local("~/Documents/ndm-datasets")
remotes::install_local("~/Documents/ndm-software")
```

## Runtime usage

Point `ndm` at a local analysis root explicitly:

```r
library(ndm)

cfg <- ndm_create_config(
  model_type = "DecoderOnly",
  analysis_root = "/path/to/CovidSuperlearner/Analysis2",
  float_type = "32",
  force_to_gpu = FALSE
)

runtime_env <- ndm_prepare_runtime(cfg)
ndm_prepare_data(runtime_env, analysis_root = cfg$analysis_root, generator = "sim")
model <- ndm_build_model(runtime_env, analysis_root = cfg$analysis_root)
trained <- ndm_train(model, analysis_root = cfg$analysis_root)
```

The runtime helpers now honor `analysis_root` end-to-end, including
`ndm_source_runtime_calibration()`, `ndm_source_runtime_results_get()`, and
`ndm_source_runtime_results_analyze()`.

## Analysis2

The package runners are deprecated:

- `ndm_run_analysis2_real()`
- `ndm_run_analysis2_sim()`

Run the project-local entrypoints instead:

- `/path/to/CovidSuperlearner/Analysis2/SuperLModel_MasterReal.R`
- `/path/to/CovidSuperlearner/Analysis2/SuperLModel_MasterSim.R`

Those local scripts should use `ndm_create_config(..., analysis_root = <project>/Analysis2)`
and the runtime sourcing helpers from this package.
