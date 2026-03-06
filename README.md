# ndm

`ndm` is an R package for neural disease modeling with a `reticulate`-managed
JAX backend. It wraps the active Phase 1 runtime in an R-first interface,
ships built-in epidemic model specifications, and vendors the legacy analysis
code inside the package so users do not need a separate external checkout to
run it.

This repository is most useful for researchers who want to work from R while
preserving the existing JAX, TFRecord, and `.tex` epidemic-specification
workflow.

## Why use `ndm`?

- R-native wrappers for provisioning and initializing a JAX backend
- Built-in SEIR/SEIRS model presets plus import/export for custom `.tex`
  specifications
- TFRecord readers and batch helpers for existing training and inference data
- Vendored runtime code under `inst/extracted/Analysis`, so the package is
  self-contained at execution time
- Both low-level building blocks and higher-level orchestration helpers for
  fitting legacy models from R

## Current scope

Phase 1 is intentionally conservative. The package currently:

- supports `DecoderOnly` and `NeuralODE`
- defaults to `DecoderOnly`
- supports transformer backbones only
- preserves the established `.tex` epidemic specification workflow
- consumes existing TFRecords rather than regenerating them
- vendors the active runtime bundle into the package

The package does not currently aim to be a fully turnkey, raw-data-to-results
epidemic modeling framework. If you need Mamba, latent attention, or a fully
new end-to-end data pipeline, that is outside the current scope.

## Package surface

| Area | Main functions | What they do |
| --- | --- | --- |
| Backend | `ndm_build_backend()`, `ndm_check_backend()`, `ndm_initialize_backend()` | Provision and verify the Python/JAX runtime used through `reticulate` |
| Model specs | `ndm_model_spec_presets()`, `ndm_model_spec()`, `ndm_model_spec_from_tex()`, `ndm_model_spec_to_tex()` | Work with bundled or custom epidemic model definitions |
| TFRecords | `ndm_tfrecord_fields_real()`, `ndm_tfrecord_fields_sim()`, `ndm_load_tfrecord_bundle()`, `ndm_attach_tfrecord_bundle()` | Inspect TFRecord contracts and load existing datasets |
| Runtime | `ndm_runtime_paths()`, `ndm_prepare_runtime()`, `ndm_prepare_data()` | Load the vendored legacy runtime into an isolated R environment |
| Modeling | `ndm_build_model()`, `ndm_train()`, `ndm_predict()`, `ndm_loss()`, `ndm_fit()` | Build, fit, and evaluate models once the runtime has been prepared |

## Installation

`ndm` is not on CRAN. Install it from a checkout of this repository:

```r
install.packages(c("remotes", "reticulate"))
remotes::install_local(".")
```

The package depends on:

- R 4.1 or newer
- a conda installation that `reticulate` can find
- Python packages for JAX-based modeling
- TensorFlow if you want to read TFRecords through the package helpers

If `conda` is not already available, `reticulate::install_miniconda()` is the
easiest setup path for most R users.

## Backend setup

After installing the package, build and initialize the Python backend:

```r
library(ndm)

# If needed:
# reticulate::install_miniconda()

ndm_build_backend(conda_env = "ndm_software_env")
ndm_check_backend(conda_env = "ndm_software_env")

backend <- ndm_initialize_backend(
  conda_env = "ndm_software_env",
  float_type = "32"
)

backend$default_backend
```

`ndm_build_backend()` tries to choose a sensible JAX install strategy:

- Apple Silicon: installs Metal-enabled JAX
- Linux with supported NVIDIA drivers: attempts CUDA wheels
- otherwise: falls back to CPU-only JAX

## Typical workflow

Most users will interact with `ndm` in four steps:

### 1. Create a runtime configuration

```r
cfg <- ndm_create_config(
  model_type = "DecoderOnly",
  backbone = "transformer",
  float_type = "32",
  force_to_gpu = FALSE
)

print(cfg)
```

### 2. Choose a built-in or custom epidemic specification

```r
ndm_model_spec_presets()

spec <- ndm_model_spec(preset = "seirs_dynamic_beta")

custom_spec <- ndm_model_spec_from_tex("path/to/model_spec.tex")
ndm_model_spec_to_tex(spec, "spec_copy.tex")
```

### 3. Inspect or load TFRecord data

Use the schema helpers to confirm the expected fields before reading existing
datasets:

```r
ndm_tfrecord_fields_real()
ndm_tfrecord_fields_sim()
ndm_tfrecord_dtype_map("real")
```

To load a training/inference bundle from existing TFRecords:

```r
bundle <- ndm_load_tfrecord_bundle(
  train_file = "path/to/train.tfrecord",
  inference_file = "path/to/inference.tfrecord",
  batch_size = 32L,
  kind = "real"
)
```

### 4. Prepare the vendored runtime and fit a model

`ndm` exposes both low-level and high-level wrappers around the vendored Phase 1
runtime:

```r
runtime_env <- ndm_prepare_runtime(config = cfg)
ndm_attach_tfrecord_bundle(runtime_env, bundle)
```

From there, you can either:

- use `ndm_fit()` if your workflow already matches the legacy runtime
  conventions, or
- use the lower-level helpers `ndm_prepare_data()`, `ndm_build_model()`, and
  `ndm_train()` after supplying the runtime globals expected by the extracted
  code

The important constraint is that the packaged wrappers preserve the historical
runtime contract. For realistic end-to-end training, you still need the
workflow-specific globals, output locations, and TFRecord layout expected by
the vendored scripts.

## Built-in model presets

The package bundles five preset epidemic specifications under
`inst/extdata/model_specs/`:

| Preset | Summary |
| --- | --- |
| `seir_fixed` | Classical fixed-beta, fixed-global SEIR baseline |
| `seirs_dynamic_beta` | SEIRS with dynamic local transmission and fixed global rates |
| `seirs_dynamic_beta_multi_outcome` | Dynamic local transmission with a multi-outcome observation model |
| `seirs_dynamic_beta_dynamic_global` | Dynamic local transmission plus dynamic global rates |
| `seirs_dynamic_beta_dynamic_global_multi_outcome` | Dynamic local transmission, dynamic global rates, and multi-outcome observations |

These presets can be inspected with `ndm_model_spec_presets()` and converted to
or from `.tex` files as needed.

## Input expectations

Potential users should know up front what `ndm` expects:

- Existing TFRecords are the main packaged input path for training and
  inference
- `.tex` model definitions remain a supported interchange format
- The runtime is JAX-based but controlled from R through `reticulate`
- The package vendors the legacy analysis code, but it does not erase the
  original runtime's dependence on workflow-specific globals

In practice, `ndm` is a strong fit if you want an R-facing interface around the
historical modeling stack, not if you are looking for a brand-new general
purpose epidemiology framework.

## Examples in this repository

The quickest way to see real usage patterns is to read the tests:

- [`tests/testthat/test-config.R`](tests/testthat/test-config.R): configuration defaults
- [`tests/testthat/test-specs.R`](tests/testthat/test-specs.R): preset and `.tex` model specs
- [`tests/testthat/test-tfrecord.R`](tests/testthat/test-tfrecord.R): TFRecord schemas and data helpers
- [`tests/testthat/test-sim-fit.R`](tests/testthat/test-sim-fit.R): the most complete scripted simulation fit example in the repo

## Validation status

The current automated tests cover:

- configuration defaults
- built-in spec registration and import/export
- TFRecord field and dtype contracts
- vendored runtime path resolution and isolated runtime setup
- simulated fit smoke tests when the required Python dependencies are available

That means the package surface is already testable, but full parity with every
historical run still depends on providing the matching runtime configuration and
data assets.
