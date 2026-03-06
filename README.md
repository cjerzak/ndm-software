# ndm

`ndm` is an R-first package for neural disease modeling with a
reticulate-managed JAX backend.

Phase 1 in this repository is intentionally conservative:

- preserve both `DecoderOnly` and `NeuralODE` model types
- default to `DecoderOnly`
- support transformer backbones only
- exclude latent attention and Mamba
- preserve the established `.tex` epidemic specification workflow through import and
  export helpers
- consume existing TFRecords rather than regenerating them
- vendor the active runtime code into the package so `ndm` is
  self-contained at execution time

## Current package surface

- `ndm_create_config()`: build explicit runtime configuration objects
- `ndm_build_backend()` / `ndm_initialize_backend()`: provision and initialize
  the Python JAX environment
- `ndm_model_spec_*()`: work with built-in and custom epidemic model specs
- `ndm_read_tfrecord_dataset()` / `ndm_load_tfrecord_bundle()`: read existing
  TFRecords for training and inference
- `ndm_prepare_runtime()` / `ndm_prepare_data()`: load the vendored runtime into
  an isolated environment
- `ndm_build_model()` / `ndm_train()` / `ndm_predict()` / `ndm_loss()`: wrappers
  over the vendored modeling pipeline
- `ndm_fit()`: end-to-end orchestration wrapper around runtime, data, build, and
  train steps

## Built-in model presets

The package ships these built-in epidemic specs under
`inst/extdata/model_specs/`:

- `seir_fixed`
- `seirs_dynamic_beta`
- `seirs_dynamic_beta_multi_outcome`
- `seirs_dynamic_beta_dynamic_global`
- `seirs_dynamic_beta_dynamic_global_multi_outcome`

The structured `ndm_model_spec` object is the canonical internal
representation. `.tex` remains a supported import/export format for
compatibility with the established workflow.

## Testing status

The repository currently includes package tests for:

- configuration defaults
- built-in spec registration and import/export
- TFRecord field and dtype contracts
- packaged runtime path resolution and isolated environment setup

Phase 1 parity tests against full historical runs still need real run-specific
configurations and TFRecord locations.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███╗   ██╗██████╗ ███╗   ███╗      n d m                                 ║
║   ████╗  ██║██╔══██╗████╗ ████║   Neural Disease Modeling                  ║
║   ██╔██╗ ██║██║  ██║██╔████╔██║                                            ║
║   ██║╚██╗██║██║  ██║██║╚██╔╝██║      R • reticulate • JAX Backend          ║
║   ██║ ╚████║██████╔╝██║ ╚═╝ ██║                                            ║
║   ╚═╝  ╚═══╝╚═════╝ ╚═╝     ╚═╝                                            ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║     Transformer Backbone               Epidemic Compartments                 ║
║   ┌─────────────────────┐            ┌────────────────────────────┐         ║
║   │   ●───▶●───▶●       │            │   S ──▶ E ──▶ I ──▶ R      │         ║
║   │  DecoderOnly        │            │ Dynamic β • Multi-outcome  │         ║
║   │     NeuralODE       │            │   Global Dynamics          │         ║
║   └──────────▲──────────┘            └─────────────▲──────────────┘         ║
║              │                                    │                        ║
║              └────────────── ndm_fit() ───────────┘                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

     Built-in: seir_fixed • seirs_dynamic_beta • seirs_dynamic_beta_multi_outcome
              • seirs_dynamic_beta_dynamic_global • TFRecords • .tex support
              • ndm_create_config() • ndm_build_model() •
			  ndm_predict()
```
