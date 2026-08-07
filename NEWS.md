# ndm 0.4.0

- Model and run APIs now default to `compute_backend = "auto"`, selecting a
  supported JAX GPU when available and otherwise running on CPU. Explicit
  `"cpu"` and generic JAX `"gpu"` policies are also supported; macOS uses CPU.
- `force_to_gpu` remains as a deprecated compatibility alias. Backend objects
  now report requested and resolved policies plus selected-device provenance,
  and portable placement no longer assumes NVIDIA CUDA.

# ndm 0.3.0

- Multidisease WHO workflows can now attach a manifest-defined annual
  covariate panel with a closed, ordered numeric schema. Panel, manifest, and
  schema SHA-256 identities are embedded in the canonical dataset contract and
  verified again before training.
- Multidisease grids may define `inferenceSupportInputs` independently from
  `dataInputs`, allowing optional masked covariates without changing the
  trajectory-defined inference cohort.

# ndm 0.2.0

- Real and simulation model preparation now consumes validated canonical
  ndmdatasets TFRecord pairs and manifests. Create them with
  `ndm_bootstrap_real_tfrecords()` or `ndm_bootstrap_sim_tfrecords()` before
  training; use `SkipTfRecords = TRUE` only for the in-memory compatibility
  path.
- Multidisease workflows now use a serial
  `ndm_bootstrap_multidisease_tfrecords()` preflight that shares the loader,
  schema-v3 dataset/source/seed/support contract, and read-only validation path
  used by training. Training binds the expected producer through
  `NDM_TFRECORD_PRODUCER_CONTRACT`.
- Canonical simulation targets retain their policy channel for diagnostics, but
  training loss is restricted to the model's declared outcome channels.
- Training-stage and dataset-iterator errors now preserve their original
  condition and traceback instead of being replaced by generic loader or
  malformed-batch messages.
