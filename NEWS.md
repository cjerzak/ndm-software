# ndm 0.5.0

- Trajectory videos after every run. `ndm_run_real()`, `ndm_run_sim()` and
  `ndm_run_multidisease()` (and the Analysis2 command-line runners, which
  reach the same entry point) now call `ndmviz::ndm_viz_after_run()` when the
  `ndmviz` package (https://github.com/cjerzak/ndm-viz) is installed, so each
  fit's `trajectories/rolling.mp4` (and `training.mp4` when it kept several
  checkpoints) appears in its result folder. Skipped for dry runs and when
  `NDM_VIZ_AUTO=FALSE`; never fails a run; writes nothing an analysis reads.
- Simulation analytics keep the evaluation batch's trajectories: every
  checkpoint writes `trajectories_sim<af>_i<i>.csv` (past and future truth,
  posterior-mean forecast, its scale, persistence) next to `res<af>_i<i>.csv`,
  so simulation videos show trajectories rather than only skill curves. The
  write is guarded and cannot affect the fit.

- `enable_kv_cache_training` now defaults to `TRUE` across the config, run,
  Analysis2-spec, and runtime layers: DecoderOnly training uses the KV-cached
  rollout by default. The cached training path was verified gradient-equivalent
  to the un-cached reference rollout (forward and gradient parity at float32
  reordering tolerance across production topology, left padding, interior mask
  holes, and both residual modes) and is several times faster per training
  step at production batch sizes (about 7x for batch 32, horizon 12, on CPU).
  Set `enable_kv_cache_training = FALSE` to reproduce historical un-cached
  training exactly. The strict rule that `enable_kv_cache_training = TRUE`
  requires `enable_kv_cache = TRUE` is unchanged, so cache-off callers must
  now pass `enable_kv_cache_training = FALSE` explicitly.
- Added a regression test asserting cached-vs-uncached training loss
  trajectory and post-training prediction parity.

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
