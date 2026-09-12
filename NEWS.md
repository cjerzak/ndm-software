# ndm 0.5.0

- Training defaults now favour step time over activation memory.
  `transformer_activation_checkpointing` defaults to `FALSE`: measured on CPU it
  cost +27% per DecoderOnly step at `ModelDims` 128 / depth 4 and +6% at 256 /
  depth 8, it changes nothing for NeuralODE steps, and at those sizes the
  activations it saves are not the binding constraint. Set it to `TRUE` when
  activation memory binds. The NeuralODE optimisation solve now starts at
  `neuralode_optim_dt0 = 0.1` rather than `1e-3`, since the controller otherwise
  spent its first steps ramping up (about 26% of the solve), and it stops at the
  last saved time instead of integrating and differentiating one further unit
  that is never read. `MaxSteps` is now 4096 rather than `10^6` for real and
  multidisease runs; the largest solve observed was 241 steps, and a solve that
  exceeds the cap still reports a solver failure rather than truncating silently.
- NeuralODE training for structures without global-neural terms now
  integrates with a fixed-step RK4 scan instead of the adaptive diffrax solve.
  The adaptive solve's backward pass cost about seven times its forward pass
  (checkpointed while-loop machinery), while a plain `lax.scan` differentiated
  with ordinary reverse-mode AD has the usual ratio and a cheaper forward.
  `ndm_create_config()` exposes `neuralode_train_integrator` (`"auto"`,
  `"fixed_rk4"`, `"diffrax"`) and `neuralode_train_substeps`, the number of
  RK4 steps per unit time (`NULL` chooses automatically). Prediction and
  analytics always use the adaptive solve, and the integrator choice leaves
  inference output bit-identical. At identical weights on a fixed batch, the
  fixed-step training solve matches the adaptive one to 4e-3 on state
  trajectories, 1e-5 on the loss and 8e-3 on gradients for the fixed-global
  SEIRS structure with one substep; measured back to back on CPU at
  `ModelDims` 128 / depth 4, its training step is 2.5x faster.
  Structures with dynamic global rates keep the adaptive solve under `"auto"`.
  Their dynamics are stiff at initialisation: the adaptive solver needs up to
  about 120 local steps on some examples, and a fixed RK4 step at 1, 1/2 or
  1/4 unit time drives the compartments of those examples to overflow (20, 22
  and 26 of 32 examples survive) while the global path itself integrates
  exactly. Forcing `"fixed_rk4"` on such a structure is allowed but fails the
  solver-success check on stiff examples.
- The transformer no longer computes final-layer outputs for tokens it
  discards. The DecoderOnly prefill and the NeuralODE encoder each keep one
  token, but the last layer ran its output projection, feed-forward block and
  output residual aggregation over the whole sequence. `transformer_run()` now
  takes a `select_position`, applied after the last layer's attention and its
  K/V cache writes (the cache still holds every position), so those final
  stages run on one token. Every remaining stage is position-wise, and the
  selected output is bit-identical to the full run in both residual modes and
  both compute precisions; `RunTransformerBackbone()` keeps its full-sequence
  contract. Measured back to back on CPU at `ModelDims` 128 / depth 4, the
  training step is about 10% faster for DecoderOnly and 8% for NeuralODE.
- Full attention residual aggregation is cheaper without changing its result.
  Each residual source now records its inverse RMS once, when it is created,
  and every later aggregation reuses it: the logits are one matrix-vector
  product per source against the scaled pseudo-query, so the normalised
  `[sources, T, D]` key stack is never materialised, and the weighted sum is
  accumulated source by source instead of through a second stack. Per forward
  pass this computes 2L+1 normalisations rather than 2L^2+3L+1. Outputs and
  gradients match the previous implementation to 2e-7 and 1e-6 relative in
  float32 (bit-identical forward in BF16). Measured back to back on CPU at
  `ModelDims` 256 / depth 8, the DecoderOnly step is about 10% faster; the
  remaining cost of the feature is the aggregation itself, not its layout.
- The training loop spends less time on the host between compiled steps.
  Explicit garbage collection (`gc()` plus a full Python `gc.collect()`, about
  200 ms per call) now runs at checkpoints and every `TrainingGcInterval`
  iterations (default 200) instead of every ten; the per-step diagnostics that
  were fetched as 36 separate device-to-host transfers (loss, gradient norm,
  twelve loss components, nineteen solver-diagnostic fields, acceptance) are
  packed on the device and fetched in three; observation masks are validated
  inside the compiled step and folded into update acceptance, with the host-side
  copy and detailed error kept only for the failure path; and TFRecord datasets
  are attached with `prefetch(2)` so parsing overlaps device compute. Host-side
  column names and error text are unchanged, and custom step functions without
  the packed outputs fall back to the previous per-field path. The ~350-400 ms
  gc spikes every tenth iteration are gone from the telemetry.
- Compiled programs are now reused within a session and the learning-rate
  telemetry table is built in one vectorized call. `ndm_initialize_backend()`
  points JAX's persistent compilation cache at the session temporary directory,
  so grid rows after the first reuse compiled train steps instead of paying
  6-15 s to compile each one. Set `NDM_JAX_CACHE_DIR` to a durable path to reuse
  compilations across sessions, or to `""` to disable the cache.
- Fixed dynamic-global NeuralODE structures failing at the first training step
  with a `dot_general` contracting-dimension mismatch. The local neural field's
  first layer was built two inputs wider than the vector it receives, because
  `in_features` carried a hard-coded
  `+ 2 * grepl(model_tex_loc, "DynamicBeta_DynamicGlobal")` term. The width is
  now derived from `nDimODEOutput_ts_mean` plus the base and extra input counts,
  so it stays tied to the concatenation the vector field actually builds. The
  `seirs_dynamic_beta_dynamic_global` preset now trains and is covered by the
  NeuralODE structure smoke tests; the multi-outcome variant shares the same code
  path. Unchanged, and worth revisiting separately: `encneural_inputs` is parsed
  from the first rule mentioning `Neural`, which for these structures is a
  `Neural2` rule, so the four global rates enter the local field as extra inputs.
- Reduced transformer training memory with rematerialized sublayers and
  residual-source aggregation, native grouped-query attention, context-only
  decoder prefill, and parameter/optimizer buffer donation. Update rejection
  remains inside the compiled step and preserves state for diagnostics.
  New FP32 CUDA models default to BF16 transformer activations and K/V caches,
  retaining master, optimizer, prediction-head, and ODE precision. Configs expose
  precision/checkpointing/donation controls and artifacts preserve their
  resolved transformer precision across devices.
- Full attention residual transformers now aggregate all residual sources at
  the output, including in cached decoding. This adds `AttnResOutput`
  parameters; earlier full attention residual checkpoints must be rebuilt.
  Zero-initialized residual-attention queries are excluded from adaptive
  gradient clipping so their initial updates are not suppressed.
- Fixed convolution initialization when input channels exceed model width,
  masked non-finite targets contaminating MSE and disabled auxiliary losses,
  and cosine learning-rate decay ending one warmup period too early. Learning
  rate telemetry now uses the optimizer's zero-based update count.

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
