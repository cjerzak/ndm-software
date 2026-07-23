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
