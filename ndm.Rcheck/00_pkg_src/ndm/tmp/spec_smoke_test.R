library(ndm)

spec <- ndm_model_spec(preset = "seirs_dynamic_beta_dynamic_global")
cat("Preset:", spec$preset, "\n")
cat("Model type:", spec$model_type, "\n")
cat("Source path:", spec$source_path, "\n")
cat("First 120 chars:\n")
cat(substr(ndm_model_spec_to_tex(spec), 1, 120), "\n")
