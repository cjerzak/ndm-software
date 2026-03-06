library(ndm)

paths <- ndm_runtime_paths()
print(paths[c("analysis_root", "helper_fxns", "build_model")])

env <- ndm_new_runtime_env()
ndm_set_runtime_globals(env, list(ModelType = "DecoderOnly", BackboneType = "transformer"))
cat("Runtime globals set:", paste(ls(env), collapse = ", "), "\n")
