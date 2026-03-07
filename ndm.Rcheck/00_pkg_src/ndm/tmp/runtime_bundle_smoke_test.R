if (requireNamespace("pkgload", quietly = TRUE) && file.exists("DESCRIPTION")) {
  pkgload::load_all(".", quiet = TRUE, export_all = FALSE)
} else {
  library(ndm)
}

analysis_root <- normalizePath(
  file.path("tests", "fixtures", "analysis_runtime", "Analysis"),
  winslash = "/",
  mustWork = TRUE
)
paths <- ndm_runtime_paths(analysis_root)
print(paths[c("analysis_root", "helper_fxns", "build_model")])

env <- ndm_new_runtime_env()
ndm_set_runtime_globals(env, list(ModelType = "DecoderOnly", BackboneType = "transformer"))
cat("Runtime globals set:", paste(ls(env), collapse = ", "), "\n")
