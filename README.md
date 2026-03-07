# ndm

`ndm` is an R package for neural disease modeling with an R-first interface to
a `reticulate`-managed JAX runtime. It provides:

- backend setup helpers for the Python/JAX environment
- built-in compartment-model specifications
- TFRecord utilities
- thin runtime wrappers around a caller-supplied local `Analysis` or `Analysis2`
  tree

## Installation

Install the package and its companion dataset package from local checkouts:

```r
install.packages(c("remotes", "reticulate"))
remotes::install_local("~/Documents/ndm-datasets")
remotes::install_local("~/Documents/ndm-software")
```

If you need the Python/JAX backend for a real analysis run, provision it once:

```r
library(ndm)

ndm_build_backend()
ndm_check_backend()
```

## Simple End-to-End Example

The example below creates a tiny temporary `Analysis2` runtime and runs the
full `ndm_fit()` workflow. In real use, replace the temporary runtime with your
project's `Analysis2` directory.

```r
library(ndm)

make_minimal_analysis2 <- function() {
  analysis_root <- file.path(tempdir(), "ndm-demo", "Analysis2")
  dirs <- file.path(
    analysis_root,
    c(
      "SetupEnv",
      "SetupData",
      "ModelDefiners",
      "ModelTrainers",
      "ResultsGet",
      "ResultsAnalyze",
      "ModelStructureTex"
    )
  )
  vapply(dirs, dir.create, logical(1), recursive = TRUE, showWarnings = FALSE)

  write_file <- function(path, lines = "invisible(NULL)") {
    writeLines(lines, file.path(analysis_root, path))
  }

  write_file("SetupEnv/SuperLModel_helperFxns.R", "helper_loaded <- TRUE")
  write_file("SetupEnv/SuperLModel_MasterImports.R", c(
    "backend_loaded <- TRUE",
    "backend_settings <- list(",
    "  floatType = get0('floatType', inherits = TRUE, ifnotfound = NULL),",
    "  force2GPU = get0('force2GPU', inherits = TRUE, ifnotfound = NULL)",
    ")"
  ))

  write_file("SetupData/SuperLModel_CalibrateML.R")
  write_file("SetupData/SuperLModel_DataGenerator_Real.R", "data_mode <- 'real'")
  write_file("SetupData/SuperLModel_DataGenerator_Sim.R", c(
    "ndm_source_extracted(file.path('SetupEnv', 'SuperLModel_helperFxns.R'))",
    "data_mode <- 'sim'"
  ))

  write_file("ModelDefiners/SuperLModel_BuildML.R", c(
    "ModelList <- list(source = NDM_INTERNAL_ANALYSIS_DIR)",
    "state <- list(stage = 'built')",
    "PriorList <- list()",
    "PolicyList <- list()",
    "GetPredSaveAtInfo_default <- list(1L)",
    "GetPred_inference <- function(ModelList, x, state, PriorList, PolicyList, GetPredSaveAtInfo, seed) list(list(kind = 'inference', batch = x), state)",
    "GetPred_train_jit <- function(ModelList, x, state, PriorList, PolicyList, GetPredSaveAtInfo, seed) list(list(kind = 'train', batch = x), state)",
    "getLoss_train <- function(ModelList, x, y, y_mask, i, state, PriorList, PolicyList, GetPredSaveAtInfo, seed) list(list(kind = 'loss', iteration = i), state)",
    "gradLoss_jax <- function(ModelList, x, y, y_mask, i, state, PriorList, PolicyList, GetPredSaveAtInfo, seed) list(list(list(0, state)), ModelList)"
  ))

  write_file(
    "ModelTrainers/SuperLModel_TrainDefine.R",
    "opt_state <- list(stage = 'defined')"
  )
  write_file("ModelTrainers/SuperLModel_TrainDo.R", c(
    "state$stage <- 'trained'",
    "opt_state <- list(stage = 'trained')"
  ))

  write_file("ResultsGet/SuperLModel_GetAnalytics.R")
  write_file("ResultsAnalyze/SuperLModel_GenFigs.R")

  normalizePath(analysis_root, winslash = "/", mustWork = TRUE)
}

analysis_root <- make_minimal_analysis2()

cfg <- ndm_create_config(
  model_type = "DecoderOnly",
  analysis_root = analysis_root,
  float_type = "32",
  force_to_gpu = FALSE
)

trained <- ndm_fit(
  config = cfg,
  model_spec = ndm_model_spec(
    preset = "seirs_dynamic_beta",
    model_type = cfg$model_type
  ),
  data_generator = "sim"
)

trained$state$stage
#> [1] "trained"
```

`ndm_fit()` is the shortest path through the package. If you want more control,
the same flow can be run step by step with `ndm_prepare_runtime()`,
`ndm_prepare_data()`, `ndm_build_model()`, and `ndm_train()`.

## Typical Project Usage

When you already have a project checkout, point `analysis_root` at that local
runtime and keep the rest of the flow the same:

```r
library(ndm)

cfg <- ndm_create_config(
  model_type = "DecoderOnly",
  analysis_root = "/path/to/CovidSuperlearner/Analysis2",
  float_type = "32",
  force_to_gpu = FALSE
)

trained <- ndm_fit(
  config = cfg,
  model_spec = ndm_model_spec(
    preset = "seirs_dynamic_beta",
    model_type = cfg$model_type
  ),
  data_generator = "sim"
)
```
