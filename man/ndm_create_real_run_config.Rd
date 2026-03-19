% Manual documentation for package-native run configs and runners.
\name{ndm_create_real_run_config}
\alias{ndm_create_real_run_config}
\alias{ndm_create_sim_run_config}
\alias{ndm_create_multidisease_run_config}
\alias{ndm_run_real}
\alias{ndm_run_sim}
\alias{ndm_run_multidisease}
\title{Package-native run configurations and workflow runners}
\usage{
ndm_create_real_run_config(
  project_root = getwd(),
  analysis_name = "RealApril15",
  grid = NULL,
  grid_file = file.path("Data", "RunGrids", "RealGrids", sprintf("RealGrid_\%s.csv",
    analysis_name)),
  outer = 3L,
  model_type = NULL,
  respect_grid_model_type = FALSE,
  resave_tfrecords = FALSE,
  tfrecord_dir = file.path("Data", "RunTFRecords", "RealTFRecords",
    analysis_name),
  raw_data_dir = file.path("Data", "MainData"),
  outcome_metric = "inc_death",
  data_subset = "high_income",
  dry_run = FALSE
)

ndm_create_sim_run_config(
  project_root = getwd(),
  analysis_name = "BigSimsLatest",
  grid = NULL,
  grid_file = file.path("Data", "RunGrids", "SimGrids", sprintf("SimGrid_\%s.csv",
    analysis_name)),
  outer = 1L,
  model_type = NULL,
  respect_grid_model_type = FALSE,
  resave_tfrecords = TRUE,
  tfrecord_dir = file.path("Data", "RunTFRecords", "SimTFRecords",
    analysis_name),
  dry_run = FALSE
)

ndm_create_multidisease_run_config(
  project_root = getwd(),
  analysis_name = "RealLatest",
  grid = NULL,
  grid_file = file.path("Data", "RunGrids", "RealGrids", sprintf("RealGrid_\%s.csv",
    analysis_name)),
  outer = 1L,
  model_type = NULL,
  respect_grid_model_type = FALSE,
  resave_tfrecords = FALSE,
  tfrecord_dir = file.path("Data", "RunTFRecords", "RealTFRecords",
    analysis_name),
  outcome_metric = "CountValue",
  data_subset = "all",
  disease_names = c("Covid", "Flu"),
  data_format = "IHME",
  dry_run = FALSE
)

ndm_run_real(config = ndm_create_real_run_config())

ndm_run_sim(config = ndm_create_sim_run_config())

ndm_run_multidisease(config = ndm_create_multidisease_run_config())
}
\arguments{
\item{project_root}{Working project directory used for grids, TFRecords, and outputs.}

\item{analysis_name}{Analysis label threaded through output paths.}

\item{grid}{Optional in-memory grid data frame. When supplied, \code{grid_file} must be \code{NULL}.}

\item{grid_file}{Optional CSV path for the run grid.}

\item{outer}{Integer vector of outer-iteration rows to execute.}

\item{model_type}{Optional model family override.}

\item{respect_grid_model_type}{Logical scalar indicating whether grid rows may override the requested model type.}

\item{resave_tfrecords}{Logical scalar controlling TFRecord regeneration.
Supported for real and simulation workflows. Multidisease workflows reject
\code{TRUE} because the legacy regeneration path has been retired.}

\item{tfrecord_dir}{Optional TFRecord output directory.}

\item{raw_data_dir}{Real-data input directory.}

\item{outcome_metric}{Outcome metric name for real-data or multidisease runs.}

\item{data_subset}{Optional real-data subset name.}

\item{disease_names}{Character vector of multidisease names.}

\item{data_format}{Multidisease data format.}

\item{dry_run}{Logical scalar indicating whether the run should stop after resolving paths and grid rows.}

\item{config}{A package-native run configuration created by the matching \code{ndm_create_*_run_config()} helper.}
}
\value{
The \code{ndm_create_*_run_config()} helpers return classed lists describing
the requested workflow. The \code{ndm_run_*()} functions return the underlying
workflow result, or a dry-run preview when \code{config$dry_run} is \code{TRUE}.
}
\description{
Package-native configuration constructors and workflow runners for the
self-contained real, simulation, and multidisease execution paths.
}
\details{
\code{dry_run = TRUE} resolves paths and selected rows without executing the
underlying workflow. Dry runs can use lightweight in-memory grids such as the
\code{BaseID} / \code{ModelType} preview shown in \code{README.md}.

Non-dry execution expects Analysis2-compatible grids. The tested simulation
grids in this package are produced by \code{ndmdatasets::ndm_sim_build_grid()}
and then augmented with fields such as \code{paddingMethod},
\code{lookahead}, \code{n_time_steps}, \code{n_inference_batches},
\code{scaling_batches}, and either \code{model_spec_name} or
\code{model_tex_loc}. The tested real and multidisease grids include
\code{BaseID}, \code{ContextLength}, \code{evaluationTime},
\code{initialTransform}, \code{initialNormType}, \code{paddingMethod},
\code{OSSType}, \code{dataInputs}, \code{ModelType}, \code{ModelDepth},
\code{ModelDims}, \code{nSamplesTrain}, \code{nObsInference},
\code{floatType}, and either \code{model_spec_name} or \code{model_tex_loc}.

Non-dry runner workflows also require the full execution stack: the
\code{ndmdatasets} package, the helper packages used by the runtime, and a
backend environment aligned with \code{NDM_SOFTWARE_CONDA_ENV} (or the
platform-specific fallback used by the runner layer).
}
\examples{
\dontrun{
preview_grid <- data.frame(
  BaseID = c(1L, 2L),
  ModelType = c("DecoderOnly", "NeuralODE"),
  stringsAsFactors = FALSE
)

cfg <- ndm_create_sim_run_config(
  project_root = tempdir(),
  grid = preview_grid,
  outer = 1:2,
  dry_run = TRUE
)

ndm_run_sim(cfg)
}
}
