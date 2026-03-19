\name{ndm-package}
\alias{ndm-package}
\alias{ndm}
\title{Neural Disease Modeling Software}
\description{
\code{ndm} is an R-first package for neural disease modeling with a
reticulate-managed JAX backend. The package currently provides backend setup,
epidemic model spec import/export, TFRecord readers, package-managed runtime
helpers, package-native runtime/model helpers, package-native run APIs, and
artifact helpers for saving and restoring trained models.
}
\details{
The current implementation preserves both historical model families,
\code{"DecoderOnly"} and \code{"NeuralODE"}, while defaulting to
\code{"DecoderOnly"}. The supported backbone is the transformer path only.
Latent attention and Mamba are intentionally excluded.

The package treats the structured \code{ndm_model_spec} object as the canonical
internal representation and supports \code{.tex} import/export for compatibility
with the original epidemic specification workflow.

For executable workflows rather than dry-run previews, install the helper
packages listed under \code{Suggests}, initialize the backend with
\code{\link{ndm_initialize_backend}}, and align that backend with
\code{NDM_SOFTWARE_CONDA_ENV} when using the package-native runner helpers.
When \code{NDM_SOFTWARE_CONDA_ENV} is unset, the runner layer currently falls
back to \code{jax_cpu} on macOS and \code{ndm_software_env} elsewhere.
}
\section{Key functions}{
  \itemize{
    \item \code{\link{ndm_create_config}}
    \item \code{\link{ndm_build_backend}}
    \item \code{\link{ndm_initialize_backend}}
    \item \code{\link{ndm_model_spec}}
    \item \code{\link{ndm_model_spec_from_tex}}
    \item \code{\link{ndm_model_spec_to_tex}}
    \item \code{\link{ndm_read_tfrecord_dataset}}
    \item \code{\link{ndm_load_tfrecord_bundle}}
    \item \code{\link{ndm_prepare_runtime}}
    \item \code{\link{ndm_prepare_data}}
    \item \code{\link{ndm_build_model}}
    \item \code{\link{ndm_train}}
    \item \code{\link{ndm_predict}}
    \item \code{\link{ndm_loss}}
    \item \code{\link{ndm_fit}}
    \item \code{\link{ndm_create_real_run_config}}
    \item \code{\link{ndm_run_real}}
    \item \code{\link{ndm_run_sim}}
    \item \code{\link{ndm_run_multidisease}}
    \item \code{\link{ndm_save_model}}
    \item \code{\link{ndm_load_model}}
    \item \code{\link{ndm_resume_training}}
  }
}
\author{
Connor T. Jerzak \email{connor.jerzak@austin.utexas.edu}
}
