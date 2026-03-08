\name{ndm-package}
\alias{ndm-package}
\alias{ndm}
\title{Neural Disease Modeling Software}
\description{
\code{ndm} is an R-first package for neural disease modeling with a
reticulate-managed JAX backend. The package currently provides backend setup,
epidemic model spec import/export, TFRecord readers, a bundled internal
analysis runtime, and package-native runtime/model helpers.
}
\details{
The current implementation preserves both historical model families,
\code{"DecoderOnly"} and \code{"NeuralODE"}, while defaulting to
\code{"DecoderOnly"}. The supported backbone is the transformer path only.
Latent attention and Mamba are intentionally excluded.

The package treats the structured \code{ndm_model_spec} object as the canonical
internal representation and supports \code{.tex} import/export for compatibility
with the original epidemic specification workflow.
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
  }
}
\author{
Connor T. Jerzak \email{connor.jerzak@austin.utexas.edu}
}
