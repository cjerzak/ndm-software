ndm_test_build_ml_loss_function <- function() {
  source_path <- testthat::test_path(
    "..", "..", "tools", "runtime_source", "ndm_runtime",
    "ModelDefiners", "SuperLModel_BuildML.R"
  )
  if (!file.exists(source_path)) {
    testthat::skip("runtime source tree is not available in this installed-package context")
  }

  find_assignment <- function(expr, target) {
    if (!is.call(expr)) {
      return(NULL)
    }
    if (identical(expr[[1L]], as.name("<-")) &&
        identical(expr[[2L]], as.name(target))) {
      return(expr[[3L]])
    }
    child_indices <- seq_along(expr)[-1L]
    for (child_index in child_indices) {
      if (identical(expr[[child_index]], quote(expr = ))) {
        next
      }
      child <- expr[[child_index]]
      found <- find_assignment(child, target)
      if (!is.null(found)) {
        return(found)
      }
    }
    NULL
  }

  expressions <- parse(file = source_path, keep.source = FALSE)
  for (expr in expressions) {
    found <- find_assignment(expr, "getLoss_train")
    if (!is.null(found)) {
      return(found)
    }
  }
  stop("Could not locate the getLoss_train definition.", call. = FALSE)
}

test_that("runtime loss selects model outcome channels at its boundary", {
  loss_function <- ndm_test_build_ml_loss_function()
  loss_source <- paste(deparse(loss_function, width.cutoff = 500L), collapse = "\n")
  runtime_source <- paste(
    readLines(
      testthat::test_path(
        "..", "..", "tools", "runtime_source", "ndm_runtime",
        "ModelDefiners", "SuperLModel_BuildML.R"
      ),
      warn = FALSE
    ),
    collapse = "\n"
  )

  selector_position <- regexpr(
    ".ndm_select_model_targets(", loss_source, fixed = TRUE
  )[[1L]]
  prediction_position <- regexpr(
    "GetPred_output <- GetPred_train(", loss_source, fixed = TRUE
  )[[1L]]

  expect_gt(selector_position, 0L)
  expect_gt(prediction_position, 0L)
  expect_lt(selector_position, prediction_position)
  expect_match(loss_source, "y = y", fixed = TRUE)
  expect_match(loss_source, "y_mask = y_mask", fixed = TRUE)
  expect_match(loss_source, "n_outcomes = nOutcomes", fixed = TRUE)
  expect_match(loss_source, "jnp = jnp", fixed = TRUE)
  expect_match(
    runtime_source,
    paste(
      '  .ndm_select_model_targets <- utils::getFromNamespace(',
      '    ".ndm_select_model_targets",',
      '    "ndm"',
      '  )',
      sep = "\n"
    ),
    fixed = TRUE
  )
})

test_that("runtime likelihood and NeuralODE MSE share selected targets", {
  loss_function <- ndm_test_build_ml_loss_function()
  loss_source <- paste(deparse(loss_function, width.cutoff = 500L), collapse = "\n")

  expect_match(loss_source, "loss_y <- model_targets$y", fixed = TRUE)
  expect_match(loss_source, "loss_y_mask <- model_targets$y_mask", fixed = TRUE)
  expect_match(loss_source, "y = loss_y", fixed = TRUE)
  expect_match(loss_source, "mask = loss_y_mask", fixed = TRUE)
  expect_match(
    loss_source,
    "observation_mask <- loss_y_mask$astype(GetPred_output$y_mu$dtype)",
    fixed = TRUE
  )
  expect_match(
    loss_source,
    "jnp$square(GetPred_output$y_mu - loss_y) * observation_mask",
    fixed = TRUE
  )
  expect_false(grepl("(^|\\n)[[:space:]]*y <- model_targets\\$y", loss_source))
  expect_false(grepl("(^|\\n)[[:space:]]*y_mask <- model_targets\\$y_mask", loss_source))
})

test_that("runtime loss rejects insufficient channels before tracing prediction", {
  loss_environment <- new.env(parent = baseenv())
  prediction_called <- FALSE
  loss_environment$.ndm_select_model_targets <- getFromNamespace(
    ".ndm_select_model_targets", "ndm"
  )
  loss_environment$nOutcomes <- 2L
  loss_environment$jnp <- NULL
  loss_environment$GetPred_train <- function(...) {
    prediction_called <<- TRUE
    stop("prediction should not be traced", call. = FALSE)
  }
  loss_function <- eval(ndm_test_build_ml_loss_function(), envir = loss_environment)

  expect_error(
    loss_function(
      ModelList = NULL,
      x = NULL,
      y = array(1, dim = c(1L, 2L, 1L)),
      y_mask = array(TRUE, dim = c(1L, 2L, 1L)),
      i = 1L,
      state = NULL,
      PriorList = NULL,
      PolicyList = NULL,
      GetPredSaveAtInfo = NULL,
      seed = NULL
    ),
    "`y` has 1 outcome channel(s), but `n_outcomes` is 2.",
    fixed = TRUE
  )
  expect_false(prediction_called)
})
