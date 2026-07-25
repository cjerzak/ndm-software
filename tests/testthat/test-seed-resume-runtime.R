ndm_test_runtime_assignment <- function(relative_path, target) {
  expressions <- ndm_test_runtime_source_expressions(relative_path)
  find_assignment <- function(expr) {
    if (!is.call(expr)) {
      return(NULL)
    }
    if (identical(expr[[1L]], as.name("<-")) &&
        is.symbol(expr[[2L]]) &&
        identical(as.character(expr[[2L]]), target)) {
      return(expr)
    }
    for (child_index in seq_along(expr)[-1L]) {
      child <- expr[[child_index]]
      if (identical(child, quote(expr = ))) {
        next
      }
      found <- find_assignment(child)
      if (!is.null(found)) {
        return(found)
      }
    }
    NULL
  }
  for (expr in expressions) {
    found <- find_assignment(expr)
    if (!is.null(found)) {
      return(found)
    }
  }
  stop("Could not locate runtime assignment `", target, "`.", call. = FALSE)
}

test_that("BuildML seed derivation preserves the full public integer domain", {
  seed_assignment <- ndm_test_runtime_assignment(
    file.path("ModelDefiners", "SuperLModel_BuildML.R"),
    "ndm_runtime_run_seed"
  )
  key_assignment <- ndm_test_runtime_assignment(
    file.path("ModelDefiners", "SuperLModel_BuildML.R"),
    "ndm_runtime_seed_key"
  )
  boundary_seeds <- c(0, 349, 350, .Machine$integer.max)
  derived <- vector("list", length(boundary_seeds))

  for (seed_index in seq_along(boundary_seeds)) {
    runtime_env <- new.env(parent = baseenv())
    runtime_env$ndm_runtime_get0 <- function(name, ifnotfound = NULL) {
      if (identical(name, "SEED_")) {
        return(boundary_seeds[[seed_index]])
      }
      ifnotfound
    }
    runtime_env$JaxKey <- function(seed) {
      c(0L, as.integer(seed))
    }
    runtime_env$jax <- list(
      random = list(
        fold_in = function(key, domain_tag) {
          c(root_seed = key[[2L]], domain_tag = as.integer(domain_tag))
        }
      )
    )

    eval(seed_assignment, envir = runtime_env)
    eval(key_assignment, envir = runtime_env)
    derived[[seed_index]] <- runtime_env$ndm_runtime_seed_key(6152228L)
  }

  expect_identical(
    vapply(derived, `[[`, integer(1L), "root_seed"),
    as.integer(boundary_seeds)
  )
  expect_identical(
    vapply(derived, `[[`, integer(1L), "domain_tag"),
    rep.int(6152228L, length(boundary_seeds))
  )
  expect_length(unique(vapply(derived, `[[`, integer(1L), "root_seed")), 4L)
  expect_identical(
    runtime_env$ndm_runtime_seed_key(6152228L),
    derived[[4L]]
  )
  expect_false(identical(
    runtime_env$ndm_runtime_seed_key(6152228L),
    runtime_env$ndm_runtime_seed_key(6152229L)
  ))
})

test_that("BuildML has no multiplicative run-seed key construction", {
  runtime_source <- ndm_test_runtime_source_text(
    "ModelDefiners/SuperLModel_BuildML.R"
  )

  expect_match(runtime_source, "ndm_runtime_seed_key(6152228L)", fixed = TRUE)
  expect_false(grepl("SEED_[[:space:]]*\\*", runtime_source))
  expect_false(grepl("\\*[[:space:]]*SEED_", runtime_source))
})

test_that("JAX keeps boundary run seeds distinct after domain separation", {
  skip_on_cran()
  ndm_require_backend_test_stack(
    "runtime seed boundary tests",
    modules = c("jax", "numpy"),
    packages = "reticulate"
  )
  jax <- reticulate::import("jax")
  np <- reticulate::import("numpy")
  seed_assignment <- ndm_test_runtime_assignment(
    file.path("ModelDefiners", "SuperLModel_BuildML.R"),
    "ndm_runtime_run_seed"
  )
  key_assignment <- ndm_test_runtime_assignment(
    file.path("ModelDefiners", "SuperLModel_BuildML.R"),
    "ndm_runtime_seed_key"
  )
  boundary_seeds <- c(0, 349, 350, .Machine$integer.max)
  key_signature <- function(seed) {
    runtime_env <- new.env(parent = baseenv())
    runtime_env$ndm_runtime_get0 <- function(name, ifnotfound = NULL) {
      if (identical(name, "SEED_")) {
        return(seed)
      }
      ifnotfound
    }
    runtime_env$JaxKey <- jax$random$PRNGKey
    runtime_env$jax <- jax
    eval(seed_assignment, envir = runtime_env)
    eval(key_assignment, envir = runtime_env)
    key <- runtime_env$ndm_runtime_seed_key(6152228L)
    paste(
      format(
        as.numeric(np$array(key)),
        scientific = FALSE,
        trim = TRUE
      ),
      collapse = ":"
    )
  }
  derived <- vapply(boundary_seeds, key_signature, character(1L))

  expect_length(unique(derived), length(boundary_seeds))
  expect_identical(key_signature(350L), derived[[3L]])
})

test_that("training iteration keys distinguish zero and maximum run seeds", {
  skip_on_cran()
  ndm_require_backend_test_stack(
    "training iteration seed boundary tests",
    modules = c("jax", "numpy"),
    packages = "reticulate"
  )
  jax <- reticulate::import("jax")
  np <- reticulate::import("numpy")
  seed_assignment <- ndm_test_runtime_assignment(
    file.path("ModelDefiners", "SuperLModel_BuildML.R"),
    "ndm_runtime_run_seed"
  )
  seed_key_assignment <- ndm_test_runtime_assignment(
    file.path("ModelDefiners", "SuperLModel_BuildML.R"),
    "ndm_runtime_seed_key"
  )
  iteration_key_assignment <- ndm_test_runtime_assignment(
    file.path("ModelTrainers", "SuperLModel_TrainDo.R"),
    "ndm_training_iteration_key"
  )
  key_signature <- function(run_seed, iteration) {
    runtime_env <- new.env(parent = baseenv())
    runtime_env$ndm_runtime_get0 <- function(name, ifnotfound = NULL) {
      if (identical(name, "SEED_")) {
        return(run_seed)
      }
      ifnotfound
    }
    runtime_env$JaxKey <- jax$random$PRNGKey
    runtime_env$jax <- jax
    eval(seed_assignment, envir = runtime_env)
    eval(seed_key_assignment, envir = runtime_env)
    eval(iteration_key_assignment, envir = runtime_env)
    paste(
      format(
        as.numeric(np$array(
          runtime_env$ndm_training_iteration_key(iteration)
        )),
        scientific = FALSE,
        trim = TRUE
      ),
      collapse = ":"
    )
  }

  zero_first <- key_signature(0L, 1L)
  max_first <- key_signature(.Machine$integer.max, 1L)
  expect_false(identical(zero_first, max_first))
  expect_false(identical(zero_first, key_signature(0L, 2L)))
  expect_identical(max_first, key_signature(.Machine$integer.max, 1L))
})

test_that("completed training resumes have an empty iteration sequence", {
  train_do_source <- ndm_test_runtime_source_text(
    "ModelTrainers/SuperLModel_TrainDo.R"
  )
  sequence_assignment <- ndm_test_runtime_assignment(
    file.path("ModelTrainers", "SuperLModel_TrainDo.R"),
    "ndm_training_iteration_sequence"
  )
  runtime_env <- new.env(parent = baseenv())
  eval(sequence_assignment, envir = runtime_env)

  expect_match(
    train_do_source,
    "for(i in ndm_training_iteration_sequence(i_, nSGD_model))",
    fixed = TRUE
  )

  completed_env <- new.env(parent = emptyenv())
  completed_env$in_loss_vec <- rep(1, 500L)
  completed_env$i_ <- 500L
  completed_state <- ndm:::.ndm_collect_training_state(completed_env)

  partial_env <- new.env(parent = emptyenv())
  partial_env$in_loss_vec <- c(rep(1, 498L), NA_real_, NA_real_)
  partial_env$i_ <- 498L
  partial_state <- ndm:::.ndm_collect_training_state(partial_env)

  restored_completed_env <- new.env(parent = emptyenv())
  ndm:::.ndm_restore_training_state(
    restored_completed_env,
    completed_state
  )
  restored_partial_env <- new.env(parent = emptyenv())
  ndm:::.ndm_restore_training_state(
    restored_partial_env,
    partial_state
  )

  expect_identical(completed_state$last_completed_iteration, 500L)
  expect_identical(completed_state$resume_iteration, 501L)
  expect_identical(partial_state$last_completed_iteration, 498L)
  expect_identical(partial_state$resume_iteration, 499L)
  expect_identical(restored_completed_env$i_, 501L)
  expect_identical(restored_partial_env$i_, 499L)
  expect_identical(
    runtime_env$ndm_training_iteration_sequence(
      restored_completed_env$i_,
      500L
    ),
    integer()
  )
  expect_identical(
    runtime_env$ndm_training_iteration_sequence(500L, 500L),
    500L
  )
  expect_identical(
    runtime_env$ndm_training_iteration_sequence(499L, 500L),
    499:500
  )

  update_count <- 0L
  for (iteration in runtime_env$ndm_training_iteration_sequence(
    restored_completed_env$i_,
    500L
  )) {
    update_count <- update_count + 1L
  }
  expect_identical(update_count, 0L)
})
