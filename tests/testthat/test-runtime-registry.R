runtime_registry_expr_name <- function(relative_path) {
  sprintf(
    ".ndm_stage_expr_%s",
    gsub("[^A-Za-z0-9]+", "_", sub("\\.R$", "", relative_path))
  )
}

runtime_registry_parent_paths <- function(path, depth = 6L) {
  path <- normalizePath(path, winslash = "/", mustWork = FALSE)
  parents <- character(depth + 1L)
  parents[[1L]] <- path

  for (i in seq_len(depth)) {
    parents[[i + 1L]] <- normalizePath(
      file.path(parents[[i]], ".."),
      winslash = "/",
      mustWork = FALSE
    )
  }

  unique(parents[nzchar(parents)])
}

runtime_registry_source_root <- function() {
  source_root <- try(ndm:::.ndm_runtime_source_root(), silent = TRUE)
  if (!inherits(source_root, "try-error") && !is.null(source_root) && dir.exists(source_root)) {
    return(source_root)
  }

  start_paths <- unique(c(
    getwd(),
    tryCatch(testthat::test_path("..", ".."), error = function(...) ""),
    system.file(package = "ndm")
  ))
  candidate_roots <- unique(unlist(lapply(start_paths[nzchar(start_paths)], runtime_registry_parent_paths)))
  candidate_rel_paths <- c(
    file.path("tools", "runtime_source", "ndm_runtime"),
    file.path("00_pkg_src", "ndm", "tools", "runtime_source", "ndm_runtime")
  )

  for (root in candidate_roots) {
    for (rel_path in candidate_rel_paths) {
      candidate <- file.path(root, rel_path)
      if (dir.exists(candidate)) {
        return(normalizePath(candidate, winslash = "/", mustWork = TRUE))
      }
    }
  }

  testthat::skip("runtime source tree is not available in this installed-package context")
}

runtime_source_relative_files <- function(source_root) {
  entries <- list.files(source_root, recursive = TRUE, full.names = FALSE)
  entries <- entries[file.info(file.path(source_root, entries))$isdir %in% FALSE]
  sort(entries)
}

runtime_source_text <- function(source_root, relative_path) {
  paste(readLines(file.path(source_root, relative_path), warn = FALSE), collapse = "\n")
}

runtime_registry_escape_non_ascii <- function(lines) {
  vapply(
    enc2utf8(lines),
    function(line) {
      code_points <- utf8ToInt(line)
      escaped <- vapply(
        code_points,
        function(code_point) {
          if (code_point <= 127L) {
            intToUtf8(code_point)
          } else if (code_point <= 65535L) {
            sprintf("\\\\u%04X", code_point)
          } else {
            sprintf("\\\\U%08X", code_point)
          }
        },
        character(1),
        USE.NAMES = FALSE
      )
      paste0(escaped, collapse = "")
    },
    character(1),
    USE.NAMES = FALSE
  )
}

runtime_registry_expected_expr <- function(path) {
  eval(
    parse(
      text = paste(
        runtime_registry_escape_non_ascii(
          capture.output(dput(parse(file = path, keep.source = FALSE)))
        ),
        collapse = "\n"
      )
    ),
    envir = baseenv()
  )
}

expect_runtime_expr_matches_source <- function(expr_name, relative_path) {
  ns <- asNamespace("ndm")
  source_root <- runtime_registry_source_root()
  source_path <- file.path(source_root, relative_path)

  expect_true(exists(expr_name, envir = ns, inherits = FALSE), info = relative_path)
  expect_identical(
    get(expr_name, envir = ns, inherits = FALSE),
    runtime_registry_expected_expr(source_path),
    info = relative_path
  )
}

test_that("generated runtime registry stays in sync with runtime source files", {
  ns <- asNamespace("ndm")
  runtime_registry <- get(".ndm_runtime_stage_registry", envir = ns, inherits = FALSE)

  expect_true(length(runtime_registry) > 0L)

  for (relative_path in names(runtime_registry)) {
    expect_runtime_expr_matches_source(
      runtime_registry_expr_name(relative_path),
      relative_path
    )
  }
})

test_that("generated runtime special expressions stay in sync with runtime source files", {
  special_runtime_files <- c(
    ".ndm_run_impl_expr" = "SetupEnv/Analysis2_api.R",
    ".ndm_multidisease_driver_expr" = "SetupEnv/Analysis2_legacy_multidisease_driver.R"
  )

  for (expr_name in names(special_runtime_files)) {
    expect_runtime_expr_matches_source(expr_name, special_runtime_files[[expr_name]])
  }
})

test_that("generated embedded runtime sources stay in sync with runtime source files", {
  ns <- asNamespace("ndm")
  source_root <- runtime_registry_source_root()
  embedded_sources <- get(".ndm_embedded_runtime_sources", envir = ns, inherits = FALSE)
  relative_paths <- runtime_source_relative_files(source_root)

  expect_setequal(names(embedded_sources), relative_paths)

  for (relative_path in relative_paths) {
    expect_identical(
      embedded_sources[[relative_path]],
      runtime_source_text(source_root, relative_path),
      info = relative_path
    )
  }
})

test_that("runtime source keeps NeuralODE and transformer regression fixes", {
  source_root <- runtime_registry_source_root()
  build_ml <- runtime_source_text(source_root, "ModelDefiners/SuperLModel_BuildML.R")
  sim_data <- runtime_source_text(source_root, "SetupData/SuperLModel_DataGenerator_Sim.R")
  transformer <- runtime_source_text(source_root, "ModelDefiners/SuperLModel_BackboneTransformer.R")
  parser <- runtime_source_text(source_root, "ModelDefiners/SuperLModel_ParseDynamicODE.R")
  analysis2_api <- runtime_source_text(source_root, "SetupEnv/Analysis2_api.R")

  expect_match(build_ml, "neuralode_variational", fixed = TRUE)
  expect_match(build_ml, "jnp$mean(GetPred_output$KL_TERM)", fixed = TRUE)
  expect_match(build_ml, "testWithoutSampling <- !isTRUE(neuralode_variational)", fixed = TRUE)
  expect_match(sim_data, "\"YTrue_out\" = (YTrue_out_ <- jnp$expand_dims(YTrue_d_out,2L))", fixed = TRUE)
  expect_false(grepl("astype\\(jaxFloatType,1L\\)", transformer))
  expect_false(grepl("eval\\(sprintf", parser))
  expect_match(analysis2_api, "ndm.analysis2.expose_runtime_env", fixed = TRUE)
})
