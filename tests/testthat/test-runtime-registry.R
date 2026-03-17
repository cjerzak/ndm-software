runtime_registry_expr_name <- function(relative_path) {
  sprintf(
    ".ndm_stage_expr_%s",
    gsub("[^A-Za-z0-9]+", "_", sub("\\.R$", "", relative_path))
  )
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
  source_root <- testthat::test_path("..", "..", "tools", "runtime_source", "ndm_runtime")
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
