ndm_readme_path <- function() {
  path <- normalizePath(
    testthat::test_path("..", "..", "README.md"),
    winslash = "/",
    mustWork = FALSE
  )

  if (!file.exists(path)) {
    testthat::skip("README.md is not available in the current test environment.")
  }

  path
}

ndm_extract_readme_r_blocks <- function(path, tag = NULL) {
  lines <- readLines(path, warn = FALSE)
  blocks <- list()
  in_block <- FALSE
  block_lines <- character()
  block_start <- NA_integer_

  for (i in seq_along(lines)) {
    line <- lines[[i]]

    if (!in_block) {
      if (grepl("^```\\s*r\\b", line)) {
        in_block <- TRUE
        block_lines <- character()
        block_start <- i + 1L
      }
      next
    }

    if (grepl("^```\\s*$", line)) {
      blocks[[length(blocks) + 1L]] <- list(
        code = block_lines,
        start_line = block_start
      )
      in_block <- FALSE
      next
    }

    block_lines <- c(block_lines, line)
  }

  if (in_block) {
    stop("README.md contains an unterminated R code block.", call. = FALSE)
  }

  if (is.null(tag)) {
    return(blocks)
  }

  tagged_blocks <- Filter(function(block) {
    nonempty <- which(nzchar(trimws(block$code)))
    if (length(nonempty) == 0L) {
      return(FALSE)
    }

    marker <- trimws(block$code[[nonempty[[1L]]]])
    identical(marker, sprintf("# readme-test: %s", tag))
  }, blocks)

  lapply(tagged_blocks, function(block) {
    nonempty <- which(nzchar(trimws(block$code)))
    marker_idx <- nonempty[[1L]]

    list(
      code = paste(block$code[-marker_idx], collapse = "\n"),
      start_line = block$start_line
    )
  })
}
