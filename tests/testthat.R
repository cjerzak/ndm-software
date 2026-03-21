library(testthat)
library(ndm)

ndm_cleanup_test_detritus <- function(root = tempdir()) {
  roots <- unique(c(root, dirname(root)))
  roots <- vapply(roots, normalizePath, character(1), winslash = "/", mustWork = FALSE)
  roots <- roots[nzchar(roots) & dir.exists(roots)]
  if (length(roots) == 0L) {
    return(invisible(NULL))
  }

  entries <- unlist(
    lapply(
      roots,
      function(path) {
        list.files(
          path,
          all.files = TRUE,
          full.names = TRUE,
          recursive = TRUE,
          include.dirs = TRUE,
          no.. = TRUE
        )
      }
    ),
    use.names = FALSE
  )
  basenames <- basename(entries)
  removable <- unique(entries[
    basenames == "__pycache__" |
      basenames == "etc" |
      grepl("^__autograph_generated_file.*\\.py$", basenames)
  ])
  removable <- sort(unique(removable), decreasing = TRUE)
  if (length(removable) > 0L) {
    unlink(removable, recursive = TRUE, force = TRUE)
  }

  invisible(NULL)
}

local({
  on.exit(ndm_cleanup_test_detritus(), add = TRUE)
  test_check("ndm")
})
