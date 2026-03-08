test_that("README tutorial blocks run directly from README.md", {
  readme <- ndm_readme_path()
  tutorial_blocks <- ndm_extract_readme_r_blocks(readme, tag = "tutorial")

  expect_gt(length(tutorial_blocks), 0L)

  env <- new.env(parent = globalenv())

  for (block in tutorial_blocks) {
    err <- tryCatch(
      {
        eval(parse(text = block$code), envir = env)
        NULL
      },
      error = function(e) e
    )

    if (!is.null(err)) {
      fail(
        sprintf(
          "README tutorial block starting at line %d failed: %s",
          block$start_line,
          conditionMessage(err)
        )
      )
    }
  }
})
