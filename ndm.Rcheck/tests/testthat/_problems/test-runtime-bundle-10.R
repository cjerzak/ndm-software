# Extracted from test-runtime-bundle.R:10

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "ndm", path = "..")
attach(test_env, warn.conflicts = FALSE)

# test -------------------------------------------------------------------------
payload_env <- new.env(parent = emptyenv())
load(testthat::test_path("..", "..", "R", "sysdata.rda"), envir = payload_env)
