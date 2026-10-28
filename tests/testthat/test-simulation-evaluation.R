test_that("full inference panels include singleton tails and reject count mismatches", {
  collect <- function(batches, expected) {
    k <- 0L
    .ndm_collect_inference_panel(expected,
      function() { k <<- k + 1L; if (k > length(batches)) list(completed = TRUE) else list(batch = batches[[k]], completed = FALSE) },
      function(x) nrow(x), function(x, ids) list(value = x, ids = matrix(ids, ncol = 1L)))
  }
  x <- matrix(seq_len(15L), 5L, 3L)
  split <- collect(list(x[1:2,,drop=FALSE], x[3:4,,drop=FALSE], x[5,,drop=FALSE]), 5L)
  expect_identical(split$value, x)
  expect_identical(split, collect(list(x), 5L))
  expect_identical(as.vector(split$ids), 1:5)
  expect_error(collect(list(x[1:4,,drop=FALSE]), 5L), "ended early")
  expect_error(collect(list(x), 4L), "extra records")
  expect_error(collect(list(x[FALSE,,drop=FALSE]), 5L), "Malformed")
})

test_that("initial-state KL uses the correct direction and population scale", {
  p <- matrix(c(60, 40, 0, 0), 1L)
  q <- matrix(c(50, 50, 0, 0), 1L)
  kl <- .ndm_population_weighted_kl(p, q)
  expect_equal(kl$value, 60 * log(1.2) + 40 * log(.8))
  expect_equal(.ndm_population_weighted_kl(p, p)$value, 0)
  expect_equal(.ndm_population_weighted_kl(10*p, 10*q)$value, 10*kl$value)
  expect_identical(kl$status, "ok")
  mismatch <- .ndm_population_weighted_kl(p, matrix(c(100, 0, 0, 0), 1L))
  expect_identical(mismatch$status, "support_mismatch")
  expect_identical(mismatch$value, Inf)
  expect_equal(mismatch$infinite_count, 1L)
  expect_error(.ndm_population_weighted_kl(p, q * 2), "equal positive")
  expect_error(.ndm_population_weighted_kl(p, q * NA), "finite nonnegative")
  expect_error(.ndm_population_weighted_kl(p*0, q*0), "equal positive")
})

test_that("evaluation receipt commits the exact result bytes", {
  file <- tempfile(fileext = ".csv")
  on.exit(unlink(c(file, paste0(file, ".evaluation.rds"))))
  receipt <- .ndm_atomic_result_csv(data.frame(x = 1:3), file, list(checkpoint_step = 10L))
  expect_identical(receipt, readRDS(paste0(file, ".evaluation.rds")))
  expect_identical(receipt$result_sha256, digest::digest(file = file, algo = "sha256"))
})
