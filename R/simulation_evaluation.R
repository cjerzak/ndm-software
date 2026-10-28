# Inference deliberately keeps only host-side summaries, never model draw trees.
.ndm_collect_inference_panel <- function(expected_count, next_batch, batch_size, reduce_batch) {
  if (length(expected_count) != 1L || !is.finite(expected_count) ||
      expected_count < 1 || expected_count != floor(expected_count)) {
    stop("Canonical inference requires a positive manifest record count.", call. = FALSE)
  }
  parts <- list(); seen <- 0L
  repeat {
    item <- next_batch()
    if (isTRUE(item$completed)) break
    sizes <- batch_size(item$batch)
    if (!length(sizes) || anyNA(sizes) || any(sizes < 1L) || any(sizes != sizes[[1L]])) {
      stop("Malformed or inconsistent canonical inference batch.", call. = FALSE)
    }
    n <- sizes[[1L]]
    if (seen + n > expected_count) stop("Canonical inference contains extra records.", call. = FALSE)
    ids <- seq.int(seen + 1L, seen + n)
    part <- reduce_batch(item$batch, ids)
    if (!is.list(part) || !length(part) || any(vapply(part, nrow, integer(1L)) != n)) {
      stop("Inference reduction lost its canonical example axis.", call. = FALSE)
    }
    parts[[length(parts) + 1L]] <- part
    seen <- seen + n
  }
  if (seen != expected_count) stop("Canonical inference ended early: ", seen, " of ", expected_count, " records.", call. = FALSE)
  fields <- names(parts[[1L]])
  if (any(!vapply(parts, function(x) identical(names(x), fields), logical(1L)))) {
    stop("Canonical inference batches have inconsistent fields.", call. = FALSE)
  }
  stats::setNames(lapply(fields, function(nm) do.call(rbind, lapply(parts, `[[`, nm))), fields)
}

.ndm_population_weighted_kl <- function(truth, estimate, tolerance = 1e-5) {
  truth <- as.matrix(truth); estimate <- as.matrix(estimate)
  if (!identical(dim(truth), dim(estimate)) || !nrow(truth) ||
      any(!is.finite(truth)) || any(!is.finite(estimate)) ||
      any(truth < 0) || any(estimate < 0)) {
    stop("KL initial states must be aligned finite nonnegative masses.", call. = FALSE)
  }
  mass <- rowSums(truth); estimated_mass <- rowSums(estimate)
  if (any(mass <= 0 | estimated_mass <= 0) ||
      any(abs(mass - estimated_mass) > tolerance * pmax(mass, estimated_mass))) {
    stop("KL initial states must have equal positive population totals.", call. = FALSE)
  }
  p <- truth / mass; q <- estimate / estimated_mass
  contribution <- matrix(0, nrow(truth), ncol(truth))
  positive <- p > 0
  contribution[positive] <- truth[positive] * (log(p[positive]) - log(q[positive]))
  values <- rowSums(contribution)
  # A KL can be slightly negative only through floating-point roundoff.
  values[is.finite(values) & values < 0 & values >= -tolerance * mass] <- 0
  if (any(values < 0)) stop("Negative population-weighted KL beyond tolerance.", call. = FALSE)
  list(value = stats::median(values), per_example = values,
       status = if (any(is.infinite(values))) "support_mismatch" else "ok",
       infinite_count = sum(is.infinite(values)))
}

.ndm_atomic_result_csv <- function(data, file, receipt) {
  dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(".evaluation-", tmpdir = dirname(file))
  on.exit(unlink(temporary), add = TRUE)
  data.table::fwrite(data, temporary)
  receipt$schema <- "ndm-evaluation-v2"
  receipt$result_sha256 <- digest::digest(file = temporary, algo = "sha256")
  receipt$result_file <- basename(file)
  receipt_file <- paste0(file, ".evaluation.rds")
  temporary_receipt <- tempfile(".receipt-", tmpdir = dirname(file))
  on.exit(unlink(temporary_receipt), add = TRUE)
  saveRDS(receipt, temporary_receipt)
  # A receipt is the commit marker; a crash between renames is incomplete.
  if (!file.rename(temporary, file) || !file.rename(temporary_receipt, receipt_file)) {
    stop("Could not atomically publish inference output and receipt.", call. = FALSE)
  }
  invisible(receipt)
}
