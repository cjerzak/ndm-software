.ndm_report_columns <- function(data, prefix) {
  columns <- grep(paste0("^", prefix, "[0-9]+$"), names(data), value = TRUE)
  indices <- as.integer(sub(prefix, "", columns, fixed = TRUE))
  if (anyDuplicated(indices)) stop("Duplicate numeric horizon columns for ", prefix, call. = FALSE)
  columns[order(indices)]
}

.ndm_result_metadata <- function(data, env, evaluation_horizon) {
  data$training_horizon <- as.integer(env$nTimesLookahead %||% env$VI_TotalTimesInLikelihood)
  data$evaluation_horizon <- as.integer(evaluation_horizon)
  data$nParamsTotal <- env$nParamsTotal
  data$nParamsTrainable <- env$nParamsTrainable
  data$nParamsNonTrainable <- env$nParamsNonTrainable
  data$nParamsModel <- env$nParamsTotal # retained total-count alias
  if (!"ExperimentID" %in% names(data)) data$ExperimentID <- paste(
    if (isTRUE(env$SimMode)) "sim" else "real", env$AnalysisName, env$AnalysisDate, sep = ":")
  if (!"ReplicateID" %in% names(data)) data$ReplicateID <- "1"
  if (!"RunID" %in% names(data)) data$RunID <- env$RUN_ID %||% paste0("legacy:", env$OUTER_ITERATION)
  data$ModelType <- env$ModelType
  data$model_semantics_version <- 2L
  data
}

.ndm_report_key <- function(data, columns) {
  if (!nrow(data)) return(character())
  if (!all(columns %in% names(data))) stop("Missing report identity fields: ", paste(setdiff(columns, names(data)), collapse = ", "), call. = FALSE)
  if (anyNA(data[, columns, drop = FALSE])) stop("Report identity fields cannot be missing.", call. = FALSE)
  # Length-prefix encoding avoids collisions when IDs contain separators.
  fields <- lapply(data[, columns, drop = FALSE], function(x) paste0(nchar(as.character(x)), ":", x))
  do.call(paste, c(fields, sep = "|"))
}

.ndm_report_validate <- function(data, training_horizon = NULL, experiment_id = NULL) {
  data <- as.data.frame(data)
  if (!nrow(data)) stop("No result rows to report.", call. = FALSE)
  if (!is.null(training_horizon)) {
    if (length(training_horizon) != 1L) stop("training_horizon override must be scalar.", call. = FALSE)
    if ("training_horizon" %in% names(data) && any(as.numeric(data$training_horizon) != training_horizon)) stop("training_horizon override conflicts with saved metadata.", call. = FALSE)
    data$training_horizon <- training_horizon
  }
  if (!"training_horizon" %in% names(data)) stop("Historical results require explicit training_horizon metadata or an override; evaluation length is insufficient.", call. = FALSE)
  raw <- length(.ndm_report_columns(data, "Truth_l")) > 0L
  primary <- .ndm_report_columns(data, if (raw) "Truth_l" else "SkillTime")
  if (!length(primary)) stop("No horizon result columns found.", call. = FALSE)
  horizon <- as.integer(sub(if (raw) "Truth_l" else "SkillTime", "", primary, fixed = TRUE))
  if (!identical(horizon, seq_len(max(horizon)))) stop("Missing structural horizon columns.", call. = FALSE)
  if (!"evaluation_horizon" %in% names(data)) data$evaluation_horizon <- max(horizon)
  for (field in c("training_horizon", "evaluation_horizon", "i_in_sgd", "nSGD")) {
    if (!field %in% names(data)) stop("Missing result metadata: ", field, call. = FALSE)
    data[[field]] <- suppressWarnings(as.numeric(data[[field]]))
    if (any(!is.finite(data[[field]]) | data[[field]] < 1 | data[[field]] != floor(data[[field]]))) stop("Invalid result metadata: ", field, call. = FALSE)
  }
  if (any(data$training_horizon > data$evaluation_horizon) || any(data$evaluation_horizon > max(horizon))) stop("Reported horizon exceeds available evaluation columns.", call. = FALSE)
  companions <- if (raw) "Pred_l" else c("RSSPredTime", "RSSBaselineTime")
  for (prefix in companions) {
    if (!identical(.ndm_report_columns(data, prefix), paste0(prefix, horizon))) stop("Misaligned or missing horizon columns for ", prefix, call. = FALSE)
  }
  if (!"ExperimentID" %in% names(data)) data$ExperimentID <- experiment_id %||% "legacy"
  if (!"ReplicateID" %in% names(data)) data$ReplicateID <- "1"
  if (!"pair_id" %in% names(data)) {
    fallback <- c("BaseID", "nSamplesTrain", "ModelDepth", "ModelDims")
    data$pair_id <- .ndm_report_key(data, fallback)
  }
  if (!"RunID" %in% names(data)) {
    if (!"model_id" %in% names(data)) {
      data$RunID <- .ndm_report_key(data, c("ExperimentID", "ReplicateID", "pair_id", "ModelType"))
    } else data$RunID <- as.character(data$model_id)
  }
  data$PairID_recon <- .ndm_report_key(data, c("ExperimentID", "ReplicateID", "pair_id", "training_horizon", "evaluation_horizon", "i_in_sgd"))
  data
}

.ndm_report_aggregate_real <- function(data) {
  truth_cols <- .ndm_report_columns(data, "Truth_l")
  pred_cols <- .ndm_report_columns(data, "Pred_l")
  prev_cols <- .ndm_report_columns(data, "PreviousSequenceTruth_l")
  if (!length(prev_cols)) stop("Real results require previous-sequence truth for the persistence baseline.", call. = FALSE)
  group_fields <- c("ExperimentID", "ReplicateID", "RunID", "ModelType", "pair_id", "i_in_sgd", "training_horizon", "evaluation_horizon")
  group <- .ndm_report_key(data, group_fields)
  example_key <- .ndm_report_key(data, c(group_fields, "location_id", "time_anchor_id"))
  if (anyDuplicated(example_key)) stop("Duplicate real-result example at the same run and checkpoint.", call. = FALSE)
  series_cols <- grep("^(Truth_l|Pred_l|PredBase_l|PreviousSequenceTruth_l|SliceSkillTime|RSSPredTime|RSSBaselineTime|SkillTime)[0-9]+$", names(data), value = TRUE)
  metadata <- setdiff(names(data), c(series_cols, "location_id", "time_anchor_id"))
  rows <- lapply(split(seq_len(nrow(data)), group), function(indices) {
    slice <- data[indices, , drop = FALSE]
    invariant <- vapply(slice[, metadata, drop = FALSE], function(x) length(unique(x)) == 1L, logical(1L))
    required <- c(group_fields, "nSGD", "PairID_recon", intersect(c("nParamsTrainable", "nParamsTotal", "nParamsNonTrainable"), names(slice)))
    if (any(!invariant[required])) stop("Conflicting metadata within one run/checkpoint.", call. = FALSE)
    out <- slice[1L, metadata[invariant], drop = FALSE]
    numeric_matrix <- function(columns) as.matrix(data.frame(lapply(slice[, columns, drop = FALSE], as.numeric)))
    previous <- numeric_matrix(prev_cols)
    last <- max.col(is.finite(previous), ties.method = "last")
    baseline <- previous[cbind(seq_len(nrow(previous)), last)]
    baseline[rowSums(is.finite(previous)) == 0L] <- NA_real_
    truth <- numeric_matrix(truth_cols); pred <- numeric_matrix(pred_cols)
    for (h in seq_len(as.integer(out$evaluation_horizon))) {
      ok <- is.finite(truth[, h]) & is.finite(pred[, h]) & is.finite(baseline)
      rb <- if (any(ok)) mean((baseline[ok] - truth[ok, h])^2) else NA_real_
      rp <- if (any(ok)) mean((pred[ok, h] - truth[ok, h])^2) else NA_real_
      out[[paste0("RSSBaselineTime", h)]] <- rb
      out[[paste0("RSSPredTime", h)]] <- rp
      out[[paste0("SkillTime", h)]] <- 1 - (.001 + sqrt(rp)) / (.001 + sqrt(rb))
      out[[paste0("ValidExamplesTime", h)]] <- sum(ok)
    }
    out
  })
  as.data.frame(data.table::rbindlist(rows, fill = TRUE))
}

.ndm_report_summaries <- function(data) {
  for (prefix in c("Skill", "RSSPred", "RSSBaseline")) {
    values <- counts <- numeric(nrow(data))
    for (row in seq_len(nrow(data))) {
      h <- as.integer(data$training_horizon[[row]])
      columns <- paste0(prefix, "Time", seq.int(max(1L, round(.75 * h)), h))
      if (!all(columns %in% names(data))) stop("Missing summary horizon columns.", call. = FALSE)
      v <- as.numeric(data[row, columns, drop = TRUE]); ok <- is.finite(v)
      counts[[row]] <- sum(ok)
      values[[row]] <- if (any(ok)) mean(v[ok]) else NA_real_
    }
    data[[paste0(prefix, "Summary")]] <- values
    data[[paste0(prefix, "SummaryCount")]] <- counts
  }
  data$is_final <- data$i_in_sgd == data$nSGD
  if ("nParamsTrainable" %in% names(data)) {
    data$nParamsTrainable <- as.numeric(data$nParamsTrainable)
    if (any(!is.finite(data$nParamsTrainable) | data$nParamsTrainable <= 0)) stop("Invalid trainable parameter count.", call. = FALSE)
    data$UpdatesPerParam <- data$i_in_sgd / data$nParamsTrainable
    if ("nSamplesTrain" %in% names(data)) data$DataParamRatio <- as.numeric(data$nSamplesTrain) / data$nParamsTrainable
  }
  data
}

.ndm_report_pair <- function(data) {
  selected <- data[data$ModelType %in% c("DecoderOnly", "NeuralODE"), , drop = FALSE]
  unique_key <- .ndm_report_key(selected, c("PairID_recon", "ModelType"))
  if (anyDuplicated(unique_key)) stop("Ambiguous duplicate model rows for one experiment, replicate, pair and checkpoint; supply distinct ReplicateID values.", call. = FALSE)
  decoder <- selected[selected$ModelType == "DecoderOnly", , drop = FALSE]
  neural <- selected[selected$ModelType == "NeuralODE", , drop = FALSE]
  paired <- merge(decoder, neural, by = "PairID_recon", suffixes = c("_Decoder", "_NeuralODE"))
  paired$is_final_pair <- paired$is_final_Decoder & paired$is_final_NeuralODE & paired$nSGD_Decoder == paired$nSGD_NeuralODE
  paired$WinNeuralODE <- ifelse(is.finite(paired$SkillSummary_Decoder) & is.finite(paired$SkillSummary_NeuralODE),
                              paired$SkillSummary_NeuralODE >= paired$SkillSummary_Decoder, NA)
  unmatched <- rbind(decoder[!decoder$PairID_recon %in% neural$PairID_recon, , drop = FALSE],
                     neural[!neural$PairID_recon %in% decoder$PairID_recon, , drop = FALSE])
  list(paired = paired, final_pairs = paired[paired$is_final_pair, , drop = FALSE], unmatched = unmatched)
}

#' Summarize and pair saved forecast results
#'
#' Comparisons match experiment, replicate, pair, training/evaluation horizons
#' and checkpoint. Final comparisons require both runs to reach the same
#' declared training budget. Summary windows span round(0.75 * H) through H,
#' with a minimum starting horizon of one. Missing observations have explicit
#' counts; missing structural columns and ambiguous duplicates are errors.
#'
#' @param results A result data frame or list of per-file data frames. A list
#'   validates each file before combining columns.
#' @param training_horizon Explicit training horizon for historical files that
#'   lack it. Never inferred from a longer evaluation horizon.
#' @param experiment_id Identity for historical results lacking ExperimentID.
#' @param output_dir Optional directory for CSV tables and a PDF per horizon
#'   cohort. Existing report files with these names are replaced.
#' @returns A list with results, paired, final_pairs, unmatched, and cohorts.
#' @export
ndm_summarize_results <- function(results, training_horizon = NULL, experiment_id = NULL, output_dir = NULL) {
  files <- if (is.data.frame(results)) list(results) else results
  if (!is.list(files) || !length(files)) stop("results must contain result data frames.", call. = FALSE)
  validated <- lapply(files, .ndm_report_validate, training_horizon = training_horizon, experiment_id = experiment_id)
  raw <- vapply(validated, function(x) length(.ndm_report_columns(x, "Truth_l")) > 0, logical(1L))
  if (length(unique(raw)) != 1L) stop("Cannot combine raw real predictions and aggregate simulation results.", call. = FALSE)
  data <- as.data.frame(data.table::rbindlist(validated, fill = TRUE))
  if (all(raw)) data <- .ndm_report_aggregate_real(data)
  data <- .ndm_report_summaries(data)
  paired <- .ndm_report_pair(data)
  cohort_key <- paste0("train", data$training_horizon, "_eval", data$evaluation_horizon)
  cohort_indices <- split(seq_len(nrow(data)), cohort_key)
  cohorts <- lapply(cohort_indices, function(i) {
    rows <- data[i, , drop = FALSE]; pairs <- .ndm_report_pair(rows)
    final <- pairs$final_pairs
    data.frame(training_horizon = rows$training_horizon[[1L]], evaluation_horizon = rows$evaluation_horizon[[1L]],
      result_rows = nrow(rows), matched_checkpoints = nrow(pairs$paired), unmatched_rows = nrow(pairs$unmatched),
      final_pairs = nrow(final), valid_final_pairs = sum(!is.na(final$WinNeuralODE)),
      neural_win_fraction = if (any(!is.na(final$WinNeuralODE))) mean(final$WinNeuralODE, na.rm = TRUE) else NA_real_)
  })
  report <- c(list(results = data), paired, list(cohorts = do.call(rbind, cohorts)))
  if (!is.null(output_dir)) .ndm_write_report(report, output_dir, cohort_indices)
  report
}

.ndm_write_report <- function(report, output_dir, cohort_indices) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  for (name in names(report)) data.table::fwrite(report[[name]], file.path(output_dir, paste0(name, ".csv")))
  for (key in names(cohort_indices)) {
    data <- report$results[cohort_indices[[key]], , drop = FALSE]
    pairs <- .ndm_report_pair(data)$final_pairs
    grDevices::pdf(file.path(output_dir, paste0("forecast_", key, ".pdf")), width = 10, height = 5)
    tryCatch({
      graphics::par(mfrow = c(1, 2))
      h <- as.integer(data$evaluation_horizon[[1L]])
      curves <- t(vapply(seq_len(h), function(i) {
        decoder <- as.numeric(pairs[[paste0("SkillTime", i, "_Decoder")]])
        neural <- as.numeric(pairs[[paste0("SkillTime", i, "_NeuralODE")]])
        ok <- is.finite(decoder) & is.finite(neural)
        if (any(ok)) c(DecoderOnly = mean(decoder[ok]), NeuralODE = mean(neural[ok])) else
          c(DecoderOnly = NA_real_, NeuralODE = NA_real_)
      }, c(DecoderOnly = 0., NeuralODE = 0.)))
      if (any(is.finite(curves))) {
        graphics::matplot(seq_len(h), curves, type = "b", pch = 1, lty = 1,
          xlab = "Forecast horizon", ylab = "Mean matched final skill", main = key)
        graphics::abline(v = data$training_horizon[[1L]], lty = 3)
        graphics::legend("bottomleft", colnames(curves), col = seq_len(ncol(curves)), lty = 1, bty = "n")
      } else { graphics::plot.new(); graphics::title("No matched finite final skill observations") }
      ok <- is.finite(pairs$SkillSummary_Decoder) & is.finite(pairs$SkillSummary_NeuralODE)
      if (any(ok)) {
        graphics::plot(pairs$SkillSummary_Decoder[ok], pairs$SkillSummary_NeuralODE[ok],
          xlab = "Decoder final skill", ylab = "NeuralODE final skill", main = "Matched final checkpoints")
        graphics::abline(0, 1, lty = 2)
      } else { graphics::plot.new(); graphics::title("No matched finite final pairs") }
    }, finally = grDevices::dev.off())
  }
  invisible(report)
}
