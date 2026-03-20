.ndm_condition_message <- function(x) {
  cond <- attr(x, "condition")
  if (!is.null(cond)) {
    return(conditionMessage(cond))
  }
  as.character(x)
}

.ndm_numeric_summary <- function(x, np = NULL) {
  value <- try(
    if (is.null(np)) {
      x
    } else {
      reticulate::py_to_r(np$asarray(x))
    },
    silent = TRUE
  )
  if (inherits(value, "try-error")) {
    return(list(
      error = .ndm_condition_message(value),
      dims = integer(),
      length = 0L,
      finite_fraction = NA_real_,
      n_nonfinite = NA_integer_,
      n_nan = NA_integer_,
      n_inf = NA_integer_,
      min = NA_real_,
      max = NA_real_,
      mean = NA_real_,
      first_nonfinite_index = integer()
    ))
  }

  value_num <- suppressWarnings(as.numeric(value))
  dims <- dim(value)
  if (is.null(dims)) {
    dims <- as.integer(length(value_num))
  } else {
    dims <- as.integer(dims)
  }

  finite <- is.finite(value_num)
  first_nonfinite_index <- integer()
  if (length(value_num) > 0L && any(!finite)) {
    first_bad <- which(!finite)[[1L]]
    if (is.null(dim(value))) {
      first_nonfinite_index <- as.integer(first_bad)
    } else {
      first_nonfinite_index <- as.integer(arrayInd(first_bad, .dim = dim(value)))
    }
  }

  finite_values <- value_num[finite]
  list(
    dims = dims,
    length = length(value_num),
    finite_fraction = if (length(value_num) == 0L) 1 else mean(finite),
    n_nonfinite = sum(!finite),
    n_nan = sum(is.nan(value_num)),
    n_inf = sum(is.infinite(value_num)),
    min = if (length(finite_values) > 0L) min(finite_values) else NA_real_,
    max = if (length(finite_values) > 0L) max(finite_values) else NA_real_,
    mean = if (length(finite_values) > 0L) mean(finite_values) else NA_real_,
    first_nonfinite_index = first_nonfinite_index
  )
}

.ndm_first_nonfinite_name <- function(summaries) {
  if (is.null(names(summaries))) {
    return(NA_character_)
  }
  for (name in names(summaries)) {
    summary <- summaries[[name]]
    if (!is.null(summary$error)) {
      return(name)
    }
    finite_fraction <- summary$finite_fraction
    if (is.na(finite_fraction) || finite_fraction < 1) {
      return(name)
    }
  }
  NA_character_
}

.ndm_plot_log_series_safe <- function(values,
                                      main = "",
                                      empty_label = "no finite positive values",
                                      ...) {
  values_num <- suppressWarnings(as.numeric(values))
  plot_values <- values_num[is.finite(values_num) & values_num > 0]
  if (length(plot_values) > 0L) {
    plot(plot_values, log = "y", main = main, ...)
  } else {
    plot.new()
    title(main = main)
    text(0.5, 0.5, labels = empty_label, cex = 0.8)
  }
  invisible(plot_values)
}

.ndm_write_nonfinite_report <- function(report,
                                        holder_folder,
                                        outer_iteration,
                                        iteration) {
  dir.create(holder_folder, recursive = TRUE, showWarnings = FALSE)
  outer_label <- if (length(outer_iteration) == 0L || is.na(outer_iteration[[1L]])) {
    "NA"
  } else {
    as.character(as.integer(outer_iteration[[1L]]))
  }
  iteration_label <- if (length(iteration) == 0L || is.na(iteration[[1L]])) {
    "NA"
  } else {
    as.character(as.integer(iteration[[1L]]))
  }
  report_path <- file.path(
    holder_folder,
    sprintf("nonfinite_outer%s_i%s.rds", outer_label, iteration_label)
  )
  saveRDS(report, report_path)
  .ndm_normalize_path(report_path, must_work = TRUE)
}
