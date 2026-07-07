.ndm_nsgd_default_grid_file <- function(mode,
                                        project_root,
                                        analysis_name) {
  file.path(
    project_root,
    "Data",
    "RunGrids",
    switch(
      mode,
      sim = "SimGrids",
      real = "RealGrids",
      multidisease = "RealGrids",
      stop("Unsupported nSGD calibration mode: ", mode, call. = FALSE)
    ),
    switch(
      mode,
      sim = sprintf("SimGrid_%s.csv", analysis_name),
      real = sprintf("RealGrid_%s.csv", analysis_name),
      multidisease = sprintf("RealGrid_%s.csv", analysis_name)
    )
  )
}

.ndm_nsgd_has_multidisease_signal <- function(analysis_name = NULL,
                                              grid_file = NULL) {
  signals <- c(
    basename(as.character(grid_file %||% "")),
    as.character(analysis_name %||% "")
  )
  any(grepl("MULTIDISEASE", signals, ignore.case = TRUE))
}

.ndm_nsgd_resolve_active_grid_file <- function(mode,
                                               project_root,
                                               analysis_name,
                                               grid_file = NULL) {
  candidate <- grid_file %||% .ndm_nsgd_default_grid_file(
    mode = mode,
    project_root = project_root,
    analysis_name = analysis_name
  )
  if (is.null(candidate) || !nzchar(candidate)) {
    return(NULL)
  }
  candidate <- .ndm_path_join_if_relative(project_root, candidate)
  .ndm_normalize_path(candidate, must_work = FALSE)
}

.ndm_nsgd_candidate_grid_files <- function(mode,
                                           project_root,
                                           analysis_name,
                                           active_grid_file = NULL) {
  sim_dir <- file.path(project_root, "Data", "RunGrids", "SimGrids")
  real_dir <- file.path(project_root, "Data", "RunGrids", "RealGrids")

  files <- switch(
    mode,
    sim = {
      if (dir.exists(sim_dir)) {
        Sys.glob(file.path(sim_dir, "*.csv"))
      } else {
        character()
      }
    },
    real = {
      if (!dir.exists(real_dir)) {
        character()
      } else {
        real_files <- Sys.glob(file.path(real_dir, "*.csv"))
        real_files[!grepl("MULTIDISEASE", basename(real_files), ignore.case = TRUE)]
      }
    },
    multidisease = {
      multidisease_scope <- .ndm_nsgd_has_multidisease_signal(
        analysis_name = analysis_name,
        grid_file = active_grid_file
      )
      if (!multidisease_scope) {
        character()
      } else if (!dir.exists(real_dir)) {
        character()
      } else {
        real_files <- Sys.glob(file.path(real_dir, "*.csv"))
        real_files[grepl("MULTIDISEASE", basename(real_files), ignore.case = TRUE)]
      }
    },
    stop("Unsupported nSGD calibration mode: ", mode, call. = FALSE)
  )

  if (!is.null(active_grid_file) && file.exists(active_grid_file)) {
    files <- c(active_grid_file, files)
  }

  unique(vapply(
    files[file.exists(files)],
    .ndm_normalize_path,
    character(1),
    must_work = TRUE
  ))
}

.ndm_nsgd_candidate_scope <- function(mode,
                                      analysis_name,
                                      active_grid_file,
                                      active_grid_present,
                                      candidate_grid_files) {
  if (mode == "sim") {
    return("mode_folder_sim")
  }
  if (mode == "real") {
    return("mode_folder_real")
  }
  if (.ndm_nsgd_has_multidisease_signal(
    analysis_name = analysis_name,
    grid_file = active_grid_file
  )) {
    return("mode_folder_multidisease")
  }
  if (isTRUE(active_grid_present) || length(candidate_grid_files) > 0L) {
    return("active_grid_only_multidisease")
  }
  "fallback_multidisease"
}

.ndm_nsgd_extract_max_samples <- function(x,
                                          source_label) {
  if (is.null(x) || !"nSamplesTrain" %in% names(x)) {
    return(NULL)
  }

  samples <- suppressWarnings(as.integer(x$nSamplesTrain))
  samples <- samples[!is.na(samples)]
  if (length(samples) == 0L) {
    return(NULL)
  }

  data.frame(
    source = as.character(source_label),
    max_nSamplesTrain = max(samples),
    stringsAsFactors = FALSE
  )
}

.ndm_resolve_nsgd_calibration <- function(mode,
                                          project_root,
                                          analysis_name,
                                          n_epoches_max,
                                          grid = NULL,
                                          grid_file = NULL,
                                          fallback_n_samples_train = NULL,
                                          batch_cap = 32L) {
  project_root <- .ndm_normalize_path(project_root, must_work = TRUE)
  analysis_name <- as.character(analysis_name %||% "")
  n_epoches_max <- as.integer(n_epoches_max)
  batch_cap <- max(1L, as.integer(batch_cap))

  active_grid_file <- .ndm_nsgd_resolve_active_grid_file(
    mode = mode,
    project_root = project_root,
    analysis_name = analysis_name,
    grid_file = grid_file
  )
  candidate_grid_files <- .ndm_nsgd_candidate_grid_files(
    mode = mode,
    project_root = project_root,
    analysis_name = analysis_name,
    active_grid_file = active_grid_file
  )

  candidate_sources <- list()
  if (!is.null(grid)) {
    candidate_sources[[length(candidate_sources) + 1L]] <- .ndm_nsgd_extract_max_samples(
      as.data.frame(grid, stringsAsFactors = FALSE),
      source_label = "<active_grid>"
    )
  }

  for (grid_path in candidate_grid_files) {
    grid_dt <- as.data.frame(data.table::fread(grid_path), stringsAsFactors = FALSE)
    candidate_sources[[length(candidate_sources) + 1L]] <- .ndm_nsgd_extract_max_samples(
      grid_dt,
      source_label = grid_path
    )
  }

  candidate_sources <- Filter(Negate(is.null), candidate_sources)
  if (length(candidate_sources) == 0L && !is.null(fallback_n_samples_train)) {
    fallback_samples <- suppressWarnings(as.integer(fallback_n_samples_train))
    fallback_samples <- fallback_samples[!is.na(fallback_samples)]
    if (length(fallback_samples) > 0L) {
      candidate_sources[[1L]] <- data.frame(
        source = "<fallback_nSamplesTrain>",
        max_nSamplesTrain = max(fallback_samples),
        stringsAsFactors = FALSE
      )
    }
  }

  if (length(candidate_sources) == 0L) {
    stop(
      "Unable to resolve an nSGD calibration anchor because no candidate grid with `nSamplesTrain` was found for mode `",
      mode,
      "`.",
      call. = FALSE
    )
  }

  candidate_sources <- unique(do.call(rbind, candidate_sources))
  anchor_max_n_samples_train <- max(candidate_sources$max_nSamplesTrain)
  anchor_n_batch <- min(batch_cap, anchor_max_n_samples_train)
  resolved_n_sgd <- max(
    1L,
    as.integer(round(n_epoches_max * (anchor_max_n_samples_train / anchor_n_batch)))
  )

  list(
    policy = "anchored_mode_max_nSamplesTrain",
    mode = as.character(mode),
    anchor_scope = .ndm_nsgd_candidate_scope(
      mode = mode,
      analysis_name = analysis_name,
      active_grid_file = active_grid_file,
      active_grid_present = !is.null(grid),
      candidate_grid_files = candidate_grid_files
    ),
    candidate_sources = candidate_sources,
    active_grid_file = active_grid_file,
    anchor_max_n_samples_train = as.integer(anchor_max_n_samples_train),
    anchor_n_batch = as.integer(anchor_n_batch),
    n_epoches_max = as.integer(n_epoches_max),
    resolved_n_sgd = as.integer(resolved_n_sgd)
  )
}

.ndm_nsgd_calibration_globals <- function(calibration) {
  list(
    nSGDPolicy = as.character(calibration$policy),
    nSGDAnchorScope = as.character(calibration$anchor_scope),
    nSGDAnchorMaxSamplesTrain = as.integer(calibration$anchor_max_n_samples_train),
    nSGDAnchorBatch = as.integer(calibration$anchor_n_batch)
  )
}

.ndm_nsgd_calibration_message <- function(mode,
                                          calibration) {
  sprintf(
    "Resolved %s nSGD policy [%s]: anchor max nSamplesTrain=%s, anchor batch=%s, nSGD=%s, scope=%s",
    mode,
    calibration$policy,
    calibration$anchor_max_n_samples_train,
    calibration$anchor_n_batch,
    calibration$resolved_n_sgd,
    calibration$anchor_scope
  )
}

.ndm_small_run_n_checkpoints <- function(n_samples_train,
                                         n_sgd,
                                         default = 10L) {
  n_samples <- max(1L, as.integer(round(as.numeric(n_samples_train))))
  max_sgd <- max(1L, as.integer(round(as.numeric(n_sgd))))
  if (n_samples < 32L) {
    return(max(1L, min(3L, max_sgd)))
  }
  max(1L, min(as.integer(round(as.numeric(default))), max_sgd))
}
