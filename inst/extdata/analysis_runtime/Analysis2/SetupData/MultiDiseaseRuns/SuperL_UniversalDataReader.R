# Content of UniversalDataReader.R
if(dataFormat == "Tycho"){
  ReSaveRawData <- TRUE
  if( ReSaveRawData ){
    
    # CDC (1982 onwards; reports); 
    # CDC wonder (1993 onwards); 
    # 1950-1981 (public health reports)
    
    # load in tycho data 
    options(error = NULL)
    #dataloc <- "./Data/MultiDiseaseRuns/TB-USA/US.427099000/"
    dataloc <- "./Data/MultiDiseaseRuns/TB-USA-2"
    csvs_ <- (csvs_ <- list.files(dataloc))[grepl(csvs_,pattern="\\.csv")]
    
    raw_input_data <- c()
    for(csv_ in csvs_){
      raw_input_data <- rbind(raw_input_data,
                              read.csv(sprintf("%s/%s",dataloc, csv_)))
    }
    
    # define period duration (in weeks)
    raw_input_data$PeriodDurationWeeks <- difftime(as.Date(raw_input_data$PeriodEndDate),
                    as.Date(raw_input_data$PeriodStartDate),units = "weeks")
    # View(raw_input_data[order(raw_input_data$PeriodDurationWeeks,decreasing=T),])
    # plot(raw_input_data$PeriodDurationWeeks)
    
    library(dplyr); library(lubridate)
    # load in population data 
    pop_file <- "./Data/MultiDiseaseRuns/population_state_year.csv"
    #pop_url  <- "https://raw.githubusercontent.com/JoshData/historical-state-population-csv/primary/historical_state_population_by_year.csv"
    #if (!file.exists(pop_file)) {
    #  dir.create(dirname(pop_file), recursive = TRUE, showWarnings = FALSE)
    #  download.file(pop_url, destfile = pop_file, quiet = TRUE, mode = "wb")
    #}
    population_df <- read.csv(pop_file, col.names = c("state_abbr", "year", "population")) %>%
      dplyr::mutate(state_abbr = as.character(state_abbr),
                    year        = as.integer(year),
                    population  = as.numeric(population))
    
    ## ---- UNIVERSAL TRANSLATION:  Project Tycho  -------------------------------
    raw_input_data
    tycho_df <- raw_input_data %>%
      mutate(
        location_id   = Admin1ISO,              # “US‑WI”
        location_name = Admin1Name,             # “WISCONSIN”
        location_id_numeric =
          as.integer(factor(location_id, levels = sort(unique(location_id)))) - 1L,
        date_start = as.Date(PeriodStartDate),
        week_id    = as.integer(difftime(date_start,
                                         min(date_start, na.rm = TRUE),
                                         units = "weeks")),
        time_id  = week_id,
        targetTime_id = time_id,
        year       = lubridate::year(date_start),     # <‑‑ NEW
        CountValue = as.numeric(CountValue)
      ) %>%
      select(location_id, location_id_numeric, location_name,
             date_start, year, time_id, targetTime_id, CountValue) %>%
      arrange(location_id, time_id)
    
    ## Pull the two‑letter state abbrev from the Admin1ISO code
    tycho_df <- tycho_df %>% 
                  mutate(state_abbr = sub("^US-", "", location_id))
    
    ## Merge and fill the (rare) missing values using last‑observation‑carried‑forward
    tycho_df <- tycho_df %>%
      left_join(population_df, by = c("state_abbr", "year")) %>%
      group_by(state_abbr) %>%
      tidyr::fill(population, .direction = "down") %>%   # if early years missing
      tidyr::fill(population, .direction = "up") %>%     # if late years missing
      ungroup()
    
    ## Convert counts to fractions (and an interpretable per‑100 k rate)
    tycho_df <- tycho_df %>%
      mutate(
        CountFraction   = CountValue / population,
        CasesPerPop    = CountFraction * 1e5
      )
    
    # Hand off the population‑standardised series to downstream code
    truth_df_red  <- tycho_df %>%
      select(location_id, location_id_numeric, location_name,
             time_id, targetTime_id,
             CountFraction, CasesPerPop)            %>%
      rename(CountValue = CountFraction)
    
    # sanity check -- should be ≤ 1 everywhere (real outbreaks rarely hit that, obviously)
    summary(truth_df_red$CasesPerPop)
    
    # Quick visual check for one state
    if(FALSE){
      #View(truth_df_red[truth_df_red$location_name=="TEXAS",])
      #View(raw_input_data[raw_input_data$Admin1Name=="TEXAS",])
      #STATE <- "TEXAS"
      STATE <- "WISCONSIN"
      
      sort(tapply(raw_input_data$PeriodDurationWeeks, raw_input_data$Admin1Name,sd))
      
      plot(table(as.Date(raw_input_data$PeriodStartDate)))
      plot(as.Date(raw_input_data[raw_input_data$Admin1Name==STATE,]$PeriodStartDate),
           raw_input_data[raw_input_data$Admin1Name==STATE,]$CountValue)
      
      library(ggplot2)
      filter(truth_df_red, location_name == STATE) |>
        ggplot(aes(x = time_id, y = CountValue * 1e5)) +
        geom_line() +
        labs(y = "TB cases per 100,000", x = "Epi-time index")
    }
    
    # Collapse duplicates
    truth_df_red <- truth_df_red %>%
      group_by(location_id, location_id_numeric, location_name,
               time_id, targetTime_id) %>%
      summarise(CountValue = sum(CountValue, na.rm = TRUE), .groups = "drop")
    
    # write to disk
    if(ReSaveRawData){
      write.csv(truth_df_red, "./Data/MultiDiseaseRuns/Tycho_processed.csv", row.names = FALSE)
    }
  }
  
  # 3. Expose universally‑named objects expected by the downstream scripts -----
  truth_df_red           <- truth_df_red           # Targets (Y)
  input_df_red           <- truth_df_red           # Past values as covariates (X)
  true_value_names       <- "CountValue"
  all_true_value_names   <- true_value_names   # keeps interface identical to multi‑outcome cases
  
  # Autoregressive covariate set – past observations only
  dataInputs_colnames_past    <- "CountValue"

  # Union needed later in normalisation step
  dataInputs_colnames         <- c(dataInputs_colnames_past,
                                   dataInputs_colnames_future)
  
  # How many outcomes are we forecasting?
  nOutcomes <- length(true_value_names)
  nPlaces <- length(unique(truth_df_red$location_id))
}
if(dataFormat == "WHO"){
  my_data <- read.csv('./Data/MultiDiseaseRuns/DiseasePool/incidence-of-tuberculosis-sdgs.csv')
  
  library(dplyr); library(lubridate); library(tidyr)
  who_df <- my_data %>%
    # Standardize column names
    dplyr::rename(
      location_name = Entity,
      location_id   = Code,
      year          = Year,
      CasesPerPop  = Estimated.incidence.of.all.forms.of.tuberculosis
    ) %>%
    # Types & minimal cleaning
    mutate(
      year         = as.integer(year),
      CasesPerPop = as.numeric(CasesPerPop),
      # Fill blank/NA codes (e.g., continents/regions) with the entity name
      location_id  = dplyr::if_else(is.na(location_id) | trimws(location_id) == "",
                                    location_name, location_id),
      date_start   = as.Date(paste0(year, "-01-01")),
      theweek_id      = as.integer(difftime(date_start,
                                         min(date_start, na.rm = TRUE),
                                         units = "weeks")),
      time_id      = (year - min(my_data$Year)), # 0 indexed 
      targetTime_id = time_id,
      # Convert WHO incidence (per pop size) to a population fraction to match Tycho
      CountFraction = CasesPerPop / 1e3
    ) %>%
    mutate(
      location_id_numeric =
        as.integer(factor(location_id, levels = sort(unique(location_id)))) - 1L
    ) %>%
    arrange(location_id, time_id)
  
  ## If you want *countries only* and to drop regional aggregates, uncomment:
  ## who_df <- who_df %>% dplyr::filter(nchar(location_id) == 3)
  
  # Hand off in the same shape Tycho produces
  truth_df_red <- who_df %>%
    select(location_id, location_id_numeric, location_name,
           time_id, targetTime_id, CountFraction) %>%
    rename(CountValue = CountFraction) %>%
    group_by(location_id, location_id_numeric, location_name,
             time_id, targetTime_id) %>%
    summarise(CountValue = sum(CountValue, na.rm = TRUE), .groups = "drop")
  
  # Optional write (kept parallel to Tycho branch)
  # write.csv(truth_df_red, "./Data/MultiDiseaseRuns/WHO_processed.csv", row.names = FALSE)
  
  # drop 
  truth_df_red <- truth_df_red[!is.na(truth_df_red$CountValue),]
  
  # 3. Expose universally‑named objects expected by the downstream scripts -----
  truth_df_red         <- truth_df_red           # Targets (Y)
  input_df_red         <- truth_df_red           # Past values as covariates (X)
  truth_df_red$Covariate1 <- truth_df_red$CountValue
  true_value_names     <- "CountValue"
  all_true_value_names <- true_value_names
  
  # Autoregressive covariate set – past observations only
  dataInputs_colnames_future <- NULL 
  dataInputs_colnames_past <- c("CountValue","Covariate1")
  # Union needed later in normalisation step
  dataInputs_colnames <- c(dataInputs_colnames_past, dataInputs_colnames_future)
  outcome_metric <- "CountValue"
  
  # How many outcomes are we forecasting?
  nOutcomes <- length(true_value_names)
  nPlaces <- length(unique(truth_df_red$location_id))
  
  hist(truth_df_red[outcome_metric][[1]])

  # check for NAs
  colMeans(truth_df_red[,dataInputs_colnames_past],na.rm=T)
  colMeans(truth_df_red[,dataInputs_colnames_past],na.rm=T)
  #head(truth_df_red)
}
if(dataFormat == "IHME"){
  ihme_candidates <- c(
    "./Data/MultiDiseaseRuns/IHMEData/IHME-GBD_2021_DATA-ea2ad67b-1/IHME-GBD_2021_DATA-ea2ad67b-1.csv",
    "./Data/MultiDiseaseRuns/DiseasePool/IHMEData/IHME-GBD_2021_DATA-ea2ad67b-1/IHME-GBD_2021_DATA-ea2ad67b-1.csv"
  )
  ihme_file <- ihme_candidates[file.exists(ihme_candidates)][1]
  if (is.na(ihme_file) || !nzchar(ihme_file)) {
    stop(
      "IHME data file not found. Checked: ",
      paste(ihme_candidates, collapse = ", "),
      call. = FALSE
    )
  }
  my_data <- read.csv(ihme_file)
  table(my_data$year); table(my_data$metric_name)
  table(my_data$cause_name)
  my_data <- my_data[which(my_data$sex_name=="Both" & 
                             my_data$metric_name == "Rate" & 
                             my_data$cause_name == "HIV/AIDS and sexually transmitted infections" & 
                             my_data$age_name == "All ages"),]
  #View(my_data)
  #head(my_data)
  #hist(my_data$val)
  #hist(my_data$upper)
  
  library(dplyr); library(lubridate); library(tidyr)
  
  ## ---- IHME → universal format ---------------------------------------------
    # Choose a single measure: prefer Prevalence, else Incidence, else first available.
  if (exists("desired_measure")) {
    my_data <- dplyr::filter(my_data, measure_name == desired_measure)
  } else {
    preferred_measures <- c("Prevalence")
    chosen_measure <- intersect(preferred_measures, unique(my_data$measure_name))
    if (length(chosen_measure) == 0) chosen_measure <- unique(my_data$measure_name)[1]
    my_data <- dplyr::filter(my_data, measure_name == chosen_measure[1])
  }
  
  # Convert IHME metrics to a population fraction suitable for downstream use.
  ihme_df <- my_data %>%
    mutate(
      year       = as.integer(year),
      date_start = as.Date(paste0(year, "-01-01")),
      CountFraction_tmp = dplyr::case_when(
        # IHME "Rate" is per 100,000 population
        metric_name == "Rate"    ~ as.numeric(val) / 1e5,
        # IHME "Percent": robust to either 0–1 or 0–100 encoding
        metric_name == "Percent" ~ dplyr::if_else(as.numeric(val) > 1,
                                                  as.numeric(val) / 100,
                                                  as.numeric(val)),
        TRUE                     ~ NA_real_
      ),
      # prefer "Rate" over "Percent" if both are available in a given year/location
      metric_rank = dplyr::case_when(
        metric_name == "Rate"    ~ 1L,
        metric_name == "Percent" ~ 2L,
        TRUE                     ~ 99L
      )
    ) %>%
    filter(!is.na(CountFraction_tmp)) %>%
    arrange(location_id, year, metric_rank) %>%
    group_by(location_id, location_name, year) %>%
    slice(1L) %>%
    ungroup() %>%
    mutate(
      CountFraction = CountFraction_tmp,
      time_id       = year - min(year, na.rm = TRUE),  # 0-indexed
      targetTime_id = time_id,
      location_id   = as.character(location_id),
      location_id_numeric =
        as.integer(factor(location_id, levels = sort(unique(location_id)))) - 1L
    ) %>%
    arrange(location_id, time_id)
  
  # Hand off in the same shape as the other branches
  truth_df_red <- ihme_df %>%
    select(location_id, location_id_numeric, location_name,
           time_id, targetTime_id, CountFraction) %>%
    rename(CountValue = CountFraction) %>%
    group_by(location_id, location_id_numeric, location_name,
             time_id, targetTime_id) %>%
    summarise(CountValue = sum(CountValue, na.rm = TRUE), .groups = "drop")
  
  # Drop any remaining missing rows
  truth_df_red <- truth_df_red[!is.na(truth_df_red$CountValue), ]
  
  # 3. Expose universally‑named objects expected by the downstream scripts -----
  input_df_red             <- truth_df_red
  truth_df_red$Covariate1  <- truth_df_red$CountValue
  true_value_names         <- "CountValue"
  all_true_value_names     <- true_value_names
  dataInputs_colnames_future <- NULL
  dataInputs_colnames_past   <- c("CountValue","Covariate1")
  dataInputs_colnames        <- c(dataInputs_colnames_past, dataInputs_colnames_future)
  outcome_metric             <- "CountValue"
  
  # How many outcomes/places?
  nOutcomes <- length(true_value_names)
  nPlaces   <- length(unique(truth_df_red$location_id))
}
# universal sanity checks 
if(FALSE){ 
  truth_df_red_ <- truth_df_red[which(truth_df_red$location_id %in% 
                                sample(unique(truth_df_red$location_id),1)),]
  plot(truth_df_red_$targetTime_id,
       truth_df_red_$CountValue)
}


# IN PROGRESS: 
# Working on "Universal Data Reader" to take 
# data from various orgs (Tycho, etc.) and convert to format for our models 
