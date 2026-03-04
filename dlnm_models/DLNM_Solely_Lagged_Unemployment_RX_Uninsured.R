#################################################################################################################################################################
### 2/24/26, EB: Now fitting a DLNM with 3 cross-bases: one for Unemployment, one for RX, and one for Uninsured. I'll start with a linear exposure response here as well.


library(arrow)
library(dlnm)
library(splines)
library(mgcv)
library(data.table)
library(ggplot2)
library(dplyr)

############################
####### Data import ########
############################
wide_to_long <- function(
    file,
    value_name,
    fips_col = "FIPS",
    drop_bad_fips=TRUE
) {
  dt <- fread(file)
  
  # Ensure FIPS is character and zero-padded
  # dt[, (fips_col) := sprintf("%05s", get(fips_col))]
  #  dt[, (fips_col) := sprintf("%05d", as.integer(trimws(get(fips_col))))]
  dt[, fips_raw := trimws(get(fips_col))]
  
  if (drop_bad_fips) {
    # Drop missing/blank FIPS
    dt <- dt[!is.na(fips_raw) & fips_raw != ""]
  }
  
  dt[, fips_int := suppressWarnings(as.integer(fips_raw))]
  
  if (drop_bad_fips) {
    # Drop rows where conversion failed (non-numeric garbage)
    dt <- dt[!is.na(fips_int)]
  } else {
    # Fail loudly if you didn't ask to drop
    if (any(is.na(dt$fips_int))) {
      bad <- unique(dt[is.na(fips_int), fips_raw])[1:min(10, uniqueN(dt[is.na(fips_int), fips_raw]))]
      stop("Invalid FIPS encountered in ", file, ". Examples: ", paste(bad, collapse = ", "))
    }
  }
  
  dt[, (fips_col) := sprintf("%05d", fips_int)]
  dt[, c("fips_raw", "fips_int") := NULL]
  
  # Identify columns that START with a 4-digit year
  year_cols <- grep("^\\d{4}", names(dt), value = TRUE)
  
  if (length(year_cols) == 0) {
    stop("No year columns detected in file: ", file)
  }
  
  # Melt to long format
  long_dt <- melt(
    dt,
    id.vars = fips_col,
    measure.vars = year_cols,
    variable.name = "Year_raw",
    value.name = value_name
  )
  
  # Extract the 4-digit year from column names
  long_dt[, Year := as.integer(sub("^([0-9]{4}).*", "\\1", Year_raw))]
  
  # Drop raw column name
  long_dt[, Year_raw := NULL]
  
  return(long_dt)
}


data_loader <- function()
{
  mort <- wide_to_long(
    file = "data/Processed/Mortality/Mortality_final_rates.csv",
    value_name = "MR"
  )
  
  rx <- wide_to_long(
    file = "data/Processed/Prescriptions/Prescription_dispensing_rates.csv",
    value_name = "DR"
  )
  
  # pop <- wide_to_long(
  #   file = "data/Processed/Population/county_population_2010_2022.csv",
  #   value_name = "POP",
  #   drop_bad_fips=TRUE
  # )
  
  unemp <- wide_to_long(
    file = "data/Processed/Unemployment/Unemployment_rates_CT_fixed.csv",
    value_name = "Unemployment"
  )
  
  uninsured <- read_parquet("data/Processed/Uninsured/SAHIE_Uninsured_rates_CT_fixed.parquet")
  setDT(uninsured)
  # Standardize join keys to match df
  setnames(uninsured, old = "year", new = "Year")
  
  
  # Now construct SVI long-format df
  svi_files <- list.files(
    path = "data/Processed/SVI/",
    pattern = "\\.csv$",
    full.names = TRUE
  )
  svi_files <- svi_files[
    basename(svi_files) != "Unemployment_final_rates.csv"
  ]
  
  svi_long_list <- lapply(svi_files, function(f) {
    
    # Infer variable name from filename
    var_name <- tools::file_path_sans_ext(basename(f))
    
    wide_to_long(
      file = f,
      value_name = var_name
    )
  })
  
  svi <- Reduce(
    function(x, y) merge(x, y, by = c("FIPS", "Year"), all = TRUE),
    svi_long_list
  )
  
  # str(svi)
  # summary(svi)
  
  # Now combine into a single dataframe
  df <- mort
  df <- merge(df, rx, by = c("FIPS", "Year"), all = FALSE)
  # df <- merge(df, pop, by = c("FIPS", "Year"), all = FALSE)
  df <- merge(df, svi, by = c("FIPS", "Year"), all = FALSE)
  df <- merge(df, unemp, by = c("FIPS", "Year"), all = FALSE)
  df <- merge(df, uninsured, by = c("FIPS", "Year"), all = FALSE)
  
  ### Setting missing RX rates to 0 for now (they're coded as -9)
  df[DR < 0, DR := 0]
  
  setorder(df, FIPS, Year)
  # setnames(df, "DR", "RX")
  
  summary(df)
  
  return(df)
}

data <- data_loader()


eps <- 0.01
data <- data %>%
  arrange(FIPS, Year) %>%
  group_by(FIPS) %>%
  mutate(
    logMR = log(MR + eps),
    logMR_lead = lead(logMR, n = 1)
  ) %>%
  ungroup()


############################
####### DLNM Block #########
############################
lag_max <- 4
lag_df <- 3
exposure_df <- 2

### Filtering to just rows where we have data
data_model <- data %>%
  filter(!is.na(logMR_lead))

### Constructing cross-bases
cb_unemp <- crossbasis(
  data_model$Unemployment,
  lag = lag_max,
  argvar = list(fun = "lin"),        # linear exposure-response
  # argvar = list(fun = "ns", df = exposure_df),
  arglag = list(fun = "ns", df = lag_df), # smooth lag basis
  group = "FIPS"
)

cb_rx <- crossbasis(
  data_model$DR,
  lag = lag_max,
  argvar = list(fun = "lin"),
  arglag = list(fun = "ns", df = lag_df),
  group = "FIPS"
)


cb_unins <- crossbasis(
  data_model$uninsured_rate,
  lag = lag_max,
  argvar = list(fun = "lin"),
  arglag = list(fun = "ns", df = lag_df),
  group = "FIPS"
)


### Fitting simple model, just lagged Unemployment and year fixed-effects
model_joint_unemp_rx_unins_dlnm <- glm(
  logMR_lead ~ cb_unemp + cb_rx + cb_unins + factor(Year),
  family = gaussian(),
  data = data_model
)

summary(model_joint_unemp_rx_unins_dlnm)




#################
### Plotting


pred_unemp <- crosspred(
  cb_unemp,
  model_joint_unemp_rx_unins_dlnm,
  cen = median(data$Unemployment, na.rm = TRUE)
)
### Get approximate quantiles
v25_unemp <- as.numeric(quantile(data_model$Unemployment, 0.25, na.rm = TRUE))
v75_unemp <- as.numeric(quantile(data_model$Unemployment, 0.75, na.rm = TRUE))

### Snap them to the nearest values crosspred actually predicted at
### plot needs the actual gridpoints crosspred used, so we grab the 
### closest one to make the lag-specific plots for our quantiles
v25_gridpoint_unemp <- pred_unemp$predvar[ which.min(abs(pred_unemp$predvar - v25_unemp)) ]
v75_gridpoint_unemp <- pred_unemp$predvar[ which.min(abs(pred_unemp$predvar - v75_unemp)) ]


### Unemployment plots
plot(
  pred_unemp,
  var = v25_gridpoint_unemp,
  type = "l",
  ci = "area",
  col = "black",
  lwd = 2,
  xlab = "Lag (Years)",
  ylab = "Change in log(Mortality Rate) at t+1",
  main = "Lag-Specific Effect of Unemployment (25th%)\non Next-Year Log Mortality",
  cex.main = 1
)

abline(h = 0, lty = 2)

plot(
  pred_unemp,
  var = v75_gridpoint_unemp,
  type = "l",
  ci = "area",
  col = "black",
  lwd = 2,
  xlab = "Lag (Years)",
  ylab = "Change in log(Mortality Rate) at t+1",
  main = "Lag-Specific Effect of Unemployment (75th%)\non Next-Year Log Mortality",
  cex.main = 1
)

abline(h = 0, lty = 2)


plot(
  pred_unemp,
  "overall",
  ci = "area",
  col = "black",
  lwd = 2,
  xlab = "Unemployment Rate (%)",
  ylab = "Cumulative Change in log(Mortality Rate)",
  main = paste("Cumulative Effect of Unemployment\nAcross Lags 0–", lag_max, " on Next-Year Mortality"),
  cex.main = 1
)

abline(h = 0, lty = 2)

plot(
  pred_unemp,
  "overall",
  ci = "area",
  col = "black",
  lwd = 2,
  xlab = "Unemployment Rate (%)",
  ylab = "Cumulative Relative Risk (RR)",
  main = paste("Cumulative Relative Risk of Mortality\nAcross Lags 0–", lag_max),
  exp = TRUE
)

abline(h = 1, lty = 2)



pred_rx <- crosspred(
  cb_rx,
  model_joint_unemp_rx_unins_dlnm,
  cen = median(data$DR, na.rm = TRUE)
)
### Get approximate quantiles
v25_rx <- as.numeric(quantile(data_model$DR, 0.25, na.rm = TRUE))
v75_rx <- as.numeric(quantile(data_model$DR, 0.75, na.rm = TRUE))

### Snap them to the nearest values crosspred actually predicted at
### plot needs the actual gridpoints crosspred used, so we grab the 
### closest one to make the lag-specific plots for our quantiles
v25_gridpoint_rx <- pred_rx$predvar[ which.min(abs(pred_rx$predvar - v25_rx)) ]
v75_gridpoint_rx <- pred_rx$predvar[ which.min(abs(pred_rx$predvar - v75_rx)) ]



### RX Dispensing Rate plots
plot(
  pred_rx,
  var = v25_gridpoint_rx,
  type = "l",
  ci = "area",
  col = "black",
  lwd = 2,
  xlab = "Lag (Years)",
  ylab = "Change in log(Mortality Rate) at t+1",
  main = "Lag-Specific Effect of RX (25th%)\non Next-Year Log Mortality",
  cex.main = 1
)

abline(h = 0, lty = 2)

plot(
  pred_rx,
  var = v75_gridpoint_rx,
  type = "l",
  ci = "area",
  col = "black",
  lwd = 2,
  xlab = "Lag (Years)",
  ylab = "Change in log(Mortality Rate) at t+1",
  main = "Lag-Specific Effect of RX (75th%)\non Next-Year Log Mortality",
  cex.main = 1
)

abline(h = 0, lty = 2)


plot(
  pred_rx,
  "overall",
  ci = "area",
  col = "black",
  lwd = 2,
  xlab = "RX Dispensing Rate (%)",
  ylab = "Cumulative Change in log(Mortality Rate)",
  main = paste("Cumulative Effect of RX \nAcross Lags 0–", lag_max, " on Next-Year Mortality"),
  cex.main = 1
)

abline(h = 0, lty = 2)

plot(
  pred_rx,
  "overall",
  ci = "area",
  col = "black",
  lwd = 2,
  xlab = "RX Dispensing Rate (%)",
  ylab = "Cumulative Relative Risk (RR)",
  main = paste("Cumulative Relative Risk of Mortality\nAcross Lags 0–", lag_max),
  exp = TRUE
)

abline(h = 1, lty = 2)



pred_unins <- crosspred(
  cb_unins,
  model_joint_unemp_rx_unins_dlnm,
  cen = median(data$uninsured_rate, na.rm = TRUE)
)
### Get approximate quantiles
v25_unins <- as.numeric(quantile(data_model$uninsured_rate, 0.25, na.rm = TRUE))
v75_unins <- as.numeric(quantile(data_model$uninsured_rate, 0.75, na.rm = TRUE))

### Snap them to the nearest values crosspred actually predicted at
### plot needs the actual gridpoints crosspred used, so we grab the 
### closest one to make the lag-specific plots for our quantiles
v25_gridpoint_unins <- pred_unins$predvar[ which.min(abs(pred_unins$predvar - v25_unins)) ]
v75_gridpoint_unins <- pred_unins$predvar[ which.min(abs(pred_unins$predvar - v75_unins)) ]


### Uninsured Rate plots
plot(
  pred_unins,
  var = v25_gridpoint_unins,
  type = "l",
  ci = "area",
  col = "black",
  lwd = 2,
  xlab = "Lag (Years)",
  ylab = "Change in log(Mortality Rate) at t+1",
  main = "Lag-Specific Effect of Uninsured Rate (25th%)\non Next-Year Log Mortality",
  cex.main = 1
)

abline(h = 0, lty = 2)


plot(
  pred_unins,
  var = v75_gridpoint_unins,
  type = "l",
  ci = "area",
  col = "black",
  lwd = 2,
  xlab = "Lag (Years)",
  ylab = "Change in log(Mortality Rate) at t+1",
  main = "Lag-Specific Effect of Uninsured Rate (75th%)\non Next-Year Log Mortality",
  cex.main = 1
)

abline(h = 0, lty = 2)

plot(
  pred_unins,
  "overall",
  ci = "area",
  col = "black",
  lwd = 2,
  xlab = "Uninsured Rate (%)",
  ylab = "Cumulative Change in log(Mortality Rate)",
  main = paste("Cumulative Effect of Uninsured Rate \nAcross Lags 0–", lag_max, "on Next-Year Mortality"),
  cex.main = 1
)

abline(h = 0, lty = 2)

plot(
  pred_unins,
  "overall",
  ci = "area",
  col = "black",
  lwd = 2,
  xlab = "Uninsured Rate (%)",
  ylab = "Cumulative Relative Risk (RR)",
  main = paste("Cumulative Relative Risk of Mortality\nAcross Lags 0–", lag_max),
  exp = TRUE
)

abline(h = 1, lty = 2)








###########################################################################################################
### 2/28/26, EB: Followed along with a Vignette from Gasparrini, and now trying this 3CB DLNM from scratch again


library(arrow)
library(dlnm)
library(splines)
library(mgcv)
library(data.table)
library(ggplot2)
library(dplyr)

############################
####### Data import ########
############################
wide_to_long <- function(
    file,
    value_name,
    fips_col = "FIPS",
    drop_bad_fips=TRUE
) {
  dt <- fread(file)
  
  # Ensure FIPS is character and zero-padded
  # dt[, (fips_col) := sprintf("%05s", get(fips_col))]
  #  dt[, (fips_col) := sprintf("%05d", as.integer(trimws(get(fips_col))))]
  dt[, fips_raw := trimws(get(fips_col))]
  
  if (drop_bad_fips) {
    # Drop missing/blank FIPS
    dt <- dt[!is.na(fips_raw) & fips_raw != ""]
  }
  
  dt[, fips_int := suppressWarnings(as.integer(fips_raw))]
  
  if (drop_bad_fips) {
    # Drop rows where conversion failed (non-numeric garbage)
    dt <- dt[!is.na(fips_int)]
  } else {
    # Fail loudly if you didn't ask to drop
    if (any(is.na(dt$fips_int))) {
      bad <- unique(dt[is.na(fips_int), fips_raw])[1:min(10, uniqueN(dt[is.na(fips_int), fips_raw]))]
      stop("Invalid FIPS encountered in ", file, ". Examples: ", paste(bad, collapse = ", "))
    }
  }
  
  dt[, (fips_col) := sprintf("%05d", fips_int)]
  dt[, c("fips_raw", "fips_int") := NULL]
  
  # Identify columns that START with a 4-digit year
  year_cols <- grep("^\\d{4}", names(dt), value = TRUE)
  
  if (length(year_cols) == 0) {
    stop("No year columns detected in file: ", file)
  }
  
  # Melt to long format
  long_dt <- melt(
    dt,
    id.vars = fips_col,
    measure.vars = year_cols,
    variable.name = "Year_raw",
    value.name = value_name
  )
  
  # Extract the 4-digit year from column names
  long_dt[, Year := as.integer(sub("^([0-9]{4}).*", "\\1", Year_raw))]
  
  # Drop raw column name
  long_dt[, Year_raw := NULL]
  
  return(long_dt)
}


data_loader <- function()
{
  mort <- wide_to_long(
    file = "data/Processed/Mortality/Mortality_final_rates.csv",
    value_name = "MR"
  )
  
  rx <- wide_to_long(
    file = "data/Processed/Prescriptions/Prescription_dispensing_rates.csv",
    value_name = "DR"
  )
  
  # pop <- wide_to_long(
  #   file = "data/Processed/Population/county_population_2010_2022.csv",
  #   value_name = "POP",
  #   drop_bad_fips=TRUE
  # )
  
  unemp <- wide_to_long(
    file = "data/Processed/Unemployment/Unemployment_rates_CT_fixed.csv",
    value_name = "Unemployment"
  )
  
  uninsured <- read_parquet("data/Processed/Uninsured/SAHIE_Uninsured_rates_CT_fixed.parquet")
  setDT(uninsured)
  # Standardize join keys to match df
  setnames(uninsured, old = "year", new = "Year")
  
  
  # Now construct SVI long-format df
  svi_files <- list.files(
    path = "data/Processed/SVI/",
    pattern = "\\.csv$",
    full.names = TRUE
  )
  svi_files <- svi_files[
    basename(svi_files) != "Unemployment_final_rates.csv"
  ]
  
  svi_long_list <- lapply(svi_files, function(f) {
    
    # Infer variable name from filename
    var_name <- tools::file_path_sans_ext(basename(f))
    
    wide_to_long(
      file = f,
      value_name = var_name
    )
  })
  
  svi <- Reduce(
    function(x, y) merge(x, y, by = c("FIPS", "Year"), all = TRUE),
    svi_long_list
  )
  
  # str(svi)
  # summary(svi)
  
  # Now combine into a single dataframe
  df <- mort
  df <- merge(df, rx, by = c("FIPS", "Year"), all = FALSE)
  # df <- merge(df, pop, by = c("FIPS", "Year"), all = FALSE)
  df <- merge(df, svi, by = c("FIPS", "Year"), all = FALSE)
  df <- merge(df, unemp, by = c("FIPS", "Year"), all = FALSE)
  df <- merge(df, uninsured, by = c("FIPS", "Year"), all = FALSE)
  
  ### Setting missing RX rates to 0 for now (they're coded as -9)
  df[DR < 0, DR := 0]
  
  setorder(df, FIPS, Year)
  # setnames(df, "DR", "RX")
  
  summary(df)
  
  return(df)
}

### Data construction and reshaping.
data <- data_loader()

eps <- 0.01

df <- data_loader()
setorder(df, FIPS, Year)

# Define the target variable
df[, logMR := log(MR + eps)]
df[, logMR_lead := shift(logMR, type = "lead", n = 1L), by = FIPS]


### 4 counties don't have all 9 years: 3 in AK and 1 in SD
# count years per county
len_by_fips <- df[, .N, by = FIPS][order(N)]
# show the shortest
# len_by_fips[1:30]
# counties that violate the requirement
bad <- len_by_fips[N < 9]
bad_fips <- bad[, FIPS]
df_ok <- df[!FIPS %in% bad_fips]


### DLNM Specifications.

lag_max <- 2  # start with 2 or 3, not 4+, until stable

# Choose knots/boundaries globally (good practice)
unemp_boundary <- range(df$Unemployment, na.rm = TRUE)
unemp_knots    <- as.numeric(quantile(df$Unemployment, probs = c(0.25, 0.5, 0.75), na.rm = TRUE))
rx_boundary <- range(df$DR, na.rm = TRUE)
rx_knots    <- as.numeric(quantile(df$DR, probs = c(0.25, 0.5, 0.75), na.rm = TRUE))
unins_boundary <- range(df$uninsured_rate, na.rm = TRUE)
unins_knots    <- as.numeric(quantile(df$uninsured_rate, probs = c(0.25, 0.5, 0.75), na.rm = TRUE))
lag_knots <- lag_max/2  # simple start (or c(1,2) style choices later)


df_fit <- df_ok[!is.na(logMR_lead)]  # drop last year per county
# NOTE: cb_* objects include NAs for initial lags; that's expected

make_crossbases <- function(df_fit, lag_max,
                            unemp_knots, unemp_boundary,
                            rx_knots, rx_boundary,
                            unins_knots, unins_boundary,
                            lag_knots,
                            lag_fun = "ns") {
  
  # define arglag spec
  arglag_spec <- switch(
    lag_fun,
    "ns"     = list(fun="ns", knots=lag_knots, Boundary.knots=c(0, lag_max)),
    "lin"    = list(fun="lin"),
    "strata" = list(fun="strata"),
    stop("Unsupported lag_fun")
  )
  
  cb_unemp <- crossbasis(
    x      = df_fit$Unemployment,
    lag    = lag_max,
    # argvar = list(fun="ns", knots=unemp_knots, Boundary.knots=unemp_boundary),
    # argvar = list(fun = "lin"),
    argvar = list(fun="ns", df=3),
    arglag = arglag_spec,
    group  = df_fit$FIPS
  )
  
  cb_rx <- crossbasis(
    x      = df_fit$DR,
    lag    = lag_max,
    # argvar = list(fun="ns", knots=rx_knots, Boundary.knots=rx_boundary),
    # argvar = list(fun = "lin"),
    argvar = list(fun="ns", df=3),
    arglag = arglag_spec,
    group  = df_fit$FIPS
  )
  
  cb_unins <- crossbasis(
    x      = df_fit$uninsured_rate,
    lag    = lag_max,
    # argvar = list(fun="ns", knots=unins_knots, Boundary.knots=unins_boundary),
    # argvar = list(fun = "lin"),
    argvar = list(fun="ns", df=3),
    arglag = arglag_spec,
    group  = df_fit$FIPS
  )
  
  list(
    unemp = cb_unemp,
    rx    = cb_rx,
    unins = cb_unins
  )
}


### Running a loop over 4 different model specifications.

forms <- list(
  none      = logMR_lead ~ cb_unemp + cb_rx + cb_unins,
  year_fe   = logMR_lead ~ cb_unemp + cb_rx + cb_unins + factor(Year),
  county_fe = logMR_lead ~ cb_unemp + cb_rx + cb_unins + factor(FIPS),
  twfe      = logMR_lead ~ cb_unemp + cb_rx + cb_unins + factor(Year) + factor(FIPS)
)


####### Model functions:
extract_model_summary <- function(fit, cb_unemp, cb_rx, cb_unins, df_fit, lag_max) {
  # ---- centering values ----
  cen_unemp <- as.numeric(median(df_fit$Unemployment, na.rm=TRUE))
  cen_rx    <- as.numeric(median(df_fit$DR, na.rm=TRUE))
  cen_unins <- as.numeric(median(df_fit$uninsured_rate, na.rm=TRUE))
  
  # ---- exposure grids (consistent across specs) ----
  at_unemp <- as.numeric(quantile(df_fit$Unemployment, probs=seq(0.01,0.99,0.02), na.rm=TRUE))
  at_rx    <- as.numeric(quantile(df_fit$DR, probs=seq(0.01,0.99,0.02), na.rm=TRUE))
  at_unins <- as.numeric(quantile(df_fit$uninsured_rate, probs=seq(0.01,0.99,0.02), na.rm=TRUE))
  
  # ---- p25 for lag-slice ----
  p25_unemp <- as.numeric(quantile(df_fit$Unemployment, probs=0.25, na.rm=TRUE))
  p25_rx    <- as.numeric(quantile(df_fit$DR, probs=0.25, na.rm=TRUE))
  p25_unins <- as.numeric(quantile(df_fit$uninsured_rate, probs=0.25, na.rm=TRUE))
  
  # ---- cumulative over lags ----
  red_unemp <- crossreduce(cb_unemp, fit, cen=cen_unemp, at=at_unemp)
  red_rx    <- crossreduce(cb_rx,    fit, cen=cen_rx,    at=at_rx)
  red_unins <- crossreduce(cb_unins, fit, cen=cen_unins, at=at_unins)
  
  # ---- full lag-by-exposure predictions (for contour + slices) ----
  # bylag=1 ensures lags land on integers 0..lag_max for yearly data
  pred_unemp <- crosspred(cb_unemp, fit, cen=cen_unemp, at=at_unemp, bylag=1)
  pred_rx    <- crosspred(cb_rx,    fit, cen=cen_rx,    at=at_rx,    bylag=1)
  pred_unins <- crosspred(cb_unins, fit, cen=cen_unins, at=at_unins, bylag=1)
  
  list(
    # fit stats
    n        = stats::nobs(fit),
    aic      = AIC(fit),
    bic      = BIC(fit),
    dev      = deviance(fit),
    df_resid = fit$df.residual,
    lag_max  = lag_max,
    
    # per-variable bundles (keeps call-sites clean)
    unemp = list(cen=cen_unemp, at=at_unemp, p25=p25_unemp, pred=pred_unemp, red=red_unemp),
    rx    = list(cen=cen_rx,    at=at_rx,    p25=p25_rx,    pred=pred_rx,    red=red_rx),
    unins = list(cen=cen_unins, at=at_unins, p25=p25_unins, pred=pred_unins, red=red_unins)
  )
}

safe_slug <- function(x) gsub("[^A-Za-z0-9_\\-]+", "_", x)

save_dlnm_plots <- function(summary_obj, spec_id, out_dir, outcome_label = "logMR_lead") {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  lag_max <- summary_obj$lag_max
  
  # helper to make 3 plots for a single variable
  save_one <- function(var_id, var_label) {
    b <- summary_obj[[var_id]]
    pred <- b$pred
    red  <- b$red
    p25  <- b$p25
    
    base <- paste0(
      safe_slug(outcome_label),
      "__L", lag_max,
      "__", safe_slug(spec_id),
      "__", safe_slug(var_id)
    )
    
    # 1) contour
    f1 <- file.path(out_dir, paste0(base, "__contour.png"))
    grDevices::png(f1, width = 1400, height = 1000, res = 150)
    plot(
      pred, "contour",
      xlab = var_label,
      ylab = "Lag (years)",
      key.title = title("Δ log(MR_{t+1})"),
      plot.title = title(main = paste0(var_label, " cross-basis surface (", spec_id, ")"))
    )
    grDevices::dev.off()
    
    # 2) lag slice at p25
    f2 <- file.path(out_dir, paste0(base, "__lag_slice_p25.png"))
    grDevices::png(f2, width = 1400, height = 1000, res = 150)
    plot(
      pred, "slices",
      var = p25,
      ci = "lines",
      xlab = "Lag (years)",
      ylab = paste0("Δ ", outcome_label, " vs Median"),
      main = paste0(var_label, ": lag-response at 25th pct (", spec_id, ")")
    )
    abline(h = 0, lty = 3)
    grDevices::dev.off()
    
    # 3) cumulative over lags (from crossreduce)
    f3 <- file.path(out_dir, paste0(base, "__cumulative.png"))
    grDevices::png(f3, width = 1400, height = 1000, res = 150)
    plot(
      red,
      ci = "lines",
      xlab = var_label,
      ylab = paste0("Cumulative Δ ", outcome_label, " (sum over lags)"),
      main = paste0(var_label, ": cumulative association over lags (", spec_id, ")")
    )
    abline(h = 0, lty = 3)
    grDevices::dev.off()
    
    invisible(c(contour=f1, lag_slice_p25=f2, cumulative=f3))
  }
  
  files <- list(
    unemp = save_one("unemp", "Unemployment"),
    rx    = save_one("rx",    "RX dispensing (DR)"),
    unins = save_one("unins", "Uninsured rate")
  )
  
  invisible(files)
}



plot_dir <- file.path("dlnm_models", "3cb_fe_loop", "plots")

lag_funs <- c("ns", "lin", "strata")

results <- list()

for (lag_fun in lag_funs) {
  
  message("---- Lag specification: ", lag_fun, " ----")
  
  # build crossbasis objects for this lag spec
  cbs <- make_crossbases(
    df_fit, lag_max,
    unemp_knots, unemp_boundary,
    rx_knots, rx_boundary,
    unins_knots, unins_boundary,
    lag_knots,
    lag_fun = lag_fun
  )
  
  cb_unemp <- cbs$unemp
  cb_rx    <- cbs$rx
  cb_unins <- cbs$unins
  
  for (nm in names(forms)) {
    
    message("Fitting model: ", nm)
    
    fit <- glm(forms[[nm]],
               data = df_fit,
               family = gaussian(),
               na.action = na.omit)
    
    spec_id <- paste0(nm, "__lag", lag_fun)
    
    results[[spec_id]] <- extract_model_summary(
      fit,
      cb_unemp,
      cb_rx,
      cb_unins,
      df_fit,
      lag_max = lag_max
    )
    
    save_dlnm_plots(
      results[[spec_id]],
      spec_id = spec_id,
      out_dir = plot_dir,
      outcome_label = "logMR_lead"
    )
    
    saveRDS(
      fit,
      file = paste0("dlnm_models/3cb_fe_loop/dlnm_glm_",
                    spec_id,
                    "_lag", lag_max,
                    ".rds")
    )
    
    rm(fit)
    gc()
  }
}


# save the small extracted objects
saveRDS(results, file=paste0("dlnm_models/3cb_fe_loop/dlnm_glm_results_summaries_lag", lag_max, ".rds"))







plot_dir <- file.path("dlnm_models", "3cb_fe_loop", "plots")

results <- list()

for (nm in names(forms)) {
  message("Fitting model: ", nm)
  
  fit <- glm(forms[[nm]], data=df_fit, family=gaussian(), na.action=na.omit)
  
  # extract only what you need
  results[[nm]] <- extract_model_summary(fit, cb_unemp, cb_rx, cb_unins, df_fit, lag_max = lag_max)
  
  # save plots for this spec
  save_dlnm_plots(results[[nm]], spec_id = nm, out_dir = plot_dir, outcome_label = "logMR_lead")
  
  
  # optional: save the full model to disk, then drop it from memory
  saveRDS(fit, file=paste0("dlnm_models/3cb_fe_loop/dlnm_glm_", nm, "_lag", lag_max, ".rds"))
  
  rm(fit)
  gc()
}

# save the small extracted objects
saveRDS(results, file=paste0("dlnm_models/3cb_fe_loop/dlnm_glm_results_summaries_lag", lag_max, ".rds"))

# Checking AIC/BIC/Deviance of all models
comp <- rbindlist(lapply(names(results), function(nm) {
  r <- results[[nm]]
  data.table(
    model = nm,
    n = r$n,
    aic = r$aic,
    bic = r$bic,
    dev = r$dev,
    df_resid = r$df_resid
  )
}))
comp[order(aic)]


### Cumulative lag effects at variable increments
vars <- c("unemp", "rx", "unins")

for (v in vars) {
  
  cat("\n=========================================\n")
  cat("Variable:", v, "\n")
  cat("=========================================\n")
  
  r_none   <- results[["none"]][[v]]
  r_year   <- results[["year_fe"]][[v]]
  r_county <- results[["county_fe"]][[v]]
  r_twfe   <- results[["twfe"]][[v]]
  
  # Sanity check: make sure grids match
  if (!all.equal(r_none$predvar, r_twfe$predvar)) {
    stop("Prediction grids differ for variable: ", v)
  }
  
  dt_out <- data.table(
    exposure  = r_none$predvar,
    fit_none  = r_none$fit,
    fit_year  = r_year$fit,
    fit_county = r_county$fit,
    fit_twfe  = r_twfe$fit,
    diff      = r_twfe$fit - r_none$fit
  )
  
  print(dt_out)
}
