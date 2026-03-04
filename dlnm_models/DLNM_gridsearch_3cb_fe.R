### 3/1/26, EB: Fitting a model with 3 cross-basis terms, one each for Unemployment, RX, and Uninsured. I'll include SVI if I get this working.
### This runs a grid search over lag_max values, and different fixed-effect specifications, saving the models, model summary stats, and contour plots for each model.
library(data.table)
library(dlnm)
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


### -------------------------
### Config
### -------------------------
eps <- 0.01
lag_grid <- c(2L, 3L, 4L)
p_lagcurve <- 0.25

# Root output folder: dlnm_models/3cb_fe_loop/grid_run_YYYYMMDD_HHMM
run_id <- format(Sys.time(), "%Y%m%d_%H%M")
run_dir <- file.path("dlnm_models", "3cb_fe_loop", paste0("grid_run_", run_id))
dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)

### -------------------------
### Data construction (once)
### -------------------------
df <- data_loader()

df <- df[, .(
  FIPS, Year,
  MR, DR, Unemployment, uninsured_rate,
  .SD
), .SDcols = setdiff(names(df), c("FIPS", "Year"))]

setorder(df, FIPS, Year)

df[, logMR := log(MR + eps)]
df[, logMR_lead := shift(logMR, type = "lead", n = 1L), by = FIPS]

# Drop counties without full 9 years (your current rule)
len_by_fips <- df[, .N, by = FIPS]
bad_fips <- len_by_fips[N < 9, FIPS]
df_ok <- df[!FIPS %in% bad_fips]

# Drop last year per county (lead outcome missing)
df_ok <- df_ok[!is.na(logMR_lead)]

# (Recommended for grid search) drop rows with missing exposures up-front
# If you want to allow missingness later, remove these lines and handle carefully.
df_ok <- df_ok[!is.na(Unemployment) & !is.na(DR) & !is.na(uninsured_rate)]

# Ensure stable ordering for group-based lags
setorder(df_ok, FIPS, Year)

### -------------------------
### FE formula templates (constant)
### -------------------------
make_forms <- function() {
  list(
    none      = logMR_lead ~ cb_unemp + cb_rx + cb_unins,
    year_fe   = logMR_lead ~ cb_unemp + cb_rx + cb_unins + factor(Year),
    county_fe = logMR_lead ~ cb_unemp + cb_rx + cb_unins + factor(FIPS),
    twfe      = logMR_lead ~ cb_unemp + cb_rx + cb_unins + factor(Year) + factor(FIPS)
  )
}

### This one uses make_df_cb
make_forms_from_dfcb <- function(df_cb) {
  # cb_cols <- setdiff(names(df_cb), c("FIPS", "Year", "logMR_lead"))
  cb_cols <- grep("^(cbU_|cbR_|cbI_)", names(df_cb), value = TRUE)
  rhs_cb <- paste(cb_cols, collapse = " + ")
  
  list(
    none      = as.formula(paste("logMR_lead ~", rhs_cb)),
    year_fe   = as.formula(paste("logMR_lead ~", rhs_cb, "+ factor(Year)")),
    county_fe = as.formula(paste("logMR_lead ~", rhs_cb, "+ factor(FIPS)")),
    twfe      = as.formula(paste("logMR_lead ~", rhs_cb, "+ factor(Year) + factor(FIPS)"))
  )
}

### -------------------------
### Making the cross-bases
### -------------------------
### Because of lagged years, was getting NA error from GLM.
### This build cb terms off full panel years
make_df_cb <- function(df_ok, lag_max,
                       unemp_knots, unemp_boundary,
                       rx_knots, rx_boundary,
                       unins_knots, unins_boundary) {
  
  lag_knots <- lag_max / 2
  
  # Build crossbasis objects on df_ok (must be ordered!)
  cb_unemp <- crossbasis(
    x=df_ok$Unemployment, lag=lag_max,
    argvar=list(fun="ns", knots=unemp_knots, Boundary.knots=unemp_boundary),
    arglag=list(fun="ns", knots=lag_knots, Boundary.knots=c(0, lag_max)),
    group=df_ok$FIPS
  )
  
  cb_rx <- crossbasis(
    x=df_ok$DR, lag=lag_max,
    argvar=list(fun="ns", knots=rx_knots, Boundary.knots=rx_boundary),
    arglag=list(fun="ns", knots=lag_knots, Boundary.knots=c(0, lag_max)),
    group=df_ok$FIPS
  )
  
  cb_unins <- crossbasis(
    x=df_ok$uninsured_rate, lag=lag_max,
    argvar=list(fun="ns", knots=unins_knots, Boundary.knots=unins_boundary),
    arglag=list(fun="ns", knots=lag_knots, Boundary.knots=c(0, lag_max)),
    group=df_ok$FIPS
  )
  
  # Expand to explicit columns so glm() never sees NA inside an object
  XU <- as.data.table(cb_unemp); setnames(XU, paste0("cbU_", names(XU)))
  XR <- as.data.table(cb_rx);    setnames(XR, paste0("cbR_", names(XR)))
  XI <- as.data.table(cb_unins); setnames(XI, paste0("cbI_", names(XI)))
  
  # Assemble analysis table (include Year and FIPS for FE terms)
  df_cb <- cbind(
    df_ok[, .(FIPS, Year, logMR_lead, Unemployment, DR, uninsured_rate)],
    XU, XR, XI
  )
  
  # Drop rows with any NA (mainly first lag_max rows per county; plus any true missingness)
  keep <- complete.cases(df_cb)
  df_cb2 <- df_cb[keep]
  
  # Small usage report
  usage <- data.table(
    lag_max     = lag_max,
    rows_before = nrow(df_cb),
    rows_after  = nrow(df_cb2),
    dropped     = nrow(df_cb) - nrow(df_cb2),
    year_t_min  = min(df_cb2$Year),
    year_t_max  = max(df_cb2$Year),
    year_y_min  = min(df_cb2$Year + 1L),
    year_y_max  = max(df_cb2$Year + 1L),
    n_counties  = uniqueN(df_cb2$FIPS)
  )
  
  # Return everything needed downstream
  list(
    df_cb = df_cb2,
    usage = usage,
    cb_unemp = cb_unemp,
    cb_rx = cb_rx,
    cb_unins = cb_unins,
    lag_knots_used = lag_knots
  )
}

### -------------------------
### Plot helpers
### -------------------------
### Mismatched column names in cb_* and fit objects, crosspred can't reconcile without this helper function:
get_cb_block <- function(fit, cb, prefix) {
  b <- coef(fit)
  V <- vcov(fit)
  
  cb_cols <- colnames(cb)                 # e.g., "v1.l1" "v1.l2" ...
  fit_cols <- paste0(prefix, cb_cols)     # e.g., "cbU_v1.l1" ...
  
  # Ensure all required columns are present
  miss <- setdiff(fit_cols, names(b))
  if (length(miss) > 0) {
    stop("Missing basis coefficients in fit for prefix ", prefix,
         ". Examples: ", paste(head(miss, 5), collapse=", "))
  }
  
  b_sub <- b[fit_cols]
  V_sub <- V[fit_cols, fit_cols, drop = FALSE]
  
  # Rename to match cb colnames so crosspred can align safely
  names(b_sub) <- cb_cols
  colnames(V_sub) <- cb_cols
  rownames(V_sub) <- cb_cols
  
  list(coef = b_sub, vcov = V_sub)
}

crosspred_cumulative_from_fit <- function(cb, fit, prefix, cen, at) {
  blk <- get_cb_block(fit, cb, prefix)
  at <- sort(unique(as.numeric(at)))
  
  pr <- crosspred(cb, coef = blk$coef, vcov = blk$vcov, cen = cen, at = at)
  
  # Return a tiny object similar to crossreduce output (predvar/fit/se)
  list(
    predvar = pr$predvar,
    fit     = as.numeric(pr$allfit),
    se      = as.numeric(pr$allse),
    low     = as.numeric(pr$alllow),
    high    = as.numeric(pr$allhigh),
    ci.level = pr$ci.level
  )
}

############# TROUBLESHOOTING
save_surface_contour <- function(cb, fit, varname, out_png,
                                 cen, at_grid, prefix) {
  cb_block <- get_cb_block(fit, cb, prefix)
  
  pr <- crosspred(cb,
                  coef = cb_block$coef,
                  vcov = cb_block$vcov,
                  cen = cen,
                  at  = at_grid)
  x <- as.numeric(pr$predvar)
  
  # z is exposure x lag
  z <- as.matrix(pr$matfit)
  
  # Build y from the lag column names (lag0, lag1, ...)
  lag_names <- colnames(z)
  y <- suppressWarnings(as.numeric(sub("^lag", "", lag_names)))
  if (anyNA(y)) y <- 0:(ncol(z) - 1)
  
  # Sanity checks: contour needs dim(z) == c(length(x), length(y))
  stopifnot(nrow(z) == length(x))
  stopifnot(ncol(z) == length(y))
  
  png(out_png, width = 1200, height = 900, res = 130)
  on.exit(dev.off(), add = TRUE)
  
  contour(
    x = x,
    y = y,
    z = z,
    xlab = varname,
    ylab = "Lag (years)",
    main = paste0(varname, " surface (contour)"),
    nlevels = 12
  )
}


# lag-response curve at a chosen exposure value
save_lag_curve <- function(cb, fit, prefix, value, out_png, varname, cen) {
  cb_block <- get_cb_block(fit, cb, prefix)

  pr <- crosspred(cb,
                  coef = cb_block$coef,
                  vcov = cb_block$vcov,
                  cen = cen,
                  at  = value)
  
  mat <- as.matrix(pr$matfit)
  se  <- as.matrix(pr$matse)
  
  # For at=value (single exposure), mat is 1 x (lag+1) with colnames lag0, lag1, ...
  lag <- as.numeric(sub("^lag", "", colnames(mat)))
  y   <- as.numeric(mat[1, ])
  s   <- as.numeric(se[1, ])
  
  png(out_png, width = 1200, height = 900, res = 130)
  on.exit(dev.off(), add = TRUE)

  plot(lag, y, type = "b", xlab = "Lag (years)", ylab = "Log effect",
       main = paste0(varname, " lag-response @ value=", signif(value, 4)))
  lines(lag, y + 1.96 * se, type = "b", lty = 2)
  lines(lag, y - 1.96 * se, type = "b", lty = 2)
}

### -------------------------
### Fit summary extractor (compact)
### -------------------------
extract_fit_stats <- function(fit, model_name, lag_max) {
  data.table(
    model = model_name,
    lag_max = lag_max,
    n = nobs(fit),
    aic = AIC(fit),
    bic = BIC(fit),
    dev = deviance(fit),
    df_resid = fit$df.residual
  )
}

### -------------------------
### Main grid loop
### -------------------------
# forms <- make_forms()

for (L in lag_grid) {
  message("\n===============================")
  message("Running lag_max = ", L)
  message("===============================")
  
  # lag-specific folder
  lag_dir <- file.path(run_dir, paste0("lag_", sprintf("%02d", L)))
  dir.create(lag_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Build knots/boundaries on df_ok (analysis sample base)
  # (You could also compute on df for global stability; this is simpler & consistent.)
  unemp_boundary <- range(df_ok$Unemployment, na.rm = TRUE)
  unemp_knots    <- as.numeric(quantile(df_ok$Unemployment, probs = c(0.25, 0.5, 0.75), na.rm = TRUE))
  
  rx_boundary <- range(df_ok$DR, na.rm = TRUE)
  rx_knots    <- as.numeric(quantile(df_ok$DR, probs = c(0.25, 0.5, 0.75), na.rm = TRUE))
  
  unins_boundary <- range(df_ok$uninsured_rate, na.rm = TRUE)
  unins_knots    <- as.numeric(quantile(df_ok$uninsured_rate, probs = c(0.25, 0.5, 0.75), na.rm = TRUE))
  
  #############################################################################
  # --- build basis-expanded analysis table and log usage ---
  obj <- make_df_cb(
    df_ok = df_ok,
    lag_max = L,
    unemp_knots = unemp_knots, unemp_boundary = unemp_boundary,
    rx_knots = rx_knots, rx_boundary = rx_boundary,
    unins_knots = unins_knots, unins_boundary = unins_boundary
  )
  
  df_cb <- obj$df_cb
  cb_unemp <- obj$cb_unemp
  cb_rx    <- obj$cb_rx
  cb_unins <- obj$cb_unins
  lag_knots <- obj$lag_knots_used
  
  # Save usage log for this lag_max
  print(obj$usage)
  fwrite(obj$usage, file.path(lag_dir, "usage.csv"))
  
  # FE formulas built from df_cb columns (not cb objects)
  forms <- make_forms_from_dfcb(df_cb)
  
  #############################################################################
  
  # Save the basis spec used (for reproducibility)
  spec <- list(
    lag_max = L,
    p_lagcurve = p_lagcurve,
    knots = list(
      unemp_knots = unemp_knots, unemp_boundary = unemp_boundary,
      rx_knots = rx_knots, rx_boundary = rx_boundary,
      unins_knots = unins_knots, unins_boundary = unins_boundary,
      lag_knots = lag_knots
    )
  )

  saveRDS(spec, file = file.path(lag_dir, "spec.rds"))
  
  fit_stats_all <- list()
  results_reduced <- list()
  
  # FE loop
  for (fe_name in names(forms)) {
    message("  Fitting FE spec: ", fe_name)
    
    fe_dir <- file.path(lag_dir, paste0("fe_", fe_name))
    plot_dir <- file.path(fe_dir, "plots")
    dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
    
    # Fit
    fit <- glm(forms[[fe_name]],
               data = df_cb,
               family = gaussian(),
               na.action = na.fail
    )
    
    # Save full model
    saveRDS(fit, file = file.path(fe_dir, "model.rds"))
    
    # Fit stats
    fit_stats_all[[fe_name]] <- data.table(
      model = fe_name, lag_max = L,
      n = nobs(fit), aic = AIC(fit), bic = BIC(fit),
      dev = deviance(fit), df_resid = fit$df.residual
    )
    
    # ---- plotting (unchanged idea, but use df_cb for centering/grids if you want) ----
    # Use df_ok or df_cb for centering; I recommend df_cb so plots reflect estimation sample.
    cen_unemp <- median(df_cb$Unemployment, na.rm=TRUE)
    cen_rx    <- median(df_cb$DR, na.rm=TRUE)
    cen_unins <- median(df_cb$uninsured_rate, na.rm=TRUE)
    
    at_unemp_grid <- as.numeric(quantile(df_cb$Unemployment, probs=seq(0.05,0.95,0.02), na.rm=TRUE))
    at_rx_grid    <- as.numeric(quantile(df_cb$DR, probs=seq(0.05,0.95,0.02), na.rm=TRUE))
    at_unins_grid <- as.numeric(quantile(df_cb$uninsured_rate, probs=seq(0.05,0.95,0.02), na.rm=TRUE))
    
    # Surface contours via crosspred (uses cb_* objects + fitted model)
    save_surface_contour(cb_unemp, fit, "unemp",
                         out_png=file.path(plot_dir, "surface_unemp.png"),
                         cen=cen_unemp, at_grid=at_unemp_grid,
                         prefix="cbU_"
    )
    save_surface_contour(cb_rx, fit, "rx",
                         out_png=file.path(plot_dir, "surface_rx.png"),
                         cen=cen_rx, at_grid=at_rx_grid,
                         prefix="cbR_"
    )
    save_surface_contour(cb_unins, fit, "unins",
                         out_png=file.path(plot_dir, "surface_unins.png"),
                         cen=cen_unins, at_grid=at_unins_grid,
                         prefix="cbI_"
    )
    
    # Lag curves @ p25
    val_unemp <- as.numeric(quantile(df_cb$Unemployment, 0.25, na.rm=TRUE))
    val_rx    <- as.numeric(quantile(df_cb$DR, 0.25, na.rm=TRUE))
    val_unins <- as.numeric(quantile(df_cb$uninsured_rate, 0.25, na.rm=TRUE))
    
    save_lag_curve(cb_unemp, fit, prefix='cbU_', value=val_unemp, cen=cen_unemp,
                   out_png=file.path(plot_dir, "lagcurve_unemp_p25.png"), varname="unemp"
    )
    save_lag_curve(cb_rx, fit, prefix='cbR_', value=val_rx, cen=cen_rx,
                   out_png=file.path(plot_dir, "lagcurve_rx_p25.png"), varname="rx"
    )
    save_lag_curve(cb_unins, fit, prefix='cbI_', value=val_unins, cen=cen_unins,
                   out_png=file.path(plot_dir, "lagcurve_unins_p25.png"), varname="unins"
    )
    
    results_reduced[[fe_name]] <- list(
      unemp_cum = crosspred_cumulative_from_fit(
        cb_unemp, fit, prefix="cbU_", cen=cen_unemp,
        at=quantile(df_cb$Unemployment, probs=seq(0.05,0.95,0.05), na.rm=TRUE)
      ),
      rx_cum = crosspred_cumulative_from_fit(
        cb_rx, fit, prefix="cbR_", cen=cen_rx,
        at=quantile(df_cb$DR, probs=seq(0.05,0.95,0.05), na.rm=TRUE)
      ),
      unins_cum = crosspred_cumulative_from_fit(
        cb_unins, fit, prefix="cbI_", cen=cen_unins,
        at=quantile(df_cb$uninsured_rate, probs=seq(0.05,0.95,0.05), na.rm=TRUE)
      )
    )
    rm(fit)
    gc()
  }
  
  # Save per-lag summary tables/objects
  fit_comp <- rbindlist(fit_stats_all)
  fwrite(fit_comp, file = file.path(lag_dir, "fit_stats.csv"))
  saveRDS(results_reduced, file = file.path(lag_dir, "reduced_results.rds"))
  
  message("Saved lag_max=", L, " outputs to: ", lag_dir)
}

message("\nDONE. Grid run saved to: ", run_dir)


###############################################################################
library(data.table)

run_dir  <- "dlnm_models/3cb_fe_loop/grid_run_20260302_0958"
lag_grid <- c(2, 3, 4)
fe_grid  <- c("none", "year_fe", "county_fe", "twfe")
vars     <- c("unemp", "rx", "unins")

plot_cum_curve <- function(obj, out_png, xlab, main, ylab = expression(Delta*log(MR))) {
  x <- as.numeric(obj$predvar)
  y <- as.numeric(obj$fit)
  
  # CI: prefer stored low/high, otherwise compute from se
  if (!is.null(obj$low) && !is.null(obj$high)) {
    lo <- as.numeric(obj$low)
    hi <- as.numeric(obj$high)
  } else {
    se <- as.numeric(obj$se)
    lo <- y - 1.96 * se
    hi <- y + 1.96 * se
  }
  
  # drop non-finite
  ok <- is.finite(x) & is.finite(y) & is.finite(lo) & is.finite(hi)
  x <- x[ok]; y <- y[ok]; lo <- lo[ok]; hi <- hi[ok]
  
  # sort by x
  o <- order(x)
  x <- x[o]; y <- y[o]; lo <- lo[o]; hi <- hi[o]
  
  png(out_png, width = 1200, height = 900, res = 130)
  on.exit(dev.off(), add = TRUE)
  
  ylim <- range(c(lo, hi), finite = TRUE)
  
  plot(x, y, type = "n", xlab = xlab, ylab = ylab, main = main, ylim = ylim)
  
  # CI ribbon
  polygon(
    x = c(x, rev(x)),
    y = c(hi, rev(lo)),
    col = grDevices::adjustcolor("grey70", alpha.f = 0.5),
    border = NA
  )
  
  # mean curve
  lines(x, y, lwd = 2)
  
  # reference line at 0
  abline(h = 0, lty = 3)
}

# Optional nicer x labels
xlabels <- list(
  unemp = "Unemployment rate (%)",
  rx    = "RX dispensing rate",
  unins = "Uninsured rate (%)"
)

for (L in lag_grid) {
  lag_dir <- file.path(run_dir, paste0("lag_", sprintf("%02d", L)))
  reduced_path <- file.path(lag_dir, "reduced_results.rds")
  if (!file.exists(reduced_path)) stop("Missing: ", reduced_path)
  
  reduced <- readRDS(reduced_path)
  
  for (fe in intersect(names(reduced), fe_grid)) {
    fe_dir <- file.path(lag_dir, paste0("fe_", fe))
    plot_dir <- file.path(fe_dir, "plots")
    dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
    
    for (v in vars) {
      obj <- reduced[[fe]][[paste0(v, "_cum")]]
      if (is.null(obj)) next
      
      out_png <- file.path(plot_dir, paste0("cumcurve_", v, ".png"))
      main <- paste0(v, " cumulative exposure–response | lag_max=", L, " | ", fe)
      
      plot_cum_curve(
        obj = obj,
        out_png = out_png,
        xlab = xlabels[[v]] %||% v,
        main = main
      )
    }
  }
  
  message("Wrote cumulative plots for lag_max=", L)
}



##############################################################################
### Diagnostic checking
library(data.table)

vars <- c("Unemployment", "DR", "uninsured_rate")

# 1. Within-county SD and range
within_stats <- rbindlist(lapply(vars, function(v) {
  
  dt <- df_ok[, .(
    mean_within = mean(get(v), na.rm=TRUE),
    sd_within   = sd(get(v), na.rm=TRUE),
    min_within  = min(get(v), na.rm=TRUE),
    max_within  = max(get(v), na.rm=TRUE)
  ), by = FIPS]
  
  data.table(
    variable = v,
    mean_sd_within = mean(dt$sd_within, na.rm=TRUE),
    median_sd_within = median(dt$sd_within, na.rm=TRUE),
    mean_range_within = mean(dt$max_within - dt$min_within, na.rm=TRUE),
    median_range_within = median(dt$max_within - dt$min_within, na.rm=TRUE)
  )
}))

print(within_stats)

# 2. Between county variation
between_stats <- rbindlist(lapply(vars, function(v) {
  
  data.table(
    variable = v,
    total_sd = sd(df_ok[[v]], na.rm=TRUE)
  )
}))

print(between_stats)

# 3. Variation ratio
variation_ratio <- merge(within_stats, between_stats, by="variable")

variation_ratio[, within_to_total_sd_ratio := mean_sd_within / total_sd]

print(variation_ratio)


# 4. Within-county time trend dominance
trend_r2 <- function(dt, var) {
  dt[, {
    y <- get(var)
    ok <- is.finite(y) & is.finite(Year)
    if (sum(ok) < 3) return(list(r2 = NA_real_))
    fit <- lm(y[ok] ~ Year[ok])
    list(r2 = summary(fit)$r.squared)
  }, by = FIPS]
}

vars <- c("Unemployment","DR","uninsured_rate")
r2_dt <- rbindlist(lapply(vars, function(v){
  tmp <- trend_r2(df_ok, v)
  tmp[, variable := v]
  tmp
}), fill=TRUE)

r2_dt[, .(
  n = .N,
  mean_r2 = mean(r2, na.rm=TRUE),
  median_r2 = median(r2, na.rm=TRUE),
  p90_r2 = quantile(r2, 0.9, na.rm=TRUE)
), by=variable]

# 5. Outcome persistence
ar_dt <- df_ok[, {
  ok <- is.finite(logMR) & is.finite(logMR_lead)
  if (sum(ok) < 3) return(list(cor_ar1 = NA_real_))
  list(cor_ar1 = cor(logMR[ok], logMR_lead[ok]))
}, by=FIPS]

summary(ar_dt$cor_ar1)

# 6. Within-demeaned Correlations btwn Unemp, RX, Unins
dm <- copy(df_ok)
for (v in c("Unemployment","DR","uninsured_rate")) {
  dm[, paste0(v,"_dm") := get(v) - mean(get(v), na.rm=TRUE), by=FIPS]
}

# pairwise correlations on demeaned variables (pooled)
cor(dm$Unemployment_dm, dm$DR_dm, use="complete.obs")
cor(dm$Unemployment_dm, dm$uninsured_rate_dm, use="complete.obs")
cor(dm$DR_dm, dm$uninsured_rate_dm, use="complete.obs")


# 7. Linear TWFE model as baseline
# Model A: TWFE, no lagged outcome
mA <- glm(
  logMR_lead ~ Unemployment + DR + uninsured_rate + factor(FIPS) + factor(Year),
  data = df_ok,
  family = gaussian(),
  na.action = na.omit
)

# Model B: TWFE + lagged outcome
mB <- glm(
  logMR_lead ~ logMR + Unemployment + DR + uninsured_rate + factor(FIPS) + factor(Year),
  data = df_ok,
  family = gaussian(),
  na.action = na.omit
)

# Compare key coefficients
coef_compare <- data.table(
  term = c("logMR", "Unemployment", "DR", "uninsured_rate"),
  beta_A = c(NA, coef(mA)["Unemployment"], coef(mA)["DR"], coef(mA)["uninsured_rate"]),
  beta_B = c(coef(mB)["logMR"], coef(mB)["Unemployment"], coef(mB)["DR"], coef(mB)["uninsured_rate"])
)

print(coef_compare)

# Also compare fit stats
fit_compare <- data.table(
  model = c("TWFE", "TWFE + lagged logMR"),
  n = c(nobs(mA), nobs(mB)),
  aic = c(AIC(mA), AIC(mB)),
  bic = c(BIC(mA), BIC(mB)),
  df_resid = c(mA$df.residual, mB$df.residual),
  dev = c(deviance(mA), deviance(mB))
)

print(fit_compare)