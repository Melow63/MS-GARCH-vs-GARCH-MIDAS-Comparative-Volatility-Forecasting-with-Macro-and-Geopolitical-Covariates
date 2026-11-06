# Run from the repository root: Rscript R/mfGARCH_forecasting.R
library(mfGARCH)

# 1. Load and prepare data
df <- read.csv("data/processed/data_for_R.csv")
df$date       <- as.Date(df$Date)
df$log_return <- df$log_return * 100
df$week_date  <- as.Date(cut(df$date, "week"))
df$month_date <- as.Date(cut(df$date, "month"))

# Aggregate NFCI weekly and EPU monthly
nfci_weekly <- aggregate(NFCI ~ week_date, data = df, FUN = mean)
df$NFCI <- NULL
df <- merge(df, nfci_weekly, by = "week_date", all.x = TRUE)

epu_monthly <- aggregate(EPU ~ month_date, data = df, FUN = mean)
df$EPU <- NULL
df <- merge(df, epu_monthly, by = "month_date", all.x = TRUE)

# Sort by date (merge can disorder rows)
df <- df[order(df$date), ]

# 2. Train / test split
train <- df[df$date < as.Date("2023-01-01"), ]
test  <- df[df$date >= as.Date("2023-01-01"), ]

cat("Train:", nrow(train), "| Test:", nrow(test), "\n")

# 3. Estimate GARCH-MIDAS NFCI+EPU on training set only
fit_train <- fit_mfgarch(
  data         = train,
  y            = "log_return",
  x            = "NFCI",
  x.two        = "EPU",
  K            = 52,
  K.two        = 12,
  low.freq     = "week_date",
  low.freq.two = "month_date"
)

print(fit_train$par)
cat("BIC train:", fit_train$bic, "\n")

# 4. Retrieve estimated parameters from training set
p      <- fit_train$par
mu     <- p["mu"]
alpha  <- p["alpha"]
beta   <- p["beta"]
gamma  <- p["gamma"]
m      <- p["m"]
theta1 <- p["theta"]
w2_1   <- p["w2"]
theta2 <- p["theta.two"]
w2_2   <- p["w2.two"]

# Beta weighting function (normalized, w1=1 fixed)
beta_weights <- function(K, w2) {
  k <- 1:K
  w <- (1 - k/(K+1))^(w2 - 1)
  w / sum(w)
}

phi1 <- beta_weights(52, w2_1)   # NFCI weights (52 weeks)
phi2 <- beta_weights(12, w2_2)   # EPU weights (12 months)

cat("NFCI weights (first 5):", round(phi1[1:5], 4), "\n")
cat("EPU weights (first 5):", round(phi2[1:5], 4), "\n")

# 5. Monthly rolling window: reestimate at the start of each test month
test_months <- unique(format(test$date, "%Y-%m"))
ms_var_list <- list()

for (ym in test_months) {
  cutoff    <- as.Date(paste0(ym, "-01"))
  hist_data <- df[df$date < cutoff, ]
  month_data <- test[format(test$date, "%Y-%m") == ym, ]

  fit_roll <- tryCatch(
    fit_mfgarch(
      data         = hist_data,
      y            = "log_return",
      x            = "NFCI",
      x.two        = "EPU",
      K            = 52,
      K.two        = 12,
      low.freq     = "week_date",
      low.freq.two = "month_date"
    ),
    error = function(e) NULL
  )

  if (is.null(fit_roll)) next

  # One-month-ahead forecast: tau.forecast * last g
  tau_f  <- fit_roll$tau.forecast
  g_last <- tail(fit_roll$g, 1)
  h_pred <- tau_f * g_last

  month_data$h_pred <- h_pred
  ms_var_list[[ym]] <- month_data[, c("date", "log_return", "h_pred")]

  cat("Month", ym, "| h_pred =", round(h_pred, 4), "\n")
}

# 6. Assemble and export
midas_forecast <- do.call(rbind, ms_var_list)
rownames(midas_forecast) <- NULL

cat("Number of forecasts:", nrow(midas_forecast), "\n")
head(midas_forecast)

write.csv(
  midas_forecast,
  "results/midas_oos_forecast.csv",
  row.names = FALSE
)
