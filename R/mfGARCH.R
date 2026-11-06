# Run from the repository root: Rscript R/mfGARCH.R
library(mfGARCH)

df <- read.csv("data/processed/data_for_R.csv")

df$date <- as.Date(df$Date)
df$log_return <- df$log_return * 100
df$week_date <- as.Date(cut(df$date, "week"))

nfci_weekly <- aggregate(NFCI ~ week_date, data = df, FUN = mean)
head(nfci_weekly)

df$NFCI <- NULL
df <- merge(df, nfci_weekly, by = "week_date", all.x = TRUE)
head(df)

fit_nfci <- fit_mfgarch(
  data     = df,
  y        = "log_return",
  x        = "NFCI",
  K        = 52,
  low.freq = "week_date"
)
print(fit_nfci)


df$month_date <- as.Date(cut(df$date, "month"))

epu_monthly <- aggregate(EPU ~ month_date, data = df, FUN = mean)
head(epu_monthly)

df$EPU <- NULL
df <- merge(df, epu_monthly, by = "month_date", all.x = TRUE)
head(df)

fit_epu <- fit_mfgarch(
  data = df,
  y = "log_return",
  x = "EPU",
  K = 12,
  low.freq = "month_date"
)
print(fit_epu)


gpr_monthly <- aggregate(GPR ~ month_date, data = df, FUN = mean)
head(gpr_monthly)

df$GPR <- NULL
df <- merge(df, gpr_monthly, by = "month_date", all.x = TRUE)
head(df)

fit_gpr <- fit_mfgarch(
  data = df,
  y = "log_return",
  x = "GPR",
  K = 12,
  low.freq = "month_date"
)
print(fit_gpr)


fit_nfci_epu <- fit_mfgarch(
  data     = df,
  y        = "log_return",
  x        = "NFCI",
  x.two = "EPU",
  K        = 52,
  K.two = 12,
  low.freq = "week_date",
  low.freq.two = "month_date"
)
print(fit_nfci_epu)


fit_nfci_gpr <- fit_mfgarch(
  data     = df,
  y        = "log_return",
  x        = "NFCI",
  x.two = "GPR",
  K        = 52,
  K.two = 12,
  low.freq = "week_date",
  low.freq.two = "month_date"
)
print(fit_nfci_gpr)


fit_epu_gpr <- fit_mfgarch(
  data     = df,
  y        = "log_return",
  x        = "EPU",
  x.two = "GPR",
  K        = 12,
  K.two = 12,
  low.freq = "month_date",
  low.freq.two = "month_date"
)
print(fit_epu_gpr)


results <- data.frame(
  Model = c("NFCI", "EPU", "GPR", "NFCI+EPU", "NFCI+GPR", "EPU+GPR"),
  BIC   = c(fit_nfci$bic, fit_epu$bic, fit_gpr$bic, fit_nfci_epu$bic, fit_nfci_gpr$bic, fit_epu_gpr$bic),
  LLH   = c(fit_nfci$llh, fit_epu$llh, fit_gpr$llh, fit_nfci_epu$llh, fit_nfci_gpr$llh, fit_epu_gpr$llh)
)
print(results)


write.csv(results, "results/garch_midas_results.csv", row.names = FALSE)

