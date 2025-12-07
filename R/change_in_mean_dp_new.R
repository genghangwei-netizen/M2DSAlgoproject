#--- Estimate noise variance using differenced variance ---
estimate_sigma_sq <- function(data) {
  T <- length(data)
  if (T < 2) return(1)
  # formula: sum((y_i - y_{i-1})^2) / (2 * (T - 1))
  return(sum(diff(data)^2) / (2 * (T - 1)))
}

#--- Precompute cumulative sums for O(1) cost evaluation ---
precompute_cost <- function(data) {
  T <- length(data)
  sums <- numeric(T + 1)
  sums_sq <- numeric(T + 1)

  for (t in 1:T) {
    sums[t + 1] <- sums[t] + data[t]
    sums_sq[t + 1] <- sums_sq[t] + data[t]^2
  }

  cost_segment <- function(s, t) {
    n <- t - s
    if (n <= 0) return(0)
    sum_y <- sums[t + 1] - sums[s + 1]
    sum_y_sq <- sums_sq[t + 1] - sums_sq[s + 1]
    return(sum_y_sq - (sum_y^2 / n))
  }

  return(list(cost_segment = cost_segment, sums = sums, sums_sq = sums_sq))
}

# --- Dynamic Programming (O(T^2)) for Penalized Change Point Detection ---
dp_changepoint <- function(data, beta) {
  T <- length(data)
  precomp <- precompute_cost(data)
  cost_segment <- precomp$cost_segment

  Q <- numeric(T + 1)
  P <- integer(T + 1)
  Q[1] <- 0

  for (t in 1:T) {
    min_Q <- Inf
    s_opt <- 0

    for (s in 0:(t - 1)) {
      current_cost <- Q[s + 1] + cost_segment(s, t) + beta
      if (current_cost < min_Q) {
        min_Q <- current_cost
        s_opt <- s
      }
    }

    Q[t + 1] <- min_Q
    P[t + 1] <- s_opt
  }

  # Backtracking
  changepoints <- integer(0)
  idx <- T + 1

  while (P[idx] != 0) {
    cp <- P[idx]
    changepoints <- c(cp, changepoints)
    idx <- cp + 1
  }

  return(list(cost = Q[T + 1], K_opt = length(changepoints), changepoints = changepoints))
}

# --- Internal: Scan multiple penalty values ---
dp_get_scan_data <- function(data, betas) {
  results <- data.frame(
    beta = numeric(length(betas)),
    cost = numeric(length(betas)),
    K_opt = integer(length(betas))
  )

  for (i in seq_along(betas)) {
    res <- dp_changepoint(data, betas[i])
    results[i, ] <- c(betas[i], res$cost, res$K_opt)
  }

  return(results)
}

# --- Automatic penalty selection using IC or noise-level formulas ---
dp_auto_beta <- function(data, penalty_type = "BIC", p = 1) {
  T <- length(data)
  beta <- 0

  if (penalty_type == "AIC") {
    beta <- 2 * p
  } else if (penalty_type == "BIC") {
    beta <- p * log(T)
  } else if (penalty_type == "HQ") {
    beta <- 2 * p * log(log(T))
  } else if (penalty_type == "Killick2LogN") {
    beta <- 2 * log(T)
  } else if (penalty_type %in% c("LowNoise", "MidNoise", "HighNoise")) {
    sigma_sq_hat <- estimate_sigma_sq(data)
    if (penalty_type == "HighNoise") beta <- 2 * sigma_sq_hat * log(T)
    if (penalty_type == "MidNoise")  beta <- 1 * sigma_sq_hat * log(T)
    if (penalty_type == "LowNoise")  beta <- 0.5 * sigma_sq_hat * log(T)
  } else {
    stop(paste("Invalid penalty_type:", penalty_type))
  }

  message(paste("Using penalty type:", penalty_type, "with Beta =", round(beta, 4)))
  return(dp_changepoint(data, beta))
}

# --- Penalty scan + visualization of cost and number of change points ---
dp_plot_penalty_analysis <- function(data, max_beta_scan = 20, p = 1) {

  T <- length(data)

  beta_aic     <- 2 * p
  beta_hq      <- 2 * p * log(log(T))
  beta_bic     <- p * log(T)
  beta_killick <- 2 * log(T)

  sigma_sq <- estimate_sigma_sq(data)
  beta_yao <- 2 * sigma_sq * log(T)

  beta_lines <- data.frame(
    beta = c(beta_aic, beta_hq, beta_bic, beta_killick, beta_yao),
    label = c("AIC", "HQ", "BIC", "Killick", "Yao & Au"),
    color = c("orange", "purple", "red", "darkgreen", "blue"),
    lty = c(2, 2, 1, 1, 3)
  )

  betas <- seq(0.5, max_beta_scan, 0.5)
  results <- dp_get_scan_data(data, betas)

  old_par <- par(mfrow = c(1, 2))
  on.exit(par(old_par))

  plot_with_betas <- function(x, y, y_label, title, type = 'l', y_factor = 1.05) {
    plot(x, y, type = type, main = title,
         xlab = expression(Penalty ~ beta), ylab = y_label,
         lwd = 2, col = ifelse(y_label == "Optimal Total Cost", "darkblue", "firebrick"),
         xlim = c(0, max_beta_scan), ylim = c(min(y) * 0.9, max(y) * y_factor))
    grid(lty = 3)

    for (i in 1:nrow(beta_lines)) {
      abline(v = beta_lines$beta[i], col = beta_lines$color[i], lty = beta_lines$lty[i])
      text(beta_lines$beta[i], max(y) * (0.15 + i * 0.1),
           labels = beta_lines$label[i], col = beta_lines$color[i],
           pos = 4, offset = 0.3, srt = 90,cex=0.8)
    }
  }

  plot_with_betas(results$beta, results$cost,
                  y_label = "Optimal Total Cost",
                  title = "Total Cost vs Penalty")

  plot_with_betas(results$beta, results$K_opt,
                  y_label = "Optimal Number of Change Points",
                  title = "Change Point Count vs Penalty",
                  type = 's', y_factor = 1.1)

  message("\nOptimal K at standard penalties:")
  summary_data <- data.frame(
    Penalty = beta_lines$label,
    Beta = round(beta_lines$beta, 4),
    K = sapply(beta_lines$beta, function(b) dp_changepoint(data, b)$K_opt)
  )
  print(summary_data)
}

# --- PELT: Pruned Dynamic Programming (O(T)) ---
dp_changepoint_pelt <- function(data, beta) {
  T <- length(data)
  if (T == 0) return(list(cost = 0, K_opt = 0, changepoints = integer(0)))

  precomp <- precompute_cost(data)
  cost_segment <- precomp$cost_segment

  Q <- numeric(T + 1)
  P <- integer(T + 1)
  Q[1] <- 0

  R <- 0

  for (t in 1:T) {
    min_Q <- Inf
    s_opt <- 0

    for (s in R) {
      cost_val <- Q[s + 1] + cost_segment(s, t) + beta
      if (cost_val < min_Q) {
        min_Q <- cost_val
        s_opt <- s
      }
    }

    Q[t + 1] <- min_Q
    P[t + 1] <- s_opt

    R_new <- integer(0)
    for (s in R) {
      if (Q[t + 1] > Q[s + 1] + cost_segment(s, t))
        R_new <- c(R_new, s)
    }
    R <- c(R_new, t)
  }

  changepoints <- integer(0)
  idx <- T + 1
  while (P[idx] != 0) {
    cp <- P[idx]
    changepoints <- c(cp, changepoints)
    idx <- cp + 1
  }

  return(list(cost = Q[T + 1], K_opt = length(changepoints), changepoints = changepoints))
}

# --- Simple function execution timer ---
time_function_call <- function(expr) {
  start <- Sys.time()
  result <- eval(substitute(expr))
  end <- Sys.time()
  return(list(result = result, time_ms = as.numeric(difftime(end, start, units = "secs")) * 1000))
}
