##############################################
## Kernel Change-Point Detection 
##############################################

## -----------------------------
## 1. Kernel Functions & Kernel Matrix
## -----------------------------

# x: vector or matrix (n×d)
rbf_kernel <- function(x, sigma = NULL) {
  x <- as.matrix(x)
  n <- nrow(x)
  
  # Squared distance matrix
  dist2 <- as.matrix(dist(x))^2
  
  if (is.null(sigma)) {
    # Median heuristic
    dvec <- dist2[upper.tri(dist2)]
    sig2 <- stats::median(dvec)
    if (!is.finite(sig2) || sig2 <= 0) sig2 <- 1
    sigma <- sqrt(sig2 / 2)
  }
  
  K <- exp(- dist2 / (2 * sigma^2))
  attr(K, "sigma") <- sigma
  K
}

# Simple linear kernel
linear_kernel <- function(x) {
  x <- as.matrix(x)
  K <- x %*% t(x)
  K
}

# Intersection kernel
intersection_kernel <- function(x) {
  x <- as.matrix(x)
  n <- nrow(x)
  d <- ncol(x)
  
  K <- matrix(0, n, n)
  
  for (i in 1:n) {
    xi_mat <- matrix(x[i, ], nrow = n, ncol = d, byrow = TRUE)
    K[i, ] <- rowSums(pmin(xi_mat, x))
  }
  
  attr(K, "sigma") <- NA_real_
  K
}

# Unified interface: return kernel matrix given data and kernel type
kernel_matrix <- function(x,
                          kernel = c("rbf", "linear", "intersection"),
                          sigma = NULL) {
  x <- as.matrix(x)
  kernel <- match.arg(kernel)
  
  if (kernel == "rbf") {
    K <- rbf_kernel(x, sigma = sigma)
  } else if (kernel == "linear") {
    K <- linear_kernel(x)
    attr(K, "sigma") <- NA_real_
  } else if (kernel == "intersection") {
    K <- intersection_kernel(x)
    attr(K, "sigma") <- NA_real_
  }
  
  K
}

## -----------------------------
## 2. Segment cost matrix (kernel least squares loss)
## -----------------------------
segment_cost_matrix_from_K <- function(K) {
  K <- as.matrix(K)
  n <- nrow(K)
  
  # Prefix sums for diagonal
  diagK <- diag(K)
  csum_diag <- c(0, cumsum(diagK))
  
  # 2D prefix sums for quick block sums
  S <- matrix(0, n + 1, n + 1)
  for (i in 1:n) {
    row_cum <- cumsum(K[i, ])
    S[i + 1, 2:(n + 1)] <- S[i, 2:(n + 1)] + row_cum
  }
  
  # Helper: sum_{p=i}^j sum_{q=i}^j K[p,q]
  block_sum <- function(i, j) {
    S[j + 1, j + 1] - S[i, j + 1] - S[j + 1, i] + S[i, i]
  }
  
  # Cost matrix
  C <- matrix(Inf, n, n)
  for (i in 1:n) {
    for (j in i:n) {
      len <- j - i + 1
      sum_diag <- csum_diag[j + 1] - csum_diag[i]
      sum_block <- block_sum(i, j)
      C[i, j] <- sum_diag - sum_block / len
    }
  }
  
  C
}

## -----------------------------
## 3. Dynamic Programming
## -----------------------------
kernel_segmentation_dp <- function(C, Dmax = NULL) {
  C <- as.matrix(C)
  n <- nrow(C)
  
  if (is.null(Dmax)) Dmax <- n
  Dmax <- min(Dmax, n)
  
  # dp[D, j]: minimal cost for first j points with D segments
  dp  <- matrix(Inf, Dmax, n)
  last <- matrix(NA_integer_, Dmax, n)
  
  # Case D = 1
  dp[1, ] <- C[1, ]
  last[1, ] <- 0L
  
  for (D in 2:Dmax) {
    for (j in D:n) {
      best_val <- Inf
      best_t <- NA_integer_
      for (t in (D-1):(j-1)) {
        val <- dp[D-1, t] + C[t + 1, j]
        if (val < best_val) {
          best_val <- val
          best_t <- t
        }
      }
      dp[D, j] <- best_val
      last[D, j] <- best_t
    }
  }
  
  list(dp = dp, last = last)
}

# Backtracking to find segment boundaries
backtrack_segmentation <- function(last, D, n) {
  ends <- integer(D)
  cur_end <- n
  for (d in D:1) {
    ends[d] <- cur_end
    cur_end <- last[d, cur_end]
  }
  ends
}

## -----------------------------
## 4. Estimate vmax (optional)
## -----------------------------
estimate_vmax_from_K <- function(K,
                                 t_left = 0.05,
                                 t_right = 0.95) {
  K <- as.matrix(K)
  n <- nrow(K)
  idx1 <- 1:max(1, floor(t_left * n))
  idx2 <- min(n, ceiling(t_right * n)):n
  
  est_tr <- function(idx) {
    m <- length(idx)
    if (m <= 1) return(0)
    Ksub <- K[idx, idx, drop = FALSE]
    diag_mean <- mean(diag(Ksub))
    sumK <- sum(Ksub)
    mu2 <- sumK / m^2
    diag_mean - mu2
  }
  
  v1 <- est_tr(idx1)
  v2 <- est_tr(idx2)
  vmax <- max(v1, v2)
  if (!is.finite(vmax) || vmax <= 0) {
    vmax <- 1
  }
  vmax
}

## -----------------------------
## 5. Low-rank kernel matrix
## -----------------------------
low_rank_kernel_matrix <- function(K, L) {
  K <- as.matrix(K)
  n <- nrow(K)
  L <- min(L, n - 1)
  
  if (L <= 0) stop("Target rank L must be positive.")
  
  eig <- eigen(K)
  
  lambda <- eig$values[1:L]
  U <- eig$vectors[, 1:L, drop = FALSE]
  lambda[lambda < 0] <- 0
  
  K_L <- U %*% diag(lambda, nrow = L) %*% t(U)
  K_L <- (K_L + t(K_L)) / 2
  
  return(K_L)
}

## -----------------------------
## 6. Main Function: Kernel Change-Point Detection
## -----------------------------
kernel_change_point <- function(x,
                                kernel = c("rbf", "linear", "intersection"),
                                sigma = NULL,
                                Dmax = NULL,
                                C = 2,
                                vmax = NULL,
                                rank = NULL, # optional low-rank parameter
                                t_left = 0.05,
                                t_right = 0.95,
                                verbose = TRUE){
  x <- as.matrix(x)
  n <- nrow(x)
  kernel <- match.arg(kernel)
  is_low_rank <- FALSE 
  
  if (is.null(Dmax)) Dmax <- min(50, n)
  
  ## Step 1: Kernel matrix (Full Rank)
  K_full <- kernel_matrix(x, kernel = kernel, sigma = sigma)
  sigma_used <- attr(K_full, "sigma")
  K <- K_full
  
  ## Step 1.5: Low-Rank Approximation
  if (!is.null(rank) && rank > 0 && rank < n) {
    K <- low_rank_kernel_matrix(K_full, rank)
    is_low_rank <- TRUE
    if (verbose) cat(paste("Using low-rank approximation (L =", rank, ")\n"))
  }
  
  ## Step 2: Cost matrix
  Cmat <- segment_cost_matrix_from_K(K)
  
  ## Step 3 & 4: DP segmentation & vmax estimate
  dp_res <- kernel_segmentation_dp(Cmat, Dmax = Dmax)
  dp <- dp_res$dp
  last <- dp_res$last
  
  if (is.null(vmax)) {
    # estimate vmax from full-rank K
    vmax <- estimate_vmax_from_K(K_full, t_left = t_left, t_right = t_right)
  }
  
  ## Step 5 & 6: Compute criterion and select best D
  total_cost <- dp[, n]
  D_vec <- 1:Dmax
  emp_risk <- total_cost / n
  penalty <- C * vmax * D_vec / n * (1 + log(n / D_vec))
  crit <- emp_risk + penalty
  
  D_hat <- which.min(crit)
  ends_hat <- backtrack_segmentation(last, D_hat, n)
  cps_hat <- ends_hat[-length(ends_hat)]
  
  ## Verbose output
  if (verbose) {
    cat("\n================ Kernel Change-Point Detection ================\n")
    cat("Method: ", ifelse(is_low_rank, "Low-Rank Approx", "Full Rank"), "\n")
    cat("Data size n          :", n, "\n")
    cat("Kernel               :", kernel, "\n")
    
    if (kernel == "rbf") {
      if (is.null(sigma)) {
        sigma_str <- paste0("auto (", round(sigma_used, 4), ")")
      } else {
        sigma_str <- as.character(sigma)
      }
    } else {
      sigma_str <- "N/A (non-RBF kernel)"
    }
    cat("Sigma (RBF only)     :", sigma_str, "\n")
    cat("Rank L               :", ifelse(is_low_rank, rank, "Full"), "\n")
    cat("Dmax                 :", Dmax, "\n")
    cat("Penalty constant C   :", C, "\n")
    cat("Estimated vmax       :", round(vmax, 4), "\n")
    cat("--------------------------------------------------------------\n")
    cat("Selected D_hat       :", D_hat, "\n")
    cat("Estimated CPs        :", ifelse(length(cps_hat)==0, "None", paste(cps_hat, collapse=", ")), "\n")
    cat("Segment ends         :", paste(ends_hat, collapse=", "), "\n")
    cat("==============================================================\n\n")
  }
  
  ## Return result list
  return(list(
    D_hat = D_hat,
    change_points = cps_hat,
    ends = ends_hat,
    crit = crit,
    emp_risk = emp_risk,
    penalty = penalty,
    total_cost = total_cost,
    K = K_full, 
    K_used = K, 
    sigma = sigma_used,
    vmax = vmax,
    dp = dp,
    last = last,
    x = x,
    rank = ifelse(is_low_rank, rank, n)
  ))
}

## -----------------------------
## 7. Plotting & Utility Functions
## -----------------------------

plot_kcp_signal <- function(x,
                            res,
                            main = "Kernel Change-Point Detection") {
  x <- as.numeric(x)
  n <- length(x)
  plot(x, type = "l",
       xlab = "t",
       ylab = "x_t",
       main = main)
  abline(v = res$change_points, col = "red", lty = 2)
}

plot_kcp_crit <- function(res,
                          main = "Criterion vs Number of Segments") {
  D_vec <- seq_along(res$crit)
  plot(D_vec, res$crit, type = "b",
       xlab = "D (number of segments)",
       ylab = "criterion",
       main = main)
  abline(v = res$D_hat, col = "red", lty = 2)
}

annotate_kernel_matrix <- function(K, change_points, main="Annotated Kernel Matrix") {
  K <- as.matrix(K)
  n <- nrow(K)
  
  df <- expand.grid(
    Var1 = 1:n,
    Var2 = 1:n
  )
  df$K <- as.vector(K)
  
  p <- ggplot(df, aes(x = Var1, y = Var2, fill = K)) +
    geom_tile() +
    scale_fill_gradient(low="black", high="yellow") +
    coord_fixed() +
    theme_minimal(base_size = 14) +
    labs(title = main, x = "Index", y = "Index")
  
  for (cp in change_points) {
    p <- p +
      geom_vline(xintercept = cp, color = "red", linetype = "dashed", size=1) +
      geom_hline(yintercept = cp, color = "red", linetype = "dashed", size=1)
  }
  
  segs <- c(1, change_points, n)
  for (i in 1:(length(segs)-1)) {
    xstart <- segs[i]
    xend   <- segs[i+1]
    p <- p +
      annotate("rect",
               xmin = xstart, xmax = xend,
               ymin = xstart, ymax = xend,
               fill = NA, color="cyan", size=1.2
      )
  }
  print(p)
}

calculate_matches <- function(A, T_star, threshold) {
  if (length(A) == 0 || length(T_star) == 0) return(0)
  
  matched_count <- 0
  for (a in A) {
    min_dist <- min(abs(a - T_star))
    if (min_dist <= threshold) {
      matched_count <- matched_count + 1
    }
  }
  return(matched_count)
}
