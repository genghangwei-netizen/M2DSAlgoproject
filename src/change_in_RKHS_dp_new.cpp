// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

using namespace Rcpp;
using namespace Eigen;

#include <algorithm>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <iostream>


// ============================================================
// 1. Kernel Functions & Kernel Matrix
// ============================================================

// Compute squared Euclidean distance matrix between rows of X (n x d)
inline Eigen::MatrixXd squared_distance_matrix(const Eigen::MatrixXd& X) {
  const int n = static_cast<int>(X.rows());
  const int d = static_cast<int>(X.cols());
  Eigen::MatrixXd dist2 = Eigen::MatrixXd::Zero(n, n);

  for (int i = 0; i < n; ++i) {
    dist2(i, i) = 0.0;
    for (int j = i + 1; j < n; ++j) {
      double s = 0.0;
      for (int k = 0; k < d; ++k) {
        double diff = X(i, k) - X(j, k);
        s += diff * diff;
      }
      dist2(i, j) = s;
      dist2(j, i) = s;
    }
  }
  return dist2;
}

// RBF kernel: if sigma <= 0, use median heuristic.
inline Eigen::MatrixXd rbf_kernel(const Eigen::MatrixXd& X, double& sigma) {
  const int n = static_cast<int>(X.rows());
  Eigen::MatrixXd dist2 = squared_distance_matrix(X);

  // Median heuristic for sigma if not provided
  if (!(sigma > 0.0)) {
    std::vector<double> vals;
    vals.reserve(n * (n - 1) / 2);
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        vals.push_back(dist2(i, j));
      }
    }
    if (!vals.empty()) {
      std::nth_element(vals.begin(),
                       vals.begin() + vals.size() / 2,
                       vals.end());
      double med = vals[vals.size() / 2];
      if (!std::isfinite(med) || med <= 0.0) med = 1.0;
      sigma = std::sqrt(med / 2.0);
    } else {
      sigma = 1.0;
    }
  }

  Eigen::MatrixXd K(n, n);
  const double denom = 2.0 * sigma * sigma;
  for (int i = 0; i < n; ++i) {
    K(i, i) = 1.0;
    for (int j = i + 1; j < n; ++j) {
      double val = std::exp(-dist2(i, j) / denom);
      K(i, j) = val;
      K(j, i) = val;
    }
  }
  return K;
}

// Simple linear kernel: K = X X^T
inline Eigen::MatrixXd linear_kernel(const Eigen::MatrixXd& X) {
  return X * X.transpose();
}

// Intersection kernel
// Rows are samples, columns are features (non-negative, e.g. histograms)
inline Eigen::MatrixXd intersection_kernel(const Eigen::MatrixXd& X) {
  const int n = static_cast<int>(X.rows());
  const int d = static_cast<int>(X.cols());
  Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n, n);

  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      double s = 0.0;
      for (int k = 0; k < d; ++k) {
        s += std::min(X(i, k), X(j, k));
      }
      K(i, j) = s;
      K(j, i) = s;
    }
  }
  return K;
}

enum class KernelType { RBF, LINEAR, INTERSECTION };

inline KernelType kernel_from_string(const std::string& name) {
  if (name == "rbf" || name == "RBF") return KernelType::RBF;
  if (name == "linear" || name == "LINEAR") return KernelType::LINEAR;
  if (name == "intersection" || name == "INTERSECTION") return KernelType::INTERSECTION;
  // default
  return KernelType::RBF;
}

// Unified interface: given data matrix X and kernel type, return kernel matrix.
// sigma is only meaningful for RBF; for others we set it to NaN.
inline Eigen::MatrixXd kernel_matrix(const Eigen::MatrixXd& X,
                            const std::string& kernel_name,
                            double& sigma_used) {
  KernelType k = kernel_from_string(kernel_name);
  switch (k) {
  case KernelType::RBF: {
    double sigma = sigma_used;  // may be <=0 (auto)
    Eigen::MatrixXd K = rbf_kernel(X, sigma);
    sigma_used = sigma;
    return K;
  }
  case KernelType::LINEAR:
    sigma_used = std::numeric_limits<double>::quiet_NaN();
    return linear_kernel(X);
  case KernelType::INTERSECTION:
    sigma_used = std::numeric_limits<double>::quiet_NaN();
    return intersection_kernel(X);
  default: {
      double sigma = sigma_used;
      Eigen::MatrixXd K = rbf_kernel(X, sigma);
      sigma_used = sigma;
      return K;
    }
  }
}

// ============================================================
// 2. Segment cost matrix (kernel least squares loss)
// ============================================================
//
// C(i, j) = sum_{t=i}^j ||Phi(X_t) - mean||^2
//         = sum diag(K) - (1/len) * sum_{p=i}^j sum_{q=i}^j K(p, q)
// (R indices are 1-based; here everything is 0-based.)

inline Eigen::MatrixXd segment_cost_matrix_from_K(const Eigen::MatrixXd& K) {
  const int n = static_cast<int>(K.rows());
  const double INF = std::numeric_limits<double>::infinity();

  // Prefix sums of the diagonal
  VectorXd diagK = K.diagonal();
  VectorXd csum_diag(n + 1);
  csum_diag(0) = 0.0;
  for (int i = 0; i < n; ++i) {
    csum_diag(i + 1) = csum_diag(i) + diagK(i);
  }

  // 2D prefix sums: S(i+1, j+1) = sum_{p<=i, q<=j} K(p,q)
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(n + 1, n + 1);
  for (int i = 0; i < n; ++i) {
    double row_cum = 0.0;
    for (int j = 0; j < n; ++j) {
      row_cum += K(i, j);
      S(i + 1, j + 1) = S(i, j + 1) + row_cum;
    }
  }

  auto block_sum = [&](int i, int j) {
    // i, j are 0-based, i <= j
    // sum_{p=i}^j sum_{q=i}^j K(p,q)
    double res = S(j + 1, j + 1)
    - S(i,     j + 1)
    - S(j + 1, i)
    + S(i,     i);
    return res;
  };

  Eigen::MatrixXd C = Eigen::MatrixXd::Constant(n, n, INF);
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      int len = j - i + 1;
      double sum_diag  = csum_diag(j + 1) - csum_diag(i);
      double sum_block = block_sum(i, j);
      C(i, j) = sum_diag - sum_block / static_cast<double>(len);
    }
  }
  return C;
}

// ============================================================
// 3. Dynamic Programming for segmentation
// ============================================================
//
// dp[D, j]: minimal cost for first j points (1..j) with D segments
// (R is 1-based indexing; here we use 0-based indices in code)

struct KernelDPResult {
  Eigen::MatrixXd dp;            // (Dmax x n)
  Eigen::MatrixXi last; // (Dmax x n), last[D-1, j] = index t (0-based)
};

inline KernelDPResult kernel_segmentation_dp(const Eigen::MatrixXd& C, int Dmax) {
  const int n = static_cast<int>(C.rows());
  const double INF = std::numeric_limits<double>::infinity();

  if (Dmax <= 0) Dmax = n;
  Dmax = std::min(Dmax, n);

  Eigen::MatrixXd dp = Eigen::MatrixXd::Constant(Dmax, n, INF);
  Eigen::MatrixXi last = Eigen::MatrixXi::Constant(Dmax, n, -1);

  // D = 1 -> dp[0, j] = C(0, j)
  for (int j = 0; j < n; ++j) {
    dp(0, j) = C(0, j);
    last(0, j) = -1;   // no previous split
  }

  // For D >= 2
  for (int D = 2; D <= Dmax; ++D) {
    int row = D - 1;
    int prev_row = D - 2;
    // j must be at least D-1 (each segment >= 1 point)
    for (int j = D - 1; j < n; ++j) {
      double best_val = INF;
      int best_t = -1;
      for (int t = D - 2; t <= j - 1; ++t) {
        double val = dp(prev_row, t) + C(t + 1, j);
        if (val < best_val) {
          best_val = val;
          best_t = t;
        }
      }
      dp(row, j)  = best_val;
      last(row, j) = best_t;
    }
  }

  KernelDPResult res;
  res.dp = dp;
  res.last = last;
  return res;
}

// Backtracking to get segment ends for a given D (1-based indices)
inline std::vector<int> backtrack_segmentation(const Eigen::MatrixXi& last,
                                               int D,
                                               int n)
{
  std::vector<int> ends(D);
  int cur_end = n - 1;  // last index (0-based)
  for (int d = D; d >= 1; --d) {
    int row = d - 1;
    ends[d - 1] = cur_end + 1; // 1-based
    cur_end = last(row, cur_end);
  }
  return ends;
}

// ============================================================
// 4. Estimate vmax from K
// ============================================================
//
// Estimate v_max using left and right margins of the time series.
// v_max ~ max( tr(Sigma_left), tr(Sigma_right) ) where
// tr(Sigma) ≈ E||Φ(X)||^2 - ||mean||^2.

inline double estimate_vmax_from_K(const Eigen::MatrixXd& K,
                                   double t_left  = 0.05,
                                   double t_right = 0.95)
{
  const int n = static_cast<int>(K.rows());
  int idx1_end   = std::max(1, static_cast<int>(std::floor(t_left  * n)));
  int idx2_start = std::min(n, static_cast<int>(std::ceil (t_right * n)));

  auto est_tr = [&](int start_1based, int end_1based) -> double {
    int m = end_1based - start_1based + 1;
    if (m <= 1) return 0.0;

    int s0 = start_1based - 1;
    int e0 = end_1based   - 1;

    double diag_sum = 0.0;
    for (int i = s0; i <= e0; ++i) {
      diag_sum += K(i, i);
    }
    double diag_mean = diag_sum / static_cast<double>(m);

    double sumK = 0.0;
    for (int i = s0; i <= e0; ++i) {
      for (int j = s0; j <= e0; ++j) {
        sumK += K(i, j);
      }
    }
    double mu2 = sumK / static_cast<double>(m * m);

    return diag_mean - mu2;
  };

  double v1 = est_tr(1, idx1_end);
  double v2 = est_tr(idx2_start, n);
  double vmax = std::max(v1, v2);
  if (!std::isfinite(vmax) || vmax <= 0.0) {
    vmax = 1.0;
  }
  return vmax;
}

// ============================================================
// 5. Low-rank kernel matrix
// ============================================================
//
// Eigen-decomposition of K (symmetric), keep largest L eigenvalues.

inline Eigen::MatrixXd low_rank_kernel_matrix(const Eigen::MatrixXd& K, int L) {
  const int n = static_cast<int>(K.rows());
  if (n != K.cols()) {
    throw std::runtime_error("low_rank_kernel_matrix: K must be square");
  }

  L = std::min(L, n - 1);
  if (L <= 0) {
    throw std::runtime_error("low_rank_kernel_matrix: target rank L must be positive");
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(K);
  if (es.info() != Eigen::Success) {
    throw std::runtime_error("Eigen decomposition failed");
  }

  VectorXd evals = es.eigenvalues();   // ascending
  Eigen::MatrixXd evecs = es.eigenvectors();  // columns are eigenvectors

  Eigen::MatrixXd U(n, L);
  VectorXd lambda(L);
  for (int i = 0; i < L; ++i) {
    int idx = n - 1 - i;          // largest eigenvalues
    double val = evals(idx);
    if (val < 0.0) val = 0.0;
    lambda(i) = val;
    U.col(i) = evecs.col(idx);
  }

  Eigen::MatrixXd Lambda = lambda.asDiagonal();
  Eigen::MatrixXd K_L = U * Lambda * U.transpose();
  K_L = 0.5 * (K_L + K_L.transpose()); // enforce symmetry

  return K_L;
}

// ============================================================
// 6. Main function: Kernel Change-Point Detection
// ============================================================

struct KernelCPResult {
  int D_hat;                        // selected number of segments
  std::vector<int> change_points;   // 1-based indices of CPs
  std::vector<int> ends;            // 1-based segment ends
  VectorXd crit;                      // criterion(D), D=1..Dmax
  VectorXd emp_risk;
  VectorXd penalty;
  VectorXd total_cost;
  Eigen::MatrixXd K;                         // full kernel matrix
  Eigen::MatrixXd K_used;                    // kernel actually used (full or low-rank)
  double sigma_used;
  double vmax;
  Eigen::MatrixXd dp;
  Eigen::MatrixXi last;
  Eigen::MatrixXd X;
  int rank;                         // effective rank used
};

inline KernelCPResult kernel_change_point(const Eigen::MatrixXd& X,
                                          const std::string& kernel_name = "rbf",
                                          double sigma = -1.0,
                                          int Dmax = -1,
                                          double Cpen = 2.0,
                                          double vmax_input = -1.0,
                                          int rank = -1,          // optional low-rank L
                                          double t_left = 0.05,
                                          double t_right = 0.95,
                                          bool verbose = true)
{
  const int n = static_cast<int>(X.rows());
  if (Dmax <= 0) Dmax = std::min(50, n);
  Dmax = std::min(Dmax, n);

  bool is_low_rank = false;

  // Step 1: full-rank kernel
  double sigma_used = sigma;
  Eigen::MatrixXd K_full = kernel_matrix(X, kernel_name, sigma_used);
  Eigen::MatrixXd K_used = K_full;

  // Step 1.5: low-rank approximation
  if (rank > 0 && rank < n) {
    K_used = low_rank_kernel_matrix(K_full, rank);
    is_low_rank = true;
    if (verbose) {
      std::cout << "Using low-rank approximation (L = " << rank << ")\n";
    }
  } else {
    rank = n; // full rank
  }

  // Step 2: cost matrix
  Eigen::MatrixXd Cmat = segment_cost_matrix_from_K(K_used);

  // Step 3 & 4: DP + vmax
  KernelDPResult dp_res = kernel_segmentation_dp(Cmat, Dmax);
  Eigen::MatrixXd dp = dp_res.dp;
  Eigen::MatrixXi last = dp_res.last;

  double vmax = vmax_input;
  if (!(vmax > 0.0)) {
    vmax = estimate_vmax_from_K(K_full, t_left, t_right);
  }

  // Step 5 & 6: criterion & select D
  VectorXd total_cost(Dmax);
  VectorXd emp_risk(Dmax);
  VectorXd penalty(Dmax);
  VectorXd crit(Dmax);

  for (int D = 1; D <= Dmax; ++D) {
    int row = D - 1;
    total_cost(row) = dp(row, n - 1);
    emp_risk(row)   = total_cost(row) / static_cast<double>(n);
    double pen = Cpen * vmax * static_cast<double>(D) / static_cast<double>(n)
      * (1.0 + std::log(static_cast<double>(n) / static_cast<double>(D)));
    penalty(row) = pen;
    crit(row)    = emp_risk(row) + pen;
  }

  // choose D_hat
  int D_hat = 1;
  double best_val = crit(0);
  for (int D = 2; D <= Dmax; ++D) {
    int idx = D - 1;
    if (crit(idx) < best_val) {
      best_val = crit(idx);
      D_hat = D;
    }
  }

  std::vector<int> ends_hat = backtrack_segmentation(last, D_hat, n);
  std::vector<int> cps_hat;
  if (!ends_hat.empty()) {
    cps_hat.assign(ends_hat.begin(), ends_hat.end() - 1);
  }

  if (verbose) {
    std::cout << "\n================ Kernel Change-Point Detection ================\n";
    std::cout << "Method: " << (is_low_rank ? "Low-Rank Approx" : "Full Rank") << "\n";
    std::cout << "Data size n          : " << n << "\n";
    std::cout << "Kernel               : " << kernel_name << "\n";

    if (kernel_name == "rbf" || kernel_name == "RBF") {
      if (!(sigma > 0.0)) {
        std::cout << "Sigma (RBF only)     : auto (" << sigma_used << ")\n";
      } else {
        std::cout << "Sigma (RBF only)     : " << sigma << "\n";
      }
    } else {
      std::cout << "Sigma (RBF only)     : N/A (non-RBF kernel)\n";
    }
    std::cout << "Rank L               : " << rank << "\n";
    std::cout << "Dmax                 : " << Dmax << "\n";
    std::cout << "Penalty constant C   : " << Cpen << "\n";
    std::cout << "Estimated vmax       : " << vmax << "\n";
    std::cout << "--------------------------------------------------------------\n";
    std::cout << "Selected D_hat       : " << D_hat << "\n";
    std::cout << "Estimated CPs        : ";
    if (cps_hat.empty()) {
      std::cout << "None\n";
    } else {
      for (size_t i = 0; i < cps_hat.size(); ++i) {
        std::cout << cps_hat[i] << (i + 1 < cps_hat.size() ? ", " : "\n");
      }
    }
    std::cout << "Segment ends         : ";
    for (size_t i = 0; i < ends_hat.size(); ++i) {
      std::cout << ends_hat[i] << (i + 1 < ends_hat.size() ? ", " : "\n");
    }
    std::cout << "==============================================================\n\n";
  }

  KernelCPResult res;
  res.D_hat         = D_hat;
  res.change_points = cps_hat;
  res.ends          = ends_hat;
  res.crit          = crit;
  res.emp_risk      = emp_risk;
  res.penalty       = penalty;
  res.total_cost    = total_cost;
  res.K             = K_full;
  res.K_used        = K_used;
  res.sigma_used    = sigma_used;
  res.vmax          = vmax;
  res.dp            = dp;
  res.last          = last;
  res.X             = X;
  res.rank          = rank;
  return res;
}

// ============================================================
// 7. Utility: match count between estimated and true CPs
// ============================================================
//
// Count how many A[i] are within "threshold" from any element in T_star.

inline int calculate_matches(const std::vector<int>& A,
                             const std::vector<int>& T_star,
                             int threshold) {
  if (A.empty() || T_star.empty()) return 0;

  int matched_count = 0;
  for (int a : A) {
    int min_dist = std::numeric_limits<int>::max();
    for (int t : T_star) {
      int d = std::abs(a - t);
      if (d < min_dist) min_dist = d;
    }
    if (min_dist <= threshold) {
      matched_count++;
    }
  }
  return matched_count;
}


// [[Rcpp::export]]
Rcpp::List kernel_change_point_cpp(const Eigen::MatrixXd& X,
                                   std::string kernel = "rbf",
                                   double sigma = -1.0,
                                   int Dmax = 50,
                                   double Cpen = 2.0,
                                   double vmax = -1.0,
                                   int rank = -1,
                                   double t_left = 0.05,
                                   double t_right = 0.95,
                                   bool verbose = true)
{
  auto res = kernel_change_point(
    X, kernel, sigma, Dmax, Cpen, vmax, rank, t_left, t_right, verbose
  );

  return Rcpp::List::create(
    _["D_hat"]         = res.D_hat,
    _["change_points"] = res.change_points,
    _["ends"]          = res.ends,
    _["crit"]          = res.crit,
    _["emp_risk"]      = res.emp_risk,
    _["penalty"]       = res.penalty,
    _["total_cost"]    = res.total_cost
  );
}


