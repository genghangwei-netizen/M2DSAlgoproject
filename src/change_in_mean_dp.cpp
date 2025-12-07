// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

using namespace Rcpp;
using namespace Eigen;

#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <chrono>
#include <functional>
#include <iostream>


// ============================================================
// 1. Helper: estimate noise variance via differenced variance
// ============================================================
//
// estimate_sigma_sq(data) = sum( (y_i - y_{i-1})^2 ) / (2 * (T-1))

inline double estimate_sigma_sq(const VectorXd& data) {
  const int T = static_cast<int>(data.size());
  if (T < 2) {
    return 1.0;
  }

  double sum_diff2 = 0.0;
  for (int i = 1; i < T; ++i) {
    double d = data(i) - data(i - 1);
    sum_diff2 += d * d;
  }

  return sum_diff2 / (2.0 * (T - 1));
}

// ============================================================
// 2. Helper: precompute prefix sums for O(1) segment cost
// ============================================================
//
// cost_segment(s, t) = sum_{i=s+1}^t (y_i - mean)^2
// with s in [0..T-1], t in [1..T], s < t
// Implementation uses prefix sums over data and data^2.

struct PrecomputedCost {
  VectorXd sums;     // length T+1, sums[0] = 0, sums[k] = sum_{i=0..k-1} data[i]
  VectorXd sums_sq;  // same for squares

  // cost of segment (s, t] in terms of positions 0..T
  // segment indices in data: s..t-1 (inclusive)
  double cost_segment(int s, int t) const {
    int n = t - s;
    if (n <= 0) return 0.0;

    double sum_y    = sums[t]    - sums[s];
    double sum_y_sq = sums_sq[t] - sums_sq[s];
    double cost     = sum_y_sq - (sum_y * sum_y) / static_cast<double>(n);
    return cost;
  }
};

inline PrecomputedCost precompute_cost(const VectorXd& data) {
  const int T = static_cast<int>(data.size());
  PrecomputedCost pc;
  pc.sums    = VectorXd::Zero(T + 1);
  pc.sums_sq = VectorXd::Zero(T + 1);

  for (int t = 0; t < T; ++t) {
    pc.sums[t + 1]    = pc.sums[t]    + data(t);
    pc.sums_sq[t + 1] = pc.sums_sq[t] + data(t) * data(t);
  }
  return pc;
}

// ============================================================
// 3. O(T^2) Dynamic Programming for penalized change-point detection
// ============================================================
//
// Equivalent of dp_changepoint(data, beta) in R.

struct DPMeanResult {
  double cost;                 // optimal total cost Q_T
  int    K_opt;                // optimal number of change points
  std::vector<int> cpts;       // positions of change points (0-based, like R's s)
};

inline DPMeanResult dp_changepoint(const VectorXd& data, double beta) {
  const int T = static_cast<int>(data.size());
  PrecomputedCost pc = precompute_cost(data);

  // Q[t+1] stores Q_t (optimal cost up to t), with Q_0 = 0
  std::vector<double> Q(T + 1, 0.0);
  // P[t+1] stores optimal split position s_opt for Q_t
  std::vector<int> P(T + 1, 0);

  Q[0] = 0.0;   // Q_0

  // DP loop (O(T^2))
  for (int t = 1; t <= T; ++t) {
    double min_Q = std::numeric_limits<double>::infinity();
    int s_opt = 0;

    for (int s = 0; s < t; ++s) {
      double cost_last_segment = pc.cost_segment(s, t);
      double current_cost = Q[s] + cost_last_segment + beta;

      if (current_cost < min_Q) {
        min_Q = current_cost;
        s_opt = s;
      }
    }

    Q[t] = min_Q;
    P[t] = s_opt;
  }

  // Backtracking
  std::vector<int> changepoints;
  int current_cp_index = T;

  while (P[current_cp_index] != 0) {
    int cp_pos = P[current_cp_index];
    changepoints.insert(changepoints.begin(), cp_pos); // prepend
    current_cp_index = cp_pos;
  }

  DPMeanResult res;
  res.cost  = Q[T];
  res.K_opt = static_cast<int>(changepoints.size());
  res.cpts  = changepoints;
  return res;
}

// ============================================================
// 4. Scan over beta values (equivalent of dp_get_scan_data)
// ============================================================

struct BetaScanRow {
  double beta;
  double cost;
  int    K_opt;
};

inline std::vector<BetaScanRow> dp_get_scan_data(const VectorXd& data,
                                                 const std::vector<double>& betas)
{
  std::vector<BetaScanRow> results;
  results.reserve(betas.size());

  for (double beta_val : betas) {
    DPMeanResult r = dp_changepoint(data, beta_val);

    BetaScanRow row;
    row.beta  = beta_val;
    row.cost  = r.cost;
    row.K_opt = r.K_opt;
    results.push_back(row);
  }

  return results;
}

// ============================================================
// 5. Automatic beta choice (AIC, BIC, HQ, Killick, noise-based)
// ============================================================

enum class PenaltyType {
  AIC,
  BIC,
  HQ,
  Killick2LogN,
  LowNoise,
  MidNoise,
  HighNoise
};

inline PenaltyType penalty_from_string(const std::string& s) {
  if (s == "AIC")           return PenaltyType::AIC;
  if (s == "BIC")           return PenaltyType::BIC;
  if (s == "HQ")            return PenaltyType::HQ;
  if (s == "Killick2LogN")  return PenaltyType::Killick2LogN;
  if (s == "LowNoise")      return PenaltyType::LowNoise;
  if (s == "MidNoise")      return PenaltyType::MidNoise;
  if (s == "HighNoise")     return PenaltyType::HighNoise;
  // default
  return PenaltyType::BIC;
}

inline DPMeanResult dp_auto_beta(const VectorXd& data,
                                 const std::string& penalty_type_str = "BIC",
                                 int p = 1)
{
  const int T = static_cast<int>(data.size());
  PenaltyType pt = penalty_from_string(penalty_type_str);

  double beta = 0.0;

  switch (pt) {
  case PenaltyType::AIC:
    beta = 2.0 * p;
    break;
  case PenaltyType::BIC:
    beta = p * std::log(static_cast<double>(T));
    break;
  case PenaltyType::HQ:
    beta = 2.0 * p * std::log(std::log(static_cast<double>(T)));
    break;
  case PenaltyType::Killick2LogN:
    beta = 2.0 * std::log(static_cast<double>(T));
    break;
  case PenaltyType::LowNoise:
  case PenaltyType::MidNoise:
  case PenaltyType::HighNoise: {
    double sigma_sq_hat = estimate_sigma_sq(data);
    if (pt == PenaltyType::HighNoise) {
      beta = 2.0 * sigma_sq_hat * std::log(static_cast<double>(T));
    } else if (pt == PenaltyType::MidNoise) {
      beta = 1.0 * sigma_sq_hat * std::log(static_cast<double>(T));
    } else { // LowNoise
      beta = 0.5 * sigma_sq_hat * std::log(static_cast<double>(T));
    }
    break;
  }
  default:
    beta = p * std::log(static_cast<double>(T));
    break;
  }

  std::cout << "Using penalty type: " << penalty_type_str
            << " with Beta = " << beta << "\n";

  return dp_changepoint(data, beta);
}

// ============================================================
// 6. Penalty analysis data (C++ version of dp_plot_penalty_analysis)
// ============================================================

struct BetaLineInfo {
  double beta;
  std::string label;
};

struct PenaltyAnalysisResult {
  std::vector<BetaScanRow> scan_results;    // (beta, cost, K_opt)
  std::vector<BetaLineInfo> standard_betas; // AIC/HQ/BIC/Killick/Yao & Au
};

inline PenaltyAnalysisResult dp_penalty_analysis(const VectorXd& data,
                                                 double max_beta_scan = 20.0,
                                                 int p = 1)
{
  const int T = static_cast<int>(data.size());

  double beta_aic     = 2.0 * p;
  double beta_hq      = 2.0 * p * std::log(std::log(static_cast<double>(T)));
  double beta_bic     = p * std::log(static_cast<double>(T));
  double beta_killick = 2.0 * std::log(static_cast<double>(T));

  double sigma_sq_corrected = estimate_sigma_sq(data);
  double beta_yao_au_corrected = 2.0 * sigma_sq_corrected * std::log(static_cast<double>(T));

  std::vector<BetaLineInfo> beta_lines = {
    {beta_aic,     "AIC (2)"},
    {beta_hq,      "HQ"},
    {beta_bic,     "BIC (log n)"},
    {beta_killick, "Killick (2 log n)"},
    {beta_yao_au_corrected, "Yao & Au"}
  };

  std::vector<double> beta_range;
  for (double b = 0.5; b <= max_beta_scan + 1e-12; b += 0.5) {
    beta_range.push_back(b);
  }

  std::vector<BetaScanRow> scan_results = dp_get_scan_data(data, beta_range);

  std::cout << "\nOptimal Change Point Count (K_opt) at Standard Penalties:\n";
  std::cout << "Penalty_Type,\tBeta_Value,\tK_Optimal\n";

  for (const auto& bl : beta_lines) {
    DPMeanResult r = dp_changepoint(data, bl.beta);
    std::cout << bl.label << ",\t"
              << bl.beta << ",\t"
              << r.K_opt << "\n";
  }

  PenaltyAnalysisResult out;
  out.scan_results   = std::move(scan_results);
  out.standard_betas = std::move(beta_lines);
  return out;
}

// ============================================================
// 7. PELT
// ============================================================

struct PELTResult {
  double cost;
  int    K_opt;
  std::vector<int> cpts;
};

inline PELTResult dp_changepoint_pelt(const VectorXd& data, double beta) {
  const int T = static_cast<int>(data.size());
  if (T == 0) {
    return {0.0, 0, {}};
  }

  PrecomputedCost pc = precompute_cost(data);

  std::vector<double> Q(T + 1, 0.0);
  std::vector<int>    P(T + 1, 0);

  Q[0] = 0.0;

  std::vector<int> R;
  R.push_back(0);

  for (int t = 1; t <= T; ++t) {
    double min_Q = std::numeric_limits<double>::infinity();
    int s_opt = 0;

    for (int s : R) {
      double cost_last_segment = pc.cost_segment(s, t);
      double current_cost = Q[s] + cost_last_segment + beta;
      if (current_cost < min_Q) {
        min_Q = current_cost;
        s_opt = s;
      }
    }

    Q[t] = min_Q;
    P[t] = s_opt;

    std::vector<int> R_new;
    R_new.reserve(R.size() + 1);

    for (int s : R) {
      if (Q[t] > Q[s] + pc.cost_segment(s, t)) {
        R_new.push_back(s);
      }
    }

    R_new.push_back(t);
    R.swap(R_new);
  }

  std::vector<int> changepoints;
  int current_cp_index = T;

  while (P[current_cp_index] != 0) {
    int cp_pos = P[current_cp_index];
    changepoints.insert(changepoints.begin(), cp_pos);
    current_cp_index = cp_pos;
  }

  PELTResult res;
  res.cost  = Q[T];
  res.K_opt = static_cast<int>(changepoints.size());
  res.cpts  = changepoints;
  return res;
}

// ============================================================
// 8. Timing utility (optionnel)
// ============================================================

template <typename Func>
struct TimedCallResult {
  typename std::result_of<Func()>::type result;
  double time_ms;
};

template <typename Func>
TimedCallResult<Func> time_function_call(Func f) {
  using clock = std::chrono::high_resolution_clock;
  auto start = clock::now();
  auto result = f();
  auto end = clock::now();
  std::chrono::duration<double> diff = end - start;
  TimedCallResult<Func> out{std::move(result), diff.count() * 1000.0};
  return out;
}

// ============================================================
// 9. Wrappers exportés vers R
// ============================================================

// [[Rcpp::export]]
Rcpp::List dp_changepoint_cpp(const Eigen::VectorXd& data, double beta) {
  auto res = dp_changepoint(data, beta);
  return Rcpp::List::create(
    _["cost"]         = res.cost,
    _["K_opt"]        = res.K_opt,
    _["changepoints"] = res.cpts
  );
}

// [[Rcpp::export]]
Rcpp::List dp_changepoint_pelt_cpp(const Eigen::VectorXd& data, double beta) {
  auto res = dp_changepoint_pelt(data, beta);
  return Rcpp::List::create(
    _["cost"]         = res.cost,
    _["K_opt"]        = res.K_opt,
    _["changepoints"] = res.cpts
  );
}
