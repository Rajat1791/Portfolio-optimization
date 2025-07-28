import numpy as np

# Constants
RISK_FREE_RATE = 0.01  # Annual risk-free rate

# Sharpe Ratio Calculation
def sharpe_ratio(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)  # Portfolio return
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Portfolio volatility
    return (port_return - RISK_FREE_RATE) / port_volatility

# Objective Function to Maximize (Sharpe Ratio)
def objective_function(weights, mean_returns, cov_matrix):
    return -sharpe_ratio(weights, mean_returns, cov_matrix)  # Negative for minimization

# Cat Swarm Optimization Algorithm Implementation
def cat_swarm_optimization(mean_returns, cov_matrix, num_cats=50, mix_rate=0.2, iterations=100):
    num_assets = len(mean_returns)
    cat_position = np.random.dirichlet(np.ones(num_assets), size=num_cats)
    cat_velocity = np.zeros((num_cats, num_assets))
    best_cat = cat_position[np.argmin([objective_function(cat, mean_returns, cov_matrix) for cat in cat_position])]

    for iteration in range(iterations):
        for i in range(num_cats):
            if np.random.rand() < mix_rate:
                # Seeking Mode
                candidate_position = best_cat + np.random.uniform(-0.1, 0.1, num_assets)
                candidate_position = np.clip(candidate_position, 0, 1)
                candidate_position /= np.sum(candidate_position)
                if objective_function(candidate_position, mean_returns, cov_matrix) < objective_function(cat_position[i], mean_returns, cov_matrix):
                    cat_position[i] = candidate_position
            else:
                # Tracing Mode
                cat_velocity[i] = np.random.uniform(-1, 1, num_assets) * (best_cat - cat_position[i])
                cat_position[i] += cat_velocity[i]
                cat_position[i] = np.clip(cat_position[i], 0, 1)
                cat_position[i] /= np.sum(cat_position[i])

        best_cat = cat_position[np.argmin([objective_function(cat, mean_returns, cov_matrix) for cat in cat_position])]

    best_index = np.argmin([objective_function(cat, mean_returns, cov_matrix) for cat in cat_position])
    best_weights = cat_position[best_index]
    return best_weights, sharpe_ratio(best_weights, mean_returns, cov_matrix), np.dot(best_weights, mean_returns), np.sqrt(np.dot(best_weights.T, np.dot(cov_matrix, best_weights)))

# Using the prepared data
optimal_weights, optimal_sharpe, optimal_return, optimal_volatility = cat_swarm_optimization(annual_returns, annual_cov_matrix)

print(f"Optimal Weights: {optimal_weights}")
print(f"Optimal Sharpe Ratio: {optimal_sharpe:.2f}")
print(f"Expected Annual Return (%): {optimal_return * 100:.2f}")
print(f"Annual Volatility (%): {optimal_volatility * 100:.2f}")

Revised CSO code

import numpy as np
import pandas as pd
import yfinance as yf

# CSO Parameters
MR = 0.2         # Mixing ratio
SMP = 5          # Seeking memory pool
SRD = 0.2        # Seeking range of the selected dimension
CDC = True       # Counts of dimension to change
SPC = False      # Self-position considering
n_cats = 30
iterations = 100
risk_free_rate = 0.01
transaction_cost = 0.001

# Load data
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "PSX", "BRK-B", "JNJ", "V", "PG", "JPM",
           "UNH", "INTC", "VZ", "HD", "T", "BAC", "MA", "COP", "PFE", "MRK",
           "NFLX", "WMT", "PEP", "KO", "CSCO", "CMCSA", "XOM", "ABT", "NVDA", "CVX",
           "PYPL", "ORCL", "ACN", "ADBE", "GILD", "NKE", "LLY", "IBM", "TXN", "LMT",
           "NEE", "HON", "BMY", "SLB", "AMGN", "CAT", "TGT", "AMT", "QCOM", "UNP", "GE"]

data = yf.download(tickers, start="2017-12-01", end="2025-07-01")['Adj Close']
returns = data.pct_change().dropna()

# Objective function
def objective(weights, mean_returns, cov_matrix, prev_weights=None):
    weights = np.clip(weights, 0, 1)
    weights /= np.sum(weights)
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    tc = transaction_cost * np.sum(np.abs(weights - prev_weights)) if prev_weights is not None else 0
    return -((port_return - risk_free_rate) / port_vol) + tc

# CSO Optimization
def cso_optimize(mean_returns, cov_matrix, n_cats=30, iterations=100):
    n_dim = len(mean_returns)
    cats = np.random.dirichlet(np.ones(n_dim), size=n_cats)
    velocities = np.zeros((n_cats, n_dim))
    mode = np.array([1]*int(n_cats * MR) + [0]*(n_cats - int(n_cats * MR)))

                for _ in range(SMP):
                    candidate = cats[i].copy()
                    change_dim = np.random.randint(0, n_dim, 1 if CDC else int(SRD * n_dim))
                    for d in change_dim:
                        perturb = candidate[d] * SRD
                        candidate[d] += np.random.uniform(-perturb, perturb)
                    candidate = np.clip(candidate, 0, 1)
                    candidate /= np.sum(candidate)
                    candidates.append(candidate)
                fitness = [objective(c, mean_returns, cov_matrix) for c in candidates]
                best_idx = np.argmin(fitness)
                cats[i] = candidates[best_idx]
            else:  # Tracing mode
                best_cat = cats[np.argmin([objective(c, mean_returns, cov_matrix) for c in cats])]
                velocities[i] += np.random.rand() * (best_cat - cats[i])
                cats[i] += velocities[i]
                cats[i] = np.clip(cats[i], 0, 1)
                cats[i] /= np.sum(cats[i])
    best_idx = np.argmin([objective(c, mean_returns, cov_matrix) for c in cats])
    return cats[best_idx]

# Define Market Regimes
regimes = {
    "Pre-COVID": ("2018-01-01", "2019-12-31"),
    "COVID": ("2020-01-01", "2020-12-31"),
    "Post-COVID": ("2021-01-01", "2022-12-31"),
    "Recent": ("2023-01-01", "2025-07-01"),
    "Overall": ("2018-01-01", "2025-07-01")
}

# Validation
results = []
for label, (start, end) in regimes.items():
    sub_data = returns.loc[start:end]
    mu = sub_data.mean() * 252
    cov = sub_data.cov() * 252
    weights = cso_optimize(mu.values, cov.values)
    ret = np.dot(weights, mu.values)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov.values, weights)))
    sharpe = (ret - risk_free_rate) / vol
    port_ret = sub_data @ weights
    var_95 = np.percentile(port_ret, 5)
    cvar_95 = np.mean(port_ret[port_ret <= var_95])
    results.append([label, round(ret, 4), round(vol, 4), round(-cvar_95, 4), round(sharpe, 4)])

# Display results
df = pd.DataFrame(results, columns=["Period", "Ann.Return", "Ann.Vol", "CVaR(95%)", "Sharpe"])
print(df)
import matplotlib.pyplot as plt

# Plot settings
metrics = ["Ann.Return", "Ann.Vol", "CVaR(95%)", "Sharpe"]
x = np.arange(len(df["Period"]))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
for i, metric in enumerate(metrics):
    ax.bar(x + i*width, df[metric], width, label=metric)

# Aesthetics
ax.set_title("Portfolio Performance Across Market Regimes (CSO with Transaction Cost)", fontsize=14)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(df["Period"], rotation=45)
ax.set_ylabel("Metric Value")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
