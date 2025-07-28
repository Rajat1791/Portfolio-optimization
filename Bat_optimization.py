import numpy as np

# Constants
RISK_FREE_RATE = 0.01  # Annual risk-free rate

# Sharpe Ratio Calculation
def sharpe_ratio(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)  # Portfolio return
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Portfolio volatility
    return (port_return - RISK_FREE_RATE) / port_volatility

# Objective Function to Minimize (Negative of Sharpe Ratio)
def objective_function(weights, mean_returns, cov_matrix):
    return -sharpe_ratio(weights, mean_returns, cov_matrix)

# Bat Algorithm Implementation
def bat_algorithm(mean_returns, cov_matrix, population_size=50, max_iterations=100, A=0.5, r=0.5, Qmin=0, Qmax=2, d=0.9):
    num_assets = len(mean_returns)
    # Initialize bats' position and velocity
    positions = np.random.dirichlet(np.ones(num_assets), size=population_size)
    velocity = np.zeros((population_size, num_assets))
    frequencies = np.zeros(population_size)
    loudness = A * np.ones(population_size)  # Initial loudness
    pulse_rate = r * np.ones(population_size)  # Initial pulse rate

    # Best solution found
    best_idx = np.argmin([objective_function(pos, mean_returns, cov_matrix) for pos in positions])
    best = positions[best_idx]

    for t in range(max_iterations):
        for i in range(population_size):
            Q = np.random.uniform(Qmin, Qmax)  # Frequency
            frequencies[i] = Q
            velocities = velocity[i] + (positions[i] - best) * frequencies[i]
            positions[i] += velocities

            # Apply simple bounds/limits
            positions[i] = np.clip(positions[i], 0, 1)
            positions[i] /= np.sum(positions[i])  # Normalize to ensure they sum to 1

            # Pulse rate
            if np.random.rand() > pulse_rate[i]:
                positions[i] = best + 0.001 * np.random.randn(num_assets)
                positions[i] = np.clip(positions[i], 0, 1)
                positions[i] /= np.sum(positions[i])

            # Loudness
            if objective_function(positions[i], mean_returns, cov_matrix) <= objective_function(best, mean_returns, cov_matrix) and np.random.rand() < loudness[i]:
                best = positions[i]
                pulse_rate[i] *= 1 - np.exp(-0.9 * t)
                loudness[i] *= d

    # Recalculate the best solution found
    best_idx = np.argmin([objective_function(pos, mean_returns, cov_matrix) for pos in positions])
    best = positions[best_idx]
    return best, sharpe_ratio(best, mean_returns, cov_matrix), np.dot(best, mean_returns), np.sqrt(np.dot(best.T, np.dot(cov_matrix, best)))

# Using the prepared data
optimal_weights, optimal_sharpe, optimal_return, optimal_volatility = bat_algorithm(annual_returns, annual_cov_matrix)

print(f"Optimal Weights: {optimal_weights}")
print(f"Optimal Sharpe Ratio: {optimal_sharpe}")
print(f"Expected Annual Return (%): {optimal_return * 100}")
print(f"Annual Volatility (%): {optimal_volatility * 100}")

Revised BAT code

import numpy as np
import pandas as pd
import yfinance as yf

# Parameters
risk_free_rate = 0.01
transaction_cost = 0.001
alpha = 0.9  # Loudness
gamma = 0.9  # Pulse rate
fmin, fmax = 0, 2

# Download data
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "PSX", "BRK-B", "JNJ", "V", "PG", "JPM",
           "UNH", "INTC", "VZ", "HD", "T", "BAC", "MA", "COP", "PFE", "MRK",
           "NFLX", "WMT", "PEP", "KO", "CSCO", "CMCSA", "XOM", "ABT", "NVDA", "CVX",
           "PYPL", "ORCL", "ACN", "ADBE", "GILD", "NKE", "LLY", "IBM", "TXN", "LMT",
           "NEE", "HON", "BMY", "SLB", "AMGN", "CAT", "TGT", "AMT", "QCOM", "UNP", "GE"]

data = yf.download(tickers, start="2017-12-01", end="2025-07-01")['Adj Close']
returns = data.pct_change().dropna()
annual_returns = returns.mean() * 252
annual_cov_matrix = returns.cov() * 252

# Objective
def objective(weights, mean_returns, cov_matrix, prev_weights=None):
    weights = np.clip(weights, 0, 1)
    weights /= np.sum(weights)
    port_return = np.dot(weights, mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    tc_penalty = transaction_cost * np.sum(np.abs(weights - prev_weights)) if prev_weights is not None else 0
    sharpe = (port_return - risk_free_rate) / port_volatility
    return -sharpe + tc_penalty

# BAT Optimization
def bat_optimize(mean_returns, cov_matrix, iterations=100, n_bats=30):
    dim = len(mean_returns)
    positions = np.random.dirichlet(np.ones(dim), size=n_bats)
    velocities = np.zeros((n_bats, dim))
    fitness = np.array([objective(w, mean_returns, cov_matrix) for w in positions])
    best = positions[np.argmin(fitness)]

    for _ in range(iterations):
        for i in range(n_bats):
            freq = fmin + (fmax - fmin) * np.random.rand()
            velocities[i] += (positions[i] - best) * freq
            new = positions[i] + velocities[i]
            new = np.clip(new, 0, 1)
            new /= np.sum(new)

            if np.random.rand() > gamma:
                epsilon = np.random.normal(0, 0.001, dim)
                new = best + epsilon
                new = np.clip(new, 0, 1)
                new /= np.sum(new)

            new_fit = objective(new, mean_returns, cov_matrix)
            if new_fit < fitness[i] and np.random.rand() < alpha:
                positions[i] = new
                fitness[i] = new_fit
                if new_fit < objective(best, mean_returns, cov_matrix):
                    best = new
    return best

# Market Regimes
regimes = {
    "Pre-COVID": ("2018-01-01", "2019-12-31"),
    "COVID": ("2020-01-01", "2020-12-31"),
    "Post-COVID": ("2021-01-01", "2022-12-31"),
    "Recent": ("2023-01-01", "2025-07-01"),
    "Overall": ("2018-01-01", "2025-07-01")
}

results = []
for label, (start, end) in regimes.items():
    sub_data = returns.loc[start:end]
    mu = sub_data.mean() * 252
    cov = sub_data.cov() * 252
    weights = bat_optimize(mu.values, cov.values)
    ret = np.dot(weights, mu.values)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov.values, weights)))
    sharpe = (ret - risk_free_rate) / vol
    port_ret = sub_data @ weights
    var_95 = np.percentile(port_ret, 5)
    cvar_95 = np.mean(port_ret[port_ret <= var_95])
    results.append([label, round(ret, 4), round(vol, 4), round(-cvar_95, 4), round(sharpe, 4)])

# Show Results
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
ax.set_title("Portfolio Performance Across Market Regimes (BAT with Transaction Cost)", fontsize=14)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(df["Period"], rotation=45)
ax.set_ylabel("Metric Value")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
