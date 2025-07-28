import numpy as np
import pandas as pd

# Constants
RISK_FREE_RATE = 0.01  # Annual risk-free rate of return

# Sharpe Ratio Calculation
def sharpe_ratio(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)  # Annual return
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Annual volatility
    return (port_return - RISK_FREE_RATE) / port_volatility

# Objective Function
def objective_function(weights, mean_returns, cov_matrix):
    return -sharpe_ratio(weights, mean_returns, cov_matrix)  # Maximize Sharpe Ratio by minimizing its negative

# Constraint: Weights must sum to 1
constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

# Bounds for each weight
bounds = tuple((0, 1) for asset in range(len(annual_returns)))

# Bacterial Foraging Optimization Simulation (Simplified example)
def bfo_optimize(mean_returns, cov_matrix, population_size=50, max_iterations=100):
    dimensions = len(mean_returns)
    population = np.random.dirichlet(np.ones(dimensions), size=population_size)  # Initialize population

    for _ in range(max_iterations):
        for i in range(population_size):
            weights = population[i]
            perturbations = np.random.normal(0, 0.01, dimensions)  # Small perturbations
            new_weights = weights + perturbations
            new_weights = np.clip(new_weights, 0, 1)  # Enforce bounds
            new_weights /= np.sum(new_weights)  # Normalize weights

            if objective_function(new_weights, mean_returns, cov_matrix) < objective_function(weights, mean_returns, cov_matrix):
                population[i] = new_weights  # Accept new weights

    best_idx = np.argmin([objective_function(w, mean_returns, cov_matrix) for w in population])
    best_weights = population[best_idx]
    return best_weights, sharpe_ratio(best_weights, mean_returns, cov_matrix), np.dot(best_weights, mean_returns), np.sqrt(np.dot(best_weights.T, np.dot(cov_matrix, best_weights)))

optimal_weights, optimal_sharpe, optimal_return, optimal_volatility = bfo_optimize(annual_returns, annual_cov_matrix)

print(f"Optimal Weights: {optimal_weights}")
print(f"Optimal Sharpe Ratio: {optimal_sharpe}")
print(f"Expected Annual Return (%): {optimal_return * 100}")
print(f"Annual Volatility (%): {optimal_volatility * 100}")

Revised BFO code

import numpy as np
import pandas as pd
import yfinance as yf

# Define tickers
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "PSX", "BRK-B", "JNJ", "V", "PG", "JPM",
           "UNH", "INTC", "VZ", "HD", "T", "BAC", "MA", "COP", "PFE", "MRK",
           "NFLX", "WMT", "PEP", "KO", "CSCO", "CMCSA", "XOM", "ABT", "NVDA", "CVX",
           "PYPL", "ORCL", "ACN", "ADBE", "GILD", "NKE", "LLY", "IBM", "TXN", "LMT",
           "NEE", "HON", "BMY", "SLB", "AMGN", "CAT", "TGT", "AMT", "QCOM", "UNP", "GE"]

# Download price data
raw_data = yf.download(tickers, start="2017-12-01", end="2025-07-01")['Adj Close']
returns = raw_data.pct_change().dropna()

# Define market regimes
periods = {
    "Pre-COVID": ("2018-01-01", "2019-12-31"),
    "COVID": ("2020-01-01", "2020-12-31"),
    "Post-COVID": ("2021-01-01", "2022-12-31"),
    "Recent": ("2023-01-01", "2025-07-01"),
    "Overall": ("2018-01-01", "2025-07-01")
}

# Parameters
risk_free_rate = 0.01
transaction_cost = 0.001
pop_size = 50
max_iter = 100

# Sharpe Ratio function with transaction cost penalty
def objective(weights, mean_returns, cov_matrix, prev_weights):
    port_ret = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_ret - risk_free_rate) / port_vol if port_vol != 0 else 0
    tc = transaction_cost * np.sum(np.abs(weights - prev_weights)) if prev_weights is not None else 0
    return -(sharpe - tc)

# BFO Optimizer
def bfo_optimize(mean_returns, cov_matrix, prev_weights, pop_size=50, max_iter=100):
    n = len(mean_returns)
    pop = np.random.dirichlet(np.ones(n), size=pop_size)

    for _ in range(max_iter):
        for i in range(pop_size):
            weights = pop[i]
            perturb = np.random.normal(0, 0.01, n)
            new_weights = np.clip(weights + perturb, 0, 1)
            new_weights /= new_weights.sum()

            if objective(new_weights, mean_returns, cov_matrix, prev_weights) < objective(weights, mean_returns, cov_matrix, prev_weights):
                pop[i] = new_weights

    scores = [objective(w, mean_returns, cov_matrix, prev_weights) for w in pop]
    return pop[np.argmin(scores)]

# Evaluation function
def calculate_metrics(returns, weights):
    port_ret = returns @ weights
    ann_return = np.mean(port_ret) * 252
    ann_vol = np.std(port_ret) * np.sqrt(252)
    var_95 = np.percentile(port_ret, 5)
    cvar_95 = np.mean(port_ret[port_ret <= var_95])
    sharpe_ratio = (ann_return - risk_free_rate) / ann_vol if ann_vol != 0 else 0
    return ann_return, ann_vol, -cvar_95, sharpe_ratio

# Validate across regimes
results = []
prev_weights = None
for label, (start, end) in periods.items():
    data_slice = returns.loc[start:end]
    mu = data_slice.mean() * 252
    cov = data_slice.cov() * 252
    weights = bfo_optimize(mu.values, cov.values, prev_weights, pop_size, max_iter)
    ann_ret, ann_vol, cvar, sharpe = calculate_metrics(data_slice, weights)
    results.append([label, round(ann_ret, 4), round(ann_vol, 4), round(cvar, 4), round(sharpe, 4)])
    prev_weights = weights  # update for next period

# Display result
df_results = pd.DataFrame(results, columns=["Period", "Ann.Return", "Ann.Vol", "CVaR(95%)", "Sharpe"])
print(df_results)
import matplotlib.pyplot as plt

# Plot settings
metrics = ["Ann.Return", "Ann.Vol", "CVaR(95%)", "Sharpe"]
x = np.arange(len(df_results["Period"]))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
for i, metric in enumerate(metrics):
    ax.bar(x + i*width, df_results[metric], width, label=metric)

# Aesthetics
ax.set_title("Portfolio Performance Across Market Regimes (BFO with Transaction Cost)", fontsize=14)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(df_results["Period"], rotation=45)
ax.set_ylabel("Metric Value")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
