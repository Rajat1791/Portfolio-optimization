# Constants
RISK_FREE_RATE = 0.01  # Annual risk-free rate

# Sharpe Ratio Calculation
def sharpe_ratio(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)  # Portfolio return
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Portfolio volatility
    return (port_return - RISK_FREE_RATE) / port_volatility

# Objective Function to Minimize (negative of Sharpe Ratio)
def objective_function(weights, mean_returns, cov_matrix):
    return -sharpe_ratio(weights, mean_returns, cov_matrix)

# Define bounds and constraints for the optimizer
bounds = tuple((0, 1) for asset in range(len(annual_returns)))
constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

def fireworks_algorithm(mean_returns, cov_matrix, population_size=50, m=50, a=0.04, b=0.8, max_generations=100):
    dimensions = len(mean_returns)
    population = np.random.dirichlet(np.ones(dimensions), size=population_size)

    for generation in range(max_generations):
        for i in range(population_size):
            base_firework = population[i]
            sparks = []

            # Generate sparks by explosion
            num_sparks = int(m * (1 - a + a * (generation / max_generations)))
            amplitudes = b * (1 - generation / max_generations)

            for _ in range(num_sparks):
                spark = base_firework + np.random.normal(0, amplitudes, dimensions)
                spark = np.clip(spark, 0, 1)
                spark /= np.sum(spark)
                sparks.append(spark)

            # Select the best spark based on Sharpe ratio
            best_spark = min(sparks, key=lambda x: objective_function(x, mean_returns, cov_matrix))
            if objective_function(best_spark, mean_returns, cov_matrix) < objective_function(base_firework, mean_returns, cov_matrix):
                population[i] = best_spark

    # Find the best solution in the final population
    best_idx = np.argmin([objective_function(w, mean_returns, cov_matrix) for w in population])
    best_weights = population[best_idx]
    return best_weights, sharpe_ratio(best_weights, mean_returns, cov_matrix), np.dot(best_weights, mean_returns), np.sqrt(np.dot(best_weights.T, np.dot(cov_matrix, best_weights)))

# Run the Fireworks Algorithm
optimal_weights, optimal_sharpe, optimal_return, optimal_volatility = fireworks_algorithm(annual_returns, annual_cov_matrix)

print(f"Optimal Weights: {optimal_weights}")
print(f"Optimal Sharpe Ratio: {optimal_sharpe}")
print(f"Expected Annual Return (%): {optimal_return * 100}")
print(f"Annual Volatility (%): {optimal_volatility * 100}")

Revised FWA code

!pip install yfinance

import numpy as np
import pandas as pd
import yfinance as yf

# Constants
RISK_FREE_RATE = 0.01
TC = 0.001
DIMENSIONS = 51
POP_SIZE = 20
ITERATIONS = 50
np.random.seed(42)

# Load data
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "PSX", "BRK-B", "JNJ", "V", "PG", "JPM",
           "UNH", "INTC", "VZ", "HD", "T", "BAC", "MA", "COP", "PFE", "MRK",
           "NFLX", "WMT", "PEP", "KO", "CSCO", "CMCSA", "XOM", "ABT", "NVDA", "CVX",
           "PYPL", "ORCL", "ACN", "ADBE", "GILD", "NKE", "LLY", "IBM", "TXN", "LMT",
           "NEE", "HON", "BMY", "SLB", "AMGN", "CAT", "TGT", "AMT", "QCOM", "UNP", "GE"]

data = yf.download(tickers, start="2017-12-01", end="2025-07-01")['Adj Close']
returns = data.pct_change().dropna()

# Define regimes
regimes = {
    "Pre-COVID": ("2018-01-01", "2019-12-31"),
    "COVID": ("2020-01-01", "2020-12-31"),
    "Post-COVID": ("2021-01-01", "2022-12-31"),
    "Recent": ("2023-01-01", "2025-07-01"),
    "Overall": ("2018-01-01", "2025-07-01")
}

# Objective function
def objective(w, mu, cov, prev_w=None):
    w = np.clip(w, 0, 1)
    w /= np.sum(w)
    ret = np.dot(w, mu)
    vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    tc_penalty = TC * np.sum(np.abs(w - prev_w)) if prev_w is not None else 0
    return -((ret - RISK_FREE_RATE) / vol - tc_penalty)

# Lightweight Fireworks Algorithm
def fireworks_optimize(mu, cov, prev_w=None):
    pop = np.random.dirichlet(np.ones(DIMENSIONS), size=POP_SIZE)
    best_w = None
    best_fitness = float('inf')

    for _ in range(ITERATIONS):
        new_pop = []
        for w in pop:
            w = np.clip(w, 0, 1)
            w /= np.sum(w)
            f = objective(w, mu, cov, prev_w)
            if f < best_fitness:
                best_fitness = f
                best_w = w
            # 3 sparks per candidate
            for _ in range(3):
                spark = w + np.random.normal(0, 0.02, DIMENSIONS)
                spark = np.clip(spark, 0, 1)
                spark /= np.sum(spark)
                new_pop.append(spark)
        pop = np.array(new_pop[:POP_SIZE])  # Truncate to fixed size
    return best_w

# Metrics
def get_metrics(ret, w):
    port_ret = ret @ w
    ann_ret = port_ret.mean() * 252
    ann_vol = port_ret.std() * np.sqrt(252)
    cvar = -np.mean(port_ret[port_ret <= np.percentile(port_ret, 5)])
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol
    return round(ann_ret, 4), round(ann_vol, 4), round(cvar, 4), round(sharpe, 4)

# Run validation
results = []
for regime, (start, end) in regimes.items():
    r = returns.loc[start:end]
    mu = r.mean().values * 252
    cov = r.cov().values * 252
    w = fireworks_optimize(mu, cov)
    metrics = get_metrics(r, w)
    results.append([regime] + list(metrics))

# Output
df_fwa = pd.DataFrame(results, columns=["Period", "Ann.Return", "Ann.Vol", "CVaR(95%)", "Sharpe"])
print(df_fwa)
import matplotlib.pyplot as plt

# Plot settings
metrics = ["Ann.Return", "Ann.Vol", "CVaR(95%)", "Sharpe"]
x = np.arange(len(df_fwa["Period"]))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
for i, metric in enumerate(metrics):
    ax.bar(x + i*width, df_fwa[metric], width, label=metric)

# Aesthetics
ax.set_title("Portfolio Performance Across Market Regimes (FWA with Transaction Cost)", fontsize=14)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(df_fwa["Period"], rotation=45)
ax.set_ylabel("Metric Value")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
