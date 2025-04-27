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

import numpy as np
import pandas as pd

# Assuming df is your price DataFrame with assets as columns and dates as index
# Adjusted Close prices only
window_train = 252  # 1 year
window_test = 21    # 1 month
risk_free_rate = 0.01

returns = data.pct_change().dropna()
dates = returns.index
n_assets = returns.shape[1]

def sharpe_ratio(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return (port_return - risk_free_rate) / port_volatility

def objective_function(weights, mean_returns, cov_matrix):
    return -sharpe_ratio(weights, mean_returns, cov_matrix)

def bfo_optimize(mean_returns, cov_matrix, population_size=50, max_iterations=100):
    dimensions = len(mean_returns)
    population = np.random.dirichlet(np.ones(dimensions), size=population_size)
    for _ in range(max_iterations):
        for i in range(population_size):
            weights = population[i]
            perturbations = np.random.normal(0, 0.01, dimensions)
            new_weights = weights + perturbations
            new_weights = np.clip(new_weights, 0, 1)
            new_weights /= np.sum(new_weights)
            if objective_function(new_weights, mean_returns, cov_matrix) < objective_function(weights, mean_returns, cov_matrix):
                population[i] = new_weights
    best_idx = np.argmin([objective_function(w, mean_returns, cov_matrix) for w in population])
    return population[best_idx]

# Store validation results
results = []

for start in range(0, len(returns) - window_train - window_test, window_test):
    train_data = returns.iloc[start:start + window_train]
    test_data = returns.iloc[start + window_train:start + window_train + window_test]

    mean_returns_train = train_data.mean() * 252
    cov_matrix_train = train_data.cov() * 252

    # Optimize weights using BFO
    weights = bfo_optimize(mean_returns_train.values, cov_matrix_train.values)

    # Test portfolio on unseen data
    test_portfolio_returns = test_data @ weights
    cumulative_return = (1 + test_portfolio_returns).prod() - 1
    annualized_return = cumulative_return * (252 / window_test)
    volatility = test_portfolio_returns.std() * np.sqrt(252)
    sharpe = (annualized_return - risk_free_rate) / volatility

    results.append({
        "start_date": dates[start + window_train],
        "end_date": dates[start + window_train + window_test - 1],
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe
    })

# Convert to DataFrame
validation_df = pd.DataFrame(results)

# Summary stats
print(validation_df.describe())