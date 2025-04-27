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

rolling_results = []
window_train = 252
window_test = 21
returns = data.pct_change().dropna()
dates = returns.index

for start in range(0, len(returns) - window_train - window_test, window_test):
    train = returns.iloc[start:start + window_train]
    test = returns.iloc[start + window_train:start + window_train + window_test]

    mean_ret_train = train.mean() * 252
    cov_train = train.cov() * 252

    weights, ret, risk, sharpe = bat_algorithm(mean_ret_train.values, cov_train.values)
    test_returns = test @ weights
    cumulative_return = (1 + test_returns).prod() - 1
    ann_return = cumulative_return * (252 / window_test)
    ann_volatility = test_returns.std() * np.sqrt(252)
    sharpe_ratio = (ann_return - RISK_FREE_RATE) / ann_volatility

    rolling_results.append({
        'start_date': dates[start + window_train],
        'end_date': dates[start + window_train + window_test - 1],
        'annualized_return': ann_return,
        'volatility': ann_volatility,
        'sharpe_ratio': sharpe_ratio
    })

bat_validation_df = pd.DataFrame(rolling_results)
# Summary stats
print(bat_validation_df.describe())
