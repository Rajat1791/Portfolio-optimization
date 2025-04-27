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

rolling_results = []
window_train = 252  # ~1 year
window_test = 21    # ~1 month
returns = data.pct_change().dropna()
dates = returns.index

for start in range(0, len(returns) - window_train - window_test, window_test):
    train = returns.iloc[start:start + window_train]
    test = returns.iloc[start + window_train:start + window_train + window_test]

    mean_ret_train = train.mean() * 252
    cov_train = train.cov() * 252

    weights, sharpe, ret, risk = cat_swarm_optimization(mean_ret_train.values, cov_train.values)

    test_returns = test @ weights
    cumulative_return = (1 + test_returns).prod() - 1
    ann_return = cumulative_return * (252 / window_test)
    ann_volatility = test_returns.std() * np.sqrt(252)
    ann_sharpe = (ann_return - RISK_FREE_RATE) / ann_volatility

    rolling_results.append({
        'start_date': dates[start + window_train],
        'end_date': dates[start + window_train + window_test - 1],
        'annualized_return': ann_return,
        'volatility': ann_volatility,
        'sharpe_ratio': ann_sharpe
    })

cso_validation_df = pd.DataFrame(rolling_results)
# Summary stats
print(cso_validation_df.describe())