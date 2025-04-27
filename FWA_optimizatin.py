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

window_train = 252  # 1 year
window_test = 21    # 1 month
returns = data.pct_change().dropna()
dates = returns.index

def fireworks_algorithm(mean_returns, cov_matrix, population_size=50, m=50, a=0.04, b=0.8, max_generations=100):
    dimensions = len(mean_returns)
    population = np.random.dirichlet(np.ones(dimensions), size=population_size)

    for generation in range(max_generations):
        for i in range(population_size):
            base_firework = population[i]
            sparks = []
            num_sparks = int(m * (1 - a + a * (generation / max_generations)))
            amplitudes = b * (1 - generation / max_generations)

            for _ in range(num_sparks):
                spark = base_firework + np.random.normal(0, amplitudes, dimensions)
                spark = np.clip(spark, 0, 1)
                spark /= np.sum(spark)
                sparks.append(spark)

            best_spark = min(sparks, key=lambda x: objective_function(x, mean_returns, cov_matrix))
            if objective_function(best_spark, mean_returns, cov_matrix) < objective_function(base_firework, mean_returns, cov_matrix):
                population[i] = best_spark

    best_idx = np.argmin([objective_function(w, mean_returns, cov_matrix) for w in population])
    return population[best_idx]

fwa_results = []

for start in range(0, len(returns) - window_train - window_test, window_test):
    train_data = returns.iloc[start:start + window_train]
    test_data = returns.iloc[start + window_train:start + window_train + window_test]

    mean_returns_train = train_data.mean() * 252
    cov_matrix_train = train_data.cov() * 252

    weights = fireworks_algorithm(mean_returns_train.values, cov_matrix_train.values)

    test_portfolio_returns = test_data @ weights
    cumulative_return = (1 + test_portfolio_returns).prod() - 1
    annualized_return = cumulative_return * (252 / window_test)
    volatility = test_portfolio_returns.std() * np.sqrt(252)
    sharpe = (annualized_return - RISK_FREE_RATE) / volatility

    fwa_results.append({
        "start_date": dates[start + window_train],
        "end_date": dates[start + window_train + window_test - 1],
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe
    })

# Convert to DataFrame
fwa_validation_df = pd.DataFrame(fwa_results)