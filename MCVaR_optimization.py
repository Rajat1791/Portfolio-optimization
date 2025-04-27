data = yf.download(tickers, start="2013-01-01", end="2023-01-01")['Close']
returns = data.pct_change().dropna()

# Calculate expected returns and covariance
expected_returns = returns.mean().values
covariance_matrix = returns.cov().values
n_assets = len(tickers)

# Optimization variables
weights = cp.Variable(n_assets)
portfolio_return = cp.sum(cp.multiply(expected_returns, weights))
portfolio_loss = -returns.values @ weights
VaR = cp.Variable()
condition = VaR + (1/0.05) * cp.pos(portfolio_loss - VaR)  # CVaR calculation
expected_shortfall = cp.sum(condition)/len(returns)

# Regularization and constraints
lambda_reg = 0.1
objective = cp.Maximize(portfolio_return - expected_shortfall - lambda_reg * cp.norm(weights, 2))

# Constraints: Sum of weights equals 1, weights are non-negative
constraints = [cp.sum(weights) == 1, weights >= 0]

# Solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Extract optimized values
optimized_weights = weights.value
optimized_return = portfolio_return.value
optimized_cvar = expected_shortfall.value
optimized_risk = cp.quad_form(weights, covariance_matrix).value

print("Optimal Portfolio Weights:", optimized_weights)
print("Optimized Portfolio Return:", optimized_return)
print("Optimized Portfolio Conditional VaR:", optimized_cvar)
print("Optimized Portfolio Risk (Variance):", optimized_risk)