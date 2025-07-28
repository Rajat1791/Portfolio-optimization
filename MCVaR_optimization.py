data = yf.download(tickers, start="2013-01-01", end="2023-01-01")['Adj Close']
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

Revised MCVaR code

# Define market regimes
periods = {
    "Pre-COVID": ("2018-01-01", "2019-12-31"),
    "COVID": ("2020-01-01", "2020-12-31"),
    "Post-COVID": ("2021-01-01", "2022-12-31"),
    "Recent": ("2023-01-01", "2025-07-01"),
    "Overall": ("2018-01-01", "2025-07-01")
}

# Optimization function
def solve_mean_cvar(returns, alpha=0.95, lambda_cvar=0.1, tc=0.001, prev_weights=None):
    mu = returns.mean().values
    T, N = returns.shape
    w = cp.Variable(N)
    VaR = cp.Variable()
    portfolio_loss = -returns.values @ w
    CVaR = VaR + (1 / ((1 - alpha) * T)) * cp.sum(cp.pos(portfolio_loss - VaR))
    if prev_weights is None:
        tc_penalty = 0
    else:
        delta_w = w - prev_weights
        tc_penalty = tc * cp.norm1(delta_w)
    objective = cp.Maximize(mu @ w - lambda_cvar * CVaR - tc_penalty)
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)
    return w.value

# Evaluation function
def calculate_metrics(returns, weights):
    port_ret = returns @ weights
    ann_return = np.mean(port_ret) * 252
    ann_vol = np.std(port_ret) * np.sqrt(252)
    var_95 = np.percentile(port_ret, 5)
    cvar_95 = np.mean(port_ret[port_ret <= var_95])
    sharpe_ratio = ann_return / ann_vol if ann_vol != 0 else 0
    return ann_return, ann_vol, -cvar_95, sharpe_ratio

# Load and process data
raw_data = yf.download(tickers, start="2017-12-01", end="2025-07-01")['Adj Close']
returns = raw_data.pct_change().dropna()

# Loop over all regimes
results = []
for label, (start, end) in periods.items():
    data_slice = returns.loc[start:end]
    weights = solve_mean_cvar(data_slice)
    metrics = calculate_metrics(data_slice, weights)
    results.append([label] + [round(m, 4) for m in metrics])

# Show results
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
ax.set_title("Portfolio Performance Across Market Regimes (MCVaR with Transaction Cost)", fontsize=14)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(df_results["Period"], rotation=45)
ax.set_ylabel("Metric Value")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
