import numpy as np
import matplotlib.pyplot as plt
import torch
import cvxpy as cp
from tqdm import tqdm
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from data_tools import PortfolioDataset, download_data
from dateutil.relativedelta import relativedelta
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, train_loader, test_loader, criterion, verbose=True, device=device):
    model.eval()
    losses_over_train = []
    with torch.no_grad():
        for A, y, beta_true in train_loader:
            A, y, beta_true = A.to(device), y.to(device), beta_true.to(device)
            beta_pred, lambda_pred = model(A, y)
            loss = criterion(beta_pred, beta_true)
            losses_over_train.append(loss.item())
    if verbose:
        print(f'Train Loss: {sum(losses_over_train) / len(train_loader):.6f}')

    losses_over_test = []
    with torch.no_grad():
        for A, y, beta_true in test_loader:
            A, y, beta_true = A.to(device), y.to(device), beta_true.to(device)
            beta_pred, lambda_pred = model(A, y)
            loss = criterion(beta_pred, beta_true)
            losses_over_test.append(loss.item())
    if verbose:
        print(f'Test Loss: {sum(losses_over_test) / len(test_loader):.6f}')

    return losses_over_train, losses_over_test


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device=device):
    model.train()
    train_losses = []
    test_losses = []
    memory_stored = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", position=0)
        for _, (A, y, beta_true) in progress_bar:
            A, y, beta_true = A.to(device), y.to(device), beta_true.to(device)
            optimizer.zero_grad()
            init_memory = torch.cuda.memory_allocated()
            beta_pred, lambda_pred = model(A, y)
            after_forward_memory = torch.cuda.memory_allocated()
            memory_stored += after_forward_memory - init_memory
            loss = criterion(beta_pred, beta_true)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Running Loss: {running_loss / len(train_loader):.6f}')
        losses_over_train, losses_over_test = evaluate_model(model, train_loader, test_loader, criterion)
        train_losses.append(sum(losses_over_train) / len(train_loader))
        test_losses.append(sum(losses_over_test) / len(test_loader))
    print(f"Average memory stored from forward pass: {(memory_stored/(len(train_loader) * num_epochs)) / 1e6:.2f} MB")  # Needs to run on CUDA!
    return train_losses, test_losses


def lasso_regression_error_fixed_lambda(lambda_, test_loader):
    mses = []

    for entry in test_loader:
        As, ys, betas = entry
        for i in range(betas.shape[0]):
            A = As[i]
            y = ys[i]
            beta_true = betas[i]

            lasso = Lasso(alpha=lambda_, fit_intercept=False, max_iter=10000)
            lasso.fit(A, y)

            beta_pred = lasso.coef_

            mse = mean_squared_error(beta_true, beta_pred)
            mses.append(mse)

    average_mse = np.mean(mses)
    print(f"(Lasso regression - Test Set) Average MSE with L1 penalty: {average_mse:.6f}")


def backtest(tickers, rebalance_dates, model):
    start_date = (rebalance_dates[0] - relativedelta(months=2)).strftime('%Y-%m-%d')
    end_date = rebalance_dates[-1].strftime('%Y-%m-%d')
    all_data = download_data(tickers, start_date=start_date, end_date=end_date)

    portfolio_values_pred = [1.0]
    portfolio_values_real = [1.0]
    portfolio_values_equal = [1.0]
    dates = [rebalance_dates[0]]
    portfolio_returns_pred = []
    portfolio_returns_real = []
    portfolio_returns_equal = []

    for i in range(len(rebalance_dates) - 1):
        rebalance_date = rebalance_dates[i]
        next_rebalance_date = rebalance_dates[i + 1]

        estimation_start_date = rebalance_date - relativedelta(months=2)
        estimation_end_date = rebalance_date
        holding_start_date = rebalance_date
        holding_end_date = next_rebalance_date

        returns_estimation = all_data.loc[estimation_start_date:estimation_end_date]
        returns_holding = all_data.loc[holding_start_date:holding_end_date]

        returns = returns_estimation.values
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        subseries_len = 28
        dataset = PortfolioDataset(returns_tensor, subseries_len=subseries_len)

        n_assets = len(tickers)

        with torch.no_grad():
            features_batch, mu_tags_batch, sigma_tags_batch = dataset[0]
            features_batch = features_batch.unsqueeze(0)

            mu_hat, L_hat = model(features_batch)
            Sigma_hat = torch.matmul(L_hat, L_hat.transpose(1, 2))
            epsilon = 1e-5

            mu_pred = mu_hat[0].cpu().numpy()
            Sigma_pred = Sigma_hat[0].cpu().numpy()
            Sigma_pred += epsilon * np.eye(n_assets)

            mu_real = np.mean(returns, axis=0)
            Sigma_real = np.cov(returns, rowvar=False)
            Sigma_real += epsilon * np.eye(n_assets)

        def mean_variance_optimization(mu, Sigma, risk_aversion=1.0):
            n = len(mu)
            w = cp.Variable(n)
            objective = cp.Maximize(mu @ w - risk_aversion / 2 * cp.quad_form(w, Sigma))
            constraints = [cp.sum(w) == 1, w >= 0]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.SCS)
            return w.value

        w_pred = mean_variance_optimization(mu_pred, Sigma_pred)
        w_real = mean_variance_optimization(mu_real, Sigma_real)

        w_pred = np.maximum(w_pred, 0)
        w_pred /= np.sum(w_pred)
        w_real = np.maximum(w_real, 0)
        w_real /= np.sum(w_real)
        w_equal = np.ones(n_assets) / n_assets

        returns_holding_values = returns_holding.values
        pred_returns = returns_holding_values @ w_pred
        real_returns = returns_holding_values @ w_real
        equal_returns = returns_holding_values @ w_equal

        portfolio_returns_pred.extend(pred_returns)
        portfolio_returns_real.extend(real_returns)
        portfolio_returns_equal.extend(equal_returns)

        cumulative_return_pred = np.prod(1 + pred_returns)
        cumulative_return_real = np.prod(1 + real_returns)
        cumulative_return_equal = np.prod(1 + equal_returns)

        new_value_pred = portfolio_values_pred[-1] * cumulative_return_pred
        new_value_real = portfolio_values_real[-1] * cumulative_return_real
        new_value_equal = portfolio_values_equal[-1] * cumulative_return_equal

        portfolio_values_pred.append(new_value_pred)
        portfolio_values_real.append(new_value_real)
        portfolio_values_equal.append(new_value_equal)
        dates.append(next_rebalance_date)

    def calculate_drawdown(portfolio_values):
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        return drawdown

    drawdown_pred = calculate_drawdown(portfolio_values_pred)
    drawdown_real = calculate_drawdown(portfolio_values_real)
    drawdown_equal = calculate_drawdown(portfolio_values_equal)

    # Calculate Sharpe ratios
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
        excess_returns = np.array(returns) - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns)

    sharpe_pred = calculate_sharpe_ratio(portfolio_returns_pred)
    sharpe_real = calculate_sharpe_ratio(portfolio_returns_real)
    sharpe_equal = calculate_sharpe_ratio(portfolio_returns_equal)

    print(f"Sharpe Ratio (Predicted Portfolio): {sharpe_pred:.4f}")
    print(f"Sharpe Ratio (Sample Portfolio): {sharpe_real:.4f}")
    print(f"Sharpe Ratio (Equally Weighted Portfolio): {sharpe_equal:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(dates, portfolio_values_pred, marker='o', label='Predicted MV Portfolio')
    plt.plot(dates, portfolio_values_real, marker='o', label='Sample MV Portfolio')
    plt.plot(dates, portfolio_values_equal, marker='o', label='Equally Weighted Portfolio')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Backtest from 2024-01-01 to 2024-10-01 (yyyy-mm-dd)')
    plt.legend()
    plt.grid(True)
    plt.savefig('./portfolio_backtest.svg', format='svg', dpi=300)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.fill_between(dates, drawdown_pred, color='blue', alpha=0.3, label='Predicted Portfolio Drawdown')
    plt.fill_between(dates, drawdown_real, color='orange', alpha=0.3, label='Sample Portfolio Drawdown')
    plt.fill_between(dates, drawdown_equal, color='green', alpha=0.3, label='Equally Weighted Portfolio Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.title('Portfolio Drawdown from 2024-01-01 to 2024-10-01 (yyyy-mm-dd)')
    plt.legend()
    plt.grid(True)
    plt.savefig('./portfolio_drawdown.svg', format='svg', dpi=300)
    plt.show()