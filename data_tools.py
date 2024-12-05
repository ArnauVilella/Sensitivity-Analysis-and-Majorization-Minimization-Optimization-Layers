import numpy as np
import torch
import yfinance as yf
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class SparseRegressionDataset(Dataset):
    def __init__(self, dataset_params):
        self.data = []
        for _ in range(dataset_params['num_entries']):
            A = np.random.randn(dataset_params['n'], dataset_params['p'])

            sparsity = np.clip(np.exp(-dataset_params['decay_rate'] * np.random.rand()), 0, 1)

            beta = np.zeros(dataset_params['p'])
            num_nonzero = max(1, int(sparsity * dataset_params['p']))
            nonzero_indices = np.random.choice(dataset_params['p'], num_nonzero, replace=False)
            beta[nonzero_indices] = np.random.randn(num_nonzero)

            epsilon = np.random.randn(dataset_params['n']) * dataset_params['noise_std']

            y = A @ beta + epsilon

            self.data.append([torch.tensor(A, dtype=torch.float32),
                              torch.tensor(y, dtype=torch.float32),
                              torch.tensor(beta, dtype=torch.float32)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PortfolioDataset(Dataset):
    def __init__(self, returns_data, subseries_len=128):
        self.returns_data = returns_data
        self.subseries_len = subseries_len
        self.mu_tags = []
        self.sigma_tags = []

        for i in range(len(returns_data) - subseries_len):
            subseries = returns_data[i:i + subseries_len]
            sample_mu = subseries.mean(dim=0)
            sample_sigma = torch.cov(subseries.T)
            self.mu_tags.append(sample_mu)
            self.sigma_tags.append(sample_sigma)

    def __len__(self):
        return len(self.returns_data) - self.subseries_len

    def __getitem__(self, idx):
        subseries = self.returns_data[idx:idx + self.subseries_len]
        mu_tag = self.mu_tags[idx]
        sigma_tag = self.sigma_tags[idx]
        return subseries, mu_tag, sigma_tag


def get_data_loaders(dataset_params, batch_size=32, test_size=0.2):
    dataset = SparseRegressionDataset(dataset_params)
    train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=test_size, random_state=42)

    train_set = torch.utils.data.Subset(dataset, train_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_indices, test_indices


def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    return returns