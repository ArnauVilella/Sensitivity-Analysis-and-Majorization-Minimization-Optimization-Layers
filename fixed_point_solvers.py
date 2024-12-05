import numpy as np
import torch


def forward_iteration(f, x0, max_iter=50, tol=1e-2):
    """
    Source: https://implicit-layers-tutorial.org/deep_equilibrium_models/
    """
    f0 = f(x0)
    res = []
    for k in range(max_iter):
        x = f0
        f0 = f(x)
        res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
        if (res[-1] < tol):
            break
    return f0, res

def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=2.0):
    """
    Source: https://implicit-layers-tutorial.org/deep_equilibrium_models/
    """
    x0_flat = x0.view(x0.size(0), -1)
    bsz, d = x0_flat.shape
    X = torch.zeros(bsz, m, d, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0_flat, f(x0).view(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        GT = G.transpose(1, 2)
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, GT) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]  # (bsz x n)

        X_new = beta * torch.bmm(alpha.unsqueeze(1), F[:, :n])[:, 0] + (1 - beta) * torch.bmm(alpha.unsqueeze(1), X[:, :n])[:, 0]
        X[:, k % m] = X_new
        F_new = f(X_new.view_as(x0)).view(bsz, -1)
        F[:, k % m] = F_new
        res_norm = (F_new - X_new).norm().item() / (1e-5 + F_new.norm().item())
        res.append(res_norm)
        if res_norm < tol:
            break
    return X[:, k % m].view_as(x0), res