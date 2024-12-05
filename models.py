import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DNNWithLASSO(nn.Module):
    def __init__(self, hparams, dataset_params, MM_func=None, fixed_point_solver=None):
        super(DNNWithLASSO, self).__init__()
        layers = []
        in_dim = hparams["input_size"]
        for h_dim in hparams["hidden_dims"]:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(hparams["hidden_dims"][-1], 1))
        self.dnn = nn.Sequential(*layers)
        self.MM_func = MM_func
        self.fixed_point_solver = fixed_point_solver
        if not MM_func:
            A = cp.Parameter((dataset_params['n'], dataset_params['p']))
            y = cp.Parameter(dataset_params['n'])
            beta = cp.Variable(dataset_params['p'])
            lam = cp.Parameter(nonneg=True)
            objective_fn = cp.norm(A @ beta - y, 2) + lam * cp.norm(beta, 1)
            constraints = []
            problem = cp.Problem(cp.Minimize(objective_fn), constraints)
            self.opt_layer = CvxpyLayer(problem, parameters=[A, y, lam], variables=[beta])

    def forward(self, A, y):
        A_flat = A.view(A.size(0), -1)
        y_flat = y.view(y.size(0), -1)
        input_tensor = torch.cat((A_flat, y_flat), dim=1)
        lambda_ = self.dnn(input_tensor)
        lambda_ = torch.relu(lambda_)
        if self.MM_func:
            if self.fixed_point_solver:
                beta = self.MM_func(A, y, lambda_, self.fixed_point_solver, epsilon=1e-5)
            else:
                beta = self.MM_func(A, y, lambda_, epsilon=1e-5)
        else:
            beta, = self.opt_layer(A, y, lambda_.squeeze())

        return beta, lambda_


class DEQFixedPoint(nn.Module):
    """
    Source: https://implicit-layers-tutorial.org/deep_equilibrium_models/
    """
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

    def forward(self):
        x0 = self.f.initial_guess()
        with torch.no_grad():
            z, self.forward_res = self.solver(self.f, x0, **self.kwargs)
        z = self.f(z)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0)
        def backward_hook(grad):
            g, self.backward_res = self.solver(
                lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                grad,
                **self.kwargs
            )
            return g
        if torch.is_grad_enabled():
            z.register_hook(backward_hook)
        return z


class MM_update(nn.Module):
    def __init__(self, X, y, lam, c):
        super().__init__()
        self.X = X
        self.y = y
        self.lam = lam
        self.c = c

    def forward(self, beta):
        bar_beta = (1 / self.c) * (
            torch.bmm(self.X.transpose(1, 2), self.y.unsqueeze(2) - torch.bmm(self.X, beta))
        ) + beta
        return soft_threshold(bar_beta, self.lam.unsqueeze(2) / (2 * self.c))

    def initial_guess(self):
        bsz, _, p = self.X.shape
        return torch.zeros(bsz, p, 1, device=self.X.device, dtype=self.X.dtype, requires_grad=True)


class PortfolioNet(nn.Module):
    def __init__(self, input_dim, n_assets):
        super(PortfolioNet, self).__init__()
        self.fc_mu = nn.Linear(input_dim, n_assets)
        self.fc_L = nn.Linear(input_dim, n_assets * (n_assets + 1) // 2)

    def forward(self, x):
        mu = self.fc_mu(x.reshape(x.shape[0], -1))
        L_flat = self.fc_L(x.reshape(x.shape[0], -1))

        indices = torch.tril_indices(x.shape[2], x.shape[2], offset=0)
        L = torch.zeros(x.shape[0], x.shape[2], x.shape[2])
        for i in range(x.shape[0]):
            L[i, indices[0], indices[1]] = L_flat[i]
        diag_indices = torch.arange(x.shape[2])
        L[:, diag_indices, diag_indices] = torch.relu(L[:, diag_indices, diag_indices]) + 1e-5

        return mu, L


class SensitivityAnalysisLayer:
    def __init__(self, variables, parameters, objective, inequalities, equalities, **cvxpy_opts):
        self.variables = variables
        self.parameters = parameters
        self.objective = objective
        self.inequalities = inequalities
        self.equalities = equalities
        self.cvxpy_opts = cvxpy_opts

        self.cp_inequalities = [ineq(*variables, *parameters) <= 0 for ineq in inequalities]
        self.cp_equalities = [eq(*variables, *parameters) == 0 for eq in equalities]
        self.problem = cp.Problem(
            cp.Minimize(objective(*variables, *parameters)),
            self.cp_inequalities + self.cp_equalities
        )

    def forward(self, *batch_params):
        out = []
        batch_size = batch_params[0].shape[0]
        for batch in range(batch_size):
            params = [p[batch] for p in batch_params]
            with torch.no_grad():
                for i, p in enumerate(self.parameters):
                    p.value = params[i].cpu().double().numpy()
                try:
                    self.problem.solve(**self.cvxpy_opts, max_iter=100000)
                except:
                    print("Ill conditioned case. Eigenvalues: ")
                    print(torch.linalg.eigvals(params[0]))
                z_star = \
                [torch.tensor(v.value, dtype=params[0].dtype, device=params[0].device) for v in self.variables][0]

            Sigma_hat = torch.matmul(params[0], params[0].transpose(0, 1))
            mu_hat = params[1]
            optimal_value = torch.matmul(z_star.unsqueeze(0),
                                         torch.matmul(Sigma_hat, z_star.unsqueeze(1))).squeeze() - torch.sum(
                mu_hat * z_star)

            out.append(optimal_value)
        out = torch.stack(out, dim=0)
        return out


def soft_threshold(beta, threshold):
    return torch.sign(beta) * torch.maximum(torch.abs(beta) - threshold, torch.tensor(0.0, dtype=beta.dtype))


def MM_unrolled(X, y, lam, epsilon=1e-5):
    bsz = X.shape[0]
    p = X.shape[2]
    beta = torch.zeros((bsz, p), dtype=torch.float32, requires_grad=True, device=device).unsqueeze(2)
    converged = False
    iter_count = 0
    max_iter = 10000

    c = (torch.linalg.eigvals(torch.bmm(X.transpose(1, 2), X)).real.max() + lam).unsqueeze(2)

    while not converged and iter_count < max_iter:
        beta_old = beta.clone()
        f = MM_update(X, y, lam, c)
        beta = f(beta)
        if torch.sqrt(torch.sum((beta - beta_old) ** 2)).item() < epsilon:
            converged = True
        iter_count += 1

    return beta.squeeze()


def MM_deq(X, y, lam, fixed_point_solver, epsilon=1e-5):
    c = (torch.linalg.eigvals(torch.bmm(X.transpose(1, 2), X)).real.max() + lam).unsqueeze(2)

    f = MM_update(X, y, lam, c)
    deq = DEQFixedPoint(f, fixed_point_solver, tol=epsilon, max_iter=1000)
    beta = deq()

    return beta.squeeze()


def create_cvxpy_layer(n, lambda_):
    x = cp.Variable(n)
    P_sqrt = cp.Parameter((n, n))
    mu_param = cp.Parameter(n)

    quad_form = cp.sum_squares(P_sqrt @ x)
    objective = cp.Minimize(0.5 * quad_form - lambda_ * mu_param.T @ x)
    constraints = [cp.sum(x) == 1, x >= 0]
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp(), "Problem is not DPP compliant."
    cvxpylayer = CvxpyLayer(problem, parameters=[mu_param, P_sqrt], variables=[x])
    return cvxpylayer