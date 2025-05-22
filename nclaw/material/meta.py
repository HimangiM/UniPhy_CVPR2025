from typing import Optional

from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange

from .abstract import Elasticity, Plasticity
from .utils import get_nonlinearity, get_norm, init_weight
from collections import OrderedDict
from nclaw.warp import SVD

class MLPBlock(nn.Module):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            no_bias: bool,
            norm: Optional[str],
            nonlinearity: Optional[str]) -> None:

        super().__init__()
        if norm == 'wn':
            self.fc = nn.utils.weight_norm(nn.Linear(in_planes, out_planes, not no_bias))
        else:
            self.fc = nn.Linear(in_planes, out_planes, bias=not no_bias and norm is None)
        self.norm = get_norm(norm, out_planes, dim=1, affine=not no_bias)
        self.nonlinearity = get_nonlinearity(nonlinearity)

    def forward(self, x: Tensor) -> Tensor:

        x = self.fc(x)
        x = self.norm(x)
        x = self.nonlinearity(x)
        return x

class MetaElasticity(Elasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.flatten = Rearrange('b d1 d2 -> b (d1 d2)', d1=self.dim, d2=self.dim)
        self.unflatten = Rearrange('b (d1 d2) -> b d1 d2', d1=self.dim, d2=self.dim)

        self.normalize_input: bool = cfg.normalize_input

    def forward(self, F: Tensor, trajectory_latent: Tensor) -> Tensor:
        raise NotImplementedError


class PlainMetaElasticity(MetaElasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = self.dim * self.dim
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim * self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)

        if self.normalize_input:
            x = self.flatten(F - I)
        else:
            x = self.flatten(F)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        P = self.unflatten(x)
        cauchy = torch.matmul(P, self.transpose(F))
        return cauchy

class PolarMetaElasticity(MetaElasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = self.dim * self.dim
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim * self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)

        U, sigma, Vh = self.svd(F)

        R = torch.matmul(U, Vh)
        S = torch.matmul(torch.matmul(self.transpose(Vh), torch.diag_embed(sigma)), Vh)

        if self.normalize_input:
            x = self.flatten(S - I)
        else:
            x = self.flatten(S)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        x = self.unflatten(x)
        x = 0.5 * (self.transpose(x) + x)
        P = torch.matmul(R, x)
        cauchy = torch.matmul(P, self.transpose(F))
        return cauchy


class InvariantMetaElasticity(MetaElasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = 3
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim * self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor) -> Tensor:
        U, sigma, Vh = self.svd(F)

        R = torch.matmul(U, Vh)

        Ft = self.transpose(F)

        if self.normalize_input:
            I1 = sigma.sum(dim=1) - 3.0
            I2 = torch.diagonal(torch.matmul(Ft, F), dim1=1, dim2=2).sum(dim=1) - 1.0
            I3 = torch.linalg.det(F) - 1.0
        else:
            I1 = sigma.sum(dim=1)
            I2 = torch.diagonal(torch.matmul(Ft, F), dim1=1, dim2=2).sum(dim=1)
            I3 = torch.linalg.det(F)

        x = torch.stack([I1, I2, I3], dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        x = self.unflatten(x)
        x = 0.5 * (self.transpose(x) + x)
        P = torch.matmul(R, x)
        cauchy = torch.matmul(P, self.transpose(F))
        return cauchy

class InvariantFullMetaElasticity(MetaElasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = self.dim + self.dim * self.dim + 1
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim * self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)
        U, sigma, Vh = self.svd(F)
        R = torch.matmul(U, Vh)

        Ft = self.transpose(F)
        FtF = torch.matmul(Ft, F)

        if self.normalize_input:
            I1 = sigma - 1.0
            I2 = self.flatten(FtF - I)
            I3 = torch.linalg.det(F).unsqueeze(dim=1) - 1.0
        else:
            I1 = sigma
            I2 = self.flatten(FtF)
            I3 = torch.linalg.det(F).unsqueeze(dim=1)

        x = torch.cat([I1, I2, I3], dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        x = self.unflatten(x)
        x = 0.5 * (self.transpose(x) + x)
        P = torch.matmul(R, x)
        cauchy = torch.matmul(P, Ft)
        return cauchy

class SVDMetaElasticity(MetaElasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = self.dim
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor) -> Tensor:
        Ft = self.transpose(F)

        U, sigma, Vh = self.svd(F)

        if self.normalize_input:
            x = sigma - 1.0
        else:
            x = sigma
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)

        P = torch.matmul(torch.matmul(U, torch.diag_embed(x)), Vh)

        cauchy = torch.matmul(P, Ft)
        return cauchy

# http://viterbi-web.usc.edu/~jbarbic/isotropicMaterialEditor/XuSinZhuBarbic-Siggraph2015.pdf
class SplineMetaElasticity(MetaElasticity):

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.num_side_points: int = cfg.num_side_points
        self.xk_min: float = 0.0
        self.xk_max: float = cfg.xk_max
        self.yk_min: float = -cfg.yk_max
        self.yk_max: float = cfg.yk_max

        self.npoints = 2 * self.num_side_points + 1
        left_points = np.linspace(self.xk_min, 1.0, cfg.num_side_points + 1)
        right_points = np.linspace(1.0, self.xk_max , cfg.num_side_points + 1)
        xk = torch.tensor(left_points.tolist()[:-1] + [1.0] + right_points.tolist()[1:])
        self.register_buffer('xk', xk)

        w = torch.tensor([
            [-1.0, 3.0, -3.0, 1.0],
            [3.0, -6.0, 3.0, 0.0],
            [-3.0, 3.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ]).view(1, 4, 4)
        self.register_buffer('w', w)

        # if cfg.E is not None and cfg.nu is not None:
        #     E = cfg.E
        #     nu = cfg.nu
        #     mu = E / (2 * (1 + nu))
        #     la = E * nu / ((1 + nu) * (1 - 2 * nu))

        #     self.yk_f = nn.Parameter(la * xk - 3 * la + 2 * mu * (xk - 1))
        #     self.yk_g = nn.Parameter(torch.ones_like(xk) * la)
        #     self.yk_h = nn.Parameter(torch.zeros_like(xk))
        # else:

        self.yk_f = nn.Parameter(torch.linspace(self.yk_min, self.yk_max, xk.size(0)))
        self.yk_g = nn.Parameter(torch.linspace(self.yk_min, self.yk_max, xk.size(0)))
        self.yk_h = nn.Parameter(torch.linspace(self.yk_min, self.yk_max, xk.size(0)))

    def get_ak(self, yk):
        ak_1 = 2 / 3 * yk[0] + 2 / 3 * yk[1] - 1 / 3 * yk[2]
        ak_else = yk[1:-1] - 1 / 6 * yk[:-2] + 1 / 6 * yk[2:]
        return torch.cat([ak_1.unsqueeze(0), ak_else], dim=0)

    def get_bk(self, yk):
        bk_else = yk[1:-1] + 1 / 6 * yk[:-2] - 1 / 6 * yk[2:]
        bk_m = 2 / 3 * yk[-1] + 2 / 3 * yk[-2] - 1 / 3 * yk[-3]
        return torch.cat([bk_else, bk_m.unsqueeze(0)], dim=0)

    def get_func(self, yk, lambd):
        indices = torch.searchsorted(self.xk, lambd, right=False).view(-1)
        indices[indices < 0] = 0
        indices[indices > self.num_side_points - 1] = self.num_side_points - 1

        ak = self.get_ak(yk)
        bk = self.get_bk(yk)

        y_left = yk[indices].view_as(lambd)
        y_right = yk[indices + 1].view_as(lambd)
        a = ak[indices].view_as(lambd)
        b = bk[indices].view_as(lambd)
        temp_right = torch.stack([y_left, a, b, y_right], dim=2)

        xi = (lambd - self.xk[indices].view_as(lambd)) / (self.xk[indices + 1].view_as(lambd) - self.xk[indices].view_as(lambd))
        xi_vector = torch.stack([xi**3, xi**2, xi, torch.ones_like(xi)], dim=2) # batch, #lambda, 4

        temp_left = torch.matmul(xi_vector, self.w) # batch, #lambda, 4
        func = (temp_left * temp_right).sum(dim=2) # batch, #lambda

        return func

    def forward(self, F: Tensor) -> Tensor:
        U, sigma, Vh = self.svd(F)

        f = self.get_func(self.yk_f, sigma)

        areas = torch.stack([
            sigma[:, 0] * sigma[:, 1],
            sigma[:, 1] * sigma[:, 2],
            sigma[:, 0] * sigma[:, 2]], dim=1)
        g = self.get_func(self.yk_g, areas)

        g1 = g[:, [0, 0, 2]] * sigma[:, [1, 0, 0]]
        g2 = g[:, [2, 1, 1]] * sigma[:, [2, 2, 1]]

        volume = (sigma[:, 0] * sigma[:, 1] * sigma[:, 2]).unsqueeze(1)
        h = self.get_func(self.yk_h, volume) * sigma[:, [1, 0, 0]] * sigma[:, [2, 2, 1]]

        new_sigma = f + g1 + g2 + h
        P = torch.matmul(torch.matmul(U, torch.diag_embed(new_sigma)), Vh)

        Ft = self.transpose(F)

        cauchy = torch.matmul(P, Ft)
        return cauchy


class MetaPlasticity(Plasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.alpha: float = cfg.alpha

        self.flatten = Rearrange('b d1 d2 -> b (d1 d2)', d1=self.dim, d2=self.dim)
        self.unflatten = Rearrange('b (d1 d2) -> b d1 d2', d1=self.dim, d2=self.dim)

        self.normalize_input: bool = cfg.normalize_input

    def forward(self, F: Tensor, trajectory_latent: Tensor) -> Tensor:
        raise NotImplementedError


class PlainMetaPlasticity(MetaPlasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = self.dim * self.dim
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim * self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)

        if self.normalize_input:
            x = self.flatten(F - I)
        else:
            x = self.flatten(F)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        delta_Fp = self.alpha * self.unflatten(x)
        Fp = delta_Fp + F
        return Fp


class PolarMetaPlasticity(MetaPlasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = self.dim * self.dim
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim * self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)

        U, sigma, Vh = self.svd(F)

        R = torch.matmul(U, Vh)
        S = torch.matmul(torch.matmul(self.transpose(Vh), torch.diag_embed(sigma)), Vh)

        if self.normalize_input:
            x = self.flatten(S - I)
        else:
            x = self.flatten(S)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        x = self.unflatten(x)
        x = 0.5 * (self.transpose(x) + x)
        delta_Fp = self.alpha * torch.matmul(R, x)
        Fp = delta_Fp + F
        return Fp


class InvariantFullMetaPlasticity(MetaPlasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = 3 + 9 + 1
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim * self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)
        U, sigma, Vh = self.svd(F)
        R = torch.matmul(U, Vh)

        Ft = self.transpose(F)
        FtF = torch.matmul(Ft, F)

        I1 = sigma - 1.0
        I2 = self.flatten(FtF - I)
        I3 = torch.linalg.det(F).unsqueeze(dim=1) - 1.0

        invariants = torch.cat([I1, I2, I3], dim=1)
        x = invariants
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        x = self.unflatten(x)
        x = 0.5 * (self.transpose(x) + x)
        delta_Fp = self.alpha * torch.matmul(R, x)
        Fp = delta_Fp + F
        return Fp

class StressLatentConditioned(torch.nn.Module):
    def __init__(self, hidden_size, embed_dim, normalize_input) -> None:
        super().__init__()

        self.normalize_input: bool = normalize_input
        self.activation = torch.nn.GELU()
        self.dim = 3
        self.transpose = Rearrange('b d1 d2 -> b d2 d1', d1=self.dim, d2=self.dim)
        self.flatten = Rearrange('b d1 d2 -> b (d1 d2)', d1=self.dim, d2=self.dim)
        self.unflatten = Rearrange('b (d1 d2) -> b d1 d2', d1=self.dim, d2=self.dim)
        self.svd = SVD()

        self.stress_model = nn.Sequential(OrderedDict([
                                          ('fc1_stress', nn.Linear(16+9+embed_dim, hidden_size, bias=True)),
                                          ('act1_stress', self.activation),
                                          ('fc2_stress', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act2_stress', self.activation),
                                          ('fc3_stress', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act3_stress', self.activation),
                                          ('fc4_stress', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act4_stress', self.activation),
                                          ('fc5_stress', nn.Linear(hidden_size, 9, bias=True))
                                          ]))

    def forward(self, F: Tensor, C: Tensor, trajectory_latent: Tensor, traj_ids: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)
        # U, sigma, Vh = self.svd(F)
        U, sigma, V = torch.svd(F)  # plasticine
        Vh = self.transpose(V)      # plasticine
        R = torch.matmul(U, Vh)
        det_F = torch.linalg.det(F)
        det_F = torch.where(det_F < 0, torch.tensor(1e-9, dtype=det_F.dtype, device=det_F.device), det_F)

        Ft = self.transpose(F)
        FtF = torch.matmul(Ft, F)

        I1 = sigma
        I2 = self.flatten(FtF)
        I3 = det_F.unsqueeze(dim=1)
        I4 = torch.log(det_F).unsqueeze(dim=1)
        I5 = torch.max(F[:, 0, 0], torch.Tensor([1e-6]).cuda()).unsqueeze(dim=1)
        I6 = torch.log(torch.max(F[:, 0, 0], torch.Tensor([1e-6]).cuda())).unsqueeze(dim=1)

        strain = torch.cat([I1, I2, I3, I4, I5, I6], dim=1)
        C_flatten = self.flatten(C)
        out_stress = self.stress_model(torch.cat([strain, C_flatten, trajectory_latent.weight[traj_ids].unsqueeze(0).repeat(strain.shape[0], 1)], dim=-1))
        x = self.unflatten(out_stress)
        x = 0.5 * (self.transpose(x) + x)
        P = torch.matmul(R, x)
        cauchy = torch.matmul(P, Ft)
        return cauchy

class FprojLatentConditioned(torch.nn.Module):
    def __init__(self, hidden_size, embed_dim, normalize_input, alpha) -> None:
        super().__init__()

        self.normalize_input: bool = normalize_input
        self.alpha: float = alpha
        self.activation = torch.nn.GELU()
        self.dim = 3
        self.transpose = Rearrange('b d1 d2 -> b d2 d1', d1=self.dim, d2=self.dim)
        self.flatten = Rearrange('b d1 d2 -> b (d1 d2)', d1=self.dim, d2=self.dim)
        self.unflatten = Rearrange('b (d1 d2) -> b d1 d2', d1=self.dim, d2=self.dim)
        self.svd = SVD()

        self.fproj_model = nn.Sequential(OrderedDict([
                                          ('fc1_fproj', nn.Linear(27+3+embed_dim, hidden_size, bias=True)),
                                          ('act1_fproj', self.activation),
                                          ('fc2_fproj', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act2_fproj', self.activation),
                                          ('fc3_fproj', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act3_fproj', self.activation),
                                          ('fc4_fproj', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act4_fproj', self.activation),
                                          ('fc5_fproj', nn.Linear(hidden_size, 9, bias=True)),
                                          ]))

    def forward(self, F: Tensor, trajectory_latent: Tensor, traj_ids: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)
        # U, sigma, Vh = self.svd(F)
        U, sigma, V = torch.svd(F)  # plasticine
        Vh = self.transpose(V)      # plasticine
        R = torch.matmul(U, Vh)

        U_flatten = self.flatten(U)
        sigma = sigma
        Vh_flatten = self.flatten(Vh)
        F_flatten = self.flatten(F)

        invariants = torch.cat([F_flatten, U_flatten, sigma, Vh_flatten, trajectory_latent.weight[traj_ids].unsqueeze(0).repeat(F_flatten.shape[0], 1)], dim=1)
        out_fproj = self.fproj_model(invariants)
        x = self.unflatten(out_fproj)
        x = 0.5 * (self.transpose(x) + x)
        delta_Fp = self.alpha * torch.matmul(R, x)
        Fp = delta_Fp + F
        return Fp

class SplineMetaPlasticity(MetaPlasticity):

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.num_side_points: int = cfg.num_side_points
        self.xk_min: float = 0.0
        self.xk_max: float = cfg.xk_max
        self.yk_min: float = -cfg.yk_max
        self.yk_max: float = cfg.yk_max

        self.npoints = 2 * self.num_side_points + 1
        left_points = np.linspace(self.xk_min, 1.0, cfg.num_side_points + 1)
        right_points = np.linspace(1.0, self.xk_max , cfg.num_side_points + 1)
        xk = torch.tensor(left_points.tolist()[:-1] + [1.0] + right_points.tolist()[1:])
        self.register_buffer('xk', xk)

        w = torch.tensor([
            [-1.0, 3.0, -3.0, 1.0],
            [3.0, -6.0, 3.0, 0.0],
            [-3.0, 3.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ]).view(1, 4, 4)
        self.register_buffer('w', w)

        self.yk_f = nn.Parameter(torch.zeros_like(xk))
        self.yk_g = nn.Parameter(torch.zeros_like(xk))
        self.yk_h = nn.Parameter(torch.zeros_like(xk))

    def get_ak(self, yk):
        ak_1 = 2 / 3 * yk[0] + 2 / 3 * yk[1] - 1 / 3 * yk[2]
        ak_else = yk[1:-1] - 1 / 6 * yk[:-2] + 1 / 6 * yk[2:]
        return torch.cat([ak_1.unsqueeze(0), ak_else], dim=0)

    def get_bk(self, yk):
        bk_else = yk[1:-1] + 1 / 6 * yk[:-2] - 1 / 6 * yk[2:]
        bk_m = 2 / 3 * yk[-1] + 2 / 3 * yk[-2] - 1 / 3 * yk[-3]
        return torch.cat([bk_else, bk_m.unsqueeze(0)], dim=0)

    def get_func(self, yk, lambd):
        indices = torch.searchsorted(self.xk, lambd, right=False).view(-1)
        indices[indices < 0] = 0
        indices[indices > self.num_side_points - 1] = self.num_side_points - 1

        ak = self.get_ak(yk)
        bk = self.get_bk(yk)

        y_left = yk[indices].view_as(lambd)
        y_right = yk[indices + 1].view_as(lambd)
        a = ak[indices].view_as(lambd)
        b = bk[indices].view_as(lambd)
        temp_right = torch.stack([y_left, a, b, y_right], dim=2)

        xi = (lambd - self.xk[indices].view_as(lambd)) / (self.xk[indices + 1].view_as(lambd) - self.xk[indices].view_as(lambd))
        xi_vector = torch.stack([xi**3, xi**2, xi, torch.ones_like(xi)], dim=2) # batch, #lambda, 4

        temp_left = torch.matmul(xi_vector, self.w) # batch, #lambda, 4
        func = (temp_left * temp_right).sum(dim=2) # batch, #lambda

        return func

    def forward(self, F: Tensor) -> Tensor:
        U, sigma, Vh = self.svd(F)

        f = self.get_func(self.yk_f, sigma)

        areas = torch.stack([
            sigma[:, 0] * sigma[:, 1],
            sigma[:, 1] * sigma[:, 2],
            sigma[:, 0] * sigma[:, 2]], dim=1)
        g = self.get_func(self.yk_g, areas)

        g1 = g[:, [0, 0, 2]] * sigma[:, [1, 0, 0]]
        g2 = g[:, [2, 1, 1]] * sigma[:, [2, 2, 1]]

        volume = (sigma[:, 0] * sigma[:, 1] * sigma[:, 2]).unsqueeze(1)
        h = self.get_func(self.yk_h, volume) * sigma[:, [1, 0, 0]] * sigma[:, [2, 2, 1]]

        new_sigma = f + g1 + g2 + h
        delta_Fp = self.alpha * torch.matmul(torch.matmul(U, torch.diag_embed(new_sigma)), Vh)

        Fp = delta_Fp + F
        return Fp


class SVDMetaPlasticity(MetaPlasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = self.dim
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor) -> Tensor:
        U, sigma, Vh = self.svd(F)

        if self.normalize_input:
            x = sigma - 1.0
        else:
            x = sigma
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)

        delta_Fp = self.alpha * torch.matmul(torch.matmul(U, torch.diag_embed(x)), Vh)
        Fp = delta_Fp + F

        return Fp

class StressNN(torch.nn.Module):
    def __init__(self, hidden_size, embed_dim, normalize_input) -> None:
        super().__init__()

        self.normalize_input: bool = normalize_input
        self.activation = torch.nn.GELU()
        self.dim = 3
        self.transpose = Rearrange('b d1 d2 -> b d2 d1', d1=self.dim, d2=self.dim)
        self.flatten = Rearrange('b d1 d2 -> b (d1 d2)', d1=self.dim, d2=self.dim)
        self.unflatten = Rearrange('b (d1 d2) -> b d1 d2', d1=self.dim, d2=self.dim)
        self.svd = SVD()

        self.stress_model = nn.Sequential(OrderedDict([
                                          ('fc1_stress', nn.Linear(16+9+embed_dim, hidden_size, bias=True)),
                                          ('act1_stress', self.activation),
                                          ('fc2_stress', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act2_stress', self.activation),
                                          ('fc3_stress', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act3_stress', self.activation),
                                          ('fc4_stress', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act4_stress', self.activation),
                                          ('fc5_stress', nn.Linear(hidden_size, 9, bias=True))
                                          ]))

    def forward(self, F: Tensor, C: Tensor, trajectory_latent: Tensor, traj_ids: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)
        U, sigma, Vh = self.svd(F)
        # U, sigma, V = torch.svd(F)  # plasticine
        # Vh = self.transpose(V)      # plasticine
        R = torch.matmul(U, Vh)

        Ft = self.transpose(F)
        FtF = torch.matmul(Ft, F)

        if self.normalize_input:
            I1 = sigma - 1.0                                                                          # B x 3
            I2 = self.flatten(FtF - I)                                                                # B x 9
            I3 = torch.linalg.det(F).unsqueeze(dim=1) - 1.0                                           # B x 1
            I4 = torch.log(torch.linalg.det(F)).unsqueeze(dim=1) - 1.0                                # B x 1
            I5 = torch.max(F[:, 0, 0], torch.Tensor([1e-6]).cuda()).unsqueeze(dim=1) - 1.0            # B x 1
            I6 = torch.log(torch.max(F[:, 0, 0], torch.Tensor([1e-6]).cuda())).unsqueeze(dim=1) - 1.0 # B x 1
        else:
            I1 = sigma
            I2 = self.flatten(FtF)
            I3 = torch.linalg.det(F).unsqueeze(dim=1)
            I4 = torch.log(torch.linalg.det(F)).unsqueeze(dim=1)
            I5 = torch.max(F[:, 0, 0], torch.Tensor([1e-6]).cuda()).unsqueeze(dim=1)
            I6 = torch.log(torch.max(F[:, 0, 0], torch.Tensor([1e-6]).cuda())).unsqueeze(dim=1)

        strain = torch.cat([I1, I2, I3, I4, I5, I6], dim=1)
        C_flatten = self.flatten(C)
        out_stress = self.stress_model(torch.cat([strain, C_flatten, trajectory_latent.weight[traj_ids].unsqueeze(0).repeat(strain.shape[0], 1)], dim=-1))
        x = self.unflatten(out_stress)
        x = 0.5 * (self.transpose(x) + x)
        P = torch.matmul(R, x)
        cauchy = torch.matmul(P, Ft)
        return cauchy

class FprojNN(torch.nn.Module):
    def __init__(self, hidden_size, embed_dim, normalize_input, alpha) -> None:
        super().__init__()

        self.normalize_input: bool = normalize_input
        self.alpha: float = alpha
        self.activation = torch.nn.GELU()
        self.dim = 3
        self.transpose = Rearrange('b d1 d2 -> b d2 d1', d1=self.dim, d2=self.dim)
        self.flatten = Rearrange('b d1 d2 -> b (d1 d2)', d1=self.dim, d2=self.dim)
        self.unflatten = Rearrange('b (d1 d2) -> b d1 d2', d1=self.dim, d2=self.dim)
        self.svd = SVD()

        self.fproj_model = nn.Sequential(OrderedDict([
                                          ('fc1_fproj', nn.Linear(27+3+embed_dim, hidden_size, bias=True)),
                                          ('act1_fproj', self.activation),
                                          ('fc2_fproj', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act2_fproj', self.activation),
                                          ('fc3_fproj', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act3_fproj', self.activation),
                                          ('fc4_fproj', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act4_fproj', self.activation),
                                          ('fc5_fproj', nn.Linear(hidden_size, 9, bias=True)),
                                          ]))

    def forward(self, F: Tensor, trajectory_latent: Tensor, traj_ids: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)
        U, sigma, Vh = self.svd(F)
        # U, sigma, V = torch.svd(F)  # plasticine
        # Vh = self.transpose(V)      # plasticine
        R = torch.matmul(U, Vh)

        U_flatten = self.flatten(U - I)
        sigma = sigma - 1.0
        Vh_flatten = self.flatten(Vh - I)
        F_flatten = self.flatten(F - I)

        invariants = torch.cat([F_flatten, U_flatten, sigma, Vh_flatten, trajectory_latent.weight[traj_ids].unsqueeze(0).repeat(F_flatten.shape[0], 1)], dim=1)
        out_fproj = self.fproj_model(invariants)
        x = self.unflatten(out_fproj)
        x = 0.5 * (self.transpose(x) + x)
        delta_Fp = self.alpha * torch.matmul(R, x)
        Fp = delta_Fp + F
        return Fp

class StressNNEval(torch.nn.Module):
    def __init__(self, hidden_size, embed_dim, normalize_input) -> None:
        super().__init__()

        self.normalize_input: bool = normalize_input
        self.activation = torch.nn.GELU()
        self.dim = 3
        self.transpose = Rearrange('b d1 d2 -> b d2 d1', d1=self.dim, d2=self.dim)
        self.flatten = Rearrange('b d1 d2 -> b (d1 d2)', d1=self.dim, d2=self.dim)
        self.unflatten = Rearrange('b (d1 d2) -> b d1 d2', d1=self.dim, d2=self.dim)
        self.svd = SVD()

        self.stress_model = nn.Sequential(OrderedDict([
                                          ('fc1_stress', nn.Linear(16+9+embed_dim, hidden_size, bias=True)),
                                          ('act1_stress', self.activation),
                                          ('fc2_stress', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act2_stress', self.activation),
                                          ('fc3_stress', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act3_stress', self.activation),
                                          ('fc4_stress', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act4_stress', self.activation),
                                          ('fc5_stress', nn.Linear(hidden_size, 9, bias=True))
                                          ]))

    def forward(self, F: Tensor, C: Tensor, trajectory_latent: Tensor, traj_ids: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)
        # U, sigma, Vh = self.svd(F)
        U, sigma, V = torch.svd(F)  # plasticine
        Vh = self.transpose(V)      # plasticine
        R = torch.matmul(U, Vh)

        Ft = self.transpose(F)
        FtF = torch.matmul(Ft, F)

        if self.normalize_input:
            I1 = sigma - 1.0                                                                          # B x 3
            I2 = self.flatten(FtF - I)                                                                # B x 9
            I3 = torch.linalg.det(F).unsqueeze(dim=1) - 1.0                                           # B x 1
            I4 = torch.log(torch.linalg.det(F)).unsqueeze(dim=1) - 1.0                                # B x 1
            I5 = torch.max(F[:, 0, 0], torch.Tensor([1e-6]).cuda()).unsqueeze(dim=1) - 1.0            # B x 1
            I6 = torch.log(torch.max(F[:, 0, 0], torch.Tensor([1e-6]).cuda())).unsqueeze(dim=1) - 1.0 # B x 1
        else:
            I1 = sigma
            I2 = self.flatten(FtF)
            I3 = torch.linalg.det(F).unsqueeze(dim=1)
            I4 = torch.log(torch.linalg.det(F)).unsqueeze(dim=1)
            I5 = torch.max(F[:, 0, 0], torch.Tensor([1e-6]).cuda()).unsqueeze(dim=1)
            I6 = torch.log(torch.max(F[:, 0, 0], torch.Tensor([1e-6]).cuda())).unsqueeze(dim=1)

        strain = torch.cat([I1, I2, I3, I4, I5, I6], dim=1)
        C_flatten = self.flatten(C)
        out_stress = self.stress_model(torch.cat([strain, C_flatten, trajectory_latent.weight[traj_ids].unsqueeze(0).repeat(strain.shape[0], 1)], dim=-1))
        x = self.unflatten(out_stress)
        x = 0.5 * (self.transpose(x) + x)
        P = torch.matmul(R, x)
        cauchy = torch.matmul(P, Ft)
        return cauchy

class FprojNNEval(torch.nn.Module):
    def __init__(self, hidden_size, embed_dim, normalize_input, alpha) -> None:
        super().__init__()

        self.normalize_input: bool = normalize_input
        self.alpha: float = alpha
        self.activation = torch.nn.GELU()
        self.dim = 3
        self.transpose = Rearrange('b d1 d2 -> b d2 d1', d1=self.dim, d2=self.dim)
        self.flatten = Rearrange('b d1 d2 -> b (d1 d2)', d1=self.dim, d2=self.dim)
        self.unflatten = Rearrange('b (d1 d2) -> b d1 d2', d1=self.dim, d2=self.dim)
        self.svd = SVD()

        self.fproj_model = nn.Sequential(OrderedDict([
                                          ('fc1_fproj', nn.Linear(27+3+embed_dim, hidden_size, bias=True)),
                                          ('act1_fproj', self.activation),
                                          ('fc2_fproj', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act2_fproj', self.activation),
                                          ('fc3_fproj', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act3_fproj', self.activation),
                                          ('fc4_fproj', nn.Linear(hidden_size, hidden_size, bias=True)),
                                          ('act4_fproj', self.activation),
                                          ('fc5_fproj', nn.Linear(hidden_size, 9, bias=True)),
                                          ]))

    def forward(self, F: Tensor, trajectory_latent: Tensor, traj_ids: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)
        # U, sigma, Vh = self.svd(F)
        U, sigma, V = torch.svd(F)  # plasticine
        Vh = self.transpose(V)      # plasticine
        R = torch.matmul(U, Vh)

        U_flatten = self.flatten(U - I)
        sigma = sigma - 1.0
        Vh_flatten = self.flatten(Vh - I)
        F_flatten = self.flatten(F - I)

        invariants = torch.cat([F_flatten, U_flatten, sigma, Vh_flatten, trajectory_latent.weight[traj_ids].unsqueeze(0).repeat(F_flatten.shape[0], 1)], dim=1)
        out_fproj = self.fproj_model(invariants)
        x = self.unflatten(out_fproj)
        x = 0.5 * (self.transpose(x) + x)
        delta_Fp = self.alpha * torch.matmul(R, x)
        Fp = delta_Fp + F
        return Fp

class StressLatentConditioned2(MetaElasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = self.dim + self.dim * self.dim + 1 + cfg.latent_dim
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim * self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor, trajectory_latent: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)
        U, sigma, Vh = self.svd(F)
        R = torch.matmul(U, Vh)

        Ft = self.transpose(F)
        FtF = torch.matmul(Ft, F)

        if self.normalize_input:
            I1 = sigma - 1.0
            I2 = self.flatten(FtF - I)
            I3 = torch.linalg.det(F).unsqueeze(dim=1) - 1.0
        else:
            I1 = sigma
            I2 = self.flatten(FtF)
            I3 = torch.linalg.det(F).unsqueeze(dim=1)

        x = torch.cat([I1, I2, I3, trajectory_latent.weight.repeat(I1.shape[0], 1)], dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        x = self.unflatten(x)
        x = 0.5 * (self.transpose(x) + x)
        P = torch.matmul(R, x)
        cauchy = torch.matmul(P, Ft)
        return cauchy

class FprojLatentConditioned2(MetaPlasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = 3 + 9 + 1 + cfg.latent_dim
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim * self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor, trajectory_latent: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)
        U, sigma, Vh = self.svd(F)
        R = torch.matmul(U, Vh)

        Ft = self.transpose(F)
        FtF = torch.matmul(Ft, F)

        I1 = sigma - 1.0
        I2 = self.flatten(FtF - I)
        I3 = torch.linalg.det(F).unsqueeze(dim=1) - 1.0

        invariants = torch.cat([I1, I2, I3, trajectory_latent.weight.repeat(I1.shape[0], 1)], dim=1)
        x = invariants
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        x = self.unflatten(x)
        x = 0.5 * (self.transpose(x) + x)
        delta_Fp = self.alpha * torch.matmul(R, x)
        Fp = delta_Fp + F
        return Fp