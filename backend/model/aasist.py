"""
AASIST Implementation (Compatible with clovaai/aasist)
Adapted for AgriNetra Forensic Engine
"""
import random
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x):
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map(self, x):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x1, x2, master=None):
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
        x = self.input_drop(x)
        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)
        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        return x1, x2, master

    def _update_master(self, x, master):
        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)
        return master

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        att_board[:, :num_type1, :num_type1, :] = torch.matmul(att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(att_map[:, num_type1:, :num_type1, :], self.att_weight12)
        return F.softmax(att_board / self.temp, dim=-2)

    def _project(self, x, att_map):
        return self.proj_with_att(torch.matmul(att_map.squeeze(-1), x)) + self.proj_without_att(x)

    def _project_master(self, x, master, att_map):
        x1 = self.proj_with_attM(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * self.k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        h = torch.gather(h, 1, idx)
        return h

class SincConv(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        self.sample_rate = sample_rate
        
        f = int(self.sample_rate / 2) * np.linspace(0, 1, 257)
        fmel = 2595 * np.log10(1 + f / 700)
        filbandwidthsmel = np.linspace(np.min(fmel), np.max(fmel), self.out_channels + 1)
        self.mel = 700 * (10**(filbandwidthsmel / 2595) - 1)
        
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin, fmax = self.mel[i], self.mel[i + 1]
            hHigh = (2*fmax/self.sample_rate) * np.sinc(2*fmax*self.hsupp.numpy()/self.sample_rate)
            hLow = (2*fmin/self.sample_rate) * np.sinc(2*fmin*self.hsupp.numpy()/self.sample_rate)
            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hHigh - hLow)

    def forward(self, x, mask=False):
        filters = self.band_pass.to(x.device).view(self.out_channels, 1, self.kernel_size)
        if mask:
            A = int(np.random.uniform(0, 20))
            A0 = random.randint(0, filters.shape[0] - A)
            filters[A0:A0 + A, :, :] = 0
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size//2)

class ResidualBlock(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], (2, 3), padding=(1, 1))
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], (2, 3), padding=(0, 1))
        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv2d(nb_filts[0], nb_filts[1], (1, 3), padding=(0, 1))
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x
        out = self.selu(self.bn1(x)) if not self.first else x
        out = self.bn2(self.conv1(out))
        out = self.conv2(self.selu(out))
        if self.downsample:
            identity = self.conv_downsample(identity)
        out += identity
        return self.mp(out)

class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()
        filts = d_args["filts"]
        self.conv_time = SincConv(filts[0], d_args["first_conv"])
        self.encoder = nn.Sequential(
            ResidualBlock(filts[1], first=True),
            ResidualBlock(filts[2]),
            ResidualBlock(filts[3]),
            ResidualBlock(filts[4]),
            ResidualBlock(filts[4]),
            ResidualBlock(filts[4])
        )
        self.pos_S = nn.Parameter(torch.randn(1, 60, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, d_args["gat_dims"][0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, d_args["gat_dims"][0]))
        self.GAT_S = GraphAttentionLayer(filts[-1][-1], d_args["gat_dims"][0], temperature=d_args["temperatures"][0])
        self.GAT_T = GraphAttentionLayer(filts[-1][-1], d_args["gat_dims"][0], temperature=d_args["temperatures"][1])
        self.HtrgGAT_11 = HtrgGraphAttentionLayer(d_args["gat_dims"][0], d_args["gat_dims"][1], temperature=d_args["temperatures"][2])
        self.HtrgGAT_12 = HtrgGraphAttentionLayer(d_args["gat_dims"][1], d_args["gat_dims"][1], temperature=d_args["temperatures"][2])
        self.HtrgGAT_21 = HtrgGraphAttentionLayer(d_args["gat_dims"][0], d_args["gat_dims"][1], temperature=d_args["temperatures"][2])
        self.HtrgGAT_22 = HtrgGraphAttentionLayer(d_args["gat_dims"][1], d_args["gat_dims"][1], temperature=d_args["temperatures"][2])
        self.pool_S = GraphPool(d_args["pool_ratios"][0], d_args["gat_dims"][0], 0.3)
        self.pool_T = GraphPool(d_args["pool_ratios"][1], d_args["gat_dims"][0], 0.3)
        self.pool_hS1 = GraphPool(d_args["pool_ratios"][2], d_args["gat_dims"][1], 0.3)
        self.pool_hT1 = GraphPool(d_args["pool_ratios"][2], d_args["gat_dims"][1], 0.3)
        self.out_layer = nn.Linear(5 * d_args["gat_dims"][1], 2)

    def forward(self, x, Freq_aug=False):
        # Handle 2D features if provided directly (user suggested LFCC)
        if len(x.shape) == 3 and x.shape[1] > 1:
            # User wants LFCC integration - mapping to encoder
            # If x is [batch, feature, time], we treat it like conv output
            # This is a fallback mapping for the user's LFCC requirement
            x = x.unsqueeze(1) # [B, 1, F, T]
            e = self.encoder(x)
        else:
            # Official RAW waveform flow
            if len(x.shape) == 2: x = x.unsqueeze(1)
            x = self.conv_time(x, mask=Freq_aug)
            x = F.max_pool2d(torch.abs(x.unsqueeze(1)), (3, 3))
            e = self.encoder(x)

        e_S, _ = torch.max(torch.abs(e), dim=3)
        e_S = e_S.transpose(1, 2) + self.pos_S
        out_S = self.pool_S(self.GAT_S(e_S))

        e_T, _ = torch.max(torch.abs(e), dim=2)
        out_T = self.pool_T(self.GAT_T(e_T.transpose(1, 2)))

        master1 = self.master1.expand(x.size(0), -1, -1)
        out_T1, out_S1, m1 = self.HtrgGAT_11(out_T, out_S, master=master1)
        out_T_a, out_S_a, m_a = self.HtrgGAT_12(self.pool_hT1(out_T1), self.pool_hS1(out_S1), master=m1)
        
        # Simplified fusion for inference engine
        last_hidden = torch.cat([
            torch.max(torch.abs(out_T1), dim=1)[0],
            torch.mean(out_T1, dim=1),
            torch.max(torch.abs(out_S1), dim=1)[0],
            torch.mean(out_S1, dim=1),
            m1.squeeze(1)
        ], dim=1)
        
        return last_hidden, self.out_layer(last_hidden)
