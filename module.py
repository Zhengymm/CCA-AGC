import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv


def contrastive_loss_batch(z1, z2, temperature=1):
    batch_size = 2000
    z1 = F.normalize(z1, dim=-1, p=2)
    z2 = F.normalize(z2, dim=-1, p=2)
    # device = z1.device
    # pos_mask = torch.eye(z1.size(0), dtype=torch.float32)
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / temperature)
    indices = torch.arange(0, num_nodes)
    losses = []

    # neg_mask = 1 - pos_mask

    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]
        # intra_sim = f(torch.mm(z1[mask], z1.t()))  # [B, N]
        intra_sim_11 = f(torch.mm(z1[mask], z1.t()))  # [B, N]
        intra_sim_22 = f(torch.mm(z2[mask], z2.t()))  # [B, N]
        inter_sim = f(torch.mm(z1[mask], z2.t()))  # [B, N]

        loss_12 = -torch.log(
            inter_sim[:, i * batch_size:(i + 1) * batch_size].diag()
            / (intra_sim_11.sum(1) + inter_sim.sum(1)
               - intra_sim_11[:, i * batch_size:(i + 1) * batch_size].diag()))
        loss_21 = -torch.log(
            inter_sim[:, i * batch_size:(i + 1) * batch_size].diag()
            / (intra_sim_22.sum(1) + inter_sim.sum(1)
               - intra_sim_22[:, i * batch_size:(i + 1) * batch_size].diag()))
        losses.append(loss_12+loss_21)

    return torch.cat(losses)


def contrastive_cross_view(h1, h2, temperature=1):
    z1 = F.normalize(h1, dim=-1, p=2)
    z2 = F.normalize(h2, dim=-1, p=2)
    f = lambda x: torch.exp(x / temperature)
    intra_sim = f(torch.mm(z1, z1.t()))
    inter_sim = f(torch.mm(z1, z2.t()))

    # loss = -torch.log(inter_sim.diag() / (intra_sim.sum(dim=-1) + inter_sim.sum(dim=-1) - intra_sim.diag()))

    pos_mask = torch.eye(z1.size(0), dtype=torch.float32).to(z1.device)

    neg_mask = 1 - pos_mask
    pos = (inter_sim * pos_mask).sum(1)  # pos <=> between_sim.diag()
    neg = (intra_sim * neg_mask).sum(1)  # neg <=> refl_sim.sum(1) - refl_sim.diag()

    loss = -torch.log(pos / (inter_sim.sum(1) + neg))  # inter_sim.sum(1) = (inter_sim*neg_mask).sum(1) + pos

    return loss


class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, x):
        return self.net(x)


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden,
                 activation, base_model=GCNConv):
        super(Encoder, self).__init__()
        self.base_model = base_model

        self.gcn1 = base_model(in_channels, hidden)
        self.gcn2 = base_model(hidden, out_channels)

        self.activation = nn.PReLU() if activation == 'prelu' else nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.activation(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x


class Contra(Module):
    def __init__(self,
                 encoder,
                 hidden_size,
                 projection_size,
                 projection_hidden_size,
                 n_cluster,
                 v=1):
        super().__init__()

        # backbone encoder
        self.encoder = encoder

        # projection layer for representation contrastive
        self.rep_projector = MLP(hidden_size, projection_size, projection_hidden_size)
        # t-student cluster layer for clustering
        self.cluster_layer = nn.Parameter(torch.Tensor(n_cluster, hidden_size), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = v

    def kl_cluster(self, z1: torch.Tensor, z2: torch.Tensor):
        q1 = 1.0 / (1.0 + torch.sum(torch.pow(z1.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)  # q1 n*K
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        q2 = 1.0 / (1.0 + torch.sum(torch.pow(z2.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q2 = q2.pow((self.v + 1.0) / 2.0)
        q2 = (q2.t() / torch.sum(q2, 1)).t()

        return q1, q2

    def forward(self, feat, adj):
        h = self.encoder(feat, adj)
        z = self.rep_projector(h)

        return h, z

