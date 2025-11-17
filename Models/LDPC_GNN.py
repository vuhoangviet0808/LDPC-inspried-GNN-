import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN, LayerNorm, Dropout, GELU, LeakyReLU
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.utils import dropout_node
from torch_geometric.nn import GraphNorm
from torch_scatter import scatter_add
from Models.GNN import APHetNet, MLP


class LDPCConvLayer(nn.Module):
    """
    LDPC-inspired message passing layer with regularization.

    It builds a virtual bipartite factor graph between UEs (variable nodes) and
    parity/check nodes (CHK) and performs a few LDPC-like iterations of
    variable-to-check and check-to-variable message passing.

    Includes dropout and layer normalization to reduce overfitting.
    """
    def __init__(self, ue_dim, chk_dim, edge_dim, out_channel, num_checks=8, degree=3, num_iter=2, dropout_prob=0.1):
        super(LDPCConvLayer, self).__init__()
        self.ue_dim = ue_dim
        self.chk_dim = chk_dim
        self.edge_dim = edge_dim
        self.out_channel = out_channel
        self.num_checks = num_checks
        self.degree = degree
        self.num_iter = num_iter
        self.dropout_prob = dropout_prob

        # MLPs for messages with dropout for regularization
        self.v2c = MLP([ue_dim + edge_dim + chk_dim, out_channel], batch_norm=True, dropout_prob=dropout_prob)
        self.c2v = MLP([out_channel + chk_dim + ue_dim, out_channel], batch_norm=True, dropout_prob=dropout_prob)

        # Node update for UE
        self.update_ue = MLP([out_channel + ue_dim, out_channel], batch_norm=True, dropout_prob=dropout_prob)
        self.gamma = nn.Parameter(torch.full((out_channel - ue_dim if out_channel>ue_dim else 1,), 1e-3))
        
        # Layer norms for stability
        self.ln_chk = LayerNorm(out_channel)
        self.ln_ue = LayerNorm(out_channel)
        self.dropout = Dropout(dropout_prob)

    def forward(self, x_ue, batch):
        # x_ue: [total_UES, ue_dim]
        device = x_ue.device
        num_graphs = batch.num_graphs
        num_ues_total = x_ue.shape[0]
        if num_ues_total % num_graphs != 0:
            raise ValueError("UEs must be divisible by graphs in batch")
        num_ues = num_ues_total // num_graphs

        # build CHK nodes per graph
        total_chks = num_graphs * self.num_checks
        x_chk = torch.zeros((total_chks, self.chk_dim), device=device)

        # prepare random bipartite edges (UE -> CHK) per graph
        srcs = []
        dsts = []
        for g in range(num_graphs):
            ue_offset = g * num_ues
            chk_offset = g * self.num_checks
            for chk in range(self.num_checks):
                # pick `degree` random UEs from this graph
                uidx = torch.randperm(num_ues, device=device)[:self.degree] + ue_offset
                srcs.append(uidx)
                dsts.append(torch.full((self.degree,), chk + chk_offset, device=device, dtype=torch.long))

        if len(srcs) == 0:
            return x_ue

        src_flat = torch.cat(srcs).long()
        dst_flat = torch.cat(dsts).long()
        edge_index = torch.stack([src_flat, dst_flat], dim=0)

        # simple (zero) edge attributes
        edge_attr = torch.zeros((edge_index.shape[1], self.edge_dim), device=device)

        # Now LDPC-like message passing for num_iter iterations
        for it in range(self.num_iter):
            # V -> C messages (per edge)
            x_src = x_ue[edge_index[0]]
            x_dst = x_chk[edge_index[1]]
            m_v2c = self.v2c(torch.cat([x_src, edge_attr, x_dst], dim=1))
            m_v2c = self.dropout(m_v2c)

            # aggregate at check nodes with layer norm
            chk_agg = scatter_add(m_v2c, edge_index[1], dim=0, dim_size=total_chks)
            chk_agg = self.ln_chk(chk_agg)

            # extrinsic messages: for each edge, remove its own contribution
            ext = chk_agg[edge_index[1]] - m_v2c

            # C -> V messages
            m_c2v = self.c2v(torch.cat([ext, x_chk[edge_index[1]], x_src], dim=1))
            m_c2v = self.dropout(m_c2v)

            # aggregate at UE nodes with layer norm
            ue_agg = scatter_add(m_c2v, edge_index[0], dim=0, dim_size=num_ues_total)
            ue_agg = self.ln_ue(ue_agg)

            # update UE features
            tmp = torch.cat([x_ue, ue_agg], dim=1)
            tmp = self.update_ue(tmp)
            # simple residual/gating with dropout
            if x_ue.shape[1] < tmp.shape[1]:
                x_ue = torch.cat([x_ue[:,:x_ue.shape[1]], tmp], dim=1)
            else:
                x_ue = x_ue + self.dropout(tmp)

        return x_ue


class LDPCHetNet(APHetNet):
    """
    AP + LDPC hybrid network with regularization.

    - Runs the APHetNet message passing for AP-UE interactions
    - Adds an LDPC-inspired bipartite factor graph among UEs
      to perform constraint-like message passing with dropout.
    """
    def __init__(self, metadata, dim_dict, out_channels, num_layers=0, hid_layers=4, edge_conv=False,
                 num_checks=8, chk_degree=3, ldpc_iter=2, ldpc_dropout=0.1):
        super(LDPCHetNet, self).__init__(metadata, dim_dict, out_channels, num_layers, hid_layers, edge_conv)
        # We'll take UE dim as variable node dimension
        self.ldpc = LDPCConvLayer(ue_dim=out_channels, chk_dim=out_channels, edge_dim=self.edge_dim,
                                  out_channel=out_channels, num_checks=num_checks, degree=chk_degree, 
                                  num_iter=ldpc_iter, dropout_prob=ldpc_dropout)

    def forward(self, batch):
        # First APHetNet computations
        x_dict, edge_attr_dict, edge_index_dict = super(LDPCHetNet, self).forward(batch)

        # Now do LDPC-inspired updates on UE features
        x_ue = x_dict['UE']
        x_ue = self.ldpc(x_ue, batch)
        x_dict['UE'] = x_ue

        return x_dict, edge_attr_dict, edge_index_dict
