import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU, LayerNorm, Dropout, Sigmoid

# Lightweight MLP helper (self-contained)
def MLP(channels, batch_norm=False, dropout_prob=0):
    layers = []
    for i in range(1, len(channels)):
        layers.append(Seq(Lin(channels[i - 1], channels[i])))
        if batch_norm:
            layers.append(LayerNorm(channels[i]))
        if dropout_prob:
            layers.append(Dropout(dropout_prob))
        layers.append(LeakyReLU(negative_slope=0.1))
    return Seq(*layers)



class LDPCConvLayer(nn.Module):
    """LDPC-like layer (variable-check bipartite message passing).

    - If `deterministic=True`, the check graph is constructed deterministically
      per-graph (based on graph index and offsets) to reduce train/test variance.
    """
    def __init__(self, feat_dim, num_checks=4, degree=2, num_iter=2, dropout=0.1, deterministic=False):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_checks = num_checks
        self.degree = degree
        self.num_iter = num_iter
        self.det = deterministic

        self.v2c = MLP([feat_dim * 2, feat_dim], batch_norm=True, dropout_prob=dropout)
        self.c2v = MLP([feat_dim * 2, feat_dim], batch_norm=True, dropout_prob=dropout)
        self.ln_chk = LayerNorm(feat_dim)
        self.ln_ue = LayerNorm(feat_dim)
        self.drop = Dropout(dropout)

    def build_edges(self, num_graphs, num_ues, device):
        # deterministic mapping: for each graph g and check j, connect UEs (g*base + (j + k) % num_ues)
        srcs = []
        dsts = []
        for g in range(num_graphs):
            ue_offset = g * num_ues
            chk_offset = g * self.num_checks
            for chk in range(self.num_checks):
                if self.det:
                    # deterministic selection (wrap-around)
                    uids = [(ue_offset + ((chk + k) % num_ues)) for k in range(self.degree)]
                    uidx = torch.tensor(uids, dtype=torch.long, device=device)
                else:
                    # random selection per-batch
                    uidx = torch.randperm(num_ues, device=device)[:self.degree] + ue_offset
                srcs.append(uidx)
                dsts.append(torch.full((self.degree,), chk + chk_offset, device=device, dtype=torch.long))
        src_flat = torch.cat(srcs).long()
        dst_flat = torch.cat(dsts).long()
        edge_index = torch.stack([src_flat, dst_flat], dim=0)
        return edge_index

    def forward(self, x_ue, batch):
        device = x_ue.device
        num_graphs = batch.num_graphs
        N = x_ue.shape[0]
        if N % num_graphs != 0:
            raise ValueError("UEs must be divisible by num_graphs")
        num_ues = N // num_graphs

        edge_index = self.build_edges(num_graphs, num_ues, device)
        E = edge_index.shape[1]

        # zero-initialized check features
        total_chks = num_graphs * self.num_checks
        x_chk = torch.zeros((total_chks, self.feat_dim), device=device, dtype=x_ue.dtype)

        for _ in range(self.num_iter):
            x_src = x_ue[edge_index[0]]  # (E, F)
            x_dst = x_chk[edge_index[1]]  # (E, F)

            m_v2c = self.v2c(torch.cat([x_src, x_dst], dim=1))
            m_v2c = self.drop(m_v2c)

            chk_agg = scatter_add(m_v2c, edge_index[1], dim=0, dim_size=total_chks)
            chk_agg = self.ln_chk(chk_agg)

            ext = chk_agg[edge_index[1]] - m_v2c

            m_c2v = self.c2v(torch.cat([ext, x_src], dim=1))
            m_c2v = self.drop(m_c2v)

            ue_agg = scatter_add(m_c2v, edge_index[0], dim=0, dim_size=N)
            ue_agg = self.ln_ue(ue_agg)

            upd = ue_agg
            x_ue = x_ue + self.drop(upd)

        return x_ue


class LDPCStandaloneSimple(nn.Module):
    """Standalone small heterogeneous network + LDPC block.

    Designed to be a lightweight, self-contained comparison to APHetNet.
    """
    def __init__(self, dim_dict, hidden=32, num_mp_layers=2, ldpc_checks=4, ldpc_degree=2, ldpc_iter=2, ldpc_dropout=0.1, deterministic_ldpc=False):
        super().__init__()
        self.ue_in = dim_dict['UE']
        self.ap_in = dim_dict['AP']
        self.edge_in = dim_dict['edge']
        self.hidden = hidden
        self.num_mp_layers = num_mp_layers

        # projections
        self.proj_ue = MLP([self.ue_in, hidden], batch_norm=True, dropout_prob=0)
        self.proj_ap = MLP([self.ap_in, hidden], batch_norm=True, dropout_prob=0)
        self.proj_edge = MLP([self.edge_in, hidden], batch_norm=True, dropout_prob=0)

        # simple message MLPs
        self.msg_ue2ap = MLP([hidden * 3, hidden], batch_norm=True, dropout_prob=0.1)
        self.upd_ap = MLP([hidden * 2, hidden], batch_norm=True, dropout_prob=0.1)
        self.msg_ap2ue = MLP([hidden * 3, hidden], batch_norm=True, dropout_prob=0.1)
        self.upd_ue = MLP([hidden * 2, hidden], batch_norm=True, dropout_prob=0.1)

        # LDPC block
        self.ldpc = LDPCConvLayer(feat_dim=hidden, num_checks=ldpc_checks, degree=ldpc_degree, num_iter=ldpc_iter, dropout=ldpc_dropout, deterministic=deterministic_ldpc)

        # power head
        self.power_head = MLP([hidden, hidden], batch_norm=True, dropout_prob=0)
        self.power_head = nn.Sequential(*[self.power_head, Seq(Lin(hidden, 1)), Sigmoid()])

    def forward(self, batch):
        x_dict, edge_index_dict, edge_attr_dict = batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict

        x_ue = self.proj_ue(x_dict['UE'])
        x_ap = self.proj_ap(x_dict['AP'])
        # project edges
        for k, v in edge_attr_dict.items():
            edge_attr_dict[k] = self.proj_edge(v)

        # message passing
        for _ in range(self.num_mp_layers):
            # UE->AP
            et = ('UE', 'up', 'AP')
            if et in edge_index_dict:
                ei = edge_index_dict[et]
                src, dst = ei[0], ei[1]
                m = self.msg_ue2ap(torch.cat([x_ue[src], edge_attr_dict[et], x_ap[dst]], dim=1))
                agg_ap = scatter_add(m, dst, dim=0, dim_size=x_ap.shape[0])
                x_ap = self.upd_ap(torch.cat([x_ap, agg_ap], dim=1)) + x_ap

            # AP->UE
            et2 = ('AP', 'down', 'UE')
            if et2 in edge_index_dict:
                ei2 = edge_index_dict[et2]
                src2, dst2 = ei2[0], ei2[1]
                m2 = self.msg_ap2ue(torch.cat([x_ap[src2], edge_attr_dict[et2], x_ue[dst2]], dim=1))
                agg_ue = scatter_add(m2, dst2, dim=0, dim_size=x_ue.shape[0])
                x_ue = self.upd_ue(torch.cat([x_ue, agg_ue], dim=1)) + x_ue

        # LDPC updates (operate on hidden UE features)
        x_ue = self.ldpc(x_ue, batch)

        # power output
        dl_power = torch.exp(self.power_head(x_ue))
        # attach as last-dim (keep original features left intact)
        x_dict['UE'] = torch.cat([x_dict['UE'][:, :self.ue_in], dl_power], dim=1)

        return x_dict, edge_attr_dict, edge_index_dict
