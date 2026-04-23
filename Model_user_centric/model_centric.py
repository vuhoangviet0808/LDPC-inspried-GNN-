import torch
import torch.nn as nn
import torch.nn.functional as F


def MLP(channels, batch_norm=False, dropout_prob=0.0):
    layers = []
    for i in range(1, len(channels)):
        layers.append(nn.Linear(channels[i - 1], channels[i]))
        if batch_norm:
            layers.append(nn.LayerNorm(channels[i]))
        if dropout_prob > 0:
            layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.LeakyReLU(negative_slope=0.1))
    return nn.Sequential(*layers)


def _scatter_sum(src, index, dim_size):
    out = src.new_zeros((dim_size,) + src.shape[1:])
    out.index_add_(0, index, src)
    return out


class LDPCUserCentricGNN(nn.Module):
    """
    User-centric LDPC-inspired GNN

    - Association head: VN -> CN viewpoint
        a_mk in [0, 1]
    - Power head: CN -> VN viewpoint
        p_raw >= 0
    - Effective power:
        p_eff = a_soft * p_raw

    Output edge feature on ('CN','to','VN'):
        [edge_feat_base, a_soft, p_eff]
    """

    def __init__(
        self,
        vn_dim,
        cn_dim,
        edge_dim,
        message_dim=32,
        hidden=64,
        num_iter=4,
        gamma_idx=1,
        power_limit=1.0,
        assoc_temperature=1.0,
    ):
        super().__init__()
        self.num_iter = num_iter
        self.gamma_idx = gamma_idx
        self.power_limit = float(power_limit)
        self.edge_dim = edge_dim
        self.assoc_temperature = assoc_temperature

        # ===== Message passing blocks =====
        self.vn2cn = MLP(
            [vn_dim + edge_dim, message_dim],
            batch_norm=True,
            dropout_prob=0.0,
        )
        self.cn_upd = MLP(
            [cn_dim + message_dim, cn_dim],
            batch_norm=True,
            dropout_prob=0.0,
        )

        self.cn2vn = MLP(
            [cn_dim + edge_dim + message_dim, message_dim],
            batch_norm=True,
            dropout_prob=0.0,
        )
        self.vn_upd = MLP(
            [vn_dim + message_dim, vn_dim],
            batch_norm=True,
            dropout_prob=0.0,
        )

        # ===== Association head (UE-centric / VN->CN) =====
        # Input: VN feature + edge feature + current CN aggregated context
        self.assoc_mlp = MLP(
            [vn_dim + edge_dim + message_dim, hidden],
            batch_norm=False,
            dropout_prob=0.1,
        )
        self.assoc_out = nn.Linear(hidden, 1)

        # ===== Power head (AP-centric / CN->VN) =====
        # Input: edge feature + CN feature + VN feature
        self.power_mlp = MLP(
            [edge_dim + cn_dim + vn_dim, hidden],
            batch_norm=False,
            dropout_prob=0.1,
        )
        self.power_out = nn.Linear(hidden, 1)

    def forward(self, batch):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict

        edge_key = ('CN', 'to', 'VN')
        edge_index = edge_index_dict[edge_key]    # [2, E]
        cn_idx = edge_index[0]
        vn_idx = edge_index[1]

        E = edge_index.size(1)
        N_cn = x_dict['CN'].size(0)
        N_vn = x_dict['VN'].size(0)

        edge_attr = edge_attr_dict[edge_key]      # [E, edge_dim]
        gamma = edge_attr[:, self.gamma_idx].view(-1)  # [E]

        # initialize edge logits
        p_logits = torch.zeros(E, device=edge_attr.device, dtype=edge_attr.dtype)
        a_logits = torch.zeros(E, device=edge_attr.device, dtype=edge_attr.dtype)

        for _ in range(self.num_iter):
            # ==========================================================
            # 1) Estimate overload from previous effective power
            # ==========================================================
            a_soft_prev = torch.sigmoid(a_logits / self.assoc_temperature)   # [E]
            p_raw_prev = F.relu(p_logits)                                    # [E]
            p_eff_prev = a_soft_prev * p_raw_prev                            # [E]

            weighted = p_eff_prev * gamma
            S_per_cn = _scatter_sum(weighted, cn_idx, N_cn)                  # [N_cn]

            scale_cn = torch.ones_like(S_per_cn)
            mask_ol = S_per_cn > self.power_limit
            scale_cn[mask_ol] = self.power_limit / S_per_cn[mask_ol]

            scale_edge_msg = scale_cn[cn_idx].view(-1, 1)                    # [E,1]

            # ==========================================================
            # 2) VN -> CN message
            # ==========================================================
            vn_feats = x_dict['VN'][vn_idx]                                  # [E, vn_dim]
            inp_vn2cn = torch.cat([vn_feats, edge_attr], dim=1)
            m_vn2cn = self.vn2cn(inp_vn2cn)                                  # [E, message_dim]

            agg_cn = _scatter_sum(m_vn2cn, cn_idx, N_cn)                     # [N_cn, message_dim]

            cn_inp = torch.cat([x_dict['CN'], agg_cn], dim=1)
            x_dict['CN'] = self.cn_upd(cn_inp)

            # ==========================================================
            # 3) Association head from VN -> CN viewpoint
            # ==========================================================
            # UE-centric: use VN feature and the AP-side aggregated context
            agg_cn_per_edge = agg_cn[cn_idx]                                 # [E, message_dim]

            assoc_inp = torch.cat([vn_feats, edge_attr, agg_cn_per_edge], dim=1)
            h_assoc = self.assoc_mlp(assoc_inp)
            a_logits = self.assoc_out(h_assoc).view(-1)

            # ==========================================================
            # 4) CN -> VN message with constraint-aware scaling
            # ==========================================================
            m_excl = agg_cn_per_edge - m_vn2cn                               # [E, message_dim]
            cn_feats = x_dict['CN'][cn_idx]                                  # [E, cn_dim]

            inp_cn2vn = torch.cat([cn_feats, edge_attr, m_excl], dim=1)
            m_cn2vn = self.cn2vn(inp_cn2vn)
            m_cn2vn = m_cn2vn * scale_edge_msg

            agg_vn = _scatter_sum(m_cn2vn, vn_idx, N_vn)
            vn_inp = torch.cat([x_dict['VN'], agg_vn], dim=1)
            x_dict['VN'] = self.vn_upd(vn_inp)

            # ==========================================================
            # 5) Power head from CN -> VN viewpoint
            # ==========================================================
            cn_feats_new = x_dict['CN'][cn_idx]
            vn_feats_new = x_dict['VN'][vn_idx]

            power_inp = torch.cat([edge_attr, cn_feats_new, vn_feats_new], dim=1)
            h_power = self.power_mlp(power_inp)
            p_logits = self.power_out(h_power).view(-1)

        # ==============================================================
        # Final outputs
        # ==============================================================
        a_soft = torch.sigmoid(a_logits / self.assoc_temperature)   # [E]
        p_raw = F.relu(p_logits)                                    # [E]
        p_eff = a_soft * p_raw                                      # [E]

        # Save on CN->VN edges:
        # [base_edge_feat, association_score, effective_power]
        edge_attr_dict[edge_key] = torch.cat(
            [
                edge_attr[:, :self.edge_dim],
                a_soft.view(-1, 1),
                p_eff.view(-1, 1),
            ],
            dim=1
        )

        return x_dict, edge_attr_dict, edge_index_dict