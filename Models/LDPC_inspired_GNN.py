import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN, LayerNorm, Dropout, GELU, LeakyReLU
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.utils import dropout_node, dropout_edge
from torch_geometric.nn import GraphNorm


def MLP(channels, batch_norm=False, dropout_prob=0):
    layers = []
    for i in range(1, len(channels)):
        layers.append(Seq(Lin(channels[i - 1], channels[i])))
        if batch_norm:
            # layers.append(BN(channels[i]))
            layers.append(LayerNorm(channels[i]))
        if dropout_prob:
            layers.append(Dropout(dropout_prob))  # Add dropout after batch norm or activation
        # layers.append(ReLU())
        # layers.append(GELU())
        # layers.append(nn.SiLU()) # Shit
        layers.append(LeakyReLU(negative_slope=0.1))
    return Seq(*layers)

class VN2CNConv(MessagePassing):
    def __init__(self, vn_dim, cn_dim, msg_dim, out_cn_dim=None, drop_p=0.0):
        super().__init__(aggr='add')  # giống sum trong BP
        self.vn_dim = vn_dim
        self.cn_dim = cn_dim
        self.msg_dim = msg_dim
        self.drop_p = drop_p

        if out_cn_dim is None:
            out_cn_dim = cn_dim

        # message: từ VN sang CN
        self.msg_mlp = MLP([vn_dim, msg_dim], batch_norm=False, dropout_prob=drop_p)
        # update: CN nhận (x_cn, aggregated_msg)
        self.upd_mlp = MLP([cn_dim + msg_dim, out_cn_dim], batch_norm=False, dropout_prob=drop_p)

    def forward(self, x_vn, x_cn, edge_index_vc):
        # edge_index_vc: [2, E], 0: VN, 1: CN
        msg_agg = self.propagate(edge_index_vc, x=(x_vn, x_cn))  # [num_cn, msg_dim]
        cn_input = torch.cat([x_cn, msg_agg], dim=-1)
        x_cn_new = self.upd_mlp(cn_input)
        if x_cn_new.shape[-1] == x_cn.shape[-1]:
            x_cn_new = x_cn + x_cn_new  # residual
        return x_cn_new

    def message(self, x_j):
        # x_j: feature tại VN
        return self.msg_mlp(x_j)


class CN2VNConv(MessagePassing):
    def __init__(self, cn_dim, vn_dim, msg_dim, out_vn_dim=None, drop_p=0.0):
        super().__init__(aggr='add')
        self.cn_dim = cn_dim
        self.vn_dim = vn_dim
        self.msg_dim = msg_dim
        self.drop_p = drop_p

        if out_vn_dim is None:
            out_vn_dim = vn_dim

        self.msg_mlp = MLP([cn_dim, msg_dim], batch_norm=False, dropout_prob=drop_p)
        self.upd_mlp = MLP([vn_dim + msg_dim, out_vn_dim], batch_norm=False, dropout_prob=drop_p)

    def forward(self, x_cn, x_vn, edge_index_cv):
        # edge_index_cv: [2, E], 0: CN, 1: VN
        msg_agg = self.propagate(edge_index_cv, x=(x_cn, x_vn))  # [num_vn, msg_dim]
        vn_input = torch.cat([x_vn, msg_agg], dim=-1)
        x_vn_new = self.upd_mlp(vn_input)
        if x_vn_new.shape[-1] == x_vn.shape[-1]:
            x_vn_new = x_vn + x_vn_new
        return x_vn_new

    def message(self, x_j):
        return self.msg_mlp(x_j)
# class LDPCGNN(nn.Module):
#     def __init__(
#         self,
#         metadata,
#         dim_dict,          # {'VN': ..., 'CN': ..., 'edge': ...}
#         hidden_dim=32,     # dim ẩn cho VN/CN sau embed
#         msg_dim=32,        # dim message trên cạnh
#         num_iter=5,        # số vòng VN→CN→VN
#         hid_layers=4       # hidden cho power_edge MLP
#     ):
#         super().__init__()

#         self.metadata = metadata
#         self.vn_in_dim = dim_dict['VN']       # dim feature VN gốc (LLR, phi, ...)
#         self.cn_in_dim = dim_dict['CN']       # dim feature CN gốc
#         self.edge_dim = dim_dict['edge']      # số feature edge gốc (large_scale, var, ...)

#         self.hidden_dim = hidden_dim
#         self.msg_dim = msg_dim
#         self.num_iter = num_iter

#         # Embed node ban đầu
#         self.vn_embed = MLP([self.vn_in_dim, hidden_dim], batch_norm=False, dropout_prob=0.0)

#         # CN có thể dùng feature gốc rồi embed, hoặc dùng 1 vector learnable
#         self.cn_embed_in = MLP([self.cn_in_dim, hidden_dim], batch_norm=False, dropout_prob=0.0)

  
#         self.v2c_layers = nn.ModuleList()
#         self.c2v_layers = nn.ModuleList()
#         for _ in range(num_iter):
#             self.v2c_layers.append(
#                 VN2CNConv(hidden_dim, hidden_dim, msg_dim, out_cn_dim=hidden_dim, drop_p=0.1)
#             )
#             self.c2v_layers.append(
#                 CN2VNConv(hidden_dim, hidden_dim, msg_dim, out_vn_dim=hidden_dim, drop_p=0.1)
#             )

#         power_in_dim = hidden_dim + self.edge_dim + hidden_dim
#         self.power_edge_core = MLP(
#             [power_in_dim, hid_layers],
#             batch_norm=True,
#             dropout_prob=0.1
#         )
#         self.power_edge = nn.Sequential(
#             self.power_edge_core,
#             nn.Linear(hid_layers, 1)
#         )

#     def forward(self, batch):
#         x_dict = batch.x_dict
#         edge_index_dict = batch.edge_index_dict
#         edge_attr_dict = batch.edge_attr_dict



#         x_vn_raw = x_dict['VN']         
#         x_cn_raw = x_dict['CN']         

#         edge_index_vc = edge_index_dict[('VN', 'to', 'CN')]
#         edge_index_cv = edge_index_dict[('CN', 'to', 'VN')]

#         x_vn = self.vn_embed(x_vn_raw)
#         x_cn = self.cn_embed_in(x_cn_raw)

      
#         for k in range(self.num_iter):
#             x_cn = self.v2c_layers[k](x_vn, x_cn, edge_index_vc)
#             x_vn = self.c2v_layers[k](x_cn, x_vn, edge_index_cv)

#         x_dict = {
#             'VN': x_vn,
#             'CN': x_cn,
#         }
#         ei = edge_index_cv  
#         cn_idx = ei[0]
#         vn_idx = ei[1]

#         x_cn_edge = x_vn.new_zeros((ei.size(1), self.hidden_dim))
#         x_vn_edge = x_vn.new_zeros((ei.size(1), self.hidden_dim))

#         x_cn_edge = x_cn[cn_idx]   
#         x_vn_edge = x_vn[vn_idx]   
#         edge_feat = edge_attr_dict[('CN', 'to', 'VN')]  

#         power_input = torch.cat([x_cn_edge, edge_feat, x_vn_edge], dim=-1)
#         edge_power = self.power_edge(power_input)       
#         edge_attr_dict[('CN', 'to', 'VN')] = torch.cat(
#             [edge_feat[:, :self.edge_dim], edge_power],
#             dim=-1
#         )

#         return x_dict, edge_attr_dict, edge_index_dict
    
# New LDPCGNN
class LDPCGNN(nn.Module):
    def __init__(
        self,
        metadata,
        dim_dict,          # {'VN': ..., 'CN': ..., 'edge': ...}  (edge = số feature gốc, KHÔNG tính power)
        hidden_dim=32,
        msg_dim=32,
        num_iter=5,
        hid_layers=4
    ):
        super().__init__()

        self.metadata = metadata
        self.vn_in_dim = dim_dict['VN']
        self.cn_in_dim = dim_dict['CN']
        self.edge_dim  = dim_dict['edge']   # số feature edge GỐC (vd: [log(1+beta), gamma, ...])

        self.hidden_dim = hidden_dim
        self.msg_dim    = msg_dim
        self.num_iter   = num_iter

        # VN / CN embed
        self.vn_embed     = MLP([self.vn_in_dim, hidden_dim], batch_norm=False, dropout_prob=0.0)
        self.cn_embed_in  = MLP([self.cn_in_dim, hidden_dim], batch_norm=False, dropout_prob=0.0)

        # Message passing VN->CN và CN->VN
        self.v2c_layers = nn.ModuleList()
        self.c2v_layers = nn.ModuleList()
        for _ in range(num_iter):
            self.v2c_layers.append(
                VN2CNConv(hidden_dim, hidden_dim, msg_dim, out_cn_dim=hidden_dim, drop_p=0.1)
            )
            self.c2v_layers.append(
                CN2VNConv(hidden_dim, hidden_dim, msg_dim, out_vn_dim=hidden_dim, drop_p=0.1)
            )

        # MLP sinh power_raw trên cạnh ('CN','to','VN')
        # input = [x_cn_edge, edge_attr_goc, x_vn_edge]
        power_in_dim = hidden_dim + self.edge_dim + hidden_dim
        self.power_edge_core = MLP(
            [power_in_dim, hid_layers],
            batch_norm=True,
            dropout_prob=0.1
        )
        self.power_edge = nn.Sequential(
            self.power_edge_core,
            nn.Linear(hid_layers, 1)    # power_raw_{mk}
        )

    def forward(self, batch):
        # ===== 1. Lấy dict từ batch =====
        x_dict         = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict  = batch.edge_attr_dict

        # Node gốc
        x_vn_raw = x_dict['VN']    # [num_vn, vn_in_dim]
        x_cn_raw = x_dict['CN']    # [num_cn, cn_in_dim]

        # Edge index
        edge_index_vc = edge_index_dict[('VN', 'to', 'CN')]  # VN -> CN
        edge_index_cv = edge_index_dict[('CN', 'to', 'VN')]  # CN -> VN

        # ===== 2. Embed VN/CN =====
        x_vn = self.vn_embed(x_vn_raw)   # [num_vn, hidden_dim]
        x_cn = self.cn_embed_in(x_cn_raw)# [num_cn, hidden_dim]

        # ===== 3. LDPC-style message passing (num_iter vòng) =====
        for k in range(self.num_iter):
            x_cn = self.v2c_layers[k](x_vn, x_cn, edge_index_vc)  # VN -> CN
            x_vn = self.c2v_layers[k](x_cn, x_vn, edge_index_cv)  # CN -> VN

        # Cập nhật lại x_dict
        x_dict = {
            'VN': x_vn,
            'CN': x_cn,
        }

        # ===== 4. Sinh power_raw trên cạnh ('CN','to','VN') =====
        ei = edge_index_cv              # [2, E], 0: CN index m, 1: VN index k
        cn_idx = ei[0]
        vn_idx = ei[1]

        # Feature node theo cạnh
        x_cn_edge = x_cn[cn_idx]        # [E, hidden_dim]
        x_vn_edge = x_vn[vn_idx]        
        edge_feat_base = edge_attr_dict[('CN', 'to', 'VN')]  # [E, edge_dim]
        assert edge_feat_base.size(1) >= self.edge_dim

        power_input = torch.cat(
            [x_cn_edge, edge_feat_base[:, :self.edge_dim], x_vn_edge],
            dim=-1
        )   # [E, hidden_dim + edge_dim + hidden_dim]

        power_raw = self.power_edge(power_input).squeeze(-1)   # [E]

        # ===== 5. Áp constraint: sum_k p_{mk} * gamma_{mk} <= 1 với mỗi CN (AP) =====
        # Giả sử gamma_{mk} lưu ở cột 1 của edge_attr (giống channel_var):
        gamma_edge = edge_feat_base[:, 1]    # [E]  <-- chỉnh index nếu bạn lưu gamma ở cột khác

        num_cn = x_cn.size(0)

        # Tính sum_k p_{mk} * gamma_{mk} theo từng CN (AP)
        # sử dụng scatter_add:
        sum_pg = power_raw * gamma_edge      # [E]
        sum_pg_per_cn = torch.zeros(num_cn, device=power_raw.device)
        sum_pg_per_cn.scatter_add_(0, cn_idx, sum_pg)

        # scale_m = 1 nếu thỏa, oặc = 1 / sum_pg nếu sum_pg > 1
        ones = torch.ones_like(sum_pg_per_cn)
        scale_m = torch.where(
            sum_pg_per_cn > 1.0,
            1.0 / (sum_pg_per_cn ),
            ones
        )   # [num_cn]

        # scale cho từng edge
        scale_edge = scale_m[cn_idx]         # [E]
        power_constrained = power_raw * scale_edge  # [E]

        # Gắn lại vào edge_attr: [edge_feat_base, power_constrained]
        edge_attr_dict[('CN', 'to', 'VN')] = torch.cat(
            [edge_feat_base[:, :self.edge_dim], power_constrained.unsqueeze(-1)],
            dim=-1
        )

        # ===== 6. Trả kết quả giống model GNN cũ =====
        return x_dict, edge_attr_dict, edge_index_dict