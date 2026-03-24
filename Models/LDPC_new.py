import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.centralized_train import component_calculate, rate_from_component, rate_calculation

def MLP(channels, batch_norm=False, dropout_prob=0):
    layers = []
    for i in range(1, len(channels)):
        layers.append(nn.Linear(channels[i - 1], channels[i]))
        if batch_norm:
            layers.append(nn.LayerNorm(channels[i]))
        if dropout_prob:
            layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.LeakyReLU())
    return nn.Sequential(*layers)




class LDPCPowerGNN(nn.Module):
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
    ):
        super().__init__()
        self.num_iter     = num_iter
        self.gamma_idx    = gamma_idx
        self.power_limit  = float(power_limit)
        self.edge_dim     = edge_dim  # số feature edge gốc (không tính power)

        # VN <-> CN message MLPs
        self.vn2cn = MLP([vn_dim + edge_dim, message_dim],
                         batch_norm=False, dropout_prob=0.0)
        self.cn_upd = MLP([cn_dim + message_dim, cn_dim],
                          batch_norm=False, dropout_prob=0.0)

        self.cn2vn = MLP([cn_dim + edge_dim + message_dim, message_dim],
                         batch_norm=False, dropout_prob=0.0)
        self.vn_upd = MLP([vn_dim + message_dim, vn_dim],
                          batch_norm=False, dropout_prob=0.0)

        # Edge MLP -> logit s_e
        self.edge_mlp = MLP(
            [edge_dim + cn_dim + vn_dim, hidden],
            batch_norm=False,
            dropout_prob=0.1
        )
        self.edge_out = nn.Linear(hidden, 1)

    def forward(self, batch):
        x_dict          = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict  = batch.edge_attr_dict

        edge_key = ('CN', 'to', 'VN')

        edge_index = edge_index_dict[edge_key]  # (2, E)
        cn_idx = edge_index[0]
        vn_idx = edge_index[1]

        E    = edge_index.size(1)
        N_cn = x_dict['CN'].size(0)
        N_vn = x_dict['VN'].size(0)

        edge_attr = edge_attr_dict[edge_key]          #    [E, edge_dim_goc]
        gamma     = edge_attr[:, self.gamma_idx].view(-1)  # (E,)

        # khởi tạo logits per-edge
        s_logits = torch.zeros(E, device=gamma.device, dtype=gamma.dtype)

        for _ in range(self.num_iter):
            s_pos_prev = F.relu(s_logits)             # [E]
            weighted   = s_pos_prev * gamma           # [E]
            S_per_cn   = _scatter_sum(weighted, cn_idx, N_cn)  


            scale_cn = torch.ones_like(S_per_cn)
            mask_ol  = S_per_cn > self.power_limit
            scale_cn[mask_ol] =   self.power_limit /scale_cn[mask_ol]

            scale_edge     = scale_cn[cn_idx]        
            scale_edge_msg = scale_edge.view(-1, 1)

            # ===== 2) VN -> CN messages =====
            vn_feats   = x_dict['VN'][vn_idx]         # (E, vn_dim)
            inp_vn2cn  = torch.cat([vn_feats, edge_attr], dim=1)
            m_vn2cn    = self.vn2cn(inp_vn2cn)        # (E, message_dim)

            agg_cn     = _scatter_sum(m_vn2cn, cn_idx, N_cn)  # (N_cn, message_dim)

            cn_inp     = torch.cat([x_dict['CN'], agg_cn], dim=1)
            x_dict['CN'] = self.cn_upd(cn_inp)

            # ===== 3) CN -> VN messages (đưa constraint vào) =====
            agg_cn_per_edge = agg_cn[cn_idx]          # (E, message_dim)
            m_excl          = agg_cn_per_edge - m_vn2cn

            cn_feats   = x_dict['CN'][cn_idx]         # (E, cn_dim)

            inp_cn2vn  = torch.cat([cn_feats, edge_attr, m_excl], dim=1)
            m_cn2vn    = self.cn2vn(inp_cn2vn)
            m_cn2vn    = m_cn2vn * scale_edge_msg     # AP quá tải gửi message yếu hơn

            agg_vn     = _scatter_sum(m_cn2vn, vn_idx, N_vn)
            vn_inp     = torch.cat([x_dict['VN'], agg_vn], dim=1)
            x_dict['VN'] = self.vn_upd(vn_inp)

            # ===== 4) Cập nhật s_logits với feature mới =====
            inp_edge   = torch.cat([edge_attr, cn_feats, vn_feats], dim=1)
            h          = self.edge_mlp(inp_edge)
            s_logits   = self.edge_out(h).view(-1)
 
        # s_pos = F.relu(s_logits)          # [E]
        # weighted_final = s_pos * gamma
        # S_per_cn_final = _scatter_sum(weighted_final, cn_idx, N_cn)  # [N_cn]

        # denom = S_per_cn_final[cn_idx]    # [E]
        # p = torch.zeros_like(s_pos)
        # mask_denom = denom > 0
        # p[mask_denom] = (
        #     s_pos[mask_denom] * self.power_limit / denom[mask_denom]
        # )
        p = F.relu(s_logits)
        edge_attr_dict[edge_key] = torch.cat(
            [edge_attr[:, :self.edge_dim], p.view(-1, 1)],
            dim=1
        )

        return x_dict, edge_attr_dict, edge_index_dict

def _scatter_sum(src, index, dim_size):
    out = src.new_zeros((dim_size,) + src.shape[1:])
    out.index_add_(0, index, src)
    return out

def constraint_loss(p, edge_index, gamma, n_ap, power_limit=1.0, reduction='mean'):
    src = edge_index[0]
    weighted = p * gamma
    S_per_ap = _scatter_sum(weighted, src, dim_size=n_ap)
    violation = F.relu(S_per_ap - float(power_limit))
    loss = violation.pow(2)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def combined_loss(task_loss, p, edge_index, gamma, n_ap, alpha=1.0, power_limit=1.0):
    c_loss = constraint_loss(p, edge_index, gamma, n_ap, power_limit=power_limit, reduction='mean')
    if task_loss is None:
        return alpha * c_loss
    return task_loss + alpha * c_loss, c_loss


def loss_function(
        graphData,
        nodeFeatDict,   # x_dict
        edgeDict,       # edge_attr_dict
        edge_index_dict,
        tau, rho_p, rho_d, num_antenna,
        epochRatio=1,
        eval_mode=False,
        alpha=1.0,
        gamma_idx=1,
        power_limit=1.0
    ):
    num_graph = graphData.num_graphs

    # ----- đổi sang CN / VN -----
    # Nếu bạn vẫn để tên 'AP'/'UE' thì đổi CN->AP, VN->UE và key edge tương ứng.
    num_APs = graphData['CN'].x.shape[0] // num_graph   # CN ~ AP
    num_UEs = graphData['VN'].x.shape[0] // num_graph   # VN ~ UE

    edge_key = ('CN', 'to', 'VN')   # tương đương ('AP','down','UE') cũ

    # edge feature: [E, edge_dim_total] (đã có power ở cột cuối)
    edge_feat = edgeDict[edge_key]
    edge_dim_total = edge_feat.size(-1)

    # reshape về [num_graph, num_APs, num_UEs, edge_dim_total]
    edge_feat_4d = edge_feat.view(num_graph, num_APs, num_UEs, edge_dim_total)

    # large-scale fading log(1+beta) ở cột 0
    large_scale = edge_feat_4d[:, :, :, 0]
    large_scale = torch.expm1(large_scale)

    # phi_matrix từ VN (UE) features
    phi_matrix = graphData['VN'].x.view(num_graph, num_UEs, -1)

    # channel_var / gamma ở cột 1 (giống bạn dùng trước đây)
    channel_var = edge_feat_4d[:, :, :, 1]

    # power_matrix từ cột cuối cùng
    power_matrix = edge_feat_4d[:, :, :, -1]

    # ----- tính rate -----
    all_DS, all_PC, all_UI = component_calculate(
        power_matrix, channel_var, large_scale, phi_matrix, rho_d=rho_d
    )
    rate = rate_from_component(
        all_DS, all_PC, all_UI, num_antenna, rho_d=rho_d
    )

    p_flat = power_matrix.reshape(-1)                 
    gamma_flat = edge_feat[:, gamma_idx]              
    edge_index = edge_index_dict[edge_key]            
    n_ap_total = graphData['CN'].x.size(0)
    if torch.isnan(rate).any():
        print(power_matrix)
        raise ValueError('Nan in rate')

    # ----- Eval mode -----
    if eval_mode:
        min_rate, _ = torch.min(rate, dim=1)  # [num_graph]

        full = torch.ones_like(power_matrix)
        rate_full_one = rate_calculation(
            full, large_scale, channel_var, phi_matrix, rho_d, num_antenna
        )
        min_rate_one, _ = torch.min(rate_full_one, dim=1)
        c_loss = constraint_loss(p_flat, edge_index, gamma_flat, n_ap_total, power_limit=power_limit, reduction='mean')

        return min_rate, min_rate_one, c_loss

    # ----- Train mode -----
    epochRatio = min(1.0, epochRatio)

    min_rate_detach, _ = torch.min(rate.detach(), dim=1)
    min_rate, _ = torch.min(rate, dim=1)
    task_loss = torch.mean(-min_rate)

    

    total_loss, c_loss = combined_loss(
        task_loss,
        p_flat,
        edge_index,
        gamma_flat,
        n_ap_total,
        alpha=alpha,
        power_limit=power_limit
    )

    return total_loss, c_loss, torch.mean(min_rate_detach)

def train_model( epochRatio,
        dataLoader, model, optimizer,
        tau, rho_p, rho_d, num_antenna
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    
    total_loss = 0.0
    total_c_loss = 0.0
    total_graphs = 0
    for batch in dataLoader:
        optimizer.zero_grad(set_to_none=True) 
        batch = batch.to(device)
        num_graph = batch.num_graphs
        
        x_dict, edge_dict, edge_index = model(batch)
        loss, c_loss, _ = loss_function(
            batch, x_dict, edge_dict,edge_index,
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna,
            epochRatio=epochRatio
        )
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * num_graph
        total_c_loss += c_loss.item() * num_graph
        total_graphs += num_graph

    return total_loss/total_graphs, total_c_loss / total_graphs


@torch.no_grad()
def eval_model(
        dataLoader, model,
        tau, rho_p, rho_d, num_antenna
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    total_min_rate = 0.0
    total_c_loss = 0.0
    total_graphs = 0

    for batch in dataLoader:
        batch = batch.to(device)
        num_graph = batch.num_graphs
        
        x_dict, edge_dict, edge_index_dict = model(batch)

        # loss_function trả về (min_rate_batch, min_rate_full_one_batch)
        min_rate_batch, _, c_loss = loss_function(
            batch, x_dict, edge_dict, edge_index_dict,
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna,
            eval_mode=True
        )
        # min_rate_batch: shape [num_graph]

        # lấy mean theo batch
        batch_mean = min_rate_batch.mean().item()
        total_min_rate += batch_mean * num_graph
        total_c_loss += c_loss.item() * num_graph
        total_graphs   += num_graph

    return total_min_rate / total_graphs, total_c_loss / total_graphs
