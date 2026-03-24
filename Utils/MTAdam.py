import math
import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
from .centralized_train import component_calculate, rate_from_component, rate_calculation

import math
import torch
from torch.optim.optimizer import Optimizer


class MTAdam(Optimizer):
    r"""
    Multi-Task Adam (MTAdam)

    - Hỗ trợ nhiều loss cùng lúc: loss_array = [L1, L2, ...].
    - Mỗi loss có exp_avg / exp_avg_sq riêng như nhiều "đầu Adam".
    - Chuẩn hóa norm grad của từng loss theo một "anchor" (loss đầu tiên có norm > 0)
      với hệ số trơn beta3.
    - ranks: trọng số tương đối cho từng loss (ví dụ [1.0, beta_constraint]).

    Cách dùng:
        optimizer = MTAdam(model.parameters(), lr=1e-3, betas=(0.9, 0.999, 0.9))
        optimizer.step(loss_array=[task_loss, c_loss], ranks=[1.0, beta_c])
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999, 0.9),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta3: {betas[2]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

        self.training_step = 0

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, loss_array, ranks=None, closure=None):
        """
        loss_array: list/tuple các scalar loss (tensor).
        ranks: list/tuple các trọng số (float hoặc tensor) cùng chiều với loss_array.
        """
        if closure is not None:
            with torch.enable_grad():
                closure()

        if not isinstance(loss_array, (list, tuple)):
            raise ValueError("loss_array must be a list/tuple of losses")

        num_losses = len(loss_array)
        if num_losses == 0:
            return None

        if ranks is None:
            ranks = [1.0] * num_losses
        if len(ranks) != num_losses:
            raise ValueError("ranks must have same length as loss_array")

        # Convert ranks to tensors (trên CPU hoặc GPU như loss)
        rank_tensors = []
        for loss, r in zip(loss_array, ranks):
            if isinstance(r, torch.Tensor):
                rank_tensors.append(r.to(loss.device))
            else:
                rank_tensors.append(torch.tensor(float(r), device=loss.device))

        self._update_weights(loss_array, rank_tensors)
        self.training_step += 1
        return None

    def _update_weights(self, loss_array, ranks):
        num_losses = len(loss_array)

        # ----- 1. Backward từng loss + cập nhật state riêng cho từng loss -----
        for loss_index, (loss, rank) in enumerate(zip(loss_array, ranks)):
            # backward cho loss này, giữ graph cho các loss sau
            loss.backward(retain_graph=True)

            for group in self.param_groups:
                beta1, beta2, beta3 = group["betas"]
                lr = group["lr"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                amsgrad = group["amsgrad"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad

                    if grad.is_sparse:
                        raise RuntimeError("MTAdam does not support sparse gradients")

                    state = self.state[p]

                    # Khởi tạo state lần đầu
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg_list"] = []
                        state["exp_avg_sq_list"] = []
                        if amsgrad:
                            state["max_exp_avg_sq_list"] = []
                        state["norms"] = []

                        # khởi tạo list cho tất cả loss
                        for _ in range(len(loss_array)):
                            state["exp_avg_list"].append(torch.zeros_like(p))
                            state["exp_avg_sq_list"].append(torch.zeros_like(p))
                            if amsgrad:
                                state["max_exp_avg_sq_list"].append(torch.zeros_like(p))
                            state["norms"].append(torch.zeros(1, device=p.device))

                    exp_avg_list = state["exp_avg_list"]
                    exp_avg_sq_list = state["exp_avg_sq_list"]
                    norms = state["norms"]
                    if amsgrad:
                        max_exp_avg_sq_list = state["max_exp_avg_sq_list"]

                    # ---- chuẩn hóa norm gradient cho loss_index này ----
                    g_norm = grad.norm()
                    if state["step"] == 0:
                        norms[loss_index] = g_norm
                    else:
                        norms[loss_index] = beta3 * norms[loss_index] + (1 - beta3) * g_norm

                    # anchor = loss đầu tiên có norm > 1e-10
                    anchor_norm = None
                    for n in norms:
                        if n > 1e-10:
                            anchor_norm = n
                            break

                    if anchor_norm is not None and norms[loss_index] > 1e-10:
                        # scale grad để norm của loss này ≈ anchor_norm, nhân thêm rank
                        grad = grad * (anchor_norm * rank / norms[loss_index])

                    # ---- Adam update riêng cho loss_index ----
                    if weight_decay != 0.0:
                        grad = grad.add(p, alpha=weight_decay)

                    exp_avg = exp_avg_list[loss_index]
                    exp_avg_sq = exp_avg_sq_list[loss_index]

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    if amsgrad:
                        max_exp_avg_sq = max_exp_avg_sq_list[loss_index]
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        max_exp_avg_sq_list[loss_index] = max_exp_avg_sq

                    # lưu lại
                    exp_avg_list[loss_index] = exp_avg
                    exp_avg_sq_list[loss_index] = exp_avg_sq

                    # xoá grad cho loss này, để loss sau dùng lại p.grad sạch
                    p.grad.detach_()
                    p.grad.zero_()

            # end for group
        # end for each loss

        # sau khi xử lý hết các loss → tăng step
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "step" in state:
                    state["step"] += 1

        # ----- 2. Kết hợp update từ tất cả loss và áp dụng lên parameters -----
        for group in self.param_groups:
            beta1, beta2, _ = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            amsgrad = group["amsgrad"]

            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    continue

                step = state["step"]
                exp_avg_list = state["exp_avg_list"]
                exp_avg_sq_list = state["exp_avg_sq_list"]
                if amsgrad:
                    max_exp_avg_sq_list = state["max_exp_avg_sq_list"]

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                if bias_correction1 < 1e-8:  # tránh chia 0
                    bias_correction1 = 1e-8

                # tính denom (hoặc max_denom) cho tất cả loss
                denom_list = []
                for i in range(len(exp_avg_list)):
                    if amsgrad:
                        max_exp_avg_sq = max_exp_avg_sq_list[i]
                        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    else:
                        exp_avg_sq = exp_avg_sq_list[i]
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    denom_list.append(denom)

                # dùng max denom để chuẩn hóa (như code gốc)
                max_denom = denom_list[0]
                for d in denom_list[1:]:
                    max_denom = torch.max(max_denom, d)

                step_size = lr / bias_correction1

                # tổng hợp update từ tất cả loss
                total_update = torch.zeros_like(p)
                for i in range(len(exp_avg_list)):
                    exp_avg = exp_avg_list[i]
                    update_i = -step_size * (exp_avg / max_denom)
                    total_update.add_(update_i)

                p.add_(total_update)

def _scatter_sum(src, index, dim_size):
    out = src.new_zeros((dim_size,) + src.shape[1:])
    out.index_add_(0, index, src)
    return out


def constraint_loss(p, edge_index, gamma, n_ap, power_limit=1.0, reduction='mean', alpha=1e6):
    src = edge_index[0]
    weighted = p * gamma
    S_per_ap = _scatter_sum(weighted, src, dim_size=n_ap)
    violation = alpha*F.relu(S_per_ap - float(power_limit))
    # print(violation)
    # print(violation)
    loss = violation
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
    
def rate_constraint_loss(
    rate,                  # [G, K]
    target_rate=0.1,       # bps/Hz, tuỳ bạn chọn
    reduction='mean',
    mode='min'             # 'min' hoặc 'per_user'
):
    if mode == 'min':
        # min-rate mỗi graph
        min_rate, _ = torch.min(rate, dim=1)     # [G]
        violation = F.relu(target_rate - min_rate)  # [G], chỉ phạt nếu < target
        loss = violation.pow(2)
    else:
        # phạt từng user
        violation = F.relu(target_rate - rate)   # [G, K]
        loss = violation.pow(2)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def combined_loss(task_loss, p, edge_index, gamma, n_ap, alpha=1e3, power_limit=1.0):
    c_loss = constraint_loss(p, edge_index, gamma, n_ap, power_limit=power_limit, reduction='mean')
    if task_loss is None:
        return alpha * c_loss
    return task_loss + alpha * c_loss, c_loss


def ldpc_loss_mtadam(
        graphData,
        nodeFeatDict,   # x_dict
        edgeDict,       # edge_attr_dict
        edge_index_dict,
        tau, rho_p, rho_d, num_antenna,
        epochRatio=1,
        eval_mode=False,
        gamma_idx=1,
        power_limit=1.0
    ):
    num_graph = graphData.num_graphs

    # CN ~ AP, VN ~ UE
    num_APs = graphData['CN'].x.shape[0] // num_graph
    num_UEs = graphData['VN'].x.shape[0] // num_graph

    edge_key = ('CN', 'to', 'VN')

    # edge feature: [E, edge_dim_total] (cột cuối là power do GNN sinh ra)
    edge_feat = edgeDict[edge_key]
    edge_dim_total = edge_feat.size(-1)

    # reshape: [G, M, K, F]
    edge_feat_4d = edge_feat.view(num_graph, num_APs, num_UEs, edge_dim_total)

    # large-scale log(1+beta) ở cột 0
    large_scale = edge_feat_4d[:, :, :, 0]
    large_scale = torch.expm1(large_scale)

    # phi_matrix từ VN features
    phi_matrix  = graphData['VN'].x.view(num_graph, num_UEs, -1)

    # channel_var / gamma ở cột 1
    channel_var = edge_feat_4d[:, :, :, 1]

    # power_matrix từ cột cuối
    power_matrix = edge_feat_4d[:, :, :, -1]

    # ===== Tính rate =====
    all_DS, all_PC, all_UI = component_calculate(
        power_matrix, channel_var, large_scale, phi_matrix, rho_d=rho_d
    )
    rate = rate_from_component(
        all_DS, all_PC, all_UI, num_antenna, rho_d=rho_d
    )

    if torch.isnan(rate).any():
        print(power_matrix)
        raise ValueError('Nan in rate')

    # Flatten cho constraint loss
    p_flat     = power_matrix.reshape(-1)             # [E]
    gamma_flat = edge_feat[:, gamma_idx]/rho_d              # [E]
    edge_index = edge_index_dict[edge_key]            # [2, E]
    n_ap_total = graphData['CN'].x.size(0)

    # ===== Eval mode =====
    if eval_mode:
        min_rate, _ = torch.min(rate, dim=1)  # [num_graph]

        full = torch.ones_like(power_matrix)
        rate_full_one = rate_calculation(
            full, large_scale, channel_var, phi_matrix, rho_d, num_antenna
        )
        min_rate_one, _ = torch.min(rate_full_one, dim=1)

        c_loss = constraint_loss(
            p_flat, edge_index, gamma_flat, n_ap_total,
            power_limit=power_limit, reduction='mean'
        )

        return min_rate, min_rate_one, c_loss

    # ===== Train mode =====
    epochRatio = min(1.0, epochRatio)

    min_rate_detach, _ = torch.min(rate.detach(), dim=1)   # [num_graph]
    min_rate, _ = torch.min(rate, dim=1)                   # [num_graph]

    # Task loss: max–min rate
    task_loss = torch.mean(-min_rate)

    # Constraint loss: phạt vi phạm ∑ p_{mk} γ_{mk} > power_limit
    c_loss = constraint_loss(
        p_flat, edge_index, gamma_flat, n_ap_total,
        power_limit=power_limit, reduction='mean'
    )

    return task_loss, c_loss, torch.mean(min_rate_detach)


def ldpc_train_mtadam(
        epochRatio,
        dataLoader, model, optimizer: MTAdam,
        tau, rho_p, rho_d, num_antenna,
        alpha=1.0,
        gamma_idx=1,
        power_limit=1.0,
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    
    total_task_loss = 0.0
    total_c_loss    = 0.0
    total_graphs    = 0

    for batch in dataLoader:
        optimizer.zero_grad(set_to_none=True)
        batch = batch.to(device)
        num_graph = batch.num_graphs
        
        x_dict, edge_dict, edge_index_dict = model(batch)

        task_loss, c_loss, _ = ldpc_loss_mtadam(
            batch, x_dict, edge_dict, edge_index_dict,
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna,
            epochRatio=epochRatio,
            gamma_idx=gamma_idx,
            power_limit=power_limit,
        )

        # multi-loss update
        optimizer.step(
            loss_array=[task_loss, c_loss],
            ranks=[1.0, alpha]     
        )
        
        total_task_loss += task_loss.item() * num_graph
        total_c_loss    += c_loss.item() * num_graph
        total_graphs    += num_graph

    return (
        total_task_loss / total_graphs,
        total_c_loss   / total_graphs,
    )


@torch.no_grad()
def ldpc_eval_mtadam(
        dataLoader, model,
        tau, rho_p, rho_d, num_antenna,
        gamma_idx=1,
        power_limit=1.0,
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    total_min_rate = 0.0
    total_c_loss   = 0.0
    total_graphs   = 0

    for batch in dataLoader:
        batch = batch.to(device)
        num_graph = batch.num_graphs
        
        x_dict, edge_dict, edge_index_dict = model(batch)

        # ldpc_loss_mtadam trả về (min_rate_batch, min_rate_one, c_loss) ở eval_mode
        min_rate_batch, _, c_loss = ldpc_loss_mtadam(
            batch, x_dict, edge_dict, edge_index_dict,
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna,
            eval_mode=True,
            gamma_idx=gamma_idx,
            power_limit=power_limit,
        )

        # min_rate_batch: [num_graph]
        batch_mean_rate = min_rate_batch.mean().item()
        total_min_rate += batch_mean_rate * num_graph
        total_c_loss   += c_loss.item() * num_graph
        total_graphs   += num_graph

    return (
        total_min_rate / total_graphs,   # avg min-rate
        total_c_loss   / total_graphs,   # avg constraint loss
    )


def beta3_schedule(epoch, num_epochs, beta3_start=0.1, beta3_end=0.9):
    t = epoch / max(1, num_epochs - 1)
    beta3 = beta3_start + 0.5 * (beta3_end - beta3_start) * (1 - math.cos(math.pi * t))
    return float(beta3)

def update_alpha(alpha_old, c_loss_epoch,
                 c_high=1e-2, c_low=1e-4,
                 up_factor=1.05, down_factor=0.9,
                 alpha_min=0.1, alpha_max=20.0):
    alpha = alpha_old
    if c_loss_epoch > c_high:
        alpha = min(alpha * up_factor, alpha_max)
    elif c_loss_epoch < c_low:
        alpha = max(alpha * down_factor, alpha_min)
    return float(alpha)


def rate_constraint_loss_sum(
    rate,
    target_rate=0.1,
    reduction='mean',
    mode='per_user',   # 'per_user' hoặc 'min'
):
    """
    rate: [G, K]  (G graph, K UE)
    mode='per_user': phạt từng UE có R < target_rate
    mode='min': phạt theo min_k R_{g,k} cho mỗi graph
    """
    if mode == 'per_user':
        violation = F.relu(target_rate - rate)   # [G,K]
        loss = violation.pow(2)
    elif mode == 'min':
        min_rate, _ = torch.min(rate, dim=1)     # [G]
        violation = F.relu(target_rate - min_rate)
        loss = violation.pow(2)
    else:
        raise ValueError(f"Unknown rate constraint mode: {mode}")

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
    


def ldpc_loss_mtadam_sum(
        graphData,
        nodeFeatDict,   # x_dict
        edgeDict,       # edge_attr_dict
        edge_index_dict,
        tau, rho_p, rho_d, num_antenna,
        epochRatio=1,
        eval_mode=False,
        gamma_idx=1,
        power_limit=1.0,
        target_rate=0.1,
        rate_mode='per_user',   # hoặc 'min'
    ):
    """
    Task: maximize *sum rate* per graph.
    Train mode  -> trả về: task_loss, power_c_loss, rate_c_loss, mean_sum_rate_detach
    Eval mode   -> trả về: sum_rate_per_graph, power_c_loss, rate_c_loss
    """
    num_graph = graphData.num_graphs
    num_APs   = graphData['CN'].x.shape[0] // num_graph
    num_UEs   = graphData['VN'].x.shape[0] // num_graph

    edge_key = ('CN', 'to', 'VN')
    edge_feat      = edgeDict[edge_key]      # [E, F]
    edge_dim_total = edge_feat.size(-1)

    edge_feat_4d = edge_feat.view(num_graph, num_APs, num_UEs, edge_dim_total)

    # large-scale, phi, channel_var, power_matrix
    large_scale  = torch.expm1(edge_feat_4d[:, :, :, 0])
    phi_matrix   = graphData['VN'].x.view(num_graph, num_UEs, -1)
    channel_var  = edge_feat_4d[:, :, :, 1]
    power_matrix = edge_feat_4d[:, :, :, -1]

    # ===== tính rate =====
    all_DS, all_PC, all_UI = component_calculate(
        power_matrix, channel_var, large_scale, phi_matrix, rho_d=rho_d
    )
    rate = rate_from_component(
        all_DS, all_PC, all_UI, num_antenna, rho_d=rho_d
    )   # [G, K]

    if torch.isnan(rate).any():
        print(power_matrix)
        raise ValueError("NaN in rate")

    # ===== power-constraint loss =====
    p_flat     = power_matrix.reshape(-1)
    gamma_flat = edge_feat[:, gamma_idx] / rho_d
    edge_index = edge_index_dict[edge_key]
    n_ap_total = graphData['CN'].x.size(0)

    power_c_loss = constraint_loss(
        p_flat,
        edge_index,
        gamma_flat,
        n_ap_total,
        power_limit=power_limit,
        reduction='mean'
    )

    # ===== rate-constraint loss =====
    rate_c_loss = rate_constraint_loss_sum(
        rate,
        target_rate=target_rate,
        reduction='mean',
        mode=rate_mode,
    )

    if eval_mode:
        # sum-rate mỗi graph
        sum_rate_per_graph = rate.sum(dim=1)   # [G]
        return sum_rate_per_graph, power_c_loss, rate_c_loss

    # ===== train mode =====
    epochRatio = min(1.0, epochRatio)

    sum_rate_detach = rate.detach().sum(dim=1)  # [G]
    sum_rate        = rate.sum(dim=1)           # [G]

    # Task loss: maximize sum-rate
    task_loss = torch.mean(-sum_rate)

    return task_loss, power_c_loss, rate_c_loss, torch.mean(sum_rate_detach)

def ldpc_train_mtadam_sum(
        epochRatio,
        dataLoader,
        model,
        optimizer,
        tau, rho_p, rho_d, num_antenna,
        alpha_power,      # weight cho power constraint
        alpha_rate,       # weight cho rate constraint
        beta3,
        gamma_idx=1,
        power_limit=1.0,
        target_rate=0.1,
        rate_mode='per_user',
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    
    total_task_loss = 0.0
    total_p_c_loss  = 0.0
    total_r_c_loss  = 0.0
    total_graphs    = 0

    # set beta3 cho optimizer (nếu muốn set ở đây)
    for group in optimizer.param_groups:
        b1, b2, _ = group["betas"]
        group["betas"] = (b1, b2, beta3)

    for batch in dataLoader:
        optimizer.zero_grad(set_to_none=True)
        batch = batch.to(device)
        num_graph = batch.num_graphs
        
        x_dict, edge_dict, edge_index_dict = model(batch)

        task_loss, p_c_loss, r_c_loss, _ = ldpc_loss_mtadam_sum(
            graphData       = batch,
            nodeFeatDict    = x_dict,
            edgeDict        = edge_dict,
            edge_index_dict = edge_index_dict,
            tau             = tau,
            rho_p           = rho_p,
            rho_d           = rho_d,
            num_antenna     = num_antenna,
            epochRatio      = epochRatio,
            eval_mode       = False,
            gamma_idx       = gamma_idx,
            power_limit     = power_limit,
            target_rate     = target_rate,
            rate_mode       = rate_mode,
        )

        # Multi-task Adam: 3 task
        optimizer.step(
            loss_array=[task_loss, p_c_loss, r_c_loss],
            ranks=[1.0, alpha_power, alpha_rate],
        )

        total_task_loss += task_loss.item() * num_graph
        total_p_c_loss  += p_c_loss.item()  * num_graph
        total_r_c_loss  += r_c_loss.item()  * num_graph
        total_graphs    += num_graph

    return (
        total_task_loss / total_graphs,
        total_p_c_loss  / total_graphs,
        total_r_c_loss  / total_graphs,
    )

@torch.no_grad()
def ldpc_eval_mtadam_sum(
        dataLoader,
        model,
        tau, rho_p, rho_d, num_antenna,
        gamma_idx=1,
        power_limit=1.0,
        target_rate=0.1,
        rate_mode='per_user',
    ):
    """
    Eval (sum-rate task):
      - mean_sum_rate      : trung bình sum-rate per graph
      - mean_power_c_loss  : power constraint loss trung bình
      - mean_rate_c_loss   : rate constraint loss trung bình
      - frac_rate_viol     : tỉ lệ vi phạm rate (theo mode)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    total_sum_rate     = 0.0
    total_power_c_loss = 0.0
    total_rate_c_loss  = 0.0
    total_graphs       = 0

    total_violated = 0   # số UE hoặc graph vi phạm
    total_units    = 0   # tổng UE hoặc graph

    for batch in dataLoader:
        batch = batch.to(device)
        num_graph = batch.num_graphs

        x_dict, edge_dict, edge_index_dict = model(batch)

        sum_rate_batch, power_c_loss_batch, rate_c_loss_batch = ldpc_loss_mtadam_sum(
            graphData       = batch,
            nodeFeatDict    = x_dict,
            edgeDict        = edge_dict,
            edge_index_dict = edge_index_dict,
            tau             = tau,
            rho_p           = rho_p,
            rho_d           = rho_d,
            num_antenna     = num_antenna,
            epochRatio      = 1.0,
            eval_mode       = True,
            gamma_idx       = gamma_idx,
            power_limit     = power_limit,
            target_rate     = target_rate,
            rate_mode       = rate_mode,
        )
        # sum_rate_batch: [G]

        total_sum_rate     += sum_rate_batch.sum().item()
        total_power_c_loss += power_c_loss_batch.item() * num_graph
        total_rate_c_loss  += rate_c_loss_batch.item()  * num_graph
        total_graphs       += num_graph

        # để đo frac_rate_viol, cần lại full rate:
        num_APs   = batch['CN'].x.shape[0] // num_graph
        num_UEs   = batch['VN'].x.shape[0] // num_graph
        edge_key  = ('CN', 'to', 'VN')
        edge_feat = edge_dict[edge_key]
        edge_dim_total = edge_feat.size(-1)
        edge_feat_4d   = edge_feat.view(num_graph, num_APs, num_UEs, edge_dim_total)

        large_scale  = torch.expm1(edge_feat_4d[:, :, :, 0])
        phi_matrix   = batch['VN'].x.view(num_graph, num_UEs, -1)
        channel_var  = edge_feat_4d[:, :, :, 1]
        power_matrix = edge_feat_4d[:, :, :, -1]

        all_DS, all_PC, all_UI = component_calculate(
            power_matrix, channel_var, large_scale, phi_matrix, rho_d=rho_d
        )
        rate = rate_from_component(
            all_DS, all_PC, all_UI, num_antenna, rho_d=rho_d
        )   # [G,K]

        if rate_mode == 'per_user':
            viol_mask = rate < target_rate   # [G,K]
            total_violated += viol_mask.sum().item()
            total_units    += rate.numel()
        else:  # 'min'
            min_rate, _ = torch.min(rate, dim=1)  # [G]
            viol_mask = min_rate < target_rate
            total_violated += viol_mask.sum().item()
            total_units    += num_graph

    if total_graphs == 0:
        return 0.0, 0.0, 0.0, 0.0

    mean_sum_rate     = total_sum_rate     / total_graphs
    mean_power_c_loss = total_power_c_loss / total_graphs
    mean_rate_c_loss  = total_rate_c_loss  / total_graphs
    frac_rate_viol    = (
        total_violated / total_units if total_units > 0 else 0.0
    )

    return mean_sum_rate, mean_power_c_loss, mean_rate_c_loss, frac_rate_viol
