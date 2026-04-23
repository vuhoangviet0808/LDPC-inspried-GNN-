import math
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from .centralized_train import component_calculate, rate_from_component, rate_calculation


# =========================================================
# MTAdam
# =========================================================
class MTAdam(Optimizer):
    """
    Multi-Task Adam (MTAdam)

    - Hỗ trợ nhiều loss cùng lúc: loss_array = [L1, L2, ...]
    - Mỗi loss có exp_avg / exp_avg_sq riêng
    - Chuẩn hóa norm grad của từng loss theo anchor với beta3
    - ranks: trọng số tương đối cho từng loss
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
        for loss_index, (loss, rank) in enumerate(zip(loss_array, ranks)):
            loss.backward(retain_graph=True)

            for group in self.param_groups:
                beta1, beta2, beta3 = group["betas"]
                weight_decay = group["weight_decay"]
                amsgrad = group["amsgrad"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad

                    if grad.is_sparse:
                        raise RuntimeError("MTAdam does not support sparse gradients")

                    state = self.state[p]

                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg_list"] = []
                        state["exp_avg_sq_list"] = []
                        if amsgrad:
                            state["max_exp_avg_sq_list"] = []
                        state["norms"] = []

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

                    g_norm = grad.norm()
                    if state["step"] == 0:
                        norms[loss_index] = g_norm
                    else:
                        norms[loss_index] = beta3 * norms[loss_index] + (1.0 - beta3) * g_norm

                    anchor_norm = None
                    for n in norms:
                        if n > 1e-10:
                            anchor_norm = n
                            break

                    if anchor_norm is not None and norms[loss_index] > 1e-10:
                        grad = grad * (anchor_norm * rank / norms[loss_index])

                    if weight_decay != 0.0:
                        grad = grad.add(p, alpha=weight_decay)

                    exp_avg = exp_avg_list[loss_index]
                    exp_avg_sq = exp_avg_sq_list[loss_index]

                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    if amsgrad:
                        max_exp_avg_sq = max_exp_avg_sq_list[loss_index]
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        max_exp_avg_sq_list[loss_index] = max_exp_avg_sq

                    exp_avg_list[loss_index] = exp_avg
                    exp_avg_sq_list[loss_index] = exp_avg_sq

                    p.grad.detach_()
                    p.grad.zero_()

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "step" in state:
                    state["step"] += 1

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

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                if bias_correction1 < 1e-8:
                    bias_correction1 = 1e-8

                denom_list = []
                for i in range(len(exp_avg_list)):
                    if amsgrad:
                        max_exp_avg_sq = max_exp_avg_sq_list[i]
                        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    else:
                        exp_avg_sq = exp_avg_sq_list[i]
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    denom_list.append(denom)

                max_denom = denom_list[0]
                for d in denom_list[1:]:
                    max_denom = torch.max(max_denom, d)

                step_size = lr / bias_correction1

                total_update = torch.zeros_like(p)
                for i in range(len(exp_avg_list)):
                    exp_avg = exp_avg_list[i]
                    update_i = -step_size * (exp_avg / max_denom)
                    total_update.add_(update_i)

                p.add_(total_update)


# =========================================================
# Utils
# =========================================================
def _scatter_sum(src, index, dim_size):
    out = src.new_zeros((dim_size,) + src.shape[1:])
    out.index_add_(0, index, src)
    return out


def beta3_schedule(epoch, num_epochs, beta3_start=0.1, beta3_end=0.9):
    t = epoch / max(1, num_epochs - 1)
    beta3 = beta3_start + 0.5 * (beta3_end - beta3_start) * (1.0 - math.cos(math.pi * t))
    return float(beta3)


def update_alpha(
    alpha_old,
    c_loss_epoch,
    c_high=1e-2,
    c_low=1e-4,
    up_factor=1.05,
    down_factor=0.9,
    alpha_min=1e-7,
    alpha_max=1e-3,
):
    alpha = alpha_old
    if c_loss_epoch > c_high:
        alpha = min(alpha * up_factor, alpha_max)
    elif c_loss_epoch < c_low:
        alpha = max(alpha * down_factor, alpha_min)
    return float(alpha)


# =========================================================
# Constraint / regularization losses
# =========================================================
def constraint_loss_usercentric(
    p_eff,                # [E]
    edge_index,           # [2, E]
    gamma,                # [E]
    n_ap,
    power_limit=1.0,
    reduction='mean',
    alpha=1e6,
):
    src = edge_index[0]
    weighted = p_eff * gamma
    S_per_ap = _scatter_sum(weighted, src, dim_size=n_ap)
    violation = alpha * F.relu(S_per_ap - float(power_limit))
    loss = violation

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def rate_constraint_loss_usercentric(
    rate,                 # [G, K]
    target_rate=1.0,
    reduction='mean',
    mode='per_user',      # 'per_user' or 'min'
):
    if mode == 'per_user':
        violation = F.relu(target_rate - rate)
        loss = violation.pow(2)
    elif mode == 'min':
        min_rate, _ = torch.min(rate, dim=1)
        violation = F.relu(target_rate - min_rate)
        loss = violation.pow(2)
    else:
        raise ValueError(f"Unknown rate constraint mode: {mode}")

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def association_sparsity_loss(a_soft, reduction='mean'):
    """
    a_soft: [G, M, K]
    """
    loss = a_soft
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def association_cardinality_loss(
    a_soft,               # [G, M, K]
    target_ap_per_ue=4,
    reduction='mean',
    mode='leq',           # 'leq' or 'eq'
):
    """
    Số AP phục vụ mỗi UE.
    """
    ap_per_ue = a_soft.sum(dim=1)   # [G, K]

    if mode == 'leq':
        violation = F.relu(ap_per_ue - float(target_ap_per_ue))
        loss = violation.pow(2)
    elif mode == 'eq':
        loss = (ap_per_ue - float(target_ap_per_ue)).pow(2)
    else:
        raise ValueError(f"Unknown cardinality mode: {mode}")

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def association_ap_load_loss(
    a_soft,               # [G, M, K]
    max_ue_per_ap=4,
    reduction='mean',
    mode='leq',           # 'leq' or 'eq'
):
    """
    Số UE mà mỗi AP phục vụ.
    """
    ue_per_ap = a_soft.sum(dim=2)   # [G, M]

    if mode == 'leq':
        violation = F.relu(ue_per_ap - float(max_ue_per_ap))
        loss = violation.pow(2)
    elif mode == 'eq':
        loss = (ue_per_ap - float(max_ue_per_ap)).pow(2)
    else:
        raise ValueError(f"Unknown AP-load mode: {mode}")

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def association_cover_loss(
    a_soft,               # [G, M, K]
    min_ap_per_ue=1.0,
    reduction='mean',
):
    """
    Mỗi UE phải được ít nhất min_ap_per_ue AP phục vụ.
    """
    ap_per_ue = a_soft.sum(dim=1)   # [G, K]
    violation = F.relu(float(min_ap_per_ue) - ap_per_ue)
    loss = violation.pow(2)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


# =========================================================
# Edge output parser
# =========================================================
def extract_usercentric_matrices(
    graphData,
    edgeDict,
    edge_base_dim,
    edge_key=('CN', 'to', 'VN'),
):
    """
    Model output edge format:
        [base_edge_feat, a_soft, p_eff]

    Return:
        large_scale  : [G, M, K]
        channel_var  : [G, M, K]
        assoc_matrix : [G, M, K]
        power_matrix : [G, M, K]
        phi_matrix   : [G, K, d]
        edge_feat    : [E, F]
        edge_feat_4d : [G, M, K, F]
    """
    num_graph = graphData.num_graphs
    num_APs = graphData['CN'].x.shape[0] // num_graph
    num_UEs = graphData['VN'].x.shape[0] // num_graph

    edge_feat = edgeDict[edge_key]
    edge_dim_total = edge_feat.size(-1)

    edge_feat_4d = edge_feat.view(num_graph, num_APs, num_UEs, edge_dim_total)

    large_scale = torch.expm1(edge_feat_4d[:, :, :, 0])
    channel_var = edge_feat_4d[:, :, :, 1]
    assoc_matrix = edge_feat_4d[:, :, :, edge_base_dim]
    power_matrix = edge_feat_4d[:, :, :, edge_base_dim + 1]
    phi_matrix = graphData['VN'].x.view(num_graph, num_UEs, -1)

    return (
        large_scale,
        channel_var,
        assoc_matrix,
        power_matrix,
        phi_matrix,
        edge_feat,
        edge_feat_4d,
    )


# =========================================================
# Main loss for user-centric max-sum-rate
# =========================================================
def ldpc_loss_mtadam_sum_usercentric(
    graphData,
    nodeFeatDict,
    edgeDict,
    edge_index_dict,
    tau,
    rho_p,
    rho_d,
    num_antenna,
    edge_base_dim,
    epochRatio=1.0,
    eval_mode=False,
    gamma_idx=1,
    power_limit=1.0,
    target_rate=1.0,
    rate_mode='per_user',
    target_ap_per_ue=4,
    assoc_card_mode='leq',
    max_ue_per_ap=4,
    ap_load_mode='leq',
    min_ap_per_ue=1.0,
):
    edge_key = ('CN', 'to', 'VN')

    (
        large_scale,
        channel_var,
        assoc_matrix,
        power_matrix,
        phi_matrix,
        edge_feat,
        edge_feat_4d,
    ) = extract_usercentric_matrices(
        graphData=graphData,
        edgeDict=edgeDict,
        edge_base_dim=edge_base_dim,
        edge_key=edge_key,
    )

    # ===== Rate =====
    all_DS, all_PC, all_UI = component_calculate(
        power_matrix, channel_var, large_scale, phi_matrix, rho_d=rho_d
    )
    rate = rate_from_component(
        all_DS, all_PC, all_UI, num_antenna, rho_d=rho_d
    )  # [G, K]

    if torch.isnan(rate).any():
        print(power_matrix)
        raise ValueError("NaN in rate")

    # ===== Power constraint =====
    p_flat = power_matrix.reshape(-1)
    gamma_flat = edge_feat[:, gamma_idx] / rho_d
    edge_index = edge_index_dict[edge_key]
    n_ap_total = graphData['CN'].x.size(0)

    power_c_loss = constraint_loss_usercentric(
        p_eff=p_flat,
        edge_index=edge_index,
        gamma=gamma_flat,
        n_ap=n_ap_total,
        power_limit=power_limit,
        reduction='mean',
    )

    # ===== QoS / rate =====
    rate_c_loss = rate_constraint_loss_usercentric(
        rate=rate,
        target_rate=target_rate,
        reduction='mean',
        mode=rate_mode,
    )

    # ===== Association regularizers / constraints =====
    assoc_sparse_loss = association_sparsity_loss(
        assoc_matrix,
        reduction='mean',
    )

    assoc_card_loss = association_cardinality_loss(
        assoc_matrix,
        target_ap_per_ue=target_ap_per_ue,
        reduction='mean',
        mode=assoc_card_mode,
    )

    assoc_ap_load_loss = association_ap_load_loss(
        assoc_matrix,
        max_ue_per_ap=max_ue_per_ap,
        reduction='mean',
        mode=ap_load_mode,
    )

    assoc_cover_loss = association_cover_loss(
        assoc_matrix,
        min_ap_per_ue=min_ap_per_ue,
        reduction='mean',
    )

    if eval_mode:
        sum_rate_per_graph = rate.sum(dim=1)
        min_rate_per_graph, _ = torch.min(rate, dim=1)

        return (
            sum_rate_per_graph,
            min_rate_per_graph,
            power_c_loss,
            rate_c_loss,
            assoc_sparse_loss,
            assoc_card_loss,
            assoc_ap_load_loss,
            assoc_cover_loss,
        )

    epochRatio = min(1.0, epochRatio)

    sum_rate = rate.sum(dim=1)
    task_loss = torch.mean(-sum_rate)
    mean_sum_rate_detach = torch.mean(sum_rate.detach())

    return (
        task_loss,
        power_c_loss,
        rate_c_loss,
        assoc_sparse_loss,
        assoc_card_loss,
        assoc_ap_load_loss,
        assoc_cover_loss,
        mean_sum_rate_detach,
    )


# =========================================================
# Train
# =========================================================
def ldpc_train_mtadam_sum_usercentric(
    epochRatio,
    dataLoader,
    model,
    optimizer: MTAdam,
    tau,
    rho_p,
    rho_d,
    num_antenna,
    edge_base_dim,
    alpha_power=1.0,
    alpha_rate=1.0,
    alpha_assoc_sparse=1e-3,
    alpha_assoc_card=1e-2,
    alpha_assoc_ap=1e-2,
    alpha_assoc_cover=1e-2,
    beta3=0.9,
    gamma_idx=1,
    power_limit=1.0,
    target_rate=1.0,
    rate_mode='per_user',
    target_ap_per_ue=4,
    assoc_card_mode='leq',
    max_ue_per_ap=4,
    ap_load_mode='leq',
    min_ap_per_ue=1.0,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()

    for group in optimizer.param_groups:
        b1, b2, _ = group["betas"]
        group["betas"] = (b1, b2, beta3)

    total_task_loss = 0.0
    total_p_c_loss = 0.0
    total_r_c_loss = 0.0
    total_a_s_loss = 0.0
    total_a_c_loss = 0.0
    total_a_ap_loss = 0.0
    total_a_cov_loss = 0.0
    total_graphs = 0

    for batch in dataLoader:
        optimizer.zero_grad(set_to_none=True)
        batch = batch.to(device)
        num_graph = batch.num_graphs

        x_dict, edge_dict, edge_index_dict = model(batch)

        (
            task_loss,
            p_c_loss,
            r_c_loss,
            a_s_loss,
            a_c_loss,
            a_ap_loss,
            a_cov_loss,
            _,
        ) = ldpc_loss_mtadam_sum_usercentric(
            graphData=batch,
            nodeFeatDict=x_dict,
            edgeDict=edge_dict,
            edge_index_dict=edge_index_dict,
            tau=tau,
            rho_p=rho_p,
            rho_d=rho_d,
            num_antenna=num_antenna,
            edge_base_dim=edge_base_dim,
            epochRatio=epochRatio,
            eval_mode=False,
            gamma_idx=gamma_idx,
            power_limit=power_limit,
            target_rate=target_rate,
            rate_mode=rate_mode,
            target_ap_per_ue=target_ap_per_ue,
            assoc_card_mode=assoc_card_mode,
            max_ue_per_ap=max_ue_per_ap,
            ap_load_mode=ap_load_mode,
            min_ap_per_ue=min_ap_per_ue,
        )

        optimizer.step(
            loss_array=[task_loss, p_c_loss, r_c_loss, a_s_loss, a_c_loss, a_ap_loss, a_cov_loss],
            ranks=[1.0, alpha_power, alpha_rate, alpha_assoc_sparse, alpha_assoc_card, alpha_assoc_ap, alpha_assoc_cover],
        )

        total_task_loss += task_loss.item() * num_graph
        total_p_c_loss += p_c_loss.item() * num_graph
        total_r_c_loss += r_c_loss.item() * num_graph
        total_a_s_loss += a_s_loss.item() * num_graph
        total_a_c_loss += a_c_loss.item() * num_graph
        total_a_ap_loss += a_ap_loss.item() * num_graph
        total_a_cov_loss += a_cov_loss.item() * num_graph
        total_graphs += num_graph

    return (
        total_task_loss / total_graphs,
        total_p_c_loss / total_graphs,
        total_r_c_loss / total_graphs,
        total_a_s_loss / total_graphs,
        total_a_c_loss / total_graphs,
        total_a_ap_loss / total_graphs,
        total_a_cov_loss / total_graphs,
    )


# =========================================================
# Eval
# =========================================================
@torch.no_grad()
def ldpc_eval_mtadam_sum_usercentric(
    dataLoader,
    model,
    tau,
    rho_p,
    rho_d,
    num_antenna,
    edge_base_dim,
    gamma_idx=1,
    power_limit=1.0,
    target_rate=1.0,
    rate_mode='per_user',
    target_ap_per_ue=4,
    assoc_card_mode='leq',
    max_ue_per_ap=4,
    ap_load_mode='leq',
    min_ap_per_ue=1.0,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    total_sum_rate = 0.0
    total_min_rate = 0.0
    total_power_c_loss = 0.0
    total_rate_c_loss = 0.0
    total_assoc_sparse_loss = 0.0
    total_assoc_card_loss = 0.0
    total_assoc_ap_loss = 0.0
    total_assoc_cover_loss = 0.0
    total_graphs = 0

    total_rate_violated = 0
    total_rate_units = 0

    total_avg_assoc = 0.0
    total_avg_ap_per_ue = 0.0
    total_avg_ue_per_ap = 0.0

    edge_key = ('CN', 'to', 'VN')

    for batch in dataLoader:
        batch = batch.to(device)
        num_graph = batch.num_graphs

        x_dict, edge_dict, edge_index_dict = model(batch)

        (
            sum_rate_batch,
            min_rate_batch,
            power_c_loss_batch,
            rate_c_loss_batch,
            assoc_sparse_loss_batch,
            assoc_card_loss_batch,
            assoc_ap_loss_batch,
            assoc_cover_loss_batch,
        ) = ldpc_loss_mtadam_sum_usercentric(
            graphData=batch,
            nodeFeatDict=x_dict,
            edgeDict=edge_dict,
            edge_index_dict=edge_index_dict,
            tau=tau,
            rho_p=rho_p,
            rho_d=rho_d,
            num_antenna=num_antenna,
            edge_base_dim=edge_base_dim,
            epochRatio=1.0,
            eval_mode=True,
            gamma_idx=gamma_idx,
            power_limit=power_limit,
            target_rate=target_rate,
            rate_mode=rate_mode,
            target_ap_per_ue=target_ap_per_ue,
            assoc_card_mode=assoc_card_mode,
            max_ue_per_ap=max_ue_per_ap,
            ap_load_mode=ap_load_mode,
            min_ap_per_ue=min_ap_per_ue,
        )

        total_sum_rate += sum_rate_batch.sum().item()
        total_min_rate += min_rate_batch.sum().item()
        total_power_c_loss += power_c_loss_batch.item() * num_graph
        total_rate_c_loss += rate_c_loss_batch.item() * num_graph
        total_assoc_sparse_loss += assoc_sparse_loss_batch.item() * num_graph
        total_assoc_card_loss += assoc_card_loss_batch.item() * num_graph
        total_assoc_ap_loss += assoc_ap_loss_batch.item() * num_graph
        total_assoc_cover_loss += assoc_cover_loss_batch.item() * num_graph
        total_graphs += num_graph

        (
            large_scale,
            channel_var,
            assoc_matrix,
            power_matrix,
            phi_matrix,
            edge_feat,
            edge_feat_4d,
        ) = extract_usercentric_matrices(
            graphData=batch,
            edgeDict=edge_dict,
            edge_base_dim=edge_base_dim,
            edge_key=edge_key,
        )

        total_avg_assoc += assoc_matrix.mean().item() * num_graph
        total_avg_ap_per_ue += assoc_matrix.sum(dim=1).mean().item() * num_graph
        total_avg_ue_per_ap += assoc_matrix.sum(dim=2).mean().item() * num_graph

        all_DS, all_PC, all_UI = component_calculate(
            power_matrix, channel_var, large_scale, phi_matrix, rho_d=rho_d
        )
        rate = rate_from_component(
            all_DS, all_PC, all_UI, num_antenna, rho_d=rho_d
        )

        if rate_mode == 'per_user':
            viol_mask = rate < target_rate
            total_rate_violated += viol_mask.sum().item()
            total_rate_units += rate.numel()
        else:
            min_rate, _ = torch.min(rate, dim=1)
            viol_mask = min_rate < target_rate
            total_rate_violated += viol_mask.sum().item()
            total_rate_units += num_graph

    if total_graphs == 0:
        return (
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )

    mean_sum_rate = total_sum_rate / total_graphs
    mean_min_rate = total_min_rate / total_graphs
    mean_power_c_loss = total_power_c_loss / total_graphs
    mean_rate_c_loss = total_rate_c_loss / total_graphs
    mean_assoc_sparse_loss = total_assoc_sparse_loss / total_graphs
    mean_assoc_card_loss = total_assoc_card_loss / total_graphs
    mean_assoc_ap_loss = total_assoc_ap_loss / total_graphs
    mean_assoc_cover_loss = total_assoc_cover_loss / total_graphs
    mean_avg_assoc = total_avg_assoc / total_graphs
    mean_avg_ap_per_ue = total_avg_ap_per_ue / total_graphs
    mean_avg_ue_per_ap = total_avg_ue_per_ap / total_graphs

    frac_rate_viol = (
        total_rate_violated / total_rate_units if total_rate_units > 0 else 0.0
    )

    return (
        mean_sum_rate,
        mean_min_rate,
        mean_power_c_loss,
        mean_rate_c_loss,
        mean_assoc_sparse_loss,
        mean_assoc_card_loss,
        mean_assoc_ap_loss,
        mean_assoc_cover_loss,
        frac_rate_viol,
        mean_avg_ap_per_ue,
        mean_avg_ue_per_ap,
    )