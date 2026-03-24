import torch
from Utils.comm import power_from_raw, variance_calculate, rate_calculation, component_calculate, rate_from_component
import torch.nn.functional as F
import torch.nn as nn
# Centralized Training


def ldpc_loss_function(graphData, nodeFeatDict, edgeDict, tau, rho_p, rho_d, num_antenna, epochRatio=1, eval_mode=False):
    num_graph = graphData.num_graphs
    criterion = nn.MSELoss(reduction='mean') 
    
        
    # label_power = torch.sqrt(graphData.y)
    num_APs = graphData['CN'].x.shape[0]//num_graph
    num_UEs = graphData['VN'].x.shape[0]//num_graph
    
    large_scale = edgeDict['CN','to','VN'].reshape(num_graph, num_APs, num_UEs, -1)[:,:,:,0]
    large_scale = torch.expm1(large_scale)
    power_matrix_raw = edgeDict['CN','to','VN'].reshape(num_graph, num_APs, num_UEs, -1)[:,:,:,-1]
    # ap_gate = nodeFeatDict['AP'].reshape(num_graph, num_APs, -1)
    phi_matrix = graphData['VN'].x.reshape(num_graph, num_UEs, -1)
    # channel_var = variance_calculate(large_scale, phi_matrix, tau=tau, rho_p=rho_p)
    channel_var = edgeDict['CN','to','VN'].reshape(num_graph, num_APs, num_UEs, -1)[:,:,:,1]
    power_matrix = power_from_raw(power_matrix_raw, channel_var, num_antenna)
    
    all_DS, all_PC, all_UI = component_calculate(power_matrix, channel_var, large_scale, phi_matrix, rho_d=rho_d)
    rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna, rho_d=rho_d)
    
    if torch.isnan(rate).any():
        print(power_matrix_raw)
        raise ValueError('Nan in rate')
    
    
    if eval_mode:
        min_rate, _ = torch.min(rate, dim=1)
        full = torch.ones_like(power_matrix)
        rate_full_one = rate_calculation(full, large_scale, channel_var, phi_matrix, rho_d, num_antenna)
        min_rate_one, _ = torch.min(rate_full_one, dim=1)
        return min_rate, min_rate_one
    else:
        epochRatio = min(1.0, epochRatio)
        min_rate_detach, _ = torch.min(rate.detach(), dim=1)
        
        # Option 1: hard-min
        min_rate, _ = torch.min(rate, dim=1)
        loss = torch.mean(-min_rate)
        
  
        

        return loss, torch.mean(min_rate_detach.detach())


def ldpc_train( epochRatio,
        dataLoader, model, optimizer,
        tau, rho_p, rho_d, num_antenna
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    
    total_loss = 0.0
    total_graphs = 0
    for batch in dataLoader:
        optimizer.zero_grad(set_to_none=True) 
        batch = batch.to(device)
        num_graph = batch.num_graphs
        
        x_dict, edge_dict, edge_index = model(batch)
        loss, _ = ldpc_loss_function(
            batch, x_dict, edge_dict,
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna,
            epochRatio=epochRatio
        )
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * num_graph
        total_graphs += num_graph

    return total_loss/total_graphs


@torch.no_grad()
def ldpc_eval(
        dataLoader, model,
        tau, rho_p, rho_d, num_antenna
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    total_min_rate = 0.0
    total_graphs = 0
    for batch in dataLoader:
        batch = batch.to(device)
        num_graph = batch.num_graphs
        
        x_dict, edge_dict, edge_index = model(batch)
        _, min_rate = ldpc_loss_function(
            batch, x_dict, edge_dict,
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna
        )
        
        total_min_rate += min_rate.item() * num_graph
        total_graphs += num_graph

    return total_min_rate/total_graphs 


