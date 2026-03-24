import torch
import numpy as np
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, HeteroData
# 1. Location & Channel generation
def Generate_Input(num_H, tau, K, M, Pd, D=1, Hb=15, Hm=1.65, f=1900,
                    var_noise=1, Pmin=0, power_f=0.2, seed=2017, d0=0.01, d1=0.05):
    np.random.seed(seed)
    aL = (1.1 * np.log10(f) - 0.7) * Hm - (1.56 * np.log10(f) - 0.8)
    L = 46.3+33.9*np.log10(f)-13.82*np.log10(Hb)-aL
    
    random_matrix = np.random.randn(tau, tau)
    U, S, V = np.linalg.svd(random_matrix) # Pilot coodbook

    Beta_ALL = np.zeros((num_H, M, K), dtype=np.float32)
    Phii_All  = np.zeros((num_H, K, tau), dtype=np.float32)
    
        

    for each_data in range(num_H):
        # Pilot assignment
        Phii = np.zeros((K,tau))
        for k in range(K):
            Point = k % tau
            # Point = np.random.randint(1, tau+1)
            Phii[k,:] = U[Point - 1,:]
        Phii_All[each_data,:,:] = Phii
        
        
        # Random positions for APs and UEs
        AP = np.random.uniform(-D, D, size=(M, 2))
        Ter = np.random.uniform(-D, D, size=(K, 2))
        # Create an MxK large-scale coefficients beta_mk
        BETAA = np.zeros((M, K))
        # dist = np.zeros((M, K))

        for m in range(M):
            for k in range(K):
                dist = np.linalg.norm(AP[m, :] - Ter[k, :])

                if dist < d0:
                    betadB = -L - 35 * np.log10(d1) + 20 * np.log10(d1) - 20 * np.log10(d0)
                elif dist >= d0 and dist <= d1:
                    betadB = -L - 35 * np.log10(d1) + 20 * np.log10(d1) - 20 * np.log10(dist)
                else:
                    betadB = -L - 35 * np.log10(dist) + np.random.normal(0, 1) * 7

                BETAA[m, k] = 10 ** (betadB / 10) * Pd
        Beta_ALL[each_data,:,:] = BETAA
    return Beta_ALL, Phii_All



def get_cg(n):
    adj = []
    for i in range(0,n):
        for j in range(0,n):
            if(not(i==j)):
                adj.append([i,j])
    return adj




def create_graph_ldpc(Beta_all, Gamma_all, Eta_all, Phi_all, type='het', isDecentralized=True):
    num_sample, num_AP, num_UE = Beta_all.shape
    data_list = []
    if isDecentralized:
        for each_AP in range(num_AP):
            data_single_AP = []
            for each_sample in range(num_sample):
                if type=='het':
                    data = full_het_graph(
                        Beta_all[each_sample, each_AP][np.newaxis, :], 
                        Gamma_all[each_sample, each_AP][np.newaxis, :], 
                        Eta_all[each_sample, each_AP][np.newaxis, :], 
                        Phi_all[each_sample], 
                        each_AP, each_sample
                    )
                else:
                    raise ValueError(f'{type} graph is not defined!')
                data_single_AP.append(data)
            data_list.append(data_single_AP)
    else:
        for each_sample in range(num_sample):
            data = full_het_graph(
                Beta_all[each_sample], 
                Gamma_all[each_sample], 
                Eta_all[each_sample], 
                Phi_all[each_sample]
            )
            data_list.append(data)
    return data_list 


def full_het_graph(beta_single_sample, gamma_single_sample, eta_single_all, phi_single_sample, ap_id=None, sample_id=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_AP, num_UE = beta_single_sample.shape

    # Creating node features (random values for AP and UE nodes)
    ap_features = np.ones((num_AP, 1), dtype=np.float32)   # np.random.rand(num_AP, 1)  # Random feature for AP node (dim 1)
    ue_features = phi_single_sample  # Random feature for UE nodes (dim 1)

    # Concatenate features for both AP and UE nodes
    x_ap = torch.tensor(ap_features, dtype=torch.float32).to(device)
    x_ue = torch.tensor(ue_features, dtype=torch.float32).to(device)

    # Combine AP and UE node features
    x = {'CN': x_ap, 'VN': x_ue}

    # Define edges (connect AP to all UEs in a bipartite manner)
    edge_index_ap_down_ue = []
    edge_index_ue_up_ap = []

    for ap_idx in range(num_AP):
        for ue_idx in range(num_UE):
            edge_index_ap_down_ue.append([ap_idx, ue_idx])  # AP (0) to UE (ue_idx)
            # edge_index_ue_up_ap.append([ue_idx, ap_idx])  # UE (ue_idx) to AP (0)
    
    for ue_idx in range(num_UE):
        for ap_idx in range(num_AP):
            edge_index_ue_up_ap.append([ue_idx, ap_idx])  # UE (ue_idx) to AP (0)

    edge_index_ap_down_ue = torch.tensor(edge_index_ap_down_ue, dtype=torch.long).t().contiguous().to(device)
    edge_index_ue_up_ap = torch.tensor(edge_index_ue_up_ap, dtype=torch.long).t().contiguous().to(device)

    # edge_attr_ap_to_ue = torch.tensor(beta_single_sample.reshape(-1, 1), dtype=torch.float32).to(device)
    # edge_attr_ue_up_ap = torch.tensor(beta_single_sample.T.reshape(-1, 1), dtype=torch.float32).to(device)

    beta_up = beta_single_sample.reshape(-1, 1)
    gamma_up = gamma_single_sample.reshape(-1, 1)
    edge_attr_ap_to_ue = np.concatenate((beta_up, gamma_up), axis=1)
    edge_attr_ap_to_ue = torch.tensor(edge_attr_ap_to_ue, dtype=torch.float32).to(device)
    
    
    beta_down = beta_single_sample.T.reshape(-1, 1)
    gamma_down = gamma_single_sample.T.reshape(-1, 1)
    edge_attr_ue_up_ap = np.concatenate((beta_down, gamma_down), axis=1)
    edge_attr_ue_up_ap = torch.tensor(edge_attr_ue_up_ap, dtype=torch.float32).to(device)

    # Create the heterogeneous graph data
    data = HeteroData()
    data['VN'].x = x['VN']
    data['CN'].x = x['CN']
    data['VN', 'to', 'CN'].edge_index =  edge_index_ue_up_ap
    data['VN', 'to', 'CN'].edge_attr =  edge_attr_ue_up_ap
    data['CN', 'to', 'VN'].edge_index = edge_index_ap_down_ue
    data['CN', 'to', 'VN'].edge_attr = edge_attr_ap_to_ue
    
    data.y = torch.tensor([eta_single_all], dtype=torch.float32).to(device)
    
    data.ap_id = ap_id
    data.sample_id = sample_id

    return data

    

def build_loader(per_ap_datasets, batch_size, seed, drop_last=True, num_workers=0):
    n = len(per_ap_datasets[0])
    assert all(len(ds) == n for ds in per_ap_datasets), "All AP datasets must have same length."
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g).tolist()  # same random order for all APs

    loaders = []
    for ds in per_ap_datasets:
        subset = Subset(ds, order)  # fixes the order
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers))
    return loaders


def build_cen_loader_ldpc(betaMatrix, gammaMatrix, etaMatrix, phiMatrix, batchSize, isShuffle=False):
    log_large_scale = np.log1p(betaMatrix)
    deta_cen = create_graph_ldpc(log_large_scale, gammaMatrix, etaMatrix, phiMatrix, 'het', isDecentralized=False)
    loader_cen = DataLoader(deta_cen, batch_size=batchSize, shuffle=isShuffle)
    return deta_cen, loader_cen


def build_decen_loader_ldpc(betaMatrix, gammaMatrix, etaMatrix, phiMatrix, batchSize, seed=1712):
    log_large_scale = np.log1p(betaMatrix)
    data_decen = create_graph_ldpc(log_large_scale, gammaMatrix, etaMatrix, phiMatrix, 'het')
    loader_decen = build_loader(data_decen, batchSize, seed=seed, drop_last=False)
    return data_decen, loader_decen
    
