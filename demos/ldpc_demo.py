import torch
import numpy as np
from Utils.data_gen import create_graph
from torch_geometric.loader import DataLoader
from Models.GNN import LDPCHetNet

# Small demo runner – will run a single forward pass with the LDPC-inspired GNN
# 1) adjust parameters for your environment
batch_size = 4
num_ue = 6
num_ap = 30

# create toy data (reuses the same mat file creating logic from main.py)
# Note: set path to Data/dl_data... file
import scipy.io
mat = scipy.io.loadmat('Data/dl_data_2000_{}_{}.mat'.format(num_ue, num_ap))
Beta_all = mat['betas']
Phi_all = mat['Phii_cf'].transpose(0,2,1)
# Create dataset
beta_mean = np.mean(Beta_all)
beta_std = np.std(Beta_all)
# use only first few samples
Beta_small = Beta_all[:batch_size]
Phi_small = Phi_all[:batch_size]

# 'het' type dataset
data = create_graph(Beta_small, Phi_small, beta_mean, beta_std, 'het')
loader = DataLoader(data, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# build model with same dim_meta as main.py
ap_dim = data[0][0]['AP'].x.shape[1]
ue_dim = data[0][0]['UE'].x.shape[1]
edge_dim = data[0][0]['down'].edge_attr.shape[1]
metadata = [('UE','up','AP'),('AP','down','UE')]
dim_dict = {'UE':ue_dim,'AP':ap_dim,'edge':edge_dim}

model = LDPCHetNet(metadata=metadata, dim_dict=dim_dict, out_channels=32, num_layers=3, num_checks=4, chk_degree=2, ldpc_iter=3)
model.to(device)
model.eval()

for batch in loader:
    batch = batch.to(device)
    with torch.no_grad():
        x_dict, edge_attr_dict, edge_index_dict = model(batch)
    print('UE feature shape:', x_dict['UE'].shape)
    break
