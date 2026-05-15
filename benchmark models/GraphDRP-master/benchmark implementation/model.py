import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_max_pool as gmp, global_mean_pool as gap

class GraphDRPModel(nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=735, output_dim=128, dropout=0.2):
        super(GraphDRPModel, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        self.output_dim = output_dim
        
        # --- Shared Cell Line Feature (1D CNN) ---
        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8)
        self.pool_xt_3 = nn.MaxPool1d(3)
        self.fc1_xt = nn.Linear(2944, output_dim)

        # --- Shared Final Head ---
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)
        self.sigmoid = nn.Sigmoid()

    def forward_cell(self, target):
        target = target[:, None, :] 
        x = F.relu(self.conv_xt_1(target))
        x = self.pool_xt_1(x)
        x = F.relu(self.conv_xt_2(x))
        x = self.pool_xt_2(x)
        x = F.relu(self.conv_xt_3(x))
        x = self.pool_xt_3(x)
        x = x.view(-1, x.shape[1] * x.shape[2])
        return self.fc1_xt(x)

    def forward_combined(self, drug_x, cell_x):
        xc = torch.cat((drug_x, cell_x), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = nn.Dropout(0.5)(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = nn.Dropout(0.5)(xc)
        out = self.out(xc)
        return self.sigmoid(out)

class GCNNet(GraphDRPModel):
    def __init__(self, **kwargs):
        super(GCNNet, self).__init__(**kwargs)
        self.conv1 = GCNConv(78, 78)
        self.conv2 = GCNConv(78, 78 * 2)
        self.conv3 = GCNConv(78 * 2, 78 * 4)
        self.fc_g1 = nn.Linear(78 * 4, 1024)
        self.fc_g2 = nn.Linear(1024, self.output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))
        x = gmp(x, batch)
        x = self.relu(self.fc_g1(x))
        x = nn.Dropout(0.5)(x)
        drug_x = self.fc_g2(x)
        drug_x = nn.Dropout(0.5)(drug_x)
        cell_x = self.forward_cell(data.target)
        return self.forward_combined(drug_x, cell_x)

class GATNet(GraphDRPModel):
    def __init__(self, **kwargs):
        super(GATNet, self).__init__(**kwargs)
        self.gcn1 = GATConv(78, 78, heads=10, dropout=0.2)
        self.gcn2 = GATConv(78 * 10, self.output_dim, dropout=0.2)
        self.fc_g1 = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)
        x = self.fc_g1(x)
        drug_x = self.relu(x)
        cell_x = self.forward_cell(data.target)
        return self.forward_combined(drug_x, cell_x)

class GINConvNet(GraphDRPModel):
    def __init__(self, **kwargs):
        super(GINConvNet, self).__init__(**kwargs)
        dim = 32
        self.conv1 = GINConv(nn.Sequential(nn.Linear(78, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn1 = nn.BatchNorm1d(dim)
        self.conv2 = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn2 = nn.BatchNorm1d(dim)
        self.conv3 = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn3 = nn.BatchNorm1d(dim)
        self.conv4 = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn4 = nn.BatchNorm1d(dim)
        self.conv5 = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn5 = nn.BatchNorm1d(dim)
        self.fc1_xd = nn.Linear(dim, self.output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        from torch_geometric.nn import global_add_pool
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        drug_x = F.dropout(x, p=0.2, training=self.training)
        cell_x = self.forward_cell(data.target)
        return self.forward_combined(drug_x, cell_x)

class GAT_GCN(GraphDRPModel):
    def __init__(self, **kwargs):
        super(GAT_GCN, self).__init__(**kwargs)
        self.conv1 = GATConv(78, 78, heads=10)
        self.conv2 = GCNConv(78 * 10, 78 * 10)
        self.fc_g1 = nn.Linear(78 * 10 * 2, 1500)
        self.fc_g2 = nn.Linear(1500, self.output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        from torch_geometric.nn import global_max_pool as gmp, global_mean_pool as gap
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = nn.Dropout(0.2)(self.relu(self.fc_g1(x)))
        drug_x = self.fc_g2(x)
        cell_x = self.forward_cell(data.target)
        return self.forward_combined(drug_x, cell_x)

def get_model(name, **kwargs):
    models = {"GCN": GCNNet, "GAT": GATNet, "GIN": GINConvNet, "GAT_GCN": GAT_GCN}
    return models[name](**kwargs)
