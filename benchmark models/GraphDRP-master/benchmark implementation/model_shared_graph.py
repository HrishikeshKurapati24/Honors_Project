import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_add_pool, global_max_pool as gmp, global_mean_pool as gap

from benchmarking_common.strict_graph_modules import BipartiteResponseRefiner, SharedGraphRefiner


class GraphDRPSharedBase(nn.Module):
    def __init__(self, atom_dim: int, num_features_xt: int, output_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.atom_dim = atom_dim
        self.output_dim = output_dim
        self.relu = nn.ReLU()
        self.dropout_rate = float(dropout)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8)
        self.pool_xt_3 = nn.MaxPool1d(3)

        with torch.no_grad():
            probe = torch.zeros(1, 1, num_features_xt)
            probe = self.pool_xt_1(F.relu(self.conv_xt_1(probe)))
            probe = self.pool_xt_2(F.relu(self.conv_xt_2(probe)))
            probe = self.pool_xt_3(F.relu(self.conv_xt_3(probe)))
            flat_dim = int(probe.shape[1] * probe.shape[2])
        self.fc1_xt = nn.Linear(flat_dim, output_dim)

        self.cell_refiner = SharedGraphRefiner(output_dim)
        self.drug_refiner = SharedGraphRefiner(output_dim)
        self.response_refiner = BipartiteResponseRefiner(output_dim, output_dim)

        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, 1)

    def encode_cells(self, target: torch.Tensor, cell_edge_index: torch.Tensor) -> torch.Tensor:
        target = target[:, None, :]
        x = self.pool_xt_1(F.relu(self.conv_xt_1(target)))
        x = self.pool_xt_2(F.relu(self.conv_xt_2(x)))
        x = self.pool_xt_3(F.relu(self.conv_xt_3(x)))
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = self.fc1_xt(x)
        return self.cell_refiner(x, cell_edge_index)

    def predict_pair_logits(
        self,
        drug_embeddings: torch.Tensor,
        cell_embeddings: torch.Tensor,
        pair_indices: torch.Tensor,
    ) -> torch.Tensor:
        pair_features = torch.cat(
            (
                drug_embeddings[pair_indices[:, 1]],
                cell_embeddings[pair_indices[:, 0]],
            ),
            dim=1,
        )
        x = self.dropout(self.relu(self.fc1(pair_features)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.out(x).view(-1)

    def predict_pairs(
        self,
        drug_embeddings: torch.Tensor,
        cell_embeddings: torch.Tensor,
        pair_indices: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.predict_pair_logits(drug_embeddings, cell_embeddings, pair_indices))

    def refine_with_response_edges(
        self,
        cell_embeddings: torch.Tensor,
        drug_embeddings: torch.Tensor,
        response_edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.response_refiner(cell_embeddings, drug_embeddings, response_edge_index)


class GCNNetShared(GraphDRPSharedBase):
    def __init__(self, atom_dim: int, num_features_xt: int, output_dim: int = 128, dropout: float = 0.2):
        super().__init__(atom_dim=atom_dim, num_features_xt=num_features_xt, output_dim=output_dim, dropout=dropout)
        self.conv1 = GCNConv(atom_dim, atom_dim)
        self.conv2 = GCNConv(atom_dim, atom_dim * 2)
        self.conv3 = GCNConv(atom_dim * 2, atom_dim * 4)
        self.fc_g1 = nn.Linear(atom_dim * 4, 1024)
        self.fc_g2 = nn.Linear(1024, output_dim)

    def encode_drugs(self, batch, drug_edge_index: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(batch.x, batch.edge_index))
        x = self.relu(self.conv2(x, batch.edge_index))
        x = self.relu(self.conv3(x, batch.edge_index))
        x = gmp(x, batch.batch)
        x = self.dropout(self.relu(self.fc_g1(x)))
        x = self.fc_g2(x)
        x = self.dropout(x)
        return self.drug_refiner(x, drug_edge_index)


class GATNetShared(GraphDRPSharedBase):
    def __init__(self, atom_dim: int, num_features_xt: int, output_dim: int = 128, dropout: float = 0.2):
        super().__init__(atom_dim=atom_dim, num_features_xt=num_features_xt, output_dim=output_dim, dropout=dropout)
        self.gcn1 = GATConv(atom_dim, atom_dim, heads=10, dropout=0.2)
        self.gcn2 = GATConv(atom_dim * 10, output_dim, dropout=0.2)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

    def encode_drugs(self, batch, drug_edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(batch.x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.gcn1(x, batch.edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.relu(self.gcn2(x, batch.edge_index))
        x = gmp(x, batch.batch)
        x = self.relu(self.fc_g1(x))
        return self.drug_refiner(x, drug_edge_index)


class GINConvNetShared(GraphDRPSharedBase):
    def __init__(self, atom_dim: int, num_features_xt: int, output_dim: int = 128, dropout: float = 0.2):
        super().__init__(atom_dim=atom_dim, num_features_xt=num_features_xt, output_dim=output_dim, dropout=dropout)
        dim = 32
        self.conv1 = GINConv(nn.Sequential(nn.Linear(atom_dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn1 = nn.BatchNorm1d(dim)
        self.conv2 = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn2 = nn.BatchNorm1d(dim)
        self.conv3 = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn3 = nn.BatchNorm1d(dim)
        self.conv4 = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn4 = nn.BatchNorm1d(dim)
        self.conv5 = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.bn5 = nn.BatchNorm1d(dim)
        self.fc1_xd = nn.Linear(dim, output_dim)

    def encode_drugs(self, batch, drug_edge_index: torch.Tensor) -> torch.Tensor:
        x = self.bn1(F.relu(self.conv1(batch.x, batch.edge_index)))
        x = self.bn2(F.relu(self.conv2(x, batch.edge_index)))
        x = self.bn3(F.relu(self.conv3(x, batch.edge_index)))
        x = self.bn4(F.relu(self.conv4(x, batch.edge_index)))
        x = self.bn5(F.relu(self.conv5(x, batch.edge_index)))
        x = global_add_pool(x, batch.batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return self.drug_refiner(x, drug_edge_index)


class GAT_GCNShared(GraphDRPSharedBase):
    def __init__(self, atom_dim: int, num_features_xt: int, output_dim: int = 128, dropout: float = 0.2):
        super().__init__(atom_dim=atom_dim, num_features_xt=num_features_xt, output_dim=output_dim, dropout=dropout)
        self.conv1 = GATConv(atom_dim, atom_dim, heads=10)
        self.conv2 = GCNConv(atom_dim * 10, atom_dim * 10)
        self.fc_g1 = nn.Linear(atom_dim * 10 * 2, 1500)
        self.fc_g2 = nn.Linear(1500, output_dim)

    def encode_drugs(self, batch, drug_edge_index: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(batch.x, batch.edge_index))
        x = self.relu(self.conv2(x, batch.edge_index))
        x = torch.cat([gmp(x, batch.batch), gap(x, batch.batch)], dim=1)
        x = F.dropout(self.relu(self.fc_g1(x)), p=self.dropout_rate, training=self.training)
        x = self.fc_g2(x)
        return self.drug_refiner(x, drug_edge_index)


def get_model(model_type: str, atom_dim: int, num_features_xt: int, output_dim: int = 128, dropout: float = 0.2):
    models = {
        "GCN": GCNNetShared,
        "GAT": GATNetShared,
        "GIN": GINConvNetShared,
        "GAT_GCN": GAT_GCNShared,
    }
    return models[model_type](atom_dim=atom_dim, num_features_xt=num_features_xt, output_dim=output_dim, dropout=dropout)
