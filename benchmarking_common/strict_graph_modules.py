import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SharedGraphRefiner(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = GCNConv(dim, dim, add_self_loops=True)
        self.bn = nn.BatchNorm1d(dim)
        self.fuse = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0 or x.size(0) <= 1:
            return x
        shared = self.conv(x, edge_index)
        shared = F.relu(shared)
        shared = self.bn(shared)
        return F.relu(self.fuse(torch.cat((x, shared), dim=1)))


class BipartiteResponseRefiner(nn.Module):
    def __init__(self, cell_dim: int, drug_dim: int):
        super().__init__()
        self.cell_from_drug = nn.Linear(drug_dim, cell_dim)
        self.drug_from_cell = nn.Linear(cell_dim, drug_dim)
        self.cell_fuse = nn.Linear(cell_dim * 2, cell_dim)
        self.drug_fuse = nn.Linear(drug_dim * 2, drug_dim)
        self.cell_bn = nn.BatchNorm1d(cell_dim)
        self.drug_bn = nn.BatchNorm1d(drug_dim)

    def forward(
        self,
        cell_embeddings: torch.Tensor,
        drug_embeddings: torch.Tensor,
        response_edge_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if response_edge_index is None or response_edge_index.numel() == 0:
            return cell_embeddings, drug_embeddings

        cell_idx = response_edge_index[0]
        drug_idx = response_edge_index[1]
        num_cells = cell_embeddings.size(0)
        num_drugs = drug_embeddings.size(0)

        cell_messages = torch.zeros_like(cell_embeddings)
        drug_messages = torch.zeros_like(drug_embeddings)
        cell_degree = torch.zeros((num_cells, 1), dtype=cell_embeddings.dtype, device=cell_embeddings.device)
        drug_degree = torch.zeros((num_drugs, 1), dtype=drug_embeddings.dtype, device=drug_embeddings.device)

        cell_messages.index_add_(0, cell_idx, self.cell_from_drug(drug_embeddings[drug_idx]))
        drug_messages.index_add_(0, drug_idx, self.drug_from_cell(cell_embeddings[cell_idx]))
        one_cell = torch.ones((cell_idx.numel(), 1), dtype=cell_embeddings.dtype, device=cell_embeddings.device)
        one_drug = torch.ones((drug_idx.numel(), 1), dtype=drug_embeddings.dtype, device=drug_embeddings.device)
        cell_degree.index_add_(0, cell_idx, one_cell)
        drug_degree.index_add_(0, drug_idx, one_drug)

        cell_messages = cell_messages / cell_degree.clamp_min(1.0)
        drug_messages = drug_messages / drug_degree.clamp_min(1.0)
        cell_messages = self.cell_bn(cell_messages)
        drug_messages = self.drug_bn(drug_messages)

        cell_out = F.relu(self.cell_fuse(torch.cat((cell_embeddings, F.relu(cell_messages)), dim=1)))
        drug_out = F.relu(self.drug_fuse(torch.cat((drug_embeddings, F.relu(drug_messages)), dim=1)))
        return cell_out, drug_out
