import importlib.util
import os
import sys

import torch
import torch.nn as nn

from benchmarking_common.strict_graph_modules import BipartiteResponseRefiner, SharedGraphRefiner


_BASE_MODULE_NAME = "_benchmark_deepttc_base_model"
if _BASE_MODULE_NAME in sys.modules:
    _base_model = sys.modules[_BASE_MODULE_NAME]
else:
    _base_model_path = os.path.join(os.path.dirname(__file__), "model.py")
    _base_spec = importlib.util.spec_from_file_location(_BASE_MODULE_NAME, _base_model_path)
    if _base_spec is None or _base_spec.loader is None:
        raise ImportError(f"Unable to load DeepTTC base model from {_base_model_path}")
    _base_model = importlib.util.module_from_spec(_base_spec)
    sys.modules[_BASE_MODULE_NAME] = _base_model
    _base_spec.loader.exec_module(_base_model)

GeneMLP = _base_model.GeneMLP
LayerNorm = _base_model._ORIGINAL_HELPER.LayerNorm
Encoder_MultipleLayers = _base_model._ORIGINAL_HELPER.Encoder_MultipleLayers


class DrugGraphTransformer(nn.Module):
    def __init__(self, atom_dim: int, max_tokens: int = 50):
        super().__init__()
        self.hidden_size = 128
        self.max_tokens = max_tokens
        self.atom_projection = nn.Linear(atom_dim, self.hidden_size)
        self.position_embeddings = nn.Embedding(max_tokens, self.hidden_size)
        self.layer_norm = LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.encoder = Encoder_MultipleLayers(
            8,
            self.hidden_size,
            512,
            8,
            0.1,
            0.1,
        )

    def forward(self, atom_tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_length = atom_tokens.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=atom_tokens.device)
        position_ids = position_ids.unsqueeze(0).expand(atom_tokens.size(0), -1)
        emb = self.atom_projection(atom_tokens)
        emb = emb + self.position_embeddings(position_ids)
        emb = self.layer_norm(emb)
        emb = self.dropout(emb)

        ex_e_mask = attention_mask.long().unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask.float()) * -10000.0
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())
        return encoded_layers[:, 0]


class DeepTTCSharedGraph(nn.Module):
    def __init__(
        self,
        gene_input_dim: int,
        atom_dim: int,
        classifier_dropout: float = 0.1,
        max_tokens: int = 50,
    ):
        super().__init__()
        self.drug_model = DrugGraphTransformer(atom_dim=atom_dim, max_tokens=max_tokens)
        self.gene_model = GeneMLP(gene_input_dim)

        self.cell_refiner = SharedGraphRefiner(256)
        self.drug_refiner = SharedGraphRefiner(128)
        self.response_refiner = BipartiteResponseRefiner(256, 128)

        dropout = nn.Dropout(classifier_dropout)
        hidden_dims = [1024, 1024, 512]
        dims = [128 + 256] + hidden_dims + [1]
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(dropout)
        self.classifier = nn.Sequential(*layers)

    def encode_drugs(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        drug_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.drug_model(token_ids, attention_mask)
        return self.drug_refiner(embeddings, drug_edge_index)

    def encode_cells(
        self,
        expression: torch.Tensor,
        cell_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.gene_model(expression)
        return self.cell_refiner(embeddings, cell_edge_index)

    def predict_pair_logits(
        self,
        drug_embeddings: torch.Tensor,
        cell_embeddings: torch.Tensor,
        pair_indices: torch.Tensor,
    ) -> torch.Tensor:
        pair_embeddings = torch.cat(
            (
                drug_embeddings[pair_indices[:, 1]],
                cell_embeddings[pair_indices[:, 0]],
            ),
            dim=1,
        )
        return self.classifier(pair_embeddings).view(-1)

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

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        expression: torch.Tensor,
        pair_indices: torch.Tensor,
        cell_edge_index: torch.Tensor,
        drug_edge_index: torch.Tensor,
        response_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        drug_embeddings = self.encode_drugs(token_ids, attention_mask, drug_edge_index)
        cell_embeddings = self.encode_cells(expression, cell_edge_index)
        cell_embeddings, drug_embeddings = self.refine_with_response_edges(
            cell_embeddings,
            drug_embeddings,
            response_edge_index,
        )
        return self.predict_pair_logits(
            drug_embeddings=drug_embeddings,
            cell_embeddings=cell_embeddings,
            pair_indices=pair_indices,
        )
