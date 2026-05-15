from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, HeteroConv, SAGEConv

from model_flexible import (
    BranchFusion,
    DrugRepresentationModule,
    EncoderConfig,
    FingerprintDrugRepresentationModule,
    FlexibleCellLineRepresentationModule,
    SmilesBPEDrugRepresentationModule,
)


class HGTClaimFUSECDR(nn.Module):
    """
    Configurable variant of the flexible FUSE-CDR graph module used only for
    HGT-claim experiments. It mirrors the main architecture but allows
    controlled enabling/disabling of the local GraphSAGE branch and the global
    HGT branch.
    """

    def __init__(
        self,
        atom_shape: int,
        encoder_configs: List[EncoderConfig],
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        hidden_dim: int = 256,
        output_dim: int = 100,
        fusion_dim: int = 512,
        dropout: float = 0.2,
        num_local_layers: int = 2,
        num_global_layers: int = 2,
        heads: int = 4,
        drug_num_gnn_layers: int = 3,
        use_local_branch: bool = True,
        use_global_branch: bool = True,
        drug_encoder_type: str = "graph",
        drug_input_dim: Optional[int] = None,
    ):
        super().__init__()
        if not use_local_branch and not use_global_branch:
            raise ValueError("At least one graph branch must be enabled.")
        self.hidden_dim = hidden_dim
        self.drug_encoder_type = drug_encoder_type
        self.use_local_branch = use_local_branch
        self.use_global_branch = use_global_branch

        if drug_encoder_type == "graph":
            self.drug_module = DrugRepresentationModule(
                atom_feature_dim=atom_shape,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_gnn_layers=drug_num_gnn_layers,
            )
        elif drug_encoder_type == "fingerprint":
            if drug_input_dim is None:
                raise ValueError("drug_input_dim is required for fingerprint drug encoder mode")
            self.drug_module = FingerprintDrugRepresentationModule(
                input_dim=drug_input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
            )
        elif drug_encoder_type == "smiles_bpe":
            self.drug_module = SmilesBPEDrugRepresentationModule(output_dim=hidden_dim)
        else:
            raise ValueError(f"Unsupported drug_encoder_type '{drug_encoder_type}'")

        self.cell_line_module = FlexibleCellLineRepresentationModule(
            encoder_configs=encoder_configs,
            fusion_dim=fusion_dim,
            output_dim=hidden_dim,
        )

        node_types, edge_types = metadata
        if set(node_types) != {"drug", "cell"}:
            raise ValueError("Metadata node_types must be exactly ['drug', 'cell']")

        self.local_convs = nn.ModuleList()
        if use_local_branch:
            for _ in range(num_local_layers):
                conv_dict = {
                    edge_type: SAGEConv(hidden_dim, hidden_dim) for edge_type in edge_types
                }
                self.local_convs.append(HeteroConv(conv_dict, aggr="sum"))

        self.global_convs = nn.ModuleList()
        self.global_norms = nn.ModuleList()
        if use_global_branch:
            for _ in range(num_global_layers):
                self.global_convs.append(
                    HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=heads)
                )
                self.global_norms.append(nn.LayerNorm(hidden_dim))

        self.fusion = BranchFusion(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, 1),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _encode_local(
        self,
        x_dict: Dict[str, torch.Tensor],
        hetero_graph_edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        h_local_dict = x_dict.copy()
        for conv in self.local_convs:
            h_local_dict = conv(h_local_dict, hetero_graph_edge_index_dict)
            out_dict = {}
            for key, val in h_local_dict.items():
                out_dict[key] = self.dropout_layer(F.relu(val))
            h_local_dict = out_dict
        return h_local_dict

    def _encode_global(
        self,
        x_dict: Dict[str, torch.Tensor],
        hetero_graph_edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        h_global_dict = x_dict.copy()
        for i, conv in enumerate(self.global_convs):
            h_global_dict = conv(h_global_dict, hetero_graph_edge_index_dict)
            out_dict = {}
            for key, val in h_global_dict.items():
                val = self.global_norms[i](val)
                out_dict[key] = self.dropout_layer(F.relu(val))
            h_global_dict = out_dict
        return h_global_dict

    def forward(
        self,
        drug_feature: torch.Tensor,
        drug_adj: Optional[torch.Tensor],
        ibatch: Optional[torch.Tensor],
        omics_data: Dict[str, Dict[str, torch.Tensor]],
        hetero_graph_edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        drug_indices: Optional[torch.Tensor] = None,
        cell_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if self.drug_encoder_type == "graph":
            if drug_adj is None or ibatch is None:
                raise ValueError("graph drug encoder requires drug_adj and ibatch")
            x_drug = self.drug_module(drug_feature, drug_adj, ibatch)
        else:
            x_drug = self.drug_module(drug_feature)
        x_cell = self.cell_line_module(omics_data)
        x_dict = {"drug": x_drug, "cell": x_cell}

        h_local_dict = self._encode_local(x_dict, hetero_graph_edge_index_dict) if self.use_local_branch else None
        h_global_dict = self._encode_global(x_dict, hetero_graph_edge_index_dict) if self.use_global_branch else None

        if h_local_dict is not None and h_global_dict is not None:
            h_fused_dict = {}
            for key in x_dict.keys():
                fused, _ = self.fusion(h_local_dict[key], h_global_dict[key])
                h_fused_dict[key] = fused
        elif h_local_dict is not None:
            h_fused_dict = h_local_dict
        else:
            h_fused_dict = h_global_dict

        logits = None
        if drug_indices is not None and cell_indices is not None:
            z_drug_batch = h_fused_dict["drug"][drug_indices]
            z_cell_batch = h_fused_dict["cell"][cell_indices]
            z_pair = torch.cat([z_drug_batch, z_cell_batch], dim=1)
            logits = self.predictor(z_pair)

        return {
            "node_embeddings": h_fused_dict,
            "drug_embeddings": h_fused_dict["drug"],
            "cell_embeddings": h_fused_dict["cell"],
            "logits": logits,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(hidden_dim={self.hidden_dim}, "
            f"use_local_branch={self.use_local_branch}, "
            f"use_global_branch={self.use_global_branch}, "
            f"local_layers={len(self.local_convs)}, global_layers={len(self.global_convs)})"
        )
