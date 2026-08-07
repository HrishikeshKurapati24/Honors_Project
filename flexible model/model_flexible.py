"""
model_flexible.py
Locked flexible FUSE-CDR architecture:
- Drug encoder: GIN
- Graph type: Heterogeneous
- Local graph conv: GraphSAGE (SAGEConv via HeteroConv)
- Global graph conv: HGT (graph transformer branch)

NOTE:
FlexibleCellLineRepresentationModule is intentionally kept unchanged.
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINConv,
    HGTConv,
    HeteroConv,
    SAGEConv,
    global_max_pool as gmp,
)


# ====================================================================
# CONFIGURATION & REGISTRY
# ====================================================================


@dataclass
class EncoderConfig:
    category: Literal[
        "genomics",
        "epigenomics",
        "transcriptomics",
        "proteomics",
        "metabolomics",
        "pathway",
    ]
    subtype: str
    encoder_type: str
    input_dim: int
    output_dim: int = 256


class OmicsEncoderRegistry:
    """
    Registry for omics encoders.

    # HOW TO REGISTER A NEW ENCODER:
    # 1. Add a static method _build_YOURNAME_fc(in_dim) -> nn.Module.
    # 2. Add 'YOURNAME_fc' to the ENCODERS and ENCODER_OUTPUT_DIMS dictionaries.
    # 3. Intermediate dimensions should match the input feature density.
    """

    @staticmethod
    def _build_genomics_fc(in_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    @staticmethod
    def _build_chromatin_fc(in_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    @staticmethod
    def _build_transcriptomics_fc(in_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 256),
        )

    @staticmethod
    def _build_cnv_fc(in_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 256),
        )

    @staticmethod
    def _build_miRNA_fc(in_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    @staticmethod
    def _build_proteomics_fc(in_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
        )

    @staticmethod
    def _build_metabolomics_fc(in_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    @staticmethod
    def _build_pathway_fc(in_dim: int) -> nn.Module:
        return nn.Sequential(nn.Linear(in_dim, 256))

    @staticmethod
    def _build_methylation_fc(in_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )

    ENCODERS = {
        "genomics_fc": _build_genomics_fc.__func__,
        "cnv_fc": _build_cnv_fc.__func__,
        "chromatin_fc": _build_chromatin_fc.__func__,
        "methylation_fc": _build_methylation_fc.__func__,
        "transcriptomics_fc": _build_transcriptomics_fc.__func__,
        "miRNA_fc": _build_miRNA_fc.__func__,
        "proteomics_fc": _build_proteomics_fc.__func__,
        "metabolomics_fc": _build_metabolomics_fc.__func__,
        "pathway_fc": _build_pathway_fc.__func__,
    }

    ENCODER_OUTPUT_DIMS = {
        "genomics_fc": 64,
        "cnv_fc": 256,
        "chromatin_fc": 64,
        "methylation_fc": 256,
        "transcriptomics_fc": 256,
        "miRNA_fc": 128,
        "proteomics_fc": 64,
        "metabolomics_fc": 128,
        "pathway_fc": 256,
    }

    # Explicitly supported dataset-specific {category, subtype} pairs.
    CATEGORY_SUBTYPE_TO_ENCODER = {
        ("genomics", "mutation"): "genomics_fc",
        ("genomics", "cnv"): "cnv_fc",
        ("epigenomics", "chromatin"): "chromatin_fc",
        ("epigenomics", "methylation"): "methylation_fc",
        ("transcriptomics", "expression"): "transcriptomics_fc",
        ("transcriptomics", "miRNA"): "miRNA_fc",
        ("proteomics", "reverse_phase"): "proteomics_fc",
        ("metabolomics", "profile"): "metabolomics_fc",
        ("pathway", "pathway"): "pathway_fc",
    }

    @classmethod
    def resolve_encoder_type(cls, category: str, subtype: str) -> str:
        if category == "pathway" and subtype:
            return "pathway_fc"
        key = (category, subtype)
        if key not in cls.CATEGORY_SUBTYPE_TO_ENCODER:
            supported = ", ".join(
                [f"{c}/{s}" for c, s in sorted(cls.CATEGORY_SUBTYPE_TO_ENCODER.keys())]
            )
            raise KeyError(
                f"Unsupported omics pair '{category}/{subtype}'. "
                f"Supported pairs: {supported}"
            )
        return cls.CATEGORY_SUBTYPE_TO_ENCODER[key]

    @classmethod
    def build_encoder(cls, config: EncoderConfig) -> nn.Module:
        """Builds encoder and adds automated projection to match output_dim."""
        expected_encoder = cls.resolve_encoder_type(config.category, config.subtype)
        if config.encoder_type != expected_encoder:
            raise ValueError(
                f"Encoder mismatch for {config.category}/{config.subtype}: "
                f"expected '{expected_encoder}', got '{config.encoder_type}'."
            )
        if config.encoder_type not in cls.ENCODERS:
            raise KeyError(
                f"Unknown encoder_type '{config.encoder_type}'. Supported: {sorted(cls.ENCODERS)}"
            )
        base = cls.ENCODERS[config.encoder_type](config.input_dim)
        base_out = cls.ENCODER_OUTPUT_DIMS[config.encoder_type]
        if base_out != config.output_dim:
            return nn.Sequential(base, nn.Linear(base_out, config.output_dim))
        return base


# ====================================================================
# NODE REPRESENTATION MODULES (Hierarchical Fusion)
# ====================================================================


class IntraCategoryMultiHeadFusion(nn.Module):
    """
    Dedicated intra-category fusion block for subtype representations.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, subtype_reps: List[torch.Tensor]) -> torch.Tensor:
        if len(subtype_reps) == 1:
            return subtype_reps[0]
        stacked = torch.stack(subtype_reps, dim=1)
        fused, _ = self.attn(stacked, stacked, stacked)
        return fused.mean(dim=1)


class CrossModalAttention(nn.Module):
    """
    Stable directional cross-modal attention for iterative fusion.

    fused = CrossModalAttention(fused, new_modality)
    Query = fused representation, Key/Value = new modality representation.
    """

    def __init__(
        self,
        dim_fused: int,
        dim_new: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.q_proj = nn.Linear(dim_fused, output_dim)
        self.k_proj = nn.Linear(dim_new, output_dim)
        self.v_proj = nn.Linear(dim_new, output_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_q = nn.LayerNorm(output_dim)
        self.norm_ff = nn.LayerNorm(output_dim)

        self.ff = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, fused_vec: torch.Tensor, new_vec: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(fused_vec)
        k = self.k_proj(new_vec)
        _v = self.v_proj(new_vec)

        tokens = torch.stack([q, k], dim=1)  # (B,2,D)
        tokens = self.norm_q(tokens)

        attn_output, _ = self.attention(tokens, tokens, tokens)
        tokens = tokens + self.dropout(attn_output)

        ff_out = self.ff(self.norm_ff(tokens))
        tokens = tokens + self.dropout(ff_out)

        updated_fused = tokens[:, 0, :]
        return updated_fused


class AttentionFusion(nn.Module):
    """Attention fusion for multi-omics representation with pathway representation."""

    def __init__(self, omics_dim: int, pathway_dim: int, output_dim: int):
        super().__init__()
        self.omics_proj = nn.Linear(omics_dim, output_dim)
        self.pathway_proj = nn.Linear(pathway_dim, output_dim)
        self.attention = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 2),
            nn.Softmax(dim=-1),
        )

    def forward(
        self, omics_features: torch.Tensor, pathway_features: torch.Tensor
    ) -> torch.Tensor:
        omics_proj = self.omics_proj(omics_features)
        pathway_proj = self.pathway_proj(pathway_features)
        combined = torch.cat([omics_proj, pathway_proj], dim=-1)
        attn_weights = self.attention(combined)
        fused = (
            attn_weights[:, 0:1] * omics_proj + attn_weights[:, 1:2] * pathway_proj
        )
        return fused


class FlexibleCellLineRepresentationModule(nn.Module):
    """
    Implements Hierarchical Fusion:
    Level 1: Intra-category multi-head attention.
    Level 2: Inter-category iterative CrossModalAttention.
    Final: AttentionFusion with pathway representation.
    """

    def __init__(
        self,
        encoder_configs: List[EncoderConfig],
        fusion_dim: int = 512,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.output_dim = output_dim if output_dim is not None else fusion_dim
        self.encoders = nn.ModuleDict(
            {
                f"{cfg.category}_{cfg.subtype}": OmicsEncoderRegistry.build_encoder(cfg)
                for cfg in encoder_configs
            }
        )
        self.category_subtypes = {}
        for cfg in encoder_configs:
            self.category_subtypes.setdefault(cfg.category, []).append(cfg.subtype)

        # Level 1 fusion
        self.intra_fusion = nn.ModuleDict(
            {
                cat: IntraCategoryMultiHeadFusion(
                    embed_dim=fusion_dim,
                    num_heads=4,
                    dropout=0.1,
                )
                for cat, subs in self.category_subtypes.items()
                if len(subs) > 1
            }
        )

        # Level 2 / final fusion (same primitives used in reference module)
        self.cross_fuse = CrossModalAttention(
            dim_fused=fusion_dim,
            dim_new=fusion_dim,
            output_dim=fusion_dim,
            num_heads=4,
            dropout=0.1,
        )
        self.final_fusion = AttentionFusion(
            omics_dim=fusion_dim,
            pathway_dim=fusion_dim,
            output_dim=fusion_dim,
        )
        self.post_fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, self.output_dim),
        )

    def forward(self, omics_data: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        category_reps: Dict[str, torch.Tensor] = {}
        for cat, subs in self.category_subtypes.items():
            if cat not in omics_data:
                continue
            curr_subs = [
                self.encoders[f"{cat}_{s}"](omics_data[cat][s])
                for s in subs
                if s in omics_data[cat]
            ]
            if not curr_subs:
                continue
            if len(curr_subs) == 1:
                category_reps[cat] = curr_subs[0]
            else:
                category_reps[cat] = self.intra_fusion[cat](curr_subs)

        if not category_reps:
            raise ValueError("No omics data matches enabled encoders.")

        # Iterative cross-modal fusion for non-pathway categories.
        non_pathway_order = [
            cat
            for cat in self.category_subtypes.keys()
            if cat != "pathway" and cat in category_reps
        ]
        current: Optional[torch.Tensor]
        if non_pathway_order:
            current = category_reps[non_pathway_order[0]]
            for cat in non_pathway_order[1:]:
                current = self.cross_fuse(current, category_reps[cat])
        else:
            current = None

        # Dedicated final fusion with pathway representation.
        pathway_rep = category_reps.get("pathway")
        if pathway_rep is not None:
            if current is not None:
                current = self.final_fusion(current, pathway_rep)
            else:
                current = pathway_rep

        if current is None:
            raise ValueError("Unable to fuse omics data into a final representation.")
        return self.post_fusion_head(current)


# ====================================================================
# DRUG MODULE (LOCKED: GIN)
# ====================================================================


class GINLayer(nn.Module):
    """Graph Isomorphism Network layer using PyG's GINConv."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        self.gin_conv = GINConv(nn=mlp, train_eps=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.gin_conv(x, edge_index)


class DrugRepresentationModule(nn.Module):
    """
    Locked best-performing drug branch from the reference implementation:
    GIN encoder stack + pooled projection head.
    """

    def __init__(
        self,
        atom_feature_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_gnn_layers: int = 3,
    ):
        super().__init__()
        self.atom_embedding = nn.Linear(atom_feature_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList(
            [GINLayer(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)]
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(
        self, drug_feature: torch.Tensor, drug_adj: torch.Tensor, ibatch: torch.Tensor
    ) -> torch.Tensor:
        x_graph = self.atom_embedding(drug_feature)
        for gnn_layer in self.gnn_layers:
            x_graph = gnn_layer(x_graph, drug_adj)
        x_graph = gmp(x_graph, ibatch)
        x_graph = self.fc_layers(x_graph)
        return x_graph


class FingerprintDrugRepresentationModule(nn.Module):
    """
    Dense fingerprint encoder used by the native shared-input benchmark.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, drug_feature: torch.Tensor) -> torch.Tensor:
        return self.net(drug_feature)


class SmilesBPEDrugRepresentationModule(nn.Module):
    """
    BPE-SMILES transformer encoder for benchmark settings that use tokenized
    SMILES directly instead of molecular graphs.
    """

    def __init__(
        self,
        vocab_size: int = 2586,
        max_len: int = 50,
        emb_dim: int = 128,
        intermediate_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 8,
        dropout: float = 0.1,
        output_dim: int = 256,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, emb_dim)
        self.position_embeddings = nn.Embedding(max_len, emb_dim)
        self.norm = nn.LayerNorm(emb_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=intermediate_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = (
            nn.Identity()
            if emb_dim == output_dim
            else nn.Sequential(
                nn.Linear(emb_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
            )
        )

    def forward(self, drug_feature: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        token_ids, attention_mask = drug_feature
        seq_length = token_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
        emb = self.word_embeddings(token_ids) + self.position_embeddings(position_ids)
        emb = self.dropout(self.norm(emb))
        encoded = self.encoder(emb, src_key_padding_mask=(attention_mask == 0))
        cls_rep = encoded[:, 0]
        return self.proj(cls_rep)


# ====================================================================
# BRANCH FUSION
# ====================================================================


class BranchFusion(nn.Module):
    """
    Attention-based fusion for local and global graph representations.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.attn_proj = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 2),
        )

    def forward(
        self, h_local: torch.Tensor, h_global: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([h_local, h_global], dim=1)
        attn_scores = self.attn_proj(combined)
        attn_weights = F.softmax(attn_scores, dim=1)
        w_local = attn_weights[:, 0].unsqueeze(1)
        w_global = attn_weights[:, 1].unsqueeze(1)
        h_fused = w_local * h_local + w_global * h_global
        return h_fused, attn_weights


# ====================================================================
# MAIN MODEL (LOCKED ARCHITECTURE)
# ====================================================================


class FUSECDR(nn.Module):
    """
    Locked flexible FUSE-CDR (single architecture only):
    - Drug branch: GIN
    - Cell branch: FlexibleCellLineRepresentationModule (unchanged)
    - Local branch: HeteroConv + SAGEConv
    - Global branch: HGTConv
    - Fusion: BranchFusion
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
        num_layers: int = 2,
        heads: int = 4,
        drug_num_gnn_layers: int = 3,
        drug_encoder_type: str = "graph",
        drug_input_dim: Optional[int] = None,
        num_local_layers: Optional[int] = None,
        num_global_layers: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drug_encoder_type = drug_encoder_type

        local_layer_count = num_layers if num_local_layers is None else num_local_layers
        global_layer_count = num_layers if num_global_layers is None else num_global_layers
        if local_layer_count < 0 or global_layer_count < 0:
            raise ValueError("FUSECDR graph depths cannot be negative")
        if local_layer_count == 0 and global_layer_count == 0:
            raise ValueError("FUSECDR requires at least one graph branch")
        self.num_local_layers = int(local_layer_count)
        self.num_global_layers = int(global_layer_count)
        self.use_local_branch = self.num_local_layers > 0
        self.use_global_branch = self.num_global_layers > 0

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
        for _ in range(self.num_local_layers):
            conv_dict = {
                edge_type: SAGEConv(hidden_dim, hidden_dim) for edge_type in edge_types
            }
            self.local_convs.append(HeteroConv(conv_dict, aggr="sum"))

        self.global_convs = nn.ModuleList()
        for _ in range(self.num_global_layers):
            self.global_convs.append(
                HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=heads)
            )
        self.global_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(self.num_global_layers)]
        )

        self.fusion = (
            BranchFusion(hidden_dim)
            if self.use_local_branch and self.use_global_branch
            else None
        )
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

    def encode_local(
        self,
        x_dict: Dict[str, torch.Tensor],
        hetero_graph_edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not self.local_convs:
            raise RuntimeError("The local GraphSAGE branch is disabled for this model")
        h_local_dict = x_dict.copy()
        for conv in self.local_convs:
            h_local_dict = conv(h_local_dict, hetero_graph_edge_index_dict)
            h_local_dict = {
                key: self.dropout_layer(F.relu(val))
                for key, val in h_local_dict.items()
            }
        return h_local_dict

    def encode_global(
        self,
        x_dict: Dict[str, torch.Tensor],
        hetero_graph_edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not self.global_convs:
            raise RuntimeError("The global HGT branch is disabled for this model")
        h_global_dict = x_dict.copy()
        for index, conv in enumerate(self.global_convs):
            h_global_dict = conv(h_global_dict, hetero_graph_edge_index_dict)
            h_global_dict = {
                key: self.dropout_layer(F.relu(self.global_norms[index](val)))
                for key, val in h_global_dict.items()
            }
        return h_global_dict

    def fuse_branches(
        self,
        h_local_dict: Dict[str, torch.Tensor],
        h_global_dict: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if self.fusion is None:
            raise RuntimeError("Branch fusion requires both graph branches")
        h_fused_dict: Dict[str, torch.Tensor] = {}
        fusion_weights: Dict[str, torch.Tensor] = {}
        for key in h_local_dict:
            fused, weights = self.fusion(h_local_dict[key], h_global_dict[key])
            h_fused_dict[key] = fused
            fusion_weights[key] = weights
        return h_fused_dict, fusion_weights

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

        h_local_dict = (
            self.encode_local(x_dict, hetero_graph_edge_index_dict)
            if self.use_local_branch
            else None
        )
        h_global_dict = (
            self.encode_global(x_dict, hetero_graph_edge_index_dict)
            if self.use_global_branch
            else None
        )
        if h_local_dict is not None and h_global_dict is not None:
            h_fused_dict, fusion_weights = self.fuse_branches(h_local_dict, h_global_dict)
        elif h_local_dict is not None:
            h_fused_dict = h_local_dict
            fusion_weights = {}
        elif h_global_dict is not None:
            h_fused_dict = h_global_dict
            fusion_weights = {}
        else:
            raise RuntimeError("FUSECDR has no active graph branch")

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
            "input_embeddings": x_dict,
            "local_embeddings": h_local_dict,
            "global_embeddings": h_global_dict,
            "fusion_weights": fusion_weights,
            "logits": logits,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(hidden_dim={self.hidden_dim}, local_layers={len(self.local_convs)}, global_layers={len(self.global_convs)})"
