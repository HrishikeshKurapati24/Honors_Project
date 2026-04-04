import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool, GINConv, GCNConv, SAGEConv, GATConv, HGTConv, HeteroConv, Linear, TransformerConv
from torch_geometric.utils import to_scipy_sparse_matrix
# Assuming sys.path is set up correctly by main.py to find base_model and my_utiils in parent dir
from torch.nn import Parameter, Sequential, Linear, ReLU
from utils import *
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh
import math

EPS = 1e-15

# ====================================================================
# LAPLACIAN EIGENVECTOR POSITIONAL ENCODING FOR GRAPH TRANSFORMER
# ====================================================================

def compute_laplacian_eigenvectors(edge_index, num_nodes, k=20, device=None):
    """
    Compute normalized Laplacian eigenvector positional encodings.
    
    Args:
        edge_index: (2, num_edges)
        num_nodes: number of nodes
        k: positional dimensions
        device: torch device
    
    Returns:
        pos_enc: (num_nodes, k)
    """
    if num_nodes <= 1:
        return torch.zeros(num_nodes, k, device=device)

    # Build sparse adjacency
    adj = to_scipy_sparse_matrix(edge_index.detach().cpu(), num_nodes=num_nodes)

    # Degree
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    d_inv_sqrt = csr_matrix(np.diag(deg_inv_sqrt))

    # Normalized Laplacian
    laplacian = eye(num_nodes, format="csr") - d_inv_sqrt @ adj @ d_inv_sqrt

    # Compute k+1 eigenvectors and drop the first trivial component
    k_eff = min(k + 1, num_nodes - 1)
    if k_eff <= 1:
        return torch.zeros(num_nodes, k, device=device)

    try:
        eigenvalues, eigenvectors = eigsh(laplacian, k=k_eff, which="SM")
    except Exception:
        return torch.zeros(num_nodes, k, device=device)

    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx]

    # Remove trivial first eigenvector
    pos_np = eigenvectors[:, 1:k + 1]
    if pos_np.shape[1] == 0:
        return torch.zeros(num_nodes, k, device=device)

    # Pad/truncate to fixed k so projection input dim remains stable
    if pos_np.shape[1] < k:
        pad = np.zeros((num_nodes, k - pos_np.shape[1]), dtype=pos_np.dtype)
        pos_np = np.concatenate([pos_np, pad], axis=1)
    elif pos_np.shape[1] > k:
        pos_np = pos_np[:, :k]

    pos_enc = torch.from_numpy(pos_np).float()
    pos_enc = F.normalize(pos_enc, p=2, dim=0)
    if device is not None:
        pos_enc = pos_enc.to(device)

    return pos_enc


class GraphTransformerLayer(nn.Module):
    """
    Global transformer block with Laplacian positional encoding.
    Local GIN aggregation remains in DrugRepresentationModule for
    --use_transformer_drug path.
    """
    def __init__(self, d_model: int, nhead: int = 2, dropout: float = 0.2, k_pos: int = 20):
        super().__init__()
        self.d_model = d_model
        self.k_pos = k_pos
        
        # Positional encoding projection
        self.pos_encoder = nn.Linear(k_pos, d_model)
        
        # Transformer encoder layer
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        
    def forward(self, x, edge_index, batch):
        """
        Args:
            x: (num_nodes, d_model)
            edge_index: (2, num_edges)
            batch: (num_nodes,)
        
        Returns:
            x: (num_nodes, d_model)
        """
        # Process each graph separately
        num_graphs = batch.max().item() + 1
        outputs = []
        
        for graph_id in range(num_graphs):
            mask = (batch == graph_id)
            graph_nodes = x[mask]
            num_nodes_graph = graph_nodes.size(0)

            if num_nodes_graph <= 1:
                outputs.append(graph_nodes)
                continue

            # Extract subgraph edges
            edge_mask = (batch[edge_index[0]] == graph_id) & (batch[edge_index[1]] == graph_id)
            graph_edge_index = edge_index[:, edge_mask]

            # Remap global indices to local graph indices
            local_indices = torch.where(mask)[0]
            node_mapping = torch.zeros(batch.size(0), dtype=torch.long, device=x.device)
            node_mapping[local_indices] = torch.arange(num_nodes_graph, device=x.device)
            graph_edge_index = node_mapping[graph_edge_index]

            # Laplacian positional encoding
            pos_enc = compute_laplacian_eigenvectors(
                graph_edge_index,
                num_nodes_graph,
                k=self.k_pos,
                device=x.device
            )
            pos_emb = self.pos_encoder(pos_enc)
            graph_nodes = graph_nodes + pos_emb

            # Transformer global attention
            graph_nodes = graph_nodes.unsqueeze(0)
            graph_nodes = self.transformer(graph_nodes)
            graph_nodes = graph_nodes.squeeze(0)

            outputs.append(graph_nodes)
        
        return torch.cat(outputs, dim=0)

# ====================================================================
# DETERMINISTIC AUTOENCODER (AE) CLASSES FOR OMICS DATA COMPRESSION
# ====================================================================

class Autoencoder(nn.Module):
    """Deterministic autoencoder for omics data compression."""
    def __init__(self, input_dim, latent_dim=256, hidden_dims=None, dropout=0.2):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [1024, 512]

        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon


def ae_loss(x, x_recon):
    return F.mse_loss(x_recon, x)


class GenomicsAE(Autoencoder):
    def __init__(self, input_dim):
        super(GenomicsAE, self).__init__(input_dim, latent_dim=64, hidden_dims=[512, 128])


class EpigenomicsAE(Autoencoder):
    def __init__(self, input_dim):
        super(EpigenomicsAE, self).__init__(input_dim, latent_dim=256, hidden_dims=[512, 256])


class TranscriptomicsAE(Autoencoder):
    def __init__(self, input_dim):
        super(TranscriptomicsAE, self).__init__(input_dim, latent_dim=256, hidden_dims=[1024, 512])


class ProteomicsAE(Autoencoder):
    def __init__(self, input_dim):
        super(ProteomicsAE, self).__init__(input_dim, latent_dim=64, hidden_dims=[256])


class MetabolomicsAE(Autoencoder):
    def __init__(self, input_dim):
        super(MetabolomicsAE, self).__init__(input_dim, latent_dim=128, hidden_dims=[512])


class PathwayAE(Autoencoder):
    def __init__(self, input_dim):
        super(PathwayAE, self).__init__(input_dim, latent_dim=32, hidden_dims=[64])

class CrossModalAttention(nn.Module):
    """
    Stable directional cross-modal attention for iterative fusion.

    fused = CrossModalAttention(fused, new_modality)

    Query  = fused representation
    Key/Value = new modality representation
    Output = updated fused representation
    """

    def __init__(self, dim_fused, dim_new, output_dim, num_heads=4, dropout=0.1):
        super(CrossModalAttention, self).__init__()

        # projections
        self.q_proj = nn.Linear(dim_fused, output_dim)
        self.k_proj = nn.Linear(dim_new, output_dim)
        self.v_proj = nn.Linear(dim_new, output_dim)

        # multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # layer norms (pre-norm transformer style)
        self.norm_q = nn.LayerNorm(output_dim)
        self.norm_ff = nn.LayerNorm(output_dim)

        # feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, fused_vec, new_vec):

        q = self.q_proj(fused_vec)
        k = self.k_proj(new_vec)
        v = self.v_proj(new_vec)

        tokens = torch.stack([q, k], dim=1)  # (B,2,D)

        tokens = self.norm_q(tokens)

        attn_output, _ = self.attention(tokens, tokens, tokens)

        tokens = tokens + self.dropout(attn_output)

        ff_out = self.ff(self.norm_ff(tokens))

        tokens = tokens + self.dropout(ff_out)

        updated_fused = tokens[:,0,:]

        return updated_fused

class AttentionFusion(nn.Module):
    """Attention mechanism for fusing multi-omics and pathway scores"""
    def __init__(self, omics_dim, pathway_dim, output_dim):
        super(AttentionFusion, self).__init__()
        self.omics_proj = nn.Linear(omics_dim, output_dim)
        self.pathway_proj = nn.Linear(pathway_dim, output_dim)
        
        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, omics_features, pathway_features):
        # Project to same dimension
        omics_proj = self.omics_proj(omics_features)
        pathway_proj = self.pathway_proj(pathway_features)
        
        # Compute attention weights
        combined = torch.cat([omics_proj, pathway_proj], dim=-1)
        attn_weights = self.attention(combined)
        
        # Weighted fusion
        fused = attn_weights[:, 0:1] * omics_proj + attn_weights[:, 1:2] * pathway_proj
        
        return fused

class GINLayer(nn.Module):
    """Graph Isomorphism Network layer using PyTorch Geometric's GINConv"""
    def __init__(self, input_dim, output_dim, epsilon=0.0):
        super(GINLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon
        
        # MLP for transformation in GIN
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
        # Create GINConv layer
        self.gin_conv = GINConv(
            nn=self.mlp,
            train_eps=True  # Allow epsilon to be learnable
        )
        
    def forward(self, x, edge_index, batch):
        # Apply GIN convolution
        out = self.gin_conv(x, edge_index)
        return out

class GCNLayer(nn.Module):
    """Graph Convolutional Network layer using PyTorch Geometric's GCNConv"""
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create GCNConv layer
        self.gcn_conv = GCNConv(
            in_channels=input_dim,
            out_channels=output_dim
        )
        
    def forward(self, x, edge_index, batch):
        # Apply GCN convolution
        out = self.gcn_conv(x, edge_index)
        return out

class GraphSAGELayer(nn.Module):
    """Graph SAGE layer using PyTorch Geometric's SAGEConv"""
    def __init__(self, input_dim, output_dim, aggr='mean'):
        super(GraphSAGELayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create SAGEConv layer
        self.sage_conv = SAGEConv(
            in_channels=input_dim,
            out_channels=output_dim,
            aggr=aggr  # 'mean', 'max', 'add' or 'lstm'
        )
        
    def forward(self, x, edge_index, batch):
        # Apply GraphSAGE convolution
        out = self.sage_conv(x, edge_index)
        return out

class GATLayer(nn.Module):
    """Graph Attention Network layer using PyTorch Geometric's GATConv"""
    def __init__(self, input_dim, output_dim, heads=8, dropout=0.0):
        super(GATLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create GATConv layer
        # Set concat=False so output_dim remains consistent with other layers
        self.gat_conv = GATConv(
            in_channels=input_dim,
            out_channels=output_dim,
            heads=heads,
            dropout=dropout,
            concat=False  # Average attention heads to maintain output_dim
        )
        
    def forward(self, x, edge_index, batch):
        # Apply GAT convolution
        out = self.gat_conv(x, edge_index)
        return out

class DrugRepresentationModule(nn.Module):
    """Drug representation module using GIN layers with optional physicochemical branch and cross-attention.
    
    If active=True:
        - Graph branch: GNN layers to process molecular graph
        - Physicochemical branch: MLP to process 64 normalized physicochemical features
        - Cross-attention: GIN embedding as query, physicochemical as key/value
    Else:
        - Uses only the graph branch (original implementation)
    
    If use_transformer_drug=True:
        - Uses GPDRP-style architecture: 2GIN layers + 1 Transformer layer
        - This architecture is more efficient and can improve performance
    """
    def __init__(self, atom_feature_dim, hidden_dim=256, output_dim=100, num_gnn_layers=3, 
                 gnn_type='GIN', active=False, use_transformer_drug=False, dropout=0.2):
        super(DrugRepresentationModule, self).__init__()
        self.atom_feature_dim = atom_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_gnn_layers = num_gnn_layers
        self.gnn_type = gnn_type
        self.active = active
        self.use_transformer_drug = use_transformer_drug
        self.dropout = dropout
        
        # ====================================================================
        # GPDRP ARCHITECTURE: 2GIN + 1GraphTransformer (with Laplacian PE)
        # ====================================================================
        if use_transformer_drug:
            # GIN layer 1
            nn1 = Sequential(Linear(atom_feature_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            self.conv1 = GINConv(nn1)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            
            # GIN layer 2
            nn2 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            self.conv2 = GINConv(nn2)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            
            # Graph Transformer layer with Laplacian positional encoding
            self.graph_transformer = GraphTransformerLayer(
                d_model=hidden_dim,
                nhead=2,
                dropout=dropout,
                k_pos=20
            )
            
            # Dropout for GPDRP
            self.dropout_layer = nn.Dropout(dropout)
        else:
            # ====================================================================
            # ORIGINAL ARCHITECTURE: Multiple GNN layers
            # ====================================================================
            # Initial projection for original architecture
            self.atom_embedding = nn.Linear(atom_feature_dim, hidden_dim)
            
            # GNN layers
            self.gnn_layers = nn.ModuleList()
            for i in range(num_gnn_layers):
                if gnn_type == 'GIN':
                    self.gnn_layers.append(GINLayer(hidden_dim, hidden_dim))
                elif gnn_type == 'GCN':
                    self.gnn_layers.append(GCNLayer(hidden_dim, hidden_dim))
                elif gnn_type == 'GraphSAGE':
                    self.gnn_layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
                elif gnn_type == 'GAT':
                    self.gnn_layers.append(GATLayer(hidden_dim, hidden_dim))
                else:
                    raise ValueError(f"Unsupported GNN type: {gnn_type}. Choose from 'GIN', 'GCN', 'GraphSAGE', 'GAT'")
        
        # ====================================================================
        # FINAL PROJECTION LAYERS
        # ====================================================================
        if use_transformer_drug:
            # For GPDRP architecture, use simpler projection similar to GPDRP
            if not self.active:
                # Original GPDRP-style projection
                self.fc_layers = nn.Sequential(
                    nn.Linear(hidden_dim, output_dim),
                    nn.ReLU()
                )
            else:
                # When active=True, we still need the projection but it will be used in fusion_projection
                self.fc_proj = nn.Linear(hidden_dim, output_dim)
        else:
            # Original implementation
            if not self.active:
                # Final FC layers
                self.fc_layers = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, output_dim),
                    nn.BatchNorm1d(output_dim),
                    nn.ReLU()
                )
        
        # ====================================================================
        # ENHANCED IMPLEMENTATION (when active=True)
        # ====================================================================
        if self.active:
            # Physicochemical branch: 64 features -> hidden_dim
            self.physicochemical_mlp = nn.Sequential(
                nn.Linear(63, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            # Cross-attention mechanism
            # Query: graph embedding, Key/Value: physicochemical embedding
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            
            # Layer normalization for residual connection
            self.norm = nn.LayerNorm(hidden_dim)
            
            # Final projection
            self.fusion_projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU()
            )
        
    def forward(self, drug_feature, drug_adj, ibatch, physicochemical_features=None):
        # ====================================================================
        # GRAPH BRANCH: GPDRP or Original Architecture
        # ====================================================================
        if self.use_transformer_drug:
            # GPDRP Architecture: 2GIN + 1GraphTransformer (with Laplacian PE)
            # GIN layer 1
            
            x = self.conv1(drug_feature, drug_adj)
            x = self.bn1(x)
            x = F.relu(x)
            
            # GIN layer 2
            x = self.conv2(x, drug_adj)
            x = self.bn2(x)
            x = F.relu(x)
            
            # Graph Transformer layer (captures global dependencies with self-attention)
            x = self.graph_transformer(x, drug_adj, ibatch)
            
            # Global max pooling to aggregate node features
            x_graph = gmp(x, ibatch)
            
            # Apply dropout (GPDRP style)
            x_graph = self.dropout_layer(x_graph)
            
            # Apply projection if not active (for transformer architecture)
            if not self.active:
                x_graph = self.fc_layers(x_graph)
        else:
            # Original Architecture: Multiple GNN layers
            # Initial embedding
            x_graph = self.atom_embedding(drug_feature)
            
            # Apply GNN layers
            for gnn_layer in self.gnn_layers:
                x_graph = gnn_layer(x_graph, drug_adj, ibatch)
                # Keep comparisons fair: GIN already includes nonlinear MLP updates,
                # so add explicit activation after other message-passing layers.
                if self.gnn_type != 'GIN':
                    x_graph = F.relu(x_graph)
            
            # Global pooling
            x_graph = gmp(x_graph, ibatch)
        
        # ====================================================================
        # ORIGINAL FORWARD PASS (when active=False)
        # ====================================================================
        if not self.active:
            # For transformer architecture, projection already applied above
            if not self.use_transformer_drug:
                x_graph = self.fc_layers(x_graph)
            return x_graph
        
        # ====================================================================
        # ENHANCED FORWARD PASS (when active=True)
        # ====================================================================
        if physicochemical_features is None:
            raise ValueError("physicochemical_features must be provided when active=True")
        
        # Physicochemical branch
        x_physicochemical = self.physicochemical_mlp(physicochemical_features)
        
        # Cross-attention: query=graph, key/value=physicochemical
        # Reshape for attention: (batch, 1, hidden_dim)
        query = x_graph.unsqueeze(1)  # (batch, 1, hidden_dim)
        key_value = x_physicochemical.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Apply cross-attention
        attended, _ = self.cross_attention(query, key_value, key_value)
        attended = attended.squeeze(1)  # (batch, hidden_dim)
        
        # Residual connection and normalization
        fused = self.norm(attended + x_graph)
        
        # Final projection
        output = self.fusion_projection(fused)
        
        return output
        
        

class CellLineRepresentationModule(nn.Module):
    """Cell line representation with modality-specific preprocessing, optional inclusion,
    iterative cross-modal fusion, and final attention fusion with pathway activity scores.

    Modalities: genomics, epigenomics, transcriptomics, proteomics, metabolomics, pathway.
    
    Supports three variations:
    - 'original': FC-based preprocessing with attention fusion (default)
    - 'AE': deterministic autoencoder-based compression with attention fusion
    - 'FC': FC-based preprocessing with FC layer-based fusion
    """
    def __init__(self, genomics_dim, epigenomics_in_channels, transcriptomics_dim, proteomics_dim,
                 metabolomics_dim, pathway_dim, output_dim=100, fusion_dim=256, variation='original', permute_omics=False):
        super(CellLineRepresentationModule, self).__init__()
        self.output_dim = output_dim
        self.fusion_dim = fusion_dim
        self.variation = variation
        self.permute_omics = permute_omics
        self.latest_recon_loss = None
        self.latest_recon_loss = None

        # Track which modalities are enabled
        self.genomics_enabled = genomics_dim >= 1
        self.epigenomics_enabled = epigenomics_in_channels >= 1
        self.transcriptomics_enabled = transcriptomics_dim >= 1
        self.proteomics_enabled = proteomics_dim >= 1
        self.metabolomics_enabled = metabolomics_dim >= 1
        self.pathway_enabled = pathway_dim >= 1

        print(f"[{self.__class__.__name__}] Initialized with variation: {self.variation}")
        print(f"  - Genomics: {'Enabled' if self.genomics_enabled else 'Disabled'} (dim={genomics_dim})")
        print(f"  - Epigenomics: {'Enabled' if self.epigenomics_enabled else 'Disabled'} (channels={epigenomics_in_channels})")
        print(f"  - Transcriptomics: {'Enabled' if self.transcriptomics_enabled else 'Disabled'} (dim={transcriptomics_dim})")
        print(f"  - Proteomics: {'Enabled' if self.proteomics_enabled else 'Disabled'} (dim={proteomics_dim})")
        print(f"  - Metabolomics: {'Enabled' if self.metabolomics_enabled else 'Disabled'} (dim={metabolomics_dim})")
        print(f"  - Pathway: {'Enabled' if self.pathway_enabled else 'Disabled'} (dim={pathway_dim})")

        # ====================================================================
        # PREPROCESSING: ORIGINAL OR AE
        # ====================================================================
        if self.variation == 'AE':
            # Autoencoder-based compression with reconstruction targets
            if self.genomics_enabled:
                self.genomics_ae = GenomicsAE(genomics_dim)
                self.genomics_to_fusion = nn.Linear(64, self.fusion_dim)
            
            if self.epigenomics_enabled:
                self.epi_in = epigenomics_in_channels
                self.epigenomics_ae = EpigenomicsAE(self.epi_in)
                self.epi_in_adapter = None
                self.epigenomics_to_fusion = nn.Linear(256, self.fusion_dim)
            
            if self.transcriptomics_enabled:
                self.transcriptomics_ae = TranscriptomicsAE(transcriptomics_dim)
                self.transcriptomics_to_fusion = nn.Identity()
            
            if self.proteomics_enabled:
                self.proteomics_ae = ProteomicsAE(proteomics_dim)
                self.proteomics_to_fusion = nn.Linear(64, self.fusion_dim)
            
            if self.metabolomics_enabled:
                self.metabolomics_ae = MetabolomicsAE(metabolomics_dim)
                self.metabolomics_to_fusion = nn.Linear(128, self.fusion_dim)
            
            if self.pathway_enabled:
                self.pathway_ae = PathwayAE(pathway_dim)
                self.pathway_layers = nn.Sequential(
                    nn.Linear(32, self.fusion_dim),
                    nn.BatchNorm1d(self.fusion_dim),
                    nn.ReLU()
                )
        else:
            # ORIGINAL FC-based preprocessing
            if self.genomics_enabled:
                self.genomics_net = nn.Sequential(
                    nn.Linear(genomics_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64)
                )
                self.genomics_to_fusion = nn.Linear(64, self.fusion_dim)

            if self.epigenomics_enabled:
                # FC-based epigenomics preprocessing (lighter than Conv1D stack).
                self.epi_in = epigenomics_in_channels
                self.epigenomics_net = nn.Sequential(
                    nn.Linear(self.epi_in, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Linear(512, 256)
                )
                self.epi_in_adapter = None  # lazily created if runtime input dim differs
                self.epigenomics_to_fusion = nn.Linear(256, self.fusion_dim)

            if self.transcriptomics_enabled:
                self.transcriptomics_net = nn.Sequential(
                    nn.Linear(transcriptomics_dim, 1024),
                    nn.ReLU(),
                    nn.BatchNorm1d(1024),
                    nn.Dropout(0.2),
                    nn.Linear(1024, 256)
                )
                self.transcriptomics_to_fusion = nn.Linear(256, self.fusion_dim)

            if self.proteomics_enabled:
                self.proteomics_net = nn.Sequential(
                    nn.Linear(proteomics_dim, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Linear(128, 64)
                )
                self.proteomics_to_fusion = nn.Linear(64, self.fusion_dim)

            if self.metabolomics_enabled:
                self.metabolomics_net = nn.Sequential(
                    nn.Linear(metabolomics_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128)
                )
                self.metabolomics_to_fusion = nn.Linear(128, self.fusion_dim)

            if self.pathway_enabled:
                self.pathway_layers = nn.Linear(pathway_dim, self.fusion_dim)

        # ====================================================================
        # FUSION: ATTENTION-BASED OR FC-BASED
        # ====================================================================
        if self.variation == 'FC':
            # FC-based fusion
            self.fc_fusion_layers = nn.Sequential(
                nn.Linear(self.fusion_dim * 2, self.fusion_dim * 2),
                nn.BatchNorm1d(self.fusion_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.fusion_dim * 2, self.fusion_dim),
                nn.BatchNorm1d(self.fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            self.fc_pathway_fusion = nn.Sequential(
                nn.Linear(self.fusion_dim * 2, self.fusion_dim * 2),
                nn.BatchNorm1d(self.fusion_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.fusion_dim * 2, self.fusion_dim)
            )
        elif self.variation in ['original', 'AE']:
            # ATTENTION-BASED FUSION (original and AE)
            self.cross_fuse = CrossModalAttention(self.fusion_dim, self.fusion_dim, self.fusion_dim)
            self.final_fusion = AttentionFusion(self.fusion_dim, self.fusion_dim, self.fusion_dim)
        else:
             raise ValueError(f"Unknown variation: {self.variation}. Choose from 'original', 'AE', 'FC'.")
        
        # Post-fusion head (shared across all variations)
        # Scales from fusion_dim to output_dim
        self.post_fusion_head = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.BatchNorm1d(self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.fusion_dim, output_dim)
        )

    def forward(self, genomics_data=None, epigenomics_data=None, transcriptomics_data=None,
                proteomics_data=None, metabolomics_data=None, pathway_data=None):

        # Verify inputs match enabled status
        if self.genomics_enabled and genomics_data is None:
            raise ValueError("Genomics is enabled but no data provided.")
        if self.epigenomics_enabled and epigenomics_data is None:
            raise ValueError("Epigenomics is enabled but no data provided.")
        if self.transcriptomics_enabled and transcriptomics_data is None:
            raise ValueError("Transcriptomics is enabled but no data provided.")
        if self.proteomics_enabled and proteomics_data is None:
            raise ValueError("Proteomics is enabled but no data provided.")
        if self.metabolomics_enabled and metabolomics_data is None:
            raise ValueError("Metabolomics is enabled but no data provided.")
        if self.pathway_enabled and pathway_data is None:
            raise ValueError("Pathway is enabled but no data provided.")

        # ====================================================================
        # FORWARD PASS: DEPENDS ON VARIATION
        # ====================================================================
        self.latest_recon_loss = None
        if self.variation == 'AE':
            # AE-based forward pass
            reps = []
            recon_losses = []
            if genomics_data is not None and self.genomics_enabled:
                g_latent, g_recon = self.genomics_ae(genomics_data)
                reps.append(self.genomics_to_fusion(g_latent))
                recon_losses.append(ae_loss(genomics_data, g_recon))
            
            if epigenomics_data is not None and self.epigenomics_enabled:
                if epigenomics_data.dim() > 2:
                    e_in = epigenomics_data.flatten(start_dim=1)
                else:
                    e_in = epigenomics_data
                c_in = e_in.size(1)
                if c_in != self.epi_in:
                    if self.epi_in_adapter is None:
                        self.epi_in_adapter = nn.Linear(c_in, self.epi_in).to(e_in.device)
                    e_in = self.epi_in_adapter(e_in)
                e_latent, e_recon = self.epigenomics_ae(e_in)
                reps.append(self.epigenomics_to_fusion(e_latent))
                recon_losses.append(ae_loss(e_in, e_recon))
            
            if transcriptomics_data is not None and self.transcriptomics_enabled:
                t_latent, t_recon = self.transcriptomics_ae(transcriptomics_data)
                reps.append(self.transcriptomics_to_fusion(t_latent))
                recon_losses.append(ae_loss(transcriptomics_data, t_recon))
            
            if proteomics_data is not None and self.proteomics_enabled:
                p_latent, p_recon = self.proteomics_ae(proteomics_data)
                reps.append(self.proteomics_to_fusion(p_latent))
                recon_losses.append(ae_loss(proteomics_data, p_recon))
            
            if metabolomics_data is not None and self.metabolomics_enabled:
                m_latent, m_recon = self.metabolomics_ae(metabolomics_data)
                reps.append(self.metabolomics_to_fusion(m_latent))
                recon_losses.append(ae_loss(metabolomics_data, m_recon))
            
            # Validate that we have some data
            if len(reps) == 0 and not (pathway_data is not None and self.pathway_enabled):
                raise ValueError('At least one omics modality or pathway must be enabled and provided to CellLineRepresentationModule.')
            
            # Start fusion with omics data if any
            if len(reps) > 0:
                if self.permute_omics and self.training:
                    random.shuffle(reps)
                current = reps[0]
                for nxt in reps[1:]:
                    current = self.cross_fuse(current, nxt)
            else:
                current = None  # Handled below
            
            # Fuse pathway data
            if pathway_data is not None and self.pathway_enabled:
                pw_latent, pw_recon = self.pathway_ae(pathway_data)
                pw = self.pathway_layers(pw_latent)
                if current is not None:
                    current = self.final_fusion(current, pw)
                else:
                    current = pw
                recon_losses.append(ae_loss(pathway_data, pw_recon))
            
            if len(recon_losses) > 0:
                self.latest_recon_loss = torch.stack(recon_losses).mean()
            
            out = self.post_fusion_head(current)
            return out
        
        elif self.variation == 'FC':
            # Original FC preprocessing + FC-based fusion
            reps = []
            if genomics_data is not None and self.genomics_enabled:
                g = self.genomics_net(genomics_data)
                reps.append(self.genomics_to_fusion(g))

            if epigenomics_data is not None and self.epigenomics_enabled:
                if epigenomics_data.dim() > 2:
                    e_in = epigenomics_data.flatten(start_dim=1)
                else:
                    e_in = epigenomics_data
                c_in = e_in.size(1)
                if c_in != self.epi_in:
                    if self.epi_in_adapter is None:
                        self.epi_in_adapter = nn.Linear(c_in, self.epi_in).to(e_in.device)
                    e_in = self.epi_in_adapter(e_in)
                e = self.epigenomics_net(e_in)
                reps.append(self.epigenomics_to_fusion(e))

            if transcriptomics_data is not None and self.transcriptomics_enabled:
                t = self.transcriptomics_net(transcriptomics_data)
                reps.append(self.transcriptomics_to_fusion(t))

            if proteomics_data is not None and self.proteomics_enabled:
                p = self.proteomics_net(proteomics_data)
                reps.append(self.proteomics_to_fusion(p))

            if metabolomics_data is not None and self.metabolomics_enabled:
                m = self.metabolomics_net(metabolomics_data)
                reps.append(self.metabolomics_to_fusion(m))

            if len(reps) == 0 and not (pathway_data is not None and self.pathway_enabled):
                raise ValueError('At least one omics modality or pathway must be enabled and provided to CellLineRepresentationModule.')

            # FC-based fusion
            if len(reps) >= 2:
                if self.permute_omics and self.training:
                    random.shuffle(reps)
                current = reps[0]
                for nxt in reps[1:]:
                    concat = torch.cat([current, nxt], dim=-1)
                    current = self.fc_fusion_layers(concat)
            elif len(reps) == 1:
                current = reps[0]
            else:
                current = None
            
            if pathway_data is not None and self.pathway_enabled:
                pw = self.pathway_layers(pathway_data)
                if current is not None:
                    concat_pw = torch.cat([current, pw], dim=-1)
                    current = self.fc_pathway_fusion(concat_pw)
                else:
                    current = pw
            
            if len(reps) > 0:
                self.latest_recon_loss = reps[0].new_zeros(())
            out = self.post_fusion_head(current)
            return out
        
        else:
            # ORIGINAL: FC preprocessing + attention fusion
            reps = []
            if genomics_data is not None and self.genomics_enabled:
                g = self.genomics_net(genomics_data)
                reps.append(self.genomics_to_fusion(g))
            
            if epigenomics_data is not None and self.epigenomics_enabled:
                if epigenomics_data.dim() > 2:
                    e_in = epigenomics_data.flatten(start_dim=1)
                else:
                    e_in = epigenomics_data
                c_in = e_in.size(1)
                if c_in != self.epi_in:
                    if self.epi_in_adapter is None:
                        self.epi_in_adapter = nn.Linear(c_in, self.epi_in).to(e_in.device)
                    e_in = self.epi_in_adapter(e_in)
                e = self.epigenomics_net(e_in)
                reps.append(self.epigenomics_to_fusion(e))
            
            if transcriptomics_data is not None and self.transcriptomics_enabled:
                t = self.transcriptomics_net(transcriptomics_data)
                reps.append(self.transcriptomics_to_fusion(t))
            
            if proteomics_data is not None and self.proteomics_enabled:
                p = self.proteomics_net(proteomics_data)
                reps.append(self.proteomics_to_fusion(p))
            
            if metabolomics_data is not None and self.metabolomics_enabled:
                m = self.metabolomics_net(metabolomics_data)
                reps.append(self.metabolomics_to_fusion(m))
            
            if len(reps) == 0 and not (pathway_data is not None and self.pathway_enabled):
                raise ValueError('At least one omics modality or pathway must be enabled and provided to CellLineRepresentationModule.')
            
            # Attention-based fusion
            if len(reps) > 0:
                if self.permute_omics and self.training:
                    random.shuffle(reps)
                current = reps[0]
                for nxt in reps[1:]:
                    current = self.cross_fuse(current, nxt)
            else:
                current = None

            if pathway_data is not None and self.pathway_enabled:
                pw = self.pathway_layers(pathway_data)
                if current is not None:
                    current = self.final_fusion(current, pw)
                else:
                    current = pw

            if len(reps) > 0:
                self.latest_recon_loss = reps[0].new_zeros(())
            out = self.post_fusion_head(current)
            return out

    def get_autoencoder_loss(self):
        return self.latest_recon_loss

################## Stage-1 Implementation: Heterogeneous Graph Dual-Branch Model ##################


class BranchFusion(nn.Module):
    """
    Attention-based fusion module to combine Local and Global representations.
    computes:
        alpha = softmax(W * [h_local || h_global])
        h_fused = alpha_local * h_local + alpha_global * h_global
    """
    def __init__(self, input_dim):
        super(BranchFusion, self).__init__()
        self.input_dim = input_dim
        
        # Project concatenated representations to attention scores
        # Input: 2 * input_dim (concatenated)
        # Output: 2 (weights for local and global)
        self.attn_proj = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 2)
        )
        
    def forward(self, h_local, h_global):
        # Stack inputs: (N, d) -> (N, 2d)
        combined = torch.cat([h_local, h_global], dim=1)
        
        # Compute attention scores
        attn_scores = self.attn_proj(combined) # (N, 2)
        attn_weights = F.softmax(attn_scores, dim=1) # (N, 2)
        
        # Split weights
        w_local = attn_weights[:, 0].unsqueeze(1) # (N, 1)
        w_global = attn_weights[:, 1].unsqueeze(1) # (N, 1)
        
        # Weighted sum
        h_fused = w_local * h_local + w_global * h_global
        
        return h_fused, attn_weights

class SOULCDR(nn.Module):
    """
    Stage-1 Main Model: Heterogeneous Drug-Cell Response Prediction.
    
    Architecture:
    1. Node Embeddings: DrugRepresentationModule + CellLineRepresentationModule
    2. Local Branch: HeteroGCN (HeteroConv with GCN/SAGE)
    3. Global Branch: Heterogeneous Graph Transformer (HGT)
    4. Fusion: Attention-based integration of Local and Global embeddings
    5. Prediction: MLP on Drug-Cell pairs
    
    No contrastive learning in Stage-1.
    """
    def __init__(self, atom_shape, genomics_dim, epigenomics_in_channels, transcriptomics_dim,
                 proteomics_dim, metabolomics_dim, pathway_dim, 
                 metadata, # Tuple (node_types, edge_types) for HGT
                 hidden_dim=256, output_dim=100, fusion_dim=256, dropout=0.2,
                 num_layers=2, heads=4, use_graph_transformer=False,
                 graph_gnn_type='homogenous', drug_gnn_type='GCN', global_gnn_type='GCN',
                 active=False, use_transformer_drug=False, variation='original', permute_omics=False):
        
        super(SOULCDR, self).__init__()
        torch.manual_seed(0)
        self.hidden_dim = hidden_dim
        self.use_graph_transformer = use_graph_transformer
        self.graph_gnn_type = graph_gnn_type
        self.global_gnn_type = global_gnn_type
        self.active = active
        self.use_transformer_drug = use_transformer_drug
        self.variation = variation
        
        # ====================================================================
        # 1. NODE EMBEDDING MODULES
        # ====================================================================
        self.drug_module = DrugRepresentationModule(
            atom_feature_dim=atom_shape,
            hidden_dim=hidden_dim, # Internal hidden dim
            output_dim=hidden_dim, # Output must match graph dimension
            gnn_type=drug_gnn_type,
            dropout=dropout,
            active=active,
            use_transformer_drug=use_transformer_drug
        )

        self.cell_line_module = CellLineRepresentationModule(
            genomics_dim=genomics_dim,
            epigenomics_in_channels=epigenomics_in_channels,
            transcriptomics_dim=transcriptomics_dim,
            proteomics_dim=proteomics_dim,
            metabolomics_dim=metabolomics_dim,
            pathway_dim=pathway_dim,
            output_dim=hidden_dim, # Output must match graph dimension
            fusion_dim=fusion_dim,
            variation=variation,
            permute_omics=permute_omics
        )
        
        # ====================================================================
        # 2. LOCAL RELATIONSHIP BRANCH
        # ====================================================================
        self.local_convs = nn.ModuleList()
        node_types, edge_types = metadata
        
        # Helper to create GNN layer based on type
        def create_gnn(g_type, in_dim, out_dim, add_loops=False):
            if g_type == 'GCN':
                return GCNConv(in_dim, out_dim, add_self_loops=add_loops)
            elif g_type == 'GAT':
                return GATConv(in_dim, out_dim, add_self_loops=add_loops)
            elif g_type == 'SAGE':
                return SAGEConv(in_dim, out_dim)
            else:
                 raise ValueError(f"Unsupported global_gnn_type: {g_type}")

        if self.graph_gnn_type == 'heterogenous':
            # Heterogeneous: Use HeteroConv
            for _ in range(num_layers):
                conv_dict = {}
                for edge_type in edge_types:
                    # Generic GNN creation
                    conv_dict[edge_type] = create_gnn(global_gnn_type, hidden_dim, hidden_dim, add_loops=False)
                self.local_convs.append(HeteroConv(conv_dict, aggr='sum'))
        elif self.graph_gnn_type == 'homogenous':
            # Homogenous: Use standard GNN on merged graph
            # Input dimension to GNN will be hidden_dim + 2 (due to one-hot node types)
            # Actually, we should probably project it back to hidden_dim or let GNN handle it?
            # Standard: First layer takes (hidden_dim + 2) -> hidden_dim. Subsequent: hidden_dim -> hidden_dim
            # But we want residual connections or simple stacking?
            # Let's assume standard stacking.
            
            # Note: If homongenous, we merge features. 
            # drug_feat: [N_d, H] + [1,0] -> [N_d, H+2]
            # cell_feat: [N_c, H] + [0,1] -> [N_c, H+2]
            
            input_dim = hidden_dim + 2
            
            for i in range(num_layers):
                 dim_in = input_dim if i == 0 else hidden_dim
                 self.local_convs.append(create_gnn(global_gnn_type, dim_in, hidden_dim, add_loops=True)) # Homogeneous data can have loops
        else:
            raise ValueError(f"Unknown graph_gnn_type: {self.graph_gnn_type}. Choose 'heterogenous' or 'homogenous'.")

        # ====================================================================
        # 3. GLOBAL RELATIONSHIP BRANCH (Graph Transformer)
        # ====================================================================
        if self.use_graph_transformer:
            self.global_convs = nn.ModuleList()
            
            if self.graph_gnn_type == 'heterogenous':
                # HGTConv: In_channels -> hidden_dim
                for _ in range(num_layers):
                    self.global_convs.append(
                        HGTConv(hidden_dim, hidden_dim, metadata, heads=heads)
                    )
            else:
                # TransformerConv: In_channels -> hidden_dim
                # Input dimension same as Local branch logic
                input_dim = hidden_dim + 2
                
                # Edge dim for 3 types of edges (one-hot encoded)
                edge_dim = 3
                
                for i in range(num_layers):
                    dim_in = input_dim if i == 0 else hidden_dim
                    self.global_convs.append(
                        TransformerConv(dim_in, hidden_dim, heads=heads, edge_dim=edge_dim, concat=False)
                    )
                
            self.global_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # ====================================================================
        # 4. FUSION & PREDICTION
        # ====================================================================
        if self.use_graph_transformer:
            self.fusion = BranchFusion(hidden_dim)
        else:
             self.fusion = None
        
        # Prediction Head: [Drug || Cell] -> Logits
        # Input dim is 2 * hidden_dim
        # Option A: use output_dim as a bottleneck layer
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, 1) # Output logits
            # Sigmoid is applied externally in loss
        )

        # PROJECTION HEAD FOR CONTRASTIVE LEARNING (Stage-2)
        # Decouples contrastive space from prediction space
        # Two-layer MLP: (2*hidden_dim) -> hidden_dim -> hidden_dim
        self.projection_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, drug_feature, drug_adj, ibatch, 
                genomics_data, epigenomics_data, transcriptomics_data,
                proteomics_data, metabolomics_data, pathway_data,
                hetero_graph_edge_index_dict, hetero_graph_edge_weight_dict=None,
                drug_indices=None, cell_indices=None, physicochemical_features=None):
        """
        Args:
            drug_feature, drug_adj, ibatch: Inputs for Drug Module
            *_data: Inputs for Cell Module
            hetero_graph_edge_index_dict: Dict of edge indices for the hetero graph
            hetero_graph_edge_weight_dict: Dict of edge weights (optional, for Local GCN)
            drug_indices: Indices of drugs in the current batch (mapping to graph nodes)
            cell_indices: Indices of cell lines in the current batch
            physicochemical_features: (N_d, 64) for enhanced drug module
        """
        
        # 1. Generate Node Embeddings
        # ---------------------------
        # Drug embeddings (N_d, hidden_dim)

        x_drug = self.drug_module(drug_feature, drug_adj, ibatch, physicochemical_features=physicochemical_features)
            
        # Cell embeddings (N_c, hidden_dim)

        x_cell = self.cell_line_module(
            genomics_data=genomics_data, epigenomics_data=epigenomics_data, transcriptomics_data=transcriptomics_data,
            proteomics_data=proteomics_data, metabolomics_data=metabolomics_data, pathway_data=pathway_data
        )
        
        # Construct Node Feature Dictionary
        x_dict = {'drug': x_drug, 'cell': x_cell}
        
        # 2. Local Branch
        # ---------------------
        h_local_dict = {}

        if self.graph_gnn_type == 'heterogenous':
            # === HETEROGENOUS PIPELINE ===
            h_local_dict = x_dict.copy()
            for i, conv in enumerate(self.local_convs):
                if hetero_graph_edge_weight_dict is not None:
                    h_local_dict = conv(h_local_dict, hetero_graph_edge_index_dict, hetero_graph_edge_weight_dict)
                else:
                    h_local_dict = conv(h_local_dict, hetero_graph_edge_index_dict)
                
                out_dict = {}
                for key, x in h_local_dict.items():
                    x = F.relu(x)
                    x = self.dropout_layer(x)
                    out_dict[key] = x
                h_local_dict = out_dict
                
        else:
            # === HOMOGENOUS PIPELINE ===
            # 1. Merge Features with One-Hot Type Encoding
            # Drug: [Feature, 1, 0], Cell: [Feature, 0, 1]

            n_d = x_drug.shape[0]
            n_c = x_cell.shape[0]
            device = x_drug.device
            
            d_type = torch.tensor([1.0, 0.0], device=device).unsqueeze(0).expand(n_d, -1)
            c_type = torch.tensor([0.0, 1.0], device=device).unsqueeze(0).expand(n_c, -1)
            
            x_drug_aug = torch.cat([x_drug, d_type], dim=1)
            x_cell_aug = torch.cat([x_cell, c_type], dim=1)
            
            x_all = torch.cat([x_drug_aug, x_cell_aug], dim=0) # (N_d+N_c, H+2)

            
            # 2. Merge Edges & Create Edge Attributes
            # Offsets: Drug=0..N_d-1, Cell=N_d..N_d+N_c-1
            edge_indices_list = []
            edge_attrs_list = []
            
            # Helper for explicit one-hot edge types: [DC, CC, DD]
            # drug-responds-cell (DC): [1, 0, 0]
            # cell-similar-cell (CC): [0, 1, 0]
            # drug-similar-drug (DD): [0, 0, 1]
            attr_dc = torch.tensor([1.0, 0.0, 0.0], device=device)
            attr_cc = torch.tensor([0.0, 1.0, 0.0], device=device)
            attr_dd = torch.tensor([0.0, 0.0, 1.0], device=device)
            
            # Helper to process edge dict
            def process_edges(src_offset, dst_offset, etype_tuple, attr_vec):
                if etype_tuple in hetero_graph_edge_index_dict:
                    index = hetero_graph_edge_index_dict[etype_tuple]
                    # Apply offsets
                    # index shape [2, E]
                    src = index[0] + src_offset
                    dst = index[1] + dst_offset
                    index_shifted = torch.stack([src, dst], dim=0)
                    edge_indices_list.append(index_shifted)
                    
                    # Attributes
                    e_count = index.shape[1]
                    edge_attrs_list.append(attr_vec.unsqueeze(0).expand(e_count, -1))

            # Process known types from metadata structure assumption
            # ('drug', 'responds_to', 'cell')
            process_edges(0, n_d, ('drug', 'responds_to', 'cell'), attr_dc)
            
            # ('cell', 'similar_to', 'cell')
            process_edges(n_d, n_d, ('cell', 'similar_to', 'cell'), attr_cc)
            
            # ('drug', 'similar_to', 'drug')
            process_edges(0, 0, ('drug', 'similar_to', 'drug'), attr_dd)
            
            if len(edge_indices_list) > 0:
                edge_index_all = torch.cat(edge_indices_list, dim=1)
                edge_attr_all = torch.cat(edge_attrs_list, dim=0)
            else:
                 # Fallback for empty graph? (Unlikely)
                 edge_index_all = torch.empty((2,0), dtype=torch.long, device=device)
                 edge_attr_all = torch.empty((0,3), device=device)
            


            # 3. Local Convolution (Homogeneous)
            h_all = x_all

            for conv in self.local_convs:
                # GCNConv/GATConv might verify dim. Our convs init with input_dim=Hidden+2
                h_all = conv(h_all, edge_index_all) # Most PyG convs allow extra args, but we pass standard
                h_all = F.relu(h_all)
                h_all = self.dropout_layer(h_all)
            
            # Store local output for fusion extraction
            h_all_local = h_all


        # 3. Global Branch & Fusion
        # -------------------------
        h_fused_dict = {}
        
        if self.use_graph_transformer:
            if self.graph_gnn_type == 'heterogenous':
                # === HETEROGENOUS GLOBAL ===
                h_global_dict = x_dict.copy()
                for i, conv in enumerate(self.global_convs):
                    h_global_dict = conv(h_global_dict, hetero_graph_edge_index_dict)
                    out_dict = {}
                    for key, x in h_global_dict.items():
                        x = self.global_norms[i](x)
                        x = F.relu(x)
                        x = self.dropout_layer(x)
                        out_dict[key] = x
                    h_global_dict = out_dict
                
                # Fusion
                for key in x_dict.keys():
                    fused, _ = self.fusion(h_local_dict[key], h_global_dict[key])
                    h_fused_dict[key] = fused
                    
            else:
                # === HOMOGENOUS GLOBAL ===
                # Use same x_all and edge structures
                # Note: TransformerConv input dim matches x_all (with type encoding) because we init it that way
                # But Wait: We transformed x_all in Local branch?
                # Usually Global branch takes *original* features (Residual) or processed?
                # HGT logic above takes `x_dict.copy()`, i.e., original embeddings.
                # So we should use `x_all` (original concatenated) not `h_all` (local output).
                h_all_global = x_all
                

                for i, conv in enumerate(self.global_convs):
                    # TransformerConv(x, edge_index, edge_attr)
                    h_all_global = conv(h_all_global, edge_index_all, edge_attr=edge_attr_all)
                    h_all_global = self.global_norms[i](h_all_global)
                    h_all_global = F.relu(h_all_global)
                    h_all_global = self.dropout_layer(h_all_global)

                
                # Split back for Fusion
                # h_all_local: [N_d+N_c, H]
                # h_all_global: [N_d+N_c, H]
                
                # Extract
                h_local_drug = h_all_local[:n_d]
                h_local_cell = h_all_local[n_d:]
                
                h_global_drug = h_all_global[:n_d]
                h_global_cell = h_all_global[n_d:]
                
                # Fusion
                fused_drug, _ = self.fusion(h_local_drug, h_global_drug)
                fused_cell, _ = self.fusion(h_local_cell, h_global_cell)
                
                h_fused_dict['drug'] = fused_drug
                h_fused_dict['cell'] = fused_cell
                
        else:
             # Just assign local results
             if self.graph_gnn_type == 'heterogenous':
                 h_fused_dict = h_local_dict
             else:
                 h_fused_dict['drug'] = h_all_local[:n_d]
                 h_fused_dict['cell'] = h_all_local[n_d:]
            
        # 5. Prediction (Drug-Cell Pairs)
        # -------------------------------
        # Extract embeddings for the batch pairs if indices provided
        logits = None
        z_pair = None
        z_drug_batch = None
        z_cell_batch = None
        
        # Always return full node embeddings separated
        # h_fused_dict['drug'] is (N_d, H), h_fused_dict['cell'] is (N_c, H)
        
        if drug_indices is not None and cell_indices is not None:
            z_drug_batch = h_fused_dict['drug'][drug_indices]
            z_cell_batch = h_fused_dict['cell'][cell_indices]
            
            # Concatenate: [Drug || Cell]
            z_pair = torch.cat([z_drug_batch, z_cell_batch], dim=1) # (B, 2*hidden_dim)
            
            # Predict logits
            logits = self.predictor(z_pair) # (B, 1)

        # 6. Output
        # ---------
        return {
            "node_embeddings": h_fused_dict,
            "drug_embeddings": h_fused_dict['drug'], # Return FULL embeddings (main.py expects this for slicing)
            "cell_embeddings": h_fused_dict['cell'], # Return FULL embeddings
            "logits": logits
        }

    def get_autoencoder_loss(self):
        """Aggregate reconstruction loss from cell line module if AE variation is used."""
        return self.cell_line_module.get_autoencoder_loss()

    def __repr__(self):
        return '{}(hidden_dim={}, layers={})'.format(self.__class__.__name__, self.hidden_dim, len(self.local_convs))