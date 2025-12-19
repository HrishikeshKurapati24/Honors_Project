import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool, GINConv, GCNConv, SAGEConv, GATConv
from torch_geometric.utils import to_dense_adj
from base_model.GCNConv import GCNConv as CustomGCNConv
from base_model.SGConv import SGConv
from torch.nn import Parameter, Sequential, Linear, ReLU
from utils import *
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix

EPS = 1e-15

# ====================================================================
# LAPLACIAN EIGENVECTOR POSITIONAL ENCODING FOR GRAPH TRANSFORMER
# ====================================================================

def compute_laplacian_eigenvectors(edge_index, num_nodes, k=20, device=None):
    """
    Compute Laplacian eigenvector positional encodings for a graph.
    
    Args:
        edge_index: Edge connectivity in COO format (2, num_edges)
        num_nodes: Number of nodes in the graph
        k: Number of eigenvectors to use (default: 20)
        device: Device to place tensors on (default: None, uses CPU)
    
    Returns:
        pos_enc: Positional encoding tensor of shape (num_nodes, k)
    """
    # Convert to dense adjacency matrix
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
    
    # Compute degree matrix
    degree = adj.sum(dim=1)
    
    # Compute Laplacian: L = D - A
    laplacian = torch.diag(degree) - adj
    
    # Convert to numpy for eigenvalue decomposition
    laplacian_np = laplacian.cpu().numpy()
    laplacian_sparse = csr_matrix(laplacian_np)
    
    # Compute k smallest eigenvalues and eigenvectors
    # Note: eigs returns eigenvalues in ascending order
    try:
        eigenvalues, eigenvectors = eigs(laplacian_sparse, k=min(k, num_nodes-1), which='SM')
        # Sort by eigenvalues (eigs might not return in exact order)
        idx = np.argsort(eigenvalues.real)
        eigenvectors = eigenvectors[:, idx]
        
        # Take real part and convert to tensor
        pos_enc = torch.from_numpy(eigenvectors.real).float()
        
        # Move to device if specified
        if device is not None:
            pos_enc = pos_enc.to(device)
        
        # Normalize
        pos_enc = F.normalize(pos_enc, p=2, dim=0)
        
    except Exception:
        # Fallback: use identity if computation fails
        pos_enc = torch.zeros(num_nodes, k, device=device)
    
    return pos_enc


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer layer with Laplacian eigenvector positional encoding.
    Processes all nodes of a graph as a sequence using self-attention.
    """
    def __init__(self, d_model: int, nhead: int = 2, dropout: float = 0.2, k_pos: int = 20):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.k_pos = k_pos
        
        # Positional encoding projection
        self.pos_encoder = nn.Linear(k_pos, d_model)
        
        # Transformer encoder layer
        self.transformer_layer = nn.TransformerEncoderLayer(
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
            x: Node features (num_nodes, d_model)
            edge_index: Edge connectivity (2, num_edges)
            batch: Batch assignment (num_nodes,)
        
        Returns:
            x: Transformed node features (num_nodes, d_model)
        """
        # Process each graph separately
        num_graphs = batch.max().item() + 1
        outputs = []
        
        for graph_id in range(num_graphs):
            # Get nodes belonging to this graph
            mask = (batch == graph_id)
            graph_nodes = x[mask]
            num_nodes_graph = graph_nodes.size(0)
            
            # Filter edges for this graph
            edge_mask = (batch[edge_index[0]] == graph_id) & (batch[edge_index[1]] == graph_id)
            graph_edge_index = edge_index[:, edge_mask]
            
            # Adjust edge indices to be local to this graph (0-indexed)
            if graph_edge_index.size(1) > 0:
                # Create mapping from global node indices to local indices
                local_node_indices = torch.where(mask)[0]
                node_mapping = torch.zeros(batch.size(0), dtype=torch.long, device=x.device)
                node_mapping[local_node_indices] = torch.arange(num_nodes_graph, device=x.device)
                
                # Map edge indices to local indices
                graph_edge_index_local = node_mapping[graph_edge_index]
            else:
                # No edges: create empty edge index
                graph_edge_index_local = torch.zeros((2, 0), dtype=torch.long, device=x.device)
            
            # Compute Laplacian eigenvectors for this graph
            if num_nodes_graph > 1 and graph_edge_index_local.size(1) > 0:
                try:
                    pos_enc = compute_laplacian_eigenvectors(
                        graph_edge_index_local, 
                        num_nodes_graph, 
                        k=self.k_pos,
                        device=x.device
                    )
                except Exception:
                    # Fallback: use zero encoding if computation fails
                    pos_enc = torch.zeros(num_nodes_graph, self.k_pos, device=x.device)
            else:
                # Single node or no edges: use zero encoding
                pos_enc = torch.zeros(num_nodes_graph, self.k_pos, device=x.device)
            
            # Project positional encoding
            pos_emb = self.pos_encoder(pos_enc)
            
            # Add positional encoding to node features
            graph_nodes = graph_nodes + pos_emb
            
            # Reshape for transformer: (1, num_nodes, d_model)
            graph_nodes = graph_nodes.unsqueeze(0)
            
            # Apply transformer (self-attention across all nodes in the graph)
            graph_nodes = self.transformer_layer(graph_nodes)
            
            # Remove batch dimension
            graph_nodes = graph_nodes.squeeze(0)
            
            outputs.append(graph_nodes)
        
        # Concatenate all graphs back
        x = torch.cat(outputs, dim=0)
        return x

# ====================================================================
# VARIATIONAL AUTOENCODER (VAE) CLASSES FOR OMICS DATA COMPRESSION
# ====================================================================

class VariationalAutoencoder(nn.Module):
    """Base VAE class for compressing omics data"""
    def __init__(self, input_dim, latent_dim=256, hidden_dims=None):
        super(VariationalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Default hidden layers if not specified
        if hidden_dims is None:
            # Progressive compression: input_dim -> 1024 -> 512 -> 256
            hidden_dims = [1024, 512]
        
        # Encoder: input_dim -> hidden_dims -> latent_dim
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
    def encode(self, x):
        """Encode input to latent space"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """Forward pass: encode and reparameterize"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class GenomicsVAE(VariationalAutoencoder):
    """VAE for genomics data compression"""
    def __init__(self, input_dim, latent_dim=64):
        # Genomics: compress to 64 dimensions
        super(GenomicsVAE, self).__init__(input_dim, latent_dim=latent_dim, hidden_dims=[128])


class EpigenomicsVAE(nn.Module):
    """VAE for epigenomics data using Conv1d architecture"""
    def __init__(self, input_channels, input_length, latent_dim=256):
        super(EpigenomicsVAE, self).__init__()
        self.input_channels = input_channels
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        # Encoder: Conv1d layers
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 512, kernel_size=64, padding=32),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Global average pooling + FC to latent space
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
    def encode(self, x):
        """Encode input to latent space"""
        # Handle both 2D and 3D inputs
        if x.dim() == 2:  # (B, L)
            x = x.unsqueeze(1)  # (B, 1, L)
        elif x.dim() == 3 and x.size(1) != self.input_channels:
            # Project channels if needed
            if x.size(1) != self.input_channels:
                x = x[:, 0, :].unsqueeze(1)  # Take first channel
        
        h = self.encoder(x)  # (B, 256, L)
        h = F.adaptive_avg_pool1d(h, output_size=1).squeeze(-1)  # (B, 256)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """Forward pass: encode and reparameterize"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class TranscriptomicsVAE(VariationalAutoencoder):
    """VAE for transcriptomics data compression"""
    def __init__(self, input_dim, latent_dim=256):
        # Transcriptomics: compress to 256 dimensions
        super(TranscriptomicsVAE, self).__init__(input_dim, latent_dim=latent_dim, hidden_dims=[1024])


class ProteomicsVAE(VariationalAutoencoder):
    """VAE for proteomics data compression"""
    def __init__(self, input_dim, latent_dim=64):
        # Proteomics: compress to 64 dimensions
        super(ProteomicsVAE, self).__init__(input_dim, latent_dim=latent_dim, hidden_dims=[128])


class MetabolomicsVAE(VariationalAutoencoder):
    """VAE for metabolomics data compression"""
    def __init__(self, input_dim, latent_dim=128):
        # Metabolomics: compress to 128 dimensions
        super(MetabolomicsVAE, self).__init__(input_dim, latent_dim=latent_dim, hidden_dims=[256])


class PathwayVAE(VariationalAutoencoder):
    """VAE for pathway data compression"""
    def __init__(self, input_dim, latent_dim=256):
        # Pathway: compress to 256 dimensions (matching fusion_dim)
        super(PathwayVAE, self).__init__(input_dim, latent_dim=latent_dim, hidden_dims=[512, 256])

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for fusing different omics data"""
    def __init__(self, dim1, dim2, output_dim):
        super(CrossModalAttention, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.output_dim = output_dim
        
        # Linear projections for each modality
        self.linear1 = nn.Linear(dim1, output_dim)
        self.linear2 = nn.Linear(dim2, output_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        
        # Final projection
        self.final_proj = nn.Linear(output_dim, output_dim)
        
    def forward(self, x1, x2):
        # Project to same dimension
        proj1 = self.linear1(x1)
        proj2 = self.linear2(x2)
        
        # Stack for attention
        combined = torch.stack([proj1, proj2], dim=1)  # [batch, 2, output_dim]
        
        # Self-attention
        attended, _ = self.attention(combined, combined, combined)
        
        # Layer norm and residual
        attended = self.norm1(attended)
        attended = attended + combined
        
        # Final projection
        output = self.final_proj(attended)
        output = self.norm2(output)
        
        # Return mean across modalities
        return output.mean(dim=1)

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
        
        # GIN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            if gnn_type == 'GIN':
                if i == 0:
                    self.gnn_layers.append(GINLayer(hidden_dim, hidden_dim))
                else:
                    self.gnn_layers.append(GINLayer(hidden_dim, hidden_dim))
            elif gnn_type == 'GCN':
                if i == 0:
                    self.gnn_layers.append(GCNLayer(hidden_dim, hidden_dim))
                else:
                    self.gnn_layers.append(GCNLayer(hidden_dim, hidden_dim))
            elif gnn_type == 'GraphSAGE':
                if i == 0:
                    self.gnn_layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
                else:
                    self.gnn_layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
            elif gnn_type == 'GAT':
                if i == 0:
                    self.gnn_layers.append(GATLayer(hidden_dim, hidden_dim))
                else:
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
            # ====================================================================
            # ENHANCED IMPLEMENTATION (when active=True)
            # ====================================================================
            # Physicochemical branch: 64 features -> hidden_dim
            self.physicochemical_mlp = nn.Sequential(
                nn.Linear(64, 128),
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
            x = F.relu(self.conv1(drug_feature, drug_adj))
            x = self.bn1(x)
            
            # GIN layer 2
            x = F.relu(self.conv2(x, drug_adj))
            x = self.bn2(x)
            
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
        
        # Apply GIN layers
        for gnn_layer in self.gnn_layers:
                x_graph = gnn_layer(x_graph, drug_adj, ibatch)
        
        # Global pooling
            x_graph = gmp(x_graph, ibatch)
        
        # ====================================================================
        # ORIGINAL FORWARD PASS (when active=False)
        # ====================================================================
        if not self.active:
            # For transformer architecture, projection already applied above
            if not self.use_transformer_drug:
                x = self.fc_layers(x_graph)
        return x
            else:
                # Already projected in transformer branch
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
    - 'VAE': VAE-based compression with attention fusion
    - 'FC': FC-based preprocessing with FC layer-based fusion
    """
    def __init__(self, genomics_dim, epigenomics_in_channels, transcriptomics_dim, proteomics_dim,
                 metabolomics_dim, pathway_dim, output_dim=100, variation='original'):
        super(CellLineRepresentationModule, self).__init__()

        self.output_dim = output_dim
        self.fusion_dim = 256
        self.variation = variation

        # Track which modalities are enabled
        self.genomics_enabled = genomics_dim > 1
        self.epigenomics_enabled = epigenomics_in_channels > 1
        self.transcriptomics_enabled = transcriptomics_dim > 1
        self.proteomics_enabled = proteomics_dim > 1
        self.metabolomics_enabled = metabolomics_dim > 1
        self.pathway_enabled = pathway_dim > 1

        # ====================================================================
        # PREPROCESSING: ORIGINAL OR VAE
        # ====================================================================
        if self.variation == 'VAE':
            # VAE-based compression
            if self.genomics_enabled:
                self.genomics_vae = GenomicsVAE(genomics_dim, latent_dim=64)
                self.genomics_to_fusion = nn.Linear(64, self.fusion_dim)
            
            if self.epigenomics_enabled:
                self.epigenomics_vae = EpigenomicsVAE(
                    input_channels=1, 
                    input_length=1000,  # placeholder, will adapt
                    latent_dim=256
                )
                self.epigenomics_to_fusion = nn.Linear(256, self.fusion_dim)
            
            if self.transcriptomics_enabled:
                self.transcriptomics_vae = TranscriptomicsVAE(transcriptomics_dim, latent_dim=256)
                self.transcriptomics_to_fusion = nn.Linear(256, self.fusion_dim)
            
            if self.proteomics_enabled:
                self.proteomics_vae = ProteomicsVAE(proteomics_dim, latent_dim=64)
                self.proteomics_to_fusion = nn.Linear(64, self.fusion_dim)
            
            if self.metabolomics_enabled:
                self.metabolomics_vae = MetabolomicsVAE(metabolomics_dim, latent_dim=128)
                self.metabolomics_to_fusion = nn.Linear(128, self.fusion_dim)
            
            if self.pathway_enabled:
                self.pathway_vae = PathwayVAE(pathway_dim, latent_dim=256)
                self.pathway_layers = nn.Sequential(
                    nn.Linear(256, self.fusion_dim),
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
            self.epi_in = epigenomics_in_channels
            self.epi_kernel = 64
            self.epi_conv1 = nn.Conv1d(self.epi_in, 1024, kernel_size=self.epi_kernel, padding=self.epi_kernel // 2)
            self.epi_conv2 = nn.Conv1d(1024, 512, kernel_size=1)
            self.epi_bn = nn.BatchNorm1d(512)
            self.epi_proj = nn.Linear(512, 256)
            self.epigenomics_to_fusion = nn.Linear(256, self.fusion_dim)
            self.epi_in_adapter = None  # lazily created

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
            self.pathway_layers = nn.Sequential(
                nn.Linear(pathway_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, self.fusion_dim),
                nn.BatchNorm1d(self.fusion_dim),
                nn.ReLU()
            )

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
        else:
            # ATTENTION-BASED FUSION (original and VAE)
            self.cross_fuse = CrossModalAttention(self.fusion_dim, self.fusion_dim, self.fusion_dim)
            self.final_fusion = AttentionFusion(self.fusion_dim, self.fusion_dim, self.fusion_dim)
        
        # Post-fusion head (shared across all variations)
        self.post_fusion_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, genomics_data=None, epigenomics_data=None, transcriptomics_data=None,
                proteomics_data=None, metabolomics_data=None, pathway_data=None):

        # ====================================================================
        # FORWARD PASS: DEPENDS ON VARIATION
        # ====================================================================
        if self.variation == 'VAE':
            # VAE-based forward pass
        reps = []
            if genomics_data is not None and self.genomics_enabled:
                g_latent, g_mu, g_logvar = self.genomics_vae(genomics_data)
                reps.append(self.genomics_to_fusion(g_latent))
            
            if epigenomics_data is not None and self.epigenomics_enabled:
                e_latent, e_mu, e_logvar = self.epigenomics_vae(epigenomics_data)
                reps.append(self.epigenomics_to_fusion(e_latent))
            
            if transcriptomics_data is not None and self.transcriptomics_enabled:
                t_latent, t_mu, t_logvar = self.transcriptomics_vae(transcriptomics_data)
                reps.append(self.transcriptomics_to_fusion(t_latent))
            
            if proteomics_data is not None and self.proteomics_enabled:
                p_latent, p_mu, p_logvar = self.proteomics_vae(proteomics_data)
                reps.append(self.proteomics_to_fusion(p_latent))
            
            if metabolomics_data is not None and self.metabolomics_enabled:
                m_latent, m_mu, m_logvar = self.metabolomics_vae(metabolomics_data)
                reps.append(self.metabolomics_to_fusion(m_latent))
            
            if len(reps) == 0:
                raise ValueError('At least one omics modality must be enabled and provided to CellLineRepresentationModule.')
            
            # Continue with fusion (same as original/attention)
            if len(reps) == 1:
                current = reps[0]
            else:
                current = reps[0]
                for nxt in reps[1:]:
                    current = self.cross_fuse(current, nxt)
            
            if pathway_data is not None and self.pathway_enabled:
                pw_latent, pw_mu, pw_logvar = self.pathway_vae(pathway_data)
                pw = self.pathway_layers(pw_latent)
                current = self.final_fusion(current, pw)
            
            out = self.post_fusion_head(current)
            return out
        
        elif self.variation == 'FC':
            # Original FC preprocessing + FC-based fusion
            reps = []
        if genomics_data is not None and self.genomics_enabled:
            g = self.genomics_net(genomics_data)
            reps.append(self.genomics_to_fusion(g))

        if epigenomics_data is not None and self.epigenomics_enabled:
                if epigenomics_data.dim() == 2:
                    e_in = epigenomics_data.unsqueeze(1)
            else:
                    e_in = epigenomics_data
            c_in = e_in.size(1)
            if c_in != self.epi_in:
                if self.epi_in_adapter is None:
                    self.epi_in_adapter = nn.Conv1d(c_in, self.epi_in, kernel_size=1)
                        # Move adapter to same device as input and register as submodule
                        device = e_in.device
                        self.epi_in_adapter = self.epi_in_adapter.to(device)
                        self.add_module('epi_in_adapter_fc', self.epi_in_adapter)  # Unique name
                e_in = self.epi_in_adapter(e_in)
                e = self.epi_conv1(e_in)
                e = self.epi_conv2(e)
            e = self.epi_bn(e)
            e = F.relu(e)
                e = F.adaptive_avg_pool1d(e, output_size=1).squeeze(-1)
                e = self.epi_proj(e)
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

        if len(reps) == 0:
            raise ValueError('At least one omics modality must be enabled and provided to CellLineRepresentationModule.')

            # FC-based fusion
            if len(reps) >= 2:
                current = reps[0]
                for nxt in reps[1:]:
                    concat = torch.cat([current, nxt], dim=-1)
                    current = self.fc_fusion_layers(concat)
            else:
                current = reps[0]
            
            if pathway_data is not None and self.pathway_enabled:
                pw = self.pathway_layers(pathway_data)
                concat_pw = torch.cat([current, pw], dim=-1)
                current = self.fc_pathway_fusion(concat_pw)
            
            out = self.post_fusion_head(current)
            return out
        
        else:
            # ORIGINAL: FC preprocessing + attention fusion
            reps = []
            if genomics_data is not None and self.genomics_enabled:
                g = self.genomics_net(genomics_data)
                reps.append(self.genomics_to_fusion(g))
            
            if epigenomics_data is not None and self.epigenomics_enabled:
                if epigenomics_data.dim() == 2:
                    e_in = epigenomics_data.unsqueeze(1)
                else:
                    e_in = epigenomics_data
                c_in = e_in.size(1)
                if c_in != self.epi_in:
                    if self.epi_in_adapter is None:
                        self.epi_in_adapter = nn.Conv1d(c_in, self.epi_in, kernel_size=1)
                    e_in = self.epi_in_adapter(e_in)
                e = self.epi_conv1(e_in)
                e = self.epi_conv2(e)
                e = self.epi_bn(e)
                e = F.relu(e)
                e = F.adaptive_avg_pool1d(e, output_size=1).squeeze(-1)
                e = self.epi_proj(e)
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
            
            if len(reps) == 0:
                raise ValueError('At least one omics modality must be enabled and provided to CellLineRepresentationModule.')
            
            # Attention-based fusion
        if len(reps) == 1:
            current = reps[0]
        else:
            current = reps[0]
            for nxt in reps[1:]:
                current = self.cross_fuse(current, nxt)

        if pathway_data is not None and self.pathway_enabled:
            pw = self.pathway_layers(pathway_data)
            current = self.final_fusion(current, pw)

        out = self.post_fusion_head(current)
        return out

class NodeRepresentation(nn.Module):
    """Updated NodeRepresentation using new representation modules and six modalities.
    Missing modalities (None) are skipped from fusion.
    """
    def __init__(self, atom_shape, genomics_dim, epigenomics_in_channels, transcriptomics_dim,
                 proteomics_dim, metabolomics_dim, pathway_dim, gnn_type, output, active=False, 
                 variation='original', use_transformer_drug=False, dropout=0.2):
        super(NodeRepresentation, self).__init__()
        torch.manual_seed(0)

        # Drug representation module
        self.drug_module = DrugRepresentationModule(
            atom_feature_dim=atom_shape,
            output_dim=output,
            gnn_type=gnn_type,
            active=active,
            use_transformer_drug=use_transformer_drug,
            dropout=dropout
        )
        self.active = active

        # Cell line representation module
        self.cell_line_module = CellLineRepresentationModule(
            genomics_dim=genomics_dim,
            epigenomics_in_channels=epigenomics_in_channels,
            transcriptomics_dim=transcriptomics_dim,
            proteomics_dim=proteomics_dim,
            metabolomics_dim=metabolomics_dim,
            pathway_dim=pathway_dim,
            output_dim=output,
            variation=variation
        )

        # Final batch normalization
        self.batch_norm = nn.BatchNorm1d(output)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, drug_feature, drug_adj, ibatch,
                genomics_data=None, epigenomics_data=None, transcriptomics_data=None,
                proteomics_data=None, metabolomics_data=None, pathway_data=None,
                physicochemical_features=None):

        # Drug representation
        if self.active:
            x_drug = self.drug_module(drug_feature, drug_adj, ibatch, 
                                     physicochemical_features=physicochemical_features)
        else:
        x_drug = self.drug_module(drug_feature, drug_adj, ibatch)

        # Cell line representation
        x_cell = self.cell_line_module(
            genomics_data=genomics_data,
            epigenomics_data=epigenomics_data,
            transcriptomics_data=transcriptomics_data,
            proteomics_data=proteomics_data,
            metabolomics_data=metabolomics_data,
            pathway_data=pathway_data
        )

        # Combine
        x_all = torch.cat((x_cell, x_drug), 0)
        x_all = self.batch_norm(x_all)
        return x_all

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.prelu1 = nn.PReLU(hidden_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.prelu1(x)
        return x

class Summary(nn.Module):
    def __init__(self, ino, inn):
        super(Summary, self).__init__()
        self.fc1 = nn.Linear(ino + inn, 1)

    def forward(self, xo, xn):
        m = self.fc1(torch.cat((xo, xn), 1))
        m = torch.tanh(torch.squeeze(m))
        m = torch.exp(m) / (torch.exp(m)).sum()
        x = torch.matmul(m, xn)
        return x

class GraphCDR(nn.Module):
    def __init__(self, hidden_channels, encoder, summary, feat, index):
        super(GraphCDR, self).__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.feat = feat
        self.index = index
        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.act = nn.Sigmoid()
        self.fc = nn.Linear(100, 10)
        self.fd = nn.Linear(100, 10)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        glorot(self.weight)
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, 
                methylation_data, edge, proteomics_data=None, metabolomics_data=None, pathway_data=None,
                physicochemical_features=None):
        
        #---CDR_graph_edge and corrupted CDR_graph_edge
        pos_edge = torch.from_numpy(edge[edge[:, 2] == 1, 0:2].T)
        neg_edge = torch.from_numpy(edge[edge[:, 2] == -1, 0:2].T)
        # Move edges to same device as drug_feature
        device = drug_feature.device
        pos_edge = pos_edge.to(device)
        neg_edge = neg_edge.to(device)
        
        #---cell+drug node attributes
        feature = self.feat(drug_feature, drug_adj, ibatch, mutation_data, gexpr_data, 
                           methylation_data, proteomics_data, metabolomics_data, pathway_data,
                           physicochemical_features=physicochemical_features)
        
        #---cell+drug embedding from the CDR graph and the corrupted CDR graph
        pos_z = self.encoder(feature, pos_edge)
        neg_z = self.encoder(feature, neg_edge)
        
        #---graph-level embedding (summary)
        summary_pos = self.summary(feature, pos_z)
        summary_neg = self.summary(feature, neg_z)
        
        #---embedding at layer l
        cellpos = pos_z[:self.index, ]
        drugpos = pos_z[self.index:, ]
        
        #---embedding at layer 0
        cellfea = self.fc(feature[:self.index, ])
        drugfea = self.fd(feature[self.index:, ])
        cellfea = torch.sigmoid(cellfea)
        drugfea = torch.sigmoid(drugfea)
        
        #---concatenate embeddings at different layers (0 and l)
        cellpos = torch.cat((cellpos, cellfea), 1)
        drugpos = torch.cat((drugpos, drugfea), 1)
        
        #---inner product
        pos_adj = torch.matmul(cellpos, drugpos.t())
        pos_adj = self.act(pos_adj)
        
        return pos_z, neg_z, summary_pos, summary_neg, pos_adj.view(-1)

    def discriminate(self, z, summary, sigmoid=True):
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)