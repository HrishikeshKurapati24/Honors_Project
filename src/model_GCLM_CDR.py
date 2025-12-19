import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_max_pool as gmp, global_mean_pool, GINConv, SAGEConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, degree, to_dense_adj
from torch.nn import Parameter, Sequential, Linear, ReLU
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix


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
                        self.add_module('epi_in_adapter', self.epi_in_adapter)
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
                        # Move adapter to same device as input and register as submodule
                        device = e_in.device
                        self.epi_in_adapter = self.epi_in_adapter.to(device)
                        self.add_module('epi_in_adapter', self.epi_in_adapter)
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


# ====================================================================
# HELPER FUNCTIONS FOR FORMAT CONVERSION
# ====================================================================

def convert_pyg_to_user_format(drug_x, drug_edge_index, drug_batch):
    """
    Convert PyTorch Geometric format to user module format.
    
    Args:
        drug_x: Node features (num_nodes, atom_feature_dim)
        drug_edge_index: Edge connectivity (2, num_edges)
        drug_batch: Batch assignment (num_nodes,)
    
    Returns:
        drug_feature: Node features (same as drug_x)
        drug_adj: List of dense adjacency matrices, one per graph
        ibatch: Batch assignment (same as drug_batch)
    """
    device = drug_x.device
    num_graphs = drug_batch.max().item() + 1
    
    # Convert edge_index to dense adjacency matrices per graph
    drug_adj_list = []
    
    for graph_id in range(num_graphs):
        # Get nodes belonging to this graph
        mask = (drug_batch == graph_id)
        graph_nodes = torch.where(mask)[0]
        num_nodes_graph = len(graph_nodes)
        
        if num_nodes_graph == 0:
            continue
        
        # Filter edges for this graph
        edge_mask = (drug_batch[drug_edge_index[0]] == graph_id) & (drug_batch[drug_edge_index[1]] == graph_id)
        graph_edge_index = drug_edge_index[:, edge_mask]
        
        # Create local node mapping
        node_mapping = torch.zeros(drug_batch.size(0), dtype=torch.long, device=device)
        node_mapping[graph_nodes] = torch.arange(num_nodes_graph, device=device)
        
        # Map edge indices to local indices
        if graph_edge_index.size(1) > 0:
            local_edge_index = node_mapping[graph_edge_index]
            # Convert to dense adjacency matrix
            adj = to_dense_adj(local_edge_index, max_num_nodes=num_nodes_graph).squeeze(0)
        else:
            # No edges: create zero adjacency matrix
            adj = torch.zeros(num_nodes_graph, num_nodes_graph, device=device)
        
        drug_adj_list.append(adj)
    
    # For user modules, we need a single adjacency matrix that works with the batched format
    # Actually, the user modules expect edge_index format, not dense adj
    # So we'll keep edge_index but we need to adjust it
    # Actually, looking at DrugRepresentationModule.forward, it expects drug_adj which is edge_index
    # So we can just pass drug_edge_index directly
    drug_feature = drug_x
    drug_adj = drug_edge_index  # User modules use edge_index format
    ibatch = drug_batch
    
    return drug_feature, drug_adj, ibatch


# ====================================================================
# EXISTING GCLM-CDR CLASSES
# ====================================================================

class MultiOmicsDNNExtractor(nn.Module):
    """
    Multi-Omics DNN Feature Extraction Module.
    Extracts latent features from each omic type using 3-layer DNN.
    """
    def __init__(self, genomics_dim, epigenomics_dim, transcriptomics_dim, hidden_dim=128, dropout=0.2):
        super(MultiOmicsDNNExtractor, self).__init__()
        
        # Genomic DNN
        self.genomics_dnn = nn.Sequential(
            nn.Linear(genomics_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim)
        )
        
        # Epigenomic DNN
        self.epigenomics_dnn = nn.Sequential(
            nn.Linear(epigenomics_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim)
        )
        
        # Transcriptomic DNN
        self.transcriptomics_dnn = nn.Sequential(
            nn.Linear(transcriptomics_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim)
        )
    
    def forward(self, genomics_data, epigenomics_data, transcriptomics_data):
        """
        Args:
            genomics_data: (batch_size, genomics_dim)
            epigenomics_data: (batch_size, epigenomics_dim)
            transcriptomics_data: (batch_size, transcriptomics_dim)
        
        Returns:
            M_G: (batch_size, hidden_dim)
            M_E: (batch_size, hidden_dim)
            M_N: (batch_size, hidden_dim)
        """
        M_G = self.genomics_dnn(genomics_data)
        M_E = self.epigenomics_dnn(epigenomics_data)
        M_N = self.transcriptomics_dnn(transcriptomics_data)
        
        return M_G, M_E, M_N


class MultiOmicsNILayer(nn.Module):
    """
    Multi-Omics Neighborhood Interaction Layer.
    Captures correlations and complementarities among omics.
    M_1 = M_G ⊙ M_E, M_2 = M_G ⊙ M_N, M_3 = M_E ⊙ M_N
    M_C = [M_1 || M_2 || M_3]
    """
    def __init__(self):
        super(MultiOmicsNILayer, self).__init__()
    
    def forward(self, M_G, M_E, M_N):
        """
        Args:
            M_G: (batch_size, hidden_dim)
            M_E: (batch_size, hidden_dim)
            M_N: (batch_size, hidden_dim)
        
        Returns:
            M_C: (batch_size, hidden_dim * 3)
        """
        # Element-wise products
        M_1 = M_G * M_E  # ⊙: element-wise product
        M_2 = M_G * M_N
        M_3 = M_E * M_N
        
        # Concatenate
        M_C = torch.cat([M_1, M_2, M_3], dim=1)  # ||: concatenation
        
        return M_C


class GATDrugEncoder(nn.Module):
    """
    GAT-based Drug Encoder.
    Processes molecular graphs using Graph Attention Network.
    """
    def __init__(self, atom_feature_dim, hidden_dim=128, num_heads=2, num_layers=2, dropout=0.2):
        super(GATDrugEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # First GAT layer
        self.gat1 = GATConv(atom_feature_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        
        # Subsequent GAT layers
        if num_layers > 1:
            self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout, concat=False)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch):
        """
        Args:
            x: Node features (num_nodes, atom_feature_dim)
            edge_index: Edge connectivity (2, num_edges)
            batch: Batch assignment (num_nodes,)
        
        Returns:
            h_D: Drug embedding (batch_size, hidden_dim)
        """
        # GAT layer 1
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GAT layer 2
        if self.num_layers > 1:
            x = self.gat2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global Max Pooling
        h_D = gmp(x, batch)  # (batch_size, hidden_dim)
        
        return h_D


class GCLMGraphConstructor(nn.Module):
    """
    Graph Constructor for Graph Contrastive Learning.
    Constructs bipartite graph G_OD = (V, E) where:
    V = omics nodes + drug nodes
    E = edges representing sensitivity (1) or resistance (0)
    """
    def __init__(self):
        super(GCLMGraphConstructor, self).__init__()
    
    def forward(self, M_C_batch, h_D_batch, cell_indices, drug_indices, labels, device):
        """
        Construct bipartite graph from omics-drug pairs in the current batch.
        
        Args:
            M_C_batch: Multi-omics embeddings (batch_size, hidden_dim * 3)
            h_D_batch: Drug embeddings (batch_size, hidden_dim)
            cell_indices: Cell line indices for this batch (batch_size,)
            drug_indices: Drug indices for this batch (batch_size,)
            labels: Edge labels (batch_size,)
            device: Device to create tensors on
        
        Returns:
            graph_info: Dictionary with graph structure information
        """
        batch_size = len(cell_indices)
        
        # Get unique cell lines and drugs in this batch
        unique_cells, cell_inverse = torch.unique(cell_indices, return_inverse=True)
        unique_drugs, drug_inverse = torch.unique(drug_indices, return_inverse=True)
        
        num_omics_nodes = len(unique_cells)
        num_drug_nodes = len(unique_drugs)
        total_nodes = num_omics_nodes + num_drug_nodes
        
        # Create edge list: connect omics nodes to drug nodes based on batch pairs
        edge_list = []
        edge_labels = []
        
        for i in range(batch_size):
            omics_node = cell_inverse[i].item()
            drug_node = num_omics_nodes + drug_inverse[i].item()
            
            # Add bidirectional edges
            edge_list.append([omics_node, drug_node])
            edge_list.append([drug_node, omics_node])
            edge_labels.append(labels[i].item())
            edge_labels.append(labels[i].item())
        
        if len(edge_list) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
        
        edge_labels = torch.tensor(edge_labels, dtype=torch.float32, device=device)
        
        return {
            'edge_index': edge_index,
            'edge_labels': edge_labels,
            'num_omics_nodes': num_omics_nodes,
            'num_drug_nodes': num_drug_nodes,
            'total_nodes': total_nodes,
            'cell_inverse': cell_inverse,
            'drug_inverse': drug_inverse,
            'unique_cells': unique_cells,
            'unique_drugs': unique_drugs
        }


class GCLMEncoder(nn.Module):
    """
    1-layer GCN Encoder with PReLU activation for Graph Contrastive Learning.
    h_x^(1) = PReLU(Σ_{y∈N(x)∪{x}} (1/√(q_x q_y)) * W h_y^(0))
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(GCLMEncoder, self).__init__()
        self.gcn = GCNConv(input_dim, hidden_dim)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge connectivity (2, num_edges)
        
        Returns:
            h: Encoded node features (num_nodes, hidden_dim)
        """
        # Add self-loops for GCN
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # GCN convolution with degree normalization
        # PyTorch Geometric's GCNConv already includes degree normalization
        h = self.gcn(x, edge_index)
        
        # Apply PReLU (only place where PReLU is used)
        h = self.prelu(h)
        h = self.dropout(h)
        
        return h


class AttentiveReadout(nn.Module):
    """
    Attentive Readout Function for graph-level representation.
    a_x = exp(f_a([h_x^(0) || h_x^(1)])) / Σ_{x∈V} exp(...)
    T_OD = Σ_{x∈V} a_x h_x^(1)
    """
    def __init__(self, input_dim, hidden_dim):
        super(AttentiveReadout, self).__init__()
        # f_a: MLP to compute attention weights
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, h_0, h_1):
        """
        Args:
            h_0: Initial node features (num_nodes, input_dim)
            h_1: Encoded node features (num_nodes, hidden_dim)
        
        Returns:
            T_OD: Graph-level representation (hidden_dim,)
            attention_weights: Attention weights (num_nodes,)
        """
        # Concatenate initial and encoded features
        h_concat = torch.cat([h_0, h_1], dim=1)  # (num_nodes, input_dim + hidden_dim)
        
        # Compute attention scores
        attention_scores = self.attention_net(h_concat)  # (num_nodes, 1)
        attention_scores = attention_scores.squeeze(-1)  # (num_nodes,)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=0)  # (num_nodes,)
        
        # Weighted sum
        T_OD = torch.sum(attention_weights.unsqueeze(-1) * h_1, dim=0)  # (hidden_dim,)
        
        return T_OD, attention_weights


class GCLMCDR(nn.Module):
    """
    GCLM-CDR: Graph Contrastive Learning for Multi-omics Cancer Drug Response.
    Main model class combining all modules.
    """
    def __init__(self, genomics_dim, epigenomics_dim, transcriptomics_dim, atom_feature_dim,
                 hidden_dim=128, gat_heads=2, gat_layers=2, lambda_pos=0.3, lambda_neg=0.3,
                 dropout=0.2):
        super(GCLMCDR, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        
        # Multi-Omics DNN Extractor
        self.omics_dnn = MultiOmicsDNNExtractor(
            genomics_dim, epigenomics_dim, transcriptomics_dim,
            hidden_dim=hidden_dim, dropout=dropout
        )
        
        # Multi-Omics NI Layer
        self.ni_layer = MultiOmicsNILayer()
        
        # GAT Drug Encoder
        self.drug_encoder = GATDrugEncoder(
            atom_feature_dim, hidden_dim=hidden_dim,
            num_heads=gat_heads, num_layers=gat_layers, dropout=dropout
        )
        
        # Graph Constructor
        self.graph_constructor = GCLMGraphConstructor()
        
        # GCN Encoder (with PReLU)
        # Input: projected omics features (hidden_dim) + drug features (hidden_dim)
        # Project omics features to match drug feature dimension
        self.omics_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        # GCN input dimension is hidden_dim (after projection)
        self.gcn_encoder = GCLMEncoder(hidden_dim, hidden_dim, dropout=dropout)
        
        # Attentive Readout
        self.readout = AttentiveReadout(hidden_dim, hidden_dim)
        
        # Discriminator for contrastive loss: D(h, T) = σ(h^T W T)
        # Takes node embedding h and graph summary T, outputs scalar
        self.discriminator_W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.discriminator_sigmoid = nn.Sigmoid()
        
        # Prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, drug_x, drug_edge_index, drug_batch, genomics_data, epigenomics_data,
                transcriptomics_data, cell_indices, drug_indices, labels, all_drug_embeddings=None):
        """
        Forward pass through GCLM-CDR model.
        
        Args:
            drug_x: Drug node features (num_drug_nodes, atom_feature_dim) - can be all drugs or batch
            drug_edge_index: Drug edge connectivity (2, num_drug_edges)
            drug_batch: Drug batch assignment (num_drug_nodes,)
            genomics_data: Genomics data (batch_size, genomics_dim)
            epigenomics_data: Epigenomics data (batch_size, epigenomics_dim)
            transcriptomics_data: Transcriptomics data (batch_size, transcriptomics_dim)
            cell_indices: Cell line indices for this batch (batch_size,)
            drug_indices: Drug indices for this batch (batch_size,) - indices into all drugs
            labels: Edge labels (batch_size,)
            all_drug_embeddings: Pre-computed drug embeddings for all drugs (num_all_drugs, hidden_dim) - optional
        
        Returns:
            predictions: Predicted responses (batch_size,)
            h_O: Omics node embeddings (num_omics_nodes, hidden_dim)
            h_D: Drug node embeddings (num_drug_nodes, hidden_dim)
            T_OD: Graph-level representation (hidden_dim,)
            graph_info: Graph construction information
        """
        device = drug_x.device
        batch_size = len(cell_indices)
        
        # 1. Extract multi-omics features (DNN → NI Layer → M_C)
        M_G, M_E, M_N = self.omics_dnn(genomics_data, epigenomics_data, transcriptomics_data)
        M_C = self.ni_layer(M_G, M_E, M_N)  # (batch_size, hidden_dim * 3)
        
        # 2. Encode drugs (GAT → h_D)
        if all_drug_embeddings is not None:
            # Use pre-computed drug embeddings
            h_D_batch = all_drug_embeddings[drug_indices]  # (batch_size, hidden_dim)
        else:
            # Encode drugs from this batch
            h_D_batch = self.drug_encoder(drug_x, drug_edge_index, drug_batch)  # (num_drugs_in_batch, hidden_dim)
            # If we have fewer drugs than indices, we need to handle this
            if len(h_D_batch) < len(drug_indices):
                # This shouldn't happen if drug_indices are valid
                raise ValueError(f"Drug batch size ({len(h_D_batch)}) < number of drug indices ({len(drug_indices)})")
        
        # 3. Construct graph G_OD
        graph_info = self.graph_constructor(
            M_C, h_D_batch, cell_indices, drug_indices, labels, device
        )
        
        # 4. Create node features for the graph
        num_omics_nodes = graph_info['num_omics_nodes']
        num_drug_nodes = graph_info['num_drug_nodes']
        
        # Project omics features to hidden_dim and create node features
        M_C_proj = self.omics_proj(M_C)  # (batch_size, hidden_dim)
        
        # Create combined node feature matrix
        # For omics nodes: use projected M_C, averaged by unique cell
        # For drug nodes: use h_D_batch, averaged by unique drug
        node_features = torch.zeros(
            graph_info['total_nodes'], self.hidden_dim, device=device
        )
        
        # Assign omics node features (average if same cell appears multiple times)
        cell_feature_sum = torch.zeros(num_omics_nodes, self.hidden_dim, device=device)
        cell_feature_count = torch.zeros(num_omics_nodes, device=device)
        
        for i in range(batch_size):
            cell_node_idx = graph_info['cell_inverse'][i]
            cell_feature_sum[cell_node_idx] += M_C_proj[i]
            cell_feature_count[cell_node_idx] += 1
        
        for i in range(num_omics_nodes):
            if cell_feature_count[i] > 0:
                node_features[i] = cell_feature_sum[i] / cell_feature_count[i]
        
        # Assign drug node features (average if same drug appears multiple times)
        drug_feature_sum = torch.zeros(num_drug_nodes, self.hidden_dim, device=device)
        drug_feature_count = torch.zeros(num_drug_nodes, device=device)
        
        for i in range(batch_size):
            drug_node_idx = graph_info['drug_inverse'][i]
            drug_feature_sum[drug_node_idx] += h_D_batch[i]
            drug_feature_count[drug_node_idx] += 1
        
        for i in range(num_drug_nodes):
            if drug_feature_count[i] > 0:
                node_features[num_omics_nodes + i] = drug_feature_sum[i] / drug_feature_count[i]
        
        # 5. Graph encoding (GCN → H_OD)
        h_0 = node_features  # Initial node features
        h_1 = self.gcn_encoder(node_features, graph_info['edge_index'])  # Encoded features
        
        # 6. Attentive readout (T_OD)
        T_OD, attention_weights = self.readout(h_0, h_1)
        
        # 7. Extract embeddings for prediction
        # h_O: omics node embeddings
        h_O = h_1[:num_omics_nodes]  # (num_omics_nodes, hidden_dim)
        # h_D: drug node embeddings
        h_D = h_1[num_omics_nodes:]  # (num_drug_nodes, hidden_dim)
        
        # 8. Prediction: P̂_OD = Sigmoid(Linear(h_O ⊙ h_D))
        # For each pair in the batch, predict response using learned combination
        predictions = []
        for i in range(batch_size):
            cell_node_idx = graph_info['cell_inverse'][i].item()
            drug_node_idx = graph_info['drug_inverse'][i].item()
            
            # Combine embeddings using element-wise product (Hadamard product)
            pred_input = h_O[cell_node_idx] * h_D[drug_node_idx]  # (hidden_dim,)
            pred = self.predictor(pred_input.unsqueeze(0))  # (1, hidden_dim) -> (1, 1)
            predictions.append(pred)
        
        predictions = torch.cat(predictions, dim=0).squeeze(-1)  # (batch_size,)
        
        return predictions, h_O, h_D, T_OD, graph_info
    
    def contrastive_loss(self, h, T, h_neg=None, T_neg=None):
        """
        Compute contrastive loss.
        L_pos = -1/(2|V|)[Σ log D(h_x,T) + Σ log(1 - D(fh_x,T))]
        where fh_x are corrupted/shuffled node embeddings
        L_neg = -1/(2|V|)[Σ log D(ĥ_x,Ť) + Σ log(1 - D(h_x,Ť))]
        
        Args:
            h: Positive node embeddings (num_nodes, hidden_dim)
            T: Positive graph representation (hidden_dim,)
            h_neg: Negative node embeddings (num_neg_nodes, hidden_dim)
            T_neg: Negative graph representation (hidden_dim,)
        
        Returns:
            L_pos: Positive contrastive loss
            L_neg: Negative contrastive loss
        """
        # Discriminator: D(h, T) = σ(h^T W T)
        # W is a learnable matrix, T is the graph summary
        W_T = self.discriminator_W(T)  # (hidden_dim,)
        
        # For positive pairs: D(h_x, T) = σ(h_x^T W T)
        # Compute for all nodes: h @ W_T (element-wise, then sum)
        h_W_T = torch.matmul(h, W_T.unsqueeze(-1)).squeeze(-1)  # (num_nodes,)
        D_pos = self.discriminator_sigmoid(h_W_T)  # (num_nodes,)
        
        # For corrupted nodes with positive summary: D(fh_x, T) where fh_x are shuffled/corrupted nodes
        # Create corrupted/shuffled node embeddings
        h_corrupted = h[torch.randperm(h.size(0), device=h.device)]  # Shuffle node embeddings
        h_corrupted_W_T = torch.matmul(h_corrupted, W_T.unsqueeze(-1)).squeeze(-1)  # (num_nodes,)
        D_pos_corrupted = self.discriminator_sigmoid(h_corrupted_W_T)  # (num_nodes,)
        
        # L_pos = -1/(2|V|)[Σ log D(h_x, T) + Σ log(1 - D(fh_x, T))]
        term1 = -torch.mean(torch.log(D_pos + 1e-15))
        term2 = -torch.mean(torch.log(1 - D_pos_corrupted + 1e-15))
        L_pos = (term1 + term2) / 2
        
        L_neg = torch.tensor(0.0, device=h.device)
        if h_neg is not None and T_neg is not None:
            # For negative pairs: D(ĥ_x, Ť) = σ(ĥ_x^T W Ť)
            W_T_neg = self.discriminator_W(T_neg)  # (hidden_dim,)
            h_neg_W_T_neg = torch.matmul(h_neg, W_T_neg.unsqueeze(-1)).squeeze(-1)  # (num_neg_nodes,)
            D_neg = self.discriminator_sigmoid(h_neg_W_T_neg)  # (num_neg_nodes,)
            
            # For positive nodes with negative summary: D(h_x, Ť) = σ(h_x^T W Ť)
            h_W_T_neg = torch.matmul(h, W_T_neg.unsqueeze(-1)).squeeze(-1)  # (num_nodes,)
            D_pos_neg = self.discriminator_sigmoid(h_W_T_neg)  # (num_nodes,)
            
            # L_neg = -1/(2|V|)[Σ log D(ĥ_x, Ť) + Σ log(1 - D(h_x, Ť))]
            L_neg = -torch.mean(torch.log(D_neg + 1e-15)) - torch.mean(torch.log(1 - D_pos_neg + 1e-15))
            L_neg = L_neg / 2
        
        return L_pos, L_neg
    
    def compute_loss(self, predictions, labels, h_O, h_D, T_OD, graph_info):
        """
        Compute total loss: L = (1 - λ - μ)L_pre + λL_pos + μL_neg
        
        Args:
            predictions: Predicted responses (batch_size,)
            labels: True labels (batch_size,)
            h_O: Omics node embeddings (num_omics_nodes, hidden_dim)
            h_D: Drug node embeddings (num_drug_nodes, hidden_dim)
            T_OD: Positive graph representation (hidden_dim,)
            graph_info: Graph construction information
        
        Returns:
            total_loss: Total loss
            pred_loss: Prediction loss
            pos_loss: Positive contrastive loss
            neg_loss: Negative contrastive loss
        """
        # Prediction loss: Binary cross-entropy
        pred_loss = F.binary_cross_entropy(predictions, labels)
        
        # Contrastive loss
        # Positive pairs: use actual graph
        h_all = torch.cat([h_O, h_D], dim=0)  # All node embeddings
        pos_loss, neg_loss = self.contrastive_loss(h_all, T_OD)
        
        # For negative sampling, create a negative graph by shuffling drug node features
        # This creates negative pairs for contrastive learning
        if self.training and len(h_D) > 1:
            # Create negative graph by shuffling drug node features
            h_D_neg = h_D[torch.randperm(len(h_D), device=h_D.device)]
            h_all_neg = torch.cat([h_O, h_D_neg], dim=0)
            
            # Create negative summary by using mean pooling (simpler than full readout)
            T_OD_neg = torch.mean(h_all_neg, dim=0)
            
            _, neg_loss = self.contrastive_loss(h_all, T_OD, h_all_neg, T_OD_neg)
        else:
            neg_loss = torch.tensor(0.0, device=predictions.device)
        
        # Total loss
        total_loss = (1 - self.lambda_pos - self.lambda_neg) * pred_loss + \
                     self.lambda_pos * pos_loss + \
                     self.lambda_neg * neg_loss
        
        return total_loss, pred_loss, pos_loss, neg_loss


# ====================================================================
# MODIFIABLE GCLM-CDR CLASS
# ====================================================================

class GCLMCDRModifiable(nn.Module):
    """
    Modifiable GCLM-CDR that can use custom DrugRepresentationModule and CellLineRepresentationModule
    or default GCLM-CDR modules.
    """
    def __init__(self, 
                 genomics_dim, epigenomics_dim, transcriptomics_dim, atom_feature_dim,
                 proteomics_dim=0, metabolomics_dim=0, pathway_dim=0,
                 hidden_dim=128, gat_heads=2, gat_layers=2, lambda_pos=0.3, lambda_neg=0.3,
                 dropout=0.2,
                 use_custom_modules=True,  # Flag to use custom modules
                 gnn_type='GIN'):  # For DrugRepresentationModule (default: 'GIN')
        super(GCLMCDRModifiable, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.use_custom_modules = use_custom_modules
        
        if use_custom_modules:
            # Use custom modules from model.py (copied to this file)
            # Set output_dim=hidden_dim to match GCLM-CDR expectations
            self.drug_module = DrugRepresentationModule(
                atom_feature_dim=atom_feature_dim,
                hidden_dim=256,  # Internal hidden dimension
                output_dim=hidden_dim,  # Match GCLM-CDR's hidden_dim
                num_gnn_layers=3,
                gnn_type=gnn_type,
                active=False,  # No physicochemical features
                use_transformer_drug=False,  # Default
                dropout=dropout
            )
            
            self.cell_module = CellLineRepresentationModule(
                genomics_dim=genomics_dim if genomics_dim > 1 else 1,
                epigenomics_in_channels=epigenomics_dim if epigenomics_dim > 1 else 1,
                transcriptomics_dim=transcriptomics_dim if transcriptomics_dim > 1 else 1,
                proteomics_dim=proteomics_dim if proteomics_dim > 1 else 1,
                metabolomics_dim=metabolomics_dim if metabolomics_dim > 1 else 1,
                pathway_dim=pathway_dim if pathway_dim > 1 else 1,
                output_dim=hidden_dim,  # Match GCLM-CDR's hidden_dim
                variation='original'  # Default variation
            )
        else:
            # Use default GCLM-CDR modules
            # Multi-omics DNN extractor
            self.omics_dnn = MultiOmicsDNNExtractor(
                genomics_dim=genomics_dim,
                epigenomics_dim=epigenomics_dim,
                transcriptomics_dim=transcriptomics_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            
            # Neighborhood Interaction Layer
            self.ni_layer = MultiOmicsNILayer()
            
            # Drug encoder (GAT) - using GATDrugEncoder from existing GCLM-CDR
            self.drug_encoder = GATDrugEncoder(
                atom_feature_dim=atom_feature_dim,
                hidden_dim=hidden_dim,
                num_heads=gat_heads,
                num_layers=gat_layers,
                dropout=dropout
            )
            
            # Projection for omics features (since NI layer outputs hidden_dim * 3)
            self.omics_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # Graph Constructor (shared)
        self.graph_constructor = GCLMGraphConstructor()
        
        # GCN Encoder (with PReLU)
        self.gcn_encoder = GCLMEncoder(hidden_dim, hidden_dim, dropout=dropout)
        
        # Attentive Readout
        self.readout = AttentiveReadout(hidden_dim, hidden_dim)
        
        # Discriminator for contrastive loss: D(h, T) = σ(h^T W T)
        self.discriminator_W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.discriminator_sigmoid = nn.Sigmoid()
        
        # Prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, drug_x, drug_edge_index, drug_batch, genomics_data, epigenomics_data,
                transcriptomics_data, cell_indices, drug_indices, labels, 
                proteomics_data=None, metabolomics_data=None, pathway_data=None,
                all_drug_embeddings=None, all_cell_embeddings=None):
        """
        Forward pass through modifiable GCLM-CDR model.
        
        Args:
            drug_x: Drug node features (num_drug_nodes, atom_feature_dim)
            drug_edge_index: Drug edge connectivity (2, num_drug_edges)
            drug_batch: Drug batch assignment (num_drug_nodes,)
            genomics_data: Genomics data (batch_size, genomics_dim) or (num_cells, genomics_dim) if pre-computed
            epigenomics_data: Epigenomics data (batch_size, epigenomics_dim) or (num_cells, epigenomics_dim)
            transcriptomics_data: Transcriptomics data (batch_size, transcriptomics_dim) or (num_cells, transcriptomics_dim)
            cell_indices: Cell line indices for this batch (batch_size,)
            drug_indices: Drug indices for this batch (batch_size,) - indices into all drugs
            labels: Edge labels (batch_size,)
            proteomics_data: Proteomics data (optional)
            metabolomics_data: Metabolomics data (optional)
            pathway_data: Pathway data (optional)
            all_drug_embeddings: Pre-computed drug embeddings (num_all_drugs, hidden_dim) - optional
            all_cell_embeddings: Pre-computed cell embeddings (num_all_cells, hidden_dim) - optional
        
        Returns:
            predictions: Predicted responses (batch_size,)
            h_O: Omics node embeddings (num_omics_nodes, hidden_dim)
            h_D: Drug node embeddings (num_drug_nodes, hidden_dim)
            T_OD: Graph-level representation (hidden_dim,)
            graph_info: Graph construction information
        """
        device = drug_x.device
        batch_size = len(cell_indices)
        
        if self.use_custom_modules:
            # ====================================================================
            # CUSTOM MODULES: Pre-compute all embeddings, then extract by index
            # ====================================================================
            # If pre-computed embeddings are provided, use them; otherwise compute for batch
            if all_cell_embeddings is None:
                # Compute cell embeddings for this batch
                # Extract cell data for this batch
                batch_genomics = genomics_data[cell_indices] if genomics_data is not None else None
                batch_epigenomics = epigenomics_data[cell_indices] if epigenomics_data is not None else None
                batch_transcriptomics = transcriptomics_data[cell_indices] if transcriptomics_data is not None else None
                batch_proteomics = proteomics_data[cell_indices] if proteomics_data is not None else None
                batch_metabolomics = metabolomics_data[cell_indices] if metabolomics_data is not None else None
                batch_pathway = pathway_data[cell_indices] if pathway_data is not None else None
                
                M_C_batch = self.cell_module(
                    genomics_data=batch_genomics,
                    epigenomics_data=batch_epigenomics,
                    transcriptomics_data=batch_transcriptomics,
                    proteomics_data=batch_proteomics,
                    metabolomics_data=batch_metabolomics,
                    pathway_data=batch_pathway
                )  # (batch_size, hidden_dim)
            else:
                # Extract by index from pre-computed embeddings
                M_C_batch = all_cell_embeddings[cell_indices]  # (batch_size, hidden_dim)
            
            if all_drug_embeddings is None:
                # Compute drug embeddings for all drugs in batch
                # Convert PyTorch Geometric format to user module format
                drug_feature, drug_adj, ibatch = convert_pyg_to_user_format(
                    drug_x, drug_edge_index, drug_batch
                )
                # Compute embeddings for all drugs
                all_drug_embeddings_temp = self.drug_module(drug_feature, drug_adj, ibatch)
                # Extract by drug_indices
                # drug_indices are 0-indexed within the batch, need to map to actual drug batch indices
                if len(drug_indices) > 0:
                    max_idx = drug_indices.max().item()
                    if max_idx < len(all_drug_embeddings_temp):
                        h_D_batch = all_drug_embeddings_temp[drug_indices]
                    else:
                        raise ValueError(f"drug_indices contains index {max_idx} but only {len(all_drug_embeddings_temp)} drug embeddings available")
                else:
                    raise ValueError("drug_indices cannot be empty")
            else:
                # Extract by index from pre-computed embeddings
                h_D_batch = all_drug_embeddings[drug_indices]  # (batch_size, hidden_dim)
            
        else:
            # ====================================================================
            # DEFAULT MODULES: Use existing GCLM-CDR logic
            # ====================================================================
            # 1. Extract multi-omics features (DNN → NI Layer → M_C)
            M_G, M_E, M_N = self.omics_dnn(genomics_data, epigenomics_data, transcriptomics_data)
            M_C = self.ni_layer(M_G, M_E, M_N)  # (batch_size, hidden_dim * 3)
            
            # 2. Encode drugs (GAT → h_D)
            if all_drug_embeddings is not None:
                # Use pre-computed drug embeddings
                h_D_batch = all_drug_embeddings[drug_indices]  # (batch_size, hidden_dim)
            else:
                # Encode drugs from this batch
                h_D_batch = self.drug_encoder(drug_x, drug_edge_index, drug_batch)  # (num_drugs_in_batch, hidden_dim)
                # Check if drug_indices are valid (can have repeated indices for multiple pairs with same drug)
                if len(drug_indices) > 0:
                    max_idx = drug_indices.max().item()
                    if max_idx >= len(h_D_batch):
                        raise ValueError(f"drug_indices contains index {max_idx} but only {len(h_D_batch)} drug embeddings available")
                    h_D_batch = h_D_batch[drug_indices]  # Extract embeddings for pairs in this batch
                else:
                    raise ValueError("drug_indices cannot be empty")
            
            # Project M_C to hidden_dim
            M_C_batch = self.omics_proj(M_C)  # (batch_size, hidden_dim)
        
        # 3. Construct graph G_OD (same for both custom and default)
        graph_info = self.graph_constructor(
            M_C_batch, h_D_batch, cell_indices, drug_indices, labels, device
        )
        
        # 4. Create node features for the graph
        num_omics_nodes = graph_info['num_omics_nodes']
        num_drug_nodes = graph_info['num_drug_nodes']
        
        # Create combined node feature matrix
        node_features = torch.zeros(
            graph_info['total_nodes'], self.hidden_dim, device=device
        )
        
        # Assign omics node features (average if same cell appears multiple times)
        cell_feature_sum = torch.zeros(num_omics_nodes, self.hidden_dim, device=device)
        cell_feature_count = torch.zeros(num_omics_nodes, device=device)
        
        for i in range(batch_size):
            cell_node_idx = graph_info['cell_inverse'][i]
            cell_feature_sum[cell_node_idx] += M_C_batch[i]
            cell_feature_count[cell_node_idx] += 1
        
        for i in range(num_omics_nodes):
            if cell_feature_count[i] > 0:
                node_features[i] = cell_feature_sum[i] / cell_feature_count[i]
        
        # Assign drug node features (average if same drug appears multiple times)
        drug_feature_sum = torch.zeros(num_drug_nodes, self.hidden_dim, device=device)
        drug_feature_count = torch.zeros(num_drug_nodes, device=device)
        
        for i in range(batch_size):
            drug_node_idx = graph_info['drug_inverse'][i]
            drug_feature_sum[drug_node_idx] += h_D_batch[i]
            drug_feature_count[drug_node_idx] += 1
        
        for i in range(num_drug_nodes):
            if drug_feature_count[i] > 0:
                node_features[num_omics_nodes + i] = drug_feature_sum[i] / drug_feature_count[i]
        
        # 5. Graph encoding (GCN → H_OD)
        h_0 = node_features  # Initial node features
        h_1 = self.gcn_encoder(node_features, graph_info['edge_index'])  # Encoded features
        
        # 6. Attentive readout (T_OD)
        T_OD, attention_weights = self.readout(h_0, h_1)
        
        # 7. Extract embeddings for prediction
        # h_O: omics node embeddings
        h_O = h_1[:num_omics_nodes]  # (num_omics_nodes, hidden_dim)
        # h_D: drug node embeddings
        h_D = h_1[num_omics_nodes:]  # (num_drug_nodes, hidden_dim)
        
        # 8. Prediction: P̂_OD = Sigmoid(h_O h_D^T)
        # For each pair in the batch, predict response
        predictions = []
        for i in range(batch_size):
            cell_node_idx = graph_info['cell_inverse'][i].item()
            drug_node_idx = graph_info['drug_inverse'][i].item()
            
            # Combine embeddings using element-wise product (Hadamard product)
            pred_input = h_O[cell_node_idx] * h_D[drug_node_idx]  # (hidden_dim,)
            pred = self.predictor(pred_input.unsqueeze(0))  # (1, hidden_dim) -> (1, 1)
            predictions.append(pred)
        
        predictions = torch.cat(predictions, dim=0).squeeze(-1)  # (batch_size,)
        
        return predictions, h_O, h_D, T_OD, graph_info
    
    def compute_loss(self, predictions, labels, h_O, h_D, T_OD, graph_info):
        """
        Compute total loss: L = (1 - λ - μ)L_pre + λL_pos + μL_neg
        """
        # Prediction loss: L_pre = BCE(predictions, labels)
        pred_loss = F.binary_cross_entropy(predictions, labels)
        
        # Contrastive losses
        h = torch.cat([h_O, h_D], dim=0)  # All node embeddings
        pos_loss, neg_loss = self.contrastive_loss(h, T_OD, graph_info)
        
        # Final objective: L = (1 - λ - μ)L_pre + λL_pos + μL_neg
        total_loss = (1 - self.lambda_pos - self.lambda_neg) * pred_loss + self.lambda_pos * pos_loss + self.lambda_neg * neg_loss
        
        return total_loss, pred_loss, pos_loss, neg_loss
    
    def contrastive_loss(self, h, T, graph_info=None):
        """
        Compute contrastive losses L_pos and L_neg.
        L_pos = -1/(2|V|)[Σ log D(h_x,T) + Σ log(1 - D(fh_x,T))]
        where fh_x are corrupted/shuffled node embeddings
        
        Args:
            h: Node embeddings (num_nodes, hidden_dim)
            T: Graph representation (hidden_dim,)
            graph_info: Graph construction information (unused, kept for API compatibility)
        
        Returns:
            L_pos: Positive contrastive loss
            L_neg: Negative contrastive loss
        """
        # Discriminator: D(h, T) = σ(h^T W T)
        W_T = self.discriminator_W(T)  # (hidden_dim,)
        
        # For positive pairs: D(h_x, T) = σ(h_x^T W T)
        h_W_T = torch.matmul(h, W_T.unsqueeze(-1)).squeeze(-1)  # (num_nodes,)
        D_pos = self.discriminator_sigmoid(h_W_T)  # (num_nodes,)
        
        # For corrupted nodes with positive summary: D(fh_x, T) where fh_x are shuffled/corrupted nodes
        # Create corrupted/shuffled node embeddings
        h_corrupted = h[torch.randperm(h.size(0), device=h.device)]  # Shuffle node embeddings
        h_corrupted_W_T = torch.matmul(h_corrupted, W_T.unsqueeze(-1)).squeeze(-1)  # (num_nodes,)
        D_pos_corrupted = self.discriminator_sigmoid(h_corrupted_W_T)  # (num_nodes,)
        
        # L_pos = -1/(2|V|)[Σ log D(h_x, T) + Σ log(1 - D(fh_x, T))]
        term1 = -torch.mean(torch.log(D_pos + 1e-15))
        term2 = -torch.mean(torch.log(1 - D_pos_corrupted + 1e-15))
        L_pos = (term1 + term2) / 2
        
        # For negative pairs: D(h_x, T_neg) where T_neg is corrupted graph summary
        # In GCLM-CDR, we create negative graph by shuffling node features
        h_neg = h[torch.randperm(h.size(0), device=h.device)]  # Shuffle node embeddings
        T_neg = T  # Use same graph summary (or could compute from corrupted graph)
        
        h_neg_W_T = torch.matmul(h_neg, W_T.unsqueeze(-1)).squeeze(-1)  # (num_nodes,)
        D_neg = self.discriminator_sigmoid(h_neg_W_T)  # (num_nodes,)
        
        # L_neg = -1/(2|V|) Σ log (1 - D(h_x, T_neg))
        L_neg = -torch.mean(torch.log(1 - D_neg + 1e-15))
        
        return L_pos, L_neg