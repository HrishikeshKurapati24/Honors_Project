import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_max_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj, add_self_loops
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix

# Set seeds for reproducibility
seed_value = 1
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


class GPDRPCellEncoder(nn.Module):
    """
    Cell-line representation module.
    
    Architecture:
    - Input layer: pathway activity vector of 1329 features
    - Three dense layers: 512, 1024, 128 with ReLU activation
    - Dropout (0.2) after the second dense layer
    - Output: 128-dimensional embedding
    """
    def __init__(self, num_features_xc: int = 1329, output_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(num_features_xc, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, pathway_features):
        """
        Args:
            pathway_features: Pathway activity scores (batch_size, num_features_xc)
        
        Returns:
            embedding: 128-dimensional embedding (batch_size, output_dim)
        """
        x = F.relu(self.fc1(pathway_features))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return x


class GPDRPDrugEncoder(nn.Module):
    """
    Drug representation module.
    
    Architecture:
    - Two GIN layers to extract local structural features
    - Batch normalization after each GIN layer
    - Graph Transformer layer capturing global dependencies using:
      * Self-attention mechanism
      * Laplacian eigenvector positional encoding
    - Global max pooling to aggregate node features
    - Fully connected layer to output 128-dimensional embedding
    """
    def __init__(self, num_features_xd: int, dim: int = 96, output_dim: int = 128, 
                 dropout: float = 0.2, nhead: int = 2, k_pos: int = 20):
        super().__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)

        # GIN layer 1
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = nn.BatchNorm1d(dim)

        # GIN layer 2
        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = nn.BatchNorm1d(dim)

        # Graph Transformer layer with Laplacian positional encoding
        self.graph_transformer = GraphTransformerLayer(
            d_model=dim,
            nhead=nhead,
            dropout=dropout,
            k_pos=k_pos
        )

        # Fully connected layer for final projection
        self.fc_proj = Linear(dim, output_dim)

    def forward(self, x, edge_index, batch):
        """
        Args:
            x: Node features (num_nodes, num_features_xd)
            edge_index: Edge connectivity (2, num_edges)
            batch: Batch assignment (num_nodes,)
        
        Returns:
            embedding: 128-dimensional embedding (num_graphs, output_dim)
        """
        # GIN layer 1
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)

        # GIN layer 2
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)

        # Graph Transformer layer (captures global dependencies with self-attention)
        x = self.graph_transformer(x, edge_index, batch)
        
        # Global max pooling to aggregate node features
        x = global_max_pool(x, batch)
        
        # Fully connected layer to output 128-dimensional embedding
        x = F.relu(self.fc_proj(x))
        x = self.dropout(x)

        return x


class GPDRP_GIN_Transformer(torch.nn.Module):
    """
    Complete GPDRP model combining drug and cell-line encoders.
    """
    def __init__(self, n_output=1, num_features_xd=78, num_features_xc=1329,
                 dim=96, output_dim=128, dropout=0.2):
        super(GPDRP_GIN_Transformer, self).__init__()
        
        # Drug encoder
        self.drug_encoder = GPDRPDrugEncoder(
            num_features_xd=num_features_xd,
            dim=dim,
            output_dim=output_dim,
            dropout=dropout
        )
        
        # Cell-line encoder
        self.cell_encoder = GPDRPCellEncoder(
            num_features_xc=num_features_xc,
            output_dim=output_dim,
            dropout=dropout
        )
        
        # Fusion and output layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        """
        Args:
            data: PyTorch Geometric data object containing:
                - x: Drug node features
                - edge_index: Drug graph edges
                - batch: Batch assignment
                - target_ge: Cell-line pathway features
        
        Returns:
            out: Prediction output
            drug_embedding: Drug embedding for analysis
        """
        # Drug graph encoding
        x, edge_index, batch = data.x, data.edge_index, data.batch
        drug_embedding = self.drug_encoder(x, edge_index, batch)
        
        # Cell-line encoding
        target_ge = data.target_ge  # Pathway activity scores
        cell_embedding = self.cell_encoder(target_ge)

        # Fusion
        xc = torch.cat((drug_embedding, cell_embedding), dim=1)
        xc = F.relu(self.fc1(xc))
        xc = self.dropout(xc)
        xc = F.relu(self.fc2(xc))
        xc = self.dropout(xc)

        # Regression output (LN(IC50))
        out = self.out(xc)
        return out, drug_embedding


class NodeRepresentationGPDRP(nn.Module):
    """
    NodeRepresentation that reuses GPDRP encoders but matches GraphCDR expectations.
    
    This module produces a feature for the cell node (from pathway_data) and the drug node (from graph),
    concatenates them into a single tensor of shape (num_cells + num_drugs, output), and returns it.
    """
    def __init__(self,
                 atom_shape: int,
                 genomics_dim: int,
                 epigenomics_in_channels: int,
                 transcriptomics_dim: int,
                 proteomics_dim: int,
                 metabolomics_dim: int,
                 pathway_dim: int,
                 gnn_type: str,
                 output: int,
                 dim: int = 96,
                 dropout: float = 0.2):
        super().__init__()
        # Drug encoder
        self.drug_encoder = GPDRPDrugEncoder(
            num_features_xd=atom_shape,
            dim=dim,
            output_dim=output,
            dropout=dropout
        )
        # Cell encoder (uses pathway modality)
        if pathway_dim is None or pathway_dim <= 0:
            raise ValueError("NodeRepresentationGPDRP requires pathway_dim > 0 and pathway_data at runtime")
        self.cell_encoder = GPDRPCellEncoder(
            num_features_xc=pathway_dim,
            output_dim=output,
            dropout=dropout
        )
        # Final batch norm to mirror GraphCDR's NodeRepresentation behavior
        self.batch_norm = nn.BatchNorm1d(output)
        
        # Initialize
        for m in self.modules():
            if isinstance(m, (nn.Linear,)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self,
                drug_feature,
                drug_adj,
                ibatch,
                genomics_data=None,
                epigenomics_data=None,
                transcriptomics_data=None,
                proteomics_data=None,
                metabolomics_data=None,
                pathway_data=None,
                physicochemical_features=None):
        if pathway_data is None:
            raise ValueError("pathway_data is required for NodeRepresentationGPDRP forward")
        
        # Encode drug graphs -> (num_drugs, output)
        x_drug = self.drug_encoder(drug_feature, drug_adj, ibatch)
        # Encode cell-line pathway -> (num_cells, output)
        x_cell = self.cell_encoder(pathway_data)
        # Concatenate (cells first, then drugs) to match GraphCDR expectations
        x_all = torch.cat((x_cell, x_drug), dim=0)
        x_all = self.batch_norm(x_all)
        return x_all