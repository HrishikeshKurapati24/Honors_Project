import os
import csv
import numpy as np
import pandas as pd
import torch
import networkx as nx
from rdkit import Chem
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import argparse

# --- Atom Feature Extraction (Ported from original GraphDRP) ---

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index

# --- Dataset Implementation ---

class GraphDRPDataset(InMemoryDataset):
    def __init__(self, root, dataset_name, xd=None, xt=None, y=None, smile_graph=None, saliency_map=False):
        self.dataset_name = dataset_name
        self.saliency_map = saliency_map
        super(GraphDRPDataset, self).__init__(root)
        if os.path.isfile(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process_data(xd, xt, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{self.dataset_name}.pt']

    def process_data(self, xd, xt, y, smile_graph):
        data_list = []
        for i in range(len(xd)):
            smiles = xd[i]
            target = xt[i]
            label = y[i]
            
            res = smile_graph.get(smiles)
            if res is None: continue
            c_size, features, edge_index = res
            
            data = Data(x=torch.Tensor(features),
                        edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                        y=torch.FloatTensor([label]))
            
            # target is the cell-line feature vector
            data.target = torch.tensor([target], dtype=torch.float, requires_grad=self.saliency_map)
            data.c_size = torch.LongTensor([c_size])
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# --- Preprocessing Logic ---

def run_preprocessing(data_dir, output_pt_dir):
    print(f"Loading formatted data from {data_dir}...")
    
    # 1. Load Response Pairs
    response_df = pd.read_csv(os.path.join(data_dir, "response_pairs.csv"))
    
    # 2. Load Mutation Matrix
    mut_df = pd.read_csv(os.path.join(data_dir, "genomics_mutation.csv"), index_col=0)
    
    # 3. Load SMILES
    smiles_df = pd.read_csv(os.path.join(data_dir, "drug_smiles.csv"))
    smiles_dict = dict(zip(smiles_df['drug_name'], smiles_df['smiles']))
    
    # Pre-calculate unique drug graphs
    print("Generating molecular graphs...")
    unique_smiles = smiles_df['smiles'].unique()
    graph_dict = {}
    for s in unique_smiles:
        g = smile_to_graph(s)
        if g: graph_dict[s] = g
    
    # Prepare lists for Dataset
    xd, xt, y = [], [], []
    for _, row in response_df.iterrows():
        cell_id = int(row['cell_line_id'])
        drug_name = str(row['drug_name'])
        label = row['label']
        
        if cell_id in mut_df.index and drug_name in smiles_dict:
            smiles = smiles_dict[drug_name]
            if smiles in graph_dict:
                xd.append(smiles)
                xt.append(mut_df.loc[cell_id].values)
                y.append(label)
    
    # Shuffle and Split (Standard 80-10-10)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    
    train_size = int(0.8 * len(indices))
    val_size = int(0.1 * len(indices))
    
    splits = {
        'train': indices[:train_size],
        'val': indices[train_size:train_size+val_size],
        'test': indices[train_size+val_size:]
    }
    
    os.makedirs(output_pt_dir, exist_ok=True)
    for name, idxs in splits.items():
        print(f"Creating {name} dataset ({len(idxs)} samples)...")
        GraphDRPDataset(
            root=output_pt_dir,
            dataset_name=name,
            xd=[xd[i] for i in idxs],
            xt=[xt[i] for i in idxs],
            y=[y[i] for i in idxs],
            smile_graph=graph_dict
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to GraphDRP_formatted_dataset")
    parser.add_argument('--output_dir', type=str, default="processed_pt", help="Where to save .pt files")
    args = parser.parse_args()
    
    run_preprocessing(args.data_dir, args.output_dir)
