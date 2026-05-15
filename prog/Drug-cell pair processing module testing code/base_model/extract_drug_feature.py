#Extract Drug features through Deepchem
import os
import rdkit
import deepchem as dc
from rdkit import Chem
import hickle as hkl
'''
CanonicalSMILES = 'CC1CCCC2(C(O2)CC(OC(=O)CC(C(C(=O)C(C1O)C)(C)C)O)C(=CC3=CSC(=N3)C)C)C'
mol = Chem.MolFromSmiles(CanonicalSMILES)
Simles=Chem.MolToSmiles(mol)
'''
# Paths resolved relative to the project root
PROJECT_ROOT = "/Volumes/Work/Semester - 6/Honors/CDRP models testing"
drug_smiles_file = os.path.join(PROJECT_ROOT, "3OmicsStrictBenchmarking/dataset-1/CCLE/CCLE_smiles.csv")
save_dir = os.path.join(PROJECT_ROOT, "3OmicsStrictBenchmarking/dataset-1/CCLE/drug_graph_feat")

if not os.path.exists(drug_smiles_file):
    raise FileNotFoundError(f"Required input file not found: {drug_smiles_file}")

lines = open(drug_smiles_file).readlines()[1:] # skip header
pubchemid2smile = {item.split(',')[0].strip(): item.split(',')[1].strip() for item in lines}
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
featurizer = dc.feat.ConvMolFeaturizer()
for each in pubchemid2smile.keys():
	molecules = []
	molecules.append(Chem.MolFromSmiles(pubchemid2smile[each]))
	mol_object = featurizer.featurize(molecules)
	features = mol_object[0].get_atom_features()
	degree_list = mol_object[0].deg_list
	adj_list = mol_object[0].get_adjacency_list()
	hkl.dump([features,adj_list,degree_list],f'{save_dir}/{each}.hkl')