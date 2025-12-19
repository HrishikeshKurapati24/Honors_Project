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
drug_smiles_file='../../new_data/GDSC/Processed data/pubchem_smiles.txt'
save_dir='../../new_data/GDSC/Drug/drug_graph_feat'
pubchemid2smile = {item.split('\t')[0]:item.split('\t')[1].strip() for item in open(drug_smiles_file).readlines()}
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