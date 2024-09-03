import itertools
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import pairwise_distances

def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            
        # AllChem.Compute2DCoords(mol)
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, maxAttempts=1000)
            AllChem.UFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
        except:
            AllChem.Compute2DCoords(mol)
    except ValueError as e:
        logging.warning('the SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles, e))
        
    afm, adj, dist = featurize_mol(mol, add_dummy_node=True, one_hot_formal_charge=True)
    return afm, adj, dist


def featurize_mol(mol, add_dummy_node, one_hot_formal_charge):
    node_features = np.array([get_atom_features(atom, one_hot_formal_charge)
                              for atom in mol.GetAtoms()])

    adj_matrix = np.eye(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

    conf = mol.GetConformer()
    pos_matrix = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                           for k in range(mol.GetNumAtoms())])
    dist_matrix = pairwise_distances(pos_matrix)

    if add_dummy_node:
        m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
        m[1:, 1:] = node_features
        m[0, 0] = 1.
        node_features = m

        m = np.zeros((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
        m[1:, 1:] = adj_matrix
        adj_matrix = m

        m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1), 1e6)
        m[1:, 1:] = dist_matrix
        dist_matrix = m

    return node_features, adj_matrix, dist_matrix


def get_atom_features(atom, one_hot_formal_charge=True):
    attributes = []

    attributes += one_hot_vector(
        atom.GetAtomicNum(),
        [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
    )

    attributes += one_hot_vector(
        len(atom.GetNeighbors()),
        [0, 1, 2, 3, 4, 5]
    )

    attributes += one_hot_vector(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4]
    )

    if one_hot_formal_charge:
        attributes += one_hot_vector(
            atom.GetFormalCharge(),
            [-1, 0, 1]
        )
    else:
        attributes.append(atom.GetFormalCharge())

    attributes.append(atom.IsInRing())
    attributes.append(atom.GetIsAromatic())

    return np.array(attributes, dtype=np.float32)


def one_hot_vector(val, lst):
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)



if __name__ == '__main__':
    uni2mol = pd.read_csv('data/uniprot_molecules.tsv', sep='\t', header=0)
    comprehend = pd.read_csv('data/cleaned_uniprot_rhea.tsv', sep='\t', header=0)
    
    uni_seq_dict = {comprehend['Entry'][i]: comprehend['Sequence'][i]  for i in range(len(comprehend['Entry']))}    
    uni_mol_dict = {uni2mol['uniprot_id'][i]: uni2mol['molecules'][i]  for i in range(len(uni2mol['uniprot_id'])) if uni2mol['uniprot_id'][i] in uni_seq_dict}

    smiles = [mol.replace('*', 'C').split('.') for uni, mol in uni_mol_dict.items()]
    unique_smis = list(set(itertools.chain(*smiles)))

    uni_smi_dict = {}
    for smi in tqdm(unique_smis):
        rep = smiles_to_mol(smi)
        uni_smi_dict[smi] = {'node':torch.FloatTensor(rep[0]), 'adj':torch.FloatTensor(rep[1]) ,'dist':torch.FloatTensor(rep[2])}

    torch.save(uni_smi_dict, 'data/mol_graphs.pt')
    
