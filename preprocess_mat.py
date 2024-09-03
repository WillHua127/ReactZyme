import logging
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import pairwise_distances

import torch
import numpy as np
import pandas as pd
from mat import make_model
from data_utils import *

model_params = {
        'd_atom': 28,
        'd_model': 1024,
        'N': 8,
        'h': 16,
        'N_dense': 1,
        'lambda_attention': 0.33, 
        'lambda_distance': 0.33,
        'leaky_relu_slope': 0.1, 
        'dense_output_nonlinearity': 'relu', 
        'distance_matrix_kernel': 'exp', 
        'dropout': 0.0,
        'aggregation_type': 'mean'
    }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def get_pretrained_mat(model_path):
    model = make_model(**model_params)
    pretrained_state_dict = torch.load(model_path)
    model_state_dict = model.state_dict()
                                       
    for name, param in pretrained_state_dict.items():
        if 'generator' in name:
             continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        model_state_dict[name].copy_(param)

    return model
    
def mol_embedder(mat, node_feats, adjacency, distance):
    batch_mask = torch.sum(torch.abs(node_feats), dim=-1) != 0
    embedding = mat.encode(node_feats, batch_mask, adjacency, distance, None).squeeze()
    return embedding

def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        AllChem.Compute2DCoords(mol)
        # try:
        #     mol = Chem.AddHs(mol)
        #     AllChem.EmbedMolecule(mol, maxAttempts=500)
        #     AllChem.UFFOptimizeMolecule(mol)
        #     mol = Chem.RemoveHs(mol)
        # except:
        #     AllChem.Compute2DCoords(mol)
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



if __name__ == "__main__":
    home_dict = './pretrained/'
    model_name = 'mat.pt'
    mat = get_pretrained_mat(home_dict + model_name).to(device)
    mat.eval()
    
    uni_mol_embedding_dict = {}
    pos_trn_mols, pos_trn_seqs, _, _ = get_samples('data/new_time/positive_train_val_time.pt', 'data/new_time/negative_train_val_time.pt')
    pos_tst_mols, pos_tst_seqs, _, _ = get_samples('data/new_time/positive_test_time.pt', 'data/new_time/negative_test_time.pt')
    unique_mols = list(set(pos_trn_mols + pos_tst_mols))
    
    with torch.no_grad():
        for smi in tqdm(unique_mols):
            smiles = smi.replace('*', 'C').split('.')
            molecules = [smiles_to_mol(i) for i in smiles]
            embeddings = [mol_embedder(mat, FloatTensor(feat).unsqueeze(0), FloatTensor(adj).unsqueeze(0), FloatTensor(dist).unsqueeze(0)) for feat, adj, dist in molecules]
            embeddings = torch.stack([i.squeeze().mean(0) for i in embeddings], dim=0).mean(0).detach()
            uni_mol_embedding_dict[smi] = embeddings
            
            torch.cuda.empty_cache()

    
    torch.save(uni_mol_embedding_dict, 'data/mol_embedding.pt')
