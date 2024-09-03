import torch
import pandas as pd
from Levenshtein import distance

import random
from tqdm import tqdm

def get_samples(pos_path):
    pos_samples = torch.load(pos_path)
    
    pos_mols = []
    pos_seqs = []
    for unis, values in pos_samples.items():
        mol = values[0]
        seq = values[1]
        pos_mols.append(mol.replace('*', 'C'))
        pos_seqs.append(seq)
    
    assert len(pos_mols) == len(pos_seqs)
    
    return pos_mols, pos_seqs

def find_similar_sequences(target_sequence, sequence_list, n_sample=5):
    # Calculate the distance between the target sequence and each sequence in the list
    distances = [(seq, distance(target_sequence, seq)) for seq in sequence_list]
    
    # Sort the list of sequences by distance
    sorted_sequences = sorted(distances, key=lambda x: x[1])
    return [seq for seq, dist in sorted_sequences[1 :n_sample+1]]

    
if __name__ == "__main__":
    pos_trn_mols, pos_trn_seqs = get_samples('data/new_time/positive_train_val_time.pt')
    pos_tst_mols, pos_tst_seqs = get_samples('data/new_time/positive_test_time.pt')
    unique_seqs = list(set(pos_trn_seqs + pos_tst_seqs))
    unique_mols = list(set(pos_trn_mols + pos_tst_mols))

    negative_mol_dict = {}
    for mol in tqdm(unique_mols):
        negative_mol = find_similar_sequences(mol, unique_mols, n_sample=2000)
        negative_mol_dict[mol] = negative_mol
        
    torch.save(negative_mol_dict, 'data/negative_mol_dict.pt')


    
    # negative_seq_dict = {}
    # for seq in tqdm(unique_seqs):
    #     negative_seq = find_similar_sequences(seq, unique_seqs, n_sample=100)
    #     negative_seq_dict[seq] = negative_seq
        
    # torch.save(negative_seq_dict, 'data/negative_seq_dict.pt')


    
