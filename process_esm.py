import esm
import torch

from tqdm import tqdm
import pandas as pd
from data_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    home_dict = './pretrained/'
    model_name = 'esm2_t33_650M_UR50D.pt'
    model, alphabet = esm.pretrained.load_model_and_alphabet(home_dict + model_name)
    #model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model = model.to(device)
    model.eval()
    
    if model_name in {'esm2_t33_650M_UR50D.pt'}:
        layer = 33
        
    elif model_name in {'esm2_t30_150M_UR50D.pt'}:
        layer = 30
    
    
    uni_seq_embedding_dict = {}
    pos_trn_mols, pos_trn_seqs, _, _ = get_samples('data/new_time/positive_train_val_time.pt', 'data/new_time/negative_train_val_time.pt')
    pos_tst_mols, pos_tst_seqs, _, _ = get_samples('data/new_time/positive_test_time.pt', 'data/new_time/negative_test_time.pt')
    unique_seqs = list(set(pos_trn_seqs + pos_tst_seqs))
    
    with torch.no_grad():
        for seq in tqdm(unique_seqs):
            if len(seq) > 5000:
                seq = seq[:5000]
                
            toks = torch.tensor(alphabet.encode(seq)).view(1, -1).to(device)
            out = model(toks, repr_layers=[33], return_contacts=False)
            uni_seq_embedding_dict[seq] = out['representations'][33].squeeze().mean(0).detach()
            
            torch.cuda.empty_cache()
            
    torch.save(uni_seq_embedding_dict, 'data/seq_embedding.pt')
