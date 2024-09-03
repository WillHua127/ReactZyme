import argparse

import torch
from tqdm import tqdm
import pandas as pd
import os
import time
import json
import numpy as np
import sys
import esm

from transformers import EsmTokenizer, EsmForMaskedLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_struc_seq(foldseek,
                  path,
                  chains: list = None,
                  process_id: int = 0,
                  plddt_path: str = None,
                  plddt_threshold: float = 70.) -> dict:
    """
    
    Args:
        foldseek: Binary executable file of foldseek
        path: Path to pdb file
        chains: Chains to be extracted from pdb file. If None, all chains will be extracted.
        process_id: Process ID for temporary files. This is used for parallel processing.
        plddt_path: Path to plddt file. If None, plddt will not be used.
        plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.

    Returns:
        seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
        (seq, struc_seq, combined_seq).
    """
    assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(path), f"Pdb file not found: {path}"
    assert plddt_path is None or os.path.exists(plddt_path), f"Plddt file not found: {plddt_path}"
    
    tmp_save_path = f"get_struc_seq_{process_id}.tsv"
        
    cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    os.system(cmd)
    
    seq_dict = {}
    name = os.path.basename(path)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]
            
            # Mask low plddt
            if plddt_path is not None:
                with open(plddt_path, "r") as r:
                    plddts = np.array(json.load(r)["confidenceScore"])
                    
                    # Mask regions with plddt < threshold
                    indices = np.where(plddts < plddt_threshold)[0]
                    np_seq = np.array(list(struc_seq))
                    np_seq[indices] = "#"
                    struc_seq = "".join(np_seq)
            
            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]

            if chains is None or chain in chains:
                if chain not in seq_dict:
                    combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                    seq_dict[chain] = (seq, struc_seq, combined_seq)
        
    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    return seq_dict
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_path', type=str, default='./data/saprot_seq.pt')
    parser.add_argument('--model_path', type=str, default='./weights/SaProt_650M_PDB')
    args = parser.parse_args()
        
    
    print('loading data...')
    trn_data = torch.load('data/positive_train_val_time.pt')
    tst_data = torch.load('data/positive_test_time.pt')
    trn_data.update(tst_data)

    saprot_seq = torch.load(args.seq_path)
    esm_embeddings = torch.load('data/embedding/esm_seq_embedding.pt')


    print('loading model...')
    tokenizer = EsmTokenizer.from_pretrained(args.model_path)
    model = EsmForMaskedLM.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()

    
    print('processing feature...')
    uni_seq_embedding_dict ={}
    items = 0
    with torch.no_grad():
        for uni, comp in tqdm(trn_data.items()):
            seq = comp[1]
            if len(seq) > 5000:
                seq = seq[:5000]

            try:
                sa_seq = saprot_seq[uni]
                inputs = tokenizer(sa_seq, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                inputs["output_hidden_states"] = True
                outputs = model(**inputs)
                hidden = outputs['hidden_states'][-1][:, 1:-1, :]
                hidden = hidden.squeeze().mean(0).detach()
                uni_seq_embedding_dict[seq] = hidden

                torch.cuda.empty_cache()


            except:
                items += 1
                print(f'{uni} is invalid, loading from esm, {items} items loaded from esm...')
                uni_seq_embedding_dict[seq] = esm_embeddings[seq]


    torch.save(uni_seq_embedding_dict, 'data/saprot_seq_embedding.pt')
