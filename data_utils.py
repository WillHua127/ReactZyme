import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from mat import make_model

def get_samples(pos_path, neg_path):
    pos_samples = torch.load(pos_path)
    neg_samples = torch.load(neg_path)
    
    pos_mols = []
    pos_seqs = []
    for unis, values in pos_samples.items():
        mol = values[0]
        seq = values[1]
        pos_mols.append(mol.replace('*', 'C'))
        pos_seqs.append(seq)
    
    
    neg_mols = []
    neg_seqs = []
    for unis, values in neg_samples.items():
        mol = values[0]
        seq = values[1]
        neg_mols.append(mol.replace('*', 'C'))
        neg_seqs.append(seq)

    assert len(pos_mols) == len(pos_seqs)
    assert len(neg_mols) == len(neg_seqs)
    
    return pos_mols, pos_seqs, neg_mols, neg_seqs
    

def collate_fn(batch):
    mols, seqs, labels = zip(*batch)
    batch_mols = pad_sequence(mols, batch_first=True, padding_value=0)
    batch_seqs = pad_sequence(seqs, batch_first=True, padding_value=1)
    batch_labels = torch.stack(labels)
    return batch_mols, batch_seqs, batch_labels
    
    
class EnzymeDataset(Dataset):
    def __init__(self, molecules, sequences, mol_tokenizer, seq_tokenizer, positive_sample=True, max_len=7000):
        assert len(molecules) == len(sequences)
        self.len = len(sequences)
        self.mols = molecules
        self.seqs = sequences
        self.mol_tokenizer = mol_tokenizer
        self.seq_tokenizer = seq_tokenizer
        self.max_len = max_len
        self.labels = torch.ones(self.len) if positive_sample else torch.zeros(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        mols = self.mols[item]
        seqs = self.seqs[item]
        labels = self.labels[item]
        
        mol_tok = self.mol_tokenizer(mols, padding=True, truncation=True, max_length=self.max_len)['input_ids']
        seq_tok = self.seq_tokenizer(seqs, padding=True, truncation=True, max_length=self.max_len)['input_ids']
        
        return torch.tensor(mol_tok), torch.tensor(seq_tok), labels
    
def collate_fn_pretrained(batch):
    mols, seqs, labels = zip(*batch)
    batch_mols = torch.stack(mols)
    batch_seqs = torch.stack(seqs)
    batch_labels = torch.stack(labels)
    return batch_mols, batch_seqs, batch_labels
    
class EnzymeDatasetPretrained(Dataset):
    def __init__(self, molecules, sequences, mol_embedding, seq_embedding, positive_sample=True, max_len=5000):
        assert len(molecules) == len(sequences)
        self.len = len(sequences)
        self.mols = molecules
        self.seqs = sequences
        self.mol_embedding = mol_embedding
        self.seq_embedding = seq_embedding
        self.max_len = max_len
        self.labels = torch.ones(self.len) if positive_sample else torch.zeros(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        mols = self.mols[item]
        seqs = self.seqs[item]
        labels = self.labels[item]
        seqs = seqs[:self.max_len] if len(seqs) > self.max_len else seqs
        
        mol_tok = self.mol_embedding[mols]
        seq_tok = self.seq_embedding[seqs]
        
        if mol_tok.dim() == 2: mol_tok = mol_tok.sum(0)
        
        return mol_tok, seq_tok, labels
    
    
def collate_fn_pretrained_single(batch):
    data, labels = zip(*batch)
    batch_data = torch.stack(data)
    batch_labels = torch.stack(labels)
    return batch_data, batch_labels
    
class EnzymeDatasetPretrainedSingle(Dataset):
    def __init__(self, data, embedding, positive_sample=True, max_len=5000):
        self.len = len(data)
        self.data = data
        self.embedding = embedding
        self.max_len = max_len
        self.labels = torch.ones(self.len) if positive_sample else torch.zeros(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        data = self.data[item]
        labels = self.labels[item]
        data = data[:self.max_len] if len(data) > self.max_len else data
        
        emb = self.embedding[data]
        
        if emb.dim() == 2: emb = emb.sum(0)
        
        return emb, labels
    
    
def graph_collate_fn(batch):
    mols, seqs, labels = zip(*batch)
    return mols, seqs, labels
    

class GraphEnzymeDataset(Dataset):
    def __init__(self, molecules, sequences, alphabet, mol_graphs_dict, positive_sample=True, max_len=7000):
        assert len(molecules) == len(sequences)
        self.len = len(sequences)
        self.mols = molecules
        self.seqs = sequences
        self.alphabet = alphabet
        self.mol_graphs_dict = mol_graphs_dict
        self.max_len = max_len
        self.labels = torch.ones(self.len) if positive_sample else torch.zeros(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        mols = self.mols[item]
        seqs = self.seqs[item]
        labels = self.labels[item]
        
        split_mols = mols.replace('*', 'C').split('.')

        seq_tok = self.alphabet.encode(seqs[:self.max_len])
        graph_mols = [(self.mol_graphs_dict[mol]) for mol in split_mols]
        
        return graph_mols, torch.tensor(seq_tok).view(1, -1), labels

