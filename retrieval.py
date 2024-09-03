import argparse
import math
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import *
from itertools import islice

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    
class CrossAttention(nn.Module):
    def __init__(self, query_input_dim, key_input_dim, output_dim):
        super(CrossAttention, self).__init__()
        
        self.out_dim = output_dim
        self.W_Q = nn.Linear(query_input_dim, output_dim)
        self.W_K = nn.Linear(key_input_dim, output_dim)
        self.W_V = nn.Linear(key_input_dim, output_dim)
        self.scale_val = self.out_dim ** 0.5
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query_input, key_input, value_input, query_input_mask=None, key_input_mask=None):
        query = self.W_Q(query_input)
        key = self.W_K(key_input)
        value = self.W_V(value_input)

        attn_weights = torch.matmul(query, key.transpose(1, 2)) / self.scale_val
        attn_weights = self.softmax(attn_weights)
        output = torch.matmul(attn_weights, value)
        
        return output

class PretrainedNetwork(nn.Module):
    def __init__(self, mol_input_dim=1024, seq_input_dim=1280, hidden_dim=128, output_dim=64, dropout=0.0):
        super(PretrainedNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.lin_mol_embed = nn.Sequential(
                                    nn.Linear(mol_input_dim, 256, bias=False),
                                    nn.Dropout(dropout),
                                    nn.BatchNorm1d(256),
                                    nn.SiLU(),
                                    nn.Linear(256, 256, bias=False),
                                    nn.Dropout(dropout),
                                    nn.BatchNorm1d(256),
                                    nn.SiLU(),
                                    nn.Linear(256, 256, bias=False),
                                    nn.Dropout(dropout),
                                    nn.BatchNorm1d(256),
                                    nn.SiLU(),
                                    nn.Linear(256, hidden_dim, bias=False),
                                    )
        
        self.lin_seq_embed = nn.Sequential(
                                    nn.Linear(seq_input_dim, 512, bias=False),
                                    nn.Dropout(dropout),
                                    nn.BatchNorm1d(512),
                                    nn.SiLU(),
                                    nn.Linear(512, 256, bias=False),
                                    nn.Dropout(dropout),
                                    nn.BatchNorm1d(256),
                                    nn.SiLU(),
                                    nn.Linear(256, 256, bias=False),
                                    nn.Dropout(dropout),
                                    nn.BatchNorm1d(256),
                                    nn.SiLU(),
                                    nn.Linear(256, hidden_dim, bias=False),
                                    )
        
        
        self.lin_out = nn.Sequential(
                                    nn.Linear(2*hidden_dim, hidden_dim, bias=False),
                                    nn.Dropout(dropout),
                                    nn.SiLU(),
                                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                                    nn.Dropout(dropout),
                                    nn.SiLU(),
                                    nn.Linear(hidden_dim, output_dim, bias=False),
                                    nn.Dropout(dropout),
                                    nn.SiLU(),
                                    nn.Linear(output_dim, 16, bias=False),
                                    nn.Dropout(dropout),
                                    nn.Linear(16, 1, bias=False),
                                    )
        
        self.cross_attn_seq = CrossAttention(
                                query_input_dim=hidden_dim,
                                key_input_dim=hidden_dim,
                                output_dim=hidden_dim,
                            )
        
        self.cross_attn_mol = CrossAttention(
                                query_input_dim=hidden_dim,
                                key_input_dim=hidden_dim,
                                output_dim=hidden_dim,
                            )

    def forward(self, mol_src, seq_src):
        # src:(B,H)
        b_size = mol_src.size(0)
        mol_embedded = self.lin_mol_embed(mol_src) #(B,H)
        seq_embedded = self.lin_seq_embed(seq_src) #(B,H)
        
        mol_embedded = mol_embedded.reshape(b_size, 1, -1)
        seq_embedded = seq_embedded.reshape(b_size, 1, -1)
                
        _mol_embedded = self.cross_attn_mol(mol_embedded, seq_embedded, seq_embedded) #(B,H)
        _seq_embedded = self.cross_attn_seq(seq_embedded, mol_embedded, mol_embedded) #(B,H)
                
        outputs = self.lin_out(torch.cat([_mol_embedded, _seq_embedded], dim=-1))
        
        return outputs
    

def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', type=str, default='model/time/esm_mat_epoch18', help='checkpoint')
    parser.add_argument('--seq_len', type=int, default=5000, help='maximum length')
    parser.add_argument('--topk', type=int, default=1, help='topk')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--n_worker', type=int, default=0, help='number of workers')
    parser.add_argument('--split_type', type=str, default='time')
    parser.add_argument('--hidden', type=int, default=128, help='length of hidden vector')
    parser.add_argument('--mol_input_dim', type=int, default=512, help='length of hidden vector')
    parser.add_argument('--dropout', type=float, default=0., help='Adam learning rate')
    parser.add_argument('--mol_embedding_type', type=str, default='mat')
    parser.add_argument('--pro_embedding_type', type=str, default='esm')
    return parser.parse_args()

args = parse_arguments()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PretrainedNetwork(
            mol_input_dim=args.mol_input_dim,#1024,
            seq_input_dim=1280,
            hidden_dim=args.hidden,
            output_dim=64,
            dropout=args.dropout,
        ).to(args.device)

checkpoint = torch.load(args.model_path, map_location=args.device)
model.load_state_dict(checkpoint['model_state_dict'])


def topk_accuracy(logits, labels, k=1):
    asrt = torch.argsort(logits, dim=1, descending=True, stable=True)
    if (logits == 0).all(dim=-1).sum():
        rand_perm = torch.stack([torch.randperm(logits.size(1)) for _ in range(logits.size(0))])
        indices = torch.where((logits == 0).all(dim=-1) == 1)[0]
        asrt[indices] = rand_perm[indices]
    
    ranking = torch.empty(logits.shape[0], logits.shape[1], dtype = torch.long).scatter_ (1, asrt, torch.arange(logits.shape[1]).repeat(logits.shape[0], 1))
    ranking = (ranking + 1).to(labels.device)
    mean_rank = (ranking * labels.float()).sum(dim=-1) / (labels.sum(dim=-1)) # (num_seq)
    #mean_rank = (ranking * labels).sum(-1) / (((labels.argsort(dim=-1, descending=True) + 1) * labels).sum(-1))
    mean_rank = mean_rank.mean(dim=0)
    #mrr = (1.0 / ranking * labels.float()).sum(dim=-1) / ((1.0 / (labels.argsort(dim=-1, descending=True) + 1) * labels).sum(-1) + 1e-9)
    mrr = (1.0 / ranking * labels.float()).sum(dim=-1) / (labels.sum(dim=-1)) # (num_seq)
    mrr = mrr.mean(dim=0)
    
    top_accs = []
    top_accs2 = []
    for k in [1, 2, 3, 4, 5, 10, 20, 50]:
        top_acc = ((ranking <= k) * labels.float()).sum(dim=-1) / k
        top_acc = top_acc.mean(dim=0)    
        top_accs.append(top_acc)

        top_acc2 = (((ranking <= k) * labels.float()).sum(dim=-1) > 0).float()
        top_acc2 = top_acc2.mean(dim=0)
        top_accs2.append(top_acc2)
        
    return top_accs[0], top_accs[1], top_accs[2], top_accs[3], top_accs[4], top_accs[5], top_accs[6], top_accs[7], top_accs2[0], top_accs2[1], top_accs2[2], top_accs2[3], top_accs2[4], top_accs2[5], top_accs2[6], top_accs2[7], mean_rank, mrr



    
@torch.no_grad()
def test(pos_pair_loader, mol_loader, labels, k=1):
    model.eval()
    torch.set_grad_enabled(False)

    preds = []
    for (seqs, _) in tqdm(tst_pos_pair_loader):
        logits = []
        seqs = seqs.repeat(args.batch_size, 1).to(seqs.device)
        for (mols, _) in tst_molecules_loader: 
            b = mols.size(0)
            mols = mols.to(args.device)
            out = model(mols, seqs[:b, :]) 
            logits.append(out)

        logits = torch.concat(logits, dim=0).view(1, -1)
        preds.append(logits)
    preds = torch.cat(preds, dim=0).to(args.device)
        
    return preds, labels


if __name__ == '__main__':
    # pos_trn_mols, pos_trn_seqs, _, _ = get_samples(f'data/new_{args.split_type}/positive_train_val_{args.split_type}.pt', f'data/new_{args.split_type}/negative_train_val_{args.split_type}.pt')
    print('loading data...')
    pos_tst_mols, pos_tst_seqs, _, _ = get_samples(f'data/new_{args.split_type}/positive_test_{args.split_type}.pt', f'data/new_{args.split_type}/negative_test_{args.split_type}.pt')
    
    
    unique_mols = list(set(pos_tst_mols))
    unique_seqs = list(set(pos_tst_seqs))
    
    labels = torch.zeros(len(unique_seqs), len(unique_mols))
    indices = [(unique_seqs.index(seq), unique_mols.index(mol)) for seq, mol in zip(pos_tst_seqs, pos_tst_mols)]
    
    for idx in indices:
        labels[idx[0]][idx[1]] = 1
    labels = labels.to(args.device)
    
    
    mol_embedding = torch.load(f'data/embedding/{args.mol_embedding_type}_mol_embedding.pt')
    seq_embedding = torch.load(f'data/embedding/{args.pro_embedding_type}_seq_embedding.pt')
    
    print('loading data...')
    pos_seqs = EnzymeDatasetPretrainedSingle(unique_seqs, seq_embedding, positive_sample=True, max_len=args.seq_len)
    pos_mols = EnzymeDatasetPretrainedSingle(unique_mols, mol_embedding, positive_sample=True, max_len=100000)
    tst_pos_pair_loader = DataLoader(pos_seqs, batch_size=1, shuffle=False, num_workers=args.n_worker, collate_fn=collate_fn_pretrained_single)
    tst_molecules_loader = DataLoader(pos_mols, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker, collate_fn=collate_fn_pretrained_single)
    

#     unique_mols = list(set(list(pos_trn_mols + pos_tst_mols)))
#     labels = torch.tensor([unique_mols.index(mol) for mol in pos_tst_mols]).to(args.device)
    
#     pos_tst = EnzymeDataset(pos_tst_mols, pos_tst_seqs, mol_tokenizer, seq_tokenizer, positive_sample=True, max_len=args.seq_len)
#     all_molecules = EnzymeDataset(unique_mols, unique_mols, mol_tokenizer, mol_tokenizer, positive_sample=True, max_len=args.seq_len)
    
    
#     tst_pos_pair_loader = DataLoader(pos_tst, batch_size=1, shuffle=False, num_workers=args.n_worker, collate_fn=collate_fn)
#     tst_molecules_loader = DataLoader(all_molecules, batch_size=1, shuffle=False, num_workers=args.n_worker, collate_fn=collate_fn)

    preds, labels = test(tst_pos_pair_loader, tst_molecules_loader, labels, k=args.topk)
    
    
    top1_acc, top2_acc, top3_acc, top4_acc, top5_acc, top10_acc, top20_acc, top50_acc, top1_acc2, top2_acc2, top3_acc2, top4_acc2, top5_acc2, top10_acc2, top20_acc2, top50_acc2, mean_rank, mrr = topk_accuracy(preds.detach().cpu(), labels.detach().cpu())
    print(f'Pred Top1 Acc-N: {top1_acc:.4f}, Top2 Acc-N: {top2_acc:.4f}, Top3 Acc-N: {top3_acc:.4f}, Top4 Acc-N: {top4_acc:.4f}, Top5 Acc-N: {top5_acc:.4f}, Top10 Acc-N: {top10_acc:.4f}, Top20 Acc-N: {top20_acc:.4f},  Top50 Acc-N: {top50_acc:.4f}, Top1 Acc: {top1_acc2:.4f}, Top2 Acc: {top2_acc2:.4f}, Top3 Acc: {top3_acc2:.4f}, Top4 Acc: {top4_acc2:.4f}, Top5 Acc: {top5_acc2:.4f}, Top10 Acc: {top10_acc2:.4f}, Top20 Acc: {top20_acc2:.4f},  Top50 Acc: {top50_acc2:.4f}, Mean Rank: {mean_rank:.4f}, MRR: {mrr:.4f}')
    
    top1_acc, top2_acc, top3_acc, top4_acc, top5_acc, top10_acc, top20_acc, top50_acc, top1_acc2, top2_acc2, top3_acc2, top4_acc2, top5_acc2, top10_acc2, top20_acc2, top50_acc2, mean_rank, mrr = topk_accuracy(labels.detach().cpu(), labels.detach().cpu())
    print(f'Data Top1 Acc-N: {top1_acc:.4f}, Top2 Acc-N: {top2_acc:.4f}, Top3 Acc-N: {top3_acc:.4f}, Top4 Acc-N: {top4_acc:.4f}, Top5 Acc-N: {top5_acc:.4f}, Top10 Acc-N: {top10_acc:.4f}, Top20 Acc-N: {top20_acc:.4f},  Top50 Acc-N: {top50_acc:.4f}, Top1 Acc: {top1_acc2:.4f}, Top2 Acc: {top2_acc2:.4f}, Top3 Acc: {top3_acc2:.4f}, Top4 Acc: {top4_acc2:.4f}, Top5 Acc: {top5_acc2:.4f}, Top10 Acc: {top10_acc2:.4f}, Top20 Acc: {top20_acc2:.4f},  Top50 Acc: {top50_acc2:.4f}, Mean Rank: {mean_rank:.4f}, MRR: {mrr:.4f}')
    
    top1_acc, top2_acc, top3_acc, top4_acc, top5_acc, top10_acc, top20_acc, top50_acc, top1_acc2, top2_acc2, top3_acc2, top4_acc2, top5_acc2, top10_acc2, top20_acc2, top50_acc2, mean_rank, mrr = topk_accuracy(preds.transpose(0,1).detach().cpu(), labels.transpose(0,1).detach().cpu())
    print(f'Pred Transpose Top1 Acc-N: {top1_acc:.4f}, Top2 Acc-N: {top2_acc:.4f}, Top3 Acc-N: {top3_acc:.4f}, Top4 Acc-N: {top4_acc:.4f}, Top5 Acc-N: {top5_acc:.4f}, Top10 Acc-N: {top10_acc:.4f}, Top20 Acc-N: {top20_acc:.4f},  Top50 Acc-N: {top50_acc:.4f}, Top1 Acc: {top1_acc2:.4f}, Top2 Acc: {top2_acc2:.4f}, Top3 Acc: {top3_acc2:.4f}, Top4 Acc: {top4_acc2:.4f}, Top5 Acc: {top5_acc2:.4f}, Top10 Acc: {top10_acc2:.4f}, Top20 Acc: {top20_acc2:.4f},  Top50 Acc: {top50_acc2:.4f}, Mean Rank: {mean_rank:.4f}, MRR: {mrr:.4f}')
    
    top1_acc, top2_acc, top3_acc, top4_acc, top5_acc, top10_acc, top20_acc, top50_acc, top1_acc2, top2_acc2, top3_acc2, top4_acc2, top5_acc2, top10_acc2, top20_acc2, top50_acc2, mean_rank, mrr = topk_accuracy(labels.transpose(0,1).detach().cpu(), labels.transpose(0,1).detach().cpu())
    print(f'Data Transpose Top1 Acc-N: {top1_acc:.4f}, Top2 Acc-N: {top2_acc:.4f}, Top3 Acc-N: {top3_acc:.4f}, Top4 Acc-N: {top4_acc:.4f}, Top5 Acc-N: {top5_acc:.4f}, Top10 Acc-N: {top10_acc:.4f}, Top20 Acc-N: {top20_acc:.4f},  Top50 Acc-N: {top50_acc:.4f}, Top1 Acc: {top1_acc2:.4f}, Top2 Acc: {top2_acc2:.4f}, Top3 Acc: {top3_acc2:.4f}, Top4 Acc: {top4_acc2:.4f}, Top5 Acc: {top5_acc2:.4f}, Top10 Acc: {top10_acc2:.4f}, Top20 Acc: {top20_acc2:.4f},  Top50 Acc: {top50_acc2:.4f}, Mean Rank: {mean_rank:.4f}, MRR: {mrr:.4f}')



