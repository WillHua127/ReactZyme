import argparse
import math
import os
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, AUROC

from data_utils import *
from itertools import islice

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
    def __init__(self, mol_input_dim, seq_input_dim, hidden_dim, output_dim, n_layers=1, dropout=0.0):
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
                                    nn.Linear(256, hidden_dim, bias=False),
                                    )
        
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.lin_out = nn.Sequential(
                                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                                    nn.Dropout(dropout),
                                    nn.LayerNorm(hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, output_dim, bias=False),
                                    nn.Dropout(dropout),
                                    nn.LayerNorm(output_dim),
                                    nn.ReLU(),
                                    nn.Linear(output_dim, 16, bias=False),
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

    def forward(self, mol_src, seq_src, hidden=None):
        # src:(B,T)
        b_size = mol_src.size(0)
        mol_embedded = self.lin_mol_embed(mol_src) #(B,H)
        seq_embedded = self.lin_seq_embed(seq_src) #(B,H)
        
        mol_embedded = mol_embedded.reshape(b_size, 1, -1)
        seq_embedded = seq_embedded.reshape(b_size, 1, -1)

        _mol_embedded = self.cross_attn_mol(mol_embedded, seq_embedded, seq_embedded) #(B,H)
        _seq_embedded = self.cross_attn_seq(seq_embedded, mol_embedded, mol_embedded) #(B,H)
        
        embedded = torch.cat([_mol_embedded, _seq_embedded], dim=1) #(B,2T,H)
        outputs, _ = self.gru(embedded, hidden) #(B,2T,2H)
        
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_dim] +
                   outputs[:, :, self.hidden_dim:]) #(B,2T,H)

        outputs = self.lin_out(outputs.sum(1)) #(B,T,O)
        
        return outputs
    
    
        
def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--epochs', type=int, default=10000, help='number of epochs')
    parser.add_argument('--early_stopping', type=int, default=300, help='early stopping')
    parser.add_argument('--seq_len', type=int, default=5000, help='maximum length')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--n_worker', type=int, default=0, help='number of workers')
    parser.add_argument('--hidden', type=int, default=128, help='length of hidden vector')
    parser.add_argument('--mol_input_dim', type=int, default=512, help='length of hidden vector')
    parser.add_argument('--dropout', type=float, default=0., help='Adam learning rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='Adam learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-10, help='Adam weight decay')
    parser.add_argument('--split_type', type=str, default='mol_smi')
    parser.add_argument('--mol_embedding_type', type=str, default='unimol')
    parser.add_argument('--pro_embedding_type', type=str, default='esm')
    parser.add_argument('--checkpoint', type=str, default=None)
    return parser.parse_args()
    

PAD_MOL = 0
PAD_SEQ = 1
args = parse_arguments()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PretrainedNetwork(
            mol_input_dim=args.mol_input_dim, #1024,
            seq_input_dim=1280,
            hidden_dim=args.hidden,
            output_dim=64,
            dropout=args.dropout,
        ).to(args.device)

#optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

best_val_loss = float('inf')
best_tst_acc = 0
best_tst_roc = 0
if args.checkpoint is not None:
    print('loading model')
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_val_loss = checkpoint["best_loss"]

criterion = nn.BCEWithLogitsLoss(reduction='none')

accuracy = BinaryAccuracy().to('cpu')
auroc = AUROC(task="binary").to('cpu')


def train(loader, neg_weight=1, threshold=0.5):
    model.train()
    torch.set_grad_enabled(True)
    
    total_loss = 0
    pred_labels = []
    true_labels = []
    for (mols, seqs, labels) in tqdm(loader):
        optimizer.zero_grad()
        
        mols = mols.to(args.device)
        seqs = seqs.to(args.device)
        labels = labels.to(args.device)
        
        out = model(mols, seqs)
        out = out.view(-1)
        #loss = criterion(out, labels)
        
        weights = torch.ones_like(labels).to(args.device)
        weights[labels==0] = neg_weight
        loss = F.binary_cross_entropy_with_logits(out, labels, weight=weights)
        #loss = (loss * weights).mean()
        
        pred_labels.append((torch.sigmoid(out) > threshold).long())
        true_labels.append(labels)

        total_loss += loss.item() * args.batch_size
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    pred_labels = torch.cat(pred_labels, dim=-1).detach().cpu()
    true_labels = torch.cat(true_labels, dim=-1).detach().cpu()

    acc = accuracy(pred_labels, true_labels)
    roc = auroc(pred_labels, true_labels)
        
    return total_loss / len(loader.dataset), acc.item(), roc.item()
    
    
@torch.no_grad()
def test(loader, neg_weight=1, threshold=0.5):
    model.eval()
    torch.set_grad_enabled(False)

    total_loss = 0
    pred_labels = []
    true_labels = []
    
    with torch.no_grad():
        for (mols, seqs, labels) in tqdm(loader):
            mols = mols.to(args.device)
            seqs = seqs.to(args.device)
            labels = labels.to(args.device)

            out = model(mols, seqs)
            out = out.view(-1)
            #loss = criterion(out, labels)
            
            weights = torch.ones_like(labels).to(args.device)
            weights[labels==0] = neg_weight
            loss = F.binary_cross_entropy_with_logits(out, labels, weight=weights)
            #loss = (loss * weights).mean()

            pred_labels.append((torch.sigmoid(out) > threshold).long())
            true_labels.append(labels)

            total_loss += loss.item() * args.batch_size
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        
    pred_labels = torch.cat(pred_labels, dim=-1).detach().cpu()
    true_labels = torch.cat(true_labels, dim=-1).detach().cpu()

    acc = accuracy(pred_labels, true_labels)
    roc = auroc(pred_labels, true_labels)
        
    return total_loss / len(loader.dataset), acc.item(), roc.item()
    
        
if __name__ == '__main__':
    
    date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    with open(f'logger/{date}.txt', 'a') as logger:
        logger.write(f'{args}\n')
        logger.close()
        
    print('loading data...')
    pos_trn_mols, pos_trn_seqs, neg_trn_mols, neg_trn_seqs = get_samples(f'data/new_{args.split_type}/positive_train_val_{args.split_type}.pt', f'data/new_{args.split_type}/negative_train_val_{args.split_type}.pt')
    pos_tst_mols, pos_tst_seqs, neg_tst_mols, neg_tst_seqs = get_samples(f'data/new_{args.split_type}/positive_test_{args.split_type}.pt', f'data/new_{args.split_type}/negative_test_{args.split_type}.pt')
    
    trn_weight = len(pos_trn_mols) / len(neg_trn_mols)
    tst_weight = len(pos_tst_mols) / len(neg_tst_mols)
    
    mol_embedding = torch.load(f'data/embedding/{args.mol_embedding_type}_mol_embedding.pt')
    seq_embedding = torch.load(f'data/embedding/{args.pro_embedding_type}_seq_embedding.pt')

    
    print('loading data...')
    pos_trn_val = EnzymeDatasetPretrained(pos_trn_mols, pos_trn_seqs, mol_embedding, seq_embedding, positive_sample=True, max_len=args.seq_len)
    neg_trn_val = EnzymeDatasetPretrained(neg_trn_mols, neg_trn_seqs, mol_embedding, seq_embedding, positive_sample=False, max_len=args.seq_len)
    trn_val_dataset = pos_trn_val + neg_trn_val
    
    pos_tst = EnzymeDatasetPretrained(pos_tst_mols, pos_tst_seqs, mol_embedding, seq_embedding, positive_sample=True, max_len=args.seq_len)
    neg_tst = EnzymeDatasetPretrained(neg_tst_mols, neg_tst_seqs, mol_embedding, seq_embedding, positive_sample=False, max_len=args.seq_len)
    tst_dataset = pos_tst + neg_tst

    trn_size = int(0.9 * len(trn_val_dataset))
    val_size = len(trn_val_dataset) - trn_size
    trn_dataset, val_dataset = torch.utils.data.random_split(trn_val_dataset, [trn_size, val_size])
    
    
    trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker, collate_fn=collate_fn_pretrained)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker, collate_fn=collate_fn_pretrained)
    tst_loader = DataLoader(tst_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker, collate_fn=collate_fn_pretrained)

    
    current_pointer = 0
    
        
    for epoch in range(args.epochs):
        trn_loss, trn_acc, trn_roc = train(trn_loader, neg_weight=trn_weight)
        val_loss, val_acc, val_roc = test(val_loader, neg_weight=trn_weight)
        tst_loss, tst_acc, tst_roc = test(tst_loader, neg_weight=tst_weight)

        current_pointer += 1
        if trn_loss < best_val_loss:
            best_val_loss = trn_loss
            best_tst_acc = tst_acc
            best_tst_roc = tst_roc
            current_pointer = 0

            torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_loss": best_val_loss,
                            "best_acc": best_tst_acc,
                            "best_roc": best_tst_roc,
                        },
                        f'model/{args.split_type}/{args.pro_embedding_type}_{args.mol_embedding_type}_epoch{epoch}_rnn',
                    )


        print(f'Epoch: {epoch:04d}, Trn Loss: {trn_loss:.4f}, Trn Acc: {trn_acc:.4f}, Trn ROC: {trn_roc:.4f}, Val Loss: {val_loss:.4f}, Tst Loss: {tst_loss:.4f}, Tst Acc: {tst_acc:.4f}, Tst ROC: {tst_roc:.4f}, Best Val Loss: {best_val_loss:.4f}, Best Tst Acc: {best_tst_acc:.4f}, Best Tst ROC: {best_tst_roc:.4f}')

        with open(f'logger/{date}.txt', 'a') as logger:
            logger.write(f'Epoch: {epoch:04d}, Trn Loss: {trn_loss:.4f}, Trn Acc: {trn_acc:.4f}, Trn ROC: {trn_roc:.4f}, Val Loss: {val_loss:.4f}, Tst Loss: {tst_loss:.4f}, Tst Acc: {tst_acc:.4f}, Tst ROC: {tst_roc:.4f}, Best Val Loss: {best_val_loss:.4f}, Best Tst Acc: {best_tst_acc:.4f}, Best Tst ROC: {best_tst_roc:.4f}\n')
            logger.close()

        #scheduler.step()

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if current_pointer == args.early_stopping:
            break







