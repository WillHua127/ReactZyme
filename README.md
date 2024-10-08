# ReactZyme: A Benchmark for Enzyme-Reaction Prediction [[paper](https://www.arxiv.org/abs/2408.13659)]
### Official Github repository of ReactZyme ([arxiv-link](https://www.arxiv.org/abs/2408.13659)).

### Check out for our newest [EnzymeFlow](https://github.com/WillHua127/EnzymeFlow) !!!

# Data preparation

Rawdata can be downloaded from [zendo-reactzyme](https://zenodo.org/records/13635807). Once downloaded, put rawdata into 'data' folder.

(1) Rawdata

There should be 4 rawdata files including: (1) cleaned_uniprot_rhea.tsv; (2) uniprot_molecules.tsv; (3) uniprot_rhea.tsv; (4) rhea_molecules.tsv.
Additionally, there is a saprot_seq.pt for structure-aware protein sequences for SaProt after running FoldSeek.
Put these files under the 'data' folder.

(2) Processed data

And there should be 3 splits: (1) time; (2) seq-smi based; (3) mol-smi based. Put time/seq-smi/mol-smi under new_time/new_seq_smi/new_mol_smi folders, respectively. Notice that we only provide positive enzyme-reaction pairs, the design of negative samples remains an open question. Nevertheless, we provide example of negative samples generation in [prepare_negative.py](https://github.com/WillHua127/ReactZyme/blob/main/prepare_negative.py).

# Python file - utils

SaProt tips: If you want to use SaProt, you have to use FoldSeek to get structure-aware sequence representations. This can be annoying. So we provide [processed structure-aware sequences](https://zenodo.org/records/13635807) for our dataset (the 'saprot_seq.pt' file from zendo). Or if you'd like to do it on your own, you can use the function [get_struc_seq](https://github.com/WillHua127/ReactZyme/blob/main/process_saprot.py) from process_saprot.py.


(1) Processing Sequences

>[get_afdb.py](https://github.com/WillHua127/ReactZyme/blob/main/get_afdb.py): code example of fetching afdb structures for time-based split.

>[process_saprot.py](https://github.com/WillHua127/ReactZyme/blob/main/process_saprot.py): code example of processing saprot features for afbd structures.

>[process_esm.py](https://github.com/WillHua127/ReactZyme/blob/main/process_esm.py): code example of processing ESM features for sequences.



(2) Processing Reactions

>[mat.py](https://github.com/WillHua127/ReactZyme/blob/main/mat.py): code for MAT for loading model purposes.

>[process_mat.py](https://github.com/WillHua127/ReactZyme/blob/main/process_mat.py): code example of processing MAT features for reactions.

>[prepare_graphs.py](https://github.com/WillHua127/ReactZyme/blob/main/prepare_graphs.py): code for process molecular graphs.



(3) General dataloading

>[data_utils.py](https://github.com/WillHua127/ReactZyme/blob/main/data_utils.py): dataloader etc.




(4) Negative samples
>[prepare_negative.py](https://github.com/WillHua127/ReactZyme/blob/main/prepare_negative.py): code example of preparing negative samples based on reaction SMILES. Once you have the dictionary of negative pairs 'data/negative_mol_dict.pt', you can prepare negative samples for training.


(5) Unimol features

> [unimol.ipynb](https://github.com/WillHua127/ReactZyme/blob/main/unimol.ipynb): code example of generating unimol features for reactions.


# Python file - train and evaluation


(1) Train MLP


>[train.py](https://github.com/WillHua127/ReactZyme/blob/main/train.py): code for MLP training. 

You can do time-based esm-unimol training like: CUDA_VISIBLE_DEVICES=0 python train.py --split_type time --mol_embedding_type unimol --pro_embedding_type esm --batch_size 1000

>[retrieval.py](https://github.com/WillHua127/ReactZyme/blob/main/retrieval.py): code for MLP evaluation.



(2) Train Contrastive


>[train_contra.py](https://github.com/WillHua127/ReactZyme/blob/main/train_contra.py): code for MLP-contrastive training. 


>[retrieval.py](https://github.com/WillHua127/ReactZyme/blob/main/retrieval.py): code for MLP-contrastive evaluation.



(3) Train Transformer


>[train_tfmr.py](https://github.com/WillHua127/ReactZyme/blob/main/train_tfmr.py): code for Transformer training. 


>[retrieval_tfmr.py](https://github.com/WillHua127/ReactZyme/blob/main/retrieval_tfmr.py): code for Transformer evaluation.




(4) Train Bi-RNN


>[train_rnn.py](https://github.com/WillHua127/ReactZyme/blob/main/train_rnn.py): code for Bi-RNN training. 


>[retrieval_rnn.py](https://github.com/WillHua127/ReactZyme/blob/main/retrieval_rnn.py): code for Bi-RNN evaluation.



## Citation
```
@article{hua2024reactzyme,
  title={Reactzyme: A Benchmark for Enzyme-Reaction Prediction},
  author={Hua, Chenqing and Zhong, Bozitao and Luan, Sitao and Hong, Liang and Wolf, Guy and Precup, Doina and Zheng, Shuangjia},
  journal={arXiv preprint arXiv:2408.13659},
  year={2024}
}
```
