# [ReactZyme](https://www.arxiv.org/abs/2408.13659)
Official Github Repo of ReactZyme ([arxiv-link](https://www.arxiv.org/abs/2408.13659)).

Rawdata can be downloaded from [zendo-reactzyme](https://zenodo.org/records/13635807). Once downloaded, put rawdata into 'data' folder.

# Python file - utils
SaProt tips: If you want to use SaProt, you have to use FoldSeek to get structure-aware sequence representations. This can be annoying. So we provide [processed structure-aware sequences](https://zenodo.org/records/13635807) for our dataset (the 'saprot_seq.pt' file from zendo). Or if you'd like to do it on your own, you can use the function [get_struc_seq](https://github.com/WillHua127/ReactZyme/blob/main/process_saprot.py) from process_saprot.py.

(1) Processing Sequences:

[get_afdb.py](https://github.com/WillHua127/ReactZyme/blob/main/get_afdb.py): code example of fetching afdb structures for time-based split.

[process_saprot.py](https://github.com/WillHua127/ReactZyme/blob/main/process_saprot.py): code example of processing saprot features for afbd structures.

[process_esm.py](https://github.com/WillHua127/ReactZyme/blob/main/process_esm.py): code example of processing ESM features for sequences.



(2) Processing Reactions:

[mat.py](https://github.com/WillHua127/ReactZyme/blob/main/mat.py): code for MAT for loading model purposes.

[process_mat.py](https://github.com/WillHua127/ReactZyme/blob/main/process_mat.py): code example of processing MAT features for reactions.

[prepare_graphs.py](https://github.com/WillHua127/ReactZyme/blob/main/prepare_graphs.py): code for process molecular graphs.


(3) General dataloading:

[data_utils.py](https://github.com/WillHua127/ReactZyme/blob/main/data_utils.py): dataloader etc.



## Citation
```bash
@article{hua2024reactzyme,
  title={Reactzyme: A Benchmark for Enzyme-Reaction Prediction},
  author={Hua, Chenqing and Zhong, Bozitao and Luan, Sitao and Hong, Liang and Wolf, Guy and Precup, Doina and Zheng, Shuangjia},
  journal={arXiv preprint arXiv:2408.13659},
  year={2024}
}
```
