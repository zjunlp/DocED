# DocED
This repository is the official implementation of MLBiNet: A Cross-Sentence Collective Event Detection Network.

## Requirements
### To install basic requirements:
pip install requirements.txt

## Datasets
ACE2005 can be found here: https://catalog.ldc.upenn.edu/LDC2006T06

## Basic training
### To evaluate a setting with serveral random trials, execute
python run_experiments_multi.py

#### Main hyperparameters in train_MLBiNet.py include:
--tagging_mechanism,   mechanism to model event inter-dependency, you can choose one of "forward_decoder", "backward_decoder" or "bidirectional_decoder"

--num_tag_layers,   number of tagging layers, 1 indicates that we do sentence-level ED, 2 indicates that information of adjacent sentences were aggregated, ...

--max_doc_len,   maximum number of consecutive sentences are extracted as a mini-document, we can set it as 8 or 16

--tag_dim,   dimension of an uni-directional event tagging vector

--self_att_not,   whether to apply self-attention mechanism in sentence encoder 

## Main results
### Overall performance on ACE2005
![image](https://user-images.githubusercontent.com/32415352/118842889-252e6900-b8fc-11eb-9de8-dba5f82377f4.png)

### Performance on detecting multiple events collectively
![image](https://user-images.githubusercontent.com/32415352/118843522-b9003500-b8fc-11eb-8e3f-759f6d37f98a.png)

where 1/1 means one sentence that has one event; otherwise, 1/n is used.

### Performance of our proposed method with different multi-layer settings or decoder methods
![image](https://user-images.githubusercontent.com/32415352/118843910-11cfcd80-b8fd-11eb-965c-fbcde1319983.png)
