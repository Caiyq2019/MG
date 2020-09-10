# MG Training
A Pytorch implementations of 'Maximum Gaussianality Training' for author's article ["Deep Speaker Vector Normalization with Maximum Gaussianality Training"](https://arxiv.……)

## Datasets
```bash
trainingset:Voxceleb 
testset: SITW, CNCeleb
```
Following this [link](https://pan.baidu.com/s/1NZXZhKbrJUk75FDD4_p6PQ) to download the dataset 
(extraction code：8xwe)

## Run DNF with ML training
```bash
./run_ML.sh
```
## Run DNF with MG training
```bash
./run_MG.sh
```
The evaluation and scoring will be performed automatically during the training process.

## Other instructions
```bash
score.py is a python implementations of the standard kaldi consine scoring, you can also use kaldi to do the plda scoring
tsne.py can be used to draw the distribution of latent space 
```
