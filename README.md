# MG Training (Maximum Gaussianality Training)
A Pytorch implementations of MG training for author's article ["Deep Speaker Vector Normalization with Maximum Gaussianality Training"](https://arxiv.……)
This is a general Gaussian distribution training method and can be used in any task that requires Gaussian distribution in latent space. 

## Datasets
```bash
trainingset:Voxceleb 
testset: SITW, CNCeleb
```
Following this [link](https://pan.baidu.com/s/1NZXZhKbrJUk75FDD4_p6PQ) to download the dataset 
(extraction code：8xwe)

## Run DNF with ML training (original [DNF](https://github.com/Caiyq2019/Deep-normalization-for-speaker-vectors) model )
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
