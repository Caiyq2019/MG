import copy
import os
import torch
import matplotlib.pyplot as plt
import time
import numpy as np

import torch.optim as optim
import torch.nn.functional as F


def data_load(train_path, enroll_path, test_path, trails_path):
    train  = np.load(train_path) 
    enroll = np.load(enroll_path)
    test   = np.load(test_path)
    tr_vec    = train['vector']
    en_vec    = enroll['vector']
    en_label  = enroll['utt']
    te_vec    = test['vector']
    te_label  = test['utt']
    en_vec = en_vec - tr_vec.mean(axis=0)
    te_vec = te_vec - tr_vec.mean(axis=0)
    trails = load_trails(trails_path)

    return en_vec, en_label, te_vec, te_label, trails


def compute_eer(target_scores, nontarget_scores): 
    if isinstance(target_scores , list) is False:
        target_scores = list(target_scores)
    if isinstance(nontarget_scores , list) is False:
        nontarget_scores = list(nontarget_scores)
    target_scores = sorted(target_scores)
    nontarget_scores = sorted(nontarget_scores)
    target_size = len(target_scores);
    nontarget_size = len(nontarget_scores)
    for i in range(target_size-1):
        target_position = i
        nontarget_n = nontarget_size * float(target_position) / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontarget_scores[nontarget_position] < target_scores[target_position]:
            break
    th = target_scores[target_position];
    eer = target_position * 1.0 / target_size;
    return eer, th 


def compute_idr(trails_path, scores):
    trails = load_trails(trails_path)
    p=0
    n=0
    for lb in set(trails[1]):
        sco_i = scores[trails[1]==lb]
        s_index = np.argmax(sco_i)
        tar = trails[2][trails[1]==lb]
        if tar[s_index]=='target':
            p+=1
        else:
            n+=1
    idr = p/(n+p)

    return idr


def load_trails(trails_path):
    file = open(trails_path,'r')
    trails=[]
    enroll=[]
    test  =[]
    tar   =[]
    for line in file:
        line = line.strip('\n').split(' ')
        enroll.append(line[0])
        test.append(line[1])
        tar.append(line[2])

    trails.append(enroll)
    trails.append(test)
    trails.append(tar)
    trails = np.array(trails)
    return trails



def supervise_mean_var(data, label):
    assert(data.shape[0] == label.shape[0]), 'data and label must have the same length'  
    label_class = np.array(list(set(copy.deepcopy(label))))

    mean_list = []
    label_list = []
    for lb in label_class:
        data_j = data[label==lb]
        mean_j = data_j.mean(0) 
        mean_list.append(mean_j)
        label_list.append(lb)
    mean_class = np.array(mean_list)
    mlabel = np.array(label_list)

    return mean_class, mlabel


def Cosine_score(test_vec, te_label, enroll_vec, enroll_label, trails):
    mean_class, mlabel = supervise_mean_var(enroll_vec, enroll_label)

    Cosine_tensor_list=[]
    tar=[]
    for i in range(trails[0].shape[0]):
        e = mean_class[mlabel==trails[0][i]][0]
        t = test_vec[te_label==trails[1][i]][0]

        Cosine_similarity = np.dot(e, t) / (np.linalg.norm(e) * np.linalg.norm(t))

        Cosine_tensor_list.append(Cosine_similarity)
        tar.append(trails[2][i])

    Cosine_scores = np.array(Cosine_tensor_list)

    return Cosine_scores, np.array(tar)


def cosine_scoring_by_trails(train_path,enroll_path,test_path,trails_path):
    en_vec, en_label, te_vec, te_label, trails = data_load(train_path, enroll_path, test_path, trails_path)
    Cosine_scores, tar = Cosine_score(te_vec, te_label, en_vec, en_label, trails)
    eer, th = compute_eer(Cosine_scores[tar=='target'], Cosine_scores[tar=='nontarget'])
    return eer
    

'''
if __name__ == "__main__":
    train_path  = './data/xvector/vox_4k.npz'
    enroll_path = './data/xvector/Sitw/enroll.npz'
    test_path   = './data/xvector/Sitw/test.npz'
    trails_path = './data/xvector/Sitw/core-core.lst'
    eer = cosine_scoring_by_trails(train_path,enroll_path,test_path,trails_path)
    print('eer={:.2f}'.format(eer*100))
'''

















