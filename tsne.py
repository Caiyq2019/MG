# coding='utf-8'
"""t-SNE对手写数字进行可视化"""
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE


def get_data(path):
    data=np.load(path)
    
    X=data['vector']
    labels=data['utt']

    ID_list=[]
    for label in labels:
        ID_list.append(label.split('-')[0])
    x_dic={}
    for i in ID_list:
        x_dic[i]=[]
    for i in range(len(ID_list)):
        if ID_list[i] in x_dic:
           x_dic[ID_list[i]].append(X[i])
    x_choose={}
    for i in x_dic:
        if len(x_dic[i])>=250:
            x_choose[i]=x_dic[i]
    print('select speakers with utts large than 250')  

    x_vector=[]
    label_vector=[]
    n=1
    for i in x_choose:
        for j in x_choose[i]:
            x_vector.append(j)
            label_vector.append(n)
        n+=1
    n_samples, n_dim =len(x_vector), x_vector[0].shape
    n_labels=len(set(label_vector))
    print('n_samples={}, n_dim={}, n_labels={}'.format(n_samples,n_dim,n_labels))
    return x_vector, label_vector


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    if np.min(label, 0) != np.max(label, 0):
        label_min, label_max = np.min(label, 0), np.max(label, 0)
        label = (label - label_min) / (label_max - label_min)
    else:
        label=label

    fig = plt.figure()
    ax = plt.subplot(111)

    plt.scatter(data[:, 0], data[:, 1], 10, c=label, cmap=plt.cm.Spectral, alpha=0.5)
    plt.title(title)
    return fig


def main(path0, epoch):
    data, labels_color = get_data(path0)
    print('Computing t-SNE embedding epoch')
    n_labels=len(set(labels_color))
    tsne = TSNE(n_components = 2, init='pca', random_state = 0)

    result = tsne.fit_transform(data)
    
    fig = plot_embedding(result, labels_color,'t-SNE z_vector epoch{} n_speaker={}'.format(epoch, n_labels))

    if not os.path.exists('./tsne'):
        os.mkdir('./tsne' );
        
    plt.savefig("./tsne/z_vector_epoch{}_n_speaker={}.png".format(epoch, n_labels))
    plt.close()
    
 

if __name__ == '__main__':
    main()

  
