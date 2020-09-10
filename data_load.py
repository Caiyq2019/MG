import numpy as np
import torch
import pdb

class data_load:
    class Data:
        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.y = data.astype(np.int64)
    def __init__(self,path, nclass = 0):
        data, label = load_data_normalised(path, nclass)
        self.dt = self.Data(data)
        self.label = self.Data(label)

def get_data(path, n_filter=0):
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
        if len(x_dic[i])>=n_filter:
            x_choose[i]=x_dic[i]
    print('x_choose done')  

    x_vector=[]
    label_vector=[]
    n=0
    for i in x_choose:
        for j in x_choose[i]:
            x_vector.append(j)
            label_vector.append(n)
        n+=1
    x_vector = np.array(x_vector)
    label_vector = np.array(label_vector)

    print('%d samples loaded, dim=%d, labels=%d, nclasses=%d'%(x_vector.shape[0], x_vector.shape[1], label_vector.shape[0], len(set(label_vector))))
    print('label_vector1=',label_vector)

    return x_vector, label_vector



def load_data_normalised(path, n_filter):
    X, labels = get_data(path, n_filter)
    return X, labels


def dataset_prepare(n_filter = 0, test_name=None):
    
    #training set voxceleb_4k_speaker
    print("loading training data from %s"%'./data/xvector/vox_4k.npz');
    trn_data = data_load('./data/xvector/vox_4k.npz', n_filter)
    t0_tensor = torch.from_numpy(trn_data.dt.x)
    t0_label_tensor = torch.from_numpy(trn_data.label.y)
    t0_dataset = torch.utils.data.TensorDataset(t0_tensor, t0_label_tensor)


    #sitw enroll
    print("loading enrollment data from %s"%'./data/xvector/Sitw/enroll.npz');
    enr_data = data_load('./data/xvector/Sitw/enroll.npz',0)
    t1_tensor = torch.from_numpy(enr_data.dt.x)
    t1_label_tensor = torch.from_numpy(enr_data.label.y)
    t1_dataset = torch.utils.data.TensorDataset(t1_tensor,t1_label_tensor)
    #testset: verify
    print("loading enrollment data from %s"%'./data/xvector/Sitw/test.npz');
    ver_data = data_load('./data/xvector/Sitw/test.npz',0)
    t2_tensor = torch.from_numpy(ver_data.dt.x)
    t2_label_tensor = torch.from_numpy(ver_data.label.y)
    t2_dataset = torch.utils.data.TensorDataset(t2_tensor,t2_label_tensor)

   
    return t0_dataset, t1_dataset, t2_dataset

