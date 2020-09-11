import argparse
import copy
import math
import sys
import os
import time
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from tqdm import tqdm
from utils import AverageMeter

import tsne
from score import *
import model as fnn
from data_load import dataset_prepare

pi = torch.from_numpy(np.array(np.pi))


def data_normlize(train_data, test_data):    
    return (test_data - train_data.mean(axis=0)) / train_data.std(axis=0)

def supervise_mean_var(data, labels):
    assert(data.shape[0] == labels.shape[0]), 'data and label must have the same length'  
    #init class mean vectors
    label_class = np.array(list(set(copy.deepcopy(labels).cpu().numpy())))
    mean_list=[]
    for lb in label_class:
        data_j = data[ labels == lb ]
        mean_j = torch.mean(data_j, 0, True) 

        mean_list.append(mean_j)
    class_mean = torch.cat(mean_list, 0)
    return class_mean


def initial(train_data, train_label, epoch):
   if epoch == 0:
     with torch.no_grad():
      print("SCH 2: init mean by z, init var by predifined value")
      out, _ = model(train_data)
      class_mean = supervise_mean_var(out, train_label)
     model.class_mean.data   = class_mean.clone()
     print("SCH2: init the mean adaptable parameters, and var un-adaptable")
     model.class_mean.requires_grad = True


def L2_Gaussian_log_likelihood(u, a=1):
    dim = u.shape[1]
    r = (a*dim)**0.5
    L2 = torch.norm(u,dim=1,keepdim=True)
    log_det_sigma = torch.log(torch.tensor(0.5)).sum(-1, keepdim=True)
    logp_L2 = -0.5 * ((torch.pow((L2 - r),2) / (0.5) + torch.log(2 * pi) ).sum(-1, keepdim=True) + log_det_sigma)
    return logp_L2.mean()


def angle_Gaussian_log_likelihood(u):
    u1 = u.repeat(u.shape[0],1)
    u2 = u.repeat(1,u.shape[0]).reshape(-1,u.shape[1])
    cos_slr0 = torch.cosine_similarity(u1, u2, dim=1)
    cos_slr = -(torch.pow(cos_slr0,2).sum() - u.shape[0]) / (u1.shape[0] - u.shape[0])
    return cos_slr



def train(epoch):

    model.train()
    if epoch == 0:
      initial(train_data, train_label, epoch)
    class_mean =  model.class_mean

    #statistics box
    losses_ag = AverageMeter()  
    logjac_ag = AverageMeter()
    logp_ag   = AverageMeter()
    
    hinge_L2_sw = AverageMeter()
    hinge_Ang_sw = AverageMeter()
    hinge_L2_sb = AverageMeter()
    hinge_Ang_sb = AverageMeter()

    logp_L2_sw = AverageMeter()
    logp_cos_sw = AverageMeter()
    logp_L2_sb = AverageMeter()
    logp_cos_sb = AverageMeter()

    #utilities
    pbar = tqdm(total=len(train_loader.dataset))

    #strat to train
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer_model.zero_grad()

        #flow loss with class specific means
        mean_j = torch.index_select(class_mean, 0, labels)

        u_out, dnf_loss, log_probs_z, logdet = model.dnf_Gaussian_log_likelihood(data, mean_j, args.vc)
        #log_probs_dnf = dnf_loss.mean() 

        logp_L2_SW  = L2_Gaussian_log_likelihood(u_out - mean_j)
        logp_cos_SW = angle_Gaussian_log_likelihood(u_out - mean_j)
        logp_L2_SB  = L2_Gaussian_log_likelihood(mean_j) 
        logp_cos_SB = angle_Gaussian_log_likelihood(mean_j)

        hinge_L_sw = torch.max(torch.tensor(0.0).to(device), (-0.01)  + (-1.07)   - logp_L2_SW)
        hinge_A_sw = torch.max(torch.tensor(0.0).to(device), (-0.001) + (-0.0019) - logp_cos_SW)
        hinge_L_sb = torch.max(torch.tensor(0.0).to(device), (-0.01)  + (-1.07)   - logp_L2_SB)
        hinge_A_sb = torch.max(torch.tensor(0.0).to(device), (-0.001) + (-0.0019) - logp_cos_SB)
        
        if epoch < 0:
          loss = -(log_probs_z + logdet)
        else:
          loss = -(args.logpz*log_probs_z + args.logdet*logdet) + args.L2_SW*hinge_L_sw + args.cos_SW*hinge_A_sw + args.L2_SB*hinge_L_sb + args.cos_SB*hinge_A_sb 


        loss.backward()
        optimizer_model.step()
        
        #update statistics
        losses_ag.update(    loss.item(),        labels.shape[0] )
        logp_ag.update(      log_probs_z.item(), labels.shape[0] )
        logjac_ag.update(    logdet.item(),      labels.shape[0] )

        hinge_L2_sb.update(  hinge_L_sb.item(),  labels.shape[0] )
        hinge_Ang_sb.update( hinge_A_sb.item(),  labels.shape[0] )
        hinge_L2_sw.update(  hinge_L_sw.item(),  labels.shape[0] )
        hinge_Ang_sw.update( hinge_A_sw.item(),  labels.shape[0] )

        logp_L2_sw.update(  logp_L2_SW.item(),   labels.shape[0] )
        logp_cos_sw.update( logp_cos_SW.item(),  labels.shape[0] )
        logp_L2_sb.update(  logp_L2_SB.item(),   labels.shape[0] )
        logp_cos_sb.update( logp_cos_SB.item(),  labels.shape[0] )

        pbar.update(data.size(0))
        pbar.set_description('Total val/avg={:.3f}/{:.3f}  LogP val/avg={:.3f}/{:.3f} LogDet val/avg= {:.3f}/{:.3f}  hinge_L2/Ang_sw={:.3f}/{:.3f}  hinge_L2/Ang_sb={:.3f}/{:.3f}  logp_L2/cos_sw={:.3f}/{:.4f}  logp_L2/cos_sb={:.3f}/{:.4f}'.format(
            losses_ag.val, losses_ag.avg, logp_ag.val, logp_ag.avg, logjac_ag.val, logjac_ag.avg,  hinge_L2_sw.avg, hinge_Ang_sw.avg, hinge_L2_sb.avg, hinge_Ang_sb.avg, logp_L2_sw.avg, logp_cos_sw.avg, logp_L2_sb.avg, logp_cos_sb.avg ))
    
    #utility 
    pbar.close()
    

    

if __name__ == "__main__":

   if sys.version_info < (3, 6):
      print('Sorry, this code might need Python 3.6 or higher')
   else:
      print("%s"%' '.join(sys.argv))

   # Training settings
   parser = argparse.ArgumentParser(description='PyTorch Flows')
   parser.add_argument('--batch-size', type=int,   default=100,   help='input batch size for training (default: 300)')
   parser.add_argument('--epochs',     type=int,   default=1000,  help='number of epochs to train (default: 1000)')
   parser.add_argument('--lr',         type=float, default=0.0005, help='learning rate (default: 0.003)')
   parser.add_argument('--num-blocks', type=int,   default=10,    help='number of invertible blocks (default: 10)')
   parser.add_argument('--seed',       type=int,   default=1,     help='random seed (default: 1)')
   parser.add_argument('--gpu',        type=int,   default=0,     help='GPU used (default: 0)')
   parser.add_argument('--n_filter',   type=int,   default=10,    help='dryrun a small number of classes (default: 0)')
   parser.add_argument('--vc',         type=float, default=1.0,   help='variance of the class space (default: 1.0)')
   
   parser.add_argument('--logpz',  type=float, default=1.0, help='variance of the class space (default: 1.0)')
   parser.add_argument('--logdet', type=float, default=1.0, help='variance of the result space (default: 1.0)')   
   parser.add_argument('--L2_SW',  type=float, default=1.0, help='variance of the class space (default: 1.0)')
   parser.add_argument('--cos_SW', type=float, default=1.0, help='variance of the result space (default: 1.0)')
   parser.add_argument('--L2_SB',  type=float, default=1.0, help='variance of the class space (default: 1.0)')
   parser.add_argument('--cos_SB', type=float, default=1.0, help='variance of the result space (default: 1.0)')
   #parser.add_argument('--ex',     type=float, default=1,   help='variance of the result space (default: 1)')
   
   args = parser.parse_args()
   

   #working env
   os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%args.gpu
   device = torch.device("cuda")
   torch.manual_seed(args.seed)
   torch.cuda.manual_seed(args.seed) 
   kwargs = {'num_workers': 4, 'pin_memory': True}


   #load data
   print("loading data ...")
   t0_dataset, t1_dataset, t2_dataset = dataset_prepare(args.n_filter)
   train_loader = torch.utils.data.DataLoader(t0_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

   print("data loaded")
   train_data  = t0_dataset[:][0].to(device)
   train_label = t0_dataset[:][1].to(device)

   #define the network structure
   num_inputs = train_data.shape[1]
   num_hidden = train_data.shape[1]
   act = 'relu' # 'relu' 'PReLU' 'LeakyReLU' 'sigmoid' 'tanh'

   print("Flow structure: %d blocks with activation=%s"%(args.num_blocks, act))

   modules=[]
   for i in range(args.num_blocks):
      modules += [
                    fnn.MADE(num_inputs, num_hidden, act=act),
                 ]

   model = fnn.FlowSequential(*modules)
   class_mean = model.set_class_mean()


 
   for module in model.modules():
      if isinstance(module, nn.Linear):
          nn.init.orthogonal_(module.weight)
          if hasattr(module, 'bias') and module.bias is not None:
              module.bias.data.fill_(0)

   model.to(device)

   #define the optimizer and its optimizer
   optimizer_model = optim.Adam(model.parameters(), lr=args.lr)
   
   
   if not os.path.exists('./z_tr'):
      os.mkdir('./z_tr');
   if not os.path.exists('./z_Sitw'):
      os.mkdir('./z_Sitw');
   if not os.path.exists('./chkpt'):
      os.mkdir('./chkpt' );
   
   '''
   #initial
   print('\nDrawing tsne of xvector ......')
   tsne.main('./data/xvector/vox_4k.npz', -1)
   #scoing
   print('do scoring ......')
   trails_path = './data/xvector/Sitw/core-core.lst'
   eer = cosine_scoring_by_trails('./data/xvector/vox_4k.npz', './data/xvector/Sitw/enroll.npz', './data/xvector/Sitw/test.npz', trails_path)
   print('>>>> epoch = {}  cosine eer% = {:.2f}%'.format(-1, eer*100))
   file = open('result_eer','a')
   file.write('epoch = {} cosine-eer% = {:.2f} \n'.format(-1, eer*100))
   file.close()
   '''

   for epoch in range(args.epochs):
      print('\nEpoch: {}'.format(epoch))
      print(time.asctime( time.localtime(time.time()) ))
      train(epoch)
      #save model to checkpoint
      if epoch % 5 == 0:
        print("saving model to ./chkpt/model_epoch%d.pt"%(epoch))
        torch.save(model.state_dict(), './chkpt/model_epoch{}.pt'.format(epoch))
        
        #evaluation
        with torch.no_grad():
            print('do evaluation ......\n')
            path0='./z_tr/z0_epoch{}.npz'.format(epoch)
            u0, _ = model(t0_dataset[:][0].to(device))
            label0 = np.load('./data/xvector/vox_4k.npz')['utt']
            data0  = u0.cpu().detach().numpy()
            data00  = data_normlize(data0, data0)
            np.savez(path0, vector=data00, utt=label0)
            
            path1='./z_Sitw/z1_epoch{}.npz'.format(epoch)
            u1, _ = model(t1_dataset[:][0].to(device))
            label1 = np.load('./data/xvector/Sitw/enroll.npz')['utt']
            data1  = u1.cpu().detach().numpy()
            data1  = data_normlize(data0, data1)
            np.savez(path1, vector=data1, utt=label1)
            
            path2='./z_Sitw/z2_epoch{}.npz'.format(epoch)
            u2, _ = model(t2_dataset[:][0].to(device))
            label2 = np.load('./data/xvector/Sitw/test.npz')['utt']
            data2  = u2.cpu().detach().numpy()
            data2  = data_normlize(data0, data2)
            np.savez(path2, vector=data2, utt=label2)
            
            #scoing
            print('do scoring ......')
            trails_path = './data/xvector/Sitw/core-core.lst'
            eer = cosine_scoring_by_trails(path0, path1, path2, trails_path)
            print('>>>> epoch = {}  cosine eer% = {:.2f}%'.format(epoch, eer*100))
            
            file = open('result_eer','a')
            file.write('epoch = {} cosine-eer% = {:.2f} \n'.format(epoch, eer*100))
            file.close()
            
            #draw latent tsne
            print('\nDrawing tsne of latent space ......')
            tsne.main(path0,epoch)

            
   print("Training end..")
   print(time.asctime( time.localtime(time.time()) ))
