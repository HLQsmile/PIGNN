import os
# os.environ ['CUDA_VISIBLE_DEVICES']='0'
import torch as t
import torch
from torch import nn
import numpy as np
import pandas as pd
import sys
import scipy.io as sio
import math
import pickle as pkl
import math
import torch as t 
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
from sklearn.metrics import precision_score,f1_score
from torch_geometric.utils import to_undirected,remove_self_loops
from torch.nn.init import xavier_normal_,kaiming_normal_
from torch.nn.init import uniform_,kaiming_uniform_,constant
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.data import Batch,Data
from collections import Counter 
from torch.utils import data as tdata
from sklearn.model_selection import StratifiedKFold
import argparse


from dataset import (collate_func,DataLoader,ExprDataset)     
from scheduler import CosineAnnealingWarmRestarts
from model import (extract,edge_transform_func)

pathjoin = os.path.join
def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-in','--infile',type=str,default='C:/Users/DELL/Desktop/PIGNN/data/dataset.npz')
    parser.add_argument('-out-dir','--outdir',type=str,default='C:/Users/DELL/Desktop/PIGNN//results')
    parser.add_argument('-cuda','--cuda',type=bool,default=True)
    parser.add_argument('-bs','--batch_size',type=int,default=64)
    return parser

def train2(model,optimizer,train_loader,epoch,device,loss_fn =None,scheduler =None,verbose=False  ):
    model.train()

    loss_all = 0
    iters = len(train_loader)
    for idx,data in enumerate( train_loader):
        # print('epoch,idx',epoch,idx)
        # data.x = data.x + t.rand_like(data.x)*1e-5
        data = data.to(device)
        if verbose:
            print(data.y.shape,data.edge_index.shape)
        optimizer.zero_grad()
        output = model(data)
        if loss_fn is None:
            loss = F.cross_entropy(output, data.y.reshape(-1), weight=None,)
        else:
            loss = loss_fn(output, data.y.reshape(-1))

        if model.edge_weight is not None:
            l2_loss = 0 
            if isinstance(model.edge_weight,nn.Module):  # nn.ParamterList
                for edge_weight in model.edge_weight :
                    l2_loss += 0.1* t.mean((edge_weight)**2)
            elif isinstance(model.edge_weight,t.Tensor):
                l2_loss =0.1* t.mean((model.edge_weight)**2)
            # print(loss.cpu().detach().numpy(),l2_loss.cpu().detach().numpy())
            loss+=l2_loss


        
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step() #??????????????????????????????????????????

        if not (scheduler is  None):
            scheduler.step( (epoch -1) + idx/iters) # let "epoch" begin from 0 

    return loss_all / iters # len(train_dataset)

def test2(model,loader,predicts=False):
    model.eval()

    correct = 0
    y_pred =[]
    y_true=[]
    y_output=[]
    for data in loader:
        data = data.to(device)
        # print(data.y.shape)
        output = model(data)
        pred = output.max(dim=1)[1].cpu().data.numpy()
        y = data.y.cpu().data.numpy()
        y_pred.extend(pred)
        y_true.extend(y)
        y_output.extend(output.cpu().data.numpy())
        
    acc = precision_score(y_true,y_pred,average='macro')
    f1 = f1_score(y_true,y_pred,average='macro')
    if predicts:
         return acc,f1,y_true,np.array(y_pred),y_output
    else:
        return acc,f1
    
def help_bn(bn1,x):
    
    x = x.permute(1,0,2) 
    x = bn1(x) 
    x = x.permute(1,0,2) 
    return x

def weith(net,input_data):
    x, edge_index = input_data.x, input_data.edge_index

    if net.weight_edge_flag:
        one_graph_edge_weight=torch.sigmoid(net.edge_weight)#*self.edge_num.  net.edge_weight???109914???
        edge_weight = one_graph_edge_weight
    else:
        edge_weight = None
    x = net.act1(net.conv1(x.to(device), edge_index.to(device),edge_weight=edge_weight.to(device)))  #[23459,1000,8]
    x = help_bn(net.bn1,x)
    # if net.dropout_ratio >0: x = F.dropout(x, p=0.1, training=net.training)
    x = x.permute(1,2,0)  # #samples x #features x #nodes   
    x = x.unsqueeze(dim=-1) # #samples x #features x #nodes x 1 
    x = net.global_conv1(x)  # #samples x #features x #nodes x 1  
    x = net.global_act1(x)
    x = net.global_bn1(x)
    # if self.dropout_ratio >0: x = F.dropout(x, p=0.3, training=self.training)
    x = net.global_conv2(x)   #[1000,4,23459,1]
    x = net.global_act1(x)
    x = net.global_bn2(x)

    x = x.squeeze(dim=-1)  # #samples  x #features  x #nodes #[1000,4,23459]
    num_samples = x.shape[0]

    x = x .view(num_samples,-1) 

    x = net.global_fc_nn(x)
    return x

if __name__ == '__main__':


    parser = get_parser()
    args = parser.parse_args()
    print('args:',args)

    cuda_flag = args.cuda 
    npz_file = args.infile
    save_folder = args.outdir 
    batch_size = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() and cuda_flag  else 'cpu')

    prob_file =  pathjoin(save_folder,'predicted_probabilities.txt')
    pred_file = pathjoin(save_folder,'predicted_label.txt') 
    true_file = pathjoin(save_folder,'true_label.txt') 
    os.makedirs(pathjoin(save_folder,'models'),exist_ok=True)

    data= np.load(npz_file,allow_pickle=True)
    logExpr = data['logExpr'].T  # logExpr: row-cell, column-gene
    label = data['label']
    str_labels = data['str_labels']
    used_edge = edge_transform_func(data['edge_index'],)

    num_samples = logExpr.shape[0]



    init_lr =0.02
    min_lr = 0.001
    max_epoch= 40 
    batch_size = 64
    weight_decay  = 1e-4
    dropout_ratio = 0.2

    print('use wegithed cross entropy.... ')
    label_type = np.unique(label.reshape(-1))
    alpha = np.array([ np.sum(label == x) for x in label_type])
    alpha = np.max(alpha) / alpha
    alpha = np.clip(alpha,1,50)
    alpha = alpha/ np.sum(alpha)
    loss_fn = t.nn.CrossEntropyLoss(weight = t.tensor(alpha).float())
    loss_fn = loss_fn.to(device)


    dataset = ExprDataset(Expr=logExpr,edge=used_edge,y=label,device=device)
    gene_num = dataset.gene_num
    class_num = len(np.unique(label))

  
    kf = StratifiedKFold(n_splits=5,shuffle=True)
    for tr, ts in kf.split(X=label,y=label):
        train_index = tr 
        test_index = ts 
    
    train_dataset = dataset.split(t.tensor(train_index).long())
    test_dataset = dataset.split(t.tensor(test_index).long())
    # add more samples for those small celltypes
    train_dataset.duplicate_minor_types(dup_odds=50)

        
    num_workers = 0
    assert num_workers == 0 
    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=num_workers, shuffle=True,collate_fn = collate_func,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1,num_workers=num_workers,collate_fn = collate_func)

    model = extract(in_channel = dataset.num_expr_feaure , num_nodes=gene_num,
                out_channel=class_num,edge_num=dataset.edge_num,
                dropout_ratio = dropout_ratio,
                ).to(device)  

    print(model)

    optimizer = torch.optim.Adam(model.parameters(),lr=init_lr ,weight_decay=weight_decay,)
    scheduler = CosineAnnealingWarmRestarts(optimizer,2, 2, eta_min=min_lr, lr_max_decay=0.5)
    max_metric = float(0)
    max_metric_count = 0
    weights_list = []
    
    # for idx,data in enumerate(train_loader):
    #     output=model.forward(data)

    for epoch in range(1, max_epoch):  #??????16
        train_loss = train2(model,optimizer,train_loader,epoch,device,loss_fn,scheduler =scheduler )
        train_acc,train_f1= test2(model,train_loader,predicts=False)
        lr = optimizer.param_groups[0]['lr']
        # print('epoch\t%03d,lr : %.06f,loss: %.06f,T-acc: %.04f,T-f1: %.04f'%(
        #             epoch,lr,train_loss,train_acc,train_f1))

        
    #stage two 
    extend_epoch = 50 
    print('stage 2 training...')
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0) #????????????????????????train test????????????8???2
    for final_, valid_ in sss.split(dataset.y[train_index],dataset.y[train_index]):
        train_index2,valid_index2 =train_index[final_],train_index[valid_]
    valid_dataset = dataset.split(t.tensor(valid_index2).long())
    train_dataset = dataset.split(t.tensor(train_index2).long())
    if True:
        # add more samples for those small celltypes
        train_dataset.duplicate_minor_types(dup_odds=50)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=num_workers, shuffle=True,collate_fn = collate_func,drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,num_workers=num_workers,shuffle=True,collate_fn = collate_func)
    lr = optimizer.param_groups[0]['lr']
    print('stage2 initilize lr:',lr)

    max_metric = float(0)
    max_metric_count = 0
    optimizer = torch.optim.Adam(model.parameters(),lr=lr ,weight_decay=weight_decay,)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor=0.1, patience=2, verbose=True,min_lr=0.001)
    old_lr = lr 
    for epoch_idx,epoch in enumerate(range(max_epoch,(max_epoch+extend_epoch))):  
        if old_lr != lr:
            max_metric_count = 0 
            print('reset max_metric_count to 0 due to updating lr from %f to %f'%(old_lr,lr))
            old_lr = lr 

        train_loss = train2(model,optimizer,train_loader,epoch,device,loss_fn,verbose=False  )
        train_acc,train_f1= test2(model,train_loader,predicts=False)
        valid_acc,valid_f1= test2(model,valid_loader,predicts=False)

        lr = optimizer.param_groups[0]['lr']
        # print('epoch\t%03d,lr : %.06f,loss: %.06f,T-acc: %.04f,T-f1: %.04f,acc: %.04f,f1: %.04f'%(epoch,
        #                 lr,train_loss,train_acc,train_f1,valid_acc,valid_f1))
        scheduler.step(valid_f1)
        lr = optimizer.param_groups[0]['lr']

        if valid_f1 >max_metric:
            max_metric=valid_f1
            tmp_file = pathjoin(save_folder,'models','model.pth')
            weights_list.append(tmp_file)
            t.save(model,tmp_file)
            max_metric_count=0
            max_metric=valid_f1
        else:
            if epoch_idx >=2: #ignore first two epochs
                max_metric_count+=1
            if max_metric_count >3:
                print('break at epoch',epoch)
                break 

        if lr <= 0.001:
            break 
    
    test_acc,test_f1,y_true,y_pred,y_output = test2(model,test_loader,predicts=True)
    # print('F1: %.03f,Acc: %.03f'%(test_acc,test_f1))
    
    np.savetxt(prob_file,y_output,)
    t.save(model,pathjoin(save_folder,'models','final_model.pth'))
    np.savetxt(pred_file,[str_labels[x] for x in np.reshape(y_pred,-1)],fmt="%s")
    np.savetxt(true_file,[str_labels[x] for x in np.reshape(y_true,-1)],fmt="%s")
    
    print('Output feature')
    #output edge weight
    pd.DataFrame(model.edge_weight.detach().numpy()).to_csv(".csv")
    #output feature and labels
    num_samples = logExpr.shape[0]
    dataset = ExprDataset(Expr=logExpr,edge=used_edge,y=label,device=device)
    gene_num = dataset.gene_num
    dataset1= DataLoader(dataset, batch_size=200,num_workers=0, shuffle=False,collate_fn = collate_func,drop_last=False)     
    model=torch.load('C:/Users/DELL/Desktop/PIGNN/results/models/final_model.pth').to(device)
    t=np.ones((1,1024))
    l=np.ones((1))
    for idx,data in enumerate(dataset1):
        output=weith(model, data)
        d=output.cpu().detach().numpy()
        t = np.concatenate([t,d], axis=0)
        numbers=data.y.detach().numpy() #(1000,1)
        number=np.squeeze(numbers)
        numbers=[str_labels[x] for x in np.reshape(numbers.T,-1)]
        numbers= np.array(numbers, dtype=np.object)
        l=np.concatenate([l,numbers], axis=0)
    t=np.delete(t,[0,0],axis=0)
    l=np.delete(l,0)
    results_inner = {'Xtrain':t.T, 'train_labels':l.T}
    sio.savemat('/trainz.mat', {'train':results_inner})
    
