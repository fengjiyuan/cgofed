import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import torchvision.utils as utils

import os
import os.path
from collections import OrderedDict
from myModel import AlexNet
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import random
import pdb
import argparse,time
import math
from copy import deepcopy
from server import *
from myModel import *
from server import Regularization


def train(args, model, device, x,y, optimizer,criterion, task_id):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output, proto = model(data)
        loss = criterion(output[task_id], target)        
        loss.backward()
        optimizer.step()
        # 保存特征图
        if i == 0:
            Proto = proto
        else:
            Proto = torch.cat((Proto, proto), dim =0)
    return Proto


def train_projected(args,model,device,x,y,optimizer,criterion,feature_mat,task_id, reg_loss):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output, proto = model(data)
        local_loss_value = criterion(output[task_id], target)   

        # 在这里读取旧任务的模型，并计算正则化损失
        reg_loss_value = reg_loss(model)
        b1 = len(str(int(local_loss_value*1000000)))    
        b2 = len(str(int(reg_loss_value*1000000)))
        loss_value = local_loss_value + reg_loss_value/(10**(b2-b1+2))*2    # 加入正则化项的损失

        loss_value.backward()
        # Gradient Projections 
        kk = 0 
        for k, (m,params) in enumerate(model.named_parameters()):
            if k<15 and len(params.size())!=1:
                sz =  params.grad.data.size(0)
                params.grad.data = params.grad.data - (1-args.zeta) * torch.mm(params.grad.data.view(sz,-1),\
                                                        feature_mat[kk]).view(params.size())
                kk +=1
            elif (k<15 and len(params.size())==1) and task_id !=0 :
                params.grad.data.fill_(0)

        optimizer.step()
        # 保存特征图
        if i == 0:
            Proto = proto
        else:
            Proto = torch.cat((Proto, proto), dim = 0)
    return Proto


def test(args, model, device, x, y, criterion, task_id):
    model.eval()
    total_loss = 0
    total_num = 0 
    correct = 0
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r)
    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_test):
            if i+args.batch_size_test<=len(r): b=r[i:i+args.batch_size_test]
            else: b=r[i:]
            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output, _ = model(data)
            loss = criterion(output[task_id], target)
            pred = output[task_id].argmax(dim=1, keepdim=True) 
            
            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc


def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

# def adjust_learning_rate(optimizer, epoch, args):
#     for param_group in optimizer.param_groups:
#         if (epoch ==1):
#             param_group['lr']=args.lr
#         else:
#             param_group['lr'] /= args.lr_factor  

# def get_grad_matrix (net, device, x, y=None): 
#     # Collect activations by forward pass
#     r=np.arange(x.size(0))
#     np.random.shuffle(r)
#     r=torch.LongTensor(r)
#     b=r[0:125] # Take 125 random samples 
#     example_data = x[b]
#     example_data = example_data.to(device)
#     example_out  = net(example_data)
    
#     batch_list=[2*12,100,100,125,125] 
#     grad_list=[]
#     act_key=list(net.act.keys())
#     for i in range(len(net.map)):
#         bsz=batch_list[i]
#         k=0
#         if i<3:
#             ksz= net.ksize[i]
#             s=compute_conv_output_size(net.map[i],net.ksize[i])
#             mat = np.zeros((net.ksize[i]*net.ksize[i]*net.in_channel[i],s*s*bsz))
#             act = net.act[act_key[i]].detach().cpu().numpy()
#             for kk in range(bsz):
#                 for ii in range(s):
#                     for jj in range(s):
#                         mat[:,k]=act[kk,:,ii:ksz+ii,jj:ksz+jj].reshape(-1) 
#                         k +=1
#             grad_list.append(mat)
#         else:
#             act = net.act[act_key[i]].detach().cpu().numpy()
#             activation = act[0:bsz].transpose()
#             grad_list.append(activation)
#     return grad_list 


def get_grad_matrix (net, device, x, y=None): 
    # Collect activations by forward pass
    r=np.arange(x.size(0))            # 样本数量
    np.random.shuffle(r)
    r=torch.LongTensor(r)
    b=r[0:125] # Take 125 random samples 
    example_data = x[b]
    example_data = example_data.to(device)
    example_out  = net(example_data)
    
    grad_list=[] # list contains gradient of each layer

    for _, (m,params) in enumerate(net.named_parameters()):
        
        if 'weight' in m and 'bn' not in m and 'fc3' not in m:
            sz =  params.grad.data.size(0)
            grad = params.grad.data.view(sz,-1).detach().cpu().numpy()
            activation = grad.transpose()
            grad_list.append(activation)
    return grad_list   


def intra_task2_basis_weight(singular_value, param_alpha):
    # param_alpha = 5
    weight_lambda = []
    singular_value = [value ** (1/param_alpha) for value in singular_value]
    max_sv = max(singular_value)
    for sv in singular_value:
        scaled_value = sv / max_sv        
        weight_lambda.append(scaled_value)
    return weight_lambda   

# 为每个任务的基向量计算整体的衰减参数
def inter_task_basis_weight(curr_task):
    mu_1 = 1
    beta = 0.9
    mu_i = mu_1* (beta** (curr_task))
    return mu_i


def update_grad_basis_our (args, grad_list, threshold, grad_basis=[], importance_list=[]):
    # print ('Threshold: ', threshold) 
    if not grad_basis:
        for i in range(len(grad_list)):
            activation = grad_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  
            grad_basis.append(U[:,0:r])    
            # our add
            importance_list.append(np.array(intra_task2_basis_weight(S[0:r], args.param_alpha)))
    else:
        for i in range(len(grad_list)):
            activation = grad_list[i]
            U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()            
            
            
            act_hat = activation - np.dot(np.dot(grad_basis[i],grad_basis[i].transpose()),activation)
            U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
            
            
            # criteria (Eq-9)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total               
            accumulated_sval = (sval_total-sval_hat)/sval_total    
    
            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            
            
            
            Ui=np.hstack((grad_basis[i],U[:,0:r]))  
            if Ui.shape[1] > Ui.shape[0] :
                grad_basis[i]=Ui[:,0:Ui.shape[0]]
            else:
                grad_basis[i]=Ui    
    return grad_basis  

def update_grad_basis (args, grad_list, threshold, task_id, grad_basis=[], importance_list=[]):
    # print ('Threshold: ', threshold) 
    our_importance_list = []
    if not grad_basis:
        for i in range(len(grad_list)):
            activation = grad_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  
            grad_basis.append(U[:,0:r])    
            
            importance_list.append(np.array(intra_task2_basis_weight(S[0:r], args.param_alpha)))    
    else:
        for i in range(len(grad_list)):
            activation = grad_list[i]
            U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            
            
            # Projected Representation (Eq-4)
            act_proj = np.dot(np.dot(grad_basis[i],grad_basis[i].transpose()),activation)
            r_old = grad_basis[i].shape[1] # old GPM bases 
            # Uc,Sc,Vhc = np.linalg.svd(act_proj, full_matrices=False)
            # importance_new_on_old = np.dot(np.dot(grad_basis[i].transpose(),Uc[:,0:r_old])**2, Sc[0:r_old]**2) ## r_old no of elm s**2 fmt
            # importance_new_on_old = np.sqrt(importance_new_on_old)
            
            act_hat = activation - act_proj
            U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
            
            
            # criteria (Eq-5)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total               
            accumulated_sval = (sval_total-sval_hat)/sval_total
            
            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            # if r == 0:
            #     print ('Skip Updating GPM for layer: {}'.format(i+1)) 
            #     # update importances 
            #     # importance = importance_new_on_old
            #     # importance = ((args.scale_coff+1)*importance)/(args.scale_coff*importance + max(importance)) 
            #     importance = intra_task2_basis_weight(importance, args.param_alpha)
            #     # importance [0:r_old] = np.clip(importance [0:r_old]+importance_list[i][0:r_old], 0, 1)
            #     importance [0:r_old] = np.clip(importance [0:r_old], 0, 1)
            #     importance_list[i] = importance # update importance
            #     continue
            
            # 利用与其他客户端的历史任务的相似度给历史任务的基向量加权                 
            # update GPM
            Ui=np.hstack((grad_basis[i],U[:,0:r]))
            # Ui=np.hstack((grad_basis[i],U[:,0:r]*(args.beta**(task_id))))  
            # update importance 
            # importance = np.hstack((importance_new_on_old,S[0:r]))
            # importance = ((args.scale_coff+1)*importance)/(args.scale_coff*importance + max(importance))         
            # importance = intra_task2_basis_weight(importance, args.param_alpha)
            # importance [0:r_old] = np.clip(importance [0:r_old]+importance_list[i][0:r_old], 0, 1) 

            if Ui.shape[1] > Ui.shape[0] :
                grad_basis[i]=Ui[:,0:Ui.shape[0]]
                # importance_list[i] = importance[0:Ui.shape[0]]
            else:
                grad_basis[i]=Ui
                # importance_list[i] = importance

            # Calculate and store weights for the updated basis
            new_weights = calculate_weights(Ui, S, args.param_alpha)
            importance_list[i] = new_weights  # Update the importance list with new weights
            importance_list.append(np.array(intra_task2_basis_weight(S[0:r], args.param_alpha)))


    return grad_basis, importance_list  


class Client:
    def __init__(self, model, args):
        
        super(Client, self).__init__()
        # self.task_id = task_id
        self.model = model
        self.best_model = None
        self.best_loss = np.inf
        self.criterion = torch.nn.CrossEntropyLoss()
        self.patience = None
        self.lr = args.lr        
        
        
        self.personalized_global_model = None
        self.curr_AvgProto = []
        self.history_AvgProto = []
        self.history_dis = {}
        self.dis_with_other = [0]*30
        self.history_model = []
        self.weight_lambda = []
        # self.feature_list = []
        self.grad_basis = []
        self.importance_list = []


    def train_first_task(self, args, xtrain,ytrain,xvalid,yvalid, task_id, device, threshold, c_id, g_epoch):
        importance_list = []
        grad_basis =[]
        
        self.model = self.model.to(device)
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)        
        for epoch in range(1, args.l_epochs+1):
            # Train
            clock0=time.time()
            Proto = train(args, self.model, device, xtrain, ytrain, optimizer, self.criterion, task_id)
            clock1=time.time()
            tr_loss,tr_acc = test(args, self.model, device, xtrain, ytrain,  self.criterion, task_id)
            print('Epoch {:3d} | Client {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        c_id, tr_loss,tr_acc, 1000*(clock1-clock0)),end='')
            # Validate
            valid_loss,valid_acc = test(args, self.model, device, xvalid, yvalid,  self.criterion, task_id)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
            print()
    
        # 本地训练中，保存最后一个epoch的feature
        self.curr_AvgProto = Proto.mean(dim=0).detach().cpu().numpy()
        if g_epoch == args.g_epochs-1:
            # 应该在所有的epochs结束之后返回特征和模型
            self.history_AvgProto.append(self.curr_AvgProto) # 保存历史类原型
            self.history_model.append(self.model.state_dict())
        # Memory Update   %%%
        grad_list = get_grad_matrix (self.model, device, xtrain, ytrain)
        self.grad_basis , self.importance_list = update_grad_basis (args, grad_list, threshold,task_id, grad_basis, importance_list)
        return 
    

    def train_new_task(self, args, xtrain,ytrain,xvalid,yvalid, task_id, device, threshold, c_id, old_model_list, g_epoch):
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)       
        reg_loss = Regularization(self.model, old_model_list, p=2).to(device)
        grad_basis = []
        # Projection Matrix Precomputation %%%
        for i in range(len(self.model.act)):
            if len(self.importance_list[i]) !=  self.grad_basis[i].shape[1]:
                self.importance_list[i] = self.importance_list[i][:self.grad_basis[i].shape[1]]
            Uf=torch.Tensor(np.dot(self.grad_basis[i],np.dot(np.diag(self.importance_list[i]),self.grad_basis[i].transpose()))).to(device) 
            # Uf.requires_grad = False
            grad_basis.append(Uf)
            # # Uf改梯度
            # Uf=torch.Tensor(np.dot(self.grad_basis[i],self.grad_basis[i].transpose())).to(device)
            # print('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
        print ('-'*40)
        for epoch in range(1, args.l_epochs+1):
            # Train 
            clock0=time.time()
            Proto = train_projected(args, self.model,device,xtrain, ytrain,optimizer,self.criterion,grad_basis,task_id, reg_loss)
            clock1=time.time()
            tr_loss, tr_acc = test(args, self.model, device, xtrain, ytrain,self.criterion,task_id)
            print('Epoch {:3d} | Client {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                    c_id, tr_loss, tr_acc, 1000*(clock1-clock0)),end='')
            # Validate
            valid_loss,valid_acc = test(args, self.model, device, xvalid, yvalid, self.criterion,task_id)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
            print()
        
        # 本地训练中，保存最后一个epoch的feature
        self.curr_AvgProto = Proto.mean(dim=0).detach().cpu().numpy()        
        if g_epoch == args.g_epochs-1:
            # 应该在所有的epochs结束之后返回特征和模型
            self.history_AvgProto.append(self.curr_AvgProto) # 保存历史类原型
            self.history_model.append(self.model.state_dict())
        # Memory Update  %%%
        grad_list = get_grad_matrix (self.model, device, xtrain, ytrain)
        self.grad_basis , self.importance_list = update_grad_basis (args, grad_list, threshold, task_id, self.grad_basis, self.importance_list)

        return 

















