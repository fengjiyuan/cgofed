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
        # Gradient Projections    %%%
        kk = 0 
        for k, (m,params) in enumerate(model.named_parameters()):
            if k<15 and len(params.size())!=1:
                sz =  params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
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
            output,_ = model(data)
            loss = criterion(output[task_id], target)
            pred = output[task_id].argmax(dim=1, keepdim=True) 
            
            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc

# def adjust_learning_rate(optimizer, epoch, lr, lr_factor):
#     for param_group in optimizer.param_groups:
#         if (epoch ==1):
#             param_group['lr']=lr
#         else:
#             param_group['lr'] /= lr_factor  

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return


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

def update_grad_basis (grad_list, threshold, grad_basis=[]):
    # print ('Threshold: ', threshold) 
    if not grad_basis:
        for i in range(len(grad_list)):
            activation = grad_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  
            grad_basis.append(U[:,0:r])    
    
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

class Client:
    def __init__(self, model, args):
        
        super(Client, self).__init__()
        # self.task_id = task_id
        self.model = model
        self.best_model = None
        self.best_loss = np.inf
        self.grad_basis = []
        self.criterion = torch.nn.CrossEntropyLoss()
        self.patience = None
        self.lr = args.lr        
        
        
        self.personalized_global_model = None
        self.curr_AvgProto = []
        self.history_AvgProto = []
        self.history_dis = {}
        self.dis_with_other = [0]*30
        self.history_model = []


    def train_first_task(self, args, xtrain,ytrain,xvalid,yvalid, task_id, device, threshold, c_id, g_epoch):
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
        self.grad_basis = update_grad_basis (grad_list, threshold,self.grad_basis)
        return 
    

    def train_new_task(self, args, xtrain,ytrain,xvalid,yvalid, task_id, device, threshold, c_id, old_model_list, g_epoch):
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)       
        reg_loss = Regularization(self.model, old_model_list, p=2).to(device)
        feature_mat = []
        # Projection Matrix Precomputation %%%
        for i in range(len(self.model.act)):
            # Uf改梯度
            Uf=torch.Tensor(np.dot(self.grad_basis[i],self.grad_basis[i].transpose())).to(device)
            # print('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
            feature_mat.append(Uf)
        print ('-'*40)
        for epoch in range(1, args.l_epochs+1):
            # Train 
            clock0=time.time()
            Proto = train_projected(args, self.model,device,xtrain, ytrain,optimizer,self.criterion,feature_mat,task_id, reg_loss)
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
        self.grad_basis = update_grad_basis (grad_list, threshold, self.grad_basis)

        return 

    
def vision_feature(model_params):
    layer_params = model_params
    plt.figure(figsize=(10, 10))
    # last_layer_params.cpu()
#     layer_params = [torch.tensor(param) for param in layer_params]
    i = 0
    for param in layer_params:
        plt.imshow(utils.make_grid(torch.tensor(param.cpu()), normalize=True).numpy().transpose((1, 2, 0)))
        plt.axis('off')
        plt.savefig("feature{}.png".format(i), dpi=300)
        plt.show()
        i = i+1
    return 