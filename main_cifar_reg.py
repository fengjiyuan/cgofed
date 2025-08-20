import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import heapq

import torchvision
from torchvision import datasets, transforms

import os
import os.path
from collections import OrderedDict
import copy
import matplotlib.pyplot as plt
import numpy as np

import random
import pdb
import argparse,time
import math
from copy import deepcopy

## Define AlexNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class AlexNet(nn.Module):
    def __init__(self,taskcla):
        super(AlexNet, self).__init__()
        self.act=OrderedDict()
        self.map =[]
        self.ksize=[]
        self.in_channel =[]
        self.map.append(32)
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        s=compute_conv_output_size(32,4)
        s=s//2
        self.ksize.append(4)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        s=compute_conv_output_size(s,3)
        s=s//2
        self.ksize.append(3)
        self.in_channel.append(64)
        self.map.append(s)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.ksize.append(2)
        self.in_channel.append(128)
        self.map.append(256*self.smid*self.smid)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256*self.smid*self.smid,2048, bias=False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.fc2 = nn.Linear(2048,2048, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)
        self.map.extend([2048])
        
        self.taskcla = taskcla
        self.fc3=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.fc3.append(torch.nn.Linear(2048,n,bias=False))
        
    def forward(self, x):
        bsz = deepcopy(x.size(0))
        self.act['conv1']=x
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        self.act['conv2']=x
        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        self.act['conv3']=x
        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x=x.view(bsz,-1)
        self.act['fc1']=x
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        self.act['fc2']=x        
        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))
        feature_x = x
        y=[]
        for t,i in self.taskcla:
            y.append(self.fc3[t](x))
            
        return y, feature_x


class Regularization(torch.nn.Module):
    def __init__(self, model, old_model_list, p=2):
        '''
        :param model 模型
        :param p: 范数计算中的幂指数值,默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        self.model = model
        self.old_model_list = old_model_list
        self.p = p
        self.weight_list = self.get_weight(model)

    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.old_model_list, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, old_model_list, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值, 默认求2范数
        :return:
        '''
        # selected_old_weight = select_old_model(index, ID, selected_clients_num)
        reg_loss = 0
        for (name1,w1), (name2, w2) in zip(weight_list, old_model_list):
            l2_reg = torch.norm(w1-w2, p=p)
            reg_loss = reg_loss + l2_reg
        return reg_loss

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        if (epoch ==1):
            param_group['lr']=args.lr
        else:
            param_group['lr'] /= args.lr_factor  

def train(args, model, device, x,y, optimizer,criterion, task_id):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
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
    r=torch.LongTensor(r).to(device)
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
        # zeta = 0.1
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

def PFL(curr_i, clients, clients_num):
    # 先归一化，然后计算出权重
    dis_sum = 0.
    dis = clients[curr_i].dis_with_other
    max_dis = max(dis)
    min_dis = min(filter(lambda x: x > 0, dis))
    for i in range(len(dis)):
        if dis[i] != 0:
            dis[i] = (dis[i] - min_dis) / (max_dis - min_dis)
    # 取倒数求和
    for i in range(len(dis)):
        if dis[i] != 0:
            dis_sum += 1.0/dis[i]   
    # 然后求权重
    for i in range(clients_num):
        if dis[i] != 0:
            clients[curr_i].dis_with_other[i] = (1.0/dis[i])/dis_sum
    
    w_g_personalized = copy.deepcopy(clients[curr_i].model.state_dict())  
    w_g_personalized.update((key, value * 0.9) for key, value in w_g_personalized.items()) # 个性化聚合中，自己的模型占一半权重，其他客户端的所有模型，占一半权重
    for k in w_g_personalized.keys():
        for i in range(clients_num):   #遍历本轮的所有客户端
            if i != curr_i and str(clients[i].model.state_dict()[k].dtype) == 'torch.float32' :       #排除客户端curr_i自己
                w_g_personalized[k] += 0.1 * clients[curr_i].dis_with_other[i] * clients[i].model.state_dict()[k]
    return w_g_personalized

def PFL_layer_aggre(curr_i, clients, clients_num):
    # 先归一化，然后计算出权重
    dis_sum = 0.
    dis = clients[curr_i].dis_with_other
    max_dis = max(dis)
    min_dis = min(filter(lambda x: x > 0, dis))
    for i in range(len(dis)):
        if dis[i] != 0:
            dis[i] = (dis[i] - min_dis) / (max_dis - min_dis)
    # 取倒数求和
    for i in range(len(dis)):
        if dis[i] != 0:
            dis_sum += 1.0/dis[i]   
    # 然后求权重
    for i in range(clients_num):
        if dis[i] != 0:
            clients[curr_i].dis_with_other[i] = (1.0/dis[i])/dis_sum
    
    w_g_personalized = copy.deepcopy(clients[curr_i].model.state_dict())   

    l = 0
    for key, value in w_g_personalized.items():
        if l < 6:
            # w_g_personalized.update(key, value * 0.9)
            w_g_personalized[key] = value * 0.5
        l = l+1
    l = 0
    for k in w_g_personalized.keys():
        if l < 6:
            for i in range(clients_num):   #遍历本轮的所有客户端
                if i != curr_i and str(clients[i].model.state_dict()[k].dtype) == 'torch.float32' :       #排除客户端curr_i自己
                    w_g_personalized[k] += 0.5 * clients[curr_i].dis_with_other[i] * clients[i].model.state_dict()[k]
        l = l+1




    # w_g_personalized.update((key, value * 0.9) for key, value in w_g_personalized.items()) # 个性化聚合中，自己的模型占一半权重，其他客户端的所有模型，占一半权重
    # for k in w_g_personalized.keys():
    #     for i in range(clients_num):   #遍历本轮的所有客户端
    #         if i != curr_i and str(clients[i].model.state_dict()[k].dtype) == 'torch.float32' :       #排除客户端curr_i自己
    #             w_g_personalized[k] += 0.1 * clients[curr_i].dis_with_other[i] * clients[i].model.state_dict()[k]
    return w_g_personalized


# def PFL_avg(curr_i, clients, clients_num):
   
#     w_g_personalized = copy.deepcopy(clients[curr_i].model.state_dict())   
#     w_g_personalized.update((key, value * 1) for key, value in w_g_personalized.items()) # 个性化聚合中，自己的模型占一半权重，其他客户端的所有模型，占一半权重
#     for k in w_g_personalized.keys():
#         for i in range(clients_num):   #遍历本轮的所有客户端
#             if i != curr_i and str(clients[i].model.state_dict()[k].dtype) == 'torch.float32' :       #排除客户端curr_i自己
#                 w_g_personalized[k] += 0.0  * clients[i].model.state_dict()[k]
#     return w_g_personalized


# def personalized_aggregative(curr_i, clients, clients_num):
    
    
#     param_list = []
#     for i in range(clients_num):
#         if i == curr_i:
#             continue
#         else:
#             param_list.append(clients[i].state_dict())
    
#     weights = [1/clients_num]*clients_num
#     weighted_average_params = clients[curr_i].state_dict()
#     for params, weight in zip(param_list, weights):
#         for key, value in params.items():
#             weighted_average_params[key] += value * weight

#     return weighted_average_params

def compute_distance_curr_feature(curr_i, curr_feature, clients, clients_participant):
    # dis = []
    for i in range(clients_participant):   #遍历除当前client的其他客户端
        if i != curr_i :      
            # 遍历模型的每一层参数
            d = np.linalg.norm(curr_feature-clients[i].curr_AvgProto, ord=2, axis=None, keepdims=False)  # 原型保存为numpy
            # d = torch.dist(curr_feature, clients[i].curr_AvgProto, p=2)      # 原型保存为tensor
            clients[curr_i].dis_with_other[i] = float(d)
    return 

def compute_distance_with_history_AvgProto(curr_client, clients, clients_participant, task_id):
    for r in range(task_id+1):            # 遍历其他客户端的历史模型
        dis = []
        for i in range(clients_participant):          #遍历除当前client的其他客户端
            if i == curr_client:
                d = 0
                dis.append(float(d))
            else:
                d = np.linalg.norm(clients[curr_client].curr_AvgProto - clients[i].history_AvgProto[r], ord=2, axis=None, keepdims=False)  # 原型保存为numpy
                dis.append(float(d))
            # 算出了当前平均原型和其他客户端的历史平均原型的距离
        clients[curr_client].history_dis[r] = dis

def select_old_model(clients, curr_i, task_id, clients_participant, selected_clients_num):
    # 根据上一个循环计算出的关系，为每个客户端挑选需要加入正则化项的历史模型
    sum_value = 0.
    all_round_selected_list = []                # 保存被选中的历史原型，包括客户端编号，以及该历史原型与当前原型的距离
    client_list = [x for x in range(clients_participant)]
    for id in range(task_id):
        min_index = list(map(clients[curr_i].history_dis[id].index, 
                            heapq.nsmallest(selected_clients_num, clients[curr_i].history_dis[id])))
        min_index.remove(curr_i)
        selected_clients = {}
        for ind in min_index:
            selected_clients[client_list[ind]] = clients[curr_i].history_dis[id][ind]
            sum_value += clients[curr_i].history_dis[id][ind]
        all_round_selected_list.append(selected_clients)
        # 返回值为list类型，其长度是迭代训练的次数，每个元素是一个字典，键是客户端编号，值与当前客户端的距离
    # 选完客户端之后，计算正则化项
    avg_selected_model = {}
    for r in range(len(all_round_selected_list)):            # 用选中的历史模型计算正则化项
        for key, value in all_round_selected_list[r].items():        
            for layer_name, param in clients[key].history_model[r].items():     # 历史模型从内存中读取，self.history_model
                if r == 0:
                    avg_selected_model[layer_name] = value/sum_value * param
                else:
                    avg_selected_model[layer_name] = avg_selected_model[layer_name] + value/sum_value * param
    
    old_weight_list = []        
    for name, param in avg_selected_model.items():  # 只用weight参数，不用bias参数
        if ("weight" in name):
            weight = (name, param)
            old_weight_list.append(weight)
    return old_weight_list

def test(args, model, device, x, y, criterion, task_id):
    model.eval()
    total_loss = 0
    total_num = 0 
    correct = 0
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
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

def get_representation_matrix (net, device, x, y=None): 
    # Collect activations by forward pass
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:125] # Take 125 random samples 
    example_data = x[b]
    example_data = example_data.to(device)
    example_out  = net(example_data)
    
    batch_list=[2*12,100,100,125,125] 
    mat_list=[]
    act_key=list(net.act.keys())
    for i in range(len(net.map)):
        bsz=batch_list[i]
        k=0
        if i<3:
            ksz= net.ksize[i]
            s=compute_conv_output_size(net.map[i],net.ksize[i])
            mat = np.zeros((net.ksize[i]*net.ksize[i]*net.in_channel[i],s*s*bsz))
            act = net.act[act_key[i]].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:,k]=act[kk,:,ii:ksz+ii,jj:ksz+jj].reshape(-1) 
                        k +=1
            mat_list.append(mat)
        else:
            act = net.act[act_key[i]].detach().cpu().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)

    # print('-'*30)
    # print('Representation Matrix')
    # print('-'*30)
    # for i in range(len(mat_list)):
    #     print ('Layer {} : {}'.format(i+1,mat_list[i].shape))
    # print('-'*30)
    return mat_list    


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

def update_CGoFed (args, model, mat_list, threshold, task_id, feature_list=[], importance_list=[]):
    # plt.figure(figsize=(10, 6))
    # print ('Threshold: ', threshold) 
    our_importance_list = []
    if not feature_list:
        # After First Task 
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-1)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  
            # update GPM
            feature_list.append(U[:,0:r])
            # update importance (Eq-2)
            # importance = ((args.scale_coff+1)*S[0:r])/(args.scale_coff*S[0:r] + max(S[0:r])) 
            # importance_list.append(importance)
            
            importance_list.append(np.array(intra_task2_basis_weight(S[0:r], args.param_alpha)))
            
            
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            # Projected Representation (Eq-4)
            act_proj = np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
            r_old = feature_list[i].shape[1] # old GPM bases 
            Uc,Sc,Vhc = np.linalg.svd(act_proj, full_matrices=False)
            importance_new_on_old = np.dot(np.dot(feature_list[i].transpose(),Uc[:,0:r_old])**2, Sc[0:r_old]**2) ## r_old no of elm s**2 fmt
            importance_new_on_old = np.sqrt(importance_new_on_old)
            
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
            if r == 0:
                print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                # update importances 
                importance = importance_new_on_old
                # importance = ((args.scale_coff+1)*importance)/(args.scale_coff*importance + max(importance)) 
                importance = intra_task2_basis_weight(importance, args.param_alpha)
                # importance [0:r_old] = np.clip(importance [0:r_old]+importance_list[i][0:r_old], 0, 1)
                importance [0:r_old] = np.clip(importance [0:r_old], 0, 1)
                importance_list[i] = importance # update importance
                continue
            
            # 利用与其他客户端的历史任务的相似度给历史任务的基向量加权                 
            # update GPM
            Ui=np.hstack((feature_list[i],U[:,0:r]*(args.beta**(task_id))))  
            # update importance 
            importance = np.hstack((importance_new_on_old,S[0:r]))
            # importance = ((args.scale_coff+1)*importance)/(args.scale_coff*importance + max(importance))         
            importance = intra_task2_basis_weight(importance, args.param_alpha)
            importance [0:r_old] = np.clip(importance [0:r_old]+importance_list[i][0:r_old], 0, 1) 

            if Ui.shape[1] > Ui.shape[0] :
                feature_list[i]=Ui[:,0:Ui.shape[0]]
                importance_list[i] = importance[0:Ui.shape[0]]
            else:
                feature_list[i]=Ui
                importance_list[i] = importance

    return feature_list, importance_list  

def set_seed(seed):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return


class Client:
    def __init__(self, model, args):
        
        super(Client, self).__init__()
        # self.task_id = task_id
        self.model = model
        self.best_model = None
        self.best_loss = np.inf
        self.feature_list = []
        self.grad_basis = []
        self.criterion = torch.nn.CrossEntropyLoss()
        self.patience = None
        self.lr = args.lr  
        self.hyperparam = True      
        # args.lr_patience
        
        self.personalized_global_model = None
        self.curr_AvgProto = []
        self.history_AvgProto = []
        self.history_dis = {}
        self.dis_with_other = [0]*30
        self.history_model = []
        self.history_mat_list = []
        self.history_representation_matrix = []
        self.weight_lambda = []
        self.feature_list = []
        self.importance_list = []


    def train_first_task(self, args, xtrain,ytrain,xvalid,yvalid,xtest,ytest, task_id, device, threshold, c_id ,old_model_list, g_round, lr, best_loss):
        best_model=get_model(self.model)
        feature_list =[]
        importance_list = []
        optimizer = optim.SGD(self.model.parameters(), lr=lr)

        for epoch in range(1, args.local_epochs+1):
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
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=get_model(self.model)
                patience=args.lr_patience
                print(' *',end='')
            else:
                patience-=1
                if patience<=0:
                    lr/=args.lr_factor
                    print(' lr={:.1e}'.format(lr),end='')
                    if lr<args.lr_min:
                        print()
                        break
                    patience=args.lr_patience
                    adjust_learning_rate(optimizer, epoch, args)
            print()                    
        set_model_(self.model,best_model)

        # 本地训练中，保存最后一个epoch的feature
        self.curr_AvgProto = Proto.mean(dim=0).detach().cpu().numpy()
        if g_round == args.global_rounds-1:
            # 应该在所有的epochs结束之后返回特征和模型
            self.history_AvgProto.append(self.curr_AvgProto) # 保存历史类原型
            self.history_model.append(self.model.state_dict())

        # Test
        test_loss, test_acc = test(args, self.model, device, xtest, ytest,  self.criterion, task_id)
        print('G_round {:1d} | Client {:3d} | Test: loss={:.3f} , acc={:5.1f}%'.format(g_round, c_id, test_loss,test_acc))  
        print ('-'*40)
        # Memory and Importance Update  
        mat_list = get_representation_matrix(self.model, device, xtrain, ytrain)
        self.feature_list, self.importance_list = update_CGoFed(args, self.model, mat_list, threshold, task_id, feature_list, importance_list)
        return
    
    def train_next_task(self, args, xtrain,ytrain,xvalid,yvalid,xtest,ytest, task_id, device, threshold, c_id ,old_model_list, g_round, lr, best_loss):
        # model = client_model[c_id]
        optimizer = optim.SGD(self.model.parameters(), lr=args.lr)
        reg_loss = Regularization(self.model, old_model_list, p=2).to(device)

        feature_mat = []
        for i in range(len(self.model.act)):
            if len(self.importance_list[i]) !=  self.feature_list[i].shape[1]:
                self.importance_list[i] = self.importance_list[i][:self.feature_list[i].shape[1]]
            Uf=torch.Tensor(np.dot(self.feature_list[i],np.dot(np.diag(self.importance_list[i]),self.feature_list[i].transpose()))).to(device) 
            # print('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
            Uf.requires_grad = False
            feature_mat.append(Uf)
        for epoch in range(1, args.local_epochs+1):
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
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=get_model(self.model)
                patience=args.lr_patience
                print(' *',end='')
            else:
                patience-=1
                if patience<=0:
                    lr/=args.lr_factor
                    print(' lr={:.1e}'.format(lr),end='')
                    if lr<args.lr_min:
                        print()
                        break
                    patience=args.lr_patience
                    adjust_learning_rate(optimizer, epoch, args)
            print()
        set_model_(self.model,best_model)

        # 本地训练中，保存最后一个epoch的feature
        self.curr_AvgProto = Proto.mean(dim=0).detach().cpu().numpy()        
        if g_round == args.global_rounds-1:
            # 应该在所有的epochs结束之后返回特征和模型
            self.history_AvgProto.append(self.curr_AvgProto) # 保存历史类原型
            self.history_model.append(self.model.state_dict())



        # Test 
        test_loss, test_acc = test(args, self.model, device, xtest, ytest,  self.criterion,task_id)
        print('G_round {:1d} | Client {:3d} | Test: loss={:.3f} , acc={:5.1f}%'.format(g_round, c_id, test_loss,test_acc))  
        print ('-'*40)
        # Memory and Importance Update 
        mat_list = get_representation_matrix (self.model, device, xtrain, ytrain)
        self.feature_list, self.importance_list = update_CGoFed (args, self.model, mat_list, threshold, task_id, self.feature_list, self.importance_list)
        return
    
    
def fed_main(args):
    tstart=time.time()
    ## Device Setting 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    ## setup seeds
    set_seed(args.seed)
    ## Load CIFAR100 DATASET
    from dataloader import cifar100 as cf100
    ## Distribute data to the client
    data_set, taskcla = cf100.new_get(args.increment_mode, seed=args.seed, 
                                            pc_valid=args.pc_valid, clients_num=args.clients_num)

    acc_matrix=np.zeros((10,10))
  
    global_model = []
    clients = []
    ## Create the client class
    for c_id in range(args.clients_num):
        global_model.append(AlexNet(taskcla).to(device))
        clients.append(Client(global_model[c_id], args))


    task_id = 0
    task_list = []
    for k,ncla in taskcla:
        # specify threshold hyperparameter
        threshold = np.array([args.gpm_eps] * 5) + task_id * np.array([args.gpm_eps_inc] * 5)
        task_list.append(k)

        lr = args.lr 
        best_loss=np.inf
        print('*'*100)
        print ('Task ID :{:2d} ({:s}) | Learning Rate : {}'.format(task_id, data_set[k][0]['name'], lr))
        print('*'*100)

        for g_round in range(args.global_rounds):
            for c_id in range (args.clients_num):
                xtrain=data_set[k][c_id]['train']['x'].to(device)
                ytrain=data_set[k][c_id]['train']['y'].to(device)
                xvalid=data_set[k][c_id]['valid']['x'].to(device)
                yvalid=data_set[k][c_id]['valid']['y'].to(device)       
                xtest =data_set[k][c_id]['test']['x'].to(device)
                ytest =data_set[k][c_id]['test']['y'].to(device)                         
                
                lr = args.lr 
                best_loss=np.inf

                if task_id==0:
                    old_model_list = []
                    clients[c_id].train_first_task(args, xtrain,ytrain,xvalid,yvalid,xtest,ytest, k, device, threshold, c_id, old_model_list, g_round, lr, best_loss)
                else:
                    old_model_list = select_old_model(clients, c_id, task_id, args.clients_num, args.selected_clients)
                    clients[c_id].train_next_task(args, xtrain,ytrain,xvalid,yvalid,xtest,ytest, k, device, threshold, c_id, old_model_list, g_round, lr, best_loss)                
            
            # 计算原型之间的距离
            for c_id in range(args.clients_num):
                compute_distance_curr_feature(c_id, clients[c_id].curr_AvgProto, clients, args.clients_num)
                if g_round == args.global_rounds-1:       # ep_g%10=0,表示只保存和每个任务第一轮的距离及其原型，ep_g%10=9表示最后一轮
                    compute_distance_with_history_AvgProto(c_id, clients, args.clients_num, task_id)   


            # personalized aggregation
            for c_id in range(args.clients_num):
                clients[c_id].personalized_global_model = PFL_layer_aggre(c_id, clients, args.clients_num)
                global_model[c_id].load_state_dict(clients[c_id].personalized_global_model)
            
            
            # print ('-'*40)          
            #     print('Task: {:3d}, G-Epoch: {:3d}, Test: loss={:.3f} , acc={:5.1f}%'.format(ii, g_round, avg_test_loss/args.clients_num,avg_test_acc/args.clients_num))

        # save accuracy 
        jj=0
        for ii in np.array(task_list)[0:task_id+1]:
            avg_acc = 0
            for c_id in range(args.clients_num):
                xtest =data_set[ii][c_id]['test']['x'].to(device)
                ytest =data_set[ii][c_id]['test']['y'].to(device) 
                _, acc = test(args, clients[c_id].model, device, xtest, ytest,clients[c_id].criterion,ii) 
                avg_acc += acc
            acc_matrix[task_id,jj] = avg_acc/args.clients_num
            jj+= 1

        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a,j_a]),end='')
            print()
            # update task id 
        task_id +=1   
    print('-'*50)

    # Simulation Results 
    print ('Task Order : {}'.format(np.array(task_list)))
    print("Configs: seed: {} | increment_mode: {} | param_alpha: {} | beta: {} | zeta: {}".format(args.seed,args.increment_mode,args.param_alpha,args.beta,args.zeta)) 
    print ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean())) 
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    print ('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    print('-'*50)



if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='CGoFed')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=5, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--seed', type=int, default=5, metavar='S',
                        help='random seed (default: 5)')
    parser.add_argument('--pc_valid',default=0.05,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=5e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    # SGP/GPM specific 
    parser.add_argument('--scale_coff', type=int, default=10, metavar='SCF',
                        help='importance co-efficeint (default: 10)')
    parser.add_argument('--gpm_eps', type=float, default=0.97, metavar='EPS',
                        help='threshold (default: 0.97)')
    parser.add_argument('--gpm_eps_inc', type=float, default=0.003, metavar='EPSI',
                        help='threshold increment per task (default: 0.003)')
    parser.add_argument('--param_alpha', type=int, default=3, metavar='alpha',
                        help='importance co-efficeint (default: 5)')
    parser.add_argument('--zeta', type=float, default=0.08, metavar='zeta',
                        help='angle of projection (default: 0.1)')
    parser.add_argument('--beta', type=float, default=0.92, metavar='zeta',
                        help='weight of basis of each task (default: 0.9)')


    # FL specific
    parser.add_argument('--clients_num', type=int, default=5, metavar='CN',
                        help='the number of clients (default: 10)')
    parser.add_argument('--local_epochs', type=int, default=50, metavar='N',
                        help='the number of training epochs/clients (default: 1)')
    parser.add_argument('--global_rounds', type=int, default=5, metavar='N',
                        help='the number of training rounds/task (default: 5)')
    parser.add_argument('--increment_mode', type=str, default='SL', metavar='N',
                        help='incremental data is the same label or different labels (default: SL)')
    parser.add_argument('--selected_clients', type=int, default=2, metavar='N',
                        help='select the similar client (default: 2)')

    args = parser.parse_args()
    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)

    fed_main(args)


