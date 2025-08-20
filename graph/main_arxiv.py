# -*- coding: utf-8 -*-
# 使用新的数据加载函数
import torch
import numpy as np
import argparse
import torch.nn as nn
import heapq
import os
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import argparse,time
import os.path
from collections import OrderedDict
from copy import deepcopy
import copy
from data_stream import setup_seed, set_task
from torch_sparse import SparseTensor


## Define AlexNet model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def get_model(model):
    return deepcopy(model.state_dict())


def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def get_grad_matrix(model, device, task, memories):
    _, _ = model(task.x, task.edge_index)
    for i, memory in enumerate(memories):
        memory = memory.to(device, "x", "y", "adj_t")
        output, _ = model(memory.x, memory.edge_index) 
    
    target_layers = ['conv1', 'conv2']
    grad_list = []
    for _, (m,params) in enumerate(model.named_parameters()):
        if any(layer in m for layer in target_layers) and "bias" not in m and params.grad is not None:
            sz =  params.grad.size(0)
            grad = params.grad.view(sz,-1).detach()
            activation = grad.T
            grad_list.append(activation)
    return grad_list

def update_grad_basis (grad_list, threshold, grad_basis=[]):
    # print ('Threshold: ', threshold) 
    if not grad_basis:
        for i in range(len(grad_list)):
            activation = grad_list[i]
            U,S,Vh = np.linalg.svd(activation.to('cpu'), full_matrices=False)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  
            grad_basis.append(U[:,0:r])    
    
    else:
        for i in range(len(grad_list)):
            activation = grad_list[i].to('cpu') 
            U1,S1,Vh1=np.linalg.svd(activation.to('cpu'), full_matrices=False)
            sval_total = (S1**2).sum()   
            grad_list_i = grad_list[i].to('cpu')         
            act_hat = activation - np.dot(np.dot(grad_list_i, grad_list_i.T),activation)
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
            Ui=np.hstack((grad_list_i,U[:,0:r]))  
            if Ui.shape[1] > Ui.shape[0] :
                grad_basis[i]=Ui[:,0:Ui.shape[0]]
            else:
                grad_basis[i]=Ui    
    return grad_basis  

def sigmoid_np(beta, x):
    return 1 / (1 + np.exp(-x))
def sigmoid(beta, x):
    return torch.sigmoid(beta * x)

def update_grad_basis_calculate_important_sigmoid (args, grad_list, threshold, task_id, grad_basis=[], importance_list=[]):
    # print ('Threshold: ', threshold) 
    # scale_coff = 10
    if not grad_basis:
        for i in range(len(grad_list)):
            activation = grad_list[i]
            U,S,Vh = torch.linalg.svd(activation, full_matrices=False)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            sval_ratio = torch.cumsum(sval_ratio, dim=0)  # 使用 torch.cumsum 替代 np.cumsum
            r = torch.sum(sval_ratio < threshold[i]).item()  # 使用 torch.sum 替代 np.sum            
            grad_basis.append(U[:,0:r])    
            
            # 以下为新增
            importance = sigmoid(args.beta,  S[0:r])
            importance_list.append(importance)

    else:
        for i in range(len(grad_list)):
            
            activation = grad_list[i].to('cpu')
            U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()   
            if isinstance(grad_basis[i], torch.Tensor):
                g_basis = grad_basis[i].to('cpu')
            elif isinstance(grad_basis[i], np.ndarray):
                g_basis = grad_basis[i]
            # g_basis = grad_basis[i].to('cpu')  
            act_proj = np.dot(np.dot(g_basis,g_basis.T),activation)
            
            # 以下为新增
            r_old = g_basis.shape[1] # old GPM bases 
            Uc,Sc,Vhc = np.linalg.svd(act_proj, full_matrices=False)
            importance_new_on_old = np.dot(np.dot(g_basis.T, Uc[:,0:r_old])**2, Sc[0:r_old]**2) ## r_old no of elm s**2 fmt
            importance_new_on_old = np.sqrt(importance_new_on_old)            
                   
            act_hat = activation - act_proj
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
            
            # 以下为新增
            if r == 0:
                print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                # update importances 
                importance = importance_new_on_old
                importance = sigmoid_np(args.beta, importance)  
                importance [0:r_old] = np.clip(importance [0:r_old]+importance_list[i][0:r_old], 0, 1)
                importance_list[i] = importance # update importance
                continue
           
           # update GPM
            Ui=np.hstack((g_basis,U[:,0:r]))  
            # update importance  以下为新增
            if isinstance(importance_list[i], torch.Tensor):
                importance_list_i = importance_list[i].to('cpu')
            elif isinstance(importance_list[i], np.ndarray):
                importance_list_i = importance_list[i]

            importance = np.hstack((importance_new_on_old,S[0:r]))
            importance = sigmoid_np(args.beta, importance)      
            importance [0:r_old] = np.clip(importance [0:r_old]+np.array(importance_list_i[0:r_old]), 0, 1) 
           
            if Ui.shape[1] > Ui.shape[0] :
                grad_basis[i]=Ui[:,0:Ui.shape[0]]

                importance_list[i] = importance[0:Ui.shape[0]]
            else:
                grad_basis[i]=Ui    
                importance_list[i] = importance

    return grad_basis, importance_list


def calculate_average_forgetting(matrix, task_num):
    n = len(matrix)
    if n == 0:
        return 0

    total_difference = 0
    for col in range(n):
        max_value = max(row[col] for row in matrix)
        last_value = matrix[task_num][col]
        difference = max_value - last_value
        total_difference += difference

    average_difference = total_difference / ((task_num+1)*100)
    return average_difference


def compute_distance_curr_feature(curr_i, curr_feature, clients, clients_participant):
    # dis = []
    for i in range(clients_participant):   #遍历除当前client的其他客户端
        if i != curr_i :      
            # 遍历模型的每一层参数
            d = np.linalg.norm(curr_feature-clients[i].curr_AvgProto, ord=2, axis=None, keepdims=False)  # 原型保存为numpy
            # d = torch.dist(curr_feature, clients[i].curr_AvgProto, p=2)      # 原型保存为tensor
            clients[curr_i].dis_with_other[i] = float(d)
    return 

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
    w_g_personalized.update((key, value * 0.8) for key, value in w_g_personalized.items()) # 个性化聚合中，自己的模型占一半权重，其他客户端的所有模型，占一半权重
    for k in w_g_personalized.keys():
        for i in range(clients_num):   #遍历本轮的所有客户端
            if i != curr_i and str(clients[i].model.state_dict()[k].dtype) == 'torch.float32' :       #排除客户端curr_i自己
                w_g_personalized[k] += 0.2 * clients[curr_i].dis_with_other[i] * clients[i].model.state_dict()[k]
    return w_g_personalized


def test(args, model, device, task, mask, criterion, task_id):
    model.eval()
    with torch.no_grad():
        output,_ = model(task.x, task.edge_index)
        final_loss = criterion(output[task_id][mask], task.y[mask])
        pred = output[task_id][mask].argmax(dim=1, keepdim=True)
        correct = pred.eq(task.y[mask].view_as(pred)).sum().item()
        acc = 100. * correct / int(mask.sum())
    return final_loss, acc             

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


def compute_distance_with_history_AvgProto(curr_client, clients, clients_participant, task_id):
    
    # history_dis = {}
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

from torch_geometric.nn import GCNConv, global_mean_pool
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, taskcla):
        """
        GCN 模型构造函数。
        参数:
        - input_dim: 输入特征维度
        - hidden_dim: 隐藏层维度
        - taskcla: 任务列表，包含 (任务ID, 类别数) 的元组
        """        
        super(GCNModel, self).__init__()

        # 定义图卷积层
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 128)

        # 任务分类器
        self.taskcla = taskcla
        self.fc_task = nn.ModuleList()
        for t, n in self.taskcla:
            self.fc_task.append(nn.Linear(128, n))

        # Dropout 和激活函数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # 用于存储中间激活值
        self.act = OrderedDict()
        
    def forward(self, x, edge_index):
        """
        前向传播。
        参数:
        - x: 节点特征矩阵 (num_nodes, input_dim)
        - edge_index: 边索引 (2, num_edges)
        - batch: 每个节点所属的图索引 (num_nodes,)
        返回:
        - y: 每个任务的分类结果
        - feature_x: 图全局特征
        """
        # 图卷积层
        
        
        x = self.relu(self.conv1(x, edge_index))
        self.act['conv1'] = x  # 保存输入特征
        x = self.dropout(x)
        
        x = self.relu(self.conv2(x, edge_index))
        self.act['conv2'] = x  # 保存 conv1 激活值
        feature_x = self.dropout(x)
        
        x = feature_x
        # self.act['fc2'] = feature_x  # 保存 fc1 激活值
        # x = self.relu(self.fc2(feature_x))
        # 分类输出
        y = []
        for t, _ in self.taskcla:
            y.append(self.fc_task[t](x))
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
        self.grad_basis = []
        self.importance_list = []    
        
        self.memories = []
        self.memory_budget = 100   
         
    def train_first_task(self, args, task, client_node_indices, task_id, device, threshold, c_id, g_epoch):
        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)     
        feature_list =[]
        importance_list = []          

        # graph
        task.to(device)
        num_cls = torch.unique(task.y)[-1]
        # 根据客户端节点索引创建数据掩码
        node_mask = torch.zeros(task.num_nodes, dtype=torch.bool, device=device)
        node_mask[client_node_indices] = True  # 只保留客户端的节点
        task.train_mask = node_mask        


        # 更新内存区
        if g_epoch == 0:
            train_ids = task.train_mask.nonzero(as_tuple=True)[0].tolist()
            sampled_ids = np.random.choice(train_ids, self.memory_budget, replace=False)
            replayed_graph = task.subgraph(torch.tensor(sampled_ids).to(device))
            edge_index = replayed_graph.edge_index
            adj = SparseTensor(row=edge_index[0], 
                                col=edge_index[1], 
                                sparse_sizes=(replayed_graph.num_nodes, replayed_graph.num_nodes))    
            replayed_graph.adj_t = adj.t()
            self.memories.append(replayed_graph)    
        
        # Train      
        for epoch in range(1, args.l_epochs+1):
            clock0=time.time()
            # Train
            self.model.train()
            optimizer.zero_grad() 
            output, Proto = self.model(task.x, task.edge_index)
            loss = self.criterion(output[task_id][task.train_mask], task.y[task.train_mask])  
            loss.backward()
            optimizer.step()             
            # Test
            tr_loss,tr_acc = test(args, self.model, device, task, task.train_mask, self.criterion, task_id)
            print('Epoch {:3d} | Client {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,c_id,tr_loss,tr_acc,1000*(time.time()-clock0)),end='')
            valid_loss,valid_acc = test(args, self.model, device, task, task.test_mask, self.criterion, task_id) # Validate
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
            print()

        # 本地训练中，保存最后一个epoch的feature
        self.curr_AvgProto = Proto.mean(dim=0).detach().cpu().numpy()
        if g_epoch == args.g_epochs-1:
            # 应该在所有的epochs结束之后返回特征和模型
            self.history_AvgProto.append(self.curr_AvgProto) # 保存历史类原型
            self.history_model.append(self.model.state_dict())
        # Memory Update   %%%
        grad_list = get_grad_matrix (self.model, device, task, self.memories)
        if args.important is not True:
            self.grad_basis = update_grad_basis (grad_list, threshold,self.grad_basis)
        else:
            self.grad_basis, self.importance_list = update_grad_basis_calculate_important_sigmoid (args, grad_list, threshold, task_id, feature_list, importance_list)
        return 
    
    def train_new_task(self, args, task, client_node_indices, task_id, device, threshold, c_id, old_model_list, g_epoch, avg_forgetting):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        reg_loss = Regularization(self.model, old_model_list, p=2).to(device)
        feature_mat = []
        
        # Projection Matrix Precomputation %%%@@@@@
        for i in range(len(self.model.act)):
            if isinstance(self.grad_basis[i], torch.Tensor):
                grad_basis_transpose = np.array(self.grad_basis[i].to('cpu')).transpose(-1, -2)  # 转置操作
                grad_basis = np.array(self.grad_basis[i].to('cpu'))
            elif isinstance(self.grad_basis[i], np.ndarray):
                grad_basis_transpose = self.grad_basis[i].transpose(-1, -2) 
                grad_basis = self.grad_basis[i]
            Uf = torch.Tensor(np.dot(grad_basis, grad_basis_transpose)).to(device)   
            Uf.requires_grad = False
            feature_mat.append(Uf)        
        print ('-'*40)        
        
        
        # 更新内存区
        if g_epoch == 0:
            train_ids = task.train_mask.nonzero(as_tuple=True)[0].tolist()
            sampled_ids = np.random.choice(train_ids, self.memory_budget, replace=False)
            replayed_graph = task.subgraph(torch.tensor(sampled_ids).to(device))
            edge_index = replayed_graph.edge_index
            adj = SparseTensor(row=edge_index[0], 
                                col=edge_index[1], 
                                sparse_sizes=(replayed_graph.num_nodes, replayed_graph.num_nodes))    
            replayed_graph.adj_t = adj.t()
            self.memories.append(replayed_graph)       
                    
        # Train        
        for epoch in range(1, args.l_epochs+1):
            clock0=time.time()
            # Train 
            self.model.train()
            optimizer.zero_grad() 
            output, Proto = self.model(task.x, task.edge_index)
            local_loss_value = self.criterion(output[task_id][task.train_mask], task.y[task.train_mask])  
            
            # 在这里读取旧任务的模型，并计算正则化损失
            reg_loss_value = reg_loss(self.model)
            b1 = len(str(int(local_loss_value*1000000)))    
            b2 = len(str(int(reg_loss_value*1000000)))
            loss_value = local_loss_value + reg_loss_value/(10**(b2-b1+2))*2    # 加入正则化项的损失
            loss_value.backward()            
            
            # Gradient Projections    %%%
            initial_mu = 1.0  # Initial projection strength
            decay_rate = args.alpha    # Decay rate per iteration
            max_scale_value = task_id
            if sum(avg_forgetting) / len(avg_forgetting) < 0.02:            
                mu = initial_mu * (decay_rate ** task_id)            
            else:
                scale_value = task_id - max_scale_value + 1
                mu = initial_mu * (decay_rate ** scale_value)
            
            kk = 0 
            target_layers = ['conv1', 'conv2']
            for k, (m,params) in enumerate(self.model.named_parameters()):    
                if any(layer in m for layer in target_layers) and "bias" not in m and params.grad is not None: # 确保梯度存在
                    sz =  params.grad.size(0)
                    params.grad = params.grad - mu * torch.mm(params.grad.view(sz, -1), feature_mat[kk]).view(params.size())
                    kk +=1
                elif (k<4 and len(params.size())==1) and params.grad is not None:
                    params.grad.fill_(0)
                
            optimizer.step()            
            
            tr_loss, tr_acc = test(args, self.model, device, task, task.train_mask, self.criterion,task_id)
            print('Epoch {:3d} | Client {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,c_id, tr_loss, tr_acc, 1000*(time.time()-clock0)),end='')
            valid_loss,valid_acc = test(args, self.model, device, task, task.test_mask, self.criterion,task_id)  # Validate
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
            print()
        
        # 本地训练中，保存最后一个epoch的feature
        self.curr_AvgProto = Proto.mean(dim=0).detach().cpu().numpy()        
        if g_epoch == args.g_epochs-1:
            # 应该在所有的epochs结束之后返回特征和模型
            self.history_AvgProto.append(self.curr_AvgProto) # 保存历史类原型
            self.history_model.append(self.model.state_dict())
        # Memory Update  %%%
        grad_list = get_grad_matrix (self.model, device, task, self.memories)
        if args.important is not True:
            print("no important")
            self.grad_basis = update_grad_basis (grad_list, threshold, self.grad_basis)
        else:
            self.grad_basis, self.importance_list = update_grad_basis_calculate_important_sigmoid (args, grad_list, threshold, task_id, self.grad_basis, self.importance_list)
            # print("importance_list", self.importance_list)
        return                 
   
        
if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--l_epochs', type=int, default=5, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--g_epochs', type=int, default=2, metavar='N',
                        help='number of global training epochs (default: 100)')
    parser.add_argument('--seed', type=int, default=2023, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--pc_valid',default=0.05,type=float,
                        help='fraction of training data used for validation')
    parser.add_argument('--device',default=1,type=int,
                        help='GPU ID, -1 for CPU')
    parser.add_argument('--first_task_g', type=int, default=2, metavar='N',
                        help='number of global training epochs at first task (default: 100)')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    
    # CGO specific 
    parser.add_argument('--scale_coff', type=int, default=10, metavar='SCF',
                        help='importance co-efficeint (default: 10)')
    parser.add_argument('--gpm_eps', type=float, default=0.97, metavar='EPS',
                        help='threshold (default: 0.97)')
    parser.add_argument('--gpm_eps_inc', type=float, default=0.003, metavar='EPSI',
                        help='threshold increment per task (default: 0.003)')
    parser.add_argument('--alpha', type=float, default=3, metavar='alpha',
                        help='importance co-efficeint (default: 5)')
    parser.add_argument('--beta', type=float, default=0.92, metavar='zeta',
                        help='weight of basis of each task (default: 0.9)')

    # FL specific
    parser.add_argument('--task_num', type=int, default=10,
                        help='the number of task (default: 10)')
    parser.add_argument('--clients_num', type=int, default=10,
                        help='the number of clients (default: 10)')
    parser.add_argument('--selected_clients', type=int, default=2, 
                        help='history model of selected') 
    # parser.add_argument('--local_epochs', type=int, default=50, metavar='N',
    #                     help='the number of training epochs/clients (default: 1)')
    # parser.add_argument('--global_rounds', type=int, default=5, metavar='N',
    #                     help='the number of training rounds/task (default: 5)')
    # parser.add_argument('--increment_mode', type=str, default='SL', metavar='N',
    #                     help='incremental data is the same label or different labels (default: SL)')
    parser.add_argument('--important', type=bool, default=True, metavar='N')
    parser.add_argument('--classes_per_task', type=int, default=4,
                        help='the class numbers of per task (default: 10)')

    args = parser.parse_args()
    tstart=time.time()
    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)
    setup_seed(args.seed)
    
    data_set = []
    taskcla_list = []
    task_list = []
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # initial clients
    acc_matrix = np.zeros((10,10))
    clients = []
    model_g = []

    args.n_cls = 40
    args.feat_dim = 128


    # 设置任务
    taskcla_list = set_task(args.clients_num, args.task_num, args.classes_per_task)
    # 通过循环来给每个客户端分配数据集
    for c_id in range(args.clients_num):
        # 每个客户端一个全局模型
        model_g_temp = GCNModel(args.feat_dim, 128, taskcla_list[c_id]).to(device)
        model_g.append(model_g_temp)   
        # 每个客户创建一个client类
        client_temp = Client(model_g[c_id], args)
        clients.append(client_temp)

    print ('Model parameters ---')
    for k_t, (m, param) in enumerate(model_g[0].named_parameters()):
        print (k_t,m,param.shape)
    print ('-'*40)
    
    # 读取corafull数据集
    # task_file = os.path.join("/root/yx/fjy/data/streaming", f"{args.dataset_name}.streaming")
    task_file = os.path.join("/root/yx/fjy/data/streaming_10task_arxiv", "arxiv.streaming")
    data_stream = torch.load(task_file)
    
    # task loop
    # 统计本任务中的所有acc
    all_task_acc = {i: [] for i in range(10)}
    avg_forgetting = []
    performace_matrix = torch.zeros(len(data_stream.tasks)+1, len(data_stream.tasks)+1)
    
    task_id = 0
    for task in data_stream.tasks:
        print ('-'*100)
        print ('Task ID :{} | Learning Rate : {}'.format(task_id, args.lr))
        print ('-'*100)
        threshold = np.array([0.97] * 10) + np.array([0.001] * 10)
        
        # # 加载该任务的数据集
        g_epochs = args.first_task_g if task_id == 0 else args.g_epochs
        for epoch in range(g_epochs):
            # 客户端训练 
            for c_id in range(args.clients_num):
                client_node_indices = data_stream.clients_datas[task_id][c_id]['node_indices']
                if task_id==0:
                    clients[c_id].train_first_task(args, task, client_node_indices, task_id, device, threshold, c_id, epoch)   
                else:
                    old_model_list = select_old_model(clients, c_id, task_id, args.clients_num, args.selected_clients) # @ corss_task_module
                    clients[c_id].train_new_task(args, task, client_node_indices, task_id, device, threshold, c_id, old_model_list, epoch, avg_forgetting)  

            # 计算原型之间的距离
            for c_id in range(args.clients_num):
                compute_distance_curr_feature(c_id, clients[c_id].curr_AvgProto, clients, args.clients_num)
                # if epoch == args.g_epochs-1:       # ep_g%10=0,表示只保存和每个任务第一轮的距离及其原型，ep_g%10=9表示最后一轮
                if epoch == g_epochs-1:
                    compute_distance_with_history_AvgProto(c_id, clients, args.clients_num, task_id)   

            # personalized aggregation
            avg_test_acc, avg_test_loss = 0, 0
            for c_id in range(args.clients_num):
                clients[c_id].personalized_global_model = PFL(c_id, clients, args.clients_num)
                model_g[c_id].load_state_dict(clients[c_id].personalized_global_model)
                test_loss, test_acc = test(args, model_g[c_id], device, task, task.test_mask, clients[c_id].criterion, task_id)
                avg_test_acc += test_acc
                avg_test_loss += test_loss                
            print ('-'*40)
            print('G-Epoch: {:3d}, Test: loss={:.3f} , acc={:5.1f}%'.format(epoch, avg_test_loss/args.clients_num,avg_test_acc/args.clients_num))
            
            # 在每一轮中都测试全部任务的准确率
            print ('*'*40)
            t = 0
            for task_ in data_stream.tasks:
                task_.to(device)
                avg_test_acc = 0
                avg_test_loss = 0
                for c_id in range(args.clients_num):  
                    test_loss, test_acc = test(args, model_g[c_id], device, task_, task_.test_mask,  clients[c_id].criterion, t)
                    avg_test_acc += test_acc
                    avg_test_loss += test_loss               
                print('Task: {:3d}, G-Epoch: {:3d}, Test: loss={:.3f} , acc={:5.1f}%'.format(t, epoch, avg_test_loss/args.clients_num,avg_test_acc/args.clients_num)) 
                all_task_acc[t].append(avg_test_acc/args.clients_num)
                t += 1
            # print(all_task_acc)
                
        # save accuracy 
        jj=0
        for task__ in data_stream.tasks:
            task__.to(device)    
            avg_acc = 0
            for c_id in range(args.clients_num):
                model_g[c_id].load_state_dict(clients[c_id].personalized_global_model)
                _, acc = test(args, model_g[c_id], device, task__, task__.test_mask, clients[c_id].criterion,jj) 
                avg_acc += acc
            acc_matrix[task_id,jj] = avg_acc/args.clients_num
            jj+= 1

        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a,j_a]),end='')
            print()
        # 调用函数
        avg_forgetting.append(calculate_average_forgetting(acc_matrix, task_id))
        print("avg_forgetting", [round(x, 4) for x in avg_forgetting])
        
        task_id += 1
        
    print('-'*50)
    # Simulation Results 
    print ('Task Order : {}'.format(np.array(task_list)))
    print ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean())) 
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    print ('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    print('-'*50)
    # Plots
    array = acc_matrix
    df_cm = pd.DataFrame(array, index = [i for i in ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]],
                      columns = [i for i in ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]])
    sn.set(font_scale=1.4) 
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
    plt.show()    