# -*- coding: utf-8 -*-
# 使用新的数据加载函数
import torch
import numpy as np
import argparse
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import argparse,time
import os.path
from collections import OrderedDict
from copy import deepcopy
from utils import cifar100 as cf100
from data_stream import Streaming

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

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


def PFL(clients, clients_num):
    # 初始化聚合模型的状态字典
    aggregated_state_dict = None
    # 遍历每个客户端
    for client in clients:
        # 获取客户端的模型参数和样本数量
        client_state_dict = client.model.state_dict()
        client_weight = 1 / clients_num  # 权重比例（样本数 / 总样本数）
        
        if aggregated_state_dict is None:
            # 初始化全局模型参数
            aggregated_state_dict = {
                key: client_weight * param.clone()
                for key, param in client_state_dict.items()
            }
        else:
            # 加权累加客户端模型参数
            for key in aggregated_state_dict:
                aggregated_state_dict[key] += client_weight * client_state_dict[key]
    
    return aggregated_state_dict    
    
    # for key, value in w_g_personalized.items():
        
    # w_g_personalized = copy.deepcopy(clients[curr_i].model.state_dict())   
    # w_g_personalized.update((key, value * 0.5) for key, value in w_g_personalized.items()) # 个性化聚合中，自己的模型占一半权重，其他客户端的所有模型，占一半权重
    # for k in w_g_personalized.keys():
    #     for i in range(clients_num):   #遍历本轮的所有客户端
    #         if i != curr_i and str(clients[i].model.state_dict()[k].dtype) == 'torch.float32' :       #排除客户端curr_i自己
    #             w_g_personalized[k] += 0.5 * clients[i].model.state_dict()[k]
    # return w_g_personalized


def test(args, model, device, task, mask, criterion, task_id):
    model.eval()
    with torch.no_grad():
        output,_ = model(task.x, task.edge_index)
        final_loss = criterion(output[task_id][mask], task.y[mask])
        pred = output[task_id][mask].argmax(dim=1, keepdim=True)
        correct = pred.eq(task.y[mask].view_as(pred)).sum().item()
        acc = 100. * correct / int(mask.sum())
    return final_loss, acc             

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
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
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
        self.act['conv1'] = x  # 保存输入特征
        x = self.relu(self.conv1(x, edge_index))
        self.act['conv2'] = x  # 保存 conv1 激活值
        x = self.dropout(x)
        
        x = self.relu(self.conv2(x, edge_index))
        self.act['fc1'] = x  # 保存 conv2 激活值
        x = self.dropout(x)
        # 全连接层
        feature_x = self.relu(self.fc1(x))
        self.act['fc2'] = feature_x  # 保存 fc1 激活值
        x = self.relu(self.fc2(feature_x))
        # 分类输出
        y = []
        for t, _ in self.taskcla:
            y.append(self.fc_task[t](x))
        return y, feature_x 

    
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
        # self.personalized_global_model = None
        self.curr_AvgProto = []
        self.history_AvgProto = []
        self.history_dis = {}
        self.dis_with_other = [0]*30
        self.history_model = []
        self.grad_basis = []
        self.importance_list = []    
    
    def train_task(self, args, task, client_node_indices, task_id, device, c_id, g_epoch):
        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-3)   
        # graph
        task.to(device)
        num_cls = torch.unique(task.y)[-1]        
        # 根据客户端节点索引创建数据掩码
        node_mask = torch.zeros(task.num_nodes, dtype=torch.bool, device=device)
        node_mask[client_node_indices] = True  # 只保留客户端的节点
        task.train_mask = node_mask       
        
        for epoch in range(1, args.l_epochs+1):
            clock0=time.time()
            # Train
            self.model.train()
            optimizer.zero_grad() 
            output, _ = self.model(task.x, task.edge_index)
            loss = self.criterion(output[task_id][task.train_mask], task.y[task.train_mask])  
            loss.backward()
            optimizer.step()             
            # Test
            tr_loss,tr_acc = test(args, self.model, device, task, task.train_mask, self.criterion, task_id)
            print('Epoch {:3d} | Client {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,c_id,tr_loss,tr_acc,1000*(time.time()-clock0)),end='')
            valid_loss,valid_acc = test(args, self.model, device, task, task.test_mask, self.criterion, task_id) # Validate
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
            print()        
                        
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
    parser.add_argument('--clients_num', type=int, default=5,
                        help='the number of clients (default: 10)')
    parser.add_argument('--selected_clients', type=int, default=2, 
                        help='history model of selected') 
    parser.add_argument('--local_epochs', type=int, default=50, metavar='N',
                        help='the number of training epochs/clients (default: 1)')
    parser.add_argument('--global_rounds', type=int, default=5, metavar='N',
                        help='the number of training rounds/task (default: 5)')
    parser.add_argument('--increment_mode', type=str, default='SL', metavar='N',
                        help='incremental data is the same label or different labels (default: SL)')
    parser.add_argument('--test', type=int, default=1, metavar='N',
                        help='incremental data is the same label or different labels (default: SL)')
    parser.add_argument('--classes_per_task', type=int, default=10,
                        help='the class numbers of per task (default: 10)')

    args = parser.parse_args()
    tstart=time.time()
    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)
    cf100.setup_seed(args.seed)
    
    data_set = []
    taskcla_list = []
    task_list = []
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # initial clients
    acc_matrix = np.zeros((10,10))
    clients = []
    # model_g = []

    args.n_cls = 70
    args.feat_dim = 8710


    # 设置任务
    taskcla_list = cf100.set_task(args.clients_num, args.task_num, args.classes_per_task)
    # 通过循环来给每个客户端分配数据集
    model_g = GCNModel(args.feat_dim, 128, taskcla_list[0]).to(device)
    for c_id in range(args.clients_num):
        # 每个客户端一个全局模型
        # model_g_temp = GCNModel(args.feat_dim, 128, taskcla_list[c_id]).to(device)
        # model_g.append(model_g_temp)   
        # 每个客户创建一个client类
        client_temp = Client(model_g, args)
        clients.append(client_temp)

    print ('Model parameters ---')
    for k_t, (m, param) in enumerate(model_g.named_parameters()):
        print (k_t,m,param.shape)
    print ('-'*40)
    
    # 读取corafull数据集
    # task_file = os.path.join("/root/yx/fjy/data/streaming", f"{args.dataset_name}.streaming")
    task_file = os.path.join("/root/yx/fjy/data/streaming", "corafull.streaming")
    data_stream = torch.load(task_file)
    
    # task loop
    # 统计本任务中的所有acc
    all_task_acc = {i: [] for i in range(10)}
    avg_forgetting = []
    performace_matrix = torch.zeros(len(data_stream.tasks)+1, len(data_stream.tasks)+1)
    
    task_id = 0
    # for task_id in range(args.task_num):
    for task in data_stream.tasks:
        print ('-'*100)
        print ('Task ID :{} | Learning Rate : {}'.format(task_id, args.lr))
        print ('-'*100)
        
        # # 加载该任务的数据集  
        g_epochs = args.first_task_g if task_id == 0 else args.g_epochs
        for epoch in range(g_epochs):
            # 客户端训练 
            for c_id in range(args.clients_num):
                client_node_indices = data_stream.clients_datas[task_id][c_id]['node_indices']
                clients[c_id].train_task(args, task, client_node_indices, task_id, device, c_id, epoch) 

            # personalized aggregation
            criterion = torch.nn.CrossEntropyLoss()
            # avg_test_acc, avg_test_loss = 0, 0
            model_g_param = PFL(clients, args.clients_num)
            model_g.load_state_dict(model_g_param)
            avg_test_loss, avg_test_acc = test(args, model_g, device, task, task.test_mask, criterion, task_id)             
            print ('-'*40)
            print('G-Epoch: {:3d}, Test: loss={:.3f} , acc={:5.1f}%'.format(epoch, avg_test_loss,avg_test_acc))
            
            # 在每一轮中都测试全部任务的准确率
            print ('*'*40)
            t = 0
            for task_ in data_stream.tasks:
                task_.to(device)
                # avg_test_acc = 0
                # avg_test_loss = 0
                # for c_id in range(args.clients_num):  
                test_loss, test_acc = test(args, model_g, device, task_, task_.test_mask,  criterion, t)
                    # avg_test_acc += test_acc
                    # avg_test_loss += test_loss               
                print('Task: {:3d}, G-Epoch: {:3d}, Test: loss={:.3f} , acc={:5.1f}%'.format(t, epoch, test_loss,test_acc)) 
                all_task_acc[t].append(avg_test_acc/args.clients_num)
                t += 1
            # print(all_task_acc)
                
        # save accuracy 
        jj=0
        # for ii in range(task_id+1):
        for task__ in data_stream.tasks:
            task__.to(device)    
            # avg_acc = 0
            # for c_id in range(args.clients_num):
            # model_g[c_id].load_state_dict(clients[c_id].personalized_global_model)
            _, acc = test(args, model_g, device, task__, task__.test_mask, criterion,jj) 
            # avg_acc += acc
            acc_matrix[task_id,jj] = acc
            jj+= 1

        print('Accuracies =')
        for i_a in range(task_id+1):
            print('\t',end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a,j_a]),end='')
            print()
            # update task id 
            # task_id +=1   
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