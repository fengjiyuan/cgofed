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
from model import AlexNet
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import random
import pdb
import math
import time
from copy import deepcopy
from server import *
from model import *
from server import Regularization

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train(args, model, device, x, y, optimizer, scheduler, criterion, task_id):
    model.train()
    # 用 torch 打乱索引
    num_samples = x.size(0)
    indices = torch.randperm(num_samples)

    Proto = None          
    for i in range(0, num_samples, args.batch_size_train):   
        b = indices[i: min(i + args.batch_size_train, num_samples)]
        data   = x[b].to(device, non_blocking=True)
        target = y[b].to(device, non_blocking=True)
        # --- Mixup 增强 ---
        data, targets_a, targets_b, lam = mixup_data(data, target, alpha=0.2, device=device)

        optimizer.zero_grad(set_to_none=True)
        output, proto = model(data)

        head_idx = model.tid2head[int(task_id)]   # ★ 把真实任务ID映射到位置索引
        logits = output[head_idx]
        # --- Mixup loss ---
        loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
        # loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        # 保存特征图
        if Proto is None:
            Proto = proto
        else:
            Proto = torch.cat((Proto, proto), dim=0)
    scheduler.step()
    return Proto

def train_projected(args, model, device, x, y, optimizer, scheduler, criterion, feature_mat, task_id, reg_loss, avg_forgetting):
    model.train()
    num_samples = x.size(0)
    indices = torch.randperm(num_samples)   # 用 torch 打乱索引
    
    scale_value = task_id
    Proto = None

    for i in range(0, num_samples, args.batch_size_train):
        b = indices[i: min(i + args.batch_size_train, num_samples)]
        data = x[b].to(device, non_blocking=True)
        target = y[b].to(device, non_blocking=True)

        # --- Mixup 增强 ---
        data, targets_a, targets_b, lam = mixup_data(data, target, alpha=0.2, device=device)

        optimizer.zero_grad(set_to_none=True)
        output, proto = model(data)
        head_idx = model.tid2head[int(task_id)]
        logits = output[head_idx]

        # --- Mixup loss ---
        local_loss_value = mixup_criterion(criterion, logits, targets_a, targets_b, lam)

        # 正则化损失
        reg_loss_value = reg_loss(model)

        # 将正则化项缩放后加入
        b1 = len(str(int(local_loss_value * 1e6)))
        b2 = len(str(int(reg_loss_value * 1e6)))
        loss_value = local_loss_value + reg_loss_value / (10 ** (b2 - b1 + 2)) * 2

        loss_value.backward()

        # Gradient Projection
        kk = 0
        initial_mu = 1.0
        decay_rate = args.alpha
        max_scale_value = task_id

        if avg_forgetting[-1] < args.tau:
            mu = initial_mu * (decay_rate ** task_id)
        else:
            scale_value = task_id - max_scale_value + 1
            mu = initial_mu * (decay_rate ** scale_value)

        for k, (m, params) in enumerate(model.named_parameters()):
            if k < 15 and len(params.size()) != 1:
                sz = params.grad.data.size(0)
                params.grad.data = params.grad.data - mu * torch.mm(
                    params.grad.data.view(sz, -1), feature_mat[kk]
                ).view(params.size())
                kk += 1
            elif (k < 15 and len(params.size()) == 1) and task_id != 0:
                params.grad.data.fill_(0)
        optimizer.step()
        # 保存特征图
        if Proto is None:
            Proto = proto
        else:
            Proto = torch.cat((Proto, proto), dim=0)
    scheduler.step()
    return Proto


def get_grad_matrix(net, device, x, y=None):
    # Collect activations by forward pass
    r = np.arange(x.size(0))  # 样本数量
    np.random.shuffle(r)
    r = torch.LongTensor(r)
    b = r[0:250]  # Take 125 random samples
    example_data = x[b]
    example_data = example_data.to(device)
    example_out = net(example_data)

    grad_list = []  # list contains gradient of each layer

    for _, (m, params) in enumerate(net.named_parameters()):

        if 'weight' in m and 'bn' not in m and 'fc3' not in m:
            sz = params.grad.data.size(0)
            grad = params.grad.data.view(sz, -1).detach().cpu().numpy()
            activation = grad.transpose()
            grad_list.append(activation)
    return grad_list


def update_grad_basis(grad_list, threshold, grad_basis=[]):
    # print ('Threshold: ', threshold)
    if not grad_basis:
        for i in range(len(grad_list)):
            activation = grad_list[i]
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold[i])  # +1
            grad_basis.append(U[:, 0:r])

    else:
        for i in range(len(grad_list)):
            activation = grad_list[i]
            U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1 ** 2).sum()
            act_hat = activation - np.dot(np.dot(grad_basis[i], grad_basis[i].transpose()), activation)
            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria
            sval_hat = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            accumulated_sval = (sval_total - sval_hat) / sval_total

            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            Ui = np.hstack((grad_basis[i], U[:, 0:r]))
            if Ui.shape[1] > Ui.shape[0]:
                grad_basis[i] = Ui[:, 0:Ui.shape[0]]
            else:
                grad_basis[i] = Ui
    return grad_basis


def update_grad_basis_calculate_important_sigmoid(args, grad_list, threshold, task_id, grad_basis=[],
                                                  importance_list=[]):
    # print ('Threshold: ', threshold)
    # scale_coff = 10
    if not grad_basis:
        for i in range(len(grad_list)):
            activation = grad_list[i]
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold[i])  # +1
            grad_basis.append(U[:, 0:r])

            # 以下为新增
            importance = sigmoid(args.beta * S[0:r])
            importance_list.append(importance)

    else:
        for i in range(len(grad_list)):
            activation = grad_list[i]
            U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1 ** 2).sum()
            act_proj = np.dot(np.dot(grad_basis[i], grad_basis[i].transpose()), activation)

            # 以下为新增
            r_old = grad_basis[i].shape[1]  # old GPM bases
            Uc, Sc, Vhc = np.linalg.svd(act_proj, full_matrices=False)
            importance_new_on_old = np.dot(np.dot(grad_basis[i].transpose(), Uc[:, 0:r_old]) ** 2,
                                           Sc[0:r_old] ** 2)  ## r_old no of elm s**2 fmt
            importance_new_on_old = np.sqrt(importance_new_on_old)

            act_hat = activation - act_proj
            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria
            sval_hat = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            accumulated_sval = (sval_total - sval_hat) / sval_total

            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break

            # 以下为新增
            if r == 0:
                # print('Skip Updating GPM for layer: {}'.format(i + 1))
                # update importances
                importance = importance_new_on_old
                importance = sigmoid(args.beta * importance)
                importance[0:r_old] = np.clip(importance[0:r_old] + importance_list[i][0:r_old], 0, 1)
                importance_list[i] = importance  # update importance
                continue

            # update GPM
            Ui = np.hstack((grad_basis[i], U[:, 0:r]))

            # update importance  以下为新增
            importance = np.hstack((importance_new_on_old, S[0:r]))
            importance = sigmoid(args.beta * importance)
            importance[0:r_old] = np.clip(importance[0:r_old] + importance_list[i][0:r_old], 0, 1)

            if Ui.shape[1] > Ui.shape[0]:
                grad_basis[i] = Ui[:, 0:Ui.shape[0]]

                importance_list[i] = importance[0:Ui.shape[0]]
            else:
                grad_basis[i] = Ui
                importance_list[i] = importance

    return grad_basis, importance_list


class Client:
    def __init__(self, model, args):

        super(Client, self).__init__()
        # self.task_id = task_id
        self.model = model
        self.best_model = None
        self.best_loss = np.inf
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.patience = None
        self.lr = args.lr
        self.momentum = args.momentum

        self.personalized_global_model = None
        self.curr_AvgProto = []
        self.history_AvgProto = []
        self.history_dis = {}
        self.dis_with_other = [0] * 30
        self.history_model = []

        self.grad_basis = []
        self.importance_list = []

    def train_first_task(self, args, xtrain, ytrain, xvalid, yvalid, task_id, device, threshold, c_id, curr_epoch, g_epoch):
        self.model = self.model.to(device)
        # optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=5e-4, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.l_epochs, eta_min=args.lr_min)

        feature_list = []
        importance_list = []

        for epoch in range(1, args.l_epochs + 1):
            # Train
            clock0 = time.time()
            Proto = train(args, self.model, device, xtrain, ytrain, optimizer, scheduler, self.criterion, task_id)
            clock1 = time.time()
            tr_loss, tr_acc = test(args, self.model, device, xtrain, ytrain, self.criterion, task_id)
            print('Epoch {:3d} | Client {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch, \
                                                                                                            c_id,
                                                                                                            tr_loss,
                                                                                                            tr_acc,
                                                                                                            1000 * (clock1 - clock0)),end='')
            # Validate
            valid_loss, valid_acc = test(args, self.model, device, xvalid, yvalid, self.criterion, task_id)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc), end='')
            print()

        # 本地训练中，保存最后一个epoch的feature
        self.curr_AvgProto = Proto.mean(dim=0).detach().cpu().numpy()
        if curr_epoch == g_epoch - 1:
            # 应该在所有的epochs结束之后返回特征和模型
            self.history_AvgProto.append(self.curr_AvgProto)  # 保存历史类原型
            # 深拷贝到CPU，避免后续被覆盖
            sd = {k: t.detach().cpu().clone() for k, t in self.model.state_dict().items()}
            self.history_model.append(sd)
        # Memory Update   %%%
        grad_list = get_grad_matrix(self.model, device, xtrain, ytrain)

        if args.test == 1:
            # print("no important")
            self.grad_basis = update_grad_basis(grad_list, threshold, self.grad_basis)
        else:
            self.grad_basis, self.importance_list = update_grad_basis_calculate_important_sigmoid(args, grad_list,
                                                                                                  threshold, task_id,
                                                                                                  feature_list,
                                                                                                  importance_list)
        return

    def train_new_task(self, args, xtrain, ytrain, xvalid, yvalid, task_id, device, threshold, c_id, old_model_list,
                       g_epoch, avg_forgetting):
        # optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=5e-4, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.l_epochs, eta_min=args.lr_min)

        reg_loss = Regularization(self.model, old_model_list, p=2).to(device)
        feature_mat = []
        # Projection Matrix Precomputation %%%
        for i in range(len(self.model.act)):
            # Uf改梯度
            # Uf=torch.Tensor(np.dot(self.grad_basis[i],self.grad_basis[i].transpose())).to(device)
            Uf = torch.Tensor(np.dot(self.grad_basis[i],
                                     np.dot(np.diag(self.importance_list[i]), self.grad_basis[i].transpose()))).to(
                device)
            # print('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
            Uf.requires_grad = False
            feature_mat.append(Uf)
        print('-' * 40)
        for epoch in range(1, args.l_epochs + 1):
            # Train
            clock0 = time.time()
            Proto = train_projected(args, self.model, device, xtrain, ytrain, optimizer, scheduler, self.criterion, feature_mat,
                                    task_id, reg_loss, avg_forgetting)
            clock1 = time.time()
            tr_loss, tr_acc = test(args, self.model, device, xtrain, ytrain, self.criterion, task_id)
            print('Epoch {:3d} | Client {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch, \
                                                                                                            c_id,
                                                                                                            tr_loss,
                                                                                                            tr_acc,
                                                                                                            1000 * (clock1 - clock0)),end='')
            # Validate
            valid_loss, valid_acc = test(args, self.model, device, xvalid, yvalid, self.criterion, task_id)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc), end='')
            print()

        # 本地训练中，保存最后一个epoch的feature
        self.curr_AvgProto = Proto.mean(dim=0).detach().cpu().numpy()
        if g_epoch == args.g_epochs - 1:
            # 应该在所有的epochs结束之后返回特征和模型
            self.history_AvgProto.append(self.curr_AvgProto)  # 保存历史类原型
            # 深拷贝到CPU，避免后续被覆盖
            sd = {k: t.detach().cpu().clone() for k, t in self.model.state_dict().items()}
            self.history_model.append(sd)
        # Memory Update  %%%
        grad_list = get_grad_matrix(self.model, device, xtrain, ytrain)
        if args.test == True:
            # print("no important")
            self.grad_basis = update_grad_basis(grad_list, threshold, self.grad_basis)
        else:
            self.grad_basis, self.importance_list = update_grad_basis_calculate_important_sigmoid(args, grad_list,
                                                                                                  threshold, task_id,
                                                                                                  self.grad_basis,
                                                                                                  self.importance_list)
            # print("importance_list", self.importance_list)
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

    average_difference = total_difference / ((task_num + 1) * 100)
    return average_difference

def mixup_data(x, y, alpha=0.2, device='cuda'):
    '''返回混合后的样本和标签对'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


    # def vision_feature(model_params):


#     layer_params = model_params
#     plt.figure(figsize=(10, 10))
#     # last_layer_params.cpu()
# #     layer_params = [torch.tensor(param) for param in layer_params]
#     i = 0
#     for param in layer_params:
#         plt.imshow(utils.make_grid(torch.tensor(param.cpu()), normalize=True).numpy().transpose((1, 2, 0)))
#         plt.axis('off')
#         plt.savefig("feature{}.png".format(i), dpi=300)
#         plt.show()
#         i = i+1
#     return

# def update_grad_basis_calculate_important (args, grad_list, threshold, task_id, grad_basis=[], importance_list=[]):
#     # print ('Threshold: ', threshold)
#     scale_coff = 10
#     if not grad_basis:
#         for i in range(len(grad_list)):
#             activation = grad_list[i]
#             U,S,Vh = np.linalg.svd(activation, full_matrices=False)
#             sval_total = (S**2).sum()
#             sval_ratio = (S**2)/sval_total
#             r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1
#             grad_basis.append(U[:,0:r])

#             # 以下为新增
#             importance = ((scale_coff+1)*S[0:r])/(scale_coff*S[0:r] + max(S[0:r]))
#             importance_list.append(importance)

#     else:
#         for i in range(len(grad_list)):
#             activation = grad_list[i]
#             U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
#             sval_total = (S1**2).sum()
#             act_proj = np.dot(np.dot(grad_basis[i],grad_basis[i].transpose()),activation)

#             # 以下为新增
#             r_old = grad_basis[i].shape[1] # old GPM bases
#             Uc,Sc,Vhc = np.linalg.svd(act_proj, full_matrices=False)
#             importance_new_on_old = np.dot(np.dot(grad_basis[i].transpose(),Uc[:,0:r_old])**2, Sc[0:r_old]**2) ## r_old no of elm s**2 fmt
#             importance_new_on_old = np.sqrt(importance_new_on_old)

#             act_hat = activation - act_proj
#             U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
#             # criteria (Eq-9)
#             sval_hat = (S**2).sum()
#             sval_ratio = (S**2)/sval_total
#             accumulated_sval = (sval_total-sval_hat)/sval_total

#             r = 0
#             for ii in range (sval_ratio.shape[0]):
#                 if accumulated_sval < threshold[i]:
#                     accumulated_sval += sval_ratio[ii]
#                     r += 1
#                 else:
#                     break

#             # 以下为新增
#             if r == 0:
#                 print ('Skip Updating GPM for layer: {}'.format(i+1))
#                 # update importances
#                 importance = importance_new_on_old
#                 importance = ((args.scale_coff+1)*importance)/(args.scale_coff*importance + max(importance))
#                 importance [0:r_old] = np.clip(importance [0:r_old]+importance_list[i][0:r_old], 0, 1)
#                 importance_list[i] = importance # update importance
#                 continue

#            # update GPM
#             Ui=np.hstack((grad_basis[i],U[:,0:r]))

#             # update importance  以下为新增
#             importance = np.hstack((importance_new_on_old,S[0:r]))
#             importance = ((args.scale_coff+1)*importance)/(args.scale_coff*importance + max(importance))
#             importance [0:r_old] = np.clip(importance [0:r_old]+importance_list[i][0:r_old], 0, 1)

#             if Ui.shape[1] > Ui.shape[0] :
#                 grad_basis[i]=Ui[:,0:Ui.shape[0]]

#                 importance_list[i] = importance[0:Ui.shape[0]]
#             else:
#                 grad_basis[i]=Ui
#                 importance_list[i] = importance

#     return grad_basis,   importance_list



# def test(args, model, device, x, y, criterion, task_id):
#     model.eval()
#     total_loss = 0
#     total_num = 0
#     correct = 0
#     r = np.arange(x.size(0))
#     np.random.shuffle(r)
#     r = torch.LongTensor(r)
#     with torch.no_grad():
#         # Loop batches
#         for i in range(0, len(r), args.batch_size_test):
#             if i + args.batch_size_test <= len(r):
#                 b = r[i:i + args.batch_size_test]
#             else:
#                 b = r[i:]
#             data = x[b]
#             data, target = data.to(device), y[b].to(device)
#             output, _ = model(data)
#             head_idx = model.tid2head[int(task_id)]   # ★ 把真实任务ID映射到位置索引
#             logits = output[head_idx]
#             loss = criterion(logits, target)
#             pred = logits.argmax(dim=1, keepdim=True)

#             correct += pred.eq(target.view_as(pred)).sum().item()
#             total_loss += loss.data.cpu().numpy().item() * len(b)
#             total_num += len(b)

#     acc = 100. * correct / total_num
#     final_loss = total_loss / total_num
#     return final_loss, acc


# def adjust_learning_rate(optimizer, epoch, lr, lr_factor):
#     for param_group in optimizer.param_groups:
#         if (epoch ==1):
#             param_group['lr']=lr
#         else:
#             param_group['lr'] /= lr_factor