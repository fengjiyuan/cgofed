# -*- coding: UTF-8 -*-

import copy
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import heapq


def compute_distance_curr_feature(curr_i, curr_feature, clients, clients_participant):
    # dis = []
    for i in range(clients_participant):  # 遍历除当前client的其他客户端
        if i != curr_i:
            # 遍历模型的每一层参数
            d = np.linalg.norm(curr_feature - clients[i].curr_AvgProto, ord=2, axis=None, keepdims=False)  # 原型保存为numpy
            # d = torch.dist(curr_feature, clients[i].curr_AvgProto, p=2)      # 原型保存为tensor
            clients[curr_i].dis_with_other[i] = float(d)
    return


def compute_distance_with_history_AvgProto(curr_client, clients, clients_participant, task_id):
    # history_dis = {}
    for r in range(task_id + 1):  # 遍历其他客户端的历史模型
        dis = []
        for i in range(clients_participant):  # 遍历除当前client的其他客户端
            if i == curr_client:
                d = 0
                dis.append(float(d))
            else:
                d = np.linalg.norm(clients[curr_client].curr_AvgProto - clients[i].history_AvgProto[r], ord=2,
                                   axis=None, keepdims=False)  # 原型保存为numpy
                dis.append(float(d))
            # 算出了当前平均原型和其他客户端的历史平均原型的距离
        clients[curr_client].history_dis[r] = dis


def compute_distance_with_history_representation_matrix(curr_client, clients, clients_participant, task_id):
    # history_dis = {}
    for r in range(task_id + 1):  # 遍历其他客户端的历史模型
        dis = []
        for i in range(clients_participant):  # 遍历除当前client的其他客户端
            for l in range(len(clients[i].history_mat_list)):
                if i == curr_client:
                    d = 0
                    dis.append(float(d))
                else:

                    d = np.linalg.norm(clients[curr_client].curr_AvgProto - clients[i].history_AvgProto[r], ord=2,
                                       axis=None, keepdims=False)  # 原型保存为numpy
                    dis.append(float(d))
            # 算出了当前平均原型和其他客户端的历史平均原型的距离
        clients[curr_client].history_dis[r] = dis


def select_old_model(clients, curr_i, task_id, clients_participant, selected_clients_num):
    # 根据上一个循环计算出的关系，为每个客户端挑选需要加入正则化项的历史模型
    sum_value = 0.
    all_round_selected_list = []  # 保存被选中的历史原型，包括客户端编号，以及该历史原型与当前原型的距离
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
    for r in range(len(all_round_selected_list)):  # 用选中的历史模型计算正则化项
        for key, value in all_round_selected_list[r].items():
            for layer_name, param in clients[key].history_model[r].items():  # 历史模型从内存中读取，self.history_model
                if r == 0:
                    avg_selected_model[layer_name] = value / sum_value * param
                else:
                    avg_selected_model[layer_name] = avg_selected_model[layer_name] + value / sum_value * param

    old_weight_list = []
    for name, param in avg_selected_model.items():  # 只用weight参数，不用bias参数
        if ("weight" in name):
            weight = (name, param)
            old_weight_list.append(weight)
    return old_weight_list


def p_avg(curr_i, clients_participant, clients):
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
            dis_sum += 1.0 / dis[i]
            # 然后求权重
    for i in range(clients_participant):
        if dis[i] != 0:
            clients[curr_i].dis_with_other[i] = (1.0 / dis[i]) / dis_sum

    w_g_personalized = copy.deepcopy(clients[curr_i].model.state_dict())
    w_g_personalized.update(
        (key, value * 0.8) for key, value in w_g_personalized.items())  # 个性化聚合中，自己的模型占一半权重，其他客户端的所有模型，占一半权重
    for k in w_g_personalized.keys():
        for i in range(clients_participant):  # 遍历本轮的所有客户端
            if i != curr_i and str(clients[i].model.state_dict()[k].dtype) == 'torch.float32':  # 排除客户端curr_i自己
                w_g_personalized[k] += 0.2 * clients[curr_i].dis_with_other[i] * clients[i].model.state_dict()[k]
    return w_g_personalized


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
            dis_sum += 1.0 / dis[i]
            # 然后求权重
    for i in range(clients_num):
        if dis[i] != 0:
            clients[curr_i].dis_with_other[i] = (1.0 / dis[i]) / dis_sum

    w_g_personalized = copy.deepcopy(clients[curr_i].model.state_dict())
    w_g_personalized.update(
        (key, value * 0.8) for key, value in w_g_personalized.items())  # 个性化聚合中，自己的模型占一半权重，其他客户端的所有模型，占一半权重
    for k in w_g_personalized.keys():
        for i in range(clients_num):  # 遍历本轮的所有客户端
            if i != curr_i and str(clients[i].model.state_dict()[k].dtype) == 'torch.float32':  # 排除客户端curr_i自己
                w_g_personalized[k] += 0.2 * clients[curr_i].dis_with_other[i] * clients[i].model.state_dict()[k]
    return w_g_personalized


def FedAvg(models):
    w_avg = copy.deepcopy(models[0])
    for k in w_avg.keys():
        for i in range(1, len(models)):
            w_avg[k] += models[i][k]
        w_avg[k] = torch.div(w_avg[k], len(models))
    return w_avg


def test(args, model, device, x, y, criterion, task_id):
    model.eval()
    total_loss = 0
    total_num = 0
    correct = 0
    r = np.arange(x.size(0))
    np.random.shuffle(r)
    r = torch.LongTensor(r)
    with torch.no_grad():
        # Loop batches
        for i in range(0, len(r), args.batch_size_test):
            if i + args.batch_size_test <= len(r):
                b = r[i:i + args.batch_size_test]
            else:
                b = r[i:]
            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output, _ = model(data)
            loss = criterion(output[task_id], target)
            pred = output[task_id].argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item() * len(b)
            total_num += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc


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

    def to(self, device):
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
        for (name1, w1), (name2, w2) in zip(weight_list, old_model_list):
            l2_reg = torch.norm(w1 - w2, p=p)
            reg_loss = reg_loss + l2_reg
        return reg_loss