# -*- coding: utf-8 -*-
import time
import random
import torch
import numpy as np
import argparse
from copy import deepcopy
from client import Client
from client import *
from server import *
from model import *
from dataloader import cifar100 as cf100

if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='CGoFed')   
    parser.add_argument('--seed', type=int, default=2023, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--device', default=0, type=int,
                        help='GPU ID, -1 for CPU')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--pc_valid', default=0.05, type=float,
                        help='fraction of training data used for validation')
    parser.add_argument('--l_epochs', type=int, default=3, metavar='N',
                        help='number of training epochs/task (default: 200)')
    parser.add_argument('--g_epochs', type=int, default=20, metavar='N',
                        help='number of global training epochs (default: 100)')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
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
    parser.add_argument('--alpha', type=float, default=0.99, metavar='alpha',
                        help='importance co-efficeint (default: 5)')
    parser.add_argument('--beta', type=float, default=1, metavar='zeta',
                        help='weight of basis of each task (default: 0.9)')
    parser.add_argument('--tau', type=float, default=0.01, metavar='tau',
                        help='threshold tau')

    # FL specific
    parser.add_argument('--task_num', type=int, default=10,
                        help='the number of task (default: 10)')
    parser.add_argument('--clients_num', type=int, default=5,
                        help='the number of clients (default: 10)')
    parser.add_argument('--selected_clients', type=int, default=2,
                        help='history model of selected')
    parser.add_argument('--test', type=bool, default=False, metavar='N',
                        help='incremental data is the same label or different labels (default: SL)')


    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    tstart = time.time()
    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)

    # initial clients
    acc_matrix = np.zeros((args.task_num, args.task_num))
    data_set = []
    taskcla_list = []
    task_list = []
    clients = []
    model_g = []

    # 一次性将10个客户端的数据分好
    data_set, taskcla_list = cf100.get_data2(seed=args.seed, pc_valid=args.pc_valid, clients_num=args.clients_num,
                                        task_num=args.task_num)

    # 通过循环来给每个客户端分配数据集
    for c_id in range(args.clients_num):
        # 每个客户端一个全局模型
        model_g_temp = AlexNet(taskcla_list[c_id]).to(device)
        model_g.append(model_g_temp)

        # 每个客户创建一个client类
        client_temp = Client(model_g[c_id], args)
        clients.append(client_temp)

    print('Model parameters ---')
    for k_t, (m, param) in enumerate(model_g[0].named_parameters()):
        print(k_t, m, param.shape)
    print('-' * 40)

    # task loop
    # 统计本任务中的所有acc
    all_task_acc = {i: [] for i in range(10)}
    avg_forgetting = []
    for task_id in range(args.task_num):
        print('-' * 100)
        print('Task ID :{} | Learning Rate : {}'.format(task_id, args.lr))
        print('-' * 100)
        threshold = np.array([0.97] * 5) + task_id * np.array([0.003] * 5)

        if task_id == 0:
            g_epochs = 50
        else:
            g_epochs = args.g_epochs
        # for epoch in range(args.g_epochs):
        for epoch in range(g_epochs):

            for c_id in range(args.clients_num):
                
                xtrain = data_set[c_id][task_id]['train']['x']
                ytrain = data_set[c_id][task_id]['train']['y']
                xvalid = data_set[c_id][task_id]['valid']['x']
                yvalid = data_set[c_id][task_id]['valid']['y']
                task_list.append(task_id)

                if task_id == 0:
                    clients[c_id].train_first_task(args, xtrain, ytrain, xvalid, yvalid, task_id, device, threshold,
                                                   c_id, epoch, g_epochs)
                else:
                    old_model_list = select_old_model(clients, c_id, task_id, args.clients_num,
                                                      args.selected_clients)  # @ corss_task_module
                    clients[c_id].train_new_task(args, xtrain, ytrain, xvalid, yvalid, task_id, device, threshold, c_id,
                                                 old_model_list, epoch, avg_forgetting)

                    # 计算原型之间的距离
            for c_id in range(args.clients_num):
                compute_distance_curr_feature(c_id, clients[c_id].curr_AvgProto, clients, args.clients_num)
                # if epoch == args.g_epochs-1:       # ep_g%10=0,表示只保存和每个任务第一轮的距离及其原型，ep_g%10=9表示最后一轮
                if epoch == g_epochs - 1:
                    compute_distance_with_history_AvgProto(c_id, clients, args.clients_num, task_id)

            # personalized aggregation
            avg_test_acc = 0
            avg_test_loss = 0
            for c_id in range(args.clients_num):
                clients[c_id].personalized_global_model = PFL(c_id, clients, args.clients_num)
                model_g[c_id].load_state_dict(clients[c_id].personalized_global_model)
                xtest = data_set[c_id][task_id]['test']['x']
                ytest = data_set[c_id][task_id]['test']['y']
                test_loss, test_acc = test(args, model_g[c_id], device, xtest, ytest, clients[c_id].criterion, task_id)
                avg_test_acc += test_acc
                avg_test_loss += test_loss
            print('-' * 40)
            print('G-Epoch: {:3d}, Test: loss={:.3f} , acc={:5.1f}%'.format(epoch, avg_test_loss / args.clients_num,
                                                                            avg_test_acc / args.clients_num))

            # 在每一轮中都测试全部任务的准确率
            print('*' * 40)
            for t in range(args.task_num):
                avg_test_acc = 0
                avg_test_loss = 0
                for c_id in range(args.clients_num):
                    xtest = data_set[c_id][t]['test']['x']
                    ytest = data_set[c_id][t]['test']['y']
                    test_loss, test_acc = test(args, model_g[c_id], device, xtest, ytest, clients[c_id].criterion, t)
                    avg_test_acc += test_acc
                    avg_test_loss += test_loss
                print('Task: {:3d}, G-Epoch: {:3d}, Test: loss={:.3f} , acc={:5.1f}%'.format(t, epoch,
                                                                                             avg_test_loss / args.clients_num,
                                                                                             avg_test_acc / args.clients_num))
                all_task_acc[t].append(avg_test_acc / args.clients_num)
            print(all_task_acc)

        # save accuracy
        jj = 0
        for ii in range(task_id + 1):
            avg_acc = 0
            for c_id in range(args.clients_num):
                # set_model_(clients[c_id].model,w_avg)
                model_g[c_id].load_state_dict(clients[c_id].personalized_global_model)
                xtest = data_set[c_id][ii]['test']['x']
                ytest = data_set[c_id][ii]['test']['y']
                _, acc = test(args, model_g[c_id], device, xtest, ytest, clients[c_id].criterion, ii)
                avg_acc += acc
            acc_matrix[task_id, jj] = avg_acc / args.clients_num
            jj += 1

        print('Accuracies =')
        for i_a in range(task_id + 1):
            print('\t', end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a, j_a]), end='')
            print()
            # update task id
            # task_id +=1
        # 调用函数
        avg_forgetting.append(calculate_average_forgetting(acc_matrix, task_id))
        print("avg_forgetting", [round(x, 4) for x in avg_forgetting])
    print('-' * 50)
    # Simulation Results
    print('Task Order : {}'.format(np.array(task_list)))
    print('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean()))
    bwt = np.mean((acc_matrix[-1] - np.diag(acc_matrix))[:-1])
    print('Backward transfer: {:5.2f}%'.format(bwt))
    print('[Elapsed time = {:.1f} ms]'.format((time.time() - tstart) * 1000))
    print('-' * 50)
    # Plots
    array = acc_matrix
    df_cm = pd.DataFrame(array, index=[i for i in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]],
                         columns=[i for i in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]])
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
    plt.show()
























