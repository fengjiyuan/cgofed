'''
Author: Guosy_wxy 1579528809@qq.com
Date: 2024-10-15 12:16:55
LastEditors: Guosy_wxy 1579528809@qq.com
LastEditTime: 2024-12-05 07:15:55
FilePath: /CaT-CGL/data_stream.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import random
import numpy as np
from torch_geometric.transforms import RandomNodeSplit
from torch_sparse import SparseTensor
from progressbar import progressbar


class Streaming():
    def __init__(self, cls_per_task, dataset, num_clients):
        self.cls_per_task = cls_per_task
        self.num_clients = num_clients
        self.clients_datas = []
        # self.tasks = self.prepare_tasks(dataset)
        self.tasks = self.prepare_tasks_for_client2(dataset)
        # self.tasks = self.prepare_client(dataset)
        self.n_tasks = len(self.tasks)        
    def prepare_tasks(self, dataset):
        graph = dataset[0]
        tasks = []
        n_tasks = int(dataset.num_classes / self.cls_per_task)
        for k in progressbar(range(n_tasks), redirect_stdout=True): 
            start_cls = k * self.cls_per_task
            classes = list(range(start_cls, start_cls + self.cls_per_task))
            subset = sum(graph.y == cls for cls in classes).nonzero(as_tuple=False).squeeze()
            subgraph = graph.subgraph(subset)
            
            # Split to train/val/test
            transform = RandomNodeSplit(num_val=0.2, num_test=0.2)
            subgraph = transform(subgraph)

            edge_index = subgraph.edge_index
            adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(subgraph.num_nodes, subgraph.num_nodes))
            subgraph.adj_t = adj.t().to_symmetric()  # Arxiv is an directed graph.

            subgraph.task_id = k
            subgraph.classes = classes

            tasks.append(subgraph)
        return tasks

    def prepare_tasks_for_client(self, dataset):
        graph = dataset[0]
        tasks = []
        # clients_datas = []
        n_tasks = int(dataset.num_classes / self.cls_per_task)
        for k in progressbar(range(n_tasks), redirect_stdout=True): 
            start_cls = k * self.cls_per_task
            classes = list(range(start_cls, start_cls + self.cls_per_task))
            # 创建当前任务的节点子集
            subset = sum(graph.y == cls for cls in classes).nonzero(as_tuple=False).squeeze()
            subgraph = graph.subgraph(subset)
            
            # Split to train/val/test
            transform = RandomNodeSplit(num_val=0.2, num_test=0.2)
            subgraph = transform(subgraph)

            edge_index = subgraph.edge_index
            adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(subgraph.num_nodes, subgraph.num_nodes))
            subgraph.adj_t = adj.t().to_symmetric()  # Arxiv is an directed graph.
            
            # 为当前任务添加任务 ID 和类别信息
            subgraph.task_id = k
            subgraph.classes = classes

            # 划分客户端：将节点索引分配给不同客户端
            clients_data = []
            num_nodes_per_client = subgraph.num_nodes // self.num_clients            
            for client_id in range(self.num_clients):
                start_idx = client_id * num_nodes_per_client
                end_idx = (client_id + 1) * num_nodes_per_client if client_id < self.num_clients - 1 else subgraph.num_nodes
                client_node_indices = torch.arange(start_idx, end_idx)

                # 将分配给客户端的节点索引保存为子集
                client_data = {
                    'task_id': k,
                    'client_id': client_id,
                    'node_indices': client_node_indices
                }
                clients_data.append(client_data)

            tasks.append(subgraph)
            self.clients_datas.append(clients_data)
        return tasks


    def prepare_tasks_for_client2(self, dataset):    # 数据分配更平均
        graph = dataset[0]
        
        # clients_datas = []
        n_tasks = int(dataset.num_classes / self.cls_per_task)
        
        # 根据类别数量，将数据集按节点的类进行划分，确保所有任务的节点数量尽可能平均
        original_shape = graph.y.shape # 统计每个类别的节点数量
        class_counts = torch.bincount(graph.y.view(-1))
        print("Class counts:", class_counts)
        sorted_classes = torch.argsort(class_counts, descending=True)   # 获取类别数量，从大到小排列类别的索引
        print("sorted Class counts:", sorted_classes)
        most_classes = sorted_classes[:n_tasks]  # 最少的args.n_tasks个类
        print("most_classes counts:", most_classes)
        least_classes = sorted_classes[-n_tasks:]  # 最多的args.n_tasks个类
        print("least_classes counts:", least_classes)
        delete_classes = []
        print("delete_classes counts:", delete_classes)        

        # 创建任务字典
        tasks = {i: [] for i in range(n_tasks)}    
        
        # 将最多和最少的类分配到任务
        for i, clas in enumerate(most_classes):
            tasks[i % n_tasks].append(clas.item())

        for i, clas in enumerate(least_classes):
            tasks[(i + n_tasks) % n_tasks].append(clas.item())  
        
        # 获取剩下的类别
        remaining_classes = [i for i in range(len(class_counts)) if i not in sum(tasks.values(), []) and i not in delete_classes]
        
        # 随机分配剩下的类别
        random.shuffle(remaining_classes)
        for i, clas in enumerate(remaining_classes):
            tasks[i % n_tasks].append(clas)    
        
        # 更新节点标签
        class_mapping = {}
        for task_id, old_classes in enumerate(tasks.values()):
            for old_class in old_classes:
                class_mapping[old_class] = task_id * self.cls_per_task + old_classes.index(old_class)

        for del_class in delete_classes:
            class_mapping[int(del_class)] = len(class_mapping)

        # class_mapping = {int(old_class): new_class for new_class, old_class in enumerate(sorted_classes)}   # 创建类别映射关系，新的类别编号基于节点数的排序
        print("Class mapping (old to new):", class_mapping)
        new_labels = torch.tensor([class_mapping[label.item()] for label in graph.y], dtype=torch.long) # 遍历所有节点，更新节点的标签
        new_features = graph.x[torch.tensor([class_mapping[label.item()] for label in graph.y], dtype=torch.long)]
        graph.x = new_features
        graph.y = new_labels.view(original_shape)   # 将新的标签赋值给 data.y
        print("Updated labels")
        class_counts = torch.bincount(graph.y.view(-1))
        print("Class counts:", class_counts)        
        
        
        tasks_graph = []
        for k in progressbar(range(n_tasks), redirect_stdout=True): 
            start_cls = k * self.cls_per_task
            classes = list(range(start_cls, start_cls + self.cls_per_task))
            # 创建当前任务的节点子集
            subset = sum(graph.y == cls for cls in classes).nonzero(as_tuple=False).squeeze()
            subgraph = graph.subgraph(subset)
            
            # Split to train/val/test
            transform = RandomNodeSplit(num_val=0.2, num_test=0.2)
            subgraph = transform(subgraph)

            edge_index = subgraph.edge_index
            adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(subgraph.num_nodes, subgraph.num_nodes))
            subgraph.adj_t = adj.t().to_symmetric()  # Arxiv is an directed graph.
            
            # 为当前任务添加任务 ID 和类别信息
            subgraph.task_id = k
            subgraph.classes = classes

            # 划分客户端：将节点索引分配给不同客户端
            clients_data = []
            num_nodes_per_client = subgraph.num_nodes // self.num_clients            
            for client_id in range(self.num_clients):
                start_idx = client_id * num_nodes_per_client
                end_idx = (client_id + 1) * num_nodes_per_client if client_id < self.num_clients - 1 else subgraph.num_nodes
                client_node_indices = torch.arange(start_idx, end_idx)

                # 将分配给客户端的节点索引保存为子集
                client_data = {
                    'task_id': k,
                    'client_id': client_id,
                    'node_indices': client_node_indices
                }
                clients_data.append(client_data)

            tasks_graph.append(subgraph)
            self.clients_datas.append(clients_data)
        return tasks_graph

    def prepare_client(self, dataset):
        graph = dataset[0]
        tasks = []
        
        transform = RandomNodeSplit(num_val=0.2, num_test=0.2)
        graph = transform(graph)
        edge_index = graph.edge_index
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(graph.num_nodes, graph.num_nodes))
        graph.adj_t = adj.t().to_symmetric()  # Arxiv is an directed graph.
        classes = list(range(0, dataset.num_classes))
        graph.task_id = 0
        graph.classes = classes

          
        
        # 划分客户端：将节点索引分配给不同客户端
        clients_data = []
        num_nodes_per_client = graph.num_nodes // self.num_clients            
        for client_id in range(self.num_clients):
            start_idx = client_id * num_nodes_per_client
            end_idx = (client_id + 1) * num_nodes_per_client if client_id < self.num_clients - 1 else graph.num_nodes
            client_node_indices = torch.arange(start_idx, end_idx)

            # 将分配给客户端的节点索引保存为子集
            client_data = {
                'task_id': 0,
                'client_id': client_id,
                'node_indices': client_node_indices
            }
            clients_data.append(client_data)

        tasks.append(graph)      
        self.clients_datas.append(clients_data)        

        return tasks


def set_task(clients_num, tasks_num, classes_per_task):
    taskcla_list = []
    for c_id in range(clients_num):
        # Others
        n_cla = classes_per_task
        taskcla=[]
        for t in range(tasks_num):
            taskcla.append((t, n_cla))
            n_cla += classes_per_task
        taskcla_list.append(taskcla)
    return taskcla_list

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True