import os
import numpy as np
import torch
import random
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Subset, random_split


cf100_dir = '/root/yx/fjy/cgofed/data/cifar100/'
file_dir = '/root/yx/fjy/cgofed/data/binary_cifar100'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data(seed=0, pc_valid=0.10, clients_num=10, task_num=5):
    data = {}
    size = [3, 32, 32]
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        # CIFAR100
        dat = {}
        dat['train'] = datasets.CIFAR100(cf100_dir, train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dat['test'] = datasets.CIFAR100(cf100_dir, train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))

        for n in range(task_num):
            data[n] = {}
            data[n]['name'] = 'cifar100'
            data[n]['ncla'] = 10
            data[n]['train'] = {'x': [], 'y': []}
            data[n]['test'] = {'x': [], 'y': []}

        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            for image, target in loader:
                n = target.numpy()[0]
                nn = (n // 10)
                data[nn][s]['x'].append(image)  # 255
                data[nn][s]['y'].append(n % 10)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'], os.path.join(os.path.expanduser(file_dir), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(os.path.expanduser(file_dir), 'data' + str(t) + s + 'y.bin'))

    
    # 按不同顺序创建任务
    data_set = []
    taskcla_list = []
    task_list = []
    print('Task order =')
    for c_id in range(clients_num):
        ids = list(np.arange(task_num))  # 固定为10个task
        random.shuffle(ids)
        task_list.append(ids)
        print('client_',c_id, ":", ids)

    # Load binary files
    data = {}
    for i in range(task_num):
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser(file_dir), 'data' + str(i) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser(file_dir), 'data' + str(i) + s + 'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        if data[i]['ncla'] == 2:
            data[i]['name'] = 'cifar10-' + str(i)
        else:
            data[i]['name'] = 'cifar100-' + str(i)

    # Validation, 训练集分为验证集和训练集
    for t in data.keys():
        r = np.arange(data[t]['train']['x'].size(0))
        r = np.array(shuffle(r, random_state=seed), dtype=int)
        nvalid = int(pc_valid * len(r))
        ivalid = torch.LongTensor(r[:nvalid])
        itrain = torch.LongTensor(r[nvalid:])
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y'] = data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x'] = data[t]['train']['x'][itrain].clone()
        data[t]['train']['y'] = data[t]['train']['y'][itrain].clone()

    # 现在要将训练集和验证集分为若干个客户端
    # 统计每个任务训练集，测试集，验证集的样本总数
    avg_num_train = len(data[i]['train']['y'].numpy()) / clients_num
    avg_num_test = len(data[i]['test']['y'].numpy()) / clients_num
    avg_num_valid = len(data[i]['valid']['y'].numpy()) / clients_num

    noniid = True
    if noniid == False:
        for c_id in range(clients_num):
            client_data = {}
            true_t= 0
            # 将data的数据拆分为client_data
            for t in task_list[c_id]:
                client_data[true_t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
                client_data[true_t]['ncla'] = len(np.unique(data[0]['train']['y'].numpy()))
                client_data[true_t]['train'] = {'x': [], 'y': []}
                client_data[true_t]['test'] = {'x': [], 'y': []}
                client_data[true_t]['valid'] = {'x': [], 'y': []}
                client_data[true_t]['train']['x'] = data[t]['train']['x'][int(c_id * avg_num_train):int((c_id + 1) * avg_num_train)]
                client_data[true_t]['train']['y'] = data[t]['train']['y'][int(c_id * avg_num_train):int((c_id + 1) * avg_num_train)]
                client_data[true_t]['test']['x']  = data[t]['test']['x'][int(c_id * avg_num_test):int((c_id + 1) * avg_num_test)]
                client_data[true_t]['test']['y']  = data[t]['test']['y'][int(c_id * avg_num_test):int((c_id + 1) * avg_num_test)]
                client_data[true_t]['valid']['x'] = data[t]['valid']['x'][int(c_id * avg_num_valid):int((c_id + 1) * avg_num_valid)]
                client_data[true_t]['valid']['y'] = data[t]['valid']['y'][int(c_id * avg_num_valid):int((c_id + 1) * avg_num_valid)]
                if client_data[true_t]['ncla'] == 2:
                    client_data[true_t]['name'] = 'cifar10-' + str(t)
                else:
                    client_data[true_t]['name'] = 'cifar100-' + str(t)
                    # Others
                true_t = true_t + 1
            n = 0
            taskcla = []
            for t in client_data.keys():
                taskcla.append((t, client_data[t]['ncla']))
                n += client_data[t]['ncla']
            client_data['ncla'] = n
            data_set.append(client_data)
            taskcla_list.append(taskcla)
        return data_set, taskcla_list

    else:
        # ===== Dirichlet Non-IID 划分实现开始 =====
        def _dirichlet_partition_indices(y_np, n_clients, alpha, seed=0, min_per_client=2, max_retry=50):
            rng = np.random.default_rng(seed)
            y_np = np.asarray(y_np)
            classes = np.unique(y_np)

            for _ in range(max_retry):
                client_idx_list = [[] for _ in range(n_clients)]
                for c in classes:
                    cls_idx = np.where(y_np == c)[0]
                    rng.shuffle(cls_idx)
                    props = rng.dirichlet(np.full(n_clients, alpha))
                    counts = rng.multinomial(len(cls_idx), props)
                    start = 0
                    for cid, cnt in enumerate(counts):
                        if cnt > 0:
                            client_idx_list[cid].extend(cls_idx[start:start+cnt])
                        start += cnt
                # 检查最小样本约束
                if all(len(idx) >= min_per_client for idx in client_idx_list):
                    for cid in range(n_clients):
                        rng.shuffle(client_idx_list[cid])
                    return client_idx_list
            # 兜底：返回最后一次结果
            for cid in range(n_clients):
                rng.shuffle(client_idx_list[cid])
            return client_idx_list

        alpha_dir = 0.5  # 可调节 Non-IID 程度
        # 为每个客户端构造数据
        for c_id in range(clients_num):
            client_data = {}
            true_t= 0
            for t in task_list[c_id]:
                client_data[true_t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
                client_data[true_t]['train'] = {'x': [], 'y': []}
                client_data[true_t]['test']  = {'x': [], 'y': []}
                client_data[true_t]['valid'] = {'x': [], 'y': []}        
                # ===== 只对 train 做 Dirichlet Non-IID 划分 =====
                y_np = data[t]['train']['y'].numpy()
                parts = _dirichlet_partition_indices(
                    y_np, clients_num, alpha=alpha_dir,
                    seed=seed + 100 * int(t))
                idx_this_client = parts[c_id]
                client_data[true_t]['train']['x'] = data[t]['train']['x'][idx_this_client].clone()
                client_data[true_t]['train']['y'] = data[t]['train']['y'][idx_this_client].clone()
                # ===== valid 和 test 保持全局一致 (每个客户端都拿全部) =====
                client_data[true_t]['valid']['x'] = data[t]['valid']['x'].clone()
                client_data[true_t]['valid']['y'] = data[t]['valid']['y'].clone()
                client_data[true_t]['test']['x']  = data[t]['test']['x'].clone()
                client_data[true_t]['test']['y']  = data[t]['test']['y'].clone()

                # 任务元信息（保持全局 ncla）
                client_data[true_t]['ncla'] = data[t]['ncla']
                if client_data[true_t]['ncla'] == 2:
                    client_data[true_t]['name'] = 'cifar10-' + str(t)
                else:
                    client_data[true_t]['name'] = 'cifar100-' + str(t)
                true_t = true_t + 1
            # 统计该客户端的 taskcla / ncla
            n = 0
            taskcla = []
            for t in client_data.keys():
                taskcla.append((t, client_data[t]['ncla']))
                n += client_data[t]['ncla']
            client_data['ncla'] = n

            data_set.append(client_data)
            taskcla_list.append(taskcla)
        # ===== Dirichlet Non-IID 划分实现结束 =====
        return data_set, taskcla_list

def get(seed=0, pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]

    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        # CIFAR100
        dat = {}
        dat['train'] = datasets.CIFAR100(cf100_dir, train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dat['test'] = datasets.CIFAR100(cf100_dir, train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))

        for n in range(10):
            data[n] = {}
            data[n]['name'] = 'cifar100'
            data[n]['ncla'] = 10
            data[n]['train'] = {'x': [], 'y': []}
            data[n]['test'] = {'x': [], 'y': []}

        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            for image, target in loader:
                n = target.numpy()[0]
                nn = (n // 10)
                data[nn][s]['x'].append(image)  # 255
                data[nn][s]['y'].append(n % 10)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'], os.path.join(os.path.expanduser(file_dir), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(os.path.expanduser(file_dir), 'data' + str(t) + s + 'y.bin'))

    data = {}
    # ids2=list(shuffle(np.arange(10),random_state=seed))
    # print('Task order =',ids2)
    ids = list(np.arange(10))  # 固定为10个task
    random.shuffle(ids)
    print('Task order =', ids)
    for i in range(10):
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser(file_dir), 'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser(file_dir), 'data' + str(ids[i]) + s + 'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        if data[i]['ncla'] == 2:
            data[i]['name'] = 'cifar10-' + str(ids[i])
        else:
            data[i]['name'] = 'cifar100-' + str(ids[i])

    # Validation
    for t in data.keys():
        r = np.arange(data[t]['train']['x'].size(0))
        r = np.array(shuffle(r, random_state=seed), dtype=int)
        nvalid = int(pc_valid * len(r))
        ivalid = torch.LongTensor(r[:nvalid])
        itrain = torch.LongTensor(r[nvalid:])
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y'] = data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x'] = data[t]['train']['x'][itrain].clone()
        data[t]['train']['y'] = data[t]['train']['y'][itrain].clone()

    # Others
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size


def new_get(seed=0, pc_valid=0.10, clients_num=10, task_num=5):
    data_set = []
    taskcla_list = []
    task_list = []
    print('Task order =')
    for c_id in range(clients_num):
        ids = list(np.arange(task_num))  # 固定为10个task
        random.shuffle(ids)
        task_list.append(ids)
        print('client_',c_id, ":", ids)
    
    # Load binary files
    data = {}
    for c_id in range(clients_num):
        for i in task_list[c_id]:
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            for s in ['train', 'test']:
                data[i][s] = {'x': [], 'y': []}
                data[i][s]['x'] = torch.load(
                    os.path.join(os.path.expanduser(file_dir), 'data' + str(task_list[i][c_id]) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(
                    os.path.join(os.path.expanduser(file_dir), 'data' + str(task_list[i][c_id]) + s + 'y.bin'))               
            data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
            if data[i]['ncla'] == 2:
                data[i]['name'] = 'cifar10-' + str(i)
            else:
                data[i]['name'] = 'cifar100-' + str(i)

    # Validation, 训练集分为验证集和训练集
    for t in data.keys():
        r = np.arange(data[t]['train']['x'].size(0))
        r = np.array(shuffle(r, random_state=seed), dtype=int)
        nvalid = int(pc_valid * len(r))
        ivalid = torch.LongTensor(r[:nvalid])
        itrain = torch.LongTensor(r[nvalid:])
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y'] = data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x'] = data[t]['train']['x'][itrain].clone()
        data[t]['train']['y'] = data[t]['train']['y'][itrain].clone()

    # 现在要将训练集和验证集分为若干个客户端
    # 统计每个任务训练集，测试集，验证集的样本总数
    avg_num_train = len(data[i]['train']['y'].numpy()) / clients_num
    avg_num_test = len(data[i]['test']['y'].numpy()) / clients_num
    avg_num_valid = len(data[i]['valid']['y'].numpy()) / clients_num

    for c_id in range(clients_num):
        client_data = {}
        # 将data的数据拆分为client_data
        for t in data.keys():
            client_data[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
            client_data[t]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
            client_data[t]['train'] = {'x': [], 'y': []}
            client_data[t]['test'] = {'x': [], 'y': []}
            client_data[t]['valid'] = {'x': [], 'y': []}
            client_data[t]['train']['x'] = data[t]['train']['x'][
                                           int(c_id * avg_num_train):int((c_id + 1) * avg_num_train)]
            client_data[t]['train']['y'] = data[t]['train']['y'][
                                           int(c_id * avg_num_train):int((c_id + 1) * avg_num_train)]
            client_data[t]['test']['x'] = data[t]['test']['x'][int(c_id * avg_num_test):int((c_id + 1) * avg_num_test)]
            client_data[t]['test']['y'] = data[t]['test']['y'][int(c_id * avg_num_test):int((c_id + 1) * avg_num_test)]
            client_data[t]['valid']['x'] = data[t]['valid']['x'][
                                           int(c_id * avg_num_valid):int((c_id + 1) * avg_num_valid)]
            client_data[t]['valid']['y'] = data[t]['valid']['y'][
                                           int(c_id * avg_num_valid):int((c_id + 1) * avg_num_valid)]
            if client_data[t]['ncla'] == 2:
                client_data[t]['name'] = 'cifar10-' + str(t)
            else:
                client_data[t]['name'] = 'cifar100-' + str(t)
                # Others
        n = 0
        taskcla = []
        for t in client_data.keys():
            taskcla.append((t, client_data[t]['ncla']))
            n += client_data[t]['ncla']
        client_data['ncla'] = n
        data_set.append(client_data)
        taskcla_list.append(taskcla)
    return data_set, taskcla_list

def get_cifar(class_increment_mode, seed = 0, pc_valid=0.10, clients_num=10):
    
    if class_increment_mode == 'SL':
        file_path = file_dir
        class_increment = 0
    elif class_increment_mode == 'DL':
        file_path = '/root/yx/fjy/cgofed/data/binary_cifar100'
        class_increment = 1
    
    data_set = []
    taskcla = []
    taskcla_list = []
    client_list = []
    data={}
    for c_id in range(clients_num):
        ids=list(np.arange(10))     # 固定为10个task
        client_list.append(ids)

    print('Task order:')
    for i in range(len(client_list)):
        print("clinet{}".format(i), np.array(client_list[i]))         
    task_list = list(map(list, zip(*client_list)))  #转置
    # Load binary files  字典中的每个元素代表一个任务
    for i in range(len(task_list)):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser(file_path),'data'+str(i)+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser(file_path),'data'+str(i)+s+'y.bin'))
        data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))*(i*class_increment+1)
        if data[i]['ncla']==2:
            data[i]['name']='cifar10-'+str(i)
        else:
            data[i]['name']='cifar100-'+str(i)

    # 现在要将训练集和验证集分为若干个客户端
    # 统计每个任务训练集，测试集，验证集的样本总数
    train_num_c = len(data[i]['train']['y'].numpy())/clients_num
    test_num_c = len(data[i]['test']['y'].numpy())/clients_num

    for t in data.keys():
        client_data = {}
        for c_id in range(clients_num):
            client_data[c_id] = dict.fromkeys(['name','ncla','train','test','valid'])
            client_data[c_id]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))*(t*class_increment+1)
            client_data[c_id]['train']={'x':[],'y':[]}
            client_data[c_id]['test']={'x':[],'y':[]}
            client_data[c_id]['valid']={'x':[],'y':[]}  
            client_data[c_id]['train']['x']=data[t]['train']['x'][int(c_id*train_num_c):int((c_id+1)*train_num_c)]
            client_data[c_id]['train']['y']=data[t]['train']['y'][int(c_id*train_num_c):int((c_id+1)*train_num_c)]
            # client_data[c_id]['test']['x']=data[t]['test']['x']
            # client_data[c_id]['test']['y']=data[t]['test']['y']             
            client_data[c_id]['test']['x']=data[t]['test']['x'][int(c_id*test_num_c):int((c_id+1)*test_num_c)]
            client_data[c_id]['test']['y']=data[t]['test']['y'][int(c_id*test_num_c):int((c_id+1)*test_num_c)]            
            client_data[c_id]['valid']['x']=data[t]['train']['x'][-int(train_num_c*pc_valid):]
            client_data[c_id]['valid']['y']=data[t]['train']['y'][-int(train_num_c*pc_valid):]
            if client_data[c_id]['ncla']==2:
                client_data[c_id]['name']='cifar10-'+str(t)
            else:
                client_data[c_id]['name']='cifar100-'+str(t)        
    
        # Others
        data_set.append(client_data)
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))    
    return data_set,taskcla

def get2(seed=0, pc_valid=0.10, clients_num=10, task_num=5):
    data_set = []
    taskcla_list = []
    task_list = []
    for c_id in range(clients_num):
        ids = list(np.arange(task_num))  # 固定为10个task
        random.shuffle(ids)
        task_list.append(ids)
    #     print('Task order =',task_list)
    task_list = list(map(list, zip(*task_list)))  # 转置
    print('Task order =')
    print(np.array(task_list))
    # Load binary files
    data = {}
    for i in range(len(task_list)):
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            for c_id in range(clients_num):
                data[i][s]['x'] = torch.load(
                    os.path.join(os.path.expanduser(file_dir), 'data' + str(task_list[i][c_id]) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(
                    os.path.join(os.path.expanduser(file_dir), 'data' + str(task_list[i][c_id]) + s + 'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        if data[i]['ncla'] == 2:
            data[i]['name'] = 'cifar10-' + str(i)
        else:
            data[i]['name'] = 'cifar100-' + str(i)


    # Validation, 训练集分为验证集和训练集
    for t in data.keys():
        r = np.arange(data[t]['train']['x'].size(0))
        r = np.array(shuffle(r, random_state=seed), dtype=int)
        nvalid = int(pc_valid * len(r))
        ivalid = torch.LongTensor(r[:nvalid])
        itrain = torch.LongTensor(r[nvalid:])
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y'] = data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x'] = data[t]['train']['x'][itrain].clone()
        data[t]['train']['y'] = data[t]['train']['y'][itrain].clone()

    # 现在要将训练集和验证集分为若干个客户端
    # 统计每个任务训练集，测试集，验证集的样本总数
    avg_num_train = len(data[i]['train']['y'].numpy()) / clients_num
    avg_num_test = len(data[i]['test']['y'].numpy()) / clients_num
    avg_num_valid = len(data[i]['valid']['y'].numpy()) / clients_num

    for c_id in range(clients_num):
        client_data = {}
        # 将data的数据拆分为client_data
        for t in data.keys():
            client_data[t] = dict.fromkeys(['name', 'ncla', 'train', 'test', 'valid'])
            client_data[t]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
            client_data[t]['train'] = {'x': [], 'y': []}
            client_data[t]['test'] = {'x': [], 'y': []}
            client_data[t]['valid'] = {'x': [], 'y': []}
            client_data[t]['train']['x'] = data[t]['train']['x'][
                                           int(c_id * avg_num_train):int((c_id + 1) * avg_num_train)]
            client_data[t]['train']['y'] = data[t]['train']['y'][
                                           int(c_id * avg_num_train):int((c_id + 1) * avg_num_train)]
            client_data[t]['test']['x'] = data[t]['test']['x'][int(c_id * avg_num_test):int((c_id + 1) * avg_num_test)]
            client_data[t]['test']['y'] = data[t]['test']['y'][int(c_id * avg_num_test):int((c_id + 1) * avg_num_test)]
            client_data[t]['valid']['x'] = data[t]['valid']['x'][
                                           int(c_id * avg_num_valid):int((c_id + 1) * avg_num_valid)]
            client_data[t]['valid']['y'] = data[t]['valid']['y'][
                                           int(c_id * avg_num_valid):int((c_id + 1) * avg_num_valid)]
            if client_data[t]['ncla'] == 2:
                client_data[t]['name'] = 'cifar10-' + str(t)
            else:
                client_data[t]['name'] = 'cifar100-' + str(t)
                # Others
        n = 0
        taskcla = []
        for t in client_data.keys():
            taskcla.append((t, client_data[t]['ncla']))
            n += client_data[t]['ncla']
        client_data['ncla'] = n
        data_set.append(client_data)
        taskcla_list.append(taskcla)
    return data_set, taskcla_list