import os
import numpy as np
import torch
import random
# import utils
from torchvision import datasets, transforms
from sklearn.utils import shuffle

# cf100_dir = '/userhome/data/'
# file_dir = '/userhome/data/binary_cifar100'

# cf100_dir = '/home/admin/gpfl/dataset/'
# file_dir = '/home/admin/gpfl/dataset/binary_cifar100'
cf100_dir = '/root/yx/fjy/cgofed/data/cifar100/'
file_dir = '/root/yx/fjy/cgofed/data/binary_cifar100'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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
        dat['train'] = datasets.CIFAR100(cf100_dir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dat['test'] = datasets.CIFAR100(cf100_dir, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
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

    # Load binary files
    data = {}
    # ids2=list(shuffle(np.arange(10),random_state=seed))
    # print('Task order =',ids2)
    ids = list(np.arange(10))
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