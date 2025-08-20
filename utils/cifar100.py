# -*- coding: utf-8 -*-
import os,sys
import numpy as np
import torch
import random
# import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Subset, random_split

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

def load_cifar100_incremental(task_id, classes_per_task, num_clients, data_dir='./data', train_ratio=0.8, batch_size=32):
    """
    Load CIFAR-100 data for a specific task and distribute it among clients.
    
    Parameters:
        task_id (int): The incremental task index (0-based).
        classes_per_task (int): Number of classes per incremental task.
        num_clients (int): Number of clients to distribute the data.
        data_dir (str): Directory to download the dataset.
        batch_size (int): Batch size for DataLoaders.
    
    Returns:
        dict: A dictionary containing a DataLoader for each client.
    """
    # Calculate class range for the current task
    start_class = task_id * classes_per_task
    end_class = start_class + classes_per_task
    selected_classes = list(range(start_class, end_class))

    # Define transformations for CIFAR-100
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 mean and std
    ])

    # Load the CIFAR-100 dataset
    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    
    # Filter datasets for selected classes
    train_indices = [i for i, label in enumerate(train_dataset.targets) if label in selected_classes]
    test_indices = [i for i, label in enumerate(test_dataset.targets) if label in selected_classes]
    
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    # Split train_subset into train and validation subsets
    train_len = int(len(train_subset) * train_ratio)
    val_len = len(train_subset) - train_len
    train_data, val_data = random_split(train_subset, [train_len, val_len])        
    
    client_dataloaders = {}
    for client_id in range(num_clients):
        client_train_data, client_val_data = random_split(train_data, [len(train_data) // num_clients] * num_clients)[client_id], \
                                             random_split(val_data, [len(val_data) // num_clients] * num_clients)[client_id]
        
        client_dataloaders[f"client_{client_id}"] = {
            'train': DataLoader(client_train_data, batch_size=batch_size, shuffle=True, num_workers=2),
            'val': DataLoader(client_val_data, batch_size=batch_size, shuffle=False, num_workers=2),
            'test': DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        }
    return client_dataloaders







    # for client_id in range(num_clients):
    #     # Get the indices for the current client's subset
    #     start_idx = client_id * num_samples_per_client
    #     end_idx = start_idx + num_samples_per_client
    #     client_indices = indices[start_idx:end_idx]
    #     client_subset = Subset(dataset, client_indices)

    #     # Create a DataLoader for the client's subset
    #     client_dataloader = DataLoader(client_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    #     client_dataloaders[f"client_{client_id}"] = client_dataloader

    # return client_dataloaders
    

def get(seed=0,pc_valid=0.10):
    data={}
    taskcla=[]
    size=[3,32,32]

    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]

        # CIFAR100
        dat={}
        dat['train']=datasets.CIFAR100(cf100_dir,train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR100(cf100_dir,train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        # dat['train'] = datasets.CIFAR100(cf100_dir,train=True,download=False,transform=transforms.Compose([transforms.ToTensor()]))
        # dat['test']  = datasets.CIFAR100(cf100_dir,train=False,download=False,transform=transforms.Compose([transforms.ToTensor()]))
        for n in range(10):
            data[n]={}
            data[n]['name']='cifar100'
            data[n]['ncla']=10
            data[n]['train']={'x': [],'y': []}
            data[n]['test']={'x': [],'y': []}
        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            for image,target in loader:
                n=target.numpy()[0]
                nn=(n//10)
                data[nn][s]['x'].append(image) # 255 
                data[nn][s]['y'].append(n%10)

        # "Unify" and save
        for t in data.keys():
            for s in ['train','test']:
                data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,size[0],size[1],size[2])
                data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
                torch.save(data[t][s]['x'], os.path.join(os.path.expanduser(file_dir),'data'+str(t)+s+'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(os.path.expanduser(file_dir),'data'+str(t)+s+'y.bin'))

    # Load binary files
    data={}
    # ids2=list(shuffle(np.arange(10),random_state=seed))
    # print('Task order =',ids2)    
    ids=list(np.arange(10))     # 固定为10个task
    random.shuffle(ids)
    print('Task order =',ids)
    for i in range(10):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser(file_dir),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser(file_dir),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
        if data[i]['ncla']==2:
            data[i]['name']='cifar10-'+str(ids[i])
        else:
            data[i]['name']='cifar100-'+str(ids[i])

    # Validation
    for t in data.keys():
        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size


def get2(seed=0,pc_valid=0.10, clients_num=10, task_num=5):
    
    data_set = []
    taskcla_list = []
    task_list = []
    size=[3,32,32]
    
    # 随机生成任务分配列表
    task_list = [random.sample(range(task_num), task_num) for _ in range(clients_num)]
    task_list = np.array(task_list).T  # 转置矩阵

    print('Task order =')
    print(task_list)

    # for c_id in range(clients_num):
    #     ids=list(np.arange(task_num))     # 固定为10个task
    #     random.shuffle(ids)
    #     task_list.append(ids)
    # # print('Task order =',task_list)        
    # task_list = list(map(list, zip(*task_list)))  #转置
    # print('Task order =')
    # print(np.array(task_list))
    
    
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
        mean=[x/255 for x in [125.3,123.0,113.9]]    
        std=[x/255 for x in [63.0,62.1,66.7]]
        
        # CIFAR100
        dat={}
        dat['train']=datasets.CIFAR100(cf100_dir,train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR100(cf100_dir,train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))        

        data={}
        for n in range(10):
            data[n]={}
            data[n]['name']='cifar100'
            data[n]['ncla']=10
            data[n]['train']={'x': [],'y': []}
            data[n]['test']={'x': [],'y': []}            
        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            for image,target in loader:
                n=target.numpy()[0]
                nn=(n//10)
                data[nn][s]['x'].append(image) # 255 
                data[nn][s]['y'].append(n%10)        
    
        # "Unify" and save
        for t in data.keys():
            for s in ['train','test']:
                data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,size[0],size[1],size[2])
                data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
                torch.save(data[t][s]['x'], os.path.join(os.path.expanduser(file_dir),'data'+str(t)+s+'x.bin'))
                torch.save(data[t][s]['y'], os.path.join(os.path.expanduser(file_dir),'data'+str(t)+s+'y.bin'))    
    
    
    # Load binary files
    data={}
    for i in range(len(task_list)):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            for c_id in range(clients_num):
                data[i][s]['x']=torch.load(os.path.join(os.path.expanduser(file_dir),'data'+str(task_list[i][c_id])+s+'x.bin'))
                data[i][s]['y']=torch.load(os.path.join(os.path.expanduser(file_dir),'data'+str(task_list[i][c_id])+s+'y.bin'))
        data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
        if data[i]['ncla']==2:
            data[i]['name']='cifar10-'+str(i)
        else:
            data[i]['name']='cifar100-'+str(i)

    # Validation, 训练集分为验证集和训练集
    for t in data.keys():
        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # 现在要将训练集和验证集分为若干个客户端
    # 统计每个任务训练集，测试集，验证集的样本总数
    avg_num_train = len(data[i]['train']['y'].numpy())/clients_num
    avg_num_test = len(data[i]['test']['y'].numpy())/clients_num
    avg_num_valid = len(data[i]['valid']['y'].numpy())/clients_num

    for c_id in range(clients_num):
        client_data = {}
        # 将data的数据拆分为client_data
        for t in data.keys():
            client_data[t] = dict.fromkeys(['name','ncla','train','test','valid'])
            client_data[t]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
            client_data[t]['train']={'x':[],'y':[]}
            client_data[t]['test']={'x':[],'y':[]}
            client_data[t]['valid']={'x':[],'y':[]}
            client_data[t]['train']['x']=data[t]['train']['x'][int(c_id*avg_num_train):int((c_id+1)*avg_num_train)]
            client_data[t]['train']['y']=data[t]['train']['y'][int(c_id*avg_num_train):int((c_id+1)*avg_num_train)]
            client_data[t]['test']['x']=data[t]['test']['x'][int(c_id*avg_num_test):int((c_id+1)*avg_num_test)]
            client_data[t]['test']['y']=data[t]['test']['y'][int(c_id*avg_num_test):int((c_id+1)*avg_num_test)]
            client_data[t]['valid']['x']=data[t]['valid']['x'][int(c_id*avg_num_valid):int((c_id+1)*avg_num_valid)]
            client_data[t]['valid']['y']=data[t]['valid']['y'][int(c_id*avg_num_valid):int((c_id+1)*avg_num_valid)]
            if client_data[t]['ncla']==2:
                client_data[t]['name']='cifar10-'+str(t)
            else:
                client_data[t]['name']='cifar100-'+str(t)    
        # Others
        n=0
        taskcla=[]
        for t in client_data.keys():
            taskcla.append((t,client_data[t]['ncla']))
            n+=client_data[t]['ncla']
        client_data['ncla']=n
        data_set.append(client_data)
        taskcla_list.append(taskcla)
    return data_set,taskcla_list

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