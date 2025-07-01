import glob
import os
import shutil
import os.path as osp
import pandas as pd
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.data import Data
import torch
from tqdm import tqdm
import numpy as np
import torch_geometric.transforms as T

def edges_txt2list(inpath):
    with open(inpath, 'r') as f:
        data = []
        for line in f:
            data_line = line.strip('\n').split(',')
            data.append([int(i) for i in data_line])
    return data


def get_None_neuronsID():
    None_path = 'data/source_data/coarsen_skeleton/None'
    csvs = glob.glob(f'{None_path}/*')
    None_neuronsIDs = []
    for csv in csvs:
        None_neuronsID = int(csv.strip().split('.')[0].split(os.sep)[-1])
        None_neuronsIDs.append(None_neuronsID)
    indir = '/data2/liaominghui/project/HemiBrain/data/source_data/data_for_wangguojia'
    neuronID = edges_txt2list(osp.join(indir, 'neuron2ID.txt'))
    neuron2ID_dir = {}
    for i in range(len(neuronID)):
        key, value = int(neuronID[i][0]), int(neuronID[i][1])
        neuron2ID_dir[key] = value
    None_IDs = []
    for neuron in None_neuronsIDs:
        None_IDs.append(neuron2ID_dir[neuron])

    return None_IDs


def read_csv(path, ID, neuron):
    pos = pd.read_csv(path)
    pos_list = pos.values.tolist()
    pos_tensor = torch.tensor(pos_list).squeeze()
    data = Data(pos=pos_tensor, ID=ID, neuron=neuron)
    return data


def get_neuronID():
    data = {}
    df = pd.read_csv('data/Fly_source/ol_columns.csv')
    for id in range(df.shape[0]):
        neuron = df.iloc[id, 0]
        data[neuron] = int(id)
    return data


class FlySkeleton(InMemoryDataset):
    def __init__(self, root, load_data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, transform=None, pre_transform=None, pre_filter=None):
        # assert train in [True, False]
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        super().__init__(root, transform, pre_transform, pre_filter)
        assert load_data in ['data', 'train', 'val', 'test']
        if load_data == 'data':
            path = self.processed_paths[0]
        if load_data == 'train':
            path = self.processed_paths[1]
        elif load_data == 'val':
            path = self.processed_paths[2]
        elif load_data == 'test':
            path = self.processed_paths[3]
        self.data, self.slices = torch.load(path)
        

    @property
    def raw_file_names(self):
        # return the all file names in th file: raw
        # neurons_label = pd.read_csv('Data/data_for_wangguojia/neuron-label.txt', sep='\t', header=None)
        # lables = sorted(list(set(neurons_label.iloc[:, 1].tolist())))
        # return lables

        paths = glob.glob(f'{self.raw_dir}/*')
        types = []
        for path in paths:
            type = path.split('/')[-1]
            types.append(type)
        return types

    @property
    def processed_file_names(self):
        # return the all file names in th file: processed
        return ['data.pt', 'train.pt', 'val.pt', 'test.pt']

    def download(self):
        pass

    def process(self):
        data, train, val, test = self.process_set()
        torch.save(data, self.processed_paths[0])
        torch.save(train, self.processed_paths[1])
        torch.save(val, self.processed_paths[2])
        torch.save(test, self.processed_paths[3])

    def process_set(self):
        # to create BrianData using raw file and Data class
        categories = glob.glob(f'{self.raw_dir}/*')
        categories = sorted([x.split(os.sep)[-1] for x in categories])
        ###
        categories.pop(categories.index('not assigned'))
        categories.append('not assigned')
        ###
        neuron2ID = get_neuronID()
        data_list = []
        print("Creating Fly SkeletonData...")
        num_neuron_within_class = 0
        for target, category in enumerate(tqdm(categories, position=0)):
            folder = osp.join(self.raw_dir, category)
            paths = glob.glob(f'{folder}/*')
            for path in paths:
                if category != 'not assigned':
                    num_neuron_within_class += 1
                neuron = int(path.split(os.sep)[-1].split('.')[0])
                ID = neuron2ID[neuron]
                data = read_csv(path, ID, neuron)
                data.y = torch.tensor([target])
                data_list.append(data)
        

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        index = torch.randperm(num_neuron_within_class).tolist()

        train_index = index[:int(self.train_ratio * num_neuron_within_class)]
        val_index = index[int(self.train_ratio * num_neuron_within_class):int(self.train_ratio * num_neuron_within_class) + int(self.val_ratio * num_neuron_within_class)]
        test_index = index[int(self.train_ratio * num_neuron_within_class) + int(self.val_ratio * num_neuron_within_class):]

        train_list = [data_list[i] for i in train_index]
        val_list = [data_list[i] for i in val_index]
        test_list = [data_list[i] for i in test_index]
        test_list.extend(data_list[num_neuron_within_class:])
        
        return self.collate(data_list), self.collate(train_list), self.collate(val_list), self.collate(test_list)

    def __repr__(self) -> str:
        # print the BrainData class info for print(BrainData)
        return f'{self.__class__.__name__}({len(self)})'


if __name__ == '__main__':
    path = f'data/FlySkeleton'
    pre_transform, transform = T.NormalizeScale(), T.FixedPoints(1024)
    data_dataset = FlySkeleton(path, 'data', transform=transform, pre_transform=pre_transform)
    train_dataset = FlySkeleton(path, 'train', transform=transform, pre_transform=pre_transform)
    test_dataset = FlySkeleton(path, 'test', transform=transform, pre_transform=pre_transform)
    pass





























