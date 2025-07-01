import glob
import os
import shutil
import os.path as osp
import pandas as pd
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.data import Data
from typing import Union, List, Tuple
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius
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
    with open('/data2/liaominghui/project/HemiBrain/data/manc_class/manc_class_info/neuron2ID.txt', 'r') as file:
        data = {}
        for line in file:
            neuron, ID = [int(x) for x in line.strip().split(' ')]
            data[neuron] = ID
    return data


class MancSkeleton_class(InMemoryDataset):
    def __init__(self, root, load_data, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1,
                 transform=None, pre_transform=None, pre_filter=None):
        # assert train in [True, False]
        self.train_ratio, self.test_ratio, self.val_ratio, = train_ratio, test_ratio, val_ratio
        super().__init__(root, transform, pre_transform, pre_filter)
        # assert load_data in ['data', 'train001', 'test', 'val']
        if load_data == 'data':
            path = self.processed_paths[0]
        elif load_data == 'train001':
            path = self.processed_paths[1]
        elif load_data == 'train003':
            path = self.processed_paths[2]
        elif load_data == 'train005':
            path = self.processed_paths[3]
        elif load_data == 'train010':
            path = self.processed_paths[4]
        elif load_data == 'train020':
            path = self.processed_paths[5]
        elif load_data == 'train030':
            path = self.processed_paths[6]
        elif load_data == 'train040':
            path = self.processed_paths[7]
        elif load_data == 'train050':
            path = self.processed_paths[8]
        elif load_data == 'train060':
            path = self.processed_paths[9]
        elif load_data == 'train070':
            path = self.processed_paths[10]
        elif load_data == 'train080':
            path = self.processed_paths[11]
        elif load_data == 'train090':
            path = self.processed_paths[12]
        elif load_data == 'train100':
            path = self.processed_paths[13]
        elif load_data == 'test':
            path = self.processed_paths[14]
        elif load_data == 'val':
            path = self.processed_paths[15]
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
        return ['data.pt', 'train001.pt', 'train003.pt', 'train005.pt', 'train010.pt', 'train020.pt', 'train030.pt',
                'train040.pt', 'train050.pt', 'train060.pt', 'train070.pt', 'train080.pt', 'train090.pt', 'train100.pt',
                 'test.pt', 'val.pt']

    def download(self):
        pass

    def process(self):
        data, train001, train003, train005, train010, train020, train030, train040, train050, train060, train070, \
            train080, train090, train100, test, val = self.process_set()
        torch.save(data, self.processed_paths[0])
        torch.save(train001, self.processed_paths[1])
        torch.save(train003, self.processed_paths[2])
        torch.save(train005, self.processed_paths[3])
        torch.save(train010, self.processed_paths[4])
        torch.save(train020, self.processed_paths[5])
        torch.save(train030, self.processed_paths[6])
        torch.save(train040, self.processed_paths[7])
        torch.save(train050, self.processed_paths[8])
        torch.save(train060, self.processed_paths[9])
        torch.save(train070, self.processed_paths[10])
        torch.save(train080, self.processed_paths[11])
        torch.save(train090, self.processed_paths[12])
        torch.save(train100, self.processed_paths[13])
        torch.save(test, self.processed_paths[14])
        torch.save(val, self.processed_paths[15])

    def process_set(self):
        # to create BrianData using raw file and Data class
        categories = glob.glob(f'{self.raw_dir}/*')
        categories = sorted([x.split(os.sep)[-1] for x in categories])
        neuron2ID = get_neuronID()
        data_list = []
        print("Creating SkeletonData...")
        for target, category in enumerate(tqdm(categories, position=0)):
            folder = osp.join(self.raw_dir, category)
            paths = glob.glob(f'{folder}/*')
            for path in paths:
                neuron = int(path.split(os.sep)[-1].split('.')[0])
                ID = neuron2ID[neuron]
                data = read_csv(path, ID, neuron)
                data.y = torch.tensor([target])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        index = torch.randperm(len(data_list)).tolist()

        # [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        train_index001 = index[:int(self.train_ratio * len(data_list) * 0.01)]
        train_index003 = index[:int(self.train_ratio * len(data_list) * 0.03)]
        train_index005 = index[:int(self.train_ratio * len(data_list) * 0.05)]
        train_index010 = index[:int(self.train_ratio * len(data_list) * 0.1)]
        train_index020 = index[:int(self.train_ratio * len(data_list) * 0.2)]
        train_index030 = index[:int(self.train_ratio * len(data_list) * 0.3)]
        train_index040 = index[:int(self.train_ratio * len(data_list) * 0.4)]
        train_index050 = index[:int(self.train_ratio * len(data_list) * 0.5)]
        train_index060 = index[:int(self.train_ratio * len(data_list) * 0.6)]
        train_index070 = index[:int(self.train_ratio * len(data_list) * 0.7)]
        train_index080 = index[:int(self.train_ratio * len(data_list) * 0.8)]
        train_index090 = index[:int(self.train_ratio * len(data_list) * 0.9)]
        train_index100 = index[:int(self.train_ratio * len(data_list) * 1)]

        test_index = index[int(self.train_ratio * len(data_list)):int(self.test_ratio * len(data_list)) + int(
            self.train_ratio * len(data_list))]
        val_index = index[int(self.test_ratio * len(data_list)) + int(self.train_ratio * len(data_list)):]

        train_list001 = [data_list[i] for i in train_index001]
        train_list003 = [data_list[i] for i in train_index003]
        train_list005 = [data_list[i] for i in train_index005]
        train_list010 = [data_list[i] for i in train_index010]
        train_list020 = [data_list[i] for i in train_index020]
        train_list030 = [data_list[i] for i in train_index030]
        train_list040 = [data_list[i] for i in train_index040]
        train_list050 = [data_list[i] for i in train_index050]
        train_list060 = [data_list[i] for i in train_index060]
        train_list070 = [data_list[i] for i in train_index070]
        train_list080 = [data_list[i] for i in train_index080]
        train_list090 = [data_list[i] for i in train_index090]
        train_list100 = [data_list[i] for i in train_index100]


        test_list = [data_list[i] for i in test_index]
        val_list = [data_list[i] for i in val_index]

        return self.collate(data_list), self.collate(train_list001), self.collate(train_list003), self.collate(train_list005),\
            self.collate(train_list010), self.collate(train_list020), self.collate(train_list030), self.collate(train_list040),\
            self.collate(train_list050), self.collate(train_list060), self.collate(train_list070), self.collate(train_list080),\
        self.collate(train_list090), self.collate(train_list100), self.collate(test_list), self.collate(val_list)


    def __repr__(self) -> str:
        # print the BrainData class info for print(BrainData)
        return f'{self.__class__.__name__}({len(self)})'


if __name__ == '__main__':
    path = 'data/manc_class'
    pre_transform, transform = T.NormalizeScale(), T.FixedPoints(1024)
    data_dataset = MancSkeleton_class(path, 'data', transform=transform, pre_transform=pre_transform)
    pass




























