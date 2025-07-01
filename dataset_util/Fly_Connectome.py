import sys
sys.path.append("..") 
import glob
import os
import os.path as osp
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.data import Data
from typing import Union, List, Tuple
import torch
import numpy as np
from tqdm import tqdm
import torch_geometric.transforms as T
import pandas as pd


def edges_csv2list():
    df = pd.read_csv('data/Fly_source/ol_connections.csv')
    source_neuron = df.iloc[:, 0].tolist()
    target_neuron = df.iloc[:, 1].tolist()
    attr = df.iloc[:, 2].tolist()
    return [source_neuron, target_neuron, attr]

def get_neuon2IDAndLable():
    lables = []
    neuon2ID = {}
    ID2neuron = {}
    no_assiged_neuronID = []
    df = pd.read_csv('data/Fly_source/ol_columns.csv')
    for i in range(df.shape[0]):
        neuron = df.iloc[i, 0]
        lable = df.iloc[i, 2]
        if lable == 'not assigned':
            no_assiged_neuronID.append(i)
            lable = -1
        else:
            lable = int(lable)
        lables.append(lable)
        neuon2ID[neuron] = i
        ID2neuron[i] = neuron
    return lables, neuon2ID, ID2neuron, no_assiged_neuronID

def get_labels(Connect_raw_path):
    # path is '../../raw'
    density = Connect_raw_path.split(os.sep)[-2].split('e')[-1]
    skeleton_raw_path = f'/data/users/minghuiliao/project/Dor/HemiBrain/data/source_data/coarsen_skeleton_more{density}/raw'
    labels = []
    ID2labels = {}
    types = glob.glob(f'{skeleton_raw_path}/*')
    for type in types:
        type_name = type.strip().split(os.sep)[-1]
        labels.append(type_name)
        csvs = glob.glob(f'{type}/*csv')
        for csv in csvs:
            csv_name = csv.split('.')[0].split('/')[-1]
            ID2labels[int(csv_name)] = type_name
    return labels, ID2labels

def get_Connectome_raw(outdir):
    lables, neuon2ID, ID2neuron, no_assiged_neuronID = get_neuon2IDAndLable()
    if os.path.exists(osp.join(outdir, 'edge_index.npy')):
        return np.load(osp.join(outdir, 'edge_index.npy')),\
               np.load(osp.join(outdir, 'edge_attr.npy')),\
               np.load(osp.join(outdir, 'y.npy')), ID2neuron, no_assiged_neuronID
    edges = edges_csv2list()
    out_edges = [[], []]
    out_edge_attr = []
    print('selecting edge...')
    for i in tqdm(range(len(edges[0])), position=0):
        if edges[0][i] in neuon2ID and edges[1][i] in neuon2ID:
            sourceID = neuon2ID[edges[0][i]]
            targetID = neuon2ID[edges[1][i]]
        else:
            continue
        out_edges[0].append(sourceID)
        out_edges[1].append(targetID)
        out_edge_attr.append(edges[2][i])
    print('selected edge!')
    np.save(osp.join(outdir, 'edge_index.npy'), np.array(out_edges))
    np.save(osp.join(outdir, 'edge_attr.npy'), np.array(out_edge_attr))
    np.save(osp.join(outdir, 'y.npy'), np.array(lables))
    return np.array(out_edges), np.array(out_edge_attr), np.array(lables), ID2neuron, no_assiged_neuronID

class FlyConnectome(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, edge_remain_rate=1):
        self.edge_remain_rate = edge_remain_rate
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        assert edge_remain_rate <= 1 and edge_remain_rate > 0

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ['edge_index.npy', 'edge_attr,npy', 'y.npy']

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        torch.save(self.collate([self.process_set()]), self.processed_paths[0])

    def process_set(self):
        edges_index, edges_attr, y, ID2neuron, no_assiged_neuronID = get_Connectome_raw(outdir=self.raw_dir)
        train_mask = torch.ones((len(y), ), dtype=torch.bool)
        test_mask = torch.zeros((len(y), ), dtype=torch.bool)
        train_mask[no_assiged_neuronID] = False
        test_mask[no_assiged_neuronID] = True
        edges_index = torch.tensor(edges_index)
        edges_attr = torch.tensor(edges_attr)
        data = Data(edge_index=edges_index, edge_attr=edges_attr, y=torch.tensor(y))
        data.train_mask = train_mask
        data.test_mask = test_mask
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'



if __name__ == '__main__':
    path_Connectome = f'data/FlyConnectome'
    transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
    dataset = FlyConnectome(path_Connectome, transform=transform)
    pass

