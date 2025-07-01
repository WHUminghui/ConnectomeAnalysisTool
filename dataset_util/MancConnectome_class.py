import glob
import os
import os.path as osp
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.data import Data
from typing import Union, List, Tuple
import torch
import numpy as np
import sys
from tqdm import tqdm
import torch_geometric.transforms as T
def nouse():
    sys.path.append(f'{os.path.dirname(__file__)}')
    wpath = os.getcwd()
    print(f'{os.path.dirname(__file__)}')
    print(wpath)
    print(os.path.exists("../data/source_data/data_for_wangguojia/edges.txt"))


def edges_txt2list(inpath):
    with open(inpath, 'r') as f:
        data = []
        for line in f:
            data_line = line.strip('\n').split(' ')
            data.append([int(i) for i in data_line])
    return data

def get_neuronID():
    with open('/data2/liaominghui/project/HemiBrain/data/source_data/data_for_wangguojia/neuron2ID.txt', 'r') as file:
        data = {}
        for line in file:
            neuron, ID = [int(x) for x in line.strip().split(',')]
            data[neuron] = ID
    return data


def get_selected_neuronID(path):
    # path is ../../raw
    selected_neuronID = []
    neuron2ID = get_neuronID()
    categories_path = glob.glob(f'{path}/*')
    for category_path in categories_path:
        csvs_path = glob.glob(f'{category_path}/*')
        for csv_path in csvs_path:
            neuron = int(csv_path.strip().split(os.sep)[-1].split('.')[0])
            neuronID = neuron2ID[neuron]
            selected_neuronID.append(neuronID)
    selected_neuronID.sort()
    return selected_neuronID

def get_labels(Connect_raw_path):
    # path is '../../raw'
    skeleton_raw_path = '/data2/liaominghui/project/HemiBrain/data/manc_class/raw'
    labels = []
    ID2labels = {}
    classes = glob.glob(f'{skeleton_raw_path}/*')
    for clas in classes:
        class_name = clas.strip().split(os.sep)[-1]
        labels.append(class_name)
        csvs = glob.glob(f'{clas}/*csv')
        for csv in csvs:
            csv_name = csv.split('.')[0].split('/')[-1]
            ID2labels[int(csv_name)] = class_name
    return labels, ID2labels


def get_Connectome_raw(outdir, indir='/data2/liaominghui/project/HemiBrain/data/manc_class/manc_class_info'):
    if os.path.exists(osp.join(outdir, 'edge_index.npy')):
        return np.load(osp.join(outdir, 'edge_index.npy')),\
               np.load(osp.join(outdir, 'edge_attr.npy')),\
               np.load(osp.join(outdir, 'y.npy'))

    edges = edges_txt2list(osp.join(indir, 'edges.txt'))
    neuronID = edges_txt2list(osp.join(indir, 'neuron2ID.txt'))
    neuron2ID_dir = {}
    ID2neuron = []
    for i in range(len(neuronID)):
        key, value = neuronID[i][0], neuronID[i][1]
        neuron2ID_dir[key] = value
        ID2neuron.append(key)
    labels, ID2labels = get_labels(outdir)
    labels2nums = {}
    for i, type in enumerate(labels):
        labels2nums[type] = i

    temp_edges = [[], []]
    temp_edge_attr = []
    print('creteating temp_edges...')
    for edge in tqdm(edges, position=0, maxinterval=len(edges[0])):
        temp_edges[0].append(neuron2ID_dir[edge[0]])
        temp_edges[1].append(neuron2ID_dir[edge[1]])
        temp_edge_attr.append(edge[2])
    print('creteated temp_edges')

    # selected_neuronIDList = list(map(int, selected_neuronID))
    # out_edges = [[], []]
    # out_edge_attr = []
    # print('selecting edge...')
    # for i in tqdm(range(len(temp_edges[0])), position=0):
    #     if temp_edges[0][i] in selected_neuronID and temp_edges[1][i] in selected_neuronID:
    #         out_edges[0].append(selected_neuronIDList.index(temp_edges[0][i]))
    #         out_edges[1].append(selected_neuronIDList.index(temp_edges[1][i]))
    #         out_edge_attr.append(temp_edge_attr[i])
    # print('selected edge!')

    out_label = []
    for neuron in ID2neuron:
        out_label.append(labels2nums[ID2labels[neuron]])
    np.save(osp.join(outdir, 'edge_index.npy'), np.array(temp_edges))
    np.save(osp.join(outdir, 'edge_attr.npy'), np.array(temp_edge_attr))
    np.save(osp.join(outdir, 'y.npy'), np.array(out_label))

    return np.array(temp_edges), np.array(temp_edge_attr), np.array(out_label)


class MancConnectome_class(InMemoryDataset):
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
        edges_index, edges_attr, y = get_Connectome_raw(outdir=self.raw_dir)
        index = torch.randperm(len(y)).tolist()
        train_index = index[:len(y)//10*8]
        test_index = index[len(y)//10*8:len(y)//10*9]
        val_index = index[len(y)//10*9:]
        # train_index = torch.arange(len(y)//10*8, dtype=torch.long)
        # test_index = torch.arange(len(y)//10*8, len(y)//10*9, dtype=torch.long)
        # val_index = torch.arange(len(y) // 10 * 9, len(y), dtype=torch.long)
        train_mask = torch.zeros((len(y), ), dtype=torch.bool)
        val_mask = torch.zeros((len(y),), dtype=torch.bool)
        test_mask = torch.zeros((len(y), ), dtype=torch.bool)

        len_raw_edge = len(edges_attr)
        index = torch.randperm(len_raw_edge).tolist()
        remian_edge_index = index[:int(len_raw_edge * self.edge_remain_rate)]
        edge_mask = torch.zeros(len_raw_edge, dtype=torch.bool)
        edge_mask[remian_edge_index] = True
        edges_index = torch.tensor(edges_index)
        edges_attr = torch.tensor(edges_attr)
        edges_index = edges_index[:, edge_mask]
        edges_attr = edges_attr[edge_mask]

        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        data = Data(edge_index=edges_index, edge_attr=edges_attr, y=torch.tensor(y))
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        self.selected_ID = [x for x in range(len(y))]
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'



if __name__ == '__main__':
    path_Skeleton = f'data/manc_class_Connectome'
    transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
    dataset = MancConnectome_class(path_Skeleton, transform=transform)

