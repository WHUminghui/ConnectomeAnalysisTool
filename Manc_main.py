import numpy as np
import torch
import torch.nn.functional as F
import os
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from model import GCNII_Concat, BrainConnetome_cascade_X
from dataset_util.MancConnectome_class import MancConnectome_class
from dataset_util.MancSkeleton_class import MancSkeleton_class
import argparse
from focal_loss import FocalLossV2
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import os.path as osp
import random
import glob
# import wandb
# wandb.init(project='HemiBrain(BrainConnetome_concat_X)')

parser = argparse.ArgumentParser(
    description='train',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#### about model ####
parser.add_argument('--name', type=str, default='deng')

#### about train ####
parser.add_argument('--deviceID', type=int, default=2)
parser.add_argument('--train_patient', type=int, default=15)
parser.add_argument('--train_gcn_patient', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--train_lr', type=int, default=0.01)
parser.add_argument('--GCNII_Graph_weight_decay', type=float, default=0.01)
parser.add_argument('--GCNII_linear_weight_decay', type=float, default=5e-4)

args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_gcn(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    hidden_x, out = model(data.adj_t)
    out = out.log_softmax(dim=-1)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return hidden_x, float(loss)

@torch.no_grad()
def test_gcn(model, data):
    model.eval()
    _, pred = model(data.adj_t)
    pred = pred.log_softmax(dim=-1)
    pred = pred.argmax(dim=-1)
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

def train(model, hidden_x, train_loader, selected_ID, device, optimizer):
    model.train()
    loss_sum = 0
    corrects = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        _, ypre = model(hidden_x, data, selected_ID)
        loss = F.nll_loss(ypre, data.y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        with torch.no_grad():
            pred = ypre.max(1)[1]
        correct = pred.eq(data.y).sum().item()
        corrects += correct
    train_acc = corrects / len(train_loader.dataset)
    # wandb.log({'train_loss': loss_sum/i, 'train_acc': train_acc})
    return loss_sum/i, train_acc

def test(model, hidden_x, test_loader, selected_ID, device):
    model.eval()
    corrects = 0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            _, pred = model(hidden_x, data, selected_ID)
            pred = pred.max(1)[1]
        correct = pred.eq(data.y).sum().item()
        corrects += correct
    test_acc = corrects / len(test_loader.dataset)
    # wandb.log({'test_acc': test_acc})
    return test_acc

def get_accMat(model, hidden_x, test_loader, selected_ID, device, dataset_num_classes):
    model.eval()
    accMat = [[0] * dataset_num_classes] * dataset_num_classes
    accMat = np.array(accMat)
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            _, pred = model(hidden_x, data, selected_ID)
        pred = pred.max(1)[1]
        pred = pred.tolist()
        ture = data.y.tolist()
        for i in range(len(data.y)):
            accMat[pred[i]][ture[i]] += 1
    # wandb.log({'test_acc': test_acc})
    return accMat

def get_hiddenFea(model, hidden_x, data_loader, selected_ID, device, dataset_num_classes):
    model.eval()
    dataIDs = data_loader.dataset.data.ID.tolist()
    hiddenFeas = [0] * len(dataIDs)
    for data in data_loader:
        data = data.to(device)
        dataID = data.ID.tolist()
        with torch.no_grad():
            hiddenFea, _ = model(hidden_x, data, selected_ID)
        for i, id in enumerate(dataID):
            hiddenFeas[dataIDs.index(id)] = hiddenFea[i].tolist()
    # wandb.log({'test_acc': test_acc})
    return hiddenFeas

def main(seed, data):
    setup_seed(seed)
    device = torch.device(f'cuda:{args.deviceID}' if torch.cuda.is_available() else 'cpu')
    if data == 'Manc':
        path_Skeletome = 'data/manc_class'
        path_Connectome = 'data/manc_class_Connectome'
        pre_transform, transform = T.NormalizeScale(), T.FixedPoints(1024)
        data_dataset = MancSkeleton_class(path_Skeletome, 'data', transform=transform, pre_transform=pre_transform)
        train_dataset = MancSkeleton_class(path_Skeletome, 'train100', transform=transform, pre_transform=pre_transform)
        test_dataset = MancSkeleton_class(path_Skeletome, 'test', transform=transform, pre_transform=pre_transform)
        data_loader = DataLoader(data_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6)
        transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
        dataset_Connectome = MancConnectome_class(path_Connectome, transform=transform)
        categories = glob.glob(f'{data_dataset.raw_dir}/*')
        categories = sorted([x.split(os.sep)[-1] for x in categories])

    data_Connectome = dataset_Connectome[0]
    data_Connectome = data_Connectome.to(device)
    selected_ID = [x for x in range(len(data_Connectome.y))]
    data_Connectome.adj_t = gcn_norm(data_Connectome.adj_t)
    num_nodes, dataset_num_classes = len(dataset_Connectome.data.y), data_dataset.num_classes
    
    model_gnn = GCNII_Concat(num_nodes, dataset_num_features=1024, num_calss=dataset_num_classes, represent_features=512,\
                             hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5, shared_weights=True, dropout=0).to(device)
    optimizer_gcn = torch.optim.Adam([
        dict(params=model_gnn.convs.parameters(), weight_decay=args.GCNII_Graph_weight_decay),
        dict(params=model_gnn.lins.parameters(), weight_decay=args.GCNII_linear_weight_decay),
        dict(params=model_gnn.mlp.parameters(), weight_decay=args.GCNII_linear_weight_decay)
    ], lr=args.train_lr)

    model = BrainConnetome_cascade_X(dataset_num_classes, dataset_num_features=1024, represent_features=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    max_test_acc = 0
    epoch = 0
    bad = 0
    print('GCNII is pre_trainning...')
    while True:
        # if epoch==5: # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #     break
        if bad > args.train_gcn_patient:
            break
        epoch += 1
        bad += 1
        hidden_x, loss = train_gcn(model_gnn, data_Connectome, optimizer_gcn)
        train_acc, _, tmp_test_acc = test_gcn(model_gnn, data_Connectome)
        if tmp_test_acc > max_test_acc:
            bad = 0
            max_test_acc = tmp_test_acc
            print(f'Epoch: {epoch:03d}, loss: {loss:.6f}, Train: {train_acc:.4f}, Test: {tmp_test_acc:.4f}')
            continue
        if bad % (args.train_gcn_patient//5) == 0:
            print(f'Epoch: {epoch:03d}')

    max_test_acc = 0
    epoch = 0
    bad = 0
    hidden_x = torch.from_numpy(hidden_x.cpu().detach().numpy()).to(device)
    print('main train is working...')
    while True:
        # if epoch==2: # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #     break
        if bad > args.train_patient: #args.train_patient:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            break
        bad += 1
        epoch += 1
        loss, train_acc_end = train(model, hidden_x, train_loader, selected_ID, device, optimizer)
        test_acc = test(model, hidden_x, test_loader, selected_ID, device)
        if test_acc > max_test_acc:
            bad = 0
            max_test_acc = test_acc
            print(f'Epoch: {epoch:03d}, loss: {loss:.6f}, Train: {train_acc_end:.4f}, Test: {test_acc:.4f}')
            accMat = get_accMat(model, hidden_x, test_loader, selected_ID, device, dataset_num_classes)
            hiddenFeas = get_hiddenFea(model, hidden_x, data_loader, selected_ID, device, dataset_num_classes)
            continue
        if bad % (args.train_patient//5) == 0:
            print(f'Epoch: {epoch:03d}')

    dataID, dataY = data_dataset.data.ID.tolist(), data_dataset.data.y.tolist()
    return accMat, hiddenFeas, max_test_acc, dataID, dataY


if __name__ == '__main__':
    # 500
    if not os.path.exists(f'data/experiment_result/cascadeX'):
        os.makedirs(f'data/experiment_result/cascadeX')
    data = 'Manc'
    for seed in [333, 666, 888, 999, 1024]:
            print(f'##################   {data}: {seed} is working...     ##################')
            accMat, hiddenFeas, max_test_acc, dataID, dataY = main(seed, data)
            hiddenFeas = np.array(hiddenFeas)
            dataID = np.array(dataID)
            dataY = np.array(dataY)
            np.save(f'data/experiment_result/cascadeX/accMat_{data}_{seed}.npy', accMat)
            np.save(f'data/experiment_result/cascadeX/hiddenFeas_{data}_{seed}.npy', hiddenFeas)
            np.save(f'data/experiment_result/cascadeX/dataID_{data}_{seed}.npy', dataID)
            np.save(f'data/experiment_result/cascadeX/dataY_{data}_{seed}.npy', dataY)
            with open(f'data/experiment_result/cascadeX/max_test_acc{data}_{seed}.txt', 'w') as f:
                f.write(f'max_test_acc: {max_test_acc:.4f}')











