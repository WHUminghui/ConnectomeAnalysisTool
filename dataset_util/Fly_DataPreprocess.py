import os, sys
import shutil
import glob
import pandas as pd
from tqdm import tqdm
import csv
import numpy as np


def extract_xyz_from_swc(swc_file_path, csv_file_path):
    if not os.path.exists(swc_file_path):
        return
    xyz_data = []
    with open(swc_file_path, 'r') as swc_file:
        for line in swc_file:
            line = line.strip()
            if not line.startswith('#') and line:
                parts = line.split()
                if len(parts) >= 4:
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    xyz_data.append((x, y, z))

    # 写入CSV文件
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['x', 'y', 'z'])  # 写入行标题
        writer.writerows(xyz_data)  # 写入XYZ数据

def get_neuonID2lable():
    lables = {}
    df = pd.read_csv('data/Fly_source/ol_columns.csv')
    for i in range(df.shape[0]):
        neuronID = df.iloc[i, 0]
        lable = df.iloc[i, 2]
        if lable in lables:
            lables[lable].append(neuronID)
        else:
            lables[lable] = [neuronID]
    return lables

def get_Skeleon_raw():
    skeleton_path = 'data/Fly_source/skeleton'
    raw_path = 'data/FlySkeleton/raw'
    lable2neuronIDs = get_neuonID2lable()
    for lable, neuronIDs in tqdm(lable2neuronIDs.items(), position=1):
        for neuronID in neuronIDs:
            if not os.path.exists(f'{raw_path}/{lable}'):
                os.makedirs(f'{raw_path}/{lable}')
            extract_xyz_from_swc(f'{skeleton_path}/{neuronID}.swc', f'{raw_path}/{lable}/{neuronID}.csv')




edge_attr = np.load('data/HemiBrain_connectome/raw/edge_attr.npy')
edge_index = np.load('data/HemiBrain_connectome/raw/edge_index.npy')
y = np.load('data/HemiBrain_connectome/raw/y.npy')

pass






