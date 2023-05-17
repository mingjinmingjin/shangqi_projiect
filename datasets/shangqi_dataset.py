import os
import json
from torch.utils.data import Dataset
import numpy as np
import torch
from datasets.argoverse_v1_dataset import process_argoverse2
import sys
# sys.path.append("/ssd/taomingjin/venv/HiVT2")
num={}
class ShangqiDataset(Dataset):
    def __init__(self, list_path):
        with open(list_path,'r') as f:
            self.list_path=f.readlines()

    def __getitem__(self, index):
        return torch.load(self.list_path[index].strip())

    def __len__(self):
        return len(self.list_path)


def collate_fn(batch):
    label=[]
    for i,data in enumerate(batch):
        batch[i].y[:,0]=i
        label.append(batch[i].y)
    return batch,torch.cat(label,0)

# if __name__ == '__main__':
#     shangqidata = ShangqiDataset("/ssd/share/shangqi_data/last_dir_path.txt")
#     model=torch.load("/ssd/share/shangqi_data/PL069/20221101/20221101-101739hedian.pt")
#     for i in range(len(shangqidata)):
#         data=shangqidata[i]
#     res={}


