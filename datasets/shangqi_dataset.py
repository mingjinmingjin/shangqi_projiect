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
        with open(list_path, "r") as file:
            self.dir_path = file.readlines()

    def __getitem__(self, index):
        dir_path = self.dir_path[index].strip("\n")
        ego_path = os.path.join(dir_path, 'ego.csv')
        obj_path = os.path.join(dir_path, 'obj.csv')
        # temporal_data = process_argoverse2(dir_path,[ego_path, obj_path])
        # torch.save(temporal_data,dir_path+".pt")
        bagInfo_path = os.path.join(dir_path, 'bagInfo.json')  # 先打开bagInfo文件，提取出开始时间，持续时间
        with open(bagInfo_path) as f:
            json_data=json.load(f)
            start=json_data['start']
            duration1=json_data['duration']
        with open('/ssd/taomingjin/venv/HiVT2/scene_tag.json') as json_file:
            scene_tag = json.load(json_file)
        scene_data=[]
        for file_name in os.listdir(dir_path):  # 遍历文件夹下的所有文件
            if file_name.endswith('scenario_records.json'):
                file_path = os.path.join(dir_path, file_name)  # 获取文件的完整路径
                with open(file_path, 'r') as f:  # 打开文件
                    json_data = json.load(f)  # 解析JSON数据
                    scene=json_data['scenario'][0]['scene_tag']
                    scene_start=max(json_data['scenario'][0]['t0']-start,0)
                    scene_end=json_data['scenario'][0]['tN']-start

                    scene_data.append([0,scene_tag[scene],scene_start,scene_end])
                    if scene_end<0:
                        return None,None
        scene_data=np.stack(scene_data,axis=0)
        return torch.load(dir_path+".pt"),scene_data


    def __len__(self):
        return len(self.dir_path)



def collate_fn(batch):
    batch= [data for data in batch if data[0] is not None]
    data,label=list(zip(*batch))
    for i, box in enumerate(label):
        box[:, 0] = i
    boxes=np.concatenate(label,axis=0)
    target_box=torch.from_numpy(boxes)
    return data,target_box

# if __name__ == '__main__':
#     shangqidata = ShangqiDataset("/ssd/share/shangqi_data/last_dir_path.txt")
#     model=torch.load("/ssd/share/shangqi_data/PL069/20221101/20221101-101739hedian.pt")
#     for i in range(len(shangqidata)):
#         data=shangqidata[i]
#     res={}


