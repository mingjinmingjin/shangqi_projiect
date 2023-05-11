# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser
import pytorch_lightning as pl
import os
import torch
from torch.utils.data import DataLoader, random_split
from datasets.shangqi_dataset import ShangqiDataset, collate_fn
from models.hivt import HiVT
import sys
import tqdm
from PR_curve import compute_map

sys.path.append("../HiVT2")
if __name__ == '__main__':
    torch.manual_seed(1234)
    parser = ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=5)
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()
    dataset = ShangqiDataset("/ssd/share/shangqi_data/last_dir_path.txt")
    device = torch.device("cpu")
    num_workers = args.num_workers
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    num_class = 36
    # 定义训练集、验证集和测试集的比例
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # 计算每个数据集的大小
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # 使用 random_split 函数划分数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 定义 DataLoader
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                             num_workers=num_workers)

    model = HiVT(**vars(args))
    model = model.to(device)
    # 定义损失函数和优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    test_set = []
    # 初始化最小loss值为无穷大
    min_loss = float("inf")

    home_path = os.path.expanduser("~/shangqi_train")
    print(home_path)

    # 开始训练模型
    for epoch in range(args.max_epochs):
        # 在每个 epoch 开始前，将模型设置为训练模式
        model.train()
        giou_loss_train = []
        for batch_idx, (data, target) in enumerate(tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            target = target.to(device)
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()
                output = model(data)
                giou_loss, obj_loss, cls_loss = model.compute_loss(output, target)
                loss = giou_loss + obj_loss * 0.3 + cls_loss * 0.05
                giou_loss.backward(retain_graph=True)
                giou_loss_train.append(giou_loss.item())
                optimizer.step()
                if batch_idx % 3 == 0:
                    print('Epoch: {}, Batch: {}, Loss: {} , Giou Loss: {}， obj Loss: {} , cls Loss:{}'.format(epoch,
                                                                                                              batch_idx,
                                                                                                              loss.item(),
                                                                                                              giou_loss.item(),
                                                                                                              obj_loss.item(),
                                                                                                              cls_loss.item()))
                    with open(home_path+"/train_loss_per_batch.txt", "a+") as f:
                        f.write(
                            f'Epoch: {epoch}, Batch: {batch_idx}, giou Loss: {giou_loss.item()} , obj Loss: {obj_loss.item()} ,cls Loss:{cls_loss.item() * 0.1}\n')
        #每个epoch后，保存giou_loss最小的模型参数                
        if sum(giou_loss_train) / len(giou_loss_train) < loss:
            min_loss = sum(giou_loss_train) / len(giou_loss_train)
            torch.save(model.state_dict(), home_path+"/best_model_params.pth")
        print('Epoch: {}, avg Giou Loss: {}'.format(epoch, sum(giou_loss_train) / len(giou_loss_train)))
        with open(home_path+"/giou_loss_per_epoch.txt", "a+") as f:
            f.write(f'Epoch: {epoch}, giou Loss: {sum(giou_loss_train) / len(giou_loss_train)} \n')

        # 在每个 epoch 结束后，评估模型在测试集上的表现
        # model.eval()
        # test_loss = 0
        # correct = 0
        # with torch.no_grad():
        #     for data, target in test_loader:
        #         output = model(data)
        #        //测试模型
        #         test_loss += criterion(output, target).item()
        #         pred = output.argmax(dim=1, keepdim=True)
        #         correct += pred.eq(target.view_as(pred)).sum().item()
        #
        # test_loss /= len(test_loader.dataset)
        #
        # print('Epoch: {}, Test Loss: {}, Test Accuracy: {}%'.format(
        #     epoch, test_loss, 100. * correct / len(test_loader.dataset)))
