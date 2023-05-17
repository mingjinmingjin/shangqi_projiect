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
import torch.nn as nn

sys.path.append("../shangqi_project")
if __name__ == '__main__':
    torch.manual_seed(1234)
    parser = ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=5)
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()
    dataset = ShangqiDataset("/ssd/share/shangqi_data/one_obj.txt")
    device = torch.device('cpu')

    num_workers = args.num_workers
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    num_class = 36
    # 定义训练集、验证集和测试集的比例
    train_ratio = 0.8
    # val_ratio = 0.1
    test_ratio = 0.2

    # 计算每个数据集的大小
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # 使用 random_split 函数划分数据集
    train_dataset,  test_dataset = random_split(dataset, [train_size, test_size])

    # 定义 DataLoader
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                             num_workers=num_workers)

    model = HiVT(**vars(args))
    # if torch.cuda.device_count() > 1:
    #     print("使用", torch.cuda.device_count(), "个GPU进行训练。")
    # model = nn.DataParallel(model,device_ids=[0,3,4])
    model = model.to(device)
    # 定义损失函数和优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    test_set = []
    # 初始化最小loss值为无穷大
    min_reg_loss = float("inf")
    max_pre_acc=float("-inf")
    home_path = os.path.expanduser("~/train_log")
    print(home_path)

    # 开始训练模型
    for epoch in range(args.max_epochs):
        # 在每个 epoch 开始前，将模型设置为训练模式
        model.train()
        reg_loss_train = []
        cls_loss_train = []
        for batch_idx, (datas, target) in enumerate(tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            target = target.to(device)
            for data in datas:
                for name, tensor in vars(data).items():
                    if isinstance(tensor, torch.Tensor):
                        setattr(data, name, tensor.to(device))
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()
                output = model(datas)
                reg_loss, cls_loss = model.compute_loss(output, target)
                loss = reg_loss + cls_loss
                loss.backward(retain_graph=True)
                reg_loss_train.append(reg_loss.item())
                cls_loss_train.append(cls_loss.item())
                optimizer.step()
                if batch_idx % 3 == 0:
                    print('Epoch: {}, Batch: {}, Loss: {} , reg Loss: {} , cls Loss:{}'.format(epoch,
                                                                                                batch_idx,
                                                                                                loss.item(),
                                                                                                reg_loss.item(),
                                                                                                cls_loss.item()))
                    with open(home_path + "/train_loss_per_batch.txt", "a+") as f:
                        f.write(
                            f'Epoch: {epoch}, Batch: {batch_idx}, reg Loss: {reg_loss.item()}  ,cls Loss:{cls_loss.item()}\n')

        print('Epoch: {}, avg reg Loss: {}'.format(epoch, sum(reg_loss_train) / len(reg_loss_train)))
        with open(home_path + "/reg_loss_per_epoch.txt", "a+") as f:
            f.write(f'Epoch: {epoch}, giou Loss: {sum(reg_loss_train) / len(reg_loss_train)} \n')
        print('Epoch: {}, avg cls Loss: {}'.format(epoch, sum(cls_loss_train) / len(cls_loss_train)))
        with open(home_path + "/cls_loss_per_epoch.txt", "a+") as f:
            f.write(f'Epoch: {epoch}, cls Loss: {sum(cls_loss_train) / len(cls_loss_train)} \n')

        # 在每个 epoch 结束后，评估模型在测试集上的表现
        model.eval()
        test_loss = 0.
        correct = 0.
        with torch.no_grad():
            for datas, target in test_loader:
                for data in datas:
                    for name, tensor in vars(data).items():
                        if isinstance(tensor, torch.Tensor):
                            setattr(data, name, tensor.to(device))
                output = model(datas)
                # //测试模型
                reg_loss, correct_pre = model.test_model(output, target)
                test_loss += reg_loss
                correct += correct_pre

        avg_test_loss = test_loss / len(test_dataset)
        avg_correct_pre = correct / len(test_dataset)
        if avg_test_loss < min_reg_loss:
            min_reg_loss =avg_test_loss
            torch.save(model.state_dict(), home_path + "/best_model_reg_params_epoch:{}.pth".format(epoch))
        if avg_correct_pre > max_pre_acc:
            max_pre_acc =avg_correct_pre
            torch.save(model.state_dict(), home_path + "/best_model_cls_params_epoch:{}.pth".format(epoch))
        print('Epoch: {}, reg Loss: {}, cls Accuracy: {}%'.format(
            epoch, avg_test_loss, 100. * avg_correct_pre))
        with open(home_path + "/test_loss_per_epoch.txt", "a+") as f:
            f.write(f'Epoch: {epoch}, reg Loss: {avg_test_loss} , cls acc: {avg_correct_pre} \n')

