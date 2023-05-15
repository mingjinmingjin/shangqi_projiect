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
import math
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# from losses import LaplaceNLLLoss
# from losses import SoftTargetCrossEntropyLoss
from models import GlobalInteractor
from models import LocalEncoder
from models import MLPDecoder
from utils import TemporalData
from losses.Focal_CE_Loss import CrossEntropyFocalLoss
from losses.Focal_BCE_Loss import BCEFocalLosswithLogits


class HiVT(pl.LightningModule):

    def __init__(self,
                 historical_steps: int,
                 future_steps: int,
                 num_modes: int,
                 rotate: bool,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 num_temporal_layers: int,
                 num_global_layers: int,
                 local_radius: float,
                 parallel: bool,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 **kwargs) -> None:
        super(HiVT, self).__init__()
        self.save_hyperparameters()
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.anchor_num = 2
        self.dim = 40
        self.second = [1, 3.5, 7]
        self.mlp = {}
        self.device1 = torch.device("cpu")

        for i in self.second:
            self.mlp[i] = nn.Linear(int(64 * i * 10), 80)
            self.mlp[i].weight = torch.nn.Parameter(self.mlp[i].weight.to(self.device1))
            self.mlp[i].bias = torch.nn.Parameter(self.mlp[i].bias.to(self.device1))

        self.local_encoder = LocalEncoder(historical_steps=historical_steps,
                                          node_dim=node_dim,
                                          edge_dim=edge_dim,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          num_temporal_layers=num_temporal_layers,
                                          local_radius=local_radius,
                                          parallel=parallel)

        self.global_interactor = GlobalInteractor(historical_steps=historical_steps,
                                                  embed_dim=embed_dim,
                                                  edge_dim=edge_dim,
                                                  num_modes=num_modes,
                                                  num_heads=num_heads,
                                                  num_layers=num_global_layers,
                                                  dropout=dropout,
                                                  rotate=rotate)
        self.decoder = MLPDecoder(local_channels=embed_dim,
                                  global_channels=embed_dim,
                                  future_steps=future_steps,
                                  num_modes=num_modes,
                                  uncertain=True)

        self.obj_loss = BCEFocalLosswithLogits()
        self.cls_loss = CrossEntropyFocalLoss()

    def forward(self, data: TemporalData):
        embed = []
        # for data in temporaldata:
        if self.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data['rotate_angles'])
            cos_vals = torch.cos(data['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            # if data.y is not None:
            #     data.y = torch.bmm(data.y, rotate_mat)
            data['rotate_mat'] = rotate_mat
        else:
            data['rotate_mat'] = None

        local_embed = self.local_encoder(data=data)
        embed.append(local_embed[:, 0, :])
        predict = torch.stack(embed, dim=0)

        return predict,data.y

    def training_step(self, batch, batch_idx):
        pre ,target= self(batch)
        # reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        # valid_steps = reg_mask.sum(dim=-1)
        # cls_mask = valid_steps > 0
        # l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        # best_mode = l2_norm.argmin(dim=0)
        # y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        # reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        # soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()
        # cls_loss = self.cls_loss(pi[cls_mask], soft_target)
        # loss = reg_loss + cls_loss
        # self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        giou_loss, obj_loss, cls_loss = self.compute_loss(pre, target)
        loss = giou_loss + obj_loss * 0.3 + cls_loss * 0.05
        self.log('giou_loss', giou_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        pre, target = self(batch)
        # reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        # valid_steps = reg_mask.sum(dim=-1)
        # cls_mask = valid_steps > 0
        # l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        # best_mode = l2_norm.argmin(dim=0)
        # y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        # reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        # soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()
        # cls_loss = self.cls_loss(pi[cls_mask], soft_target)
        # loss = reg_loss + cls_loss
        # self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        giou_loss, obj_loss, cls_loss = self.compute_loss(pre, target)
        loss = giou_loss + obj_loss * 0.3 + cls_loss * 0.05
        self.log('giou_loss', giou_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)

    # 计算损失
    def compute_loss(self, pre, true, ):
        '''
        pre:预测框 350帧*64维
        true:真实框
        '''
        predict = {}
        # self.second: 1秒,3.5秒，7秒
        for sec in self.second:
            predict_bbox, predict_obj, predict_cls = self.get_predict_bbox_label(pre, sec)
            predict[sec] = [predict_bbox, predict_cls, predict_obj]
        reg_box = torch.cat((predict[self.second[0]][0], predict[self.second[1]][0], predict[self.second[2]][0]), dim=1)
        cls = torch.cat((predict[self.second[0]][1], predict[self.second[1]][1], predict[self.second[2]][1]), dim=1)
        obj = torch.cat((predict[self.second[0]][2], predict[self.second[1]][2], predict[self.second[2]][2]), dim=1)
        loss = self.reg_cls_obj_loss(reg_box, true, cls, obj)
        return loss

    # 测试代码
    def test(self, pre, ):
        predict = {}
        for sec in self.second:
            predict_bbox, predict_obj, predict_cls = self.get_predict_bbox_label(pre, sec)
            predict[sec] = [predict_bbox, predict_cls, predict_obj]
        reg_box = torch.cat((predict[self.second[0]][0], predict[self.second[1]][0], predict[self.second[2]][0]), dim=1)
        cls = torch.cat((predict[self.second[0]][1], predict[self.second[1]][1], predict[self.second[2]][1]), dim=1)
        obj = torch.cat((predict[self.second[0]][2], predict[self.second[1]][2], predict[self.second[2]][2]), dim=1)
        reg_box = reg_box.flatten(1, 2)
        cls = cls.flatten(1, 2)
        obj = obj.flatten(1, 2)
        anchor_info = torch.cat((reg_box, obj, cls), dim=-1)
        anchor_predict, keep = self.one_dimensional_nms(anchor_info)
        return anchor_predict, keep

    # 预测框维度：B*N*2*2，真实框维度：B*T*2
    def reg_cls_obj_loss(self, predicted_boxes, target_boxes, predict_cls, predict_obj):
        max_len, max_index = 0, 0
        padding_target = []
        for i in range(len(predicted_boxes)):
            if len(target_boxes[target_boxes[:, 0] == i]) > max_len:
                max_index = i
                max_len = len(target_boxes[target_boxes[:, 0] == i])
            padding_target.append(target_boxes[target_boxes[:, 0] == i])
        for i in range(len(padding_target)):
            if len(padding_target[i]) < max_len:
                padding_target_concat = torch.zeros((max_len - len(padding_target[i]), 4)).to(predicted_boxes.device)
                padding_target[i] = torch.cat((padding_target[i], padding_target_concat), dim=0)
        target_boxes = torch.stack(padding_target, dim=0)
        predicted_boxes = predicted_boxes.flatten(1, 2)
        batch_size, num_anchors, _ = predicted_boxes.shape
        _, num_gt_boxes, _ = target_boxes.shape

        # 计算所有预测框和所有真实框之间的IoU
        ious = torch.zeros((batch_size, num_anchors, num_gt_boxes), dtype=torch.float32).to(predicted_boxes.device)
        for b in range(batch_size):
            for a in range(num_anchors):
                anchor_start = predicted_boxes[b, a, 0]
                anchor_end = predicted_boxes[b, a, 1]
                anchor_duration = anchor_end - anchor_start
                for t in range(num_gt_boxes):
                    gt_start = target_boxes[b, t, 2]
                    gt_end = target_boxes[b, t, 3]
                    gt_duration = gt_end - gt_start
                    intersection_start = torch.max(anchor_start, gt_start)
                    intersection_end = torch.min(anchor_end, gt_end)
                    intersection_duration = torch.max(intersection_end - intersection_start, torch.tensor(0.0))
                    union_duration = anchor_duration + gt_duration - intersection_duration
                    ious[b, a, t] = intersection_duration / union_duration

        # 对于每个预测框，找到与其IoU最大的真实框
        best_gt_ious, best_gt_inds = torch.max(ious, dim=2, keepdim=True)

        # 对于每个真实框，找到与其IoU最大的预测框
        best_pred_ious, best_pred_inds = torch.max(ious, dim=1, keepdim=True)

        # 使用掩码将没有真实框的样本的回归损失设置为0(划分正负样本)
        mask = (best_gt_ious > 0.6).float()

        # 计算回归损失
        # 计算Giou Loss
        obj_boxes = torch.gather(target_boxes, 1, best_gt_inds.expand(-1, -1, 4))
        giou_total_loss = self.giou_loss(predicted_boxes, obj_boxes[:, :, -2:], best_gt_ious)
        giou_actual_loss = torch.mean(giou_total_loss[mask.squeeze(-1) > 0])  # 只对正样本进行计算
        # 计算Smooth L1 Loss
        # diff = predicted_boxes[..., 0, :] - target_boxes[..., 0]
        # diff = predicted_boxes - obj_boxes
        # loss = F.smooth_l1_loss(diff * mask.unsqueeze(-1), torch.zeros_like(diff))
        #
        # diff = predicted_boxes[..., 1, :] - target_boxes[..., 1]
        # loss += F.smooth_l1_loss(diff * mask.unsqueeze(-1), torch.zeros_like(diff))

        # 计算正负样本损失 Focal Loss
        predict_obj = predict_obj.flatten(1, 2)
        target_obj = mask
        obj_loss = self.obj_loss(predict_obj, target_obj)

        # 计算分类损失 Focal Loss
        mask = mask.flatten()
        predict_cls = predict_cls.flatten(0, 2)[mask == 1]
        true_cls = obj_boxes[..., 1].reshape(-1, 1)[mask == 1].long()
        cls_loss = self.cls_loss(predict_cls, true_cls)

        return giou_actual_loss, obj_loss, cls_loss

    def giou_loss(self, pred_boxes, target_boxes, best_gt_ious):
        '''
        真实框: [B, N, 2]
        目标框: [B, N, 2]
        '''
        # 交集
        x1 = torch.max(pred_boxes[:, :, 0], target_boxes[:, :, 0])

        x2 = torch.min(pred_boxes[:, :, 1], target_boxes[:, :, 1])
        inter = torch.clamp(x2 - x1, min=0)

        # IOU
        pred_area = pred_boxes[:, :, 1] - pred_boxes[:, :, 0]
        target_area = target_boxes[:, :, 1] - target_boxes[:, :, 0]
        union = pred_area + target_area - inter

        # 计算 iou
        iou = inter / union

        # 计算凸包位置
        enclose_x1 = torch.min(pred_boxes[:, :, 0], target_boxes[:, :, 0])
        enclose_x2 = torch.max(pred_boxes[:, :, 1], target_boxes[:, :, 1])

        # 计算凸包长度
        enclose_area = enclose_x2 - enclose_x1

        # 计算 giou
        giou = iou - (enclose_area - union) / enclose_area

        # 计算 giou loss
        giou_loss = 1 - giou

        return giou_loss

    def get_predict_bbox_label(self, local_embed, second):
        layer_size = int(second * 10)
        x = local_embed.reshape((local_embed.shape[0], 350 // layer_size, layer_size, 64)).flatten(-2, -1)
        x = self.mlp[second](x)
        num_video, sec = x.shape[0], x.shape[1]
        x = x.reshape(num_video, sec, self.anchor_num, self.dim)
        top = torch.arange(0, sec).unsqueeze(-1).to(local_embed.device)
        center = x[:, :, :, 36].sigmoid() + top
        boundL1 = torch.exp(x[:, :, 1, 37])
        boundL2 = torch.exp(x[:, :, 0, 37] - math.log(2))
        boundL = torch.stack((boundL2, boundL1), dim=-1)
        boundR1 = torch.exp(x[:, :, 1, 38])
        boundR2 = torch.exp(x[:, :, 0, 38] - math.log(2))
        boundR = torch.stack((boundR2, boundR1), dim=-1)

        left = torch.clamp(center - boundL, 0, sec)
        right = torch.clamp(center + boundR, 0, sec)
        bbox1 = torch.stack((left[:, :, 0], right[:, :, 0]), -1)
        bbox2 = torch.stack((left[:, :, 1], right[:, :, 1]), -1)
        predict_bbox = torch.stack((bbox1, bbox2), dim=-2)
        predict_bbox = predict_bbox * second
        predict_obj = x[:, :, :, 39].unsqueeze(-1).sigmoid()
        predict_cls = x[:, :, :, :36].sigmoid()
        return predict_bbox, predict_obj, predict_cls

    def one_dimensional_nms(self, boxes, threshold=0.5):
        """
        一维时间序列的非极大值抑制。

        参数：
            boxes：形状为 (B,N,2) 的 tensor ，每个元素代表一个时间段，具有开始时间和结束时间。
            threshold：得分阈值，用于过滤重叠时间段。

        返回：
            形状为 (B,M,2) 的 tensor，其中 M 是过滤后的时间段数量。
        """

        # 按照置信度排序
        indices = torch.argsort(boxes[..., 2], dim=-1, descending=True)
        boxes = boxes[torch.arange(boxes.shape[0])[:, None], indices]

        # 进行非极大值抑制
        keep = boxes[:, :, 2] > threshold
        for i in range(boxes.shape[1] - 1):
            if not keep[:, i].any():
                overlap = torch.maximum(torch.tensor([0], device=self.device1),
                                        torch.minimum(boxes[:, i, 1].reshape(-1, 1),
                                                      boxes[:, i + 1:, 1]) - torch.maximum(
                                            boxes[:, i, 0].reshape(-1, 1),
                                            boxes[:, i + 1:,
                                            0])) / \
                          torch.maximum(torch.tensor([0], device=self.device1),
                                        torch.maximum(boxes[:, i, 1].reshape(-1, 1),
                                                      boxes[:, i + 1:, 1]) - torch.minimum(
                                            boxes[:, i, 0].reshape(-1, 1),
                                            boxes[:, i + 1:,
                                            0]))

                keep[:, i + 1:] = (overlap <= threshold) * keep[:, [i]]

        # 返回保留的时间段
        cls_label = torch.argmax(boxes[:, :, 3:], dim=-1)
        boxes = torch.cat((boxes[:, :, :3], cls_label.unsqueeze(-1)), dim=-1)
        return boxes, keep

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('HiVT')
        parser.add_argument('--historical_steps', type=int, default=350)
        parser.add_argument('--future_steps', type=int, default=30)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--rotate', type=bool, default=True)
        parser.add_argument('--node_dim', type=int, default=2)
        parser.add_argument('--edge_dim', type=int, default=2)
        parser.add_argument('--embed_dim', type=int, default=64)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_temporal_layers', type=int, default=4)
        parser.add_argument('--num_global_layers', type=int, default=3)
        parser.add_argument('--local_radius', type=float, default=50)
        parser.add_argument('--parallel', type=bool, default=False)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser
