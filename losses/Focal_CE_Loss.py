import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(CrossEntropyFocalLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):

        pt = F.softmax(logits, 1)
        pt = pt.gather(1, target).view(-1)
        log_gt = torch.log(pt)

        if self.alpha is not None:
            alpha = self.alpha.gather(0, target.view(-1))
            log_gt = log_gt * alpha

        loss = -1 * (1 - pt) ** self.gamma * log_gt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss