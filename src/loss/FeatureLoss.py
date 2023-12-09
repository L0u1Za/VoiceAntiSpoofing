import torch
from torch import Tensor

class FeatureLoss(torch.nn.Module):
    def forward(self, feat_target, feat_pred,
                **batch) -> Tensor:
        loss = 0.0
        for f_t, f_p in zip(feat_target, feat_pred):
            for target, pred in zip(f_t, f_p):
                loss += torch.mean(torch.abs(target - pred[:,:,:target.shape[2]]))

        return loss
