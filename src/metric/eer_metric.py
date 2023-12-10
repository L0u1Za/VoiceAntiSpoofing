from typing import List

import torch
from torch import Tensor

from src.base.base_metric import BaseMetric
from src.base.base_text_encoder import BaseTextEncoder
from src.metric.utils import compute_eer


class EERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, predictions: Tensor, labels: Tensor, **kwargs):
        bonafide_scores = predictions[labels == 0, 0]
        other_scores = predictions[labels == 1, 1]

        eers, _ = compute_eer(bonafide_scores.detach().cpu().numpy(), other_scores.detach().cpu().numpy())

        return sum(eers) / len(eers)
