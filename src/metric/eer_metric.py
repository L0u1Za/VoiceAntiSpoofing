from typing import List

import torch
from torch import Tensor

from src.base.base_metric import BaseMetric
from src.base.base_text_encoder import BaseTextEncoder
from src.metric.utils import compute_eer

import numpy as np

class EERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, prediction: Tensor, label: Tensor, **kwargs):
        bonafide_scores = prediction[label == 0, 0]
        other_scores = prediction[label == 1, 0]

        eer, _ = compute_eer(bonafide_scores.detach().cpu().numpy(), other_scores.detach().cpu().numpy())

        return eer
