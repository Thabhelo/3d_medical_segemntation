from __future__ import annotations

from typing import Dict

import torch
from monai.metrics import DiceMetric, MeanIoU


class EvaluationMetrics:
    def __init__(self, include_background: bool = False):
        self.metrics = {
            "dice": DiceMetric(include_background=include_background, reduction="mean"),
            "iou": MeanIoU(include_background=include_background),
        }

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for name, metric in self.metrics.items():
            try:
                metric.reset()
                metric(y_pred=predictions, y=targets)
                value = metric.aggregate().item()
                results[name] = float(value)
            except Exception:
                results[name] = float("nan")
        return results


