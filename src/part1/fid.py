"""Small-sample Fréchet distance on Inception pool features.

The distance is between every restored frame and every ground-truth frame
in one pool. Callers report the frame count because this is not a
full-dataset FID.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional
from torchvision.models import Inception_V3_Weights, inception_v3

from part1.metrics import frechet_distance


class InceptionPool:
    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, transform_input=False)
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model.to(self.device)

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        features = []
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        for frame in frames:
            image = torch.from_numpy(np.transpose(frame.astype(np.float32) / 255.0, (2, 0, 1)))
            image = image.unsqueeze(0).to(self.device)
            image = functional.interpolate(image, size=(299, 299), mode="bilinear", align_corners=False)
            image = (image - mean) / std
            with torch.no_grad():
                features.append(self.model(image).flatten().detach().cpu().numpy())
        return np.stack(features)

    def distance(self, restored: np.ndarray, ground_truth: np.ndarray) -> float:
        return pool_distance(self(restored), self(ground_truth))


def pool_distance(restored_features: np.ndarray, truth_features: np.ndarray) -> float:
    return frechet_distance(*_gaussian(restored_features), *_gaussian(truth_features))


def _gaussian(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    if len(values) < 2:
        raise ValueError("FID needs at least two frames in each pool")
    mean = values.mean(axis=0)
    cov = np.cov(values, rowvar=False)
    return mean, cov
