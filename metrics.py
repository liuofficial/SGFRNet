"""Evaluation metrics used by SGFRNet."""

from __future__ import annotations

import numpy as np
import torch
from skimage import measure


def _as_binary_numpy(tensor, threshold=0.5):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    return np.asarray(tensor) >= threshold


class SamplewiseSigmoidMetric:
    """Mean sample-wise IoU (nIoU)."""

    def __init__(self, nclass=1, score_thresh=0.5):
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.reset()

    def update(self, predictions, labels):
        predictions = _as_binary_numpy(predictions, self.score_thresh)
        labels = _as_binary_numpy(labels, 0.5)
        if predictions.ndim == 3:
            predictions = predictions[:, None, ...]
        if labels.ndim == 3:
            labels = labels[:, None, ...]

        for prediction, label in zip(predictions, labels):
            intersection = np.logical_and(prediction, label).sum(dtype=np.float64)
            union = np.logical_or(prediction, label).sum(dtype=np.float64)
            self.values.append(float(intersection / union) if union > 0 else 0.0)

    def get(self):
        return float(np.mean(self.values)) if self.values else 0.0

    def reset(self):
        self.values = []


class mIoU:
    """Global foreground IoU and foreground-pixel accuracy."""

    def __init__(self):
        self.reset()

    def update(self, predictions, labels):
        predictions = _as_binary_numpy(predictions, 0.5)
        labels = _as_binary_numpy(labels, 0.5)
        self.total_intersection += np.logical_and(predictions, labels).sum(dtype=np.float64)
        self.total_union += np.logical_or(predictions, labels).sum(dtype=np.float64)
        self.total_correct_foreground += np.logical_and(predictions, labels).sum(dtype=np.float64)
        self.total_foreground += labels.sum(dtype=np.float64)

    def get(self):
        pixel_accuracy = self.total_correct_foreground / max(self.total_foreground, np.finfo(float).eps)
        miou = self.total_intersection / max(self.total_union, np.finfo(float).eps)
        return float(pixel_accuracy), float(miou)

    def reset(self):
        self.total_intersection = 0.0
        self.total_union = 0.0
        self.total_correct_foreground = 0.0
        self.total_foreground = 0.0


class PD_FA:
    """Object-level probability of detection and pixel-level false-alarm rate.

    A predicted component matches a ground-truth component when their centroids
    are less than ``matching_distance`` pixels apart. Matching is one-to-one.
    """

    def __init__(self, matching_distance=3.0):
        self.matching_distance = float(matching_distance)
        self.reset()

    def update(self, predictions, labels, size):
        prediction = _as_binary_numpy(predictions, 0.5).squeeze()
        label = _as_binary_numpy(labels, 0.5).squeeze()

        prediction_regions = measure.regionprops(measure.label(prediction, connectivity=2))
        label_regions = measure.regionprops(measure.label(label, connectivity=2))
        unmatched_predictions = set(range(len(prediction_regions)))
        matched_targets = 0

        for target in label_regions:
            target_center = np.asarray(target.centroid)
            candidates = []
            for index in unmatched_predictions:
                prediction_center = np.asarray(prediction_regions[index].centroid)
                distance = float(np.linalg.norm(prediction_center - target_center))
                if distance < self.matching_distance:
                    candidates.append((distance, index))
            if candidates:
                _, best_index = min(candidates)
                unmatched_predictions.remove(best_index)
                matched_targets += 1

        self.detected_targets += matched_targets
        self.total_targets += len(label_regions)
        self.false_alarm_pixels += sum(prediction_regions[index].area for index in unmatched_predictions)
        self.total_pixels += int(size[0]) * int(size[1])

    def get(self):
        pd_value = self.detected_targets / self.total_targets if self.total_targets else 0.0
        fa_value = self.false_alarm_pixels / self.total_pixels if self.total_pixels else 0.0
        return float(pd_value), float(fa_value)

    def reset(self):
        self.detected_targets = 0
        self.total_targets = 0
        self.false_alarm_pixels = 0
        self.total_pixels = 0
