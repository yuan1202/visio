import os

import numpy as np
import cv2

import depthai
print('depthai module: ', depthai.__file__)


def cos_dist(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def batch_cos_dist(a, b):
    return np.dot(a, b.T) / np.dot(np.linalg.norm(a, axis=1)[:, None], np.linalg.norm(b, axis=1)[:, None].T)


def frame_norm(frame, bbox):
    return (np.clip(np.array(bbox), 0, 1) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape)
    return resized.transpose(2, 0, 1)
