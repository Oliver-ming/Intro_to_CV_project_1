"""Course face and text detectors for the hybrid.

Face boxes come from OpenCV FaceDetectorYN with the YuNet ONNX file.
Text boxes come from TextDetectionModel_EAST when its frozen graph is present.
A missing weight file yields an empty mask for that prior. Boxes are not dilated.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
YUNET = ROOT / "weights" / "face_detection_yunet_2023mar.onnx"
EAST = ROOT / "weights" / "frozen_east_text_detection.pb"


def certain_mask(frame: np.ndarray) -> np.ndarray:
    image = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    _fill_boxes(mask, _face_boxes(image))
    _fill_boxes(mask, _text_boxes(image))
    return mask


def _face_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    if not YUNET.is_file():
        return []
    height, width = image.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(YUNET), "", (width, height))
    detector.setInputSize((width, height))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, faces = detector.detect(bgr)
    if faces is None:
        return []
    boxes = []
    for face in faces:
        x, y, w, h = [int(round(value)) for value in face[:4]]
        boxes.append((x, y, x + w, y + h))
    return boxes


def _text_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    if not EAST.is_file():
        return []
    height, width = image.shape[:2]
    detector = cv2.dnn_TextDetectionModel_EAST(str(EAST))
    detector.setConfidenceThreshold(0.5)
    detector.setNMSThreshold(0.4)
    detector.setInputParams(1.0, (width - width % 32, height - height % 32), (123.68, 116.78, 103.94), True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    first, second = detector.detect(bgr)
    quadrangles = first if np.asarray(first).ndim >= 2 else second
    return _text_boxes_from_detect(quadrangles, second if quadrangles is first else first, height, width)


def _text_boxes_from_detect(quadrangles, confidences, height: int, width: int) -> list[tuple[int, int, int, int]]:
    del confidences, height, width
    boxes = []
    array = np.asarray(quadrangles)
    if array.ndim < 2 or array.size == 0:
        return boxes
    if array.ndim == 2:
        array = array.reshape(1, *array.shape)
    for quad in array:
        xs = quad[:, 0]
        ys = quad[:, 1]
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    return boxes


def _fill_boxes(mask: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> None:
    height, width = mask.shape
    for x0, y0, x1, y1 in boxes:
        mask[max(y0, 0) : min(y1, height), max(x0, 0) : min(x1, width)] = True
