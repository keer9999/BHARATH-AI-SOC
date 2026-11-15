#!/usr/bin/env python3
"""
Test Suite for Hailo AI Backend & Multi-Backend Inference
AI-Enabled Smart Waste Segregation & Monitoring System
"""

import os
import sys
import tempfile
import numpy as np
import pytest

MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from config import DEFAULT_CONFIG, DetectionConfig
from inference_backend import (
    InferenceMode,
    RawDetection,
    letterbox_image,
    non_max_suppression,
    decode_yolo_detections,
    create_inference_backend,
    MockInferenceBackend,
    DevInferenceBackend,
    OnnxInferenceBackend,
    HailoInferenceBackend,
)
from waste_detector import WasteDetector, WasteFrame
from hailo_export import generate_hailo_model_script


# ─── 1. InferenceMode Normalization Tests ──────────────────────────────────────

def test_inference_mode_normalization():
    assert InferenceMode.normalize("dev") == InferenceMode.DEV
    assert InferenceMode.normalize("pytorch") == InferenceMode.DEV
    assert InferenceMode.normalize("torch") == InferenceMode.DEV
    assert InferenceMode.normalize("onnx") == InferenceMode.ONNX
    assert InferenceMode.normalize("ort") == InferenceMode.ONNX
    assert InferenceMode.normalize("hailo") == InferenceMode.HAILO
    assert InferenceMode.normalize("hailo8") == InferenceMode.HAILO
    assert InferenceMode.normalize("hailo8l") == InferenceMode.HAILO
    assert InferenceMode.normalize("mock") == InferenceMode.MOCK
    assert InferenceMode.normalize("unknown_mode") == InferenceMode.DEV


# ─── 2. Letterbox Preprocessing & Box Scaling Tests ────────────────────────────

def test_letterbox_image_properties():
    # 480x640 (aspect 4:3) image
    sample = np.zeros((480, 640, 3), dtype=np.uint8)
    padded, ratio, (pad_w, pad_h) = letterbox_image(sample, target_size=640)

    assert padded.shape == (640, 640, 3)
    assert ratio == 1.0  # max(480, 640) = 640 -> ratio = 1.0
    assert pad_w == 0.0
    assert pad_h == 80.0  # (640 - 480) / 2 = 80


def test_letterbox_square_image():
    sample = np.zeros((300, 300, 3), dtype=np.uint8)
    padded, ratio, (pad_w, pad_h) = letterbox_image(sample, target_size=640)

    assert padded.shape == (640, 640, 3)
    assert round(ratio, 4) == round(640 / 300, 4)
    assert pad_w == 0.0
    assert pad_h == 0.0


# ─── 3. Non-Maximum Suppression (NMS) Algorithm Tests ─────────────────────────

def test_non_max_suppression_disjoint_boxes():
    # Two non-overlapping boxes
    boxes = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [100.0, 100.0, 150.0, 150.0],
    ])
    scores = np.array([0.9, 0.8])
    keep = non_max_suppression(boxes, scores, conf_threshold=0.5, iou_threshold=0.45)

    assert len(keep) == 2
    assert keep == [0, 1]


def test_non_max_suppression_overlapping_boxes():
    # Two highly overlapping boxes (IoU > 0.8)
    boxes = np.array([
        [10.0, 10.0, 100.0, 100.0],
        [12.0, 12.0, 102.0, 102.0],
    ])
    scores = np.array([0.95, 0.70])
    keep = non_max_suppression(boxes, scores, conf_threshold=0.5, iou_threshold=0.45)

    # Lower scoring duplicate must be suppressed
    assert len(keep) == 1
    assert keep == [0]


# ─── 4. YOLO Tensor Decoding Tests ─────────────────────────────────────────────

def test_decode_yolo_detections_synthetic():
    # Synthetic output tensor: shape (1, 5, 8400)
    # Channel 0: cx=320, cy=320, w=100, h=200, conf=0.92 at index 42
    tensor = np.zeros((1, 5, 8400), dtype=np.float32)
    tensor[0, 0, 42] = 320.0
    tensor[0, 1, 42] = 320.0
    tensor[0, 2, 42] = 100.0
    tensor[0, 3, 42] = 200.0
    tensor[0, 4, 42] = 0.92

    orig_shape = (480, 640)
    detections = decode_yolo_detections(
        output_tensor=tensor,
        conf_threshold=0.5,
        iou_threshold=0.45,
        orig_shape=orig_shape,
        class_names={0: "bottle"},
        ratio=1.0,
        pad=(0.0, 80.0),
    )

    assert len(detections) == 1
    det = detections[0]
    assert det.class_id == 0
    assert det.class_name == "bottle"
    assert round(det.confidence, 2) == 0.92

    # Verify coordinate conversion:
    # x1 = (320 - 50 - 0) / 1.0 = 270
    # y1 = (320 - 100 - 80) / 1.0 = 140
    # x2 = (320 + 50 - 0) / 1.0 = 370
    # y2 = (320 + 100 - 80) / 1.0 = 340
    assert round(det.x1, 1) == 270.0
    assert round(det.y1, 1) == 140.0
    assert round(det.x2, 1) == 370.0
    assert round(det.y2, 1) == 340.0


# ─── 5. Mock Backend Tests ─────────────────────────────────────────────────────

def test_mock_backend_predict():
    backend = MockInferenceBackend(simulated_latency_ms=10.0)
    assert backend.load() is True

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets, latency = backend.predict(frame, conf_threshold=0.5)

    assert len(dets) == 1
    assert dets[0].class_name == "bottle"
    assert dets[0].confidence == 0.88
    assert latency == 10.0


# ─── 6. ONNX Backend Inference Tests ───────────────────────────────────────────

def test_onnx_backend_inference():
    onnx_path = os.path.join(MODULE_DIR, "yolo", "weights", "weights.onnx")
    if not os.path.exists(onnx_path):
        pytest.skip(f"ONNX model {onnx_path} not found")

    backend = OnnxInferenceBackend(onnx_path=onnx_path)
    assert backend.load() is True

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets, latency = backend.predict(frame, conf_threshold=0.35)

    assert isinstance(dets, list)
    assert latency > 0.0


# ─── 7. Hailo Backend Fallback & Error Handling Tests ──────────────────────────

def test_hailo_backend_handles_missing_runtime():
    # Tests that Hailo backend raises FileNotFoundError or falls back cleanly
    backend = HailoInferenceBackend(
        hef_path="non_existent_weights.hef",
        allow_fallback=True
    )
    loaded = backend.load()
    # On systems without Hailo hardware or driver, allow_fallback returns False cleanly
    assert loaded is False


def test_hailo_backend_strict_raises():
    backend = HailoInferenceBackend(
        hef_path="non_existent_weights.hef",
        allow_fallback=False
    )
    with pytest.raises((RuntimeError, FileNotFoundError)):
        backend.load()


# ─── 8. WasteDetector End-to-End Integration with Backends ─────────────────────

def test_waste_detector_with_mock_backend():
    config = DetectionConfig(backend="mock")
    detector = WasteDetector(config)

    assert detector.load_model() is True
    assert detector.backend.mode == InferenceMode.MOCK

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    waste_frame = detector.detect(frame, frame_id=1)

    assert isinstance(waste_frame, WasteFrame)
    assert waste_frame.frame_id == 1
    assert len(waste_frame.detections) == 1
    assert waste_frame.detections[0].class_name == "plastic"
    assert waste_frame.detections[0].confidence == 0.88


def test_waste_detector_with_onnx_backend():
    onnx_path = os.path.join(MODULE_DIR, "yolo", "weights", "weights.onnx")
    if not os.path.exists(onnx_path):
        pytest.skip("weights.onnx missing")

    config = DetectionConfig(backend="onnx", onnx_path=onnx_path)
    detector = WasteDetector(config)

    assert detector.load_model() is True
    assert detector.backend.mode == InferenceMode.ONNX

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    waste_frame = detector.detect(frame, frame_id=2)

    assert isinstance(waste_frame, WasteFrame)
    assert waste_frame.frame_id == 2


# ─── 9. Hailo Model Script (.alls) Generation Test ─────────────────────────────

def test_generate_hailo_model_script():
    with tempfile.NamedTemporaryFile(suffix=".alls", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        res_path = generate_hailo_model_script(tmp_path, input_name="images")
        assert os.path.exists(res_path)
        with open(res_path, "r") as f:
            content = f.read()
        assert "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])" in content
        assert "compiler_optimization_level=max" in content
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
