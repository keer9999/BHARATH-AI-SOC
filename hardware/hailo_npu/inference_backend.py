#!/usr/bin/env python3
"""
Dual-Mode Inference Backend Module
AI-Enabled Smart Waste Segregation & Monitoring System

Supports:
    1. Mode.DEV (PyTorch / Ultralytics YOLO) - Local development & training verification
    2. Mode.ONNX (ONNX Runtime) - Software validation of static graph
    3. Mode.HAILO (HailoRT Virtual Streams) - Hardware acceleration on Hailo-8 / Hailo-8L
    4. Mode.MOCK (Pure Software Mock) - CI / hardware-free unit tests
"""

import os
import sys
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ─── Inference Modes ──────────────────────────────────────────────────────────

class InferenceMode(str, Enum):
    """Supported inference backend modes."""
    DEV = "dev"          # PyTorch / Ultralytics YOLO (default for workstation)
    PYTORCH = "pytorch"  # Alias for DEV
    ONNX = "onnx"        # ONNX Runtime (CPU/GPU)
    HAILO = "hailo"      # Hailo-8 / Hailo-8L NPU via hailo_platform
    MOCK = "mock"        # Software mock for testing/CI

    @classmethod
    def normalize(cls, mode_val: Union[str, "InferenceMode"]) -> "InferenceMode":
        if isinstance(mode_val, cls):
            if mode_val == cls.PYTORCH:
                return cls.DEV
            return mode_val
        val_str = str(mode_val).strip().lower()
        if val_str in ("dev", "pytorch", "torch", "gpu", "cpu"):
            return cls.DEV
        elif val_str in ("onnx", "ort", "onnxruntime"):
            return cls.ONNX
        elif val_str in ("hailo", "hailo8", "hailo8l", "npu", "hef"):
            return cls.HAILO
        elif val_str in ("mock", "dummy", "test", "simulated"):
            return cls.MOCK
        else:
            logger.warning(f"Unrecognized inference mode '{mode_val}', falling back to DEV")
            return cls.DEV


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class RawDetection:
    """Standardized detection result output by any inference backend."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in original image coordinates
    confidence: float
    class_id: int
    class_name: str

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def centroid(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


# ─── Computer Vision Helpers (Preprocessing & NMS) ─────────────────────────────

def letterbox_image(
    image: np.ndarray,
    target_size: int = 640,
    fill_color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize and pad image preserving aspect ratio to (target_size, target_size).

    Returns:
        padded_image: np.ndarray (H, W, C)
        ratio: float scaling factor
        (pad_x, pad_y): float padding on left and top
    """
    import cv2

    shape = image.shape[:2]  # (h, w)
    ratio = min(target_size / shape[0], target_size / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    dw = target_size - new_unpad[0]
    dh = target_size - new_unpad[1]
    dw_half = dw / 2.0
    dh_half = dh / 2.0

    if shape[::-1] != new_unpad:
        resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    else:
        resized = image

    top = int(round(dh_half - 0.1))
    bottom = int(round(dh_half + 0.1))
    left = int(round(dw_half - 0.1))
    right = int(round(dw_half + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=fill_color
    )
    return padded, ratio, (dw_half, dh_half)


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> List[int]:
    """
    Vectorized Non-Maximum Suppression (NMS) in NumPy.

    Args:
        boxes: np.ndarray of shape (N, 4) in [x1, y1, x2, y2]
        scores: np.ndarray of shape (N,)
        conf_threshold: minimum confidence score
        iou_threshold: IoU overlap threshold

    Returns:
        List of selected box indices.
    """
    if len(boxes) == 0:
        return []

    # Filter by confidence
    valid_mask = scores >= conf_threshold
    indices = np.where(valid_mask)[0]
    if len(indices) == 0:
        return []

    boxes = boxes[indices]
    scores = scores[indices]

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(indices[i])

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = np.zeros_like(inter)
        nonzero = union > 0
        iou[nonzero] = inter[nonzero] / union[nonzero]

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep


def decode_yolo_detections(
    output_tensor: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    orig_shape: Tuple[int, int],
    class_names: Optional[Dict[int, str]] = None,
    ratio: float = 1.0,
    pad: Tuple[float, float] = (0.0, 0.0),
) -> List[RawDetection]:
    """
    Decode raw YOLOv8/11 output tensor into standardized RawDetection list.

    Handles shapes:
        - (1, 5, 8400) or (1, 4+nc, 8400)
        - (1, 8400, 5) or (1, 8400, 4+nc)
        - (5, 8400) or (8400, 5)

    Tensor layout:
        [cx, cy, w, h, class_0_prob, class_1_prob, ...]
    """
    if class_names is None:
        class_names = {0: "bottle"}

    # Squeeze batch dimension if present
    tensor = np.squeeze(output_tensor)
    if tensor.ndim != 2:
        logger.warning(f"Unexpected tensor dimensionality: {tensor.shape}")
        return []

    # Ensure shape is (num_anchors, channels) e.g. (8400, 5)
    if tensor.shape[0] < tensor.shape[1]:
        # Currently (channels, anchors) -> transpose to (anchors, channels)
        tensor = tensor.T

    num_anchors, num_channels = tensor.shape
    if num_channels < 5:
        logger.warning(f"Expected at least 5 channels (cx, cy, w, h, class_score), got {num_channels}")
        return []

    # Extract coordinates in 640x640 space
    cx = tensor[:, 0]
    cy = tensor[:, 1]
    w = tensor[:, 2]
    h = tensor[:, 3]

    # Convert to x1, y1, x2, y2
    x1 = cx - (w / 2.0)
    y1 = cy - (h / 2.0)
    x2 = cx + (w / 2.0)
    y2 = cy + (h / 2.0)

    # Revert letterbox padding and scale back to original image dimensions
    pad_x, pad_y = pad
    x1 = (x1 - pad_x) / ratio
    y1 = (y1 - pad_y) / ratio
    x2 = (x2 - pad_x) / ratio
    y2 = (y2 - pad_y) / ratio

    # Clip to original image boundaries
    orig_h, orig_w = orig_shape[:2]
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    boxes = np.column_stack([x1, y1, x2, y2])
    class_probs = tensor[:, 4:]  # (num_anchors, num_classes)

    detections: List[RawDetection] = []
    num_classes = class_probs.shape[1]

    for c in range(num_classes):
        scores = class_probs[:, c]
        keep_indices = non_max_suppression(
            boxes, scores, conf_threshold=conf_threshold, iou_threshold=iou_threshold
        )
        for idx in keep_indices:
            cls_name = class_names.get(c, f"class_{c}")
            detections.append(
                RawDetection(
                    bbox=(float(boxes[idx, 0]), float(boxes[idx, 1]),
                          float(boxes[idx, 2]), float(boxes[idx, 3])),
                    confidence=float(scores[idx]),
                    class_id=c,
                    class_name=cls_name,
                )
            )

    return detections


# ─── Inference Backend Abstraction ─────────────────────────────────────────────

class BaseInferenceBackend(ABC):
    """Abstract Base Class for modular inference backends."""

    def __init__(self, mode: InferenceMode):
        self.mode = mode
        self.is_loaded = False
        self.class_names: Dict[int, str] = {0: "bottle"}

    @abstractmethod
    def load(self) -> bool:
        """Load model weights / configure hardware."""
        pass

    @abstractmethod
    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45
    ) -> Tuple[List[RawDetection], float]:
        """
        Run inference on a single BGR frame.

        Returns:
            (detections, inference_time_ms)
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Return backend runtime metadata."""
        return {
            "mode": self.mode.value,
            "is_loaded": self.is_loaded,
            "class_names": self.class_names,
        }

    def close(self) -> None:
        """Clean up resources."""
        pass


# ─── Mode 1: Development / PyTorch Backend (Ultralytics YOLO) ──────────────────

class DevInferenceBackend(BaseInferenceBackend):
    """
    Standard development backend using PyTorch and Ultralytics YOLO.
    Runs on workstation CPU or NVIDIA GPU.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        half_precision: bool = False,
        input_size: int = 640,
    ):
        super().__init__(InferenceMode.DEV)
        self.model_path = model_path
        self.device = device
        self.half_precision = half_precision
        self.input_size = input_size
        self.yolo_model: Optional[Any] = None

    def load(self) -> bool:
        try:
            from ultralytics import YOLO
            if not os.path.exists(self.model_path):
                logger.warning(f"DEV model file not found at {self.model_path}. Trying fallback yolov8n.pt")
                self.yolo_model = YOLO("yolov8n.pt")
            else:
                self.yolo_model = YOLO(self.model_path)

            if hasattr(self.yolo_model, "names") and self.yolo_model.names:
                self.class_names = dict(self.yolo_model.names)

            self.is_loaded = True
            logger.info(f"Loaded DEV (PyTorch/Ultralytics) model: {self.model_path} on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load DEV inference backend: {e}")
            self.is_loaded = False
            return False

    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45
    ) -> Tuple[List[RawDetection], float]:
        if not self.is_loaded or self.yolo_model is None:
            return [], 0.0

        start_time = time.time()
        predict_kwargs = {
            "source": frame,
            "conf": conf_threshold,
            "iou": iou_threshold,
            "imgsz": self.input_size,
            "device": self.device,
            "verbose": False,
        }
        if self.device != "cpu" and self.half_precision:
            predict_kwargs["half"] = True

        results = self.yolo_model.predict(**predict_kwargs)
        inference_ms = (time.time() - start_time) * 1000.0

        detections: List[RawDetection] = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = str(result.names.get(class_id, "unknown")).strip().lower()
                    detections.append(
                        RawDetection(
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name,
                        )
                    )

        return detections, inference_ms


# ─── Mode 2: ONNX Runtime Backend ──────────────────────────────────────────────

class OnnxInferenceBackend(BaseInferenceBackend):
    """
    ONNX Runtime backend.
    Executes the exported static-shape ONNX model on CPU or CUDA.
    Validates fixed-shape graph behavior prior to Hailo DFC compilation.
    """

    def __init__(
        self,
        onnx_path: str,
        input_size: int = 640,
        providers: Optional[List[str]] = None,
        class_names: Optional[Dict[int, str]] = None,
    ):
        super().__init__(InferenceMode.ONNX)
        self.onnx_path = onnx_path
        self.input_size = input_size
        self.providers = providers or ["CPUExecutionProvider"]
        if class_names:
            self.class_names = class_names
        self.session: Optional[Any] = None
        self.input_name: str = "images"
        self.output_names: List[str] = []

    def load(self) -> bool:
        try:
            import onnxruntime as ort
            if not os.path.exists(self.onnx_path):
                logger.error(f"ONNX model file not found at: {self.onnx_path}")
                return False

            self.session = ort.InferenceSession(self.onnx_path, providers=self.providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]

            self.is_loaded = True
            logger.info(f"Loaded ONNX backend session from {self.onnx_path} (providers={self.providers})")
            return True
        except ImportError:
            logger.error("onnxruntime is not installed. Run 'pip install onnxruntime'.")
            return False
        except Exception as e:
            logger.error(f"Failed to load ONNX session: {e}")
            return False

    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45
    ) -> Tuple[List[RawDetection], float]:
        if not self.is_loaded or self.session is None:
            return [], 0.0

        import cv2

        start_time = time.time()
        orig_shape = frame.shape[:2]

        # 1. Preprocess: Letterbox to 640x640
        padded, ratio, pad = letterbox_image(frame, target_size=self.input_size)

        # 2. Convert BGR to RGB, transpose HWC -> CHW, normalize to [0, 1] FP32
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        input_tensor = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # (3, 640, 640)
        input_tensor = np.expand_dims(input_tensor, axis=0)   # (1, 3, 640, 640)

        # 3. Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        raw_output = outputs[0]

        # 4. Post-processing & NMS
        detections = decode_yolo_detections(
            output_tensor=raw_output,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            orig_shape=orig_shape,
            class_names=self.class_names,
            ratio=ratio,
            pad=pad,
        )

        inference_ms = (time.time() - start_time) * 1000.0
        return detections, inference_ms


# ─── Mode 3: Hailo Hardware-Accelerated Backend (Hailo-8 / Hailo-8L) ────────────

class HailoInferenceBackend(BaseInferenceBackend):
    """
    Hardware-accelerated inference backend using HailoRT (hailo_platform).
    Runs compiled .hef models on Hailo-8 (26 TOPS) and Hailo-8L (13 TOPS)
    accelerators connected via M.2 / PCIe on Raspberry Pi 5 or NVIDIA Jetson.

    Input Handling:
        When the model is compiled with input normalization in Hailo DFC,
        the Hailo NPU accepts raw UINT8 RGB tensors (shape [1, 640, 640, 3] NHWC)
        and executes RGB/normalization directly on-chip in hardware,
        saving precious CPU cycles on embedded edge devices.
    """

    def __init__(
        self,
        hef_path: str,
        input_size: int = 640,
        batch_size: int = 1,
        class_names: Optional[Dict[int, str]] = None,
        allow_fallback: bool = False,
    ):
        super().__init__(InferenceMode.HAILO)
        self.hef_path = hef_path
        self.input_size = input_size
        self.batch_size = batch_size
        if class_names:
            self.class_names = class_names
        self.allow_fallback = allow_fallback

        self._vdevice = None
        self._network_group = None
        self._infer_pipeline = None
        self._input_vstream_info = None
        self._output_vstream_infos = []
        self._input_is_uint8 = True
        self._input_is_nhwc = True

    def load(self) -> bool:
        """Initialize Hailo VDevice, configure network group, and open virtual streams."""
        try:
            from hailo_platform import (
                HEF,
                VDevice,
                ConfigureParams,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                FormatType,
                HailoStreamInterface,
            )
        except ImportError:
            msg = (
                "HailoRT Python API (hailo_platform) is not installed.\n"
                "To deploy on Raspberry Pi 5 / Jetson:\n"
                "  1. Install Hailo PCIe driver & HailoRT: sudo apt install hailo-all\n"
                "  2. Install hailo_platform wheel matching your HailoRT version."
            )
            if self.allow_fallback:
                logger.warning(f"{msg}\nProceeding with fallback mode.")
                return False
            raise RuntimeError(msg)

        if not os.path.exists(self.hef_path):
            msg = f"Hailo executable file (.hef) not found at: {self.hef_path}"
            if self.allow_fallback:
                logger.warning(msg)
                return False
            raise FileNotFoundError(msg)

        try:
            logger.info(f"Initializing Hailo VDevice and loading HEF: {self.hef_path}")
            hef = HEF(self.hef_path)

            vdevice_params = VDevice.create_params()
            self._vdevice = VDevice(vdevice_params)

            # Configure network group
            configure_params = ConfigureParams.create_from_hef(
                hef=hef, interface=HailoStreamInterface.PCIe
            )
            network_groups = self._vdevice.configure(hef, configure_params)
            if not network_groups:
                raise RuntimeError("Failed to configure network group from HEF")
            self._network_group = network_groups[0]

            # Inspect input vstream specifications
            input_vstream_infos = hef.get_input_vstream_infos()
            if input_vstream_infos:
                self._input_vstream_info = input_vstream_infos[0]
                # Check expected input format
                shape = self._input_vstream_info.shape
                # Typically (640, 640, 3) for NHWC or (3, 640, 640) for NCHW
                if len(shape) == 3 and shape[2] == 3:
                    self._input_is_nhwc = True
                else:
                    self._input_is_nhwc = False

            # Configure virtual stream parameters (UINT8 input for hardware offload, FLOAT32 output)
            input_params = InputVStreamParams.make(
                self._network_group, format_type=FormatType.UINT8
            )
            output_params = OutputVStreamParams.make(
                self._network_group, format_type=FormatType.FLOAT32
            )

            # Create InferVStreams pipeline context
            self._infer_pipeline = InferVStreams(
                self._network_group, input_params, output_params
            )
            self._infer_pipeline.__enter__()

            self.is_loaded = True
            logger.info(
                f"✅ Hailo NPU successfully initialized on {self.hef_path} "
                f"(Input layout: {'NHWC UINT8' if self._input_is_nhwc else 'NCHW'})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Hailo backend: {e}")
            self.close()
            if self.allow_fallback:
                return False
            raise

    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45
    ) -> Tuple[List[RawDetection], float]:
        if not self.is_loaded or self._infer_pipeline is None:
            return [], 0.0

        import cv2

        start_time = time.time()
        orig_shape = frame.shape[:2]

        # 1. Preprocess: Letterbox to 640x640
        padded, ratio, pad = letterbox_image(frame, target_size=self.input_size)

        # 2. Convert BGR to RGB (OpenCV is BGR, Hailo model expects RGB)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        # 3. Format input buffer: Hailo UINT8 (NHWC) vs FP32
        if self._input_is_nhwc:
            input_data = np.expand_dims(rgb.astype(np.uint8), axis=0)  # (1, 640, 640, 3)
        else:
            chw = np.transpose(rgb, (2, 0, 1))
            input_data = np.expand_dims(chw.astype(np.uint8), axis=0)  # (1, 3, 640, 640)

        # Prepare input dictionary for InferVStreams
        input_name = self._input_vstream_info.name if self._input_vstream_info else None
        if input_name:
            feed_dict = {input_name: input_data}
        else:
            feed_dict = input_data

        # 4. Hardware Inference via HailoRT
        raw_outputs = self._infer_pipeline.infer(feed_dict)

        # 5. Extract output tensor
        if isinstance(raw_outputs, dict):
            # Take the primary output tensor
            output_tensor = next(iter(raw_outputs.values()))
        else:
            output_tensor = raw_outputs

        # 6. Post-processing & NMS
        # If output is raw detection head [1, 5, 8400] or [1, 8400, 5]:
        if isinstance(output_tensor, np.ndarray) and output_tensor.size >= 8400 * 5:
            detections = decode_yolo_detections(
                output_tensor=output_tensor,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                orig_shape=orig_shape,
                class_names=self.class_names,
                ratio=ratio,
                pad=pad,
            )
        elif isinstance(output_tensor, (list, tuple)) and len(output_tensor) > 0:
            # Handle structured Hailo NMS output format if present
            detections = self._parse_hailo_nms_output(
                output_tensor, conf_threshold, orig_shape, ratio, pad
            )
        else:
            detections = []

        inference_ms = (time.time() - start_time) * 1000.0
        return detections, inference_ms

    def _parse_hailo_nms_output(
        self,
        output_tensor: Any,
        conf_threshold: float,
        orig_shape: Tuple[int, int],
        ratio: float,
        pad: Tuple[float, float],
    ) -> List[RawDetection]:
        """Parse Hailo on-chip NMS output format if model was compiled with Hailo NMS."""
        detections: List[RawDetection] = []
        orig_h, orig_w = orig_shape
        pad_x, pad_y = pad

        # Hailo NMS typically outputs a list of arrays per class: [ymin, xmin, ymax, xmax, score]
        try:
            for cls_id, cls_detections in enumerate(output_tensor):
                if not isinstance(cls_detections, np.ndarray) or len(cls_detections) == 0:
                    continue
                for det in cls_detections:
                    if len(det) < 5:
                        continue
                    score = float(det[4])
                    if score < conf_threshold:
                        continue
                    # Hailo NMS normalized coords [0, 1] relative to input_size
                    y1 = (det[0] * self.input_size - pad_y) / ratio
                    x1 = (det[1] * self.input_size - pad_x) / ratio
                    y2 = (det[2] * self.input_size - pad_y) / ratio
                    x2 = (det[3] * self.input_size - pad_x) / ratio

                    x1 = float(np.clip(x1, 0, orig_w))
                    y1 = float(np.clip(y1, 0, orig_h))
                    x2 = float(np.clip(x2, 0, orig_w))
                    y2 = float(np.clip(y2, 0, orig_h))

                    cls_name = self.class_names.get(cls_id, "plastic")
                    detections.append(
                        RawDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=score,
                            class_id=cls_id,
                            class_name=cls_name,
                        )
                    )
        except Exception as e:
            logger.warning(f"Failed to parse Hailo NMS output: {e}")

        return detections

    def close(self) -> None:
        """Safely release Hailo virtual streams and VDevice."""
        if self._infer_pipeline is not None:
            try:
                self._infer_pipeline.__exit__(None, None, None)
            except Exception:
                pass
            self._infer_pipeline = None

        if self._vdevice is not None:
            try:
                self._vdevice.release()
            except Exception:
                pass
            self._vdevice = None

        self.is_loaded = False


# ─── Mode 4: Mock Backend (Testing & CI) ────────────────────────────────────────

class MockInferenceBackend(BaseInferenceBackend):
    """
    Pure software mock backend.
    Returns deterministic or configurable dummy detections for automated tests,
    CI pipelines, or GUI layout development without requiring model weights or hardware.
    """

    def __init__(
        self,
        simulated_detections: Optional[List[RawDetection]] = None,
        simulated_latency_ms: float = 8.5,
    ):
        super().__init__(InferenceMode.MOCK)
        self.simulated_detections = simulated_detections
        self.simulated_latency_ms = simulated_latency_ms

    def load(self) -> bool:
        self.is_loaded = True
        logger.info("Loaded MOCK inference backend (simulated detections)")
        return True

    def predict(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45
    ) -> Tuple[List[RawDetection], float]:
        orig_h, orig_w = frame.shape[:2]

        if self.simulated_detections is not None:
            filtered = [
                d for d in self.simulated_detections
                if d.confidence >= conf_threshold
            ]
            return filtered, self.simulated_latency_ms

        # Generate a realistic mock plastic detection in the center of the frame
        center_x = orig_w / 2.0
        center_y = orig_h / 2.0
        box_w = orig_w * 0.25
        box_h = orig_h * 0.45

        x1 = max(0.0, center_x - box_w / 2.0)
        y1 = max(0.0, center_y - box_h / 2.0)
        x2 = min(float(orig_w), center_x + box_w / 2.0)
        y2 = min(float(orig_h), center_y + box_h / 2.0)

        mock_detection = RawDetection(
            bbox=(x1, y1, x2, y2),
            confidence=0.88,
            class_id=0,
            class_name="bottle",
        )
        return [mock_detection], self.simulated_latency_ms


# ─── Factory Function ──────────────────────────────────────────────────────────

def create_inference_backend(
    mode: Optional[Union[str, InferenceMode]] = None,
    model_path: Optional[str] = None,
    onnx_path: Optional[str] = None,
    hef_path: Optional[str] = None,
    device: str = "cpu",
    half_precision: bool = False,
    input_size: int = 640,
    allow_fallback: bool = True,
) -> BaseInferenceBackend:
    """
    Factory to instantiate the appropriate inference backend based on configuration,
    CLI argument, or environment variable (WASTE_INFERENCE_BACKEND / HAILO_BACKEND).

    Args:
        mode: Requested InferenceMode ("dev", "onnx", "hailo", "mock")
        model_path: Path to PyTorch / Ultralytics .pt checkpoint
        onnx_path: Path to ONNX .onnx model
        hef_path: Path to compiled Hailo .hef binary
        device: Device string for PyTorch ("0", "cpu")
        half_precision: Use FP16 on GPU
        input_size: Square model input resolution (640)
        allow_fallback: Fallback to DEV if Hailo/ONNX cannot be loaded

    Returns:
        Configured BaseInferenceBackend instance.
    """
    # 1. Resolve mode priority: Explicit arg > ENV var > Default "dev"
    if mode is None:
        env_mode = os.getenv("WASTE_INFERENCE_BACKEND", os.getenv("HAILO_BACKEND", "dev"))
        mode = env_mode

    selected_mode = InferenceMode.normalize(mode)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Resolve default file paths if not provided
    if model_path is None:
        model_path = os.path.join(base_dir, "yolo", "weights", "weights.pt")
    if onnx_path is None:
        onnx_path = os.path.join(base_dir, "yolo", "weights", "weights.onnx")
    if hef_path is None:
        hef_path = os.path.join(base_dir, "yolo", "weights", "weights.hef")

    # 3. Instantiate requested backend with graceful fallback
    if selected_mode == InferenceMode.HAILO:
        try:
            backend = HailoInferenceBackend(
                hef_path=hef_path,
                input_size=input_size,
                allow_fallback=allow_fallback,
            )
            if backend.load():
                return backend
            logger.warning(f"Hailo backend could not be initialized from {hef_path}.")
        except Exception as e:
            logger.warning(f"Hailo backend initialization error ({e}).")

        if allow_fallback:
            logger.info("Falling back to DEV (PyTorch/Ultralytics) backend...")
            return create_inference_backend(
                mode=InferenceMode.DEV,
                model_path=model_path,
                device=device,
                half_precision=half_precision,
                input_size=input_size,
                allow_fallback=False,
            )
        else:
            raise

    elif selected_mode == InferenceMode.ONNX:
        backend = OnnxInferenceBackend(
            onnx_path=onnx_path,
            input_size=input_size,
        )
        if backend.load():
            return backend
        elif allow_fallback:
            logger.warning(f"ONNX backend could not be loaded from {onnx_path}. Falling back to DEV mode.")
            return create_inference_backend(
                mode=InferenceMode.DEV,
                model_path=model_path,
                device=device,
                half_precision=half_precision,
                input_size=input_size,
                allow_fallback=False,
            )
        return backend

    elif selected_mode == InferenceMode.MOCK:
        backend = MockInferenceBackend()
        backend.load()
        return backend

    else:
        # Default: DEV (PyTorch / Ultralytics)
        backend = DevInferenceBackend(
            model_path=model_path,
            device=device,
            half_precision=half_precision,
            input_size=input_size,
        )
        backend.load()
        return backend
