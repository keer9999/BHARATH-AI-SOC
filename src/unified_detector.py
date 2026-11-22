#!/usr/bin/env python3
"""
src/unified_detector.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bharat AI-SoC  —  Unified Waste Detector  (single .pt  OR  ONNX Runtime)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Drop-in replacement for the multi-model WasteDetector that loads from ONE
of two formats:

  • models/unified/c2s_waste_ensemble.pt     (torch, full ensemble in 1 file)
  • models/unified/onnx/                      (ONNX Runtime per-specialist)

Usage
─────
  from src.unified_detector import UnifiedWasteDetector

  # From unified .pt (fastest load, works without ONNX Runtime)
  det = UnifiedWasteDetector.from_pt("models/unified/c2s_waste_ensemble.pt")

  # From ONNX directory (best for edge/FPGA/TensorRT deployment)
  det = UnifiedWasteDetector.from_onnx("models/unified/onnx/")

  # Run inference (accepts BGR numpy frame like OpenCV output)
  results = det.detect(frame, conf=0.25)
  for r in results:
      print(r["category"], r["confidence"], r["bbox"])
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)

# Canonical waste categories (must match EnsembleWasteDetector.WASTE_CATEGORIES)
WASTE_CATEGORIES = [
    "recyclable_plastic",
    "recyclable_glass",
    "recyclable_metal",
    "recyclable_paper_cardboard",
    "hazardous_ewaste",
    "organic_food",
    "general_waste",
]

# Display colour map (BGR)
CATEGORY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "recyclable_plastic":       (0,   200,  60),
    "recyclable_glass":         (0,   180, 240),
    "recyclable_metal":         (200, 180,   0),
    "recyclable_paper_cardboard":(50, 100, 220),
    "hazardous_ewaste":         (0,    30, 240),
    "organic_food":             (0,   140,  60),
    "general_waste":            (120, 120, 120),
}


class UnifiedWasteDetector:
    """
    Single unified waste detector backed by either a .pt ensemble or
    an ONNX Runtime session pool.

    Parameters
    ----------
    backend : str
        "pt"   — torch EnsembleWasteDetector
        "onnx" — ONNX Runtime multi-session pool
    """

    def __init__(self):
        self.backend:   str  = "none"
        self._pt_model       = None          # EnsembleWasteDetector instance
        self._ort_sessions   = {}            # role → ort.InferenceSession
        self._onnx_manifest  = []            # list of dicts from manifest JSON
        self.imgsz:    int   = 640
        self.device          = torch.device("cpu")
        self._loaded:  bool  = False

    # ── Factory constructors ──────────────────────────────────────────────────

    @classmethod
    def from_pt(
        cls,
        pt_path: Union[str, Path],
        device: str = "cpu",
    ) -> "UnifiedWasteDetector":
        """Load from a unified .pt checkpoint produced by build_unified_model.py"""
        self = cls()
        self.device = torch.device(device)
        pt_path = Path(pt_path)

        log.info(f"Loading unified .pt from {pt_path} …")
        t0 = time.perf_counter()
        ckpt = torch.load(pt_path, map_location=self.device, weights_only=False)
        self._pt_model = ckpt["model"]
        self._pt_model.eval()
        self.imgsz   = ckpt["metadata"].get("imgsz", 640)
        self.backend = "pt"
        self._loaded = True
        log.info(f"  ✓ Loaded in {(time.perf_counter()-t0)*1000:.0f} ms | "
                 f"models={len(self._pt_model.model_roles)}")
        return self

    @classmethod
    def from_onnx(
        cls,
        onnx_dir: Union[str, Path],
        providers: Optional[List[str]] = None,
    ) -> "UnifiedWasteDetector":
        """Load from an ONNX directory produced by build_unified_model.py"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX backend. "
                "Install with: pip install onnxruntime  (or onnxruntime-gpu)"
            )

        self = cls()
        onnx_dir = Path(onnx_dir)
        providers = providers or ["CPUExecutionProvider"]

        manifest_path = onnx_dir / "c2s_onnx_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"ONNX manifest not found: {manifest_path}")

        self._onnx_manifest = json.loads(manifest_path.read_text())
        log.info(f"Loading {len(self._onnx_manifest)} ONNX models from {onnx_dir} …")

        for entry in self._onnx_manifest:
            role = entry["role"]
            fp   = onnx_dir / entry["file"]
            if not fp.exists():
                log.warning(f"  [SKIP] {role}: {fp} not found")
                continue
            t0  = time.perf_counter()
            sess = ort.InferenceSession(str(fp), providers=providers)
            self._ort_sessions[role] = (sess, entry)
            self.imgsz = entry["input"]["shape"][2]
            log.info(f"  ✓ {role:<26} in {(time.perf_counter()-t0)*1000:.0f} ms")

        self.backend = "onnx"
        self._loaded = True
        return self

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> Union[torch.Tensor, np.ndarray]:
        """BGR uint8 HxWx3  →  normalised float32 1x3xHxW (torch or numpy)."""
        img = cv2.resize(frame, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis]       # (1,3,H,W)
        return img

    # ── NMS (pure numpy, no dependency) ──────────────────────────────────────

    @staticmethod
    def _nms(
        boxes:  np.ndarray,   # (N, 4)  xyxy
        scores: np.ndarray,   # (N,)
        iou_thr: float = 0.45,
    ) -> List[int]:
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep  = []
        while order.size:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
            iou   = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(iou <= iou_thr)[0] + 1]
        return keep

    # ── Main detect ──────────────────────────────────────────────────────────

    def detect(
        self,
        frame:    np.ndarray,
        conf:     float = 0.25,
        iou_thr:  float = 0.45,
    ) -> List[Dict]:
        """
        Run ensemble inference on a single BGR frame.

        Returns
        -------
        List[Dict] with keys:
          role, raw_name, category, confidence, bbox (xyxy, pixel coords)
        """
        if not self._loaded:
            raise RuntimeError("Detector not loaded. Call from_pt() or from_onnx() first.")

        h, w = frame.shape[:2]
        img_np = self._preprocess(frame)       # float32 (1,3,H,W)

        candidates: List[Dict] = []

        # ── PyTorch backend ──────────────────────────────────────────────────
        if self.backend == "pt":
            img_t = torch.from_numpy(img_np).to(self.device)
            batch = self._pt_model(img_t, conf_threshold=conf, iou_threshold=iou_thr)
            for det in batch[0]:
                # bbox is in [0,imgsz] – scale to original
                bx = [
                    det["bbox"][0] / self.imgsz * w,
                    det["bbox"][1] / self.imgsz * h,
                    det["bbox"][2] / self.imgsz * w,
                    det["bbox"][3] / self.imgsz * h,
                ]
                candidates.append({**det, "bbox": bx})

        # ── ONNX Runtime backend ─────────────────────────────────────────────
        elif self.backend == "onnx":
            for role, (sess, entry) in self._ort_sessions.items():
                try:
                    input_name = sess.get_inputs()[0].name
                    outputs = sess.run(None, {input_name: img_np})
                    raw = outputs[0][0]            # (num_dets, 4+nc)
                    if raw is None or len(raw) == 0:
                        continue

                    cmap     = entry["category_map"]
                    names    = entry["class_names"]
                    nc       = entry["num_classes"]
                    class_scores = raw[:, 4:]      # (N, nc)
                    cls_ids  = class_scores.argmax(axis=1)
                    confs    = class_scores.max(axis=1)
                    boxes    = raw[:, :4]          # cx,cy,w,h (YOLO output)

                    # Convert cx,cy,w,h → xyxy (pixel)
                    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    sx, sy = w / self.imgsz, h / self.imgsz
                    x1 = (cx - bw / 2) * sx
                    y1 = (cy - bh / 2) * sy
                    x2 = (cx + bw / 2) * sx
                    y2 = (cy + bh / 2) * sy

                    for i, (c, ci) in enumerate(zip(confs, cls_ids)):
                        if float(c) < conf:
                            continue
                        raw_name = names.get(str(int(ci)), "unknown")
                        rn_lower = raw_name.lower()
                        # Resolve category
                        cat = cmap.get("*", "general_waste")
                        for key, val in cmap.items():
                            if key != "*" and key in rn_lower:
                                cat = val
                                break
                        if rn_lower in cmap:
                            cat = cmap[rn_lower]
                        candidates.append({
                            "role":       role,
                            "raw_name":   raw_name,
                            "category":   cat,
                            "confidence": float(c),
                            "bbox":       [float(x1[i]), float(y1[i]),
                                          float(x2[i]), float(y2[i])],
                        })
                except Exception as exc:
                    log.debug(f"[ONNX {role}] inference error: {exc}")

        # ── Global NMS across all candidates ─────────────────────────────────
        if candidates:
            boxes_arr  = np.array([d["bbox"] for d in candidates], dtype=np.float32)
            scores_arr = np.array([d["confidence"] for d in candidates], dtype=np.float32)
            keep = self._nms(boxes_arr, scores_arr, iou_thr)
            candidates = [candidates[i] for i in keep]

        return candidates

    # ── Visualisation ─────────────────────────────────────────────────────────

    def draw(
        self,
        frame:   np.ndarray,
        results: List[Dict],
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes and category labels on a copy of frame."""
        out = frame.copy()
        for det in results:
            cat  = det["category"]
            conf = det["confidence"]
            x1, y1, x2, y2 = (int(v) for v in det["bbox"])
            color = CATEGORY_COLORS.get(cat, (200, 200, 200))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            label = f"{cat} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - lh - 6), (x1 + lw + 2, y1), color, -1)
            cv2.putText(out, label, (x1 + 1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    # ── Info ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        if self.backend == "pt" and self._pt_model:
            return f"UnifiedWasteDetector(pt, models={len(self._pt_model.model_roles)})"
        if self.backend == "onnx":
            return f"UnifiedWasteDetector(onnx, sessions={len(self._ort_sessions)})"
        return "UnifiedWasteDetector(not loaded)"
