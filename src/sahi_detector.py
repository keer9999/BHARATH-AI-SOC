#!/usr/bin/env python3
"""
src/sahi_detector.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bharat AI-SoC  —  SAHI-Enhanced Waste Detector
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAHI = Slicing Aided Hyper Inference
  — Divides the input frame into overlapping tiles
  — Runs YOLO on each tile independently
  — Merges tile predictions back to original image coordinates
  — Applies global weighted-box-fusion NMS to remove duplicates
  — Plus multi-scale full-image inference for large objects

Also implements two bonus improvements:
  • Adaptive tiling  — tile count / overlap tuned to frame resolution
  • Raspberry Pi 4B profile — 320px tiles, single-thread, 2-FPS throttle

Architecture Pipeline
─────────────────────
  Camera frame (BGR)
       │
       ├─► Full-image inference (512px) ──────────────────────┐
       │                                                       │
       └─► SAHI tile slicing (N overlapping tiles)            │
                │                                             │
                ├─► Tile 0 → YOLO predict                    │
                ├─► Tile 1 → YOLO predict                    │
                ├─► Tile 2 → YOLO predict   (coordinates     │
                │   ...        mapped back   to full frame)   │
                └─► Tile N → YOLO predict                    │
                                                             │
       ┌─────────────────────────────────────────────────────┘
       │  Merge all candidate boxes
       │  Weighted-Box-Fusion (WBF) or Soft-NMS
       └─► Final WasteFrame (bounding boxes, classes, confidence)

Usage
─────
  from sahi_detector import SAHIWasteDetector, SAHIConfig
  from waste_detector import WasteDetector, DetectionConfig

  base   = WasteDetector(DetectionConfig())
  base.load_model()
  sahi   = SAHIWasteDetector(base, SAHIConfig.raspberry_pi_4b())
  result = sahi.detect(frame)

Raspberry Pi 4B Profile
  SAHIConfig.raspberry_pi_4b()
  — tile_size=320, overlap_ratio=0.25, max_tiles=4, full_image_size=320
  — single-threaded, 500 ms minimum inter-frame gap
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    _CV2_OK = True
except ImportError:
    cv2 = None
    _CV2_OK = False

from waste_detector import WasteDetection, WasteDetector, WasteFrame

log = logging.getLogger(__name__)

# ─── SAHI Config ─────────────────────────────────────────────────────────────

@dataclass
class SAHIConfig:
    """
    Configuration for the SAHI slicing pipeline.

    Parameters
    ----------
    tile_size:
        Width AND height of each square tile in pixels.
        Smaller → catches smaller objects, more tiles to process.
        320 is optimal for Raspberry Pi 4B.
    overlap_ratio:
        Fraction of tile_size used as overlap between adjacent tiles
        (0.25 = 25% overlap). Prevents objects at tile edges being cut off.
    full_image_size:
        Resolution for the full-image inference pass (catches large objects).
        Set to 0 to skip full-image pass.
    max_tiles:
        Hard upper limit on tiles processed per frame (RPi safety cap).
    nms_iou_threshold:
        IoU threshold for the final cross-tile NMS / WBF merge.
    conf_threshold:
        Minimum confidence for a tile detection to be kept.
    use_weighted_box_fusion:
        True → Weighted Box Fusion (smoother, better for crowded scenes).
        False → standard hard NMS.
    min_frame_ms:
        Minimum inter-frame delay in milliseconds.
        500 ms ≈ 2 FPS — safe for Raspberry Pi 4B core temperature.
    enable_multiscale:
        Run a second full-image pass at 0.5× scale for tiny objects.
    """

    tile_size:                int   = 416
    overlap_ratio:            float = 0.30
    full_image_size:          int   = 640
    max_tiles:                int   = 9
    nms_iou_threshold:        float = 0.45
    conf_threshold:           float = 0.25
    use_weighted_box_fusion:  bool  = True
    min_frame_ms:             float = 0.0        # 0 = no throttle
    enable_multiscale:        bool  = True

    # ── Hardware profiles ────────────────────────────────────────────────────

    @classmethod
    def raspberry_pi_4b(cls) -> "SAHIConfig":
        """
        Optimised for Raspberry Pi 4B (ARM Cortex-A72, 4 GB LPDDR4).
        • 320 px tiles — fits in L2 cache, fast OpenCV resize
        • 4 tile max — stays under 2 seconds total per frame
        • 500 ms min gap — ~2 FPS, keeps CPU core temp safe
        • No multi-scale — saves CPU cycles
        """
        return cls(
            tile_size=320,
            overlap_ratio=0.25,
            full_image_size=320,
            max_tiles=4,
            nms_iou_threshold=0.45,
            conf_threshold=0.28,
            use_weighted_box_fusion=False,   # hard NMS faster on ARM
            min_frame_ms=500.0,
            enable_multiscale=False,
        )

    @classmethod
    def laptop_gpu(cls) -> "SAHIConfig":
        """Full-precision profile for a laptop/desktop GPU."""
        return cls(
            tile_size=640,
            overlap_ratio=0.30,
            full_image_size=1280,
            max_tiles=16,
            nms_iou_threshold=0.40,
            conf_threshold=0.25,
            use_weighted_box_fusion=True,
            min_frame_ms=0.0,
            enable_multiscale=True,
        )

    @classmethod
    def jetson_nano(cls) -> "SAHIConfig":
        """Jetson Nano (GPU FP16) profile."""
        return cls(
            tile_size=416,
            overlap_ratio=0.25,
            full_image_size=640,
            max_tiles=9,
            nms_iou_threshold=0.45,
            conf_threshold=0.25,
            use_weighted_box_fusion=True,
            min_frame_ms=0.0,
            enable_multiscale=True,
        )


# ─── Tile Geometry ────────────────────────────────────────────────────────────

@dataclass
class Tile:
    """Single slice of the input frame."""
    x1: int; y1: int; x2: int; y2: int    # pixel coords in ORIGINAL frame
    img: np.ndarray                         # cropped BGR image of this tile

    @property
    def w(self) -> int: return self.x2 - self.x1

    @property
    def h(self) -> int: return self.y2 - self.y1


def compute_tiles(
    frame_w: int,
    frame_h: int,
    tile_size: int,
    overlap_ratio: float,
    max_tiles: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Compute (x1, y1, x2, y2) pixel coords for all overlapping tiles.

    Strategy:
    1. Compute stride = tile_size × (1 - overlap_ratio)
    2. Enumerate a grid of tiles covering the full image
    3. Clip to image boundaries — edge tiles may be smaller than tile_size
       but are padded internally before YOLO inference
    4. Cap at max_tiles using centre-bias selection
    """
    stride = max(1, int(tile_size * (1.0 - overlap_ratio)))

    xs = list(range(0, frame_w, stride))
    ys = list(range(0, frame_h, stride))

    tiles: List[Tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            x2 = min(x + tile_size, frame_w)
            y2 = min(y + tile_size, frame_h)
            # Skip tiny slivers at the far edge (< 25% of tile_size)
            if (x2 - x) < tile_size * 0.25 or (y2 - y) < tile_size * 0.25:
                continue
            tiles.append((x, y, x2, y2))

    # If over max_tiles, select tiles biased towards the image centre
    if len(tiles) > max_tiles:
        cx, cy = frame_w / 2.0, frame_h / 2.0

        def _dist_to_centre(t: Tuple[int, int, int, int]) -> float:
            mx = (t[0] + t[2]) / 2.0
            my = (t[1] + t[3]) / 2.0
            return math.hypot(mx - cx, my - cy)

        tiles = sorted(tiles, key=_dist_to_centre)[:max_tiles]

    return tiles


def pad_to_square(img: np.ndarray, target: int) -> np.ndarray:
    """
    Resize an image to (target, target) with letterboxing (black padding).
    Preserves aspect ratio for accurate YOLO anchor matching.
    """
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    if _CV2_OK:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target, target, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
    else:
        canvas = img[:target, :target]
    return canvas


# ─── Weighted Box Fusion (WBF) ────────────────────────────────────────────────

def weighted_box_fusion(
    boxes:  np.ndarray,  # (N, 4) xyxy normalised [0,1]
    scores: np.ndarray,  # (N,)
    labels: np.ndarray,  # (N,) int
    iou_thr: float = 0.45,
    skip_box_thr: float = 0.0001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pure-numpy Weighted Box Fusion implementation.

    WBF is superior to NMS for multi-model / multi-tile ensembles:
    — NMS discards low-confidence boxes entirely → misses detections
    — WBF weights all overlapping boxes by confidence → better bounding-box
      position and higher recall in crowded scenes.

    Reference: Solovyev et al. (2021), arXiv:1910.13302
    """
    if len(boxes) == 0:
        return (np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int))

    # Sort by confidence descending
    order = np.argsort(-scores)
    boxes  = boxes[order]
    scores = scores[order]
    labels = labels[order]

    clusters: List[Dict] = []

    for i in range(len(boxes)):
        b, s, l = boxes[i], scores[i], labels[i]
        if s < skip_box_thr:
            continue
        matched = False
        for cl in clusters:
            if cl["label"] != l:
                continue
            # Compute IoU with cluster representative box
            rep = cl["fused_box"]
            xA = max(rep[0], b[0]); yA = max(rep[1], b[1])
            xB = min(rep[2], b[2]); yB = min(rep[3], b[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            area1 = (rep[2]-rep[0]) * (rep[3]-rep[1])
            area2 = (b[2]-b[0]) * (b[3]-b[1])
            union = area1 + area2 - inter
            iou   = inter / union if union > 0 else 0.0
            if iou >= iou_thr:
                # Merge into cluster with score-weighted average
                cl["boxes"].append(b); cl["scores"].append(s)
                total_s = sum(cl["scores"])
                cl["fused_box"] = np.average(
                    cl["boxes"], axis=0, weights=cl["scores"]
                )
                cl["fused_score"] = total_s / len(cl["scores"])
                matched = True
                break
        if not matched:
            clusters.append({
                "boxes":       [b],
                "scores":      [s],
                "fused_box":   b.copy(),
                "fused_score": s,
                "label":       l,
            })

    if not clusters:
        return (np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int))

    out_boxes  = np.array([c["fused_box"]   for c in clusters])
    out_scores = np.array([c["fused_score"] for c in clusters])
    out_labels = np.array([c["label"]       for c in clusters], dtype=int)
    return out_boxes, out_scores, out_labels


def hard_nms(
    boxes:  np.ndarray,  # (N, 4) xyxy
    scores: np.ndarray,  # (N,)
    iou_thr: float = 0.45,
) -> List[int]:
    """Standard hard NMS, pure numpy — no dependency on torchvision."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas  = (x2-x1+1) * (y2-y1+1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size:
        i = order[0]; keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1+1) * np.maximum(0, yy2-yy1+1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thr)[0] + 1]
    return keep


# ─── SAHIWasteDetector ────────────────────────────────────────────────────────

class SAHIWasteDetector:
    """
    SAHI-enhanced wrapper around WasteDetector.

    Drop-in replacement: same detect(frame) → WasteFrame interface.
    Preserves all existing colour rendering, statistics, and WasteAdvisor.
    """

    def __init__(
        self,
        base_detector: WasteDetector,
        config: Optional[SAHIConfig] = None,
    ):
        self.det    = base_detector
        self.cfg    = config or SAHIConfig()
        self._last_frame_t: float = 0.0
        self._frame_count:  int   = 0
        self._total_tile_calls: int = 0

        log.info(
            f"SAHIWasteDetector ready — tile={self.cfg.tile_size}px "
            f"overlap={self.cfg.overlap_ratio:.0%} "
            f"max_tiles={self.cfg.max_tiles} "
            f"wbf={self.cfg.use_weighted_box_fusion} "
            f"min_frame={self.cfg.min_frame_ms:.0f}ms"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> WasteFrame:
        """
        Run SAHI-enhanced detection on a BGR camera frame.

        Steps
        ─────
        1. Throttle if min_frame_ms > 0 (Raspberry Pi 4B protection)
        2. Run full-image inference at full_image_size (large objects)
        3. Optionally run full-image at 0.5× scale (tiny objects)
        4. Compute overlapping tile grid
        5. Run YOLO on each tile, map coordinates back to full frame
        6. Merge all candidates through WBF or hard NMS
        7. Build WasteFrame with deduplicated detections
        """
        # 1. RPi throttle
        if self.cfg.min_frame_ms > 0:
            elapsed = (time.time() - self._last_frame_t) * 1000
            if elapsed < self.cfg.min_frame_ms:
                time.sleep((self.cfg.min_frame_ms - elapsed) / 1000.0)
        t_start = time.time()
        self._last_frame_t = t_start

        h, w = frame.shape[:2]
        all_boxes:  List[np.ndarray] = []   # each (4,) xyxy pixel
        all_scores: List[float]      = []
        all_labels: List[int]        = []   # index into self.det custom_class_names or WASTE_CLASS_NAMES
        all_metas:  List[Dict]       = []   # {class_name, detected_object, source}

        # 2 & 3. Full-image passes (catches large objects)
        if self.cfg.full_image_size > 0:
            self._run_full_image(
                frame, w, h,
                self.cfg.full_image_size,
                all_boxes, all_scores, all_labels, all_metas,
                source="full",
            )
            if self.cfg.enable_multiscale:
                self._run_full_image(
                    frame, w, h,
                    max(160, self.cfg.full_image_size // 2),
                    all_boxes, all_scores, all_labels, all_metas,
                    source="half",
                )

        # 4. Tile slicing
        tile_coords = compute_tiles(w, h, self.cfg.tile_size, self.cfg.overlap_ratio, self.cfg.max_tiles)
        log.debug(f"SAHI: frame {w}×{h} → {len(tile_coords)} tiles")

        # 5. Per-tile inference
        for (tx1, ty1, tx2, ty2) in tile_coords:
            tile_img = frame[ty1:ty2, tx1:tx2]
            if tile_img.size == 0:
                continue
            self._run_tile(
                tile_img, tx1, ty1, tx2, ty2, w, h,
                all_boxes, all_scores, all_labels, all_metas,
            )
            self._total_tile_calls += 1

        # 6. Merge via WBF or NMS
        final_dets = self._merge(all_boxes, all_scores, all_labels, all_metas, w, h)

        # 7. Build WasteFrame
        waste_frame = WasteFrame(
            frame_id=frame_id,
            timestamp=t_start,
            image_shape=frame.shape,
        )
        waste_frame.detections = final_dets
        waste_frame.inference_time_ms = (time.time() - t_start) * 1000

        self._frame_count += 1
        log.debug(
            f"SAHI frame {frame_id}: {len(final_dets)} detections "
            f"({len(tile_coords)} tiles, {waste_frame.inference_time_ms:.0f} ms)"
        )
        return waste_frame

    # ── Convenience wrappers for drawing (delegate to base detector) ──────────

    def draw_detections(self, frame: np.ndarray, waste_frame: WasteFrame) -> np.ndarray:
        return self.det.draw_detections(frame, waste_frame)

    def get_statistics(self) -> Dict:
        base = self.det.get_statistics() if hasattr(self.det, "get_statistics") else {}
        base["sahi"] = {
            "frames":          self._frame_count,
            "total_tile_runs": self._total_tile_calls,
            "tile_size":       self.cfg.tile_size,
            "overlap":         self.cfg.overlap_ratio,
            "wbf":             self.cfg.use_weighted_box_fusion,
        }
        return base

    # ── Internal inference helpers ────────────────────────────────────────────

    def _infer_on_image(self, img: np.ndarray, imgsz: int) -> List[Dict]:
        """
        Run the base WasteDetector's YOLO model on a single image and return
        a flat list of raw candidate dicts (before WasteFrame wrapping).

        We re-use the existing candidate-building logic from detect() by
        calling WasteDetector._run_full_candidates() — but that is private.
        Instead, we use a lighter path: call model.predict() directly on
        the image and parse boxes ourselves, delegating class-name normalisation
        to WasteDetector.normalize_material_name().
        """
        det = self.det
        candidates: List[Dict] = []

        def _yolo_predict(model, source, conf, iou, imgsz_, device):
            if model is None:
                return None
            try:
                kwargs = {
                    "source": source,
                    "conf":   conf,
                    "iou":    iou,
                    "imgsz":  imgsz_,
                    "device": device,
                    "verbose": False,
                }
                if det.config.device != "cpu" and det.config.half_precision:
                    kwargs["half"] = True
                return model.predict(**kwargs)
            except Exception as exc:
                log.debug(f"YOLO predict error: {exc}")
                return None

        conf   = self.cfg.conf_threshold
        iou    = det.config.iou_threshold
        device = det.config.device

        # Primary model
        results = _yolo_predict(det.model, img, conf, iou, imgsz, device)
        self._parse_yolo_results(results, candidates, det.model, "primary")

        # General YOLOv8n (COCO fallback)
        if det.general_model is not None and det.general_model is not det.model:
            results = _yolo_predict(det.general_model, img, max(conf-0.08, 0.1), iou, imgsz, device)
            self._parse_yolo_results(results, candidates, det.general_model, "general", only_known=True)

        # Specialist ensemble
        specialists = [
            (det.ewaste_a_model,          WasteDetector.EWASTE_A_MAP,       "ewaste_a"),
            (det.ewaste_b_model,          WasteDetector.EWASTE_B_MAP,       "ewaste_b"),
            (det.plastic_bag_model,       {
                "plastic-bag": ("plastic", "Plastic Bag / Film Packaging"),
                "plastic_bag": ("plastic", "Plastic Bag / Film Packaging"),
                "plastic bag": ("plastic", "Plastic Bag / Film Packaging"),
                "bag":         ("plastic", "Plastic Bag / Film Packaging"),
            }, "plastic_bag"),
            (det.general_waste_model,     {
                "cardboard": ("cardboard", "Cardboard / Paper Packaging"),
                "glass":     ("glass",     "Glass Item / Bottle"),
                "metal":     ("metal",     "Metal Can / Foil"),
                "paper":     ("paper",     "Paper / Document Waste"),
                "plastic":   ("plastic",   "Plastic Item"),
                "trash":     ("general_waste", "General Waste / Rubbish"),
                "garbage":   ("general_waste", "General Waste / Rubbish"),
                "waste":     ("general_waste", "General Waste Item"),
            }, "general_waste"),
            (det.plastic_containers_model, {
                "plastic bag":    ("plastic", "Plastic Bag / Film Packaging"),
                "plastic bottle": ("plastic", "Plastic Bottle / Container"),
                "plastic glass":  ("plastic", "Plastic Cup / Drinking Vessel"),
            }, "plastic_containers"),
            (det.resin_code_model,        WasteDetector.RESIN_CODE_MAP,     "resin_code"),
            (det.food_organic_model,      WasteDetector.FOOD_ORGANIC_MAP,   "food_organic"),
            (det.bottle_model,            {"bottle": ("plastic", "Plastic Bottle / Container")}, "bottle"),
        ]

        sp_conf = max(conf - 0.06, 0.1)
        for model, label_map, tag in specialists:
            if model is None:
                continue
            results = _yolo_predict(model, img, sp_conf, iou, imgsz, device)
            self._parse_yolo_specialist(results, candidates, model, label_map, tag)

        return candidates

    def _parse_yolo_results(
        self,
        results,
        candidates: List[Dict],
        model,
        source: str,
        only_known: bool = False,
    ) -> None:
        """Parse ultralytics Results object into candidate list."""
        if not (results and len(results) > 0 and results[0].boxes is not None):
            return
        res = results[0]
        for box in res.boxes:
            cls_id   = int(box.cls[0].cpu().numpy())
            raw_name = str(res.names.get(cls_id, "")).strip().lower()
            conf     = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cat = WasteDetector.normalize_material_name(raw_name)
            if only_known and cat == "unknown":
                continue
            obj_label = WasteDetector.COCO_WASTE_MAP.get(raw_name, (cat, raw_name.title()))[1] if raw_name in WasteDetector.COCO_WASTE_MAP else raw_name.title()
            candidates.append({
                "source": source, "class_id": cls_id, "raw_name": raw_name,
                "class_name": cat, "detected_object": obj_label,
                "confidence": conf,
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
            })

    def _parse_yolo_specialist(self, results, candidates: List[Dict], model, label_map: Dict, source: str) -> None:
        """Parse specialist model results using a fixed label_map."""
        if not (results and len(results) > 0 and results[0].boxes is not None):
            return
        res = results[0]
        for box in res.boxes:
            cls_id   = int(box.cls[0].cpu().numpy())
            raw      = str(res.names.get(cls_id, "")).strip().lower()
            lookup   = raw.replace(" ", "-").replace("_", "-")
            if lookup in label_map:
                cat, obj_label = label_map[lookup]
            elif raw in label_map:
                cat, obj_label = label_map[raw]
            else:
                cat = WasteDetector.normalize_material_name(raw)
                if cat != "unknown":
                    obj_label = raw.title()
                else:
                    continue
            conf    = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            candidates.append({
                "source": source, "class_id": cls_id, "raw_name": raw,
                "class_name": cat, "detected_object": obj_label,
                "confidence": conf,
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
            })

    def _run_full_image(
        self,
        frame: np.ndarray,
        fw: int, fh: int,
        imgsz: int,
        all_boxes, all_scores, all_labels, all_metas,
        source: str,
    ) -> None:
        """Full-image inference pass."""
        candidates = self._infer_on_image(frame, imgsz)
        for c in candidates:
            all_boxes.append(np.array(c["bbox"], dtype=np.float32))
            all_scores.append(c["confidence"])
            all_labels.append(self._label_id(c["class_name"]))
            all_metas.append({**c, "source": source})

    def _run_tile(
        self,
        tile_img: np.ndarray,
        tx1: int, ty1: int, tx2: int, ty2: int,
        fw: int, fh: int,
        all_boxes, all_scores, all_labels, all_metas,
    ) -> None:
        """
        Run inference on a single tile, then map tile-local bbox coordinates
        back to full-frame coordinates.
        """
        th, tw = tile_img.shape[:2]
        # Pad tile to square for YOLO (preserves aspect ratio)
        tile_sq = pad_to_square(tile_img, self.cfg.tile_size)

        # Scale factors from tile-padded space → original tile space
        scale = min(tw, th) / self.cfg.tile_size

        candidates = self._infer_on_image(tile_sq, self.cfg.tile_size)

        for c in candidates:
            # tile_sq box → original tile coords (undo padding/scale)
            bx1, by1, bx2, by2 = c["bbox"]
            # The pad_to_square function places the image at top-left in the
            # square canvas, so we scale back using the same factor
            pad_scale = self.cfg.tile_size / max(tw, th)
            ox1 = bx1 / pad_scale
            oy1 = by1 / pad_scale
            ox2 = bx2 / pad_scale
            oy2 = by2 / pad_scale
            # Map to full-frame coordinates
            fx1 = ox1 + tx1
            fy1 = oy1 + ty1
            fx2 = ox2 + tx1
            fy2 = oy2 + ty1
            # Clip to frame bounds
            fx1 = max(0.0, min(fx1, fw))
            fy1 = max(0.0, min(fy1, fh))
            fx2 = max(0.0, min(fx2, fw))
            fy2 = max(0.0, min(fy2, fh))

            if fx2 <= fx1 or fy2 <= fy1:
                continue

            all_boxes.append(np.array([fx1, fy1, fx2, fy2], dtype=np.float32))
            all_scores.append(c["confidence"])
            all_labels.append(self._label_id(c["class_name"]))
            all_metas.append({**c,
                               "bbox": (fx1, fy1, fx2, fy2),
                               "source": f"tile_{tx1}_{ty1}"})

    def _merge(
        self,
        all_boxes:  List[np.ndarray],
        all_scores: List[float],
        all_labels: List[int],
        all_metas:  List[Dict],
        fw: int, fh: int,
    ) -> List[WasteDetection]:
        """Apply WBF or hard NMS, then build WasteDetection objects."""
        if not all_boxes:
            return []

        boxes_arr  = np.array(all_boxes,  dtype=np.float32)
        scores_arr = np.array(all_scores, dtype=np.float32)
        labels_arr = np.array(all_labels, dtype=np.int32)

        if self.cfg.use_weighted_box_fusion:
            # Normalise boxes to [0,1] for WBF
            norm = np.array([fw, fh, fw, fh], dtype=np.float32)
            boxes_norm = np.clip(boxes_arr / norm, 0.0, 1.0)
            fused_b, fused_s, fused_l = weighted_box_fusion(
                boxes_norm, scores_arr, labels_arr,
                iou_thr=self.cfg.nms_iou_threshold,
                skip_box_thr=self.cfg.conf_threshold,
            )
            # De-normalise
            final_boxes  = fused_b * norm
            final_scores = fused_s
            final_labels = fused_l
            # Pick a meta for each fused box (spatially overlapping original with highest score)
            final_metas = []
            for fb, fl in zip(final_boxes, final_labels):
                best_m = None
                best_iou = -1.0
                best_s = -1.0
                # First pass: find overlapping candidate with same label
                for m, s, lab, b in zip(all_metas, all_scores, all_labels, boxes_arr):
                    if lab == fl:
                        # Compute IoU between fused box and original box
                        xA = max(fb[0], b[0])
                        yA = max(fb[1], b[1])
                        xB = min(fb[2], b[2])
                        yB = min(fb[3], b[3])
                        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
                        a1 = (fb[2] - fb[0]) * (fb[3] - fb[1])
                        a2 = (b[2] - b[0]) * (b[3] - b[1])
                        union = a1 + a2 - inter
                        iou = inter / union if union > 0 else 0.0
                        if iou > best_iou or (abs(iou - best_iou) < 1e-4 and s > best_s):
                            best_iou = iou
                            best_s = s
                            best_m = m
                # Fallback: if no label match, pick highest scoring overlapping
                if best_m is None:
                    for m, s, b in zip(all_metas, all_scores, boxes_arr):
                        if s > best_s:
                            best_s = s
                            best_m = m
                final_metas.append(best_m if best_m is not None else all_metas[0])
        else:
            keep = hard_nms(boxes_arr, scores_arr, self.cfg.nms_iou_threshold)
            final_boxes  = boxes_arr[keep]
            final_scores = scores_arr[keep]
            final_labels = labels_arr[keep]
            final_metas  = [all_metas[k] for k in keep]

        detections: List[WasteDetection] = []
        for box, score, meta in zip(final_boxes, final_scores, final_metas):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area = float((x2 - x1) * (y2 - y1))
            det = WasteDetection(
                class_id=meta.get("class_id", 0),
                class_name=meta.get("class_name", "unknown"),
                confidence=float(score),
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                centroid=(float(cx), float(cy)),
                area=area,
                detected_object=meta.get("detected_object", ""),
            )
            if float(score) >= self.cfg.conf_threshold:
                detections.append(det)

        return detections

    # ── Helpers ───────────────────────────────────────────────────────────────

    _LABEL_INDEX: Dict[str, int] = {
        "plastic": 0, "glass": 1, "metal": 2, "paper": 3, "cardboard": 4,
        "organic": 5, "hazardous": 6, "dry_waste": 7, "wet_waste": 8,
        "general_waste": 9, "unknown": 10,
    }

    @staticmethod
    def _label_id(class_name: str) -> int:
        return SAHIWasteDetector._LABEL_INDEX.get(class_name, 10)
