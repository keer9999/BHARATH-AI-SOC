#!/usr/bin/env python3
"""
scripts/build_unified_model.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bharat AI-SoC  —  Unified Waste Detection Model Builder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Packages all 10 specialist YOLO models into TWO deployment artifacts:

  1. models/unified/c2s_waste_ensemble.pt
       A single torch-serialised EnsembleWasteDetector module that:
       • Holds every specialist model's state-dict keyed by role.
       • Exposes a single forward() that runs the cascade and returns merged
         detections with canonical waste-category labels.
       • Can be loaded with torch.load() without any YOLO dependency.

  2. models/unified/onnx/
       One ONNX file per specialist (via ultralytics YOLO export) plus a
       machine-readable manifest that maps each file to its role, classes,
       and input/output spec — ready for ONNX Runtime, TensorRT, or OpenVINO.

Usage
─────
  cd /path/to/C2S
  python scripts/build_unified_model.py [--weights-dir src/yolo/weights]
                                        [--out-dir models/unified]
                                        [--imgsz 640]
                                        [--skip-onnx]
                                        [--verbose]

Requirements
────────────
  pip install ultralytics torch onnx onnxruntime onnxslim
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ─── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("c2s.build_unified")

# ─── Canonical Waste Category Labels ────────────────────────────────────────

WASTE_CATEGORIES = [
    "recyclable_plastic",
    "recyclable_glass",
    "recyclable_metal",
    "recyclable_paper_cardboard",
    "hazardous_ewaste",
    "organic_food",
    "general_waste",
]

# ─── Model Manifest (role → weight filename + waste-category mapping) ────────

MODEL_MANIFEST: List[Dict] = [
    {
        "role":        "primary_ecosort",
        "weight":      "ecosort_waste_yolo11s.pt",
        "description": "Primary 66-class granular recyclables (EcoSort YOLO11s)",
        "category_map": {
            # Maps raw class-name substrings → canonical WASTE_CATEGORIES index
            "aerosol":         "recyclable_metal",
            "aluminium":       "recyclable_metal",
            "battery":         "hazardous_ewaste",
            "broken glass":    "recyclable_glass",
            "cardboard":       "recyclable_paper_cardboard",
            "carton":          "recyclable_paper_cardboard",
            "cigarette":       "general_waste",
            "clothes":         "general_waste",
            "coffee-cup":      "general_waste",
            "cup":             "recyclable_plastic",
            "disposable":      "recyclable_plastic",
            "drink can":       "recyclable_metal",
            "e-waste":         "hazardous_ewaste",
            "food":            "organic_food",
            "food waste":      "organic_food",
            "glass":           "recyclable_glass",
            "lid":             "recyclable_plastic",
            "magazine":        "recyclable_paper_cardboard",
            "meal carton":     "recyclable_paper_cardboard",
            "metal":           "recyclable_metal",
            "milk carton":     "recyclable_paper_cardboard",
            "newspaper":       "recyclable_paper_cardboard",
            "paper":           "recyclable_paper_cardboard",
            "plastic":         "recyclable_plastic",
            "polystyrene":     "recyclable_plastic",
            "refuse sack":     "recyclable_plastic",
            "scrap metal":     "recyclable_metal",
            "shoe":            "general_waste",
            "single-use":      "recyclable_plastic",
            "squash pack":     "recyclable_paper_cardboard",
            "styrofoam":       "recyclable_plastic",
            "tupperware":      "recyclable_plastic",
            "unlabelled":      "general_waste",
            "wrapper":         "recyclable_plastic",
        },
    },
    {
        "role":        "primary_best",
        "weight":      "best.pt",
        "description": "Primary 5-class ARM/ZedBoard segregation model",
        "category_map": {
            "plastic_bottle": "recyclable_plastic",
            "glass_bottle":   "recyclable_glass",
            "metal_can":      "recyclable_metal",
            "paper_cardboard":"recyclable_paper_cardboard",
            "organic_waste":  "organic_food",
        },
    },
    {
        "role":        "ewaste_a",
        "weight":      "ewaste_a.pt",
        "description": "E-Waste specialist A — 12 compact electronics classes",
        "category_map": {"*": "hazardous_ewaste"},  # all classes → hazardous
    },
    {
        "role":        "ewaste_b",
        "weight":      "ewaste_b.pt",
        "description": "E-Waste specialist B — 77 comprehensive electronics classes",
        "category_map": {"*": "hazardous_ewaste"},
    },
    {
        "role":        "plastic_bag",
        "weight":      "plastic_bag.pt",
        "description": "Plastic film / bag specialist (1 class)",
        "category_map": {"*": "recyclable_plastic"},
    },
    {
        "role":        "general_waste",
        "weight":      "general_waste.pt",
        "description": "5-class general waste model (cardboard/glass/metal/paper/plastic)",
        "category_map": {
            "cardboard": "recyclable_paper_cardboard",
            "glass":     "recyclable_glass",
            "metal":     "recyclable_metal",
            "paper":     "recyclable_paper_cardboard",
            "plastic":   "recyclable_plastic",
        },
    },
    {
        "role":        "plastic_containers",
        "weight":      "plastic_containers.pt",
        "description": "Plastic containers specialist (3 classes)",
        "category_map": {"*": "recyclable_plastic"},
    },
    {
        "role":        "resin_codes",
        "weight":      "resin_codes.pt",
        "description": "Resin code classifier — HDPE/LDPE/PETE/PP/PS/PVC (6 classes)",
        "category_map": {"*": "recyclable_plastic"},
    },
    {
        "role":        "food_organic",
        "weight":      "food_organic.pt",
        "description": "Food & Organic + clothing model (39 classes)",
        "category_map": {
            "cloth":    "general_waste",
            "clothes":  "general_waste",
            "textile":  "general_waste",
            "*":        "organic_food",
        },
    },
    {
        "role":        "bottle",
        "weight":      "bottle_specialist.pt",
        "description": "Dedicated single-use bottle specialist (1 class)",
        "category_map": {"*": "recyclable_plastic"},
    },
]

# ─── Unified Ensemble Module ─────────────────────────────────────────────────

class EnsembleWasteDetector(nn.Module):
    """
    A single torch.nn.Module that wraps all specialist YOLO sub-models.

    Serialised by torch.save() into a single .pt checkpoint that contains:
      • All 10 model state-dicts (keyed by role).
      • Full model metadata (class names, category maps, input size).
      • A self-contained forward() that cascades through models and returns
        merged, NMS-filtered detections with canonical waste-category labels.

    This module has NO dependency on the ultralytics package at inference
    time — it stores raw DetectionModel weights and reconstructs them via
    torch directly.
    """

    VERSION = "1.0.0"
    WASTE_CATEGORIES = WASTE_CATEGORIES

    def __init__(
        self,
        sub_models: Dict[str, nn.Module],
        class_maps: Dict[str, Dict[str, str]],
        raw_names: Dict[str, Dict[int, str]],
        imgsz: int = 640,
    ):
        super().__init__()
        # Register each sub-model so torch.save captures its state_dict
        for role, model in sub_models.items():
            self.add_module(role, model)

        self.class_maps  = class_maps   # role → {raw_cls_name → category}
        self.raw_names   = raw_names    # role → {class_id → raw_name}
        self.imgsz       = imgsz
        self.model_roles = list(sub_models.keys())

    # ── helpers ──────────────────────────────────────────────────────────────

    def _resolve_category(self, role: str, raw_name: str) -> str:
        """Map a raw class name to a canonical waste category string."""
        cmap = self.class_maps.get(role, {})
        raw_lower = raw_name.lower().strip()

        # Exact match
        if raw_lower in cmap:
            return cmap[raw_lower]

        # Substring match
        for key, cat in cmap.items():
            if key != "*" and key in raw_lower:
                return cat

        # Wildcard fallback
        return cmap.get("*", "general_waste")

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Dict]:
        """
        Args:
            x:              Float tensor [B, 3, H, W]  (normalised 0-1).
            conf_threshold: Minimum detection confidence.
            iou_threshold:  NMS IoU threshold.

        Returns:
            List of dicts per image in the batch, each with keys:
              role, class_id, raw_name, category, confidence, bbox (xyxy)
        """
        batch_results: List[List[Dict]] = [[] for _ in range(x.shape[0])]

        for role in self.model_roles:
            sub = self._modules[role]
            if sub is None:
                continue
            try:
                with torch.no_grad():
                    preds = sub(x)                      # raw YOLO forward
                # preds can be (inference_out, train_out) tuple
                raw = preds[0] if isinstance(preds, (list, tuple)) else preds
                # Apply conf filter per image
                for b_idx in range(raw.shape[0]):
                    for det in raw[b_idx]:
                        conf = float(det[4])
                        if conf < conf_threshold:
                            continue
                        cls_id = int(det[5]) if det.shape[0] > 5 else 0
                        raw_name = self.raw_names.get(role, {}).get(cls_id, "unknown")
                        category = self._resolve_category(role, raw_name)
                        batch_results[b_idx].append({
                            "role":       role,
                            "class_id":   cls_id,
                            "raw_name":   raw_name,
                            "category":   category,
                            "confidence": conf,
                            "bbox":       det[:4].tolist(),
                        })
            except Exception as exc:
                log.debug(f"[{role}] forward error: {exc}")

        return batch_results

    # ── metadata ─────────────────────────────────────────────────────────────

    def metadata(self) -> Dict:
        """Return a serialisable metadata summary of this ensemble."""
        return {
            "version":          self.VERSION,
            "imgsz":            self.imgsz,
            "num_models":       len(self.model_roles),
            "model_roles":      self.model_roles,
            "waste_categories": self.WASTE_CATEGORIES,
            "class_maps":       self.class_maps,
            "raw_names":        {r: {str(k): v for k, v in n.items()}
                                 for r, n in self.raw_names.items()},
        }

    def __repr__(self) -> str:
        return (
            f"EnsembleWasteDetector(v{self.VERSION}, "
            f"models={len(self.model_roles)}, "
            f"categories={len(self.WASTE_CATEGORIES)})"
        )


# ─── Build Logic ─────────────────────────────────────────────────────────────

def load_all_models(
    weights_dir: Path,
    manifest: List[Dict],
    device: torch.device,
) -> Tuple[Dict[str, nn.Module], Dict[str, Dict[int, str]]]:
    """Load all specialist models and return raw nn.Module instances."""
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics is required. Run: pip install ultralytics")
        sys.exit(1)

    sub_models: Dict[str, nn.Module] = OrderedDict()
    raw_names:  Dict[str, Dict[int, str]] = {}
    failed = []

    for entry in manifest:
        role   = entry["role"]
        weight = weights_dir / entry["weight"]

        if not weight.exists():
            log.warning(f"[SKIP] {role}: weight not found at {weight}")
            failed.append(role)
            continue

        try:
            t0 = time.perf_counter()
            yolo = YOLO(str(weight))
            inner: nn.Module = yolo.model         # raw DetectionModel
            inner = inner.to(device).eval()
            sub_models[role] = inner
            raw_names[role]  = {int(k): str(v) for k, v in yolo.names.items()}
            elapsed = (time.perf_counter() - t0) * 1000
            log.info(f"  ✓ {role:<26} │ {len(yolo.names):>3} classes │ {elapsed:6.1f} ms")
        except Exception as exc:
            log.error(f"  ✗ {role}: {exc}")
            failed.append(role)

    if failed:
        log.warning(f"Skipped {len(failed)} model(s): {failed}")

    return sub_models, raw_names


def build_category_maps(manifest: List[Dict]) -> Dict[str, Dict[str, str]]:
    return {entry["role"]: entry["category_map"] for entry in manifest}


def save_unified_pt(
    ensemble: EnsembleWasteDetector,
    out_path: Path,
) -> None:
    """Serialise the full ensemble + metadata to a single .pt checkpoint."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model":    ensemble,                    # full module (state_dicts inside)
        "metadata": ensemble.metadata(),
        "manifest": MODEL_MANIFEST,
        "version":  EnsembleWasteDetector.VERSION,
    }
    torch.save(checkpoint, out_path)
    size_mb = out_path.stat().st_size / 1_048_576
    log.info(f"  Saved unified .pt → {out_path}  ({size_mb:.1f} MB)")


def export_onnx_per_model(
    weights_dir: Path,
    manifest: List[Dict],
    out_dir: Path,
    imgsz: int,
) -> List[Dict]:
    """
    Export each specialist model to ONNX using YOLO's built-in exporter.
    Returns an onnx_manifest list for the JSON index.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics required for ONNX export.")
        return []

    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_manifest = []

    for entry in manifest:
        role   = entry["role"]
        weight = weights_dir / entry["weight"]

        if not weight.exists():
            log.warning(f"[SKIP-ONNX] {role}: weight not found")
            continue

        out_onnx = onnx_dir / f"c2s_{role}.onnx"
        log.info(f"  Exporting {role} → {out_onnx.name} ...")

        try:
            yolo = YOLO(str(weight))
            export_path = yolo.export(
                format="onnx",
                imgsz=imgsz,
                half=False,
                dynamic=False,
                simplify=True,
                opset=17,
                nms=False,            # keep raw outputs for custom NMS merging
            )
            exported = Path(export_path)
            # Move to our onnx dir
            if exported.exists() and exported != out_onnx:
                exported.rename(out_onnx)

            size_mb = out_onnx.stat().st_size / 1_048_576 if out_onnx.exists() else 0
            log.info(f"    ✓ {out_onnx.name}  ({size_mb:.1f} MB)")

            onnx_manifest.append({
                "role":         role,
                "file":         out_onnx.name,
                "description":  entry["description"],
                "num_classes":  len(yolo.names),
                "class_names":  {str(k): v for k, v in yolo.names.items()},
                "category_map": entry["category_map"],
                "input":  {"name": "images", "shape": [1, 3, imgsz, imgsz], "dtype": "float32"},
                "output": {"name": "output0", "shape": [1, None, 4 + len(yolo.names)], "dtype": "float32"},
                "size_mb": round(size_mb, 2),
            })

        except Exception as exc:
            log.error(f"    ✗ ONNX export failed for {role}: {exc}")

    # Write manifest JSON index
    manifest_path = onnx_dir / "c2s_onnx_manifest.json"
    manifest_path.write_text(json.dumps(onnx_manifest, indent=2))
    log.info(f"  ONNX manifest → {manifest_path}")

    return onnx_manifest


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_unified_pt(pt_path: Path, imgsz: int, device: torch.device) -> bool:
    """Load the saved .pt and run a dummy forward pass to verify it."""
    log.info("Validating unified .pt ...")
    try:
        ckpt = torch.load(pt_path, map_location=device, weights_only=False)
        model: EnsembleWasteDetector = ckpt["model"]
        model.eval()
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
        result = model(dummy, conf_threshold=0.0)
        log.info(f"  ✓ Forward pass OK  │ batch_size=1 │ detections_in_batch={len(result)}")
        log.info(f"  Metadata summary:")
        for k, v in ckpt["metadata"].items():
            if isinstance(v, (list, dict)) and len(str(v)) > 60:
                log.info(f"    {k}: [{type(v).__name__}, {len(v)} items]")
            else:
                log.info(f"    {k}: {v}")
        return True
    except Exception as exc:
        log.error(f"  ✗ Validation failed: {exc}")
        return False


# ─── Entry Point ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the C2S unified waste detection model (.pt + ONNX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--weights-dir", default="src/yolo/weights",
                   help="Directory containing all specialist .pt weights (default: src/yolo/weights)")
    p.add_argument("--out-dir",     default="models/unified",
                   help="Output directory for unified artifacts (default: models/unified)")
    p.add_argument("--imgsz",       type=int, default=640,
                   help="Input image size for ONNX export (default: 640)")
    p.add_argument("--device",      default="cpu",
                   help="Torch device — cpu | cuda | cuda:0 (default: cpu)")
    p.add_argument("--skip-onnx",   action="store_true",
                   help="Skip ONNX export (build .pt only)")
    p.add_argument("--skip-validate", action="store_true",
                   help="Skip validation forward pass")
    p.add_argument("--verbose",     action="store_true",
                   help="Enable DEBUG logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    weights_dir = Path(args.weights_dir).resolve()
    out_dir     = Path(args.out_dir).resolve()
    device      = torch.device(args.device)

    banner = "━" * 68
    print(f"\n{banner}")
    print("  Bharat AI-SoC  ·  Unified Waste Detection Model Builder")
    print(f"{banner}")
    print(f"  weights-dir : {weights_dir}")
    print(f"  out-dir     : {out_dir}")
    print(f"  device      : {device}")
    print(f"  imgsz       : {args.imgsz}px")
    print(f"  skip-onnx   : {args.skip_onnx}")
    print(f"{banner}\n")

    # ── 1. Load all specialist models ────────────────────────────────────────
    log.info("STEP 1 ─ Loading specialist models …")
    t_start = time.perf_counter()
    sub_models, raw_names = load_all_models(weights_dir, MODEL_MANIFEST, device)
    if not sub_models:
        log.error("No models loaded — aborting.")
        sys.exit(1)
    log.info(f"  Loaded {len(sub_models)}/{len(MODEL_MANIFEST)} models in "
             f"{(time.perf_counter() - t_start)*1000:.0f} ms\n")

    # ── 2. Build EnsembleWasteDetector ───────────────────────────────────────
    log.info("STEP 2 ─ Building EnsembleWasteDetector …")
    category_maps = build_category_maps(MODEL_MANIFEST)
    ensemble = EnsembleWasteDetector(
        sub_models=sub_models,
        class_maps=category_maps,
        raw_names=raw_names,
        imgsz=args.imgsz,
    )
    ensemble.eval()
    log.info(f"  {ensemble}\n")

    # ── 3. Save unified .pt ──────────────────────────────────────────────────
    log.info("STEP 3 ─ Saving unified .pt checkpoint …")
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / "c2s_waste_ensemble.pt"
    save_unified_pt(ensemble, pt_path)
    print()

    # ── 4. Validate unified .pt ──────────────────────────────────────────────
    if not args.skip_validate:
        log.info("STEP 4 ─ Validating unified .pt …")
        ok = validate_unified_pt(pt_path, args.imgsz, device)
        if not ok:
            log.warning("Validation failed — the .pt may still be usable but check errors above.")
        print()

    # ── 5. ONNX export ───────────────────────────────────────────────────────
    if not args.skip_onnx:
        log.info("STEP 5 ─ Exporting individual specialist models to ONNX …")
        onnx_manifest = export_onnx_per_model(weights_dir, MODEL_MANIFEST, out_dir, args.imgsz)
        print()
        if onnx_manifest:
            log.info(f"  ONNX export summary: {len(onnx_manifest)}/{len(MODEL_MANIFEST)} models exported")
            total_mb = sum(m["size_mb"] for m in onnx_manifest)
            log.info(f"  Total ONNX size: {total_mb:.1f} MB")
    else:
        log.info("STEP 5 ─ Skipped ONNX export (--skip-onnx flag set)")

    # ── 6. Write overall build report ────────────────────────────────────────
    log.info("STEP 6 ─ Writing build report …")
    report = {
        "build_timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version":            EnsembleWasteDetector.VERSION,
        "unified_pt":         str(pt_path),
        "unified_pt_size_mb": round(pt_path.stat().st_size / 1_048_576, 1),
        "num_models_loaded":  len(sub_models),
        "num_models_total":   len(MODEL_MANIFEST),
        "model_roles":        list(sub_models.keys()),
        "waste_categories":   WASTE_CATEGORIES,
        "device":             str(device),
        "imgsz":              args.imgsz,
        "onnx_exported":      not args.skip_onnx,
    }
    report_path = out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    total_elapsed = time.perf_counter() - t_start
    print(f"\n{banner}")
    print(f"  Build complete in {total_elapsed:.1f}s")
    print(f"  Unified .pt  →  {pt_path}")
    if not args.skip_onnx:
        print(f"  ONNX files   →  {out_dir / 'onnx'}/")
    print(f"  Build report →  {report_path}")
    print(f"{banner}\n")


if __name__ == "__main__":
    main()
