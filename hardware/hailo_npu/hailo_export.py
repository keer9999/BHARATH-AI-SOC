#!/usr/bin/env python3
"""
Hailo AI Export & Compilation Pipeline (Hailo-8 / Hailo-8L)
AI-Enabled Smart Waste Segregation & Monitoring System

Performs the two-stage export and compilation process:
  Stage 1: PyTorch Checkpoint (.pt) → Fixed-Shape ONNX (.onnx)
  Stage 2: ONNX (.onnx) → Hailo Executable Format (.hef) via Hailo Dataflow Compiler (DFC)

Usage:
  # Run both stages (Export ONNX + Compile HEF)
  python hailo_export.py --step all

  # Only export PyTorch to ONNX
  python hailo_export.py --step onnx

  # Only compile ONNX to HEF (requires hailo_sdk_client in Hailo DFC environment)
  python hailo_export.py --step hef --arch hailo8
  python hailo_export.py --step hef --arch hailo8l
"""

import argparse
import glob
import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("HailoExport")


# ─── Stage 1: Export PyTorch / Ultralytics to Fixed-Shape ONNX ─────────────────

def export_pytorch_to_onnx(
    model_path: str,
    output_onnx_path: str,
    input_size: int = 640,
    opset: int = 12,
    simplify: bool = True,
) -> str:
    """
    Export active PyTorch YOLO checkpoint to a fixed-shape ONNX model.
    Hailo Dataflow Compiler (DFC) requires static input shapes and deterministic ops.

    Args:
        model_path: Path to PyTorch .pt checkpoint (e.g. yolo/weights/weights.pt)
        output_onnx_path: Destination path for .onnx file
        input_size: Square input resolution (default 640)
        opset: ONNX operator set version (opset 11 or 12 recommended for Hailo DFC)
        simplify: Run onnx-simplifier / onnxslim to clean up graph

    Returns:
        Path to the exported ONNX file.
    """
    logger.info("=" * 65)
    logger.info("[STAGE 1] EXPORT PYTORCH CHECKPOINT TO FIXED-SHAPE ONNX")
    logger.info("=" * 65)
    logger.info(f"Source Checkpoint : {model_path}")
    logger.info(f"Target ONNX       : {output_onnx_path}")
    logger.info(f"Input Shape       : (1, 3, {input_size}, {input_size}) [Static]")
    logger.info(f"ONNX Opset        : {opset}")
    logger.info("=" * 65)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Source model weights not found at: {model_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics is required for ONNX export. Install via pip install ultralytics.")
        raise

    # Load model
    model = YOLO(model_path)

    # Export with fixed batch size (1) and fixed spatial dimensions (640x640)
    exported_file = model.export(
        format="onnx",
        imgsz=input_size,
        dynamic=False,     # CRITICAL for Hailo DFC: Hailo requires static input shapes
        simplify=simplify, # Cleans up redundant reshape/transpose ops
        opset=opset,       # Opset 12 gives maximum compatibility with Hailo DFC parser
    )

    # If ultralytics exported with a default name, copy or rename to output_onnx_path
    if os.path.abspath(exported_file) != os.path.abspath(output_onnx_path):
        import shutil
        shutil.copy2(exported_file, output_onnx_path)
        logger.info(f"Copied exported ONNX to: {output_onnx_path}")

    # Validate exported ONNX model structure
    try:
        import onnx
        onnx_model = onnx.load(output_onnx_path)
        onnx.checker.check_model(onnx_model)
        graph = onnx_model.graph
        input_info = [
            f"{i.name}: {[dim.dim_value for dim in i.type.tensor_type.shape.dim]}"
            for i in graph.input
        ]
        output_info = [
            f"{o.name}: {[dim.dim_value for dim in o.type.tensor_type.shape.dim]}"
            for o in graph.output
        ]
        logger.info(f"[SUCCESS] ONNX model verified successfully!")
        logger.info(f"   Inputs : {input_info}")
        logger.info(f"   Outputs: {output_info}")
    except Exception as e:
        logger.warning(f"ONNX graph inspection warning ({e})")

    return output_onnx_path


# ─── Model Script (.alls) Generation for Hailo Hardware Input Pipeline ─────────

def generate_hailo_model_script(
    model_script_path: str,
    input_name: str = "images",
    include_nms: bool = False,
    num_classes: int = 1,
) -> str:
    """
    Generate Hailo model script (.alls file).

    Hardware Normalization:
        Configures the Hailo on-chip input preprocessor to normalize
        UINT8 input [0, 255] to FP32 [0.0, 1.0] using: (x - mean) / std.
        Formula: (x - 0.0) / 255.0
        This allows feeding raw UINT8 BGR/RGB frames directly into the NPU
        without any CPU normalization overhead!
    """
    os.makedirs(os.path.dirname(os.path.abspath(model_script_path)), exist_ok=True)

    lines = [
        "# Hailo Model Script (.alls)",
        "# AI Smart Waste System (Hailo-8 / Hailo-8L)",
        "",
        "# 1. Hardware-accelerated input normalization:",
        "# Feed UINT8 [0, 255] directly into HailoRT. Hardware divides by 255.0 on chip.",
        f"normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])",
        "",
        "# 2. Performance & Precision Optimization settings",
        "performance_param(compiler_optimization_level=max)",
        "",
    ]

    if include_nms:
        lines.extend([
            "# 3. Optional Hailo on-chip NMS post-processing node",
            f"nms_post_process(meta_arch=yolov8, classes={num_classes})",
            "",
        ])

    content = "\n".join(lines)
    with open(model_script_path, "w") as f:
        f.write(content)

    logger.info(f"Created Hailo Model Script (.alls) at: {model_script_path}")
    return model_script_path


# ─── Calibration Dataset Preparation for INT8 Quantization ─────────────────────

def load_calibration_dataset(
    calib_data_dir: Optional[str] = None,
    num_samples: int = 64,
    input_size: int = 640,
) -> np.ndarray:
    """
    Load and preprocess calibration images for Hailo INT8 quantization.
    Hailo Dataflow Compiler requires representative images to calibrate
    weight and activation quantization scales.

    Returns:
        np.ndarray: shape (N, 640, 640, 3) in UINT8 RGB format.
    """
    import cv2
    from inference_backend import letterbox_image

    image_paths: List[str] = []

    # Priority 1: User-specified directory
    if calib_data_dir and os.path.isdir(calib_data_dir):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            image_paths.extend(glob.glob(os.path.join(calib_data_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(calib_data_dir, "**", ext), recursive=True))

    # Priority 2: Project test and dataset folders
    if not image_paths:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_dirs = [
            os.path.join(base_dir, "test_images"),
            os.path.join(base_dir, "..", "waste-detection.v1i.yolov8", "train", "images"),
            os.path.join(base_dir, "yolo", "training_data", "waste", "train", "images"),
        ]
        for cdir in candidate_dirs:
            if os.path.isdir(cdir):
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                    image_paths.extend(glob.glob(os.path.join(cdir, ext)))

    # Deduplicate and limit to num_samples
    image_paths = sorted(list(set(image_paths)))[:num_samples]

    samples: List[np.ndarray] = []
    if image_paths:
        logger.info(f"Loading {len(image_paths)} calibration images for INT8 quantization...")
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            padded, _, _ = letterbox_image(img, target_size=input_size)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            samples.append(rgb.astype(np.uint8))

    if not samples:
        logger.warning(
            f"No calibration images found in '{calib_data_dir}'. "
            "Generating synthetic calibration dataset for compilation syntax verification."
        )
        # Generate varied synthetic background frames for calibration
        for _ in range(num_samples):
            synth = np.random.randint(40, 220, (input_size, input_size, 3), dtype=np.uint8)
            samples.append(synth)

    calib_array = np.stack(samples, axis=0)  # (N, 640, 640, 3) UINT8
    logger.info(f"Prepared calibration dataset tensor: shape {calib_array.shape}, dtype={calib_array.dtype}")
    return calib_array


# ─── Stage 2: Hailo Dataflow Compiler (DFC) Compilation ────────────────────────

def compile_onnx_to_hef(
    onnx_path: str,
    output_hef_path: str,
    hw_arch: str = "hailo8",
    calib_data_dir: Optional[str] = None,
    num_calib_samples: int = 64,
    model_name: str = "waste_detector",
    model_script_path: Optional[str] = None,
) -> str:
    """
    Compile fixed-shape ONNX to Hailo Executable Format (.hef) using Hailo DFC.

    Steps:
      1. Initialize ClientRunner for target architecture ('hailo8' or 'hailo8l').
      2. Translate ONNX model to Hailo internal representation (.hn).
      3. Apply model script (.alls) for input normalization & compiler parameters.
      4. Calibrate model using quantization dataset (INT8 optimization).
      5. Compile graph to allocation stream and write final .hef binary.
    """
    logger.info("=" * 65)
    logger.info("[STAGE 2] COMPILE ONNX TO HAILO HEF BINARY")
    logger.info("=" * 65)
    logger.info(f"Target Architecture : {hw_arch.upper()} (Hailo NPU)")
    logger.info(f"Source ONNX         : {onnx_path}")
    logger.info(f"Target HEF Binary   : {output_hef_path}")
    logger.info("=" * 65)

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model file not found at: {onnx_path}")

    # Check for Hailo SDK DFC availability
    try:
        from hailo_sdk_client import ClientRunner
    except ImportError:
        logger.error(
            "\n" + "!" * 65 + "\n"
            "[INFO] hailo_sdk_client (Hailo Dataflow Compiler) is not installed on this system.\n"
            "Hailo DFC runs on an x86_64 Linux host or inside the Hailo Software Suite Docker container.\n\n"
            "To compile the generated ONNX model into a .hef binary:\n"
            "  1. Start the Hailo AI Software Suite Docker container:\n"
            "     docker run -it -v $(pwd):/workspace hailofw/hailo-ai-sw-suite:latest\n"
            "  2. Run the compiler inside the container:\n"
            f"     python module/hailo_export.py --step hef --onnx {onnx_path} --hef {output_hef_path} --arch {hw_arch}\n"
            + "!" * 65 + "\n"
        )
        raise RuntimeError("Hailo Dataflow Compiler (hailo_sdk_client) not available.")

    # 1. Prepare Model Script (.alls)
    if not model_script_path:
        base_dir = os.path.dirname(os.path.abspath(output_hef_path))
        model_script_path = os.path.join(base_dir, f"{model_name}.alls")
    generate_hailo_model_script(model_script_path, input_name="images")

    # 2. Prepare Calibration Dataset
    calib_dataset = load_calibration_dataset(
        calib_data_dir=calib_data_dir,
        num_samples=num_calib_samples,
    )

    # 3. Initialize ClientRunner
    logger.info(f"Initializing Hailo ClientRunner for {hw_arch}...")
    runner = ClientRunner(hw_arch=hw_arch)

    # 4. Translate ONNX model
    logger.info("Translating ONNX graph to Hailo internal representation...")
    runner.translate_onnx_model(
        onnx_path,
        model_name=model_name,
    )

    # 5. Load model script
    logger.info(f"Loading model script: {model_script_path}")
    runner.load_model_script(model_script_path)

    # 6. Optimize & Calibrate (INT8 Quantization)
    logger.info("Calibrating INT8 quantization with representative dataset...")
    runner.optimize(calib_dataset)

    # 7. Compile to HEF
    logger.info("Compiling neural network into Hailo-8 hardware stream...")
    hef_binary = runner.compile()

    os.makedirs(os.path.dirname(os.path.abspath(output_hef_path)), exist_ok=True)
    with open(output_hef_path, "wb") as f:
        f.write(hef_binary)

    logger.info(f"[SUCCESS] HEF compilation successful! Binary written to: {output_hef_path}")
    return output_hef_path


# ─── Full End-to-End Pipeline ──────────────────────────────────────────────────

def run_export_pipeline(
    model_path: str,
    onnx_path: str,
    hef_path: str,
    hw_arch: str = "hailo8",
    step: str = "all",
    calib_data_dir: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Run export steps: 'onnx', 'hef', or 'all'.
    """
    exported_onnx = None
    exported_hef = None

    if step in ("onnx", "all"):
        exported_onnx = export_pytorch_to_onnx(
            model_path=model_path,
            output_onnx_path=onnx_path,
        )

    if step in ("hef", "all"):
        target_onnx = exported_onnx or onnx_path
        if not os.path.exists(target_onnx):
            logger.info("Target ONNX missing. Generating ONNX first...")
            target_onnx = export_pytorch_to_onnx(model_path, onnx_path)
            exported_onnx = target_onnx

        try:
            exported_hef = compile_onnx_to_hef(
                onnx_path=target_onnx,
                output_hef_path=hef_path,
                hw_arch=hw_arch,
                calib_data_dir=calib_data_dir,
            )
        except RuntimeError as e:
            logger.warning(
                "HEF compilation skipped or could not complete on this host (Hailo DFC requires Linux/Docker). "
                f"The static ONNX file at '{target_onnx}' is ready for DFC compilation."
            )

    return exported_onnx, exported_hef


# ─── CLI Entry Point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hailo AI Export & Compilation Pipeline (Hailo-8 / Hailo-8L)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Full pipeline (PyTorch -> ONNX -> HEF)
  python hailo_export.py --step all

  # 2. Only export PyTorch checkpoint to fixed-shape ONNX
  python hailo_export.py --step onnx

  # 3. Compile ONNX to HEF for Hailo-8L (e.g. Raspberry Pi 5 AI Kit)
  python hailo_export.py --step hef --arch hailo8l --calib-dir ../waste-detection.v1i.yolov8/train/images
        """
    )
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(base_dir, "yolo", "weights", "weights.pt")
    default_onnx = os.path.join(base_dir, "yolo", "weights", "weights.onnx")
    default_hef = os.path.join(base_dir, "yolo", "weights", "weights.hef")

    parser.add_argument("--model", type=str, default=default_model,
                        help="Path to source PyTorch .pt weights")
    parser.add_argument("--onnx", type=str, default=default_onnx,
                        help="Path for exported fixed-shape ONNX model")
    parser.add_argument("--hef", type=str, default=default_hef,
                        help="Path for compiled Hailo .hef binary")
    parser.add_argument("--arch", type=str, default="hailo8", choices=["hailo8", "hailo8l"],
                        help="Hailo hardware architecture ('hailo8' or 'hailo8l')")
    parser.add_argument("--step", type=str, default="all", choices=["all", "onnx", "hef"],
                        help="Pipeline stage to run ('all', 'onnx', or 'hef')")
    parser.add_argument("--calib-dir", type=str, default=None,
                        help="Directory containing calibration images for INT8 quantization")

    args = parser.parse_args()

    try:
        run_export_pipeline(
            model_path=args.model,
            onnx_path=args.onnx,
            hef_path=args.hef,
            hw_arch=args.arch,
            step=args.step,
            calib_data_dir=args.calib_dir,
        )
    except Exception as e:
        logger.error(f"Export pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
