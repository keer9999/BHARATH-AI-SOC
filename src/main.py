#!/usr/bin/env python3
"""
Multimodal AI Waste Identification & Segregation Assistant
Bharat AI-SoC Challenge 2026–27 — Problem Statement 2

Pipeline:
  1. Capture frames from camera / image
  2. Modality 1 (Vision): YOLOv8 detects and segregates waste (Plastic, Glass, Metal, Paper, Organic)
  3. Modality 2 (Language): Llama 3.2 / WasteAdvisor provides disposal recommendations, recycling guidelines, and upcycling ideas
  4. Store segregation events in SQLite database
  5. Display HUD with segregation bin recommendations

Usage:
    python src/main.py                           # Live webcam segregation
    python src/main.py --image path/to/image.jpg # Segregate waste in a single image
    python src/main.py --test --headless         # Self-test mode (10 frames)
    python src/main.py --headless                # Headless mode (no GUI window)
"""

import argparse
import logging
import os
import sys
import time
import signal
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from config          import DEFAULT_CONFIG, SystemConfig
from waste_detector  import WasteDetector
from waste_advisor   import WasteAdvisor
from webcam_capture  import WebcamCapture
from database        import WasteDatabase
from report_generator import ReportGenerator

# ─── Bin Mapping ────────────────────────────────────────────────────────────────

BIN_MAPPING = {
    "plastic":   {"bin": "BLUE BIN (Dry Recyclable — Plastic)",     "color": (255, 140, 0)},
    "glass":     {"bin": "BLUE BIN (Dry Recyclable — Glass)",       "color": (255, 200, 0)},
    "metal":     {"bin": "BLUE BIN (Dry Recyclable — Metal)",       "color": (200, 200, 200)},
    "paper":     {"bin": "BLUE BIN (Dry Recyclable — Paper)",       "color": (240, 240, 100)},
    "cardboard": {"bin": "BLUE BIN (Dry Recyclable — Cardboard)",   "color": (200, 180, 120)},
    "organic":   {"bin": "GREEN BIN (Wet / Compostable Waste)",     "color": (50, 205, 50)},
    "hazardous": {"bin": "RED BIN (Hazardous / E-Waste)",           "color": (0, 0, 255)},
    "unknown":   {"bin": "BLACK BIN (General Non-Recyclable)",      "color": (128, 128, 128)},
}

def get_bin_info(waste_type: str) -> dict:
    key = waste_type.lower()
    for k, v in BIN_MAPPING.items():
        if k in key:
            return v
    return BIN_MAPPING["unknown"]


# ─── Logging ────────────────────────────────────────────────────────────────────

def setup_logging(config: SystemConfig):
    os.makedirs(os.path.dirname(config.log_file) or ".", exist_ok=True)

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            payload = {
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            return json.dumps(payload)

    file_handler = logging.FileHandler(config.log_file)
    file_handler.setFormatter(JsonFormatter())

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        handlers=[
            file_handler,
            console_handler,
        ]
    )

logger = logging.getLogger(__name__)


# ─── Graceful Shutdown ───────────────────────────────────────────────────────────

running = True

def signal_handler(sig, frame):
    global running
    logger.info("Shutdown signal received — finishing current frame")
    running = False

signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def set_fullscreen(cv2, window_name: str, enabled: bool) -> None:
    """Apply fullscreen immediately so the display manager can resize the window."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if enabled:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow(window_name, 1920, 1080)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
    cv2.waitKey(1)


# ─── Image Mode ─────────────────────────────────────────────────────────────────

def run_image_mode(config: SystemConfig, image_path: str) -> bool:
    """Analyse one image, classify waste, and provide multimodal disposal advisory."""
    import cv2
    detector = WasteDetector(config.detection)
    if not detector.load_model():
        logger.error("Failed to load YOLO model.")
        return False

    advisor = WasteAdvisor()
    db = WasteDatabase(config.database.db_path)

    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not read image: {image_path}")
        return False

    logger.info(f"Analyzing image: {image_path}")
    waste_frame = detector.detect(image, frame_id=0)

    print("\n" + "=" * 65)
    print(" ◈  ECOSORT AI — MULTIMODAL PERCEPTION & SEGREGATION REPORT")
    print("=" * 65)

    if not waste_frame.detections:
        print("  ℹ️  No waste objects detected in this image.")
    else:
        print(f"  Found {len(waste_frame.detections)} waste item(s):\n")
        for i, det in enumerate(waste_frame.detections, 1):
            bin_info = get_bin_info(det.class_name)
            advice = advisor.get_detection_advice(
                det.class_name,
                det.confidence,
                bbox=list(det.bbox),
                detected_object=det.detected_object,
            )

            print(f"[{i}] OBJECT: {det.class_name.upper()} ({det.confidence:.1%})")
            print(f"    SEGREGATION: {bin_info['bin']}")
            print("-" * 65)
            print(advice.get("message", "No advice available."))
            print("=" * 65)

            # Store in database
            db.insert_waste_record({
                "timestamp": datetime.now().isoformat(),
                "waste_type": det.class_name,
                "ai_confidence": det.confidence,
                "ai_class": det.detected_object or det.class_name,
                "vehicle_id": config.vehicle.vehicle_id,
            })

    return True


# ─── Main Video / Stream Pipeline ───────────────────────────────────────────────

def run_system(
    config:   SystemConfig,
    headless: bool = False,
    test_mode: bool = False,
) -> bool:
    """Main system loop: Vision Segregation + Multimodal Disposal Guidance."""
    global running

    # ── Startup Banner ────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("  BHARAT AI-SOC CHALLENGE — MULTIMODAL WASTE SEGREGATION")
    logger.info("=" * 65)
    logger.info(f"Model Path:     {config.detection.model_path}")
    logger.info(f"Database:       {config.database.db_path}")
    logger.info(f"Execution Mode: {'Test (10 synthetic frames)' if test_mode else 'Live Camera'}")
    logger.info(f"Display:        {'Headless (Console only)' if headless else 'GUI Window'}")
    logger.info("=" * 65)

    # ── Initialise Components ─────────────────────────────────────────────────
    detector = WasteDetector(config.detection)
    advisor  = WasteAdvisor()
    camera   = WebcamCapture(config.camera)
    db       = WasteDatabase(config.database.db_path)

    # ── Load AI Model ─────────────────────────────────────────────────────────
    if not detector.load_model():
        if test_mode:
            logger.warning("Running test mode with fallback detector")
        else:
            logger.error("❌ Failed to load YOLO model — aborting")
            return False

    # ── Open Camera ──────────────────────────────────────────────────────────
    if not test_mode:
        if not camera.open():
            logger.warning("⚠️  Webcam unavailable — switching to test mode")
            test_mode = True

    # ── GUI Window Setup (if display enabled) ─────────────────────────────────
    WIN_NAME = "Bharat AI-SoC Waste Segregation | [Q] Quit [F] Fullscreen"
    if not headless and not test_mode:
        import cv2
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, config.camera.width, config.camera.height)

    logger.info("✅ Multimodal waste segregation system ready — entering loop")

    frame_id          = 0
    total_segregated  = {}
    last_report_time  = time.time()
    report_interval   = config.reports.report_interval_minutes * 60

    try:
        while running:
            loop_start = time.time()

            # ── 1. Capture frame ──────────────────────────────────────────────
            if test_mode:
                try:
                    import numpy as np
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                except ImportError:
                    frame = None
            else:
                ret, frame = camera.read_frame()
                if not ret or frame is None:
                    if not headless:
                        import cv2
                        key = cv2.waitKey(30) & 0xFF
                        if key == ord("q") or key == ord("Q"):
                            break
                    time.sleep(0.05)
                    continue

            # ── 2. Modality 1: Vision Waste Identification ────────────────────
            waste_frame = detector.detect(frame, frame_id)
            frame_id += 1

            # ── 3. Modality 2: Disposal Advisory & Segregation ────────────────
            for det in waste_frame.detections:
                bin_info = get_bin_info(det.class_name)
                total_segregated[det.class_name] = total_segregated.get(det.class_name, 0) + 1

                # Store record in SQLite database
                db.insert_waste_record({
                    "timestamp": datetime.now().isoformat(),
                    "waste_type": det.class_name,
                    "ai_confidence": det.confidence,
                    "ai_class": det.detected_object or det.class_name,
                    "vehicle_id": config.vehicle.vehicle_id,
                })

            if waste_frame.detections:
                det_summary = ", ".join(
                    f"{d.class_name.upper()} ({d.confidence:.0%}) -> {get_bin_info(d.class_name)['bin'].split(' ')[0]}"
                    for d in waste_frame.detections
                )
                logger.info(f"Frame #{frame_id:04d} | Segregated: {det_summary}")
            elif frame_id % 30 == 0:
                logger.info(f"Frame #{frame_id:04d} | Scanning for waste items...")

            # ── 4. GUI Rendering ──────────────────────────────────────────────
            if not headless and not test_mode and frame is not None:
                import cv2
                annotated = detector.draw_detections(frame, waste_frame)
                
                # Overlay segregation instruction bar
                h, w = annotated.shape[:2]
                cv2.rectangle(annotated, (0, h - 45), (w, h), (20, 20, 20), -1)
                
                if waste_frame.detections:
                    top_det = max(waste_frame.detections, key=lambda d: d.confidence)
                    top_bin = get_bin_info(top_det.class_name)
                    text = f"ACTION: Put in {top_bin['bin']} | {top_det.class_name.upper()} ({top_det.confidence:.0%})"
                    cv2.putText(annotated, text, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, top_bin["color"], 2)
                else:
                    cv2.putText(annotated, "STATUS: Place waste item in camera view for segregation guidance",
                                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

                cv2.imshow(WIN_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    break

            # ── Test mode limit ───────────────────────────────────────────────
            if test_mode and frame_id >= 10:
                logger.info(f"Test mode complete after {frame_id} frames")
                break

            # Frame rate control
            elapsed    = time.time() - loop_start
            sleep_time = max(0.0, (1.0 / config.camera.fps) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        logger.error(f"Main loop error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        camera.stop()
        if not headless and not test_mode:
            import cv2
            cv2.destroyAllWindows()

        # Session summary
        det_stats = detector.get_statistics()
        logger.info("=" * 65)
        logger.info("  SESSION SEGREGATION SUMMARY")
        logger.info("=" * 65)
        logger.info(f"Total Frames Processed: {det_stats['total_frames']}")
        logger.info(f"Avg Inference Time:     {det_stats['avg_inference_ms']:.1f} ms/frame")
        logger.info(f"Segregated Categories:  {total_segregated}")
        logger.info(f"Database Stored At:     {config.database.db_path}")
        logger.info("=" * 65)

    return True


# ─── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="◈ EcoSort AI — Multimodal Waste Perception & Segregation Assistant (Bharat AI-SoC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                           # Live webcam waste segregation
  python src/main.py --image photo.jpg         # Segregate waste in a single image
  python src/main.py --test --headless         # 10-frame self-test without camera
  python src/main.py --model models/best.pt    # Custom weights override
        """
    )
    parser.add_argument("--test",     action="store_true",
                        help="Self-test mode (10 frames, no camera required)")
    parser.add_argument("--headless", action="store_true",
                        help="Run in headless mode without GUI window")
    parser.add_argument("--image",    type=str,
                        help="Identify and segregate waste from a saved photo")
    parser.add_argument("--model",    type=str,
                        help="Override YOLO model weights path (.pt)")
    args = parser.parse_args()

    config = DEFAULT_CONFIG

    if args.model:
        config.detection.custom_model_path = args.model
        config.detection.model_path = args.model

    setup_logging(config)

    if args.image:
        success = run_image_mode(config, args.image)
        sys.exit(0 if success else 1)

    success = run_system(config, headless=args.headless, test_mode=args.test)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
