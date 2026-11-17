#!/usr/bin/env python3
"""
Zynq-7000 SoC (ZedBoard) Main System Entry Point
Multimodal AI Waste Identification & Segregation Assistant
Bharat AI-SoC Challenge 2026–27 — Problem Statement 2

Target Architecture:
  - Processing System: Dual-Core ARM Cortex-A9 @ 667 MHz
  - Fast-Path Control: BRAM / OCM memory structure
  - External Storage: 8 GB SD Card / USB (/mnt/ext_storage)
  - Hardware Actuation: ZedBoard 8 On-Board LEDs (LD0-LD7) indicate waste segregation bin!

Usage on ZedBoard:
    python3 main_zynq.py                   # Autonomous camera segregation
    python3 main_zynq.py --test            # 10-frame self-test without camera
    python3 main_zynq.py --image photo.jpg # Segregate waste in a single image
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime

# ── Dynamic Path Setup ────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

# External memory path (where models, database, and modules live)
EXT_DIR = os.environ.get("C2S_EXT_STORAGE", "/mnt/ext_storage")
sys.path.insert(0, EXT_DIR)
sys.path.insert(0, os.path.join(CURRENT_DIR, "..", "external"))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", "src")))

from config_zynq import DEFAULT_CONFIG, SystemConfig
from database import WasteDatabase
from webcam_capture import WebcamCapture

try:
    from waste_detector import WasteDetector
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..", "external")))
    from waste_detector import WasteDetector

try:
    from waste_advisor import WasteAdvisor
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..", "external")))
    from waste_advisor import WasteAdvisor


# ─── Bin Mapping ────────────────────────────────────────────────────────────────

BIN_MAPPING = {
    "plastic":   {"bin": "BLUE BIN (Dry Recyclable — Plastic)",   "led": 1},
    "glass":     {"bin": "BLUE BIN (Dry Recyclable — Glass)",     "led": 2},
    "metal":     {"bin": "BLUE BIN (Dry Recyclable — Metal)",     "led": 3},
    "paper":     {"bin": "BLUE BIN (Dry Recyclable — Paper)",     "led": 4},
    "cardboard": {"bin": "BLUE BIN (Dry Recyclable — Cardboard)", "led": 4},
    "organic":   {"bin": "GREEN BIN (Wet / Compostable Waste)",   "led": 5},
    "hazardous": {"bin": "RED BIN (Hazardous / E-Waste)",         "led": 6},
    "unknown":   {"bin": "BLACK BIN (General Non-Recyclable)",    "led": 7},
}

def get_bin_info(waste_type: str) -> dict:
    key = waste_type.lower()
    for k, v in BIN_MAPPING.items():
        if k in key:
            return v
    return BIN_MAPPING["unknown"]


# ─── ZedBoard Hardware LED Controller ──────────────────────────────────────────

class ZedBoardLEDs:
    """
    Controls the 8 onboard LEDs (LD0..LD7) on ZedBoard via sysfs GPIO.
    Signals segregation bin destinations to physical operators / actuators:
      - LD0: System Heartbeat / Active
      - LD1: Plastic Bin
      - LD2: Glass Bin
      - LD3: Metal Bin
      - LD4: Paper / Cardboard Bin
      - LD5: Organic / Wet Compost Bin
      - LD6: Hazardous / Special Bin
      - LD7: General / Non-Recyclable Bin
    """
    def __init__(self):
        self.available = False
        self.base = 0
        for base in [906, 960, 1016]:
            if os.path.exists(f"/sys/class/gpio/gpiochip{base}"):
                self.base = base
                self.available = True
                break

    def set_led(self, led_index: int, state: bool):
        if not self.available or led_index < 0 or led_index > 7:
            return
        try:
            pin = self.base + led_index
            path = f"/sys/class/gpio/gpio{pin}/value"
            if not os.path.exists(path):
                with open("/sys/class/gpio/export", "w") as f:
                    f.write(str(pin))
                with open(f"/sys/class/gpio/gpio{pin}/direction", "w") as f:
                    f.write("out")
            with open(path, "w") as f:
                f.write("1" if state else "0")
        except Exception:
            pass

    def indicate_segregation(self, waste_type: str):
        """Turn on the appropriate bin LED and clear others (except heartbeat)."""
        info = get_bin_info(waste_type)
        target_led = info.get("led", 7)
        for i in range(1, 8):
            self.set_led(i, i == target_led)

    def set_heartbeat(self, state: bool):
        self.set_led(0, state)

    def clear_all(self):
        for i in range(8):
            self.set_led(i, False)


# ─── Logging Setup ─────────────────────────────────────────────────────────────

def setup_logging(config: SystemConfig):
    log_file = config.log_file
    try:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    except OSError:
        log_file = os.path.join(CURRENT_DIR, "c2s_zynq.log")
        config.log_file = log_file

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            payload = {
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "node": "ZedBoard_Zynq7000_ARM_Cortex_A9",
                "message": record.getMessage(),
            }
            return json.dumps(payload)

    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JsonFormatter())
        handlers.append(file_handler)
    except Exception as e:
        print(f"Warning: Could not open log file: {e}")

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        handlers=handlers,
    )

logger = logging.getLogger(__name__)


# ─── Graceful Shutdown ─────────────────────────────────────────────────────────

running = True

def signal_handler(sig, frame):
    global running
    logger.info("Shutdown signal caught — terminating gracefully...")
    running = False

signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ─── Main Execution Pipeline ──────────────────────────────────────────────────

def run_system(config: SystemConfig, test_mode: bool = False) -> bool:
    global running

    leds = ZedBoardLEDs()
    logger.info("=" * 65)
    logger.info("  C2S SMART WASTE SYSTEM — ZYNQ-7000 (ZedBoard ARM)")
    logger.info("=" * 65)
    logger.info(f"Target Architecture: Dual-Core ARM Cortex-A9 (ZedBoard XC7Z020)")
    logger.info(f"External Memory:     8 GB attached (Weights & Storage)")
    logger.info(f"YOLO Model Path:     {config.detection.model_path}")
    logger.info(f"Database Path:       {config.database.db_path}")
    logger.info(f"Execution Mode:      {'Test (10 synthetic frames)' if test_mode else 'Live Autonomous'}")
    logger.info("=" * 65)

    # 1. Initialize Subsystems
    detector = WasteDetector(config.detection)
    advisor  = WasteAdvisor()
    camera   = WebcamCapture(config.camera)
    db       = WasteDatabase(config.database.db_path)

    # 2. Load AI Model from External 8GB Storage
    logger.info("Loading YOLOv8 weights from external memory...")
    if not detector.load_model():
        logger.warning(f"Could not load primary model at {config.detection.model_path}")
        logger.info(f"Attempting fallback model: {config.detection.fallback_model_path}")
        config.detection.model_path = config.detection.fallback_model_path
        if not detector.load_model():
            if test_mode:
                logger.warning("Running test mode with fallback detector")
            else:
                logger.error("Failed to load any YOLO model. Aborting.")
                return False

    # 3. Open Camera (or fallback to test mode if camera not yet plugged into USB OTG)
    if not test_mode:
        if not camera.open():
            logger.warning("Camera not detected on USB OTG — running in simulated camera mode")
            test_mode = True

    logger.info("✅ All ARM subsystems initialized — entering processing loop...")

    frame_id         = 0
    heartbeat_state  = False
    total_segregated = {}

    try:
        while running:
            loop_start = time.time()

            # Heartbeat LED toggle on ZedBoard LD0
            heartbeat_state = not heartbeat_state
            leds.set_heartbeat(heartbeat_state)

            # Step 1: Capture Frame
            if test_mode:
                try:
                    import numpy as np
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                except ImportError:
                    frame = None
            else:
                ret, frame = camera.read_frame()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

            # Step 2: Modality 1 — Vision Waste Identification
            waste_frame = detector.detect(frame, frame_id)
            frame_id += 1

            # Step 3: Modality 2 — Multimodal Segregation & Advisory
            if waste_frame.detections:
                for det in waste_frame.detections:
                    bin_info = get_bin_info(det.class_name)
                    total_segregated[det.class_name] = total_segregated.get(det.class_name, 0) + 1

                    # Physical LED actuation on ZedBoard (LD1-LD7)
                    leds.indicate_segregation(det.class_name)

                    # Store in database
                    db.insert_waste_record({
                        "timestamp": datetime.now().isoformat(),
                        "waste_type": det.class_name,
                        "ai_confidence": det.confidence,
                        "ai_class": det.detected_object or det.class_name,
                        "vehicle_id": "ZEDBOARD-ARM-01",
                    })

                det_summary = ", ".join(
                    f"{d.class_name.upper()} ({d.confidence:.0%}) -> {get_bin_info(d.class_name)['bin'].split(' ')[0]}"
                    for d in waste_frame.detections
                )
                logger.info(f"ZedBoard Frame #{frame_id:04d} | Segregated: {det_summary}")
            else:
                if frame_id % 30 == 0:
                    logger.info(f"ZedBoard Frame #{frame_id:04d} | Scanning for waste items...")

            # Step 4: Self-test limit
            if test_mode and frame_id >= 10:
                logger.info(f"Self-test complete ({frame_id} frames). Exiting.")
                break

            # Rate limiting for ARM Cortex-A9
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, (1.0 / config.camera.fps) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        logger.error(f"Zynq-7000 processing error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        camera.stop()
        leds.clear_all()
        det_stats = detector.get_statistics()
        logger.info("=" * 65)
        logger.info("  ZYNQ-7000 ARM SEGREGATION SUMMARY")
        logger.info("=" * 65)
        logger.info(f"Frames Processed:   {det_stats['total_frames']}")
        logger.info(f"Avg Inference Time: {det_stats['avg_inference_ms']:.1f} ms/frame")
        logger.info(f"Total Segregated:   {total_segregated}")
        logger.info("Zynq-7000 processing stopped cleanly.")

    return True


# ─── Image Mode ─────────────────────────────────────────────────────────────────

def run_image_mode(config: SystemConfig, image_path: str) -> bool:
    """Analyse one image on ZedBoard and display multimodal disposal advisory."""
    import cv2
    detector = WasteDetector(config.detection)
    if not detector.load_model():
        logger.error("Failed to load YOLO model.")
        return False

    advisor = WasteAdvisor()
    db = WasteDatabase(config.database.db_path)
    leds = ZedBoardLEDs()

    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not read image: {image_path}")
        return False

    logger.info(f"ZedBoard analyzing image: {image_path}")
    waste_frame = detector.detect(image, frame_id=0)

    print("\n" + "=" * 65)
    print(" 🗑️  ZEDBOARD ARM — MULTIMODAL WASTE SEGREGATION REPORT")
    print("=" * 65)

    if not waste_frame.detections:
        print("  ℹ️  No waste objects detected in this image.")
    else:
        print(f"  Found {len(waste_frame.detections)} waste item(s):\n")
        for i, det in enumerate(waste_frame.detections, 1):
            bin_info = get_bin_info(det.class_name)
            leds.indicate_segregation(det.class_name)

            advice = advisor.get_detection_advice(
                det.class_name,
                det.confidence,
                bbox=list(det.bbox),
                detected_object=det.detected_object,
            )

            print(f"[{i}] OBJECT: {det.class_name.upper()} ({det.confidence:.1%})")
            print(f"    SEGREGATION: {bin_info['bin']} (LED LD{bin_info.get('led', 7)} ON)")
            print("-" * 65)
            print(advice.get("message", "No advice available."))
            print("=" * 65)

            db.insert_waste_record({
                "timestamp": datetime.now().isoformat(),
                "waste_type": det.class_name,
                "ai_confidence": det.confidence,
                "ai_class": det.detected_object or det.class_name,
                "vehicle_id": "ZEDBOARD-ARM-IMAGE",
            })

    return True


# ─── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🗑️  Zynq-7000 (ZedBoard) Multimodal AI Waste Segregation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test",  action="store_true",
                        help="Run 10-frame self-test without camera or physical storage")
    parser.add_argument("--image", type=str,
                        help="Analyze a single image file on ZedBoard")
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    setup_logging(config)

    if args.image:
        success = run_image_mode(config, args.image)
        sys.exit(0 if success else 1)

    success = run_system(config, test_mode=args.test)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
