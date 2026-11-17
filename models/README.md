# Models Directory

This directory stores on-device AI model weights and artifacts for the Bharat AI-SoC Waste Management System.

## Multimodal Vision Models (.pt Weights)

The system includes **4 complementary YOLO vision models** that cover both general segregation and fine-grained recycling classification:

| Model File | Size | Architecture | Classes | Purpose & Multimodal Role |
| :--- | :--- | :--- | :--- | :--- |
| **`best.pt`** | ~22.5 MB | YOLOv8s | **5 Classes** (`plastic_bottle`, `glass_bottle`, `metal_can`, `paper_cardboard`, `organic_waste`) | **Primary Segregation Model** — Optimized for direct bin sorting and embedded deployment on **Zynq-7000 (ZedBoard ARM)**. |
| **`ecosort_waste_yolo11s.pt`** | ~19.2 MB | YOLO11s | **66 Classes** (Aerosol, Aluminium Foil, Battery, Broken Glass, Drink Can, Food Waste, Magazine, Tupperware, etc.) | **Granular Recyclables Model** — Delivers fine-grained item recognition to provide precise resin code (#1–#7) and recycling steps via Llama/WasteAdvisor. |
| **`weights.pt`** | ~20.3 MB | YOLOv8s | **1 Class** (`bottle`) | **Dedicated Bottle Specialist** — High-precision detector for single-use plastic/glass beverage containers. |
| **`yolov8n.pt`** | ~6.5 MB | YOLOv8n | **80 COCO Classes** (cutlery, electronic devices, bags, fruit peels, books, etc.) | **Dual-Model Fusion Fallback** — Maps everyday items to waste categories using `COCO_WASTE_MAP`. |

---

## Language Model (Modality 2 — Llama 3.2 1B Instruct)

- **Model:** Meta Llama 3.2 1B Instruct (`llama-3.2-1b-instruct-q4_k_m.gguf`)
- **Format:** GGUF (4-bit quantized Q4_K_M)
- **Size:** ~807 MB
- **Role:** Takes detected objects from the vision models and generates dynamic disposal advice, upcycling guides, and recycling answers.
- **Download:**
  ```bash
  bash scripts/download_models.sh --llama
  ```
  *(If the file is not downloaded, the system seamlessly falls back to the internal 2,300-rule `WasteAdvisor` knowledge base).*

---

## Directory Hierarchy & Sync

- `models/`: Root staging directory for all weights
- `src/yolo/weights/`: Default runtime directory loaded by `WasteDetector`
- `hardware/zynq7000/external/models/`: Staged for external SD card storage on ZedBoard
