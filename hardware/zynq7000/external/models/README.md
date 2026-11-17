# Zynq-7000 (ZedBoard) External Model Storage

This directory holds the model binaries transferred to the ZedBoard external storage mount (`/mnt/ext_storage/models/`) during deployment.

## Files
- `best.pt`: Fine-tuned YOLO waste detection weights (~22.5 MB)
- `yolov8n.pt`: Base YOLOv8 nano model (~6.5 MB)

## Deployment Note
The deployment script `hardware/zynq7000/vivado/deploy_to_zedboard.sh` transfers these files automatically over SCP or during offline SD card staging via `--stage <path>`.
