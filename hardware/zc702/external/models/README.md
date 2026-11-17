# ZC702 External Model Storage

This directory holds the model binaries transferred to the ZC702 storage mount (`/mnt/models/`) during deployment.

## Files
- `best.pt`: Fine-tuned YOLO waste detection weights (~22.5 MB)
- `yolov8n.pt`: Base YOLOv8 nano model (~6.5 MB)

## Deployment Note
The deployment script `hardware/zc702/vivado/deploy_to_zc702.sh` uploads these weights to `/mnt/models/` on the board.
