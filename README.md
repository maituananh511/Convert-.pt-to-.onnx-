# Convert YOLOv9 Model from .PT to .ONNX

## Overview
This script converts a trained YOLOv9 model from PyTorch format (`.pt`) to ONNX format (`.onnx`). ONNX allows for model interoperability across different frameworks and deployment on edge devices optimized with TensorRT, OpenVINO, or ONNX Runtime.

## Prerequisites
### Dependencies
Ensure you have the required Python libraries installed:

```bash
pip install torch pyyaml
```

### YOLOv9 Setup
Download or clone the YOLOv9 repository:
Modify the `sys.path.insert(0, ...)` line in the script to match your YOLOv9 directory.

## Usage
### 1. Update Paths in the Script
Modify the following paths to match your environment:
- `opt.yaml` path: Define input size (`imgsz`) and other options.
- Model weight path (`best.pt`): Update with the correct location of your trained model.
- ONNX output path (`best.onnx`): Define where the ONNX model should be saved.

### 2. Run the Script
Execute the script:
```bash
python convert_yolov9.py
```
After execution, the converted ONNX model will be saved at the specified location.



