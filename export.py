import torch
import yaml
import sys
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Add YOLOv9 directory to Python path
sys.path.insert(0, r"D:\working\yolov9\yolov9")

from models.yolo import Model

# Load opt.yaml
opt_path = r"D:\working\yolov9\yolov9-s\yolov9-s\opt.yaml"
with open(opt_path, 'r') as f:
    opt = yaml.safe_load(f)

# Get model config path
cfg_path = Path(r"D:\working\yolov9\yolov9") / Path(opt.get("cfg", "models/detect/yolov9-s.yaml"))

# Load trained weights
weights_path = r"D:\working\yolov9\yolov9-s\yolov9-s\weights\best.pt"
checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))

# Detect number of classes from checkpoint
if "nc" in checkpoint:
    nc = checkpoint["nc"]
else:
    nc = 3

print(f"Checkpoint trained for {nc} classes")

# Load model architecture
model = Model(cfg=str(cfg_path), ch=3, nc=nc)

# Modify the last detection layer to match checkpoint
model.model[-1].nc = nc
model.model[-1].cv2.out_channels = nc
model.names = {i: f'class_{i}' for i in range(nc)}

# Ensure state_dict format is correct
if "model" in checkpoint:
    state_dict = checkpoint["model"].state_dict()
else:
    state_dict = checkpoint

# Load weights while ignoring mismatches
model.load_state_dict(state_dict, strict=False)

model.eval()

# Export to ONNX
imgsz = opt.get("imgsz", 640)
dummy_input = torch.randn(1, 3, imgsz, imgsz)
onnx_path = r"D:\working\yolov9\best.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path, 
    export_params=True, 
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'], 
    output_names=['output'], 
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"Model successfully exported to {onnx_path}")
