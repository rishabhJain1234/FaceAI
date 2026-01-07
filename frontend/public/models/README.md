# ONNX Models Required

This directory should contain two ONNX models for client-side face processing:

## Required Models

### 1. Face Detection Model
**Filename**: `det_10g.onnx` or `retinaface_mobile.onnx`
**Size**: ~2-3 MB
**Purpose**: Detect faces in images
**Input**: 640x640 RGB image
**Output**: Bounding boxes + landmarks

### 2. Face Recognition Model  
**Filename**: `w600k_r50.onnx` or `mobilefacenet.onnx`
**Size**: ~3-4 MB
**Purpose**: Generate face embeddings
**Input**: 112x112 aligned face
**Output**: 512-d embedding vector

## Where to Get Models

### Option 1: Use InsightFace Models (Recommended)
The easiest approach is to use the same models from InsightFace but in ONNX format:

```bash
# Install insightface to download models
pip install insightface

# Run this Python script to get the models:
python3 << 'EOF'
from insightface.app import FaceAnalysis
import shutil
import os

# Initialize to download models
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Models are downloaded to ~/.insightface/models/buffalo_l/
source_dir = os.path.expanduser('~/.insightface/models/buffalo_l/')
dest_dir = './public/models/'

# Copy detection model
shutil.copy(
    os.path.join(source_dir, 'det_10g.onnx'),
    os.path.join(dest_dir, 'det_10g.onnx')
)

# Copy recognition model
shutil.copy(
    os.path.join(source_dir, 'w600k_r50.onnx'),
    os.path.join(dest_dir, 'w600k_r50.onnx')
)

print("✅ Models copied successfully!")
print(f"   Detection: {dest_dir}det_10g.onnx")
print(f"   Recognition: {dest_dir}w600k_r50.onnx")
EOF
```

### Option 2: Download from ONNX Model Zoo
Visit: https://github.com/onnx/models

Look for:
- Face detection: RetinaFace, BlazeFace, or SCRFD
- Face recognition: ArcFace or MobileFaceNet

### Option 3: Download from Hugging Face
Visit: https://huggingface.co/models?library=onnx&search=face

## After Getting Models

Once you have the models:

1. Place them in this directory (`frontend/public/models/`)
2. Update `frontend/lib/faceProcessor.ts` with the correct filenames
3. Test by running the frontend and checking browser console

## Verify Models

After placing models here, verify with:

```bash
ls -lh frontend/public/models/

# Should show:
# det_10g.onnx (or retinaface_mobile.onnx) - ~2-3 MB
# w600k_r50.onnx (or mobilefacenet.onnx) - ~3-4 MB
```

## Model Format Requirements

- Format: ONNX (.onnx)
- Precision: FP32 or FP16 (FP16 preferred for smaller size)
- Optimization: Use onnx-simplifier if possible
- Quantization: INT8 quantization can reduce size further

## Troubleshooting

**Models not loading in browser?**
- Check file size (should be 2-4 MB each)
- Check browser console for errors
- Verify CORS headers (should work from public/ directory)
- Try accessing directly: http://localhost:3000/models/det_10g.onnx

**Need help?**
See `CLIENT_SIDE_MIGRATION_README.md` for detailed model conversion scripts.
