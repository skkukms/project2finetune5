# FFHQ Face Generation — Two-Stage Refiner

## Pipeline
```
z (512-dim) → G_256 (frozen) → Refiner512 → Refiner1024 → 1024×1024 image
```

## Setup
```bash
pip install -r requirements.txt
```

## Training

**Phase 1** — Train Refiner512 (256→512):
```bash
python train_refiner.py --phase 1 \
  --config configs/refiner_512.yaml \
  --g256-ckpt /path/to/ffhq256_baseline.pt
```

**Phase 2** — Train Refiner1024 (512→1024):
```bash
python train_refiner.py --phase 2 \
  --config configs/refiner_1024.yaml \
  --g256-ckpt /path/to/ffhq256_baseline.pt \
  --r512-ckpt runs/refiner_512/final.pt
```

**Resume**:
```bash
python train_refiner.py --phase 1 \
  --config configs/refiner_512.yaml \
  --resume runs/refiner_512/ckpt_000010000.pt
```

## Generate Images
```bash
python generate.py \
  --ckpt runs/refiner_1024/final.pt \
  --out samples/
```

## Export to ONNX
```bash
python export_onnx.py \
  --g256-ckpt ckpt/ffhq256_baseline.pt \
  --r512-ckpt runs/refiner_512/final.pt \
  --r1024-ckpt runs/refiner_1024/final.pt \
  --out submission.onnx
```
