# detector-optimize

Compile a trained YOLO OBB detector into TensorRT-ready artifacts for faster inference and smaller deployment footprint.

## What it does

1. Resolves trained `.pt` weights (current run by default)
2. Exports ONNX from the checkpoint
3. Compiles TensorRT engine (`.engine`), preferring `trtexec`
4. Writes a reproducible `optimize_summary.json`

Outputs are stored in:

- `artifacts/models/{model_key}/runs/{run_id}/optimize/`

## CLI

```bash
# Compile current run with fp16 (default)
uv run detector-optimize --config config.json

# Force static batch=4, keep ONNX artifact
uv run detector-optimize --config config.json --batch 4 --keep-onnx

# Force ultralytics backend instead of trtexec
uv run detector-optimize --config config.json --force-ultralytics

# Compile specific weights path
uv run detector-optimize --config config.json --weights ./best.pt
```

## Key flags

- `--precision {fp16,fp32,int8}`
- `--workspace-gb 4.0`
- `--batch 1`
- `--dynamic` (for dynamic shape ONNX/engine)
- `--trtexec-bin trtexec`
- `--force-ultralytics`

## Artifacts

- `{model_key}_{run_id}.onnx` (if `--keep-onnx`)
- `{model_key}_{run_id}_{precision}.engine`
- `trtexec.log` (when trtexec backend is used)
- `optimize_summary.json`
