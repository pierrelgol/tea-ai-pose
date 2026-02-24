# gpu-auto-tuner

Offline GPU auto-tuning for `tea-ai` training.

## What it does

- Probes the active CUDA GPU and builds a stable hardware signature.
- Runs offline detector training probes using binary search to find safe maximum batch.
- Benchmarks extended train knobs (`imgsz`, `workers`, `cache`, `amp`, `tf32`, `cudnn_benchmark`).
- Writes tuned values back into `config.json` and stores a detailed tuning report.

## Usage

```bash
uv run gpu-auto-tuner --config config.json
```

Force a re-tune even if the current GPU signature is already tuned:

```bash
uv run gpu-auto-tuner --config config.json --force
```

## Output

- Config backup: `config.json.bak.<timestamp>`
- Tuner report: `artifacts/models/<model_key>/runs/<run_id>/tuner/<gpu_signature>/tuner_report.json`

## Notes

- This tool requires CUDA and `nvidia-smi` visibility.
- `detector-train` enforces tuned GPU signature checks when `tuner.enabled=true`.
