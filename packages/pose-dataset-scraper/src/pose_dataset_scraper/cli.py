from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any

from pipeline_config import load_pipeline_config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(slots=True)
class ScrapeConfig:
    config_path: Path
    dataset_root: Path
    output_root: Path
    raw_root: Path
    run_seed: int
    split_ratio: float


DEFAULT_QUERIES = [
    "military soldier full body",
    "combatant with vest and camo",
    "police officer tactical gear",
    "gendarmerie officer full body",
    "urban armed personnel standing",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_cfg(config_path: Path) -> ScrapeConfig:
    shared = load_pipeline_config(config_path)
    dataset_name = str(shared.dataset.get("name") or shared.run["dataset"])
    dataset_root = Path(shared.paths["dataset_root"])
    output_root = dataset_root / str(shared.dataset.get("augmented_subdir", "augmented")) / dataset_name
    raw_root = dataset_root / "raw_pose_scrape"
    return ScrapeConfig(
        config_path=config_path,
        dataset_root=dataset_root,
        output_root=output_root,
        raw_root=raw_root,
        run_seed=int(shared.run.get("seed", 42)),
        split_ratio=0.8,
    )


def _discover(cfg: ScrapeConfig) -> dict[str, Any]:
    out = cfg.raw_root / "discover"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": _now_iso(),
        "license_policy": "license-safe-only",
        "queries": DEFAULT_QUERIES,
        "sources": [
            {"name": "wikimedia", "license_required": True},
            {"name": "openimages", "license_required": True},
            {"name": "government-open-data", "license_required": True},
        ],
    }
    path = out / "query_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "ok", "query_manifest": str(path), "query_count": len(DEFAULT_QUERIES)}


def _fetch(cfg: ScrapeConfig) -> dict[str, Any]:
    # This stage is intentionally conservative: it only materializes a provenance
    # scaffold and expects licensed assets to be placed under raw/images.
    img_dir = cfg.raw_root / "images"
    prov_dir = cfg.raw_root / "provenance"
    img_dir.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)
    prov_path = prov_dir / "manifest.jsonl"
    if not prov_path.exists():
        prov_path.write_text("", encoding="utf-8")
    return {
        "status": "ok",
        "images_dir": str(img_dir),
        "provenance_manifest": str(prov_path),
        "note": "place licensed images in raw_pose_scrape/images then rerun with --action build",
    }


def _dedupe(cfg: ScrapeConfig) -> dict[str, Any]:
    src = cfg.raw_root / "images"
    dst = cfg.raw_root / "deduped"
    dst.mkdir(parents=True, exist_ok=True)
    kept = 0
    seen_hash: set[str] = set()
    for p in sorted(src.glob("**/*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        digest = _sha256(p)
        if digest in seen_hash:
            continue
        seen_hash.add(digest)
        shutil.copy2(p, dst / p.name)
        kept += 1
    return {"status": "ok", "kept": kept, "output_dir": str(dst)}


def _autolabel(cfg: ScrapeConfig) -> dict[str, Any]:
    # Stub labels with empty files; operator can replace with teacher-generated labels.
    src = cfg.raw_root / "deduped"
    labels = cfg.raw_root / "labels_auto"
    labels.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in sorted(src.glob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        (labels / f"{p.stem}.txt").write_text("", encoding="utf-8")
        count += 1
    return {"status": "ok", "labels": str(labels), "count": count}


def _qc(cfg: ScrapeConfig) -> dict[str, Any]:
    deduped = cfg.raw_root / "deduped"
    labels = cfg.raw_root / "labels_auto"
    rows: list[dict[str, Any]] = []
    for p in sorted(deduped.glob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = labels / f"{p.stem}.txt"
        rows.append(
            {
                "image": p.name,
                "label": label_path.name,
                "label_exists": label_path.exists(),
                "status": "pass" if label_path.exists() else "fail",
            }
        )
    out = cfg.raw_root / "qc_report.json"
    out.write_text(json.dumps({"rows": rows, "created_at": _now_iso()}, indent=2), encoding="utf-8")
    return {"status": "ok", "report": str(out), "rows": len(rows)}


def _build_dataset(cfg: ScrapeConfig) -> dict[str, Any]:
    images_src = cfg.raw_root / "deduped"
    labels_src = cfg.raw_root / "labels_auto"

    train_img = cfg.output_root / "images" / "train"
    val_img = cfg.output_root / "images" / "val"
    train_lab = cfg.output_root / "labels" / "train"
    val_lab = cfg.output_root / "labels" / "val"
    for p in (train_img, val_img, train_lab, val_lab):
        p.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(images_src.glob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    rng = random.Random(cfg.run_seed)
    rng.shuffle(files)
    split_idx = int(len(files) * cfg.split_ratio)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    def _copy_block(items: list[Path], out_img: Path, out_lab: Path) -> int:
        copied = 0
        for img in items:
            lab = labels_src / f"{img.stem}.txt"
            if not lab.exists():
                continue
            shutil.copy2(img, out_img / img.name)
            shutil.copy2(lab, out_lab / lab.name)
            copied += 1
        return copied

    train_count = _copy_block(train_files, train_img, train_lab)
    val_count = _copy_block(val_files, val_img, val_lab)

    classes = cfg.output_root / "classes.txt"
    classes.write_text("person\n", encoding="utf-8")
    provenance = cfg.output_root / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "created_at": _now_iso(),
                "source_root": str(cfg.raw_root),
                "license_policy": "license-safe-only",
                "train_count": train_count,
                "val_count": val_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "dataset_root": str(cfg.output_root),
        "train_images": train_count,
        "val_images": val_count,
        "classes_file": str(classes),
        "provenance": str(provenance),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape and curate license-safe human pose datasets")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument(
        "--action",
        choices=["discover", "fetch", "dedupe", "autolabel", "qc", "build", "all"],
        default="all",
    )
    args = parser.parse_args()

    cfg = _resolve_cfg(args.config)

    actions = [args.action]
    if args.action == "all":
        actions = ["discover", "fetch", "dedupe", "autolabel", "qc", "build"]

    summaries: list[dict[str, Any]] = []
    for action in actions:
        if action == "discover":
            summaries.append({"action": action, **_discover(cfg)})
        elif action == "fetch":
            summaries.append({"action": action, **_fetch(cfg)})
        elif action == "dedupe":
            summaries.append({"action": action, **_dedupe(cfg)})
        elif action == "autolabel":
            summaries.append({"action": action, **_autolabel(cfg)})
        elif action == "qc":
            summaries.append({"action": action, **_qc(cfg)})
        elif action == "build":
            summaries.append({"action": action, **_build_dataset(cfg)})

    print(json.dumps({"status": "ok", "steps": summaries}, indent=2))


if __name__ == "__main__":
    main()
