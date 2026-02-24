from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

from .labels import PoseLabel, load_pose_labels


@dataclass(slots=True)
class SampleRecord:
    split: str
    stem: str
    image_path: Path | None
    gt_label_path: Path | None


def index_ground_truth(dataset_root: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for split in ("train", "val"):
        img_dir = dataset_root / "images" / split
        lab_dir = dataset_root / "labels" / split

        stems: set[str] = set()
        if img_dir.exists():
            stems |= {p.stem for p in img_dir.iterdir() if p.is_file()}
        if lab_dir.exists():
            stems |= {p.stem for p in lab_dir.iterdir() if p.is_file() and p.suffix == ".txt"}

        for stem in sorted(stems):
            image_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                p = img_dir / f"{stem}{ext}"
                if p.exists():
                    image_path = p
                    break
            gt = lab_dir / f"{stem}.txt"
            records.append(
                SampleRecord(
                    split=split,
                    stem=stem,
                    image_path=image_path,
                    gt_label_path=gt if gt.exists() else None,
                )
            )
    return records


def _png_shape(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as f:
        sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            return None
        length = struct.unpack(">I", f.read(4))[0]
        chunk_type = f.read(4)
        if length != 13 or chunk_type != b"IHDR":
            return None
        data = f.read(13)
        w, h = struct.unpack(">II", data[:8])
        return int(h), int(w)


def _bmp_shape(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as f:
        header = f.read(26)
        if len(header) < 26 or header[:2] != b"BM":
            return None
        w = struct.unpack("<I", header[18:22])[0]
        h = struct.unpack("<I", header[22:26])[0]
        return int(h), int(w)


def _jpeg_shape(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as f:
        if f.read(2) != b"\xff\xd8":
            return None
        while True:
            marker_prefix = f.read(1)
            if not marker_prefix:
                return None
            if marker_prefix != b"\xff":
                continue
            marker = f.read(1)
            if not marker:
                return None
            while marker == b"\xff":
                marker = f.read(1)
                if not marker:
                    return None
            m = marker[0]
            if m in (0xD8, 0xD9):
                continue
            seg_len_raw = f.read(2)
            if len(seg_len_raw) != 2:
                return None
            seg_len = struct.unpack(">H", seg_len_raw)[0]
            if seg_len < 2:
                return None
            if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                payload = f.read(seg_len - 2)
                if len(payload) < 5:
                    return None
                h = struct.unpack(">H", payload[1:3])[0]
                w = struct.unpack(">H", payload[3:5])[0]
                return int(h), int(w)
            f.seek(seg_len - 2, 1)


def image_shape_fast(path: Path | None) -> tuple[int, int] | None:
    if path is None or not path.exists():
        return None
    suffix = path.suffix.lower()
    try:
        if suffix == ".png":
            return _png_shape(path)
        if suffix in {".jpg", ".jpeg"}:
            return _jpeg_shape(path)
        if suffix == ".bmp":
            return _bmp_shape(path)
    except Exception:
        return None
    return None


def load_prediction_pose_labels(
    predictions_root: Path,
    model_name: str,
    split: str,
    stem: str,
    conf_threshold: float,
) -> list[PoseLabel]:
    p = predictions_root / model_name / "labels" / split / f"{stem}.txt"
    return load_pose_labels(p, is_prediction=True, conf_threshold=conf_threshold)
