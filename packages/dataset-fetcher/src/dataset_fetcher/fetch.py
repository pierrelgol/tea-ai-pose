from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

from .profiles import DatasetProfile
from .validate import validate_dataset


def _download_archive(urls: list[str], archive_path: Path) -> str:
    last_error: Exception | None = None
    for url in urls:
        try:
            urllib.request.urlretrieve(url, archive_path)
            return url
        except Exception as exc:  # pragma: no cover - network-specific
            last_error = exc
    raise RuntimeError("failed to download dataset from provided URLs") from last_error


def _fetch_zip_dataset(dataset_root: Path, dataset_dir_name: str, urls: list[str]) -> Path:
    dataset_root.mkdir(parents=True, exist_ok=True)
    target_dir = dataset_root / dataset_dir_name

    if target_dir.exists():
        if target_dir.is_symlink() or target_dir.is_file():
            target_dir.unlink()
        else:
            shutil.rmtree(target_dir)

    with tempfile.TemporaryDirectory(prefix=f"{dataset_dir_name}-") as tmp_dir:
        archive_path = Path(tmp_dir) / f"{dataset_dir_name}.zip"
        _download_archive(urls, archive_path)

        with zipfile.ZipFile(archive_path) as zip_file:
            zip_file.extractall(dataset_root)

    return target_dir


def _fetch_local_dataset(local_path: str, dataset_root: Path, dataset_dir_name: str) -> Path:
    candidate = Path(local_path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()

    expected = (dataset_root / dataset_dir_name).resolve()
    if candidate == expected:
        return expected

    # Keep operation cheap and non-destructive by using a symlink into dataset_root.
    dataset_root.mkdir(parents=True, exist_ok=True)
    link_path = dataset_root / dataset_dir_name
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        else:
            shutil.rmtree(link_path)
    link_path.symlink_to(candidate, target_is_directory=True)
    return link_path.resolve()


def _label_rel_from_images_rel(images_rel: str) -> str:
    if images_rel.startswith("images/"):
        return "labels/" + images_rel[len("images/") :]
    return images_rel.replace("/images/", "/labels/")


def _list_image_label_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    if not images_dir.exists() or not labels_dir.exists():
        return []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in image_exts:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def _deterministic_subset(
    pairs: list[tuple[Path, Path]],
    *,
    split: str,
    limit: int,
    seed: int,
) -> list[tuple[Path, Path]]:
    if len(pairs) <= limit:
        return pairs

    def _rank_key(pair: tuple[Path, Path]) -> str:
        stem = pair[0].stem
        digest = hashlib.sha1(f"{seed}:{split}:{stem}".encode("utf-8")).hexdigest()
        return digest

    ranked = sorted(pairs, key=_rank_key)
    return ranked[:limit]


def _copy_pairs(
    pairs: list[tuple[Path, Path]],
    *,
    images_dst: Path,
    labels_dst: Path,
) -> None:
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)
    for image_src, label_src in pairs:
        shutil.copy2(image_src, images_dst / image_src.name)
        shutil.copy2(label_src, labels_dst / label_src.name)


def _load_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"id list not found: {path}")
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise RuntimeError(f"id list is empty: {path}")
    return ids


def _copy_ids_subset(
    *,
    images_src_dir: Path,
    ids: list[str],
    split: str,
    limit: int,
    seed: int,
    images_dst_dir: Path,
) -> None:
    images_dst_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(ids, key=lambda x: hashlib.sha1(f"{seed}:{split}:{x}".encode("utf-8")).hexdigest())
    selected = ranked[: min(limit, len(ranked))]
    copied = 0
    copied_stems: set[str] = set()
    for image_id in selected:
        src = images_src_dir / f"{image_id}.jpg"
        if not src.exists():
            continue
        shutil.copy2(src, images_dst_dir / src.name)
        copied_stems.add(src.stem)
        copied += 1

    # Fill missing slots from available files for non-standard layouts.
    remaining = max(0, limit - copied)
    if remaining == 0:
        return

    all_images = sorted([p for p in images_src_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"])
    if not all_images:
        raise RuntimeError(f"no images found under {images_src_dir}")
    ranked = sorted(
        all_images,
        key=lambda p: hashlib.sha1(f"{seed}:{split}:{p.stem}".encode("utf-8")).hexdigest(),
    )
    filled = 0
    for src in ranked:
        if src.stem in copied_stems:
            continue
        shutil.copy2(src, images_dst_dir / src.name)
        filled += 1
        if filled >= remaining:
            break


def _fetch_coco_subset_local_dataset(profile: DatasetProfile, dataset_root: Path, dataset_dir_name: str) -> Path:
    dataset_root.mkdir(parents=True, exist_ok=True)
    target_dir = dataset_root / dataset_dir_name
    if target_dir.exists():
        shutil.rmtree(target_dir)

    source_root = Path(profile.local_path or "")
    if not source_root.is_absolute():
        source_root = (Path.cwd() / source_root).resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"source.local_path not found: {source_root}")

    train_images_src = source_root / profile.train_images_rel
    val_images_src = source_root / profile.val_images_rel
    train_labels_rel = _label_rel_from_images_rel(profile.train_images_rel)
    val_labels_rel = _label_rel_from_images_rel(profile.val_images_rel)
    train_labels_src = source_root / train_labels_rel
    val_labels_src = source_root / val_labels_rel

    train_pairs = _list_image_label_pairs(train_images_src, train_labels_src)
    val_pairs = _list_image_label_pairs(val_images_src, val_labels_src)
    if not train_pairs:
        raise RuntimeError(f"no image/label pairs found for train split at {train_images_src} and {train_labels_src}")
    if not val_pairs:
        raise RuntimeError(f"no image/label pairs found for val split at {val_images_src} and {val_labels_src}")

    seed = int(profile.subset_seed or 42)
    train_selected = _deterministic_subset(
        train_pairs,
        split="train",
        limit=int(profile.subset_train_max_images or len(train_pairs)),
        seed=seed,
    )
    val_selected = _deterministic_subset(
        val_pairs,
        split="val",
        limit=int(profile.subset_val_max_images or len(val_pairs)),
        seed=seed,
    )

    _copy_pairs(
        train_selected,
        images_dst=target_dir / profile.train_images_rel,
        labels_dst=target_dir / train_labels_rel,
    )
    _copy_pairs(
        val_selected,
        images_dst=target_dir / profile.val_images_rel,
        labels_dst=target_dir / val_labels_rel,
    )

    return target_dir


def _fetch_coco_ids_local_dataset(profile: DatasetProfile, dataset_root: Path, dataset_dir_name: str) -> Path:
    dataset_root.mkdir(parents=True, exist_ok=True)
    target_dir = dataset_root / dataset_dir_name
    if target_dir.exists():
        shutil.rmtree(target_dir)

    source_root = Path(profile.local_path or "")
    if not source_root.is_absolute():
        source_root = (Path.cwd() / source_root).resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"source.local_path not found: {source_root}")

    images_src_dir = source_root / str(profile.ids_images_rel)
    train_ids_path = source_root / str(profile.ids_train_ids_rel)
    val_ids_path = source_root / str(profile.ids_val_ids_rel)
    train_ids = _load_ids(train_ids_path)
    val_ids = _load_ids(val_ids_path)
    seed = int(profile.subset_seed or 42)
    _copy_ids_subset(
        images_src_dir=images_src_dir,
        ids=train_ids,
        split="train",
        limit=int(profile.subset_train_max_images or len(train_ids)),
        seed=seed,
        images_dst_dir=target_dir / profile.train_images_rel,
    )
    _copy_ids_subset(
        images_src_dir=images_src_dir,
        ids=val_ids,
        split="val",
        limit=int(profile.subset_val_max_images or len(val_ids)),
        seed=seed,
        images_dst_dir=target_dir / profile.val_images_rel,
    )
    return target_dir


def fetch_dataset(
    profile: DatasetProfile,
    dataset_root: Path,
    source_url_override: str | None = None,
    dataset_dir_name_override: str | None = None,
) -> Path:
    dataset_dir_name = dataset_dir_name_override or profile.dataset_dir_name

    if profile.source_type == "local_dir":
        dataset_dir = _fetch_local_dataset(profile.local_path or "", dataset_root, dataset_dir_name)
    elif profile.source_type == "coco_subset_local":
        dataset_dir = _fetch_coco_subset_local_dataset(profile, dataset_root, dataset_dir_name)
    elif profile.source_type == "coco_ids_local":
        dataset_dir = _fetch_coco_ids_local_dataset(profile, dataset_root, dataset_dir_name)
    else:
        urls = [source_url_override] if source_url_override else profile.urls
        dataset_dir = _fetch_zip_dataset(dataset_root, dataset_dir_name, urls)

    validate_dataset(profile, dataset_dir)
    return dataset_dir
