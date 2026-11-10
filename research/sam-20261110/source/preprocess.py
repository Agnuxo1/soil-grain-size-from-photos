from __future__ import annotations
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


TARGET_PPM = 5.0

LONG_MARGIN = 0.12

CROP_POSITIONS = (("long_1", 0.0), ("long_2", 0.25), ("center", 0.5), ("long_4", 0.75), ("long_5", 1.0))

PHONE_PREFIXES = (
    ("motorolaedge60fusion", "Motorola Edge 60 Fusion"),
    ("motorolaedge", "Motorola Edge"),
    ("samsunga52", "Samsung A52"),
    ("iphone14", "iPhone 14"),
    ("iphone16", "iPhone 16"),
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def normalize_text(value: object) -> str:
    transliteration = str(value).translate(str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}))
    ascii_text = unicodedata.normalize("NFKD", transliteration).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]+", "", ascii_text)

def image_index(rgb_cache: np.lib.npyio.NpzFile, data_dir: Path) -> pd.DataFrame:
    original_paths = [str(value) for value in rgb_cache["paths"]]
    sample_ids = [str(value) for value in rgb_cache["sample_ids"]]
    if len(original_paths) != len(sample_ids):
        raise ValueError("RGB feature cache has mismatched path and sample_id lengths")

    by_name: dict[str, list[Path]] = {}
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            by_name.setdefault(path.name.casefold(), []).append(path)

    ppm = pd.read_csv(data_dir / "ppm_updated.csv")
    ppm_records = {str(row.phone): row for row in ppm.itertuples(index=False)}
    rows = []
    for original, sample_id in zip(original_paths, sample_ids):
        filename = Path(original.replace("\\", "/")).name
        matches = by_name.get(filename.casefold(), [])
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one local match for {filename}; got {len(matches)}")
        path = matches[0]
        normalized_stem = normalize_text(path.stem)
        phone = next((name for prefix, name in PHONE_PREFIXES if normalized_stem.startswith(prefix)), None)
        if phone is None or phone not in ppm_records:
            raise ValueError(f"Unable to resolve camera/PPM metadata for {filename}")
        with Image.open(path) as image:
            raw_width, raw_height = image.width, image.height
        camera = ppm_records[phone]
        scale_ratio = max(raw_width, raw_height) / max(int(camera.width), int(camera.height))
        estimated_ppm = float(camera.ppm) * scale_ratio
        rows.append({
            "original_path": original,
            "filename": filename,
            "path": path,
            "sample_id": sample_id,
            "phone": phone,
            "image_width": raw_width,
            "image_height": raw_height,
            "estimated_ppm": estimated_ppm,
        })
    return pd.DataFrame(rows)

def physical_crops(image: Image.Image, patch_size: int) -> list[Image.Image]:
    width, height = image.size
    long_side, short_side = max(width, height), min(width, height)
    margin = max(0, min(int(short_side * LONG_MARGIN), (long_side - patch_size) // 2))
    long_span = max(long_side - patch_size - 2 * margin, 0)
    short_start = max((short_side - patch_size) // 2, 0)
    crops = []
    for _, position in CROP_POSITIONS:
        long_start = round(margin + position * long_span)
        if width >= height:
            box = (long_start, short_start, long_start + patch_size, short_start + patch_size)
        else:
            box = (short_start, long_start, short_start + patch_size, long_start + patch_size)
        crops.append(image.crop(box))
    return crops

class PhysicalCropDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, patch_size: int, mean: tuple[float, ...], std: tuple[float, ...], image_mode: str):
        self.frame = frame.reset_index(drop=True)
        self.patch_size = patch_size
        self.image_mode = image_mode
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row["path"]) as image:
            image.seek(0)
            canonical = ImageOps.exif_transpose(image).convert("RGB")
        scale = TARGET_PPM / float(row["estimated_ppm"])
        canonical = canonical.resize(
            (round(canonical.width * scale), round(canonical.height * scale)), Image.Resampling.LANCZOS
        )
        patches = physical_crops(canonical, self.patch_size)
        if self.image_mode == "grayscale":
            patches = [ImageOps.grayscale(patch).convert("RGB") for patch in patches]
        return torch.stack([self.transform(patch) for patch in patches]), index
