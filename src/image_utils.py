from pathlib import Path

from PIL import Image
from io import BytesIO


def load_image_bytes(path: str) -> bytes:
    with Image.open(path) as img:
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def is_image_file(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

