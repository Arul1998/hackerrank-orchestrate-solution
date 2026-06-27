from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass


def normalize_image_bytes(path: Path) -> tuple[bytes, str]:
    """Return JPEG bytes suitable for OpenAI vision APIs."""
    with Image.open(path) as image:
        fmt = (image.format or "").upper()
        raw = path.read_bytes()
        suffix = path.suffix.lower()

        # Pass through only genuine JPEG files
        if fmt == "JPEG" and suffix in {".jpg", ".jpeg"}:
            return raw, "jpeg"

        # Re-encode AVIF/WEBP/mislabeled files to JPEG
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
        return buffer.getvalue(), "jpeg"
