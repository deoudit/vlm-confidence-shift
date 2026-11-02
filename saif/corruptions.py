"""Image corruption operators T_s (ImageNet-C protocol, Hendrycks & Dietterich 2019).

Only the image is corrupted; the question is unchanged, so all behaviour
change is attributable to the visual shift.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

DEFAULT_TYPES = ["fog", "snow", "defocus_blur", "brightness"]


def corrupt_image(img: Image.Image, corruption: str, severity: int) -> Image.Image:
    """Apply corruption at severity in {1..5}; severity 0 returns the clean image.

    Note: imagecorruptions expects uint8 RGB arrays. Some corruptions (e.g. fog)
    internally resize; we corrupt at native resolution and let the VLM processor
    handle its own resizing afterwards, so clean/corrupt pairs see the same
    preprocessing path.
    """
    # NumPy 2.x removed the *_ aliases that imagecorruptions (last released
    # 2019) still uses; restore them defensively before importing the package.
    import numpy as _np
    for _old, _new in [("float_", "float64"), ("int_", "int64"),
                       ("bool_", "bool_"), ("complex_", "complex128"),
                       ("object_", "object_")]:
        if not hasattr(_np, _old):
            setattr(_np, _old, getattr(_np, _new))
    try:
        from imagecorruptions import corrupt, get_corruption_names
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pip install imagecorruptions  (also needs opencv-python, scikit-image)"
        ) from e
    if severity == 0:
        return img
    if not 1 <= severity <= 5:
        raise ValueError(f"severity must be in 0..5, got {severity}")
    if corruption not in get_corruption_names("all"):
        raise ValueError(f"unknown corruption '{corruption}'")
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    out = corrupt(arr, corruption_name=corruption, severity=severity)
    return Image.fromarray(np.uint8(out))


def condition_grid(types: list[str], severities: list[int]) -> list[tuple[str, int]]:
    """List of (type, severity) conditions, always including clean = ('clean', 0)."""
    grid = [("clean", 0)]
    grid += [(t, s) for t in types for s in severities]
    return grid
