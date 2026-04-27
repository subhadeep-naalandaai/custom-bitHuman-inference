"""
bithuman_expression/utils/facecrop.py
Python replacement for facecrop.cpython-310-x86_64-linux-gnu.so

Exposes:
  get_scaled_bbox(bbox, img_w, img_h, ratio) -> List[int]
  process_image(image, face_ratio=1.5, target_size=(512,512)) -> PIL.Image
"""

from typing import List, Tuple, Union
import numpy as np
from PIL import Image


def get_scaled_bbox(
    bbox: List[int],
    img_w: int,
    img_h: int,
    ratio: float,
) -> List[int]:
    """
    Expand a bounding box by `ratio` around its centre, clamped to image bounds.

    Args:
        bbox:  [x1, y1, x2, y2] in pixel coordinates
        img_w: image width  (used for clamping)
        img_h: image height (used for clamping)
        ratio: scale factor (e.g. 1.5 → 50 % larger on each side)

    Returns:
        [x1, y1, x2, y2] scaled and clamped
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_w = (x2 - x1) / 2.0 * ratio
    half_h = (y2 - y1) / 2.0 * ratio

    sx1 = max(0, int(cx - half_w))
    sy1 = max(0, int(cy - half_h))
    sx2 = min(img_w, int(cx + half_w))
    sy2 = min(img_h, int(cy + half_h))

    return [sx1, sy1, sx2, sy2]


def process_image(
    image: Union[Image.Image, np.ndarray],
    face_ratio: float = 1.5,
    target_size: Tuple[int, int] = (512, 512),
) -> Image.Image:
    """
    Detect the face in `image`, crop around it (scaled by face_ratio),
    and resize to target_size.

    Falls back to a full-image resize when zero or multiple faces are found.

    Args:
        image:       PIL.Image (RGB) or HxWx3 uint8 numpy array
        face_ratio:  bbox scale factor passed to get_scaled_bbox
        target_size: (width, height) of the returned image

    Returns:
        PIL.Image resized to target_size
    """
    from inference.utils.cpu_face_handler import CPUFaceHandler

    # Normalise input to PIL and numpy
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image.astype(np.uint8), "RGB")
        arr = image.astype(np.uint8)
    else:
        pil_img = image.convert("RGB")
        arr = np.array(pil_img)

    h, w = arr.shape[:2]

    handler = CPUFaceHandler(model_selection=0)
    num_faces, bbox = handler.detect(arr)

    if num_faces == 1 and len(bbox) == 4:
        x1, y1, x2, y2 = get_scaled_bbox(bbox, w, h, face_ratio)
        cropped = pil_img.crop((x1, y1, x2, y2))
    else:
        cropped = pil_img

    return cropped.resize(target_size, Image.LANCZOS)
