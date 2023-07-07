"""IO utilities for images."""

from typing import List

import numpy as np
import PIL


def load_image_paths_to_array(image_paths: List[str]):
    """Load a list of image paths into a numpy array."""
    images = []
    for image_path in image_paths:
        images.append(PIL.Image.open(image_path))
    return np.stack(images)
