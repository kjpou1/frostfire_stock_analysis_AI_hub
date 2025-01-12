import numpy as np
from PIL import Image


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess an image for DenseNet.
    """
    image = image.resize((224, 224))  # Resize to DenseNet input size
    image_array = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)
