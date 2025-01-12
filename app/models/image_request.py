from typing import List

from pydantic import BaseModel


class ImageRequest(BaseModel):
    base64_images: List[str]
