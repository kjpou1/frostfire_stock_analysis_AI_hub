from typing import List

from pydantic import BaseModel


class ImagePayloadRequest(BaseModel):
    payload: dict

    class Config:
        schema_extra = {
            "example": {
                "payload": {
                    "data": [
                        "/9j/4AAQSkZJRgABAQEAYABgAAD/...",
                        "iVBORw0KGgoAAAANSUhEUgAAABkAAAAZ...",
                    ]
                }
            }
        }
