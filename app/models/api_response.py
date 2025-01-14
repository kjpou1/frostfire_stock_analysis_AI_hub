import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class APIResponse(BaseModel):
    """
    Response model for all API endpoints.
    """

    code: int
    code_text: str
    message: str
    data: Optional[Any] = None
    timestamp: str
    request_id: str
