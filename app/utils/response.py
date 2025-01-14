import uuid
from datetime import datetime
from typing import Optional

from app.models.api_response import APIResponse  # Adjust the import path as needed


def create_response(
    code: int, code_text: str, message: str, data: Optional[dict] = None
) -> APIResponse:
    """
    Create a standardized response for API endpoints.

    Args:
        code (int): Numeric code representing the return status.
        code_text (str): Text description, e.g., "ok" or "error".
        message (str): Short description of the API call result.
        data (Optional[dict]): The result of the call (default is None).

    Returns:
        APIResponse: Standardized API response model instance.
    """
    return APIResponse(
        code=code,
        code_text=code_text,
        message=message,
        data=data,
        timestamp=datetime.utcnow().isoformat(),
        request_id=str(uuid.uuid4()),
    )
