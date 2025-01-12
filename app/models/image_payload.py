from pydantic import BaseModel, Field, ValidationError


class ImagePayload(BaseModel):
    payload: dict = Field(
        ..., description="The payload containing Base64-encoded image representations."
    )

    def __init__(self, raw_body: dict):
        """
        Custom initialization to validate and extract the payload.
        """
        # Validate the raw body structure
        if not isinstance(raw_body, dict) or "payload" not in raw_body:
            raise ValueError("Missing 'payload' key in request body.")

        payload = raw_body.get("payload")
        if not isinstance(payload, dict) or "data" not in payload:
            raise ValueError("Missing 'data' key in 'payload'.")
        if not isinstance(payload["data"], list):
            raise ValueError("'data' must be a list of Base64-encoded strings.")

        # Call the parent constructor with the validated payload
        super().__init__(payload=payload)

    @property
    def base64_images(self):
        """
        Extract the list of Base64-encoded image strings from the payload.
        """
        return self.payload["data"]
