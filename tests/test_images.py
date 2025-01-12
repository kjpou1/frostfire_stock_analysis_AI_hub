import base64

import requests

# Encode an image to Base64
with open("path/to/image.jpg", "rb") as img_file:
    base64_string = base64.b64encode(img_file.read()).decode("utf-8")

# API endpoint
url = "http://localhost:8000/detect-charts/"

# Payload
payload = {"base64_images": [base64_string]}

# Send the request
response = requests.post(url, json=payload)

# Print the response
print(response.json())
