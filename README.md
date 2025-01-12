# Frostfire Stock Analysis AI Hub

An AI-powered hub for advanced stock analysis, featuring:
- **Chart Detection**: Classify images as charts or non-charts using a TensorFlow-based DenseNet model.
- **Email Parsing**: Extract stock symbols from emails with an LLM powered by LangChain and Ollama.
- **DuckDuckGo Search**: Perform quick internet searches for additional insights.

This project uses **FastAPI** for a robust backend, supports Docker-based deployment, and is compatible with ARM devices like Raspberry Pi.

---

## Features

1. **Chart Detection API**:
   - Upload Base64-encoded images to classify charts.
   - Built with a TensorFlow DenseNet model.

2. **Email Parsing API**:
   - Analyze email content to extract stock symbols.
   - Powered by a custom Large Language Model (LLM) using LangChain.

3. **DuckDuckGo Search API**:
   - Perform quick searches with relevant results.

4. **Dockerized Deployment**:
   - Easily run the project in Docker containers.
   - Supports Raspberry Pi (ARM architecture).

5. **Health Check Endpoint**:
   - Verify server and model readiness.

---

## Installation

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- TensorFlow compatible with your architecture
- Optionally, Raspberry Pi for ARM deployment

### Clone the Repository
```bash
git clone https://github.com/kjpou1/frostfire-stock-analysis-ai-hub.git
cd frostfire-stock-analysis-ai-hub
```

### Install Dependencies
Create a virtual environment and install the required packages:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file with the following content:
```plaintext
MODEL_PATH=./tf_models/densenet_classifier.keras
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=your-ollama-model-name
HOST=0.0.0.0
PORT=8000
```

---

## Run Locally

### Using Python
```bash
python run.py
```

### Using Docker
Build and run the Docker container:
```bash
docker build -t frostfire-ai .
docker run -d -p 8000:8000 --env-file .env frostfire-ai
```

For ARM (e.g., Raspberry Pi):
```bash
docker buildx build --platform linux/arm64 -t frostfire-ai .
```

---

## API Endpoints

### **1. Detect Charts**
- **Endpoint**: `POST /detect-charts/`
- **Payload**:
    ```json
    {
        "payload": {
            "data": [
                "base64_image_1",
                "base64_image_2"
            ]
        }
    }
    ```
- **Response**:
    ```json
    {
        "code": 0,
        "code_text": "ok",
        "message": "Processed successfully.",
        "data": [
            {"index": 0, "is_chart": true},
            {"index": 1, "is_chart": false}
        ]
    }
    ```

### **2. Analyze Email**
- **Endpoint**: `POST /analyze-email/`
- **Payload**:
    ```json
    {
        "email_text": "Please review these stocks: AAPL, TSLA."
    }
    ```
- **Response**:
    ```json
    {
        "symbols": ["AAPL", "TSLA"]
    }
    ```

### **3. DuckDuckGo Search**
- **Endpoint**: `GET /search/`
- **Query Params**: `query=<your-search-term>`
- **Response**:
    ```json
    {
        "query": "latest stock news",
        "results": [
            {"title": "Stock Market Today", "link": "https://example.com", "snippet": "Latest market trends..."}
        ]
    }
    ```

---

## Development and Testing

### Run Unit Tests
Install `pytest` and run the tests:
```bash
pip install pytest
pytest tests/
```

---

## Deployment

### Deploy on Raspberry Pi
1. Build the Docker image:
    ```bash
    docker buildx build --platform linux/arm64 -t frostfire-ai .
    ```

2. Transfer the image or use Docker Hub:
    ```bash
    docker push <your-dockerhub-username>/frostfire-ai
    ```

3. Run the container:
    ```bash
    docker run -d -p 8000:8000 --env-file .env frostfire-ai
    ```

---

## Future Enhancements
- Integrate real-time stock price analysis.
- Expand chart detection to support multiple chart types.
- Add advanced LLM-based analytics.

---

## License
This project is licensed under the MIT License.

---

## Contributors
- [Kenneth Pouncey](https://github.com/kjpou1)

