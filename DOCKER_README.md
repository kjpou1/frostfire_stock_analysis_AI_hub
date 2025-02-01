## **Docker Compose Usage for Frostfire Stock Analysis AI Hub**

This project is designed for easy deployment using **Docker Compose**. The following guide provides details on how to build, run, and manage the **Frostfire Stock Analysis AI Hub** container.

---

### **Prerequisites**
Before running the container, ensure you have:
- **Docker** installed: [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose** installed: [Install Docker Compose](https://docs.docker.com/compose/install/)
- A compatible **TensorFlow version** (if running without Docker)

---

### **Docker Compose Configuration**
The `docker-compose.yml` file defines how to run the **Frostfire Stock Analysis AI Hub** as a service.

```yaml
version: "3.8"

services:
  frostfire_stock_AI_hub:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: frostfire_stock_AI_hub
    ports:
      - "8078:8078"
    environment:
      HOST: "0.0.0.0"
      PORT: 8078
      MODEL_PATH: "/app/tf_models/densenet_classifier.keras"
      OLLAMA_BASE_URL: "http://hic-svnt-macbook.local:11434"
      OLLAMA_MODEL: "llama3:8b-instruct-q8_0"
    volumes:
      - ./tf_models:/models
    restart: unless-stopped
```

---

## **Using Docker Compose**

### **1. Build and Run the Container**
Run the following command to build and start the container:
```bash
docker-compose up -d --build
```
- `-d` runs the container in the background.
- `--build` ensures any changes in the `Dockerfile` or dependencies are included.

---

### **2. Verify Running Containers**
Check if the container is running:
```bash
docker ps
```
Expected output:
```
CONTAINER ID   IMAGE                         STATUS         PORTS                    NAMES
xxxxxxxxxx     frostfire_stock_AI_hub        Up X seconds   0.0.0.0:8078->8078/tcp   frostfire_stock_AI_hub
```

---

### **3. Stop, Start, and Restart the Service**
#### **Stop the Service**
To **stop** the container without removing it:
```bash
docker-compose stop frostfire_stock_AI_hub
```

#### **Restart the Service**
To **restart** it:
```bash
docker-compose start frostfire_stock_AI_hub
```

#### **Shut Down and Remove Containers**
To **completely remove** the container:
```bash
docker-compose down
```

---

### **4. Check Logs**
To view real-time logs for debugging:
```bash
docker-compose logs -f frostfire_stock_AI_hub
```
For a one-time log check:
```bash
docker-compose logs frostfire_stock_AI_hub
```

---

### **5. Verify Restart Policy**
Check the restart policy:
```bash
docker inspect frostfire_stock_AI_hub --format='{{.HostConfig.RestartPolicy}}'
```
Expected output:
```json
{"Name":"unless-stopped","MaximumRetryCount":0}
```

If the policy is incorrect, update `docker-compose.yml` and restart.

---

## **Testing the API**
Once the container is running, test the API with **curl** or **Postman**.

#### **Check Health Endpoint**
```bash
curl -X GET "http://localhost:8078/health/" -H "accept: application/json"
```
Expected Response:
```json
{
    "code": 0,
    "code_text": "ok",
    "message": "All services are running.",
    "data": {
        "chart_detect_model": "loaded",
        "ollama_llm": "loaded"
    },
    "timestamp": "2025-02-01T10:25:55.418512",
    "request_id": "7a71b8ed-ac7b-49f6-b52d-aa5f04afc6d0"
}
```

---

### **6. Deploy on Raspberry Pi (ARM Architecture)**
If running on a Raspberry Pi or ARM-based device, build the image for **ARM64**:
```bash
docker buildx build --platform linux/arm64 -t frostfire_stock_AI_hub .
```

To **run it**:
```bash
docker run -d -p 8078:8078 --env-file .env frostfire_stock_AI_hub
```

---

### **7. Deploy to a Remote Server**
If deploying to another machine (e.g., a cloud server or Raspberry Pi), follow these steps:

#### **Push the Image to Docker Hub**
```bash
docker tag frostfire_stock_AI_hub <your-dockerhub-username>/frostfire_stock_AI_hub
docker push <your-dockerhub-username>/frostfire_stock_AI_hub
```

#### **Pull and Run the Image on the Remote Server**
On the remote machine:
```bash
docker pull <your-dockerhub-username>/frostfire_stock_AI_hub
docker-compose up -d
```

---

## **Troubleshooting**
### **1. Check Running Containers**
```bash
docker-compose ps
```
If the container is not running, inspect logs.

### **2. Restart the Container**
```bash
docker-compose restart frostfire_stock_AI_hub
```

### **3. Inspect Logs for Errors**
```bash
docker-compose logs frostfire_stock_AI_hub
```

### **4. Verify API is Reachable**
If the container is running but the API is unreachable, check:
```bash
curl -X GET "http://localhost:8078/health/"
```

---

## **Final Thoughts**
- This `docker-compose.yml` provides a **stable, restartable, and scalable** way to deploy **Frostfire Stock Analysis AI Hub**.
- The `restart: unless-stopped` policy ensures the service remains available unless manually stopped.
- The container can be easily deployed on **Raspberry Pi**, **cloud servers**, or any **Docker-compatible system**.
