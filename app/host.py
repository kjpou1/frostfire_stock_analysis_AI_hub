import asyncio
import base64
import logging
import os
from contextlib import asynccontextmanager
from io import BytesIO
from typing import List, Optional

import numpy as np
import tensorflow as tf
import uvicorn
from duckduckgo_search import DDGS
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from PIL import Image

from app.models.image_payload import ImagePayload
from app.models.model_load_error import ModelLoadError
from app.utils import preprocess

logger = logging.getLogger(__name__)


def get_chart_detect_model(app: FastAPI):
    """
    Retrieve the loaded model from FastAPI's application state.
    """
    if app.state.chart_detect_model:
        return app.state.chart_detect_model
    else:
        return None


def get_ollama_llm():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL")
    if not model_name:
        raise ValueError("OLLAMA_MODEL environment variable is not set.")
    return ChatOllama(base_url=base_url, model=model_name)


async def load_chart_model():
    model_path = os.getenv("MODEL_PATH", "./tf_models/densenet_classifier.keras")
    model = None
    try:
        if not os.path.exists(model_path):
            logger.error("Model file not found at path: %s", model_path)
            raise ModelLoadError("Model file not found.")

        logger.info("Loading model from %s...", model_path)
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully from %s.", model_path)
    except FileNotFoundError as exc:
        # Explicitly chain the FileNotFoundError to ModelLoadError
        logger.error("Model file not found at path: %s", model_path)
        raise ModelLoadError("Model file not found.") from exc
    except Exception as exc:
        # Explicitly chain other exceptions to ModelLoadError
        logger.error("Failed to load model from %s. Error: %s", model_path, exc)
        raise ModelLoadError("Failed to load the model.") from exc
    return model


async def load_ollama_llm():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL")
    if not model_name:
        raise ValueError("OLLAMA_MODEL environment variable is not set.")
    return ChatOllama(base_url=base_url, model=model_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown logic.
    """
    # Startup logic
    logger.info("Starting application...")
    app.state.chart_detect_model = await load_chart_model()
    app.state.ollama_llm = await load_ollama_llm()
    logger.info("Application started successfully.")

    # Yield control to the application
    yield

    # Shutdown logic (if needed)
    logger.info("Shutting down resources...")


class Host:
    def __init__(self, args: None):
        """
        Initialize the Host class for Frostfire Stock Analysis AI Hub.
        """
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", 8000))
        # self.densenet_model_path = os.getenv(
        #     "MODEL_PATH", "tf_models/densenet_model.keras"
        # )
        # self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # self.ollama_model = os.getenv("OLLAMA_MODEL")

        # Load the DenseNet model
        # self.chart_detect_model = self.load_model(self.densenet_model_path)

        # Initialize Ollama LLM
        # if not self.ollama_model:
        #     raise ValueError("OLLAMA_MODEL environment variable is not set.")
        # self.ollama_llm = ChatOllama(
        #     base_url=self.ollama_base_url, model=self.ollama_model
        # )

        # Initialize FastAPI
        self.app = FastAPI(
            title="Frostfire Stock Analysis AI Hub", version="1.0", lifespan=lifespan
        )
        self.setup_routes()

    def setup_routes(self):
        """
        Set up FastAPI routes.
        """

        @self.app.post("/detect-charts/")
        async def detect_charts(
            request: Request,
            chart_detect_model: tf.keras.Model = Depends(
                lambda: get_chart_detect_model(self.app)
            ),
        ):
            """
            Detect if Base64-encoded images are charts using DenseNet.

            Parameters:
            - request (Request): Raw request data.

            Returns:
            - dict: Structured response with code, code_text, message, and data.
            """
            try:
                # Parse and validate the payload using ImagePayload
                body = await request.json()
                payload = ImagePayload(body)
                base64_images = payload.base64_images
            except ValueError as e:
                self.logger.error("Invalid request payload: %s", e)
                return {
                    "code": 400,
                    "code_text": "error",
                    "message": str(e),
                    "data": None,
                }

            results = []

            for index, base64_image in enumerate(base64_images):
                try:
                    # Decode the Base64 image
                    self.logger.info("Processing Base64 image #%d", index + 1)
                    image_data = base64.b64decode(base64_image)
                    image = Image.open(BytesIO(image_data))

                    # Preprocess the image
                    processed_image = preprocess.preprocess_image(image)

                    # Predict using the DenseNet model
                    self.logger.info("Before predict for Base64 image #%d", index + 1)
                    prediction = chart_detect_model.predict(processed_image)
                    # prediction = self.chart_detect_model.predict(processed_image)
                    is_chart = bool(prediction[0][0] <= 0.5)

                    # Append result
                    results.append({"index": index, "is_chart": is_chart})
                except Exception as e:
                    # Log and append error details
                    self.logger.error(
                        "Error processing Base64 image #%d: %s", index + 1, e
                    )
                    results.append({"index": index, "error": str(e)})

            # Structured response
            return {
                "code": 0,
                "code_text": "ok",
                "message": "Processed successfully.",
                "data": results,
            }

        @self.app.post("/analyze-email/")
        async def analyze_email(email_text: str = Form(...)):
            """
            Extract stock symbols from an email using Ollama LLM.
            """
            try:
                prompt = ChatPromptTemplate.from_template(
                    "Extract stock symbols from the following email: {email_text}"
                )
                response = self.app.state.ollama_llm.run(
                    prompt.render(email_text=email_text)
                )
                return {"symbols": response.split()}
            except Exception as e:
                self.logger.error(f"Error analyzing email: {e}")
                raise HTTPException(status_code=500, detail="Failed to process email.")

        @self.app.get("/search/")
        async def search(query: str):
            """
            Perform a DuckDuckGo search.
            """
            try:
                with DDGS() as ddgs:
                    results = [
                        {"title": r["title"], "link": r["href"], "snippet": r["body"]}
                        for r in ddgs.text(
                            query, region="wt-wt", safesearch="moderate", max_results=5
                        )
                    ]
                return {"query": query, "results": results}
            except Exception as e:
                self.logger.error("Error during DuckDuckGo search: %s", e)
                raise HTTPException(status_code=500, detail="Failed to perform search.")

    def run(self):
        """
        Asynchronous method to start both MQTT and FastAPI server concurrently.
        """
        self.logger.info("Starting host process.")
        fastapi_task = None  # Initialize fastapi_task to None

        try:
            # # Start the heartbeat task
            # heartbeat_task = asyncio.create_task(self.mqtt_service.heartbeat())

            # Start FastAPI server as a task
            fastapi_task = asyncio.run(self.start_fastapi())

            # # Keep the process running until interrupted
            # while True:
            #     await asyncio.sleep(1)

        except asyncio.CancelledError:
            self.logger.info("Stopping host process.")
        finally:
            if fastapi_task:  # Check if fastapi_task is initialized
                fastapi_task.cancel()
                fastapi_task
            # await self.mqtt_service.shutdown()

    async def start_fastapi(self):
        """
        Run the FastAPI server asynchronously.
        """
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port, log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


if __name__ == "__main__":
    try:
        host = Host(None)
    except ModelLoadError as e:
        logging.error("Critical error: %s. Application cannot start.", e)
