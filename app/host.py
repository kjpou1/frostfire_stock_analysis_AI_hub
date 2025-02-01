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

from app.config.config import Config
from app.models.api_response import APIResponse
from app.models.image_payload import ImagePayload
from app.models.model_load_error import ModelLoadError
from app.utils import preprocess
from app.utils.response import create_response

logger = logging.getLogger(__name__)

CONFIG = Config()


def get_chart_detect_model(app: FastAPI):
    """
    Retrieve the loaded model from FastAPI's application state.
    """
    if app.state.chart_detect_model:
        return app.state.chart_detect_model
    else:
        return None


def get_ollama_llm():
    base_url = CONFIG.ollama_base_url
    model_name = CONFIG.ollama_model
    if not model_name:
        raise ValueError("OLLAMA_MODEL environment variable is not set.")
    return ChatOllama(base_url=base_url, model=model_name)


async def load_chart_model():
    model_path = CONFIG.model_path
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
    base_url = CONFIG.ollama_base_url
    model_name = CONFIG.ollama_model
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
        self.host = CONFIG.host
        self.port = CONFIG.port

        # Initialize FastAPI
        self.app = FastAPI(
            title="Frostfire Stock Analysis AI Hub", version="1.0", lifespan=lifespan
        )
        self.setup_routes()

    def setup_routes(self):
        """
        Set up FastAPI routes.
        """

        @self.app.post("/detect-charts/", response_model=APIResponse)
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
                return create_response(400, "error", str(e))

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
                    prediction = chart_detect_model.predict(processed_image)
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
            return create_response(0, "ok", "Processed successfully.", results)

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

        @self.app.get("/health")
        async def health_check():
            """
            Health check endpoint to verify that the model and LLM are loaded properly.
            """
            try:
                # Check if the model is loaded
                if (
                    not hasattr(self.app.state, "chart_detect_model")
                    or not self.app.state.chart_detect_model
                ):
                    raise ValueError("Chart detection model is not initialized.")

                # Check if the LLM is loaded
                if (
                    not hasattr(self.app.state, "ollama_llm")
                    or not self.app.state.ollama_llm
                ):
                    raise ValueError("Ollama LLM is not initialized.")

                return create_response(
                    0,
                    "ok",
                    "All services are running.",
                    {"chart_detect_model": "loaded", "ollama_llm": "loaded"},
                )
            except Exception as e:
                self.logger.error("Health check failed: %s", e)
                return create_response(
                    500,
                    "error",
                    str(e),
                    {"chart_detect_model": "not loaded", "ollama_llm": "not loaded"},
                )

    def run(self):
        """
        Asynchronous method to start both MQTT and FastAPI server concurrently.
        """
        self.logger.info("Starting host process.")
        fastapi_task = None  # Initialize fastapi_task to None

        try:
            # Start FastAPI server as a task
            fastapi_task = asyncio.run(self.start_fastapi())

        except asyncio.CancelledError:
            self.logger.info("Stopping host process.")
        finally:
            if fastapi_task:  # Check if fastapi_task is initialized
                fastapi_task.cancel()
                fastapi_task

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
