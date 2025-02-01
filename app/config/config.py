import os

from dotenv import load_dotenv

from app.models.singleton import SingletonMeta


class Config(metaclass=SingletonMeta):
    """
    Singleton configuration class for managing project-wide constants and directories.
    """

    _is_initialized = False  # Tracks whether the Config has already been initialized

    def __init__(self):
        # Prevent re-initialization
        if Config._is_initialized:
            return

        # Load environment variables
        load_dotenv()

        # Mark as initialized
        Config._is_initialized = True

    @staticmethod
    def get(key, default=None):
        return os.getenv(key, default)

    @classmethod
    def initialize(cls):
        """
        Explicitly initializes the Config singleton.
        This ensures that the configuration is set up before being used in the application.
        """
        if not cls._is_initialized:
            cls()

    @classmethod
    def is_initialized(cls):
        """
        Checks whether the Config singleton has been initialized.
        Returns:
            bool: True if initialized, False otherwise.
        """
        return cls._is_initialized

    @classmethod
    def reset(cls):
        """
        Resets the Config singleton for testing purposes.
        """
        cls._is_initialized = False
        cls._instances = {}

    @property
    def host(self) -> str:
        """
        Returns the HOST environment variable (defaults to 0.0.0.0)
        """
        return os.getenv("HOST", "0.0.0.0")

    @property
    def port(self) -> int:
        """
        Returns the PORT environment variable (defaults to 8000) as an integer
        """
        return int(os.getenv("PORT", 8000))

    @property
    def ollama_base_url(self) -> str:
        """
        Returns the OLLAMA_BASE_URL environment variable (defaults to http://localhost:11434)
        """
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    @property
    def ollama_model(self) -> str:
        """
        Returns the OLLAMA_MODEL environment variable.
        Raises an exception if it's not set.
        """
        model_name = os.getenv("OLLAMA_MODEL", "")
        if not model_name:
            raise ValueError("OLLAMA_MODEL environment variable is not set.")
        return model_name

    @property
    def model_path(self) -> str:
        """
        Returns the MODEL_PATH environment variable (defaults to ./tf_models/densenet_classifier.keras)
        """
        return os.getenv("MODEL_PATH", "./tf_models/densenet_classifier.keras")
