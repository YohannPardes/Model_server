import logging

from inference_server.config import get_config
from inference_server.detectors.mock import MockDetector

logger = logging.getLogger(__name__)


def load_model():
    """Load the model once and return it."""
    # Placeholder for model loading logic
    config = get_config()
    logger.info("Loading detector %r", config.detector)

    if config.detector == "mock":
        return MockDetector(config=config)

    raise ValueError(f"Unknown model name: {config.detector}")