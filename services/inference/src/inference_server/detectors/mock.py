import logging
import random
import time

from inference_server.detectors.base import BaseDetector
from inference_server.schemas import ModelPrediction

logger = logging.getLogger(__name__)

class MockDetector(BaseDetector):
    """
    A mock detector that simulates model predictions for testing purposes.
    """

    def __init__(self, config):
        self.config = config

    def predict(self, batch) -> list[ModelPrediction]:
        """
        Simulate predictions for a list of audio clips with artificial delay.

        Args:
            batch (List[AudioClip]): List of audio clips to predict.
        """
        # Simulate processing time
        time.sleep(self.config.mock_batch_overhead + len(batch) * self.config.mock_per_item_delay)

        # Simulate a model failure, to exercise the retry path (503 → worker retries)
        if random.random() < self.config.mock_failure_rate:
            logger.warning("Simulated model failure (batch of %d)", len(batch))
            raise RuntimeError("Simulated model failure")

        predictions = [random.random() for _ in batch]
        return [ModelPrediction(score=score) for score in predictions]
