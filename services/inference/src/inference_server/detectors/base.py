from abc import ABC, abstractmethod

from inference_server.schemas import ModelPrediction


class BaseDetector(ABC):
    """Abstract base class for detectors."""

    @abstractmethod
    def predict(self, batch) -> list[ModelPrediction]:
        """Return one prediction per clip in the batch. Called by the batcher, in a thread."""
