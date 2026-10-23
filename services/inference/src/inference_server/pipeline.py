import asyncio
import logging
import time
from pathlib import Path

from inference_server.preprocessing import load_clips
from inference_server.schemas import ClipPrediction, Prediction, PredictRequest

logger = logging.getLogger(__name__)


class TooManyFailedClips(RuntimeError):
    """Too many clips of one file failed in the model to give a reliable answer."""


class Pipeline:
    """This class define the logic and pipeline of the inference server.
    It is responsible for loading the audio file,
    preprocessing it, and sending it to the model for inference.
    """

    def __init__(self, batcher, config):
        self.batcher = batcher

        self.preprocessing_func = load_clips

        self.sample_rate = config.sample_rate
        self.clip_seconds = config.clip_seconds
        self.min_tail_seconds = config.min_tail_seconds
        self.upload_dir = config.upload_dir

        self.positive_threshold = config.positive_threshold
        self.clip_threshold = config.clip_threshold
        self.max_failed_clip_ratio = config.max_failed_clip_ratio
        self.model_name = config.detector

    async def process_request(self, request: PredictRequest) -> Prediction:
        """Process the request and return the result.

        Raises AudioError (unreadable file) or TooManyFailedClips; the route maps them to HTTP.
        """
        start = time.perf_counter()

        # Submit the audio to preprocessing (in a thread: decoding would block the event loop)
        clips = await asyncio.to_thread(
            self.preprocessing_func,
            request.path,
            self.sample_rate,
            self.clip_seconds,
            self.min_tail_seconds,
            self.upload_dir,
        )
        filename = Path(request.path).name
        logger.debug(
            "Preprocessed %s into %d clips in %.0f ms",
            filename,
            len(clips),
            (time.perf_counter() - start) * 1000,
        )

        # Submit the clips to the batcher, one item per clip
        tasks = [self.batcher.submit(clip) for clip in clips]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        inference_ms = (time.perf_counter() - start) * 1000
        prediction = self._poll_decision(results, inference_ms)
        logger.info(
            "Predicted %s: label=%s fake_ratio=%.2f clips=%d failed=%d (%.0f ms)",
            filename,
            prediction.label,
            prediction.fake_ratio,
            prediction.num_clips,
            prediction.failed_clip_count,
            prediction.inference_ms,
        )
        return prediction

    def _poll_decision(self, results: list, inference_ms: float) -> Prediction:
        """The function holding the logic for the final decision of the inference server."""

        # if more than max_failed_clip_ratio of the clips failed, we return an error
        failures = [result for result in results if isinstance(result, Exception)]
        failed_count = len(failures)
        if failures:
            logger.warning(
                "%d/%d clips failed, first error: %r", failed_count, len(results), failures[0]
            )
        if failed_count / len(results) > self.max_failed_clip_ratio:
            raise TooManyFailedClips(f"{failed_count}/{len(results)} clips failed")

        # Label each successful clip from its score (clip index = position in the file)
        clips = [
            ClipPrediction(
                index=i,
                score=score.score,
                label="fake" if score.score >= self.clip_threshold else "real",
            )
            for i, score in enumerate(results)
            if not isinstance(score, Exception)
        ]

        # if more than X% of the clips are positive, return positive
        positive_count = sum(1 for clip in clips if clip.label == "fake")
        positive_ratio = positive_count / len(clips)

        return Prediction(
            label="fake" if positive_ratio >= self.positive_threshold else "real",
            fake_ratio=positive_ratio,
            failed_clip_count=failed_count,
            num_clips=len(results),
            clips=clips,
            model=self.model_name,
            inference_ms=inference_ms,
        )
