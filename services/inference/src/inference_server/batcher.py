import asyncio
import contextlib
import logging
import time

import torch

logger = logging.getLogger(__name__)


class Batcher:

    def __init__(self, model, config):
        self.model = model
        self.max_batch_size = config.max_batch_size
        self.max_wait_ms = config.max_batch_wait_ms

        self._queue = asyncio.Queue(maxsize=config.max_queue_size)
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        # Start the batcher loop
        self._task = asyncio.create_task(self._run())
        logger.info("Batcher started")

    async def stop(self) -> None:
        # Stop the batcher loop
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

        # Fail requests still waiting in the queue, so no caller hangs
        pending = 0
        while not self._queue.empty():
            _, future = self._queue.get_nowait()
            if not future.done():
                future.set_exception(RuntimeError("Batcher stopped"))
                pending += 1
        logger.info("Batcher stopped (%d pending requests failed)", pending)

    async def submit(self, item):
        future = asyncio.get_running_loop().create_future()
        self._queue.put_nowait((item, future))  # raises asyncio.QueueFull when full
        return await future

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()

        while True:
            batch = []
            futures = []

            # Wait for the first item in the queue
            item, future = await self._queue.get()
            batch.append(item)
            futures.append(future)

            # Wait for more items to fill the batch or until max_wait_ms is reached
            deadline = loop.time() + self.max_wait_ms / 1000
            while len(batch) < self.max_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    item, future = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except TimeoutError:
                    break
                batch.append(item)
                futures.append(future)

            # Process the batch in a thread, so the event loop keeps accepting requests
            start = time.perf_counter()
            try:
                results = await asyncio.to_thread(self._predict, batch)
                if len(results) != len(futures):
                    raise RuntimeError(
                        f"Model returned {len(results)} results for {len(futures)} items"
                    )
            except Exception as e:
                logger.exception("Batch of %d items failed", len(batch))
                # If the model fails, set the error for each future
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
                continue

            logger.info(
                "Ran batch of %d in %.0f ms (%d waiting in queue)",
                len(batch),
                (time.perf_counter() - start) * 1000,
                self._queue.qsize(),
            )

            # Set the results for each future
            for future, result in zip(futures, results, strict=True):
                if not future.done():
                    future.set_result(result)

    def _predict(self, batch: list):
        # Runs in a worker thread: stack the items and call the model
        return self.model.predict(self._list_to_tensor(batch))

    def _list_to_tensor(self, batch: list) -> torch.Tensor:
        """Stack same-length items into one (batch_size, *item_shape) tensor.

        Preprocessing guarantees every item has the same shape (fixed clip length).
        """
        return torch.stack([torch.as_tensor(item, dtype=torch.float32) for item in batch])
