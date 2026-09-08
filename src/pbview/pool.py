"""
Manage a pool of workers with a limited queue size.
"""

__author__ = "Per Unneberg"
__contact__ = "per.unneberg@scilifelab.se"
__data__ = "2026-09-01"

from threading import BoundedSemaphore

from .logging import app_logger as logger


class MaxQueuePool:
    """This Class wraps a concurrent.futures.Executor limiting the
    size of its task queue.

    If `max_queue_size` tasks are submitted, the next call to submit
    will block until a previously submitted one is completed.

    see https://gist.github.com/noxdafox/4150eff0059ea43f6adbdd66e5d5e87e
    """

    def __init__(
        self, executor, *, max_queue_size: int, max_workers: int | None = None
    ) -> None:
        logger.info(
            "Initializing queue with %i queue slots, %i workers",
            max_queue_size,
            max_workers,
        )
        self.pool = executor(max_workers=max_workers)
        self.pool_queue: BoundedSemaphore = BoundedSemaphore(max_queue_size)

    def submit(self, fn, *args, **kwargs):
        """Submit a new task to the pool. This will block if the queue
        is full"""
        self.pool_queue.acquire()  # pylint: disable=consider-using-with
        future = self.pool.submit(fn, *args, **kwargs)
        future.add_done_callback(self.pool_queue_callback)

        return future

    def pool_queue_callback(self, _) -> None:
        """Called when a future is done. Releases one queue slot."""
        self.pool_queue.release()
