from contextlib import contextmanager

from dask.distributed import Client, LocalCluster


class DaskClient:
    def __init__(self, workers=2, threads=2, memory_limit="2GB"):
        self.n_workers = workers
        self.threads_per_worker = threads
        self.memory_limit = memory_limit
        self.cluster = None
        self.client = None

    @contextmanager
    def get_client(self):
        self.cluster = LocalCluster(
            n_workers=self.n_workers,
            threads_per_worker=self.threads_per_worker,
            memory_limit=self.memory_limit,
        )
        self.client = Client(self.cluster)

        try:
            yield self.client
        finally:
            self.client.close()
            self.cluster.close()

    def __enter__(self):
        return self.get_client()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
