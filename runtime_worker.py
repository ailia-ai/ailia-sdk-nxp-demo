"""Keep a selected runtime resident; isolate native allocators on switches."""

import multiprocessing as mp

from runtime import worker


class RuntimeWorker:
    def __init__(self, models):
        self.models = str(models)
        self.backend = None
        self.model = None
        self.process = None
        self.requests = self.responses = None

    def submit(self, job):
        if (
            self.process is None
            or self.backend != job["backend"]
            or self.model != job.get("model", "sam")
            or not self.process.is_alive()
        ):
            self.close()
            context = mp.get_context("spawn")
            self.requests, self.responses = context.Queue(), context.Queue()
            self.process = context.Process(
                target=worker, args=(self.models, self.requests, self.responses), daemon=True
            )
            self.process.start()
            self.backend = job["backend"]
            self.model = job.get("model", "sam")
        # Compare all is sequenced by the UI, one isolated backend at a time.
        self.requests.put(dict(job, single_backend=True))

    def close(self):
        if self.process is not None:
            self.requests.put(None)
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
            self.process.close()
            for channel in (self.requests, self.responses):
                channel.cancel_join_thread()
                channel.close()
        self.process = None
        self.backend = None
        self.model = None
        self.requests = self.responses = None
