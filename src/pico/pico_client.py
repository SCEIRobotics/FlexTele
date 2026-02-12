import json
from queue import Queue
from datetime import datetime

import zenoh

from loguru import logger


class PicoClient:
    def __init__(self):
        config = zenoh.Config()
        self.session = zenoh.open(config)
        self.sub = self.session.declare_subscriber(
            "teleop/vr/pico_data", self._sub_callback
        )
        self.pico_data_queue = Queue(maxsize=10)

    def _sub_callback(self, sample: zenoh.Sample):
        payload = sample.payload.to_bytes()
        pico_data = json.loads(payload)
        pico_data["timestamp"] = datetime.fromtimestamp(
            pico_data["timestamp"]
        ).strftime("%Y-%m-%d %H:%M:%S.%f")
        if self.pico_data_queue.full():
            logger.warning("Pico data queue is full, drop the oldest data")
            self.pico_data_queue.get()
        self.pico_data_queue.put(pico_data)

    def get_latest(self):
        if self.pico_data_queue.empty():
            return None
        return self.pico_data_queue.get()
