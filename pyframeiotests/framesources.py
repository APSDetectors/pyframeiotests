import threading

import numpy as np
import pvaccess
from pvapy.utility.adImageUtility import AdImageUtility
import time


class EmptyFrameSource(threading.Thread):
    def __init__(self, **kwargs):
        self.queue = kwargs.pop('queue')
        self.shape = kwargs.pop('shape')
        self.dtype = kwargs.pop('dtype')
        self.number = kwargs.pop('number')
        self.current_index = 0

        super().__init__(**kwargs)

    def run(self):
        while self.current_index < self.number:
            self.generate()

    def generate(self):
        new_frame = np.empty(shape=self.shape, dtype=self.dtype)
        self.queue.put((self.current_index, new_frame))
        # print(f"Generated frame {self.current_index}")
        self.current_index += 1


class RandomFrameSource(EmptyFrameSource):
    def __init__(self, **kwargs):
        self._rng = np.random.default_rng()
        super().__init__(**kwargs)

    def generate(self):
        new_frame = self._rng.integers(low=np.iinfo(self.dtype).max, dtype=self.dtype, size=self.shape)
        self.queue.put((self.current_index, new_frame))
        # print(f"Generated frame {self.current_index}")
        self.current_index += 1


class ZeroFrameSource(EmptyFrameSource):
    def generate(self):
        new_frame = np.zeros(shape=self.shape, dtype=self.dtype)
        self.queue.put((self.current_index, new_frame))
        # print(f"Generated frame {self.current_index}")
        self.current_index += 1


class PvaPyFrameSource(EmptyFrameSource):
    def __init__(self, **kwargs):
        self.channel = pvaccess.Channel(kwargs.pop('channel'))
        self.lock = threading.Lock()
        super().__init__(**kwargs)

    def run(self):
        self.channel.monitor(self.generate)
        self.channel.startMonitor()
        while self.current_index < self.number:
            time.sleep(0.1)
        print("Received all frames.")
        self.channel.stopMonitor()

    def generate(self, pv):
        if len(pv['value']) == 0:
            return

        id, img, *_ = AdImageUtility.reshapeNtNdArray(pv)

        self.lock.acquire()
        if self.current_index >= self.number:
            self.lock.release()
            return

        self.current_index += 1
        self.lock.release()
        self.queue.put((self.current_index-1, img))

        # print(f"received frame {self.current_index}")
