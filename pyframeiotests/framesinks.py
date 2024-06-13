import queue
import threading
import time

import h5py
import hdf5plugin
import numpy as np
import zarr


class NullFrameSink(threading.Thread):
    def __init__(self, **kwargs):
        self.queue = kwargs.pop('queue')
        self.shape = kwargs.pop('shape')
        self.dtype = kwargs.pop('dtype')
        self.number = kwargs.pop('number')
        self.outdir = kwargs.pop('outdir', './out')
        self.processed_frames = 0
        super().__init__(**kwargs)

    def run(self):
        while self.processed_frames < self.number:
            self.consume()

    def consume(self):
        idx, _ = self.queue.get()
        self.processed_frames += 1
        # print(f"Consumed frame {idx}")


class NpzFrameSink(NullFrameSink):
    def __init__(self, **kwargs):
        self.dataset = []
        super().__init__(**kwargs)

    def run(self):
        while self.processed_frames < self.number:
            self.consume()

        d = np.asarray(self.dataset)
        #print(d.shape)
        np.savez_compressed(self.outdir + f'/tmp.{self.name}.npz', d)

    def consume(self):
        idx, f = self.queue.get()
        self.dataset.append(f)
        self.processed_frames += 1
        # print(f"Consumed frame {idx}")


class H5PyFrameSink(NullFrameSink):
    def __init__(self, **kwargs):
        self.compressor = kwargs.pop('compressor', None)
        super().__init__(**kwargs)

    def run(self):
        with h5py.File(self.outdir + f'/tmp.{self.name}.h5', 'w', rdcc_nbytes=1024**2 * 8) as f:
            d = f.create_dataset('data', dtype=self.dtype, shape=(self.number, *self.shape),
                                 chunks=(1, *self.shape))
            while self.processed_frames < self.number:
                self.consume(d)

    def consume(self, d):
        idx, f = self.queue.get()
        d[idx, :, :] = f
        self.processed_frames += 1
        # print(f"Consumed frame {idx}")


class ZarrFrameSink(NullFrameSink):
    def __init__(self, **kwargs):
        self.compressor = kwargs.pop('compressor', None)
        super().__init__(**kwargs)

    def run(self):
        z = zarr.open(self.outdir + f'/tmp.{self.name}.zarr', mode='w', shape=(self.number, *self.shape),
                      dtype=self.dtype, chunks=(1, *self.shape), compressor=self.compressor)
        while self.processed_frames < self.number:
            self.consume(z)

    def consume(self, z):
        idx, f = self.queue.get()
        z[idx, :, :] = f
        self.processed_frames += 1
        # print(f"Consumed frame {idx}")


class ZarrMTFrameSink(NullFrameSink):
    def __init__(self, **kwargs):
        self.nthreads = kwargs.pop('nthreads', 5)
        super().__init__(**kwargs)
        self.processed_frames = 0
        self.lock = threading.Lock()

    def run(self):
        z = zarr.open(self.outdir + f'/tmp.{self.name}.zarr', mode='w', shape=(self.number, *self.shape),
                      dtype=self.dtype, chunks=(1, *self.shape), compressor=None,
                      synchronizer=zarr.ThreadSynchronizer())
        threads = []
        for _ in range(self.nthreads):
            t = threading.Thread(target=self.inner_run, args=(z,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    def inner_run(self, z):
        while self.processed_frames < self.number:
            self.consume(z)

    def consume(self, z):
        try:
            idx, f = self.queue.get(block=False)
        except queue.Empty:
            return
        z[idx, :, :] = f
        self.lock.acquire()
        self.processed_frames += 1
        self.lock.release()
        # print(f"{time.monotonic()}: Thread {multiprocessing.current_process().name}: Consumed frame {idx}, total {self.processed_frames}")
