"""Tempi per fase del training, attivabili con SECORS_PROFILE_TIMING=1."""
import csv
import os
from pathlib import Path
import time

import torch


class TrainingTiming:
    phases = ('collect', 'env_step', 'insert', 'returns', 'ppo', 'checkpoint', 'plot')

    def __init__(self, run_dir, device, backend):
        self.enabled = os.environ.get('SECORS_PROFILE_TIMING', '') == '1'
        self.device = device
        self.backend = backend
        self.path = Path(run_dir) / 'training_timing.csv'

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def start_episode(self):
        if self.enabled:
            self._sync()
            self.seconds = dict.fromkeys(self.phases, 0.0)
            self.start = time.perf_counter()

    def call(self, phase, function, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        self._sync()
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            self._sync()
            self.seconds[phase] += time.perf_counter() - start

    def finish_episode(self, episode):
        if not self.enabled:
            return
        total = time.perf_counter() - self.start
        header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open('a', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            if header:
                writer.writerow(['episode', 'device', 'backend', 'total_s'] +
                                [phase + '_s' for phase in self.phases] + ['other_s'])
            writer.writerow([episode, str(self.device), self.backend, total] +
                            [self.seconds[phase] for phase in self.phases] +
                            [total - sum(self.seconds.values())])
