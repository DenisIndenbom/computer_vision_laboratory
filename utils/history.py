import json
from collections import defaultdict
from copy import deepcopy


class History:
    def __init__(self, epoch: int):
        self._history: dict[str, list[int | float | None]] = defaultdict(list)
        self._batches: dict[str, list[int | float]] = defaultdict(list)
        self._current_epoch = epoch

    def add_batch(self, metrics: dict[str, int | float]):
        for name, value in metrics.items():
            self._batches[name].append(value)

    def add(self, key: str, value: int | float):
        if key not in self._history:
            self._history[key] = [None] * (self._current_epoch - 1)
        elif len(self._history[key]) >= self._current_epoch:
            raise Exception(f'Key {key} already has a value for epoch {self._current_epoch}')

        self._history[key].append(value)

    def add_many(self, metrics: dict[str, int | float], prefix: str = ''):
        for name, value in metrics.items():
            self.add(prefix + name, value)

    def commit(self, prefix: str = '') -> dict[str, int | float]:
        avg = {}

        for key, value in self._batches.items():
            avg[key] = sum(value) / len(value)
            self.add(prefix + key, avg[key])
        self._batches.clear()

        return avg

    def next_epoch(self):
        self._current_epoch += 1
        self._batches.clear()

    def load(self, path: str):
        with open(path, 'r') as file:
            self._history = json.load(file)

    def dump(self, path: str):
        with open(path, 'w') as file:
            json.dump(self._history, file)

    def to_dict(self) -> dict[str, list[int | float | None]]:
        return deepcopy(self._history)

    def __repr__(self):
        return f'History(epoch={self._current_epoch}, keys={list(self._history.keys())})'
