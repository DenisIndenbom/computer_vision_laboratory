import json


class History:
    def __init__(self, epoch: int):
        self._history: dict[str, list[int | float | None]] = {}
        self._current_epoch = epoch

    def next_epoch(self):
        self._current_epoch += 1

    def add(self, key: str, value: int | float):
        if key not in self._history:
            self._history[key] = [None] * (self._current_epoch - 1)
        elif len(self._history[key]) > self._current_epoch:
            print(self._current_epoch)
            print(self._history)
            raise Exception(f'Key {key} had already been added at this epoch!')

        self._history[key].append(value)

    def add_many(self, metrics: dict[str, int | float], prefix=''):
        for name, value in metrics.items():
            self.add(prefix+name, value)

    def load(self, path: str):
        with open(path, 'r') as file:
            self._history = json.load(file)

    def dump(self, path: str):
        with open(path, 'w') as file:
            json.dump(self._history, file)

    def to_dict(self) -> dict[str, list[int | float | None]]:
        return self._history
