from .dataloader import FoldDataLoader
from pathlib import Path

class ShenzenDataLoader(FoldDataLoader):
    def __init__(self, batch_size, lazy_load = False, shuffle = False):
        super().__init__(batch_size, lazy_load, shuffle)

        self.splits_path = Path("")
        self.csv_path = Path("")

        self._make_dataset()