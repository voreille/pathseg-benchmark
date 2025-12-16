from torch.utils.data import Dataset


class RepeatDataset(Dataset):
    """Repeat a dataset `repeats` times per epoch.

    Indexing wraps around the underlying dataset with modulo.
    """

    def __init__(self, dataset: Dataset, repeats: int = 1):
        self.dataset = dataset
        self.repeats = repeats

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]
