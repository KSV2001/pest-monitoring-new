from .pest_default_dataset import PestDefaultDataset


class TrapValidationDataset(PestDefaultDataset):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def __getitem__(self, idx):
        record = super().__getitem__(idx)
        x, y = record["img"], int(record["val_class"].item())
        return (x, y)
