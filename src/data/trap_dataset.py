from .pest_default_dataset import PestDefaultDataset


class TrapValidationDataset(PestDefaultDataset):
    """Dataset for just doing Image Classification"""
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
    
    def pull_img_label(self, idx: int):
        """Pulls only the image and the label, for trap classification"""
        img = self.read_img(self.img_paths[idx])
        val_class = self.val_classes[idx]
        return img, val_class

    def __getitem__(self, idx):
        image, y = self.pull_img_label(idx)
        if self.transforms is not None: # specialized transforms
            image = self.transforms(image)
        return (image, y)

    def __getitem__(self, idx):
        record = super().__getitem__(idx)
        x, y = record["img"], int(record["val_class"].item())
        return (x, y)


# TODO: Create a new dataset that modifies __getitem__ to not use albumentation
# transforms

class TrapValidationDatasetV3(PestDefaultDataset):
    """Dataset for just doing Image Classification, optimized."""
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def pull_img_label(self, idx: int):
        """Pulls only the image and the label, for trap classification"""
        img = self.read_img(self.img_paths[idx])
        val_class = self.val_classes[idx]
        return img, val_class

    def __getitem__(self, idx):
        image, y = self.pull_img_label(idx)
        if self.transforms is not None: # specialized transforms
            image = self.transforms(image)
        return (image, y)
    