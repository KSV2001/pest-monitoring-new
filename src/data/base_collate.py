import torch


def default_collate(batch):
    """
    Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    """

    records = {
        "img_ids": [],
        "imgs": [],
        "bbox_coords": [],
        "bbox_classes": [],
        "val_classes": [],
    }

    for record in batch:
        records["img_ids"].append(record["img_id"])
        records["imgs"].append(record["img"])
        records["bbox_coords"].append(record["bbox_coord"])
        records["bbox_classes"].append(record["bbox_class"])
        records["val_classes"].append(record["val_class"])

    records["imgs"] = torch.stack(records["imgs"], 0)

    return records
