import torch
import torchvision.transforms as transforms

from src.data.utils import Encoder


class SSDTransformer(object):
    def __init__(self, dboxes, size=(300, 300), test=False):
        self.size = size
        self.test = test
        self.dboxes = dboxes
        self.encoder = Encoder(self.dboxes)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.img_trans = transforms.Compose(
            [
                transforms.Resize(self.size),
                transforms.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
        self.trans_test = transforms.Compose(
            [transforms.Resize(self.size), transforms.ToTensor(), self.normalize]
        )

    def __call__(self, img, bboxes=None, labels=None, max_num=200):
        if self.test:
            bbox_out = torch.zeros(max_num, 4)
            label_out = torch.zeros(max_num, dtype=torch.long)
            bbox_out[: bboxes.size(0), :] = bboxes
            label_out[: labels.size(0)] = labels
            return self.trans_test(img), bbox_out, label_out

        img = self.img_trans(img).contiguous()
        bboxes, labels = self.encoder.encode(bboxes, labels)

        return img, bboxes, labels
