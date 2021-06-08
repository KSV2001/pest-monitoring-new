import torch
from detectron2.model_zoo import get


class Detectron2Network(torch.nn.Module):
    def __init__(self, config_path: str, trained: bool = False):
        super().__init__()
        self.network = get(config_path, trained)

    def forward(self, x):
        return self.network(x)
