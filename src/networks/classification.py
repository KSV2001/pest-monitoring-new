import torch

class ClassificationNet(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        _infeatures = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = torch.nn.Linear(_infeatures, num_classes)
    def forward(self, x):
        return self.backbone(x)
