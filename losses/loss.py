import torch.nn as nn


class HyConvKTLoss(nn.Module):
    """BCE with logits loss for knowledge tracing with optional L2 regularization."""

    def __init__(self, l2_weight=1e-4):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.l2_weight = l2_weight

    def forward(self, logits, targets, model=None):
        return self.bce(logits, targets)
