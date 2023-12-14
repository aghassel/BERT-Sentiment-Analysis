import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, max_length, embed_dim, n_classes, dropout):
        super().__init__()

        self.input_shape = max_length * embed_dim
        if dropout:
            self.model = nn.Sequential(
                nn.Linear(self.input_shape, 128),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(128, n_classes),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(self.input_shape, 128),
                nn.ReLU(),
                nn.Linear(128, n_classes),
            )

    def forward(self, x):
        padded_x = F.pad(x, (0, self.input_shape - x.shape[-1]))
        x = self.model(padded_x)
        return x
