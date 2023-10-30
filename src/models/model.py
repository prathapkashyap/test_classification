import torch
import numpy as np
from torch import nn
from src.data.data_loader import fetch_traget_names

class Classifier(nn.Module):
    def __init__(self, num_classes, embedding_size) -> None:
        super(Classifier, self).__init__()
        self.embedding_size = embedding_size
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        return self.fc(x)


