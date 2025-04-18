import torch
import torch.nn as nn

def ProtoNet(input_dim=(3, 224, 224)):
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Add 3 more blocks for 4 total
        nn.Flatten(),
        nn.Linear(64*14*14, 64)  # Adjust based on pooling steps
    )