import torch
import torch.nn as nn
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weights_path: str | None = None):
    model = models.efficientnet_b0(pretrained=True)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

    model.to(DEVICE)
    model.eval()
    return model
