from src.data_loader import get_dataloaders
from src.model import get_model
from src.train import train_model
from src.evaluate import evaluate_model

import torch

train_dir = "NEU-DET/train/images"
val_dir = "NEU-DET/validation/images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, val_loader, classes = get_dataloaders(train_dir, val_dir)

# Load model
model = get_model(num_classes=len(classes))

# Train
model, losses = train_model(model, train_loader, device)

# Evaluate
evaluate_model(model, val_loader, device)