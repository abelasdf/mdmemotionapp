
import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.onnx

# Parameter
DATA_DIR = "archive/test"
MODEL_PATH = "model/emotion_model.onnx"
NUM_CLASSES = len(next(os.walk(DATA_DIR))[1])
BATCH_SIZE = 16
EPOCHS = 5
IMAGE_SIZE = 64  # Da das ONNX-Modell 64x64 erwartet

# CUDA falls verfügbar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformieren der Bilder
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # da ResNet RGB erwartet
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# Datensatz laden
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Modell laden (ResNet18 + Transfer Learning)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# Loss und Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(dataloader):.4f}")

# ONNX Export
dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
torch.onnx.export(model, dummy_input, MODEL_PATH,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

print(f"✅ Modell gespeichert unter: {MODEL_PATH}")
