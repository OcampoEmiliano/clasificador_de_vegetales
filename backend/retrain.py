"""Re-train the MLP model using the local Vegetables/ dataset.
Uses a subset of data for quick training; increase SUBSET to use all data.

Usage: uv run python retrain.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from model import MLPClassifier, INPUT_SIZE, NUM_CLASSES, DEVICE

DATA_DIR = r"C:\Users\ocamp\Desktop\vision por computadora\Vegetables"
BATCH_SIZE = 32
EPOCHS = 5
SUBSET = 100  # images per class (0 = all)

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

full_train = datasets.ImageFolder(f"{DATA_DIR}/train", transform=data_transforms)
full_test = datasets.ImageFolder(f"{DATA_DIR}/test", transform=data_transforms)

def subsample(dataset, n):
    if n <= 0:
        return dataset
    indices = []
    targets = torch.tensor(dataset.targets)
    for c in range(len(dataset.classes)):
        mask = targets == c
        idx = mask.nonzero(as_tuple=True)[0]
        indices.extend(idx[:n].tolist())
    return Subset(dataset, indices)

train_set = subsample(full_train, SUBSET)
test_set = subsample(full_test, SUBSET // 2)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

print(f"Classes: {full_train.classes}")
print(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")
print(f"Device: {DEVICE}")

model = MLPClassifier(INPUT_SIZE, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}/{len(train_loader)}")

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%")

torch.save(model.state_dict(), "modelo_vegetales.pth")
print(f"Model saved to modelo_vegetales.pth")
