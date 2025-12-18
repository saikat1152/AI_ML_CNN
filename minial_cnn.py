"""
FIXED MINIMAL PYTORCH CNN - No shape errors
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")


# FIXED MODEL - Correct dimensions
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 3x64x64
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 32x64x64
        self.pool = nn.MaxPool2d(2, 2)  # 32x32x32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 64x32x32
        # After pool: 64x16x16
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # 128x16x16
        # After pool: 128x8x8

        # FIXED: Calculate correct flattened size
        # After 3 pooling layers (64→32→16→8): 8x8 feature maps
        # 128 channels * 8 * 8 = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 64x64 → 32x32
        x = self.pool(torch.relu(self.conv2(x)))  # 32x32 → 16x16
        x = self.pool(torch.relu(self.conv3(x)))  # 16x16 → 8x8

        # Flatten: (batch_size, 128, 8, 8) → (batch_size, 128*8*8)
        x = x.view(x.size(0), -1)  # FIX: Use -1 for automatic calculation

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Or use this EVEN SIMPLER model:
class EvenSimplerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Input: 3x64x64
            nn.Conv2d(3, 16, 3, padding=1),  # 16x64x64
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x32x32

            nn.Conv2d(16, 32, 3, padding=1),  # 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x16x16

            nn.Conv2d(32, 64, 3, padding=1),  # 64x16x16
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x8x8

            # Flatten automatically
            nn.Flatten(),

            # 64*8*8 = 4096
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


# Data preparation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10
train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

# Split into train/val
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Class names
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Initialize model (use the simpler one)
model = EvenSimplerCNN().to(device)  # This one won't have shape errors!

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")

# Simple training loop
print("\nTraining...")
for epoch in range(5):  # Just 5 epochs for quick test
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    print(f"Epoch {epoch + 1}: Loss: {running_loss / len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")

# Test the model
print("\nTesting...")
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = 100 * test_correct / test_total
print(f"Test Accuracy: {test_acc:.2f}%")

# Show some predictions
print("\nSample predictions:")
model.eval()
with torch.no_grad():
    # Get one batch
    images, labels = next(iter(test_loader))
    images, labels = images[:8].to(device), labels[:8].to(device)
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)

    # Move back to CPU for plotting
    images = images.cpu()

    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        ax = axes[i // 4, i % 4]
        img = images[i].permute(1, 2, 0).numpy()
        img = img * 0.5 + 0.5  # Unnormalize from [-1,1] to [0,1]
        ax.imshow(img)

        true_label = classes[labels[i]]
        pred_label = classes[predictions[i]]
        color = 'green' if predictions[i] == labels[i] else 'red'

        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=10)
        ax.axis('off')

    plt.suptitle(f'Model Accuracy: {test_acc:.1f}%', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Save the model
torch.save(model.state_dict(), 'simple_cnn_fixed.pth')
print("Model saved as 'simple_cnn_fixed.pth'")