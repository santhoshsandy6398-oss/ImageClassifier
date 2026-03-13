import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Step 1: Transform images (make them very small to save memory)
transform = transforms.Compose([
    transforms.Resize((32, 32)),   # smaller than before
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Step 2: Load dataset (expects data/train/dogs and data/train/cats)
train_data = datasets.ImageFolder("data/train", transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)  # very small batch size

# Step 3: Define a tiny CNN
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 14 * 14, 32)  # adjusted for 32x32 input
        self.fc2 = nn.Linear(32, 2)  # 2 classes: dog, cat

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = TinyCNN()

# Step 4: Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)  # simpler optimizer, less memory

# Step 5: Training loop
for epoch in range(3):  # fewer epochs
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Step 6: Save the trained model
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")
