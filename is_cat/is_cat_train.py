import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

data_dir = '../images'
cat_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
not_cat_dir = f"./not_cat"

batch_size = 32
learning_rate = 0.001
num_epochs = 10


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CatDataset(Dataset):
    def __init__(self, cat_dirs, not_cat_dir, transform=None):
        self.cat_images = [(os.path.join(cat_dir, img), 1) for cat_dir in cat_dirs for img in os.listdir(cat_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        self.not_cat_images = [(os.path.join(not_cat_dir, img), 0) for img in os.listdir(not_cat_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        self.images = self.cat_images + self.not_cat_images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


dataset = CatDataset(cat_dirs, not_cat_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with tqdm(dataloader, unit="batch") as tepoch:
        for images, labels in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            running_loss += loss.item() * images.size(0)

            tepoch.set_postfix(loss=loss.item(), accuracy=100 * correct_predictions / total_predictions)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = 100 * correct_predictions / total_predictions
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

torch.save(model, 'is_cat_model.pth')