import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from catboost import train
from fsspec.asyn import running_async
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.xpu import device
from torchvision import datasets,transforms,models
from torch.utils.data import random_split
from torch import softmax
from torch.optim.lr_scheduler import StepLR
from collections import Counter

train_ratio = 0.8

batch_size = 32
num_epochs = 15
learning_rate = 0.001

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


dataset = datasets.ImageFolder(root='images', transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

num_classes = len(dataset.classes)
train_size = int(train_ratio*len(dataset))
test_size = len(dataset)-(train_size)

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

class_correct = Counter()
class_total = Counter()
errors_by_class = Counter()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    i = 0
    for images, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        i+=1
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
    print()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/ len(train_loader):.4f}, Accuracy: {accuracy_score(all_labels, all_predictions)}")
    scheduler.step()

model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for label, pred in zip(labels, predicted):
            if label == pred:
                class_correct[label.item()] += 1
            else:
                errors_by_class[label.item()] += 1
            class_total[label.item()] += 1

sorted_errors = sorted(errors_by_class.items(), key=lambda x: x[1], reverse=True)
print("Accuracy is {:.2f}%".format(100 * sum(class_correct.values()) / sum(class_total.values())))

print("Classes sorted by most errors:")
for class_idx, num_errors in sorted_errors:
    class_name = dataset.classes[class_idx]
    error_rate = num_errors / class_total[class_idx] if class_total[class_idx] > 0 else 0
    print(f"Class '{class_name}' - Errors: {num_errors}, Error rate: {error_rate:.2f}")


torch.save(model, 'trained_model.pth')
print("Model saved as 'trained_model.pth'")