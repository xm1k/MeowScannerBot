# train_vit.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# -----------------------
# HYPERPARAMS
# -----------------------
train_ratio = 0.8
batch_size = 8              # ⬅️ важно для CPU
num_epochs = 10
learning_rate = 1e-3
data_root = "images"
num_workers = 2
save_prefix = "vit"

# -----------------------
# DEVICE
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------
# TRANSFORMS (ViT = 224)
# -----------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------
# DATASET
# -----------------------
dataset = datasets.ImageFolder(data_root, transform=train_transform)
num_classes = len(dataset.classes)
print("Classes:", num_classes)

train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
test_dataset.dataset.transform = eval_transform

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=num_workers)

# -----------------------
# MODEL: ViT (TRANSFER LEARNING)
# -----------------------
model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

# 🔒 freeze backbone
for param in model.parameters():
    param.requires_grad = False

# 🎯 new classification head
in_features = model.heads.head.in_features
model.heads.head = nn.Linear(in_features, num_classes)

model.to(device)

# -----------------------
# LOSS / OPT / SCHED
# -----------------------
criterion = nn.CrossEntropyLoss()

# ❗ оптимизатор ТОЛЬКО для головы
optimizer = optim.Adam(model.heads.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# -----------------------
# TRAIN LOOP
# -----------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nEpoch [{epoch+1}/{num_epochs}] "
          f"Loss: {running_loss/len(train_loader):.4f} "
          f"Train-Acc: {acc:.4f}")

    scheduler.step()

# -----------------------
# EVALUATION
# -----------------------
model.eval()
class_correct = Counter()
class_total = Counter()
errors_by_class = Counter()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        for l, p in zip(labels, preds):
            class_total[l.item()] += 1
            if l == p:
                class_correct[l.item()] += 1
            else:
                errors_by_class[l.item()] += 1

overall_acc = 100 * sum(class_correct.values()) / sum(class_total.values())
print(f"\nTest Accuracy: {overall_acc:.2f}%")

print("\nClasses sorted by most errors:")
for idx, errs in sorted(errors_by_class.items(), key=lambda x: x[1], reverse=True):
    name = dataset.classes[idx]
    rate = errs / class_total[idx]
    print(f"{name}: {errs} errors ({rate:.2f})")

# -----------------------
# SAVE
# -----------------------
torch.save(model, f"{save_prefix}_full.pth")
print(f"\nSaved model: {save_prefix}_full.pth")
