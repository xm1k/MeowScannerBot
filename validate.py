import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# НАСТРОЙКИ
# -----------------------------
MODEL_PATH = "vit_full.pth"
IMAGE_ROOT = "images"
BATCH_SIZE = 32
TEST_RATIO = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# ТРАНСФОРМАЦИИ (БЕЗ AUGMENTATION!)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# DATASET
# -----------------------------
dataset = datasets.ImageFolder(IMAGE_ROOT, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)

train_size = int((1 - TEST_RATIO) * len(dataset))
test_size = len(dataset) - train_size
_, test_dataset = random_split(dataset, [train_size, test_size])

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# MODEL (как у тебя!)
# -----------------------------
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.eval()
model.to(device)

# -----------------------------
# EVALUATION
# -----------------------------
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# -----------------------------
# ОСНОВНЫЕ МЕТРИКИ
# -----------------------------
print("\n=== OVERALL METRICS ===")
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision (macro): {precision_score(y_true, y_pred, average='macro'):.4f}")
print(f"Recall    (macro): {recall_score(y_true, y_pred, average='macro'):.4f}")
print(f"F1-score  (macro): {f1_score(y_true, y_pred, average='macro'):.4f}")

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
print("\n=== CLASSIFICATION REPORT ===\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
))

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
