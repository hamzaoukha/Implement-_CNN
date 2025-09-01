import os, time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

os.makedirs("outputs", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
test_ds  = datasets.MNIST("./data", train=False, transform=tfm)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.pool  = nn.MaxPool2d(2,2)
        self.fc1   = nn.Linear(64*3*3, 100)
        self.fc2   = nn.Linear(100, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = CNN().to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

def run_epoch(loader, train=True):
    if train: model.train()
    else: model.eval()
    total=correct=0; loss_sum=0.0
    for xb,yb in loader:
        xb,yb = xb.to(device), yb.to(device)
        if train: opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        if train:
            loss.backward(); opt.step()
        loss_sum += loss.item()*xb.size(0)
        pred = out.argmax(1)
        total += yb.numel()
        correct += (pred==yb).sum().item()
    return loss_sum/total, correct/total

EPOCHS=10
t0 = time.perf_counter()
train_acc_hist, test_acc_hist = [], []
for e in range(EPOCHS):
    tr_loss, tr_acc = run_epoch(train_loader, True)
    te_loss, te_acc = run_epoch(test_loader, False)
    train_acc_hist.append(tr_acc); test_acc_hist.append(te_acc)
    print(f"Epoch {e+1:02d} | train acc {tr_acc:.4f} | test acc {te_acc:.4f}")
elapsed_min = (time.perf_counter() - t0)/60.0

test_acc = test_acc_hist[-1]
print(f"[PyTorch] Test accuracy: {test_acc:.4f} | time: {elapsed_min:.2f} min")

# Confusion matrix
all_preds, all_true = [], []
model.eval()
with torch.no_grad():
    for xb, yb in test_loader:
        out = model(xb.to(device))
        all_preds.extend(out.argmax(1).cpu().numpy().tolist())
        all_true.extend(yb.numpy().tolist())

cm = confusion_matrix(all_true, all_preds)
print(classification_report(all_true, all_preds, digits=4))
disp = ConfusionMatrixDisplay(cm)
fig, ax = plt.subplots(figsize=(6,5))
disp.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title("PyTorch — Confusion Matrix (Test)")
plt.tight_layout()
plt.savefig("outputs/fig_torch_confusion_matrix.png", dpi=200)

# Append to comparison CSV and summary text (if exists)
import pandas as pd
row = pd.DataFrame([["PyTorch", round(100*test_acc, 2), round(elapsed_min, 2), int(np.argmax(test_acc_hist))+1]],
    columns=["Framework","Test Acc (%)","Train Time (min)","Convergence Epoch"])
csv_path = "outputs/mnist_cnn_comparison.csv"
if os.path.exists(csv_path):
    base = pd.read_csv(csv_path)
    base = pd.concat([base, row], ignore_index=True)
    base.to_csv(csv_path, index=False)
else:
    row.to_csv(csv_path, index=False)

with open("outputs/mnist_cnn_summary.txt", "a") as f:
    f.write(f"- PyTorch: test acc {100*test_acc:.2f}%, time {elapsed_min:.2f} min, convergence epoch {int(np.argmax(test_acc_hist))+1}\n")
