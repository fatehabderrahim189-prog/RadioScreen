"""
RadioScreen — Reference Design for Production Training
Deep Learning Pipeline (Transfer Learning on DenseNet-121)

Author: Fateh

NOTE: This script is a REFERENCE IMPLEMENTATION describing how RadioScreen
would be trained on a real, licensed public dataset (e.g., NIH ChestX-ray14,
CheXpert, or the RSNA Pneumonia Detection Challenge dataset on Kaggle).

It requires:
    pip install torch torchvision scikit-learn pillow

It is NOT executed in the development sandbox used to build this project,
because (a) those datasets require signed data-use agreements and cannot
be auto-downloaded, and (b) GPU/deep-learning frameworks were not available
in that environment. Run this on a machine with the dataset downloaded and
the dependencies installed.

DISCLAIMER: Even once trained, a model built from this script is a research
prototype. It must be clinically validated and regulatory-cleared before
any real diagnostic use. It is not a substitute for a licensed physician.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os


class ChestXrayDataset(Dataset):
    """
    Expects a CSV with columns: image_path, label (0=normal, 1=abnormal).
    Compatible with the label format of the NIH ChestX-ray14 / RSNA
    Pneumonia datasets after standard preprocessing.
    """

    def __init__(self, csv_path: str, image_root: str, train: bool = True):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["image_path"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return image, label


def build_model() -> nn.Module:
    """
    Transfer learning on DenseNet-121, pretrained on ImageNet — the same
    backbone used in the widely cited CheXNet study on chest X-ray
    classification. The final classifier is replaced for binary
    normal/abnormal screening (extend to multi-label for specific
    findings such as pneumonia, effusion, cardiomegaly, etc.).
    """
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1),
        nn.Sigmoid(),
    )
    return model


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def main(
    train_csv: str = "data/train_labels.csv",
    val_csv: str = "data/val_labels.csv",
    image_root: str = "data/images",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ChestXrayDataset(train_csv, image_root, train=True)
    val_ds = ChestXrayDataset(val_csv, image_root, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}/{epochs} — train loss: {train_loss:.4f}")

    torch.save(model.state_dict(), "radioscreen_densenet121.pt")
    print("Model saved to radioscreen_densenet121.pt")
    print("\nReminder: clinical validation and regulatory review are required "
          "before any real-world diagnostic use of this model.")


if __name__ == "__main__":
    main()
