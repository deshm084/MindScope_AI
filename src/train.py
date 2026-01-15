import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Import our custom modules (package-safe)
from src.dataset import MindScopeDataset
from src.model_fusion import MindScopeFusionModel

# --- Configuration ---
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5
MAX_GRAD_NORM = 1.0
SEED = 42

def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    losses = []
    correct = 0

    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tabular_features = batch["tabular_features"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask, tabular_features)
        loss = loss_fn(outputs, labels)
        losses.append(loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    train_acc = correct / len(data_loader.dataset)
    train_loss = sum(losses) / len(losses)
    return train_acc, train_loss

def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tabular_features = batch["tabular_features"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, tabular_features)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    val_loss = sum(losses) / len(losses)
    return accuracy, f1, val_loss

def main():
    set_seed(SEED)

    # Resolve paths relative to project root (MindScope_AI/)
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "raw" / "mental_health_multimodal.csv"
    models_dir = project_root / "models"
    model_save_path = models_dir / "mindscope_best.pth"
    models_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run: python .\\src\\data_gen.py"
        )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    full_dataset = MindScopeDataset(str(data_path))

    # Train/Val split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = MindScopeFusionModel(num_tabular_features=4, num_classes=3, freeze_bert=True)
    model.to(device)

    # Optimizer / Loss
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = -1.0

    print("\nStarting Training...")
    print("-" * 40)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_acc, val_f1, val_loss = eval_model(model, val_loader, loss_fn, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # Save best model by validation F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), str(model_save_path))
            print(f"--> New Best Model Saved! ({model_save_path}) (F1: {best_f1:.4f})")

        print("-" * 40)

if __name__ == "__main__":
    main()
