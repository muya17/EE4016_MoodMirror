import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

# Add parent directory to path so we can import modules from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from project root
from models import SimpleCNN, LiteCNN   # models.py at project root
from data.data_loader import get_dataloaders

from experiment_logger import ExperimentLogger
from utils import set_seed, compute_f1_per_class, compute_latency

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_model(model_name, dropout_rate, num_classes=7):
    if model_name == "SimpleCNN":
        return SimpleCNN(dropout_rate=dropout_rate, num_classes=num_classes)
    elif model_name == "LiteCNN":
        return LiteCNN(dropout_rate=dropout_rate, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx == 0:
            print(f"Batch shape: {images.shape}")
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)
    return avg_loss, acc, f1, cm, all_preds, all_labels

def main(config_path):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    print(f"Batch size from config: {cfg['batch_size']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Running experiment: {cfg['name']} on {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Data loaders
    # Support both 'data_path' (folder) and 'csv_path' (CSV file) keys
    data_path = cfg.get("data_path", cfg.get("csv_path"))
    if data_path is None:
        raise ValueError("Config must contain 'data_path' or 'csv_path'")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_path=data_path,
        batch_size=cfg["batch_size"],
        augment=cfg["augment"],
        num_workers=4,
        pin_memory=True
    )

    # Model
    model = create_model(cfg["model"], cfg["training"]["dropout_rate"])
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    opt_name = cfg["training"]["optimizer"]
    lr = cfg["training"]["lr"]
    wd = cfg["training"]["weight_decay"]

    if opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "SGD":
        momentum = cfg["training"].get("momentum", 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    # Scheduler
    scheduler = None
    if cfg["training"]["lr_scheduler"]["enabled"]:
        sched_type = cfg["training"]["lr_scheduler"]["type"]
        if sched_type == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, **cfg["training"]["lr_scheduler"]["params"])
        else:
            raise ValueError(f"Unknown scheduler: {sched_type}")

    # Logger
    logger = ExperimentLogger(cfg["name"], log_dir=cfg["logging"]["log_dir"])

    best_val_acc = 0.0
    best_model_path = os.path.join(logger.log_dir, "best_model.pth")

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, criterion, device)

        logger.log_epoch(epoch, train_loss, val_loss, val_acc, val_f1)
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        if scheduler is not None:
            scheduler.step()

    # Load best model and test
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc, test_f1, test_cm, _, _ = evaluate(model, test_loader, criterion, device)

    per_class_f1 = compute_f1_per_class(test_cm)

    # Latency measurement (on CPU)
    dummy_input = torch.randn(1, 1, 48, 48)
    latency = compute_latency(model, dummy_input, device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    final_metrics = {
        "experiment_name": cfg["name"],
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "test_loss": test_loss,
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": test_cm.tolist(),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "inference_latency_ms": latency * 1000,
        "best_val_accuracy": best_val_acc,
    }
    logger.save_final_metrics(final_metrics)
    logger.close()

    print(f"\nExperiment {cfg['name']} completed.")
    print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}, Latency: {latency*1000:.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)