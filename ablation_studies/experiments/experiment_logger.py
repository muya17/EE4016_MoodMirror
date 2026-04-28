import os
import json
import csv
from torch.utils.tensorboard import SummaryWriter

class ExperimentLogger:
    def __init__(self, exp_name, log_dir="../runs"):
        self.exp_name = exp_name
        self.log_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        self.csv_path = os.path.join(self.log_dir, "metrics.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_f1"])

    def log_epoch(self, epoch, train_loss, val_loss, val_acc, val_f1):
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/val", val_loss, epoch)
        self.writer.add_scalar("Accuracy/val", val_acc, epoch)
        self.writer.add_scalar("F1/val", val_f1, epoch)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_acc, val_f1])

    def save_final_metrics(self, metrics):
        for key, value in metrics.items():
            if hasattr(value, "tolist"):
                metrics[key] = value.tolist()
        with open(os.path.join(self.log_dir, "final_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    def close(self):
        self.writer.close()