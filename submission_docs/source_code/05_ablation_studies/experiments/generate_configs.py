import os
from copy import deepcopy
import yaml

def generate_configs(output_dir="../configs"):
    os.makedirs(output_dir, exist_ok=True)

    # Baseline settings (SimpleCNN, no augmentation, no dropout, no weight decay, Adam constant)
    baseline = {
        "name": "baseline_SimpleCNN",
        "model": "SimpleCNN",
        "data_path": "../../data/fer2013",
        "batch_size": 256,
        "augment": False,
        "training": {
            "epochs": 50,
            "optimizer": "Adam",
            "lr": 0.001,
            "weight_decay": 0.0,
            "dropout_rate": 0.0,
            "lr_scheduler": {"enabled": False, "type": None, "params": {}}
        },
        "logging": {"log_dir": "../runs", "save_model": True, "save_freq": 5},
        "seed": 42
    }

    def write_config(cfg):
        path = os.path.join(output_dir, f"{cfg['name']}.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Generated: {path}")

    # 1. Baseline
    write_config(baseline)

    # 2. E1 – Data augmentation (on)
    cfg = deepcopy(baseline)
    cfg["name"] = "E1_aug_on"
    cfg["augment"] = True
    write_config(cfg)

    # 3. E2 – Dropout rates (with augmentation on)
    for drop in [0.3, 0.5]:
        cfg = deepcopy(baseline)
        cfg["name"] = f"E2_dropout_{drop}"
        cfg["augment"] = True
        cfg["training"]["dropout_rate"] = drop
        write_config(cfg)

    # 4. E3 – Weight decay strengths (aug on, dropout=0.5)
    for wd in [1e-4, 1e-3]:
        cfg = deepcopy(baseline)
        cfg["name"] = f"E3_weightdecay_{wd}"
        cfg["augment"] = True
        cfg["training"]["dropout_rate"] = 0.5
        cfg["training"]["weight_decay"] = wd
        write_config(cfg)

    # 5. E4 – Optimizer & LR schedule (aug on, dropout=0.5, wd=1e-4)
    optimizers = ["Adam", "SGD"]
    schedulers = [
        (False, "constant"),
        (True, "cosine")
    ]
    for opt in optimizers:
        for sched_enabled, sched_name in schedulers:
            cfg = deepcopy(baseline)
            cfg["name"] = f"E4_{opt}_{sched_name}"
            cfg["augment"] = True
            cfg["training"]["dropout_rate"] = 0.5
            cfg["training"]["weight_decay"] = 1e-4
            cfg["training"]["optimizer"] = opt
            if opt == "SGD":
                cfg["training"]["momentum"] = 0.9
            cfg["training"]["lr_scheduler"]["enabled"] = sched_enabled
            if sched_enabled:
                cfg["training"]["lr_scheduler"]["type"] = "CosineAnnealingLR"
                cfg["training"]["lr_scheduler"]["params"] = {"T_max": 50}
            write_config(cfg)

    # 6. E5 – Architecture: LiteCNN (baseline and best settings)
    # LiteCNN baseline (same as SimpleCNN baseline but with LiteCNN model)
    lite_baseline = deepcopy(baseline)
    lite_baseline["name"] = "E5_LiteCNN_baseline"
    lite_baseline["model"] = "LiteCNN"
    write_config(lite_baseline)

    # LiteCNN best (aug on, dropout=0.3, wd=1e-4, Adam cosine)
    lite_best = deepcopy(baseline)
    lite_best["name"] = "E5_LiteCNN_best"
    lite_best["model"] = "LiteCNN"
    lite_best["augment"] = True
    lite_best["training"]["dropout_rate"] = 0.3
    lite_best["training"]["weight_decay"] = 1e-4
    lite_best["training"]["optimizer"] = "Adam"
    lite_best["training"]["lr_scheduler"]["enabled"] = True
    lite_best["training"]["lr_scheduler"]["type"] = "CosineAnnealingLR"
    lite_best["training"]["lr_scheduler"]["params"] = {"T_max": 50}
    write_config(lite_best)

if __name__ == "__main__":
    generate_configs()