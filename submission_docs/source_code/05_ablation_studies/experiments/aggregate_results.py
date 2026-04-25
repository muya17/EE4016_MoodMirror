import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_all_results(runs_dir="../runs"):
    results = []
    for json_path in glob.glob(os.path.join(runs_dir, "*/final_metrics.json")):
        with open(json_path, "r") as f:
            data = json.load(f)
            results.append(data)
    return results

def create_summary_table(results):
    rows = []
    for r in results:
        rows.append({
            "Experiment": r["experiment_name"],
            "Accuracy": r["test_accuracy"],
            "F1 (macro)": r["test_f1_macro"],
            "Params (M)": r["total_params"] / 1e6,
            "Latency (ms)": r["inference_latency_ms"],
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("Experiment").reset_index(drop=True)
    return df

def plot_training_curves(runs_dir="../runs"):
    csv_files = glob.glob(os.path.join(runs_dir, "*/metrics.csv"))
    for csv_path in csv_files:
        exp_name = os.path.basename(os.path.dirname(csv_path))
        df = pd.read_csv(csv_path)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(df["epoch"], df["train_loss"], label="Train Loss")
        ax1.plot(df["epoch"], df["val_loss"], label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"{exp_name} - Loss")
        ax1.legend()
        ax2.plot(df["epoch"], df["val_acc"], label="Val Accuracy")
        ax2.plot(df["epoch"], df["val_f1"], label="Val F1")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Metric")
        ax2.set_title(f"{exp_name} - Accuracy & F1")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(csv_path), "training_curves.png"))
        plt.close()

def plot_confusion_matrices(runs_dir="../runs", class_names=None):
    if class_names is None:
        class_names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    json_files = glob.glob(os.path.join(runs_dir, "*/final_metrics.json"))
    for json_path in json_files:
        with open(json_path, "r") as f:
            data = json.load(f)
        cm = np.array(data["confusion_matrix"])
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f"Confusion Matrix - {data['experiment_name']}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(json_path), "confusion_matrix.png"))
        plt.close()

def plot_accuracy_comparison(results):
    df = create_summary_table(results)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Experiment", y="Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.title("Test Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png")
    plt.show()

def plot_params_vs_accuracy(results):
    df = create_summary_table(results)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Params (M)", y="Accuracy", hue="Experiment", s=100)
    for _, row in df.iterrows():
        plt.text(row["Params (M)"] + 0.02, row["Accuracy"] - 0.005, row["Experiment"], fontsize=8)
    plt.title("Model Efficiency: Parameters vs. Accuracy")
    plt.xlabel("Number of Parameters (Millions)")
    plt.ylabel("Test Accuracy")
    plt.tight_layout()
    plt.savefig("params_vs_accuracy.png")
    plt.show()

def export_table_to_latex(df, filename="summary_table.tex"):
    df_rounded = df.round(4)
    latex = df_rounded.to_latex(index=False, float_format="%.4f")
    with open(filename, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to {filename}")

if __name__ == "__main__":
    results = load_all_results("../runs")
    df = create_summary_table(results)
    print("\n=== Summary Table ===\n")
    print(df.to_string(index=False))
    df.to_csv("summary_table.csv", index=False)
    plot_training_curves()
    plot_confusion_matrices()
    plot_accuracy_comparison(results)
    plot_params_vs_accuracy(results)
    export_table_to_latex(df)