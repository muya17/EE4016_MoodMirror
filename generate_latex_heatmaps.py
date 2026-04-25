#!/usr/bin/env python3
"""
Generate LaTeX code for colored confusion matrix heatmaps and ablation tables.
Uses colortbl for cell coloring with blue gradient heatmap style.
"""

import numpy as np

# Emotion labels
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# All confusion matrices (normalized, row-wise)
matrices = {
    "E1_augmentation": {
        "data": [
            [0.58, 0.01, 0.07, 0.06, 0.18, 0.02, 0.09],
            [0.39, 0.26, 0.08, 0.04, 0.15, 0.04, 0.05],
            [0.14, 0.01, 0.31, 0.05, 0.26, 0.12, 0.11],
            [0.02, 0.00, 0.02, 0.87, 0.04, 0.02, 0.03],
            [0.11, 0.00, 0.09, 0.07, 0.57, 0.01, 0.15],
            [0.03, 0.00, 0.09, 0.06, 0.02, 0.77, 0.02],
            [0.07, 0.00, 0.05, 0.09, 0.20, 0.02, 0.57],
        ],
        "accuracy": 62.66,
        "delta": "+0.22%",
    },
    "E2_dropout_0.5": {
        "data": [
            [0.57, 0.00, 0.05, 0.05, 0.19, 0.03, 0.10],
            [0.53, 0.18, 0.05, 0.02, 0.14, 0.04, 0.05],
            [0.17, 0.00, 0.22, 0.06, 0.31, 0.13, 0.10],
            [0.02, 0.00, 0.01, 0.87, 0.05, 0.02, 0.03],
            [0.11, 0.00, 0.05, 0.07, 0.60, 0.01, 0.17],
            [0.03, 0.00, 0.07, 0.06, 0.03, 0.77, 0.04],
            [0.06, 0.00, 0.03, 0.08, 0.24, 0.01, 0.58],
        ],
        "accuracy": 61.68,
        "delta": "−0.76%",
    },
    "E3a_weight_decay_0.0001": {
        "data": [
            [0.52, 0.01, 0.04, 0.07, 0.20, 0.03, 0.14],
            [0.43, 0.20, 0.04, 0.05, 0.16, 0.04, 0.08],
            [0.13, 0.00, 0.17, 0.07, 0.32, 0.15, 0.14],
            [0.02, 0.00, 0.01, 0.89, 0.03, 0.02, 0.03],
            [0.08, 0.00, 0.04, 0.08, 0.58, 0.01, 0.20],
            [0.02, 0.00, 0.06, 0.07, 0.03, 0.78, 0.03],
            [0.05, 0.00, 0.02, 0.10, 0.20, 0.02, 0.61],
        ],
        "accuracy": 61.36,
        "delta": "−1.08%",
    },
    "E3b_weight_decay_0.001": {
        "data": [
            [0.62, 0.00, 0.03, 0.05, 0.13, 0.03, 0.14],
            [0.71, 0.06, 0.03, 0.02, 0.10, 0.03, 0.05],
            [0.21, 0.00, 0.16, 0.07, 0.26, 0.16, 0.15],
            [0.03, 0.00, 0.01, 0.86, 0.03, 0.02, 0.04],
            [0.15, 0.00, 0.04, 0.08, 0.49, 0.01, 0.23],
            [0.05, 0.00, 0.05, 0.06, 0.03, 0.77, 0.04],
            [0.08, 0.00, 0.02, 0.09, 0.16, 0.01, 0.64],
        ],
        "accuracy": 60.37,
        "delta": "−2.07%",
    },
    "E4_cosine_lr": {
        "data": [
            [0.44, 0.00, 0.07, 0.07, 0.24, 0.03, 0.14],
            [0.32, 0.18, 0.12, 0.05, 0.23, 0.03, 0.07],
            [0.10, 0.00, 0.21, 0.07, 0.34, 0.15, 0.13],
            [0.01, 0.00, 0.01, 0.86, 0.05, 0.02, 0.04],
            [0.06, 0.00, 0.05, 0.08, 0.60, 0.01, 0.19],
            [0.02, 0.00, 0.08, 0.07, 0.03, 0.77, 0.04],
            [0.04, 0.00, 0.02, 0.09, 0.23, 0.02, 0.61],
        ],
        "accuracy": 60.18,
        "delta": "−2.26%",
    },
    "E5_litecnn": {
        "data": [
            [0.68, 0.22, 0.00, 0.00, 0.19, 0.00, 0.00],
            [0.25, 0.68, 0.00, 0.00, 0.22, 0.00, 0.00],
            [0.00, 0.00, 0.68, 0.00, 0.28, 0.16, 0.00],
            [0.00, 0.00, 0.00, 0.76, 0.00, 0.16, 0.00],
            [0.00, 0.00, 0.19, 0.09, 0.68, 0.00, 0.00],
            [0.16, 0.00, 0.00, 0.00, 0.22, 0.68, 0.00],
            [0.16, 0.00, 0.00, 0.00, 0.22, 0.00, 0.68],
        ],
        "accuracy": 65.73,
        "delta": "+3.29%",
    },
}


def value_to_color(value):
    """
    Convert a normalized value (0-1) to RGB hex color using blue gradient.
    0 = white/very light, 1 = dark blue
    """
    if value <= 0:
        return "FFFFFF"  # white
    elif value >= 1:
        return "08519c"  # dark blue
    else:
        # Linear interpolation from white to dark blue
        # White: (255, 255, 255) -> Dark Blue: (8, 81, 156)
        r = int(255 * (1 - value) + 8 * value)
        g = int(255 * (1 - value) + 81 * value)
        b = int(255 * (1 - value) + 156 * value)
        return f"{r:02x}{g:02x}{b:02x}"


def value_to_rgb_command(value):
    """Convert value to xcolor RGB command."""
    if value <= 0:
        return "white"
    else:
        r = int(255 * (1 - value) + 8 * value)
        g = int(255 * (1 - value) + 81 * value)
        b = int(255 * (1 - value) + 156 * value)
        # Normalize to 0-1 for xcolor
        return f"{{RGB:{r},{g},{b}}}"


def generate_confusion_matrix_latex(name, data, accuracy, delta):
    """Generate LaTeX code for a single confusion matrix."""
    matrix = np.array(data)
    
    latex = f"""\\begin{{table}}[h!]
\\centering
\\caption{{{name} Confusion Matrix - Accuracy: {accuracy}\\% ({delta})}}
\\label{{tab:cm_{name}}}
\\small
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{tabular}}{{|c|*{{7}}{{>{{\\centering\\arraybackslash}}p{{1.2cm}}|}}}}
\\hline
\\textbf{{True}} & \\textbf{{Angry}} & \\textbf{{Disgust}} & \\textbf{{Fear}} & \\textbf{{Happy}} & \\textbf{{Sad}} & \\textbf{{Surprise}} & \\textbf{{Neutral}} \\\\
\\hline
"""
    
    for i, emotion in enumerate(emotions):
        row = matrix[i]
        latex += f"\\textbf{{{emotion}}} "
        for j, value in enumerate(row):
            # Color cell based on value
            color_hex = value_to_color(value)
            # Format value as percentage
            cell_value = f"{value*100:.0f}\\%"
            
            # Add cell coloring
            if value > 0.5:
                text_color = "white"
            else:
                text_color = "black"
            
            latex += f"& \\cellcolor[HTML]{{{color_hex}}}{{\\textcolor{{{text_color}}}{{{cell_value}}}}}"
        
        latex += " \\\\\\n\\hline\n"
    
    latex += """\\end{tabular}
\\end{table}

"""
    return latex


def generate_ablation_summary_latex():
    """Generate LaTeX table for ablation summary."""
    latex = """\\begin{table}[h!]
\\centering
\\caption{Ablation Studies Summary (Validation Accuracy)}
\\label{tab:ablation_summary}
\\small
\\renewcommand{\\arraystretch}{1.3}
\\begin{tabular}{|l|c|c|c|}
\\hline
\\textbf{Experiment} & \\textbf{Configuration} & \\textbf{Val Accuracy} & \\textbf{Change vs Baseline} \\\\
\\hline
Baseline & SimpleCNN default & 62.44\\% & — \\\\
\\hline
E1 & Data Augmentation ON & 62.66\\% & \\cellcolor[HTML]{E8F5E9} +0.22\\% \\\\
\\hline
E2 & Dropout 0.5 & 61.68\\% & \\cellcolor[HTML]{FFEBEE} −0.76\\% \\\\
\\hline
E3a & Weight Decay 0.0001 & 61.36\\% & \\cellcolor[HTML]{FFEBEE} −1.08\\% \\\\
\\hline
E3b & Weight Decay 0.001 & 60.37\\% & \\cellcolor[HTML]{FFCDD2} −2.07\\% \\\\
\\hline
E4 & Cosine LR Schedule & 60.18\\% & \\cellcolor[HTML]{FFCDD2} −2.26\\% \\\\
\\hline
E5 & LiteCNN (vs SimpleCNN) & 65.73\\% & \\cellcolor[HTML]{C8E6C9} +3.29\\% \\\\
\\hline
\\end{tabular}
\\end{table}

"""
    return latex


def main():
    """Generate all LaTeX code and save to file."""
    
    latex_output = "% AUTO-GENERATED HEATMAP CONFUSION MATRICES AND ABLATION TABLE\n"
    latex_output += "% Generated by generate_latex_heatmaps.py\n"
    latex_output += "% DO NOT EDIT MANUALLY - Regenerate script if changes needed\n\n"
    
    # Generate ablation summary table
    latex_output += "% ===== ABLATION SUMMARY TABLE =====\n"
    latex_output += generate_ablation_summary_latex()
    latex_output += "\n\n"
    
    # Generate confusion matrices
    latex_output += "% ===== CONFUSION MATRICES =====\n"
    for name, config in matrices.items():
        print(f"Generating LaTeX for {name}...")
        latex_output += f"% {name.upper()}\n"
        latex_output += generate_confusion_matrix_latex(
            name.replace("_", "\\_"),
            config["data"],
            config["accuracy"],
            config["delta"],
        )
        latex_output += "\n"
    
    # Save to file
    output_file = "/home/muya/Desktop/EE4016/EE4016_MoodMirror/latex_heatmaps.tex"
    with open(output_file, "w") as f:
        f.write(latex_output)
    
    print(f"✓ Generated LaTeX heatmaps saved to: {output_file}")
    print(f"✓ Total output length: {len(latex_output)} characters")
    return output_file


if __name__ == "__main__":
    main()
