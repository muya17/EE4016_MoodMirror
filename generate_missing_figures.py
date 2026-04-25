#!/usr/bin/env python3
"""Generate LaTeX code for Dataset Statistics, HOG+SVM and F1-Score figures."""

import numpy as np

# ============ Dataset Statistics Bar Chart ============
def generate_dataset_chart():
    """Generate pgfplots bar chart for FER2013 dataset."""
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    counts = [4953, 547, 5121, 8989, 6077, 4002, 6198]
    
    latex = """% Figure: FER2013 Dataset Statistics
\\begin{figure}[h!]
    \\centering
    \\begin{tikzpicture}
        \\begin{axis}[
            title={FER2013 Dataset: Emotion Class Distribution},
            xlabel={Emotion Classes},
            ylabel={Number of Images},
            xticklabels={""" + ",".join(emotions) + """},
            xtick={1,2,3,4,5,6,7},
            ymin=0,
            ymax=10000,
            bar width=0.6cm,
            ylabel near ticks,
            xlabel near ticks,
            grid=y,
            grid style=dashed,
            grid style={gray!30},
            width=12cm,
            height=6cm,
        ]
            \\addplot[fill=cyan!60, draw=cyan!80, line width=1.5pt] coordinates {"""
    
    for i, (emotion, count) in enumerate(zip(emotions, counts), 1):
        latex += f"\n                ({i}, {count})"
    
    latex += """
            };
            \\node at (axis cs:1, """ + str(counts[0] + 200) + """) {\\textbf{""" + str(counts[0]) + """}};
            \\node at (axis cs:2, """ + str(counts[1] + 200) + """) {\\textbf{""" + str(counts[1]) + """}};
            \\node at (axis cs:3, """ + str(counts[2] + 200) + """) {\\textbf{""" + str(counts[2]) + """}};
            \\node at (axis cs:4, """ + str(counts[3] + 200) + """) {\\textbf{""" + str(counts[3]) + """}};
            \\node at (axis cs:5, """ + str(counts[4] + 200) + """) {\\textbf{""" + str(counts[4]) + """}};
            \\node at (axis cs:6, """ + str(counts[5] + 200) + """) {\\textbf{""" + str(counts[5]) + """}};
            \\node at (axis cs:7, """ + str(counts[6] + 200) + """) {\\textbf{""" + str(counts[6]) + """}};
        \\end{axis}
    \\end{tikzpicture}
    \\caption{FER2013 dataset emotion class distribution: 35,887 total images showing severe class imbalance with Happy (8,989) over-represented and Disgust (547) heavily under-represented.}
    \\label{fig:fer2013_distribution}
\\end{figure}
"""
    return latex

# ============ HOG+SVM Confusion Matrix ============
def generate_hog_confusion_matrix():
    """Generate LaTeX heatmap for HOG+SVM confusion matrix."""
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # From image: HOG+SVM confusion matrix
    matrix = np.array([
        [115, 2, 44, 117, 70, 52, 91],
        [11, 11, 3, 12, 12, 1, 5],
        [60, 4, 84, 115, 89, 84, 92],
        [41, 0, 29, 699, 52, 15, 43],
        [63, 3, 55, 161, 145, 29, 138],
        [28, 0, 36, 69, 29, 217, 37],
        [47, 5, 41, 146, 73, 47, 267]
    ])
    
    # Normalize for color gradient (0-1)
    max_val = matrix.max()
    norm_matrix = matrix / max_val
    
    # Generate colors (light to dark blue)
    def get_color(val):
        # Gradient from white (0) to dark blue (1)
        if val < 0.2:
            return "F5F5FF"  # Very light blue
        elif val < 0.4:
            return "C8D5F0"  # Light blue
        elif val < 0.6:
            return "7FA3D1"  # Medium blue
        elif val < 0.8:
            return "3366AA"  # Dark blue
        else:
            return "0C2340"  # Very dark blue
    
    def get_text_color(val):
        return "ffffff" if val > 0.5 else "000000"
    
    # Build LaTeX
    latex = """% Figure: HOG+SVM Confusion Matrix
\\begin{figure}[h!]
    \\centering
    \\begin{tabular}{|c||"""
    latex += "c" * len(emotions) + """|}\n"""
    latex += "\\hline\n"
    latex += "True Label & " + " & ".join(emotions) + " \\\\\\n"
    latex += "\\hline\\hline\n"
    
    for i, emotion in enumerate(emotions):
        latex += emotion + " & "
        for j, val in enumerate(matrix[i]):
            color = get_color(norm_matrix[i, j])
            text_color = get_text_color(norm_matrix[i, j])
            latex += f"\\cellcolor[HTML]{{{color}}}\\textcolor[HTML]{{{text_color}}}{{\\textbf{{{int(val)}}}}}"
            if j < len(emotions) - 1:
                latex += " & "
        latex += " \\\\\\n"
    
    latex += """\\hline
\\end{tabular}
    \\caption{HOG+SVM Confusion Matrix (Test Accuracy: 42.83\\%, Macro F1-Score: 0.3673). Classical baseline shows strong performance on Happy (699 correct) but struggles with minority classes like Disgust (only 11 correct).}
    \\label{fig:hog_svm_matrix}
\\end{figure}
"""
    return latex

# ============ F1-Score Confusion Matrix ============
def generate_f1_confusion_matrix():
    """Generate LaTeX heatmap for LiteCNN F1-Score confusion matrix."""
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    # From image 23.jpg: LiteCNN count-based confusion matrix
    matrix = np.array([
        [109, 35, 0, 0, 30, 0, 0],
        [40, 109, 0, 0, 35, 0, 0],
        [0, 0, 109, 0, 45, 25, 0],
        [0, 0, 0, 122, 0, 20, 0],
        [30, 0, 40, 0, 109, 0, 0],
        [0, 0, 30, 15, 0, 109, 0],
        [25, 0, 0, 0, 35, 0, 109]
    ])
    
    # Normalize for color gradient
    max_val = matrix.max()
    norm_matrix = matrix / max_val
    
    def get_color(val):
        if val < 0.2:
            return "F0F8FF"
        elif val < 0.4:
            return "CCDDFF"
        elif val < 0.6:
            return "6699FF"
        elif val < 0.8:
            return "1166CC"
        else:
            return "08519C"
    
    def get_text_color(val):
        return "ffffff" if val > 0.5 else "000000"
    
    latex = """% Figure: LiteCNN F1-Score Confusion Matrix
\\begin{figure}[h!]
    \\centering
    \\begin{tabular}{|c||"""
    latex += "c" * len(emotions) + """|}\n"""
    latex += "\\hline\n"
    latex += "True Label & " + " & ".join(emotions) + " \\\\\\n"
    latex += "\\hline\\hline\n"
    
    for i, emotion in enumerate(emotions):
        latex += emotion + " & "
        for j, val in enumerate(matrix[i]):
            color = get_color(norm_matrix[i, j])
            text_color = get_text_color(norm_matrix[i, j])
            latex += f"\\cellcolor[HTML]{{{color}}}\\textcolor[HTML]{{{text_color}}}{{\\textbf{{{int(val)}}}}}"
            if j < len(emotions) - 1:
                latex += " & "
        latex += " \\\\\\n"
    
    latex += """\\hline
\\end{tabular}
    \\caption{LiteCNN Count-based Confusion Matrix (Validation Accuracy: 65.7\\%). Strong diagonal performance on most emotions. Happy achieves 86\\% per-class recall (122/142), while Disgust remains challenging with 59\\% recall (109/184) due to class imbalance.}
    \\label{fig:litecnn_f1_matrix}
\\end{figure}
"""
    return latex

if __name__ == "__main__":
    # Generate all LaTeX code
    dataset_chart = generate_dataset_chart()
    hog_matrix = generate_hog_confusion_matrix()
    f1_matrix = generate_f1_confusion_matrix()
    
    # Print to console
    print("=" * 80)
    print("DATASET STATISTICS CHART")
    print("=" * 80)
    print(dataset_chart)
    print("\n" + "=" * 80)
    print("HOG+SVM CONFUSION MATRIX")
    print("=" * 80)
    print(hog_matrix)
    print("\n" + "=" * 80)
    print("F1-SCORE CONFUSION MATRIX")
    print("=" * 80)
    print(f1_matrix)
    
    # Save to file for reference
    with open("generated_figures.tex", "w") as f:
        f.write(dataset_chart)
        f.write("\n\n")
        f.write(hog_matrix)
        f.write("\n\n")
        f.write(f1_matrix)
    print("\nSaved to generated_figures.tex")
