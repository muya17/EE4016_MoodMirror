# Exploratory Data Analysis
# src/eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import os

# Define globally shared configurations
DATA_FILE = 'data/fer2013'
OUTPUT_FOLDER = 'plots'
EMOTIONS_MAP = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
    4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

def plot_class_distribution(df):
    """
    EDA Function 1: 
    Analyzes and visualizes the distribution of emotion labels in the dataset.
    Crucial for identifying Class Imbalance before training begins.
    """
    print("\nAnalyzing class distribution...")
    
    # Count the occurrences of each emotion label (0 to 6)
    label_counts = df['emotion'].value_counts().sort_index()
    
    # Map the numeric indices to actual emotion names
    emotion_names = [EMOTIONS_MAP[i] for i in label_counts.index]
    counts = label_counts.values
    
    # Create the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(emotion_names, counts, color='skyblue', edgecolor='black')
    
    # Add exact numbers on top of each bar for clarity
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 100, int(yval), ha='center', va='bottom')
        
    plt.title('FER2013 Dataset: Emotion Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Emotion Classes', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    save_path = os.path.join(OUTPUT_FOLDER, 'class_distribution.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Success! Class distribution chart saved to: {save_path}")


def visualize_hog_on_random_data(df):
    """
    EDA Function 2:
    Randomly selects a face from the dataset, reconstructs the 48x48 pixel matrix,
    and visualizes the HOG (Histogram of Oriented Gradients) features to verify 
    feature extraction quality on low-resolution data.
    """
    print("\nGenerating HOG visualization...")
    
    # 1. Randomly sample one image
    sample = df.sample(n=1)
    label_id = sample['emotion'].values[0]
    label_name = EMOTIONS_MAP[label_id]
    
    # Convert the pixel string back to a 2D matrix
    image_matrix = np.fromstring(
        sample['pixels'].values[0], 
        dtype=np.uint8, 
        sep=' '
    ).reshape(48, 48)

    # 2. Execute HOG Algorithm
    fd, hog_image = hog(
        image_matrix, 
        orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        visualize=True
    )

    # Enhance intensity for better visibility
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 5))

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image_matrix, cmap=plt.cm.gray) 
    ax1.set_title(f'Original Matrix (Raw Pixels)\nEmotion: {label_name}')

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('HOG Feature Visualization\n(Gradient Orientations)')
    
    save_path = os.path.join(OUTPUT_FOLDER, 'hog_visualization.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Success! HOG visualization saved to: {save_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please ensure the dataset is downloaded.")
    else:
        print("Loading FER2013 dataset into memory (this may take a few seconds)...")
        # Load the CSV exactly once, then pass it to both EDA functions
        dataset_df = pd.read_csv(DATA_FILE)
        
        # Run both Data Exploration tasks
        plot_class_distribution(dataset_df)
        visualize_hog_on_random_data(dataset_df)
        
        print("\nAll Exploratory Data Analysis (EDA) tasks completed!")