import os
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import the dataset class we just built
#basdcjhabsdi
from data_loader import FER2013Dataset
 
def extract_hog_features(dataset):
    """
    Extract HOG (Histogram of Oriented Gradients) features from the dataset.
    :param dataset: Instance of FER2013Dataset
    :return: numpy array of features and numpy array of labels
    """
    print(f"Extracting HOG features for {len(dataset.images)} images...")
    features = []
    labels = dataset.emotions
    
    for img in dataset.images:
        # Standard HOG parameters for 48x48 face images
        fd = hog(img, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualize=False)
        features.append(fd)
        
    return np.array(features), np.array(labels)

def plot_confusion_matrix(cm, classes, save_path="confusion_matrix_svm.png"):
    """
    Plot and save the confusion matrix using Seaborn.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('HOG + SVM Confusion Matrix')
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nConfusion matrix saved as {save_path}")

def run_baseline():
    """
    Main pipeline for the Classical ML Baseline (HOG + SVM).
    """
    csv_path = "data/fer2013"
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return

    # 1. Load Datasets (Using your data_loader logic without deep learning transforms)
    train_dataset = FER2013Dataset(csv_path, split='Training')
    test_dataset = FER2013Dataset(csv_path, split='PrivateTest')
    
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # 2. Extract Features
    print("\n--- Processing Training Set ---")
    X_train, y_train = extract_hog_features(train_dataset)
    
    print("\n--- Processing Test Set ---")
    X_test, y_test = extract_hog_features(test_dataset)

    # 3. Train SVM Classifier (LinearSVC is much faster than standard SVC for large datasets)
    print("\nTraining Linear SVM Classifier. This may take a few minutes...")
    svm_clf = LinearSVC(dual=False, max_iter=2000, random_state=42)
    svm_clf.fit(X_train, y_train)

    # 4. Evaluate and Print Metrics
    print("\nPredicting on Test Set...")
    y_pred = svm_clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "="*40)
    print("🎯 BASELINE RESULTS (HOG + SVM)")
    print("="*40)
    print(f"Test Accuracy : {acc * 100:.2f}%")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print("="*40)

    # 5. Save the Confusion Matrix Plot
    plot_confusion_matrix(cm, emotion_labels)

if __name__ == "__main__":
    run_baseline()