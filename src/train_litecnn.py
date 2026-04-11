# src/train_litecnn.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Import M3's custom models
from models import LiteCNN, count_parameters
# Import M2's data pipeline
from data_loader import get_dataloaders


# ==============================================================================
# HYPERPARAMETERS & CONFIGURATION (M4 can tweak these for ablation studies)
# ==============================================================================
DATA_PATH = 'data/fer2013.csv'
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4  # L2 Regularization to prevent overfitting
SAVE_DIR = 'saved_models'


def format_time(seconds):
    """Helper function to format seconds into readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hrs}h {mins}m"


def train_model():
    """
    Main training loop for LiteCNN architecture.
    Features: MPS/CUDA acceleration, LR scheduling, and best-model checkpointing.
    
    :return: Best validation accuracy achieved during training
    """
    
    # 1. Setup Device (Hardware Acceleration)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("USING APPLE SILICON (MPS) FOR HARDWARE ACCELERATION")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("USING NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("USING CPU (Training will be slower).")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 2. Load Data Pipeline (M2's Masterpiece)
    print("\nLoading Datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(DATA_PATH, BATCH_SIZE)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    
    # 3. Initialize Model, Loss, and Optimizer
    print("\nInitializing LiteCNN Architecture...")
    model = LiteCNN(num_classes=7).to(device)
    
    # Display parameter count for efficiency tracking (Section 4.6, O3)
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Cross-entropy loss for multi-class classification (Section 4.5)
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer is generally best for FER2013 (Section 4.5, Ablation E4)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # LR Scheduler: Cosine Annealing for smooth convergence (Section 4.5, Ablation E4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.0001)
    
    best_val_acc = 0.0
    best_model_path = os.path.join(SAVE_DIR, 'LiteCNN_best.pth')
    
    # 4. The Training Loop
    print("\nSTARTING TRAINING LOOP...")
    print("="*70)
    
    training_start = time.time()
    
    # Create tqdm progress bar for epochs
    epoch_pbar = tqdm(range(EPOCHS), desc="Overall Progress", unit="epoch")
    
    for epoch in epoch_pbar:
        start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100
        
        # --- VALIDATION PHASE ---
        model.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100
        
        scheduler.step()
        
        # --- Calculate Timing & Remaining Time ---
        epoch_time = time.time() - start_time
        elapsed_time = time.time() - training_start
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_epochs = EPOCHS - (epoch + 1)
        remaining_time = avg_epoch_time * remaining_epochs
        
        # --- Update Progress Bar with Metrics ---
        epoch_pbar.set_postfix({
            'Train Acc': f'{epoch_train_acc:.1f}%',
            'Val Acc': f'{epoch_val_acc:.1f}%',
            'Time': format_time(epoch_time),
            'Remaining': format_time(remaining_time),
            'Best': f'{best_val_acc:.1f}%'
        })
        
        # --- PRINT EPOCH SUMMARY ---
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
              f"Time: {epoch_time:.1f}s | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")
        
        # --- SAVE BEST MODEL ---
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"    ★ New best validation accuracy! Model saved to {best_model_path}")
    
    epoch_pbar.close()
    
    # 5. Training Complete Summary
    print("="*70)
    print(f"TRAINING COMPLETE! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"The best weights are safely stored in: {best_model_path}")
    print("="*70 + "\n")
    
    return best_val_acc


# ==============================================================================
# TESTING ON UNSEEN DATA (Final Evaluation)
# ==============================================================================
def evaluate_model(model_path):
    """
    Evaluate the trained model on the test set (unseen data).
    This provides the official accuracy metric for the final report.
    
    :param model_path: Path to the saved best model weights
    :return: Test accuracy percentage
    """
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = LiteCNN(num_classes=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    train_loader, val_loader, test_loader = get_dataloaders(DATA_PATH, BATCH_SIZE)
    
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    final_test_acc = (correct_test / total_test) * 100
    
    print("\n" + "="*70)
    print(f"🎯 TEST ACCURACY FOR LITECNN: {final_test_acc:.2f}%")
    print("="*70 + "\n")
    
    return final_test_acc


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("TRAINING LITECNN (M3 Deliverable)")
    print("Section 4.4: Model Architectures for FER2013")
    print("Target: ≥70% accuracy, <1.5M parameters, <10ms CPU (Section 4.4, O3)")
    print("="*70)
    
    # Train LiteCNN
    lite_acc = train_model()
    
    # Evaluate LiteCNN on test set
    lite_test_acc = evaluate_model(os.path.join(SAVE_DIR, 'LiteCNN_best.pth'))
    
    # Final Summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"LiteCNN Validation Accuracy: {lite_acc:.2f}% (Target: ≥70%)")
    print(f"LiteCNN Test Accuracy:       {lite_test_acc:.2f}%")
    
    if lite_acc >= 70.0:
        print("✅ LiteCNN MEETS TARGET (≥70%)")
    else:
        print("⚠️  LiteCNN BELOW TARGET - Consider hyperparameter tuning (M4 Ablation)")
    
    print("="*70 + "\n")