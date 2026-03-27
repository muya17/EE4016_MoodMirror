# src/train_clcm.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import M2's custom modules
# (Adjust the import names if your data_loader functions are named differently)
from models import CLCM
from data_loader import get_dataloaders 

# ==============================================================================
# HYPERPARAMETERS & CONFIGURATION (M4 can tweak these later)
# ==============================================================================
DATA_PATH = 'data/fer2013.csv'
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4  # L2 Regularization to prevent overfitting
SAVE_DIR = 'saved_models'

def train_model():
    """
    Main training loop for the CLCM architecture.
    Features: MPS acceleration, LR scheduling, and best-model checkpointing.
    """
    # 1. Setup Device (Hardware Acceleration)
    # Automatically uses Apple Silicon (MPS) if available, otherwise fallback to CPU
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
    # NOTE: Assuming get_dataloaders returns 3 loaders. 
    # If yours returns 2 (train, val), just remove the test_loader part.
    train_loader, val_loader, test_loader = get_dataloaders(DATA_PATH, BATCH_SIZE)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # 3. Initialize Model, Loss, and Optimizer
    print("\nInitializing CLCM (2024) Architecture...")
    model = CLCM(num_classes=7).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer is generally best for FER2013
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # LR Scheduler: Reduces learning rate if Validation Loss stops improving
    # This is CRITICAL for pushing accuracy past 60% towards the 63% target
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_acc = 0.0
    best_model_path = os.path.join(SAVE_DIR, 'clcm_best_weights.pth')

    # 4. The Training Loop
    print("\nSTARTING TRAINING LOOP...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            # Move data to the same device as the model (e.g., MPS)
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        # Disable gradient calculation for validation (saves memory & computes faster)
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100
        
        # Tell the scheduler to check the validation loss
        scheduler.step(epoch_val_loss)

        epoch_time = time.time() - start_time

        # --- PRINT EPOCH SUMMARY ---
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
              f"Time: {epoch_time:.1f}s | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")

        # --- SAVE BEST MODEL ---
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"    New best validation accuracy! Model saved to {best_model_path}")

    print("\n" + "="*50)
    print(f"TRAINING COMPLETE! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"The best weights are safely stored in: {best_model_path}")
    print("="*50 + "\n")

    # ==============================================================================
    # TESTING ON UNSEEN DATA
    # ==============================================================================
    print("\nRUNNING FINAL EVALUATION ON TEST SET...")

    # 1. Instantiate a fresh model to ensure a clean slate
    best_model = CLCM(num_classes=7).to(device)

    # 2. Load the best saved weights from the training/validation phase
    best_model.load_state_dict(torch.load(best_model_path))
    
    # 3. Set the model to evaluation mode (disables Dropout, fixes BatchNorm)
    best_model.eval()

    correct_test = 0
    total_test = 0

    # 4. Disable gradient tracking for faster computation and lower memory usage
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = best_model(inputs)
            
            # Get the index of the max log-probability as the predicted label
            _, predicted = torch.max(outputs, 1)
            
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    # 5. Calculate and print the final official accuracy
    final_test_acc = (correct_test / total_test) * 100
    print("\n" + "="*50)
    print(f"TEST ACCURACY: {final_test_acc:.2f}%")
    print("="*50 + "\n")

if __name__ == '__main__':
    train_model()