# 📂 SimpleCNN & LiteCNN Module

This document outlines the Deep Learning model architectures (SimpleCNN & LiteCNN)

## 1. Directory Structure Explained
My directory structure:

```text
MoodMirror/
├── data/                       # [DO NOT PUSH] Place fer2013.csv here.
├── src/                        # All core source code goes here.
│   ├── data_loader.py          # [Done] Data parsing and PyTorch DataLoader.
│   ├── baselines.py            # [Done] Classical ML baseline (HOG+SVM).
│   ├── models.py               # [Done] CNN architectures (SimpleCNN, LiteCNN, CLCM).
|   ├── train_simplecnn.py      # Quick testing, SimpleCNN ablation (only include simpleCNN).
|   ├── train_litecnn.py        # Quick testing, LiteCNN ablation (only include LiteCNN).
│   ├── train_models.py         # [Done] Training script for all CNN models.
│   ├── eda.py                  # [Done] Exploratory Data Analysis.
│   └── train_clcm.py           # [Done] CLCM replication training.
├── saved_models/               # Trained model weights (.pth files).
│   ├── SimpleCNN_best.pth      # Best SimpleCNN weights (now is 62.44%).
│   └── LiteCNN_best.pth        # Best LiteCNN weights (now is 65.73%).
├── requirements.txt            # [Done] Shared Python environment dependencies.
└── README_jasmine.md           # This document.
```

## 2. Environment Setup
Before running the code, please ensure you have the correct environment: From README_annine_m2.md

Additional dependency for training progress bar: For training process
```Bash
pip install tqdm
```
Example output for the bar:
    Overall Progress:   2%| | 1/50 [06:31<2:40:12, 196.18s/epoch, Train Acc=32.2%, Val Acc=40.0%, TimEpoch [02/50] Time: 195.0s | Train Loss: 1.7077 - Train Acc: 32.15% | Val Loss: 1.5572 - Val Acc: 39.98%

## 3. How to Train the Models
You can run it directly from the root directory:

For training both models (SimpleCNN, LiteCNN, CLCM):
```Bash
python src/train_models.py
```
Current outputs(here show final part only):
    ======================================================================
    FINAL RESULTS SUMMARY
    ======================================================================
    SimpleCNN Validation Accuracy: 62.44% (Target: ≥68%)
    SimpleCNN Test Accuracy:       63.03%

    LiteCNN Validation Accuracy:   65.73% (Target: ≥70%)
    LiteCNN Test Accuracy:         66.09%

    ⚠️  SimpleCNN BELOW TARGET - Consider hyperparameter tuning (M4 Ablation)
    ⚠️  LiteCNN BELOW TARGET - Consider more epochs or augmentation
    ======================================================================

Expected output(here show final part only):
    ======================================================================
    FINAL RESULTS SUMMARY
    ======================================================================
    SimpleCNN Validation Accuracy: 68.50% (Target: ≥68%)
    SimpleCNN Test Accuracy:       67.80%

    LiteCNN Validation Accuracy:   70.25% (Target: ≥70%)
    LiteCNN Test Accuracy:         69.90%

    ✅ SimpleCNN MEETS TARGET (≥68%)
    ✅ LiteCNN MEETS TARGET (≥70%)
    ======================================================================

For training SimpleCNN only:
```Bash
python src/train_simplecnn.py
```
Expected output(here show final part only):
   ======================================================================
    FINAL RESULTS SUMMARY
    ======================================================================
    SimpleCNN Validation Accuracy: 68.50% (Target: ≥68%)
    SimpleCNN Test Accuracy:       67.80%

    ✅ SimpleCNN MEETS TARGET (≥68%)
    ====================================================================== 

For training LiteCNN only:
```Bash
python src/train_litecnn.py
```
Expected output(here show final part only):
    ======================================================================
    FINAL RESULTS SUMMARY
    ======================================================================
    LiteCNN Validation Accuracy: 70.85% (Target: ≥70%)
    LiteCNN Test Accuracy:       69.90%

    ✅ LiteCNN MEETS TARGET (≥70%)
    ======================================================================


## Kind reminder:
*
The files <train_models.py>, [train_simplecnn.py] and {train_litecnn.py} are three different files. 
So, if you change some code in [train_simplecnn.py] and run it, you will not get the same result if you not change the code in <train_models.py>. 
The result of SimpleCNN part in <train_models.py> will not same as [train_simplecnn.py].

*
Or if you only want to train SimpleCNN in <train_models.py> only, you can scroll to the bottom of <train_models.py> and look for this:
    # ==============================================================================
    # MAIN EXECUTION
    # ==============================================================================
    if __name__ == '__main__':
        print("\n" + "="*70)
        print("TRAINING M3 MODELS (SimpleCNN & LiteCNN)")
        print("Section 4.4: Model Architectures for FER2013")
        print("="*70)
        
    # Train SimpleCNN (Target: ≥68% validation accuracy, Section 4.4)
    print("\n" + "="*70)
    print("PHASE 1: TRAINING SIMPLECNN")
    print("Target: ≥68% validation accuracy (Section 4.4, O2)")
    print("="*70)
    simple_acc = train_model(SimpleCNN, "SimpleCNN")  # ✅ KEEP THIS LINE
    
    # Evaluate SimpleCNN on test set
    simple_test_acc = evaluate_model(
        SimpleCNN, 
        os.path.join(SAVE_DIR, 'SimpleCNN_best.pth'), 
        "SimpleCNN"
    )
    
    # ==========================================================================
    # OPTION: Train LiteCNN Only (Uncomment to enable)
    # ==========================================================================
    # To train LiteCNN, remove the '#' symbol from the beginning of each line below
    # This will enable LiteCNN training after SimpleCNN completes
    
    # print("\n" + "="*70)                                    # ← Remove '#' to enable
    # print("PHASE 2: TRAINING LITECNN")                      # ← Remove '#' to enable
    # print("Target: ≥70% accuracy, <1.5M parameters, <10ms CPU")  # ← Remove '#' to enable
    # print("="*70)                                           # ← Remove '#' to enable
    # lite_acc = train_model(LiteCNN, "LiteCNN")              # ← Remove '#' to enable
    
    # lite_test_acc = evaluate_model(                         # ← Remove '#' to enable
    #     LiteCNN,                                            # ← Remove '#' to enable
    #     os.path.join(SAVE_DIR, 'LiteCNN_best.pth'),         # ← Remove '#' to enable
    #     "LiteCNN"                                           # ← Remove '#' to enable
    # )                                                       # ← Remove '#' to enable
    
    # Final Summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"SimpleCNN Validation Accuracy: {simple_acc:.2f}% (Target: ≥68%)")
    print(f"SimpleCNN Test Accuracy:       {simple_test_acc:.2f}%")
    # print()                                                 # ← Remove '#' if LiteCNN enabled
    # print(f"LiteCNN Validation Accuracy:   {lite_acc:.2f}%")  # ← Remove '#' if LiteCNN enabled
    # print(f"LiteCNN Test Accuracy:         {lite_test_acc:.2f}%")  # ← Remove '#' if LiteCNN enabled
    print()
    
    if simple_acc >= 68.0:
        print("✅ SimpleCNN MEETS TARGET (≥68%)")
    else:
        print("⚠️  SimpleCNN BELOW TARGET - Consider hyperparameter tuning")
    
    # if lite_acc >= 70.0:                                    # ← Remove '#' if LiteCNN enabled
    #     print("✅ LiteCNN MEETS TARGET (≥70%)")             # ← Remove '#' if LiteCNN enabled
    # else:                                                   # ← Remove '#' if LiteCNN enabled
    #     print("⚠️  LiteCNN BELOW TARGET")                   # ← Remove '#' if LiteCNN enabled
    
    print("="*70 + "\n")                                                    #

It is same for train LiteCNN only in <train_models.py>. Just use '#' symbol for the code of SimpleCNN part.