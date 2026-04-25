# MoodMirror Presentation - Complete Content Summary

**Extracted from 26 presentation slides**  
**Date**: April 25, 2026

---

## **1. PROJECT OVERVIEW**

### Team Members
- **KAPYA Zachariah Muya** (Leader)
- WANG Xinan
- CHENG Wing Yan
- YIP Chung Hon
- SIT Yan Tung

### Project Tagline
> **"Communication is more than words"**

Visual demonstration: Same text "University is great" with three different facial expressions (happy, sad, disgusted) showing how emotion changes meaning.

---

## **2. MOTIVATION & APPLICATIONS**

### Real-World Use Cases
1. **Driver Drowsiness Detection**: Alert systems for automotive safety
2. **Employee Wellbeing Monitoring**: Workplace mental health tracking
3. **Mental Health Support**: Clinical and therapeutic applications

### The Problem: Deployment Trade-off
- **Edge Devices**: Efficient but less accurate
- **Data Centers**: High accuracy but impractical
  - ResNet-50: 25.6M parameters
  - EfficientNet-B7: 66.7M parameters
  - DCNN Ensemble: 600MB+ size

**Goal**: Find the "usable middle" - lightweight yet accurate enough for practical deployment

---

## **3. DATASET: FER2013**

### Statistics
- **Total Images**: 35,887
- **Split**: 28,709 train / 3,589 validation / 3,589 test
- **Image Size**: 48×48 grayscale
- **Classes**: 7 emotions

### Class Distribution
| Emotion | Count |
|---------|-------|
| Happy | 8,989 |
| Neutral | 6,198 |
| Sad | 6,077 |
| Angry | 4,953 |
| Fear | 5,121 |
| Surprise | 4,002 |
| Disgust | 547 |

### Human Baseline
- **Human accuracy on FER2013**: ~65-68%

---

## **4. BASELINE: HOG+SVM**

### Results
- **Test Accuracy**: 42.83%
- **Macro F1 Score**: 0.3673

### Hyperparameter Tuning
- **Configurations Tested**: 12 different setups
- **Best Tuned Result**: 42.77% (slightly worse than default 43.23%)
- **Rationale**: Intentionally avoided (6,6) cells configuration as it contradicts lightweight focus

### Key Insight
Traditional feature engineering approach (HOG+SVM) significantly underperforms deep learning methods but provides computational efficiency baseline.

---

## **5. COMPETITOR MODEL: CLCM**

### Architecture
- **Parameters**: 117,383
- **Test Accuracy**: 63.11%
- **Design**: MobileNetV2-inspired with depthwise separable convolutions
- **Purpose**: Lightweight baseline to beat

### Important Note
CLCM serves as the primary comparison benchmark for our proposed LiteCNN model.

---

## **6. ARCHITECTURE DIAGRAMS**

### 6.1 SimpleCNN Architecture
**[IMAGE: Slide 11 - SimpleCNN Block Diagram]**
- 4 standard convolutional blocks
- Progressive spatial reduction: 48→24→12→6→3
- Standard Conv2D layers throughout
- Global pooling + FC classifier head
- **Parameters**: 390,599

### 6.2 LiteCNN Architecture  
**[IMAGE: Slide 13 - LiteCNN Block Diagram]**
- 4 residual blocks with depthwise separable convolutions
- Skip connections for gradient flow
- Lightweight classifier head
- Progressive channel expansion with spatial reduction
- **Parameters**: 900,903
- **Key Innovation**: DepthwiseSep + Residual connections

---

## **7. MODEL COMPARISON TABLE**

| Feature | SimpleCNN | LiteCNN |
|---------|-----------|---------|
| **Architecture** | 4 Standard Conv Blocks | 4 Residual Blocks + DepthwiseSep |
| **Parameters** | 390,599 | 900,903 |
| **Key Innovation** | Baseline | DepthwiseSep + Residual |
| **Use Case** | Quick baseline | Production-ready |
| **CPU Latency** | 0.93ms | 5.33ms |
| **Val Accuracy** | ≥62.44% | ≥65.73% |

---

## **8. TRAINING RESULTS**

### SimpleCNN
```
Evaluating SimpleCNN on VALIDATION Set
✓ Loaded weights from: saved_models\SimpleCNN_best.pth
✓ Total Parameters: 390,599

Measuring CPU Latency...
✓ Average CPU Latency: 0.47 ms

Calculating VALIDATION Accuracy...
✓ VALIDATION Accuracy: 62.44% (2241/3589)
```

### LiteCNN
```
Evaluating LiteCNN on VALIDATION Set
✓ Loaded weights from: saved_models\LiteCNN_best.pth
✓ Total Parameters: 900,903

Measuring CPU Latency...
✓ Average CPU Latency: 5.00 ms

Calculating VALIDATION Accuracy...
✓ VALIDATION Accuracy: 65.73% (2359/3589)
```

### Final Summary Table
| Model | Parameters | Latency | Val Accuracy |
|-------|-----------|---------|--------------|
| SimpleCNN | 390,599 | 0.47 ms | 62.44% |
| LiteCNN | 900,903 | 5.00 ms | 65.73% |

---

## **9. ABLATION STUDIES**

### Overview of Experiments
- **E1**: With augmentation
- **E2**: With dropout value 0.5
- **E3**: With weight decay value 0.0001 and 0.001
- **E4**: With LR schedule different
- **E5**: SimpleCNN vs LiteCNN

---

### E1: Data Augmentation
**Results**:
- **Accuracy**: 62.66% vs Baseline 62.44%
- **Change**: ↑ 0.22%
- **Conclusion**: Augmentation can increase accuracy

**[IMAGE: Confusion Matrix - E1_aug_on]**
- Best performance on "happy" (87% precision)
- Moderate performance on other classes
- Shows improved generalization over no augmentation

---

### E2: Dropout 0.5
**Results**:
- **Accuracy**: 61.68% vs Baseline 62.44%
- **Change**: ↓ 0.76%
- **Conclusion**: Dropout value 0.5 will decrease the accuracy against 0.25

**[IMAGE: Confusion Matrix - E2_dropout_0.5]**
- Higher dropout rate hurts performance
- Suggests model benefits from moderate regularization only

---

### E3: Weight Decay 0.0001
**Results**:
- **Accuracy**: 61.36% vs Baseline 62.44%
- **Change**: ↓ 1.08%
- **Conclusion**: Weight decay will decrease accuracy, negative relationship

**[IMAGE: Confusion Matrix - E3_weightdecay_0.0001]**
- Aggressive regularization hurts training
- Model needs more capacity retention

---

### E3 (Alternative): Weight Decay 0.001
**Results**:
- **Accuracy**: 60.37% vs Baseline 62.44%
- **Change**: ↓ 2.07%
- **Conclusion**: Weight decay will decrease accuracy, negative relationship

**[IMAGE: Confusion Matrix - E3_weightdecay_0.001]**
- Even worse than 0.0001
- Confirms negative relationship between weight decay and accuracy for this task

---

### E4: LR Schedule (Cosine Annealing)
**Results**:
- **Accuracy**: 60.18% vs Baseline 62.44%
- **Change**: ↓ 2.26%
- **Conclusion**: LR Schedule using cosine will decrease the accuracy

**[IMAGE: Confusion Matrix - E4_Adam_cosine]**
- Cosine annealing not beneficial for this architecture
- Suggests constant or step LR is better suited

---

### E5: SimpleCNN vs LiteCNN
**Results**:
- **Accuracy**: 65.7% vs SimpleCNN 62.44%
- **Change**: ↑ 3.26%
- **Conclusion**: LiteCNN is more accurate

**[IMAGE: Confusion Matrix - LiteCNN (Accuracy: 65.7%)]**
- Full count-based confusion matrix showing raw predictions
- Strong diagonal performance
- Clear improvement over SimpleCNN baseline

---

## **10. F1-SCORES & IMPACT ANALYSIS**

**[IMAGE: Side-by-side confusion matrices]**
- **Left**: Count-based confusion matrix for LiteCNN
- **Right**: Normalized confusion matrix highlighting per-class recall

### Key Observations
- **Happy**: 86% recall (best performing class)
- **Surprise**: 77% recall
- **Sad**: 71% recall
- **Fear**: 61% recall (most confused with neutral)
- **Angry**: 63% recall
- **Disgust**: 59% recall (difficult due to class imbalance)
- **Neutral**: 61% recall (often confused with other subtle emotions)

### Common Confusion Patterns
- Fear ↔ Neutral confusion (25% of fear predicted as neutral)
- Sad ↔ Neutral confusion
- Angry misclassified as sad (17%)

---

## **11. LIMITATIONS**

### Limitation 1: Grayscale 48×48 Input
**[IMAGE: Low-resolution pixelated face]**
- Model uses small, colorless images
- Loses color information and struggles with non-studio lighting
- Fine-grained facial details lost in low resolution

### Limitation 2: Low Confidence (Dog Test)
**[IMAGE: Smiling dog photo]**
- Model predicts "Neutral" with low confidence
- Has never seen animal faces (Domain Gap)
- Trained exclusively on human facial expressions

---

## **12. FUTURE WORK**

### 1. Temporal Smoothing
**[IMAGE: Smoothing curve diagram]**
- Reduce flickering predictions in video
- Improve stability by averaging consecutive frames

### 2. TensorFlow Lite Conversion
**[IMAGE: Mobile phone with TFLite logo]**
- Convert model for mobile deployment
- Enable lightweight edge execution on smartphones/IoT devices

### 3. Diversify Training Data
**[IMAGE: Diverse group of faces - different ages, ethnicities]**
- More varied ages (children, elderly)
- More lighting conditions (natural, indoor, outdoor)
- More ethnicities for better generalization

---

## **13. KEY FIGURES FOR REPORT**

### Must-Include Images (from presentation slides):
1. **Slide 3**: Motivation visual (same text, different emotions)
2. **Slide 4**: Deployment trade-off diagram (Edge vs Data Center)
3. **Slide 6**: Dataset statistics bar chart
4. **Slide 9**: CLCM architecture block diagram
5. **Slide 11**: SimpleCNN architecture diagram
6. **Slide 13**: LiteCNN architecture diagram
7. **Slide 14**: SimpleCNN vs LiteCNN comparison table
8. **Slide 17**: Confusion matrix E1_aug_on
9. **Slide 18**: Confusion matrix E2_dropout_0.5
10. **Slide 19**: Confusion matrix E3_weightdecay_0.0001
11. **Slide 20**: Confusion matrix E3_weightdecay_0.001
12. **Slide 21**: Confusion matrix E4_Adam_cosine
13. **Slide 22**: Confusion matrix LiteCNN (count-based)
14. **Slide 23**: F1-scores visualization (normalized confusion matrix)
15. **Slide 24**: Limitations visual (48×48 pixelation + dog test)
16. **Slide 25**: Future work illustrations

---

## **14. NUMERICAL DATA EXTRACTED**

### Complete Ablation Results Summary
| Experiment | Config | Val Accuracy | Change vs Baseline | Conclusion |
|------------|--------|--------------|-------------------|------------|
| Baseline | SimpleCNN default | 62.44% | - | Reference point |
| E1 | Augmentation ON | 62.66% | +0.22% | ✓ Slight improvement |
| E2 | Dropout 0.5 | 61.68% | -0.76% | ✗ Too aggressive |
| E3a | Weight decay 0.0001 | 61.36% | -1.08% | ✗ Hurts performance |
| E3b | Weight decay 0.001 | 60.37% | -2.07% | ✗ Even worse |
| E4 | Cosine LR schedule | 60.18% | -2.26% | ✗ Not suitable |
| E5 | LiteCNN | 65.73% | +3.29% | ✓✓ Best model |

### Latency Measurements (from presentation)
- **SimpleCNN**: 0.47ms (CPU inference)
- **LiteCNN**: 5.00ms (CPU inference)
- Note: Slightly different from previous measurements due to different test environment

### Dataset Split Details
- Training: 28,709 images (80%)
- Validation: 3,589 images (10%)
- Test: 3,589 images (10%)
- Total: 35,887 images

---

## **15. GAPS IDENTIFIED & RECOMMENDATIONS**

### What's Now Clear (from presentation):
✅ **Complete ablation study results** with exact numbers  
✅ **Confusion matrices** for all experiments available  
✅ **Architecture diagrams** exist for SimpleCNN and LiteCNN  
✅ **Motivation and use cases** well articulated  
✅ **Limitations and future work** documented  
✅ **Dataset statistics** fully detailed  

### What Still Needs Work for Report:
⚠️ **Copy actual figure images** from presentation into LaTeX report  
⚠️ **Expand methodology sections** with implementation details  
⚠️ **Add training curves** (loss/accuracy over epochs) - not in presentation  
⚠️ **Generate per-class F1 score table** from confusion matrices  
⚠️ **Add demo app screenshots** - not in presentation slides  
⚠️ **Expand related work section** with literature review  
⚠️ **Add proper bibliography** with citations  
⚠️ **Reach 25+ page requirement** (currently 17 pages)  

---

## **16. PRESENTATION NARRATIVE FLOW**

1. **Hook**: Communication is more than words (emotional context matters)
2. **Problem**: Deployment efficiency vs accuracy trade-off
3. **Solution**: LiteCNN - the "usable middle"
4. **Dataset**: FER2013 with clear statistics
5. **Baselines**: HOG+SVM (traditional), CLCM (lightweight competitor)
6. **Our Models**: SimpleCNN (baseline) → LiteCNN (proposed)
7. **Experiments**: Systematic ablation studies (E1-E5)
8. **Results**: 3.29% improvement over SimpleCNN, competitive with CLCM
9. **Analysis**: Confusion matrices and F1-score breakdown
10. **Honesty**: Acknowledged limitations (grayscale, domain gap)
11. **Vision**: Future work for production deployment

---

## **END OF SUMMARY**

**Total slides analyzed**: 26  
**Figures identified**: 16 key images  
**Experiments documented**: 6 ablations (E1-E5)  
**Models compared**: 4 (HOG+SVM, CLCM, SimpleCNN, LiteCNN)

**Next Steps**:
1. Copy slide images into `/ppt_as_img` folder references in LaTeX
2. Expand report methodology and discussion sections
3. Add training curves and additional analysis
4. Generate per-class performance tables
5. Reach 25+ page requirement
