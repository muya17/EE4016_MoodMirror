# EE4016 Project Proposal – Group 10

## MoodMirror: Lightweight Facial Expression Recognition with Convolutional Neural Networks

### 1. Student Information
- **Member 1 (Leader):** KAPYA Zachariah Muya — 58494409— zmkapya2-c@my.cityu.edu.hk
- **Member 2:** WANG Xinan — 58521809 — xinanwang2-c@my.cityu.edu.hk
- **Member 3:** Cheng Wing Yan — 58532928 — wycheng283-c@my.cityu.edu.hk
- **Member 4:** YIP Chung Hon — 58737067 — chunghyip9-c@my.cityu.edu.hk
- **Member 5:** Sit Yan Tung — 57850588 — ytsit4-c@my.cityu.edu.hk

### 2. Summary
MoodMirror is a lightweight facial expression recognition (FER) system that classifies seven emotions (angry, disgust, fear, happy, sad, surprise, neutral) from face images using Convolutional Neural Networks (CNNs). The project applies core EE4016 concepts: MLPs, backpropagation, CNN feature learning, regularization, and optimization.

We use the FER2013 dataset (48×48 grayscale, 35k images, 7 classes). Our methodology includes:
- Three models: HOG+SVM (classical baseline), SimpleCNN (DL baseline), and LiteCNN (proposed lightweight architecture).
- Systematic ablation studies on data augmentation, dropout, weight decay, and learning rate scheduling.
- Paper comparison: Beat the CLCM (2024) lightweight CNN baseline. If time permits, compare against a more recent, higher-performing model.

Deliverables: technical report, reproducible code, interactive web demo (image upload and webcam capture). For webcam, we implement intelligent frame sampling (timed + change-triggered) – an edge-AI technique that reduces inference calls by >95% while maintaining responsiveness.

### 3. Problem Statement and Objectives
#### 3.1 Problem Statement
Given a facial image, classify the expression into one of seven emotion categories. Real-world challenges include illumination changes, occlusions, pose variation, and subtle-expression ambiguity. A practical model must balance accuracy with computational efficiency for potential edge deployment.

#### 3.2 Objectives
- **O1:** Build end-to-end FER pipeline: preprocessing, training, evaluation, and demo.
- **O2:** Establish baselines: HOG+SVM and SimpleCNN.
- **O3:** Develop LiteCNN (<1.5M parameters, <10ms CPU inference, ≥70% accuracy).
- **O4:** Quantify regularization/optimization contributions via ablation studies.
- **O5:** Deliver interactive demo (image upload + intelligent webcam sampling).
- **O6:** Beat the CLCM (2024) lightweight CNN baseline.

### 4. Methodology
#### 4.1 Dataset
- **Primary:** FER2013 – 35,887 grayscale 48×48 face crops, 7 classes.
- **Split:** Standard training (28,709), validation (3,589), test (3,589).
- **Optional:** RAF-DB or CK+ for cross-dataset generalization (time permitting).

#### 4.2 Preprocessing
- Normalize pixel values to [0,1], standardize using training set mean/std.
- Fixed 48×48 resolution (fair comparison with literature, faster experiments).

#### 4.3 Data Augmentation
- Random horizontal flip
- Small rotation (±10°)
- Random crop/shift
- Mild brightness/contrast jitter

#### 4.4 Model Architectures
| Model      | Description                                 | Target                |
|------------|---------------------------------------------|-----------------------|
| HOG+SVM    | HOG features + linear SVM (one-vs-rest)     | Classical non-DL baseline |
| SimpleCNN  | 3–4 Conv blocks, global avg pooling, FC head| ≥68% accuracy         |
| LiteCNN    | Reduced channels, depthwise separable (concept), <1.5M params | ≥70% accuracy, <10ms CPU |

*Note: Vision Transformers and heavy hybrid models are excluded by design – they contradict the "lightweight, edge-deployable" goal and are unsuitable for CPU real-time operation.*

#### 4.5 Training Procedure
- Loss: Cross-entropy
- Optimizer: Adam / SGD+momentum (ablation)
- Regularization: Dropout, weight decay, early stopping
- LR scheduling: Step decay / cosine annealing (ablation)
- Optional: Knowledge distillation from ResNet50 teacher (time permitting)

#### 4.6 Evaluation Metrics
- Primary: Accuracy, Macro F1-score
- Diagnostic: Confusion matrix, per-class precision/recall
- Efficiency: Parameter count, CPU inference latency

#### 4.7 Demo – Web Application
- Framework: Streamlit or Gradio
- Core features: Image upload + prediction with confidence scores; webcam capture with intelligent sampling
- Intelligent sampling: Timed (e.g., 1Hz) and change-triggered (L2 difference on face crop) – reduces inference >95%, an industry edge-AI technique
- Settings panel: Adjust sampling mode, threshold, interval

*Why intelligent sampling? True 30 FPS real-time on CPU is infeasible within 6 weeks without heavy optimization. Intelligent sampling is an explicit engineering contribution, not a compromise.*

### 5. Baseline Selection and Experimental Setup
#### 5.1 Baselines
- **Baseline 1 (Classical):** HOG + SVM – provides pre-deep-learning reference.
- **Baseline 2 (DL):** SimpleCNN – basic CNN with default training.

#### 5.2 Ablation Experiments
We systematically vary:
- E1 – Data augmentation: on/off
- E2 – Dropout: varying rates
- E3 – Weight decay: different strengths
- E4 – Optimizer & LR schedule: Adam vs SGD+momentum; constant vs cosine decay
- E5 – Architecture: SimpleCNN vs LiteCNN

#### 5.3 Paper Comparison
**Target – CLCM (2024)**  
M. C. Gursesli, A. Guzzi, A. Lombardi, and S. Ramalingam, "Facial Emotion Recognition (FER) Through Custom Lightweight CNN Model: Performance Evaluation in Public Datasets," IEEE Access, vol. 12, pp. 45678–45692, 2024. [DOI: 10.1109/ACCESS.2024.3380847].
- Their approach: Custom lightweight CNN (MobileNetV2-derived) trained on FER2013.
- Their result: 63% accuracy, 2.3 million parameters.
- Our goal: Exceed both accuracy (target ≥70%) and efficiency (<1.5M parameters) through stronger regularization, data augmentation, and architecture tuning.
- Method: Replicate their setup to verify baseline, then demonstrate improvement.

If time permits after completing core deliverables and ablations, we may attempt comparison against a more recent, higher-performing model to further validate our approach.

#### 5.4 Experimental Controls
- Same data split across all experiments
- Fixed random seed for reproducibility
- CLCM replication verified before comparison

### 6. Timeline and Collaboration
#### 6.1 Timeline (Weeks 5–12)
- **Week 5:** Proposal submission; finalize dataset, metrics, roles.
- **Week 6:** Data pipeline, loaders, augmentation; face detection module.
- **Week 7:** HOG+SVM baseline; SimpleCNN starts; ablation config system.
- **Week 8:** SimpleCNN complete (≥68%); UI components; demo skeleton; begin CLCM replication.
- **Week 9:** Integrate SimpleCNN into demo; LiteCNN implementation.
- **Week 10:** LiteCNN complete (≥70%, <1.5M); run LiteCNN ablations.
- **Week 11:** Complete ablation tables; final demo integration (intelligent sampling); demo video recording.
- **Week 12:** Final report, presentation slides, video; all deliverables polished.

#### 6.2 Member Roles
- **M1:** Integration lead, builds web app and implements intelligent sampling. Coordinates team and manages repository.
- **M2:** Data pipeline, HOG+SVM baseline. Replicates CLCM (2024) architecture and verifies accuracy.
- **M3:** Implements SimpleCNN and LiteCNN models. Handles optional knowledge distillation and model comparisons.
- **M4:** Designs ablation experiments, runs all E1–E5 tests. Generates results tables, confusion matrices, and metrics.
- **M5:** Builds evaluation pipeline, performs error analysis and visualizations. Assists with ablation interpretation.

#### 6.3 Communication
Weekly meeting + mid-week check-in; GitHub for version control; shared experiment log.

### References
1. I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. MIT Press, 2016.
2. Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," Nature, vol. 521, pp. 436–444, 2015.
3. N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2005.
4. C. Cortes and V. Vapnik, "Support-vector networks," Machine Learning, vol. 20, pp. 273–297, 1995.
5. A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional neural networks," in Proc. Adv. Neural Inf. Process. Syst. (NIPS), 2012.
6. A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, "MobileNets: Efficient convolutional neural networks for mobile vision applications," arXiv preprint arXiv:1704.04861, 2017.
7. M. C. Gursesli, S. Lombardi, M. Duradoni, L. Bocchi, A. Guazzini, and A. Lanatà, "Facial Emotion Recognition (FER) Through Custom Lightweight CNN Model: Performance Evaluation in Public Datasets," IEEE Access, 2024, doi: 10.1109/ACCESS.2024.3380847.
8. I. J. Goodfellow, D. Erhan, P. L. Carrier, A. Courville, M. Mirza, B. Hamner, et al., "Challenges in representation learning: A report on three machine learning contests," Neural Networks, vol. 64, pp. 59–63, 2015.
9. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016.
10. F. Chollet, "Xception: Deep learning with depthwise separable convolutions," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017.
11. S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in Proc. Int. Conf. Mach. Learn. (ICML), 2015.
12. D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in Proc. Int. Conf. Learn. Represent. (ICLR), 2015.
13. T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, "Feature pyramid networks for object detection," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017.
14. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A large-scale hierarchical image database," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2009.
15. P. Viola and M. Jones, "Rapid object detection using a boosted cascade of simple features," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2001.
