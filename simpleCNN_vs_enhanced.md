## SimpleCNN Experiment Update

We also tried a stronger modified version of SimpleCNN with heavier augmentation, class-weighted loss, label smoothing, AdamW, OneCycleLR, AMP, gradient clipping, and a bottleneck layer.

However, this version performed worse than the simpler baseline, reaching only:
- Best validation accuracy: 54.39%
- Test accuracy: 60.91%

Because of this, we decided to keep the simpler SimpleCNN baseline for the final integration and presentation, and continue focusing on the LiteCNN work that was already prepared.
