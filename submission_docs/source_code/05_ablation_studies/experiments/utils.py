import random
import numpy as np
import torch
import time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_f1_per_class(confusion_matrix):
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1

def compute_latency(model, input_tensor, device, num_runs=100):
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    end = time.perf_counter()
    return (end - start) / num_runs