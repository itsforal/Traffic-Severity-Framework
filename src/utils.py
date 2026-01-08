import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

class SoftF1Loss(nn.Module):
    """
    Custom Differentiable Soft F-Beta Loss.
    Optimizes the model specifically for Recall in Safety-Critical scenarios.
    """
    def __init__(self, beta=1.0, epsilon=1e-7):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        tp = (targets * probs).sum()
        fp = ((1 - targets) * probs).sum()
        fn = (targets * (1 - probs)).sum()

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f_score = (1 + self.beta**2) * (precision * recall) / \
                 ((self.beta**2 * precision) + recall + self.epsilon)
        return 1 - f_score

def save_confusion_matrix(cm, model_name, path):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'CM - {model_name}')
    plt.savefig(path)
    plt.close()