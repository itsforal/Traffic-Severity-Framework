import torch.nn as nn
from torchvision import models
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from src.config import Config

class DeepInsightTransformer:
    """Transforms tabular data into 2D images using t-SNE mapping."""
    def __init__(self):
        self.tsne = TSNE(n_components=2, perplexity=5, random_state=Config.RANDOM_SEED)
        self.scaler = MinMaxScaler(feature_range=(0, Config.IMAGE_SIZE - 1))

    def get_feature_map(self, X_df):
        # Transpose to cluster features
        coords = self.tsne.fit_transform(X_df.T)
        return self.scaler.fit_transform(coords).astype(int)

class CrashEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_feats, 1)
        )
    def forward(self, x): return self.backbone(x)