import torch
import os

class Config:
    # File Paths
    RAW_DATA_PATH = 'data/US_Accidents_March23.csv'
    PROCESSED_DATA_PATH = 'data/US_Accidents_MASTER.csv'
    OUTPUT_DIR = 'outputs/'
    
    # Global Settings
    RANDOM_SEED = 42
    SAMPLE_SIZE_ML = 50000
    SAMPLE_SIZE_DL = 40000
    
    # DL Hyperparameters (Rahim & Hassan 2021)
    IMAGE_SIZE = 120
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    EPOCHS = 15
    BETA_VALUES = [0.5, 1.0, 1.5, 2.0]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output dir if not exists
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)