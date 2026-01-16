import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Import custom modules from src
from src.config import Config
from src.preprocess import run_full_preprocessing
from src.dl_engine import DeepInsightTransformer, CrashEfficientNet
from src.utils import SoftF1Loss, save_confusion_matrix

def train_dl_stage(X, y, stage_name, beta):
    """
    Core training loop for DeepInsight CNN Stages.
    Reproduction of the Hierarchical Inference structure.
    """
    print(f"\n--- Training {stage_name} (Beta: {beta}) ---")
    
    # 1. Feature to Image Mapping 
    transformer = DeepInsightTransformer()
    X_scaled, coords = transformer.fit_transform(X)
    
    # Create images on the fly or pre-calculate (Pre-calculate for 40k samples)
    print(f"[DL] Generating {len(X)} feature-images...")
    images = []
    for row in X_scaled:
        img = transformer.generate_image(row, coords)
        # Convert to 3-channel for EfficientNet compatibility
        img_3ch = np.stack([img]*3, axis=0) 
        images.append(img_3ch)
    
    X_imgs = torch.tensor(np.array(images), dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    # 2. Data Loader
    dataset = TensorDataset(X_imgs, y_tensor)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # 3. Model, Loss, Optimizer
    model = CrashEfficientNet().to(Config.DEVICE)
    criterion = SoftF1Loss(beta=beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 4. Training Loop
    model.train()
    for epoch in range(Config.EPOCHS):
        running_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(Config.DEVICE), batch_y.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Loss: {running_loss/len(loader):.4f}")

    # 5. Evaluation
    model.eval()
    with torch.no_grad():
        preds_all = []
        targets_all = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(Config.DEVICE)
            outputs = model(batch_x)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            preds_all.extend(preds)
            targets_all.extend(batch_y.numpy())

    # Save Confusion Matrix
    cm = confusion_matrix(targets_all, preds_all)
    cm_path = os.path.join(Config.OUTPUT_DIR, f"CM_{stage_name}_Beta_{beta}.png")
    save_confusion_matrix(cm, f"{stage_name} (Beta={beta})", cm_path)
    
    print(f"✅ {stage_name} Complete. Results saved to {Config.OUTPUT_DIR}")

def main():
    print("==================================================")
    print(" TRAFFIC ACCIDENT SEVERITY FRAMEWORK STARTING")
    print("==================================================")

    # PHASE 1: PREPROCESSING
    if not os.path.exists(Config.PROCESSED_DATA_PATH):
        print("\n[Phase 1] Preprocessing Raw Data...")
        df = run_full_preprocessing(Config.RAW_DATA_PATH)
        df.to_csv(Config.PROCESSED_DATA_PATH, index=False)
    else:
        print("\n[Phase 1] Loading existing processed data...")
        df = pd.read_csv(Config.PROCESSED_DATA_PATH)

    # PHASE 2: HIERARCHICAL DEEP LEARNING 
    print("\n[Phase 2] Starting DeepInsight Hierarchical Inference...")
    
    # --- STAGE 1: Fatal (Severity 4) vs Others ---
    # Sampling for DL to maintain computational feasibility
    df_sample = df.sample(n=min(len(df), Config.SAMPLE_SIZE_DL), random_state=Config.RANDOM_SEED)
    X = df_sample.drop(columns=['Severity'])
    y_fatal = (df_sample['Severity'] == 4).astype(int)

    # Investigate different Beta values as per report
    for beta in [1.0, 2.0]: # Beta=2.0 is Safety-Critical (Recall oriented)
        train_dl_stage(X, y_fatal, "Stage1_Fatal_vs_All", beta)

    # --- STAGE 2: Injury (3) vs PDO (1,2) ---
    # Filter out fatal cases for stage 2
    df_non_fatal = df[df['Severity'] < 4].sample(n=min(len(df), 20000), random_state=Config.RANDOM_SEED)
    X_stage2 = df_non_fatal.drop(columns=['Severity'])
    y_injury = (df_non_fatal['Severity'] == 3).astype(int)

    train_dl_stage(X_stage2, y_injury, "Stage2_Injury_vs_PDO", beta=1.0)

    print("\n==================================================")
    print(" PROJECT EXECUTION COMPLETE")
    print(f"All artifacts are available in: {Config.OUTPUT_DIR}")
    print("==================================================")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Critical Error: {str(e)}")
