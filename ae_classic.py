# =====================================================
# CLASSIC AUTOENCODER (ANOMALY DETECTION)
# 16 FEATURES
# =====================================================

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import time

# =====================================================
# 1. CONFIG
# =====================================================
TRAIN_FILE = "qae_train_16fitur_scaled_10k.csv"
MODEL_FILE = "ae_classic_model.pt"

INPUT_DIM = 16
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)

print("="*60)
print("CLASSIC AUTOENCODER TRAINING")
print("="*60)

# =====================================================
# 2. LOAD TRAIN DATA
# =====================================================
if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(TRAIN_FILE)

df_train = pd.read_csv(TRAIN_FILE)
X_train = df_train.values.astype(np.float32)

print(f"✓ Train data loaded: {X_train.shape}")

X_train = torch.tensor(X_train).to(DEVICE)

# =====================================================
# 3. AUTOENCODER MODEL
# =====================================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 4)   # latent space
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 12),
            nn.ReLU(),
            nn.Linear(12, input_dim),
            nn.Sigmoid()      # karena data 0–1
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

model = Autoencoder(INPUT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# =====================================================
# 4. TRAIN LOOP
# =====================================================
print("\n🚀 Training started...\n")
start_time = time.time()

loss_history = []

for epoch in range(EPOCHS):
    perm = torch.randperm(X_train.size(0))
    epoch_loss = 0.0

    for i in range(0, X_train.size(0), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        batch = X_train[idx]

        optimizer.zero_grad()
        recon = model(batch)
        loss = loss_fn(recon, batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / (X_train.size(0) // BATCH_SIZE)
    loss_history.append(avg_loss)

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.6f}")

print(f"\n⏱ Training time: {(time.time()-start_time)/60:.2f} minutes")

# =====================================================
# 5. SAVE MODEL
# =====================================================
torch.save(
    {
        "model_state": model.state_dict(),
        "loss": loss_history
    },
    MODEL_FILE
)

print("="*60)
print("✅ TRAINING FINISHED")
print("Model saved:", MODEL_FILE)
print("="*60)
