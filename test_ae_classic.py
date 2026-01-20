# =====================================================
# AUTOENCODER TESTING (NO LABEL)
# Output: CSV for FINAL EVALUATION
# =====================================================

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# =====================================================
# 1. CONFIG
# =====================================================
MODEL_FILE = "ae_classic_model.pt"
TEST_FILE  = "test_data_qae_processed.csv"
OUTPUT_CSV = "ae_thresholded_results.csv"

INPUT_DIM = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("AUTOENCODER TESTING (NO LABEL)")
print("=" * 60)

# =====================================================
# 2. LOAD TEST DATA (TANPA LABEL)
# =====================================================
df = pd.read_csv(TEST_FILE)

X_test = df.values.astype(np.float32)
X_test = torch.tensor(X_test).to(DEVICE)

print(f"✓ Test data loaded: {X_test.shape}")

# =====================================================
# 3. MODEL DEFINITION (HARUS SAMA SAAT TRAIN)
# =====================================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 12),
            nn.ReLU(),
            nn.Linear(12, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder(INPUT_DIM).to(DEVICE)

checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print("✓ Model loaded")

# =====================================================
# 4. RECONSTRUCTION ERROR
# =====================================================
with torch.no_grad():
    recon = model(X_test)
    mse = torch.mean((recon - X_test) ** 2, dim=1)

mse = mse.cpu().numpy()

# =====================================================
# 5. THRESHOLDING (95 PERCENTILE)
# =====================================================
threshold = np.percentile(mse, 95)
y_pred = (mse > threshold).astype(int)

# Statistik
normal_count  = np.sum(y_pred == 0)
anomaly_count = np.sum(y_pred == 1)
total         = len(y_pred)

print(f"\nThreshold (95%) : {threshold:.6f}")
print("\n=== ANOMALY STATISTICS ===")
print(f"Normal samples  : {normal_count} ({normal_count/total*100:.2f}%)")
print(f"Anomaly samples : {anomaly_count} ({anomaly_count/total*100:.2f}%)")

# =====================================================
# 6. SAVE CSV OUTPUT (UNTUK EVALUATION)
# =====================================================
df_out = pd.DataFrame({
    "reconstruction_error": mse,
    "predicted_label": y_pred
})

df_out.to_csv(OUTPUT_CSV, index=False)

print(f"\n✓ Results saved to: {OUTPUT_CSV}")
print("=" * 60)
print("✅ AE TESTING FINISHED")
print("=" * 60)
