import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from model import RNN_SGRU

# =========================
# Load Dataset
# =========================

data = pd.read_csv("dataset/creditcard.csv")

print("Dataset shape:", data.shape)

X = data.drop("Class", axis=1).astype("float32")
y = data["Class"].astype("float32")

# =========================
# Feature Scaling
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# Handle Class Imbalance
# =========================

X_resampled = X_scaled
y_resampled = y

print("After SMOTE:", X_resampled.shape)

# =========================
# Kernel PCA
# =========================

pca = PCA(n_components=10)
X_kpca = pca.fit_transform(X_resampled)

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_kpca, y_resampled, test_size=0.2, random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# =========================
# RNN-SGRU Model
# =========================

class RNN_SGRU(nn.Module):

    def __init__(self, input_size):
        super(RNN_SGRU, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = x.unsqueeze(1)

        out, _ = self.gru(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return self.sigmoid(out)


# =========================
# Model Initialization
# =========================

input_size = X_train.shape[1]

model = RNN_SGRU(input_size)

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# Training
# =========================

epochs = 10

for epoch in range(epochs):

    model.train()

    outputs = model(X_train).squeeze()

    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# =========================
# Evaluation
# =========================

model.eval()

with torch.no_grad():

    predictions = model(X_test).squeeze()

    predicted_labels = (predictions > 0.3).float()

accuracy = accuracy_score(y_test, predicted_labels)
precision = precision_score(y_test, predicted_labels)
recall = recall_score(y_test, predicted_labels)
f1 = f1_score(y_test, predicted_labels)

print("\nModel Performance")
print("----------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# =========================
# Save Model
# =========================

torch.save(model.state_dict(), "fraud_rnn_sgru_model.pth")

print("\nModel saved successfully.")