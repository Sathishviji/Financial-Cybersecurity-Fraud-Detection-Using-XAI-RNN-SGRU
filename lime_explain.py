import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lime.lime_tabular import LimeTabularExplainer

from model import RNN_SGRU

model = RNN_SGRU(input_size=10)
model.load_state_dict(torch.load("fraud_rnn_sgru_model.pth"))
model.eval()

# Load dataset
data = pd.read_csv("dataset/dirty_financial_transactions.csv")

X = data.drop("Class", axis=1).astype("float32")
y = data["Class"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (same as training)
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Use small sample
X_sample = X_pca[:1000]

# dummy prediction function for LIME
def predict_fn(x):

    x = torch.tensor(x, dtype=torch.float32)

    if len(x.shape) == 1:
        x = x.unsqueeze(0)

    with torch.no_grad():
        preds = model(x).detach().cpu().numpy()

    preds = preds.reshape(-1,1)

    return np.hstack((1 - preds, preds))


# create explainer
explainer = LimeTabularExplainer(
    X_sample,
    mode="classification"
)

# explain first transaction
exp = explainer.explain_instance(
    X_sample[0],
    predict_fn
)

print(exp.as_list())