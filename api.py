from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import torch
from model import RNN_SGRU
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = RNN_SGRU(input_size=10)
model.load_state_dict(torch.load("fraud_rnn_sgru_model.pth"))
model.eval()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    try:

        # Read file
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)

        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(file.file)

        else:
            return {"error": "Only CSV or Excel files allowed"}

        # Limit rows for performance
        df = df.head(2000)

        print("Dataset Columns:", df.columns)

        # Remove label column if exists
        X = df.drop("Class", axis=1, errors="ignore")

        # Convert text columns to numeric
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.factorize(X[col])[0]

        # Fill missing values
        X = X.fillna(0)

        # Convert everything to numeric
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Limit columns if too many
        if X.shape[1] > 30:
            X = X.iloc[:, :30]

        # Ensure at least 10 features
        while X.shape[1] < 10:
             X[f"dummy_{X.shape[1]}"] = 0

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        pca = PCA(n_components=10)
        X_pca = pca.fit_transform(X_scaled)

        results = []
        batch_size = 50

        # Batch prediction
        for i in range(0, len(X_pca), batch_size):

            batch = X_pca[i:i + batch_size]

            x = torch.tensor(batch, dtype=torch.float32)

            with torch.no_grad():
                preds = model(x).numpy()

            for p in preds:

                label = "Fraud" if p > 0.5 else "Normal"

                results.append({
                    "prediction": label,
                    "probability": float(p)
                })

        # Summary
        fraud_count = sum(1 for r in results if r["prediction"] == "Fraud")
        normal_count = len(results) - fraud_count

        summary = {
            "total": len(results),
            "fraud": fraud_count,
            "normal": normal_count
        }

        return {
            "summary": summary,
            "results": results
        }

    except Exception as e:
        return {"error": str(e)}