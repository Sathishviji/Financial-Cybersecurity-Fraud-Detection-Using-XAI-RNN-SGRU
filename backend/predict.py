import torch
import numpy as np
from rnn_sgru_model import RNN_SGRU

# load model
input_size = 10
model = RNN_SGRU(input_size)

model.load_state_dict(torch.load("fraud_rnn_sgru_model.pth"))
model.eval()

def predict_transaction(transaction):

    data = torch.tensor(transaction, dtype=torch.float32)

    with torch.no_grad():
        output = model(data.unsqueeze(0)).item()

    if output > 0.3:
        return "Fraud", output
    else:
        return "Normal", output