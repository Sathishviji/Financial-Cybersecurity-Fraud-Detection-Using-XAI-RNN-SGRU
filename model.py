import torch
import torch.nn as nn

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

        # convert (batch, features) → (batch, seq=1, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        out, _ = self.gru(x)

        out = out[:, -1, :]
        out = self.fc(out)

        return self.sigmoid(out)