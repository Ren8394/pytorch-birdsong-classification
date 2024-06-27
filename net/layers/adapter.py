import torch
import torch.nn as nn

class Residual_Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_size):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, adapter_size),
            nn.GELU(),
            nn.Linear(adapter_size, hidden_size)
        )
        self._init_weights()

    def forward(self, x):
        return x + self.adapter(x)

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)