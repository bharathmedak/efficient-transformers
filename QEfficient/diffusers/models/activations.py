import torch
from diffusers.models.activations import GELU


class QEffGELUScale(GELU):
    def forward(self, hidden_states, scale):
        hidden_states = hidden_states / scale  # BF16
        self.proj.bias = torch.nn.Parameter(self.proj.bias / scale)
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states * scale  # BF16
        hidden_states = self.gelu(hidden_states)  # BF16
        return hidden_states
