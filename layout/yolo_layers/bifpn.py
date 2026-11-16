"""Existing Windows environment implementation, preserved for checkpoint parity.

Source: env1/site-packages/ultralytics/nn/bifpn.py, SHA256
2732ab35f56606f61623af9b272f35f88db73079ac5fdbb7b5dc1bf6830358e0.
Do not replace this with a generic BiFPN or change its three-weight denominator.
"""
import torch
import torch.nn as nn


class Concat_BiFPN(nn.Module):
    def __init__(self, dimension=1):
        super(Concat_BiFPN, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)
