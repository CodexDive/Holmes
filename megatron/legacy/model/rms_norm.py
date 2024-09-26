# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
from torch import nn
from apex.normalization.fused_layer_norm import FusedRMSNormAffineMixedDtypesFunction

class RMSNorm(torch.nn.Module):

    def __init__(self,
                 dim: int,
                 eps: float = 1e-6,
                 sequence_parallel: bool = False):
        """RMS Normaliation module

        Args:
            dim (int): The width of input, i.e. hidden size
            eps (float): epsilon to use for the norm, default to 1e-6
            sequence_parallel (bool): Set to true if sequence parallelism is being used,
              this marks the weights as needing to be allreduced.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.normalized_shape = torch.Size((dim,))

        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return FusedRMSNormAffineMixedDtypesFunction.apply(x, self.weight, self.normalized_shape, self.eps)
