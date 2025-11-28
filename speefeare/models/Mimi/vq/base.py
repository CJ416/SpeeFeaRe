
'''
Base class for quantizers
'''

from dataclasses import dataclass, field
import typing as tp

import torch
from torch import nn

@dataclass
class QuantizedResult:
    z: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor
    commitment_loss: tp.Optional[torch.Tensor] = None
    metrics: dict = field(default_factory=dict)


class BaseQuantizer(nn.Module):

    def __init__(self,):
        super().__init__()
        self._ema_frozen = False
    
    def forward(self, x: torch.Tensor, frame_rate: int) -> QuantizedResult:
        '''
        Args:
            x: Tensor[B, D, T]
        Given input x, return quantized result, quantized codes, bandwidth, 
        and commitment loss. Metrics are used to log update information.

        Frame_rate is used to compute bandwidth.
        '''
        raise NotImplementedError()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        '''x to codes'''
        raise NotImplementedError()
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        '''Codes to x'''
        raise NotImplementedError()
    
    @property
    def codebook_size(self) -> int:
        """ codebook size of each codebook"""
        raise NotImplementedError()

    @property
    def total_codebooks(self) -> int:
        ''' total number of codebooks'''
        raise NotImplementedError()
    
    @property
    def num_codebooks(self) -> int:
        '''' number of codebooks used'''
        raise NotImplementedError()
    
    
    


