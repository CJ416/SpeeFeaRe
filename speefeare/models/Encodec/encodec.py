import math
from pathlib import Path
import typing as tp

import numpy as np
import torch
from torch import nn


from speefeare.models.Encodec.vq.layers import SEANetDecoder, SEANetEncoder
from speefeare.models.Encodec.vq.quantize import ResidualVectorQuantize


'''
Reference repository: https://github.com/facebookresearch/encodec/blob/main/encodec/model.py
One of the core innovations of Encodec is the use of SEANet-based encoder and decoder
'''


class EncodecModel(nn.Module):
    '''
    Encodec model operating on raw audio waveforms
    
    '''
    def __init__(
            self,
            encoder: SEANetEncoder,
            decoder: SEANetDecoder,
            quantizer: ResidualVectorQuantize,
            sample_rate: int,
            channels: int, 
    ):
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_rate = math.ceil(sample_rate // np.prod(self.encoder.ratios))

    
    def encode(self, x: torch.Tensor):
        emb = self.encoder(x)
        z_q, codes, latents, commitment_loss, codebook_loss = self.quantizer(emb)

        return codes

'''
The architecture of Encodec model is similar to SoundStream, following the encoder-quantizer-decoder design paradigm.
The key differences are:
- Encoder and Decoder are based on SEANet architecture
- Quantizer uses RVQ, however, the codebooks implemented by fairseq is subtle difference,
    - Codebook update is performed via exponential moving average (EMA) rather than direct optimization

'''