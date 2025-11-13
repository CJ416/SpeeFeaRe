import os
import sys
import math
from typing import List
from typing import Union

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
import torch

from torch import nn

from speefeare.models.DAC.vq.layers import Snake1d
from speefeare.models.DAC.vq.layers import WNConv1d
from speefeare.models.DAC.vq.layers import WNConvTranspose1d
from speefeare.models.DAC.vq.quantize import ResidualVectorQuantize


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, padding=pad, dilation=dilation),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )
    
    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y

class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim=dim//2, dilation=1),
            ResidualUnit(dim=dim//2, dilation=3),
            ResidualUnit(dim=dim//2, dilation=9),
            Snake1d(dim//2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2)
            )
        )
    
    def forward(self, x):
        return self.block(x)
    
class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()

        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(dim=d_model, stride=stride)]
        
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size = 2 * stride,
                stride = stride,
                padding = math.ceil(stride / 2),
            ),
            ResidualUnit(dim=output_dim, dilation=1),
            ResidualUnit(dim=output_dim, dilation=3),
            ResidualUnit(dim=output_dim, dilation=9),
        )
    
    def forward(self, x):
        return self.block(x)
    
class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        layers = [WNConv1d(input_channel, channels,kernel_size=7, padding=3)]

        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2** (i+1)
            layers += [DecoderBlock(input_dim=input_dim, output_dim=output_dim, stride=stride)]
        
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class DAC(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 64,
            encoder_rates: List[int] = [2, 4, 8, 8],
            latent_dim: int = None,
            decoder_dim: int =1536,
            decoder_rates: List[int] = [8, 8, 4, 2],
            n_codebooks: int = 9,
            codebook_size: int = 1024,
            codebook_dim: Union[int, list] = 8,
            quantizer_dropout: bool = False,
            sample_rate: int = 44100,
            scale: int = 2,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))        # every downsampling doubles the channels

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.scale_size = 50/(self.sample_rate / self.hop_length)
        self.encoder = Encoder(d_model=encoder_dim, strides=encoder_rates, d_latent=latent_dim)
        
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.decoder = Decoder(
            input_channel=latent_dim,
            channels=decoder_dim,
            rates=decoder_rates,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)

        # self.delay = self.get_delay()        

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data
    
    def encode(
            self,
            audio_data: torch.Tensor,
            n_quantizers: int = None,
    ):
        '''
        Parameters:
        ----------
            audio_data: Tensor[B, 1, T]
                Input audio waveform
            n_quantizers: int, optional
                Number of quantizers to use during training (for RVQ with dropout)
        Returns:
        -------
            Tensor[B, D, T']
                Quantized continuous representation of input
            Tensor[B, D * n_codebooks, T']
                Projected latents (continuous representation of input before quantization)
            Tensor[B, n_codebooks, T']
                Codebook indices
        '''
        z = self.encoder(audio_data)                         # z: b, d_latent, T' 

        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(z, n_quantizers=n_quantizers)

        return z, codes, latents, commitment_loss, codebook_loss
    
    def decode(self, z: torch.Tensor):

        return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
        # 1. Preprocess the audio_data

        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate=sample_rate)
        z, codes, latents, commitment_loss, codebook_loss = self.encode(audio_data, n_quantizers)
        x = self.decode(z)

        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }
    
if __name__ == "__main__":
    import numpy as np
    from functools import partial

    model = DAC().to("cpu")

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))


    length = 88200 * 2
    x = torch.randn(1, 1, length).to("cpu")
    x.requires_grad_(True)
    x.retain_grad()

    out = model(x)["audio"]
    


        








        



