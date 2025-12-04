'''
Traditional codec models (Encodec, Hifi-Codec) follows the paradigm of
''raw waveform -> Encoder (downsampling) -> quantizer -> Decoder (upsampling) -> reconstructed waveform''

WaveTokenizer follows vocos models' paradigm of
''raw waveform -> Encoder (downsampling) -> quantizer -> ISTFT coeffs -> reconstructed waveform''

The core innovation of WaveTokenizer:
* ConvNext convolutional blocks
* Attention-based Decoder
* Longer context window length

Core code implementation for WaveTokenizer codec model is listed below.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. Attention Module for Semantic Improvement
# - Add attention mechanism to the decoder to better capture long-range dependencies in audio signals.
# - Model can discover semantic patterns in audio data itself, no need for extra semantic model.
# -----------------------------------------------------------------------------
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        '''x: (B, dim, T)'''
        B, dim, T = x.shape
        x_in = x.permute(0, 2, 1)               # B, T, dim
        h = self.norm(x_in)
        q,k,v = self.qkv(h).chunk(3, dim=-1)    # B, T, dim
        q = q.reshape(B, T, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x_attn = (attn @ v).permute(0, 2, 1, 3).reshape(B, T, dim)

        x_out = self.proj(x_attn) + x_in

        return x_out.permute(0, 2, 1)   # B, dim, T
    

# -----------------------------------------------------------------------------
# 2. ConvNeXt Block (Backbone of Vocos)
# -----------------------------------------------------------------------------
class ConvNeXtBlock(nn.Module):
    '''Vocos Style ConvNeXt Block'''
    def __init__(self, dim, intermediate_dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # PointWise Conv
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()

        self.pwconv2 = nn.Linear(intermediate_dim, dim)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)))

    def forward(self, x):
        input = x
        x = self.dwconv(x).permute(0, 2, 1)

        x = self.norm(x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x
        
        x = x.permute(0, 2, 1)

        x = input + x
        return x
    
class WavTokenizerDecoder(nn.Module):
    def __init__(self, 
                 in_channels=512,
                 dim=512,
                 n_fft=1024,
                 hop_length=240,
                 num_layers=8
                 ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        # 1. Initial Conv1d Layer
        self.conv_in = nn.Conv1d(in_channels, dim, kernel_size=7, padding=3)

        # 2. Attention Layers
        self.attention = SelfAttentionBlock(dim, num_heads=8)

        # 3. ConvNeXt Blocks
        self.backbone = nn.ModuleList(
            ConvNeXtBlock(dim, intermediate_dim=dim*4)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # 4. Output Projection Layer to ISTFT coeffs
        out_dim = (n_fft // 2 + 1) * 2
        self.out_proj = nn.Conv1d(dim, out_dim, kernel_size=7, padding=3)

        # 5. ISTFT, torch.istft
        self.window = torch.hann_window(n_fft)

    def forward(self, z_q):
        """
        Args:
            z_q: [B, C, T_frame]
        Returns:
            audio: [B, 1, T_audio]
        """
        x = self.conv_in(z_q)   # B, dim, T_frame

        x = self.attention(x)   # B, dim, T_frame

        for blk in self.backbone:
            x = blk(x)          # B, dim, T_frame
        
        x = x.permute(0, 2, 1)  # B, T_frame, dim
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # B, dim, T_frame

        spec = self.out_proj(x) # B, out_dim, T_frame

        audio = self.istft_synthesis(spec)

        return audio

    def istft_synthesis(self, spec):
        '''
        Convert predicted STFT coefficients to waveform using ISTFT.
        '''
        B, C, T = spec.shape        # C = (n_fft // 2 + 1) * 2
        n_freq = self.n_fft // 2 + 1

        spec = spec.view(B, n_freq, 2, T).permute(0, 1, 3, 2).contiguous()
        real = spec[..., 0]
        imag = spec[..., 1]
        complex_spec = torch.complex(real, imag)

        audio = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(spec.device),
            center=True,
            length = T * self.hop_length
        )

        return audio.unsqueeze(1)

if __name__ == "__main__":
    z_q = torch.randn(1, 512, 100)
    
    decoder = WavTokenizerDecoder(in_channels=512, dim=512, n_fft=1024, hop_length=240)
    audio = decoder(z_q)
    print(f"Input Zq shape: {z_q.shape}")
    print(f"Output Audio shape: {audio.shape}")






