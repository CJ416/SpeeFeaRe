import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))


from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm

from speefeare.models.DAC.vq.layers import WNConv1d

class VectorQuantizer(nn.Module):
    '''
    Uses following tricks from Improved VQGAN
        - Factorized codes: Perform nearest neighbor lookup in low-dimensional space for imporved
            codebook usage
        - l2-normalization codes: converts euclidean distance to cosine similarity which improves training
            stability

    '''
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def decode_latents(self, latents):
        '''
        latents: Tensor[B, D, T]
        '''
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: [N, D]

        # L2-normalize latents and codebook
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # distance calculation
        # [[x1], [x2], ..., [xn]]
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()          # [n x D] @ [D x N] -> [n x N]
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(embed_id=indices)
        return z_q, indices
    
    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        '''
        embed_id: Tensor[B, T]
        '''
        return self.embed_code(embed_id).transpose(1, 2)

    def forward(self, z):
        '''
        Quantize the input tensor using a fixed codebook and returns 
        the corresponding codebook vectors

        Args:
        -----
            z: Tensor[B, D, T]
        Returns:
        ------
            Tensor[B, D, T]
                Quantized continuous representation of input
            Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook entries
            Tensor[1]
                Codebook loss to update codebook
            Tensor[B, T]
                Codebook indices
            Tensor[B, D, T]
                Projected latents (continuous representation of input before quantization)
        
        '''
        # 1. Factorized codes, Project input into low-dimensional space
        z_e = self.in_proj(z)                               # z_e: b, input_dim, T -> b, codebook_dim, T

        # 2. Quantize the latents
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction='none').mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction='none').mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )
        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e


class ResidualVectorQuantize(nn.Module):

    def __init__(
            self,
            input_dim: int = 512,
            n_codebooks: int = 9,
            codebook_size: int = 1024,
            codebook_dim: Union[int, list] = 8,
            quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim,int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]
        
        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(input_dim=input_dim, codebook_size=codebook_size, codebook_dim=codebook_dim[i])
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout
    
    def forward(self, z, n_quantizers: int = None):
        '''
        Quantize the input tensor using a fixed codebook and returns
        the corresponding codebook vectors
        Args:
        -----
            z: Tensor[B, D, T]
            n_quantizers: int
                Number of quantizers to use during training (for stochastic depth)
        Returns:
        ------
        dict
            A dictionary with the following keys:

            "z": Tensor[B, D, T]
                Quantized continuous representation of input
            "codes": Tensor[B, N, T]
                Codebook indices for each quantizer
            "latents:" Tensor[B, N*D, T]
                Projected latents (continuous representation of input before quantization) for each quantizer
            
            "vq/commitment_loss": Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook entries
            "vq/codebook_loss": Tensor[1]
                Codebook loss to updata codebook
        '''
        # import pdb
        # pdb.set_trace()

        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        n_quantizers = n_quantizers if n_quantizers is not None else self.n_codebooks

        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)
        
        for i, quantizer in enumerate(self.quantizers):
            # 1. Quantize the residual
            z_q_i, commitment_loss_i, codebook_loss_i, indice_i, latent_i = quantizer(residual)

            # 2. Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )

            z_q = z_q + z_q_i * mask[:, None, None]         # 
            residual = residual - z_q_i

            codebook_indices.append(indice_i)
            latents.append(latent_i)
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

        codes = torch.stack(codebook_indices, dim=1)        # B, N, T
        latents = torch.cat(latents, dim=1)                 # B, N*D, T

        return z_q, codes, latents, commitment_loss, codebook_loss
    
    def from_codes(self, codes: torch.Tensor):
        """
        Given the quantized codes, reconstruct the continuous representation
        Args:
        -----
            codes: Tensor[B, N, T]
                Codebook indices for each quantizer
        Returns:
        ------
            Tensor[B, D, T]
                Reconstructed continuous representation from the quantized codes
        """
        z_q = 0.0
        z_p = []

        for i, quantizer in enumerate(self.quantizers):
            code_i = codes[:, i, :]
            z_p_i = quantizer[i].decode_code(code_i)
            z_p.append(z_p_i)

            z_q_i = self.quantizer[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), codes
    
    def from_latents(self, latents: torch.Tensor):
        """
        Given the projected latents, reconstruct the continuous representation afeter quantization
        Args:
            latents: Tensor[B, N*D, T]      D is codebook_dim -> out_proj -> output_dim
                Projected latents for each quantizer

        Returns:
            Tensor[B, D, T]
                Quantized representation of input
            Tensor[B, D, T]
                Quantized representation of latent space
        
        """
        z_q = 0.0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[0]

        for i in range(n_codebooks):
            j, k = dims[i], dims[i+1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        
        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)
    


if __name__ == "__main__":
    rvq = ResidualVectorQuantize(quantizer_dropout=0.3)
    x = torch.randn(16, 512, 80)
    y = rvq(x)
    # import pdb
    # pdb.set_trace()
    # print(y)




    
    




            
        






        