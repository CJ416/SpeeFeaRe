'''
This is the core code implementation of Split Residual Vector Quantization

https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/quantization/vq.py

'''
from base import BaseQuantizer

class SplitResidualVectorQuantizer(BaseQuantizer):
    def __init__(
            self,
            *,
            n_q: int=8,
            n_q_semantic: int=1,
            **kwargs,
    ):
        super.__init__()
        assert n_q_semantic <= n_q, "n_q_semantic should be less than or equal to n_q"
        self.n_q = n_q
        self.n_q_semantic = n_q_semantic

        self.rvq_first = ResidualVectorQuantizer(
            n_q=n_q_semantic, force_projection=True, q_dropout=False, **kwargs
        )

        self.rvq_rest = ResidualVectorQuantizer(
            n_q=n_q - n_q_semantic,
            codebook_offset=1,
            force_projection=True,
            q_dropout=q_dropout,
            **kwargs,
        )
    
    def _renorm_and_add(
            self,
            first_val: torch.Tensor,
            rest_val: torch.Tensor,
            n_q_semantic: int,
            n_q_acoustic: int
    ):
        """ Renormalize the first and rest values and add them together """
        n_q = n_q_acoustic + n_q_semantic
        return (first_val * n_q_semantic + rest_val * n_q_acoustic) / n_q
    
    def forward(self, x: torch.Tensor, frame_rate: int) -> QuantizedResult:
        res_first = self.rvq_first(x, frame_rate)
        if self.n_q_semantic == self.n_q:
            return res_first
        res_rest = self.rvq_rest(x, frame_rate)

        full_quantized_emb = res_first.z + res_rest.z
        full_codes = torch.cat([res_first.codes, res_rest.codes], dim=-1)

        full_commitment_loss = self._renorm_and_add(
            res_first.commitment_loss,
            res_rest.commitment_loss,
            self.n_q_semantic,
            self.n_q - self.n_q_semantic,
        )

        return QuantizedResult(
            z=full_quantized_emb,
            codes=full_codes,
            bandwidth=res_first.bandwidth + res_rest.bandwidth,
            commitment_loss=full_commitment_loss,
            metrics={**res_first.metrics, **res_rest.metrics},
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        codes = self.rvq_first.encode(x)
        if self.n_q_semantic < self.n_q:
            codes_rest = self.rvq_rest.encode(x)
            codes = torch.cat([codes, codes_rest], dim=-1)
        
        return codes
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        z_first = self.rvq_first.decode(x[..., :self.n_q_semantic])
        if self.n_q_semantic < self.n_q:
            z_rest = self.rvq_rest.decode(x[..., self.n_q_semantic:])
            z = z_first + z_rest
        else:
            z = z_first
        return z
    

