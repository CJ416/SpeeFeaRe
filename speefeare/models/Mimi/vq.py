from _ import BaseQuantizer, 

class SplitResidualVectorQuantizer(BaseQuantizer):
    def __init__(
            self,
            *,
            n_q = 8,
            n_q_semantic = 1,
            **kwargs,
    ):
        super().__init_()
        assert n_q > n_q_semantic, "Number of semantic codebooks must be less than total number of codebooks"
        self.max_n_q = n_q
        self.n_q_semantic = n_q_semantic
        self.n_q_acoustic = n_q - n_q_semantic
        q_dropout = kwargs.pop('quantizer_dropout', 0.0)
        self.rvq = Re
        