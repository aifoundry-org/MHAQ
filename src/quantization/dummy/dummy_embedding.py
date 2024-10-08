from torch import Tensor, nn


class QuantizedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 padding_idx: int | None = None, max_norm: float | None = None, 
                 norm_type: float = 2, scale_grad_by_freq: bool = False, 
                 sparse: bool = False, _weight: Tensor | None = None, _freeze: bool = False, 
                 device=None, dtype=None) -> None:
        super().__init__(num_embeddings, embedding_dim, 
                         padding_idx, max_norm,
                         norm_type, scale_grad_by_freq, 
                         sparse, _weight, _freeze, 
                         device, dtype)
