import torch
import torch.nn.functional as F

from torch import nn, inf

from src.aux.types import QScheme
from src.quantization.gdnsq.gdnsq import Quantizer
from src.quantization.gdnsq.gdnsq_utils import QNMethod


class NoisyEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: torch.Tensor | None = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
        qscheme: QScheme = QScheme.PER_TENSOR,
        log_s_init: float = -12,
        rand_noise: bool = False,
        qnmethod: QNMethod = QNMethod.AEWGS,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            _freeze,
            device,
            dtype,
        )
        self.qscheme = qscheme
        self.rand_noise = rand_noise

        if self.qscheme == QScheme.PER_TENSOR:
            self.log_wght_s = nn.Parameter(
                torch.Tensor([log_s_init]), requires_grad=True
            )
        elif self.qscheme == QScheme.PER_CHANNEL:
            self.log_wght_s = nn.Parameter(
                torch.empty((num_embeddings, 1)).fill_(log_s_init), requires_grad=True
            )

        self._noise_ratio = nn.Parameter(torch.Tensor([1]), requires_grad=False)
        self.Q = Quantizer(
            self, torch.exp2(self.log_wght_s), 0, -inf, inf, qnmethod=qnmethod
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        s = torch.exp2(self.log_wght_s)
        self.Q.scale = s
        self.Q.rnoise_ratio.data = (
            self._noise_ratio
            if self.rand_noise
            else torch.zeros_like(self._noise_ratio)
        )
        if self.qscheme == QScheme.PER_CHANNEL:
            min = self.weight.amin(1, keepdim=True)
        elif self.qscheme == QScheme.PER_TENSOR:
            min = self.weight.amin()
        self.Q.zero_point = min

        weight = self.Q.dequantize(self.Q.quantize(self.weight))

        return F.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
