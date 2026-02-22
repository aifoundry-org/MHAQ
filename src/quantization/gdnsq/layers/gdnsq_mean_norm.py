import torch
import torch.nn as nn


class _MeanNormNd(nn.Module):
    """
    BatchNorm-like layer with mean-only normalization:
        y = (x - mean) * weight + bias
    where mean is computed per-channel over batch + spatial dims.

    Keeps BN-like buffers/params for easier swapping:
      - weight, bias (if affine=True)
      - running_mean
      - running_var (unused, kept for state_dict compatibility)
      - num_batches_tracked
    """
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,             # kept for BN API compatibility (unused)
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features, **factory_kwargs))
            self.bias = nn.Parameter(torch.zeros(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features, **factory_kwargs))
            # Unused, but kept to be compatible with BN state_dict keys
            self.register_buffer("running_var", torch.ones(num_features, **factory_kwargs))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long, device=device))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1.0)
            self.num_batches_tracked.zero_()

    def _check_input_dim(self, x: torch.Tensor):
        raise NotImplementedError

    def _reduce_dims(self, x: torch.Tensor):
        # Mean over batch + all non-channel dims, keep per-channel mean
        # Input convention is always (N, C, ...)
        return (0,) + tuple(range(2, x.dim()))

    def _param_view_shape(self, x: torch.Tensor):
        # Shape [1, C, 1, 1, ...] to broadcast over batch/spatial dims
        return (1, self.num_features) + (1,) * (x.dim() - 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)

        if x.size(1) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} channels, got {x.size(1)}"
            )

        if self.training or not self.track_running_stats:
            mean = x.mean(dim=self._reduce_dims(x), keepdim=True)

            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked.add_(1)
                    if self.momentum is None:
                        # cumulative moving average (same convention as PyTorch BN)
                        m = 1.0 / float(self.num_batches_tracked.item())
                    else:
                        m = self.momentum
                    self.running_mean.mul_(1.0 - m).add_(m * mean.view(-1).detach())
        else:
            mean = self.running_mean.view(self._param_view_shape(x))

        y = x - mean

        if self.affine:
            w = self.weight.view(self._param_view_shape(x))
            b = self.bias.view(self._param_view_shape(x))
            y = y * w + b

        return y


class MeanNorm1d(_MeanNormNd):
    def _check_input_dim(self, x: torch.Tensor):
        # BN1d supports (N, C) or (N, C, L)
        if x.dim() not in (2, 3):
            raise ValueError(f"MeanNorm1d expected 2D or 3D input, got {x.dim()}D")


class MeanNorm2d(_MeanNormNd):
    def _check_input_dim(self, x: torch.Tensor):
        if x.dim() != 4:
            raise ValueError(f"MeanNorm2d expected 4D input, got {x.dim()}D")


class MeanNorm3d(_MeanNormNd):
    def _check_input_dim(self, x: torch.Tensor):
        if x.dim() != 5:
            raise ValueError(f"MeanNorm3d expected 5D input, got {x.dim()}D")
