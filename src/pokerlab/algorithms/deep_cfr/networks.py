"""Small configurable PyTorch MLPs; PyTorch remains an optional dependency."""

from __future__ import annotations

from collections.abc import Sequence

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised in base-only installations
    torch = None
    nn = None


def require_torch():
    if torch is None:
        raise ImportError(
            "Deep CFR neural training requires PyTorch; install pokerlab[deep-cfr]"
        )
    return torch


if nn is not None:

    class _MLP(nn.Module):
        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_layers: Sequence[int],
            activation: str = "relu",
        ) -> None:
            super().__init__()
            if input_dim < 1 or output_dim < 1:
                raise ValueError("network dimensions must be positive")
            activations = {
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
                "leaky_relu": nn.LeakyReLU,
            }
            if activation not in activations:
                raise ValueError("unsupported activation")
            layers: list[nn.Module] = []
            width = input_dim
            for hidden in hidden_layers:
                if hidden < 1:
                    raise ValueError("hidden widths must be positive")
                layers.extend((nn.Linear(width, hidden), activations[activation]()))
                width = hidden
            self.hidden = nn.Sequential(*layers) if layers else nn.Identity()
            self.output = nn.Linear(width, output_dim)
            self.output_dim = output_dim
            self.reset_parameters()

        def reset_parameters(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def forward(self, features):
            return self.output(self.hidden(features))

        def masked_probabilities(self, features, legal_mask):
            if legal_mask.dtype is not torch.bool:
                legal_mask = legal_mask.to(dtype=torch.bool)
            if not torch.all(legal_mask.any(dim=-1)):
                raise ValueError("each information set needs at least one legal action")
            logits = self(features).masked_fill(~legal_mask, -torch.inf)
            return torch.softmax(logits, dim=-1)

    class AdvantageNetwork(_MLP):
        """One player's action-advantage function; legal actions mask its loss."""

    class StrategyNetwork(_MLP):
        """Average-policy network with exactly zero mass on illegal actions."""

else:

    class AdvantageNetwork:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            require_torch()

    class StrategyNetwork:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            require_torch()
