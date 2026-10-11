# Composite Muon + AdamW optimizer.
#
# Ported from nf-flow-map/src/optimizers/torch_muon.py.
# Routes parameters by shape and name:
#   - 2D+ non-embedding -> torch.optim.Muon  (Newton-Schulz momentum)
#   - 2D+ embedding     -> AdamW with weight decay
#   - 1D (biases, norms) -> AdamW without weight decay
#
# Requires PyTorch >= 2.8 for torch.optim.Muon.

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class TorchMuon(Optimizer):
    """Composite optimizer: Muon for matrix weights, AdamW for the rest.

    This wraps ``torch.optim.Muon`` (Newton-Schulz orthogonalized momentum)
    together with ``torch.optim.AdamW`` in a single optimizer object.

    Parameter routing:
        * **2D+ params** whose name does NOT contain ``"embedding"`` -> Muon
        * **2D+ params** whose name contains ``"embedding"`` -> AdamW (with decay)
        * **1D params** (biases, LayerNorm weights) -> AdamW (no decay)

    The ``params`` argument must be an iterable of ``(name, parameter)`` tuples
    (i.e., from ``model.named_parameters()``) so that parameter names can be
    inspected for the routing decision.

    Args:
        params: Iterable of ``(name, parameter)`` tuples.
        lr: Learning rate (shared by Muon and AdamW).
        muon_momentum: Momentum for Muon.
        muon_weight_decay: Weight decay for Muon params.
        muon_nesterov: Whether Muon uses Nesterov momentum.
        adamw_betas: AdamW beta coefficients.
        adamw_weight_decay: Weight decay for AdamW embedding params.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        muon_momentum: float = 0.95,
        muon_weight_decay: float = 0.1,
        muon_nesterov: bool = True,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_weight_decay: float = 1e-4,
    ) -> None:
        named = list(params)
        all_params = [p for _, p in named]
        defaults = dict(lr=lr)
        super().__init__(all_params, defaults)

        muon_params: list[torch.nn.Parameter] = []
        adamw_decay: list[torch.nn.Parameter] = []
        adamw_nodecay: list[torch.nn.Parameter] = []

        for name, p in named:
            if not p.requires_grad:
                continue
            # Generally recommended to not do Muon on embedding layers
            if p.ndim >= 2 and "embedding" not in name:
                muon_params.append(p)
            elif p.ndim >= 2:
                adamw_decay.append(p)
            else:  # Do not decay 1D parameters
                adamw_nodecay.append(p)

        self._muon = torch.optim.Muon(
            muon_params,
            lr=lr,
            weight_decay=muon_weight_decay,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
        )
        self._adamw = torch.optim.AdamW(
            [
                {"params": adamw_decay, "weight_decay": adamw_weight_decay},
                {"params": adamw_nodecay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=adamw_betas,
        )

        self.param_groups = self._muon.param_groups + self._adamw.param_groups

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._muon.step()
        self._adamw.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"muon": self._muon.state_dict(), "adamw": self._adamw.state_dict()}

    def load_state_dict(self, state_dict) -> None:
        self._muon.load_state_dict(state_dict["muon"])
        self._adamw.load_state_dict(state_dict["adamw"])
        self.param_groups = self._muon.param_groups + self._adamw.param_groups
