from __future__ import annotations

import csv
import inspect
import json
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from transferable_samplers.models.nfmd_module import NFMDModule
from transferable_samplers.models.priors.normal_distribution import NormalDistribution
from transferable_samplers.models.transferable_nfmd_module import TransferableNFMDModule
from transferable_samplers.utils.dataclasses import SystemCond
from transferable_samplers.utils.checkpoint_utils import load_checkpoint_metadata, load_state_dict_from_ckpt
from transferable_samplers.utils.results import export_metrics

ROOT = Path(__file__).resolve().parents[2]


class AffineFlow(torch.nn.Module):
    """Analytical non-volume-preserving flow for checking molecular densities."""

    def __init__(self, summed=True):
        super().__init__()
        self.log_scale = torch.nn.Parameter(torch.tensor(0.7))
        self.logdet_sum = summed
        self.blocks = []

    def _logdet(self, value, mask):
        dim = value[0].numel() if mask is None else mask.sum(dim=-1) * value.shape[-1]
        if not self.logdet_sum:
            dim = 1
        return torch.ones(value.shape[0], device=value.device) * dim * self.log_scale

    def forward_and_log_det(self, x, context, encodings=None, mask=None, permutations=None):
        return (x - 0.2) * torch.exp(-self.log_scale), -self._logdet(x, mask)

    def inverse_and_log_det(self, z, context, encodings=None, mask=None, permutations=None):
        return z * torch.exp(self.log_scale) + 0.2, self._logdet(z, mask)


def make_model(transferable=False):
    net = AffineFlow(summed=transferable)
    kwargs = {} if transferable else {"eval_logdet_sum": True}
    cls = TransferableNFMDModule if transferable else NFMDModule
    return cls(
        net=net, optimizer=partial(torch.optim.AdamW, lr=1e-3), prior=NormalDistribution(), nfmd_t_max=0.05, **kwargs
    )


@pytest.mark.parametrize("transferable", [False, True])
def test_generated_density_matches_analytical_gaussian_and_forward_density(transferable):
    model = make_model(transferable)
    model._trainer = SimpleNamespace(datamodule=SimpleNamespace(num_atoms=4))
    condition = SystemCond(encodings={"atom_type": torch.zeros(4, dtype=torch.long)})
    x, energy = model.generate_proposal(model.net, 16, condition)
    gaussian = torch.distributions.Normal(0.2, torch.exp(model.net.log_scale))
    expected = -gaussian.log_prob(x).sum(dim=(1, 2))
    torch.testing.assert_close(energy, expected)
    torch.testing.assert_close(model.proposal_energy(model.net, x, condition), expected)
    if not transferable:
        assert model.net.logdet_sum is False  # training reduction restored after evaluation


def test_transferable_loss_ignores_padding_and_normalizes_active_dimensions():
    model = make_model(True)
    x = torch.randn(2, 3, 3)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    padded = torch.cat([x, torch.zeros(2, 2, 3)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(2, 2, dtype=torch.bool)], dim=1)
    first = model._compute_mle_loss_per_sample(x, None, None, mask)
    second = model._compute_mle_loss_per_sample(padded, None, None, padded_mask)
    torch.testing.assert_close(first, second)
    torch.manual_seed(47)
    loss = model._compute_nfmd_loss(x * mask[..., None], None, None, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.net.log_scale.grad is not None and torch.isfinite(model.net.log_scale.grad)


def test_checkpoint_loader_uses_ema_weights_and_preserves_metadata(tmp_path):
    checkpoint = tmp_path / "reference.ckpt"
    torch.save(
        {
            "state_dict": {"net.weight": torch.tensor([2.0])},
            "current_model_state": {"net.weight": torch.tensor([3.0])},
            "epoch": 949,
            "global_step": 237500,
        },
        checkpoint,
    )
    torch.testing.assert_close(load_state_dict_from_ckpt(str(checkpoint))["net.weight"], torch.tensor([2.0]))
    assert load_checkpoint_metadata(str(checkpoint)) == {"epoch": 949, "global_step": 237500}


def test_metric_exports_keep_per_system_and_aggregate_values(tmp_path):
    metrics = {
        "test/AL/resampled/energy-w2": torch.tensor(1.5),
        "test/mean/resampled/energy-w2": 1.5,
        "test/AL/undefined": float("nan"),
    }
    metadata = {"seed": 50, "world_size": 2, "sampler": "SNISSampler"}
    export_metrics(tmp_path, metrics, metadata)
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["metadata"] == metadata
    assert payload["metrics"]["test/AL/resampled/energy-w2"] == 1.5
    assert payload["metrics"]["test/AL/undefined"] is None
    with (tmp_path / "metrics.csv").open(newline="") as stream:
        rows = {row["metric"]: row["value"] for row in csv.DictReader(stream)}
    assert rows["test/mean/resampled/energy-w2"] == "1.5"
    assert rows["test/AL/undefined"] == ""
    assert not list(tmp_path.glob("*.tmp"))


@pytest.fixture
def config_environment(monkeypatch, tmp_path):
    for variable in ("NFMD_DATA_DIR", "NFMD_CACHE_DIR", "NFMD_OUTPUT_DIR"):
        monkeypatch.setenv(variable, str(tmp_path))
    monkeypatch.setenv("WANDB_PROJECT", "nfmd-tests")
    monkeypatch.setenv("WANDB_ENTITY", "nfmd-tests")
    return tmp_path


@pytest.mark.parametrize("system,mode", [(s, m) for s in ("GYDPETGTWG", "transferable") for m in ("train", "eval")])
def test_nfmd_recipes_compose_and_instantiate(system, mode, config_environment):
    tmp_path = config_environment
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name=mode,
            overrides=[f"experiment=nfmd/{system}_{mode}", f"paths.output_dir={tmp_path}", f"paths.work_dir={ROOT}"],
        )
        assert cfg.seed == (47 if mode == "train" else 50)
        assert cfg.model.objective == "nfmd"
        assert cfg.model.prior.std == 1.0
        assert cfg.model.nfmd_t_max == 0.05
        assert cfg.data.batch_size == 2048
        assert cfg.logger.wandb.project == "nfmd-tests"
        assert cfg.callbacks.sampling_evaluation.sampler.num_samples == (200000 if system == "GYDPETGTWG" else 10000)
        if system == "GYDPETGTWG":
            assert (cfg.model.net.num_blocks, cfg.model.net.num_layers, cfg.model.net.clip_range) == (6, 6, 3)
            cfg.model.net.img_size = 12
            cfg.model.net.num_layers = 1
        else:
            assert (cfg.model.net.num_blocks, cfg.model.net.layers_per_block, cfg.model.net.clip_range) == (8, 8, 5)
            cfg.model.net.max_num_tokens = 5
            cfg.model.net.layers_per_block = 1
        cfg.model.net.num_blocks = 2
        cfg.model.net.channels = 64
        model = hydra.utils.instantiate(cfg.model)
        assert isinstance(model, NFMDModule | TransferableNFMDModule)
        datamodule = hydra.utils.instantiate(cfg.data)
        assert datamodule.batch_size == 2048
        for value in cfg.callbacks.values():
            if value and "_target_" in value:
                hydra.utils.instantiate(value)


def test_all_baseline_configs_have_valid_constructor_arguments(config_environment):
    tmp_path = config_environment
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        for recipe in sorted((ROOT / "configs/experiment/baselines").rglob("*.yaml")):
            mode = recipe.parent.name
            experiment = recipe.relative_to(ROOT / "configs/experiment").with_suffix("").as_posix()
            cfg = compose(
                config_name=mode,
                overrides=[f"experiment={experiment}", f"paths.output_dir={tmp_path}", f"paths.work_dir={ROOT}"],
            )

            def check(value):
                if isinstance(value, dict):
                    if "_target_" in value:
                        target = hydra.utils.get_object(value["_target_"])
                        kwargs = {key: item for key, item in value.items() if not key.startswith("_")}
                        parameters = inspect.signature(target)
                        if value.get("_partial_"):
                            parameters.bind_partial(**kwargs)
                        else:
                            parameters.bind(**kwargs)
                    for child in value.values():
                        check(child)
                elif isinstance(value, list):
                    for child in value:
                        check(child)

            check(OmegaConf.to_container(cfg, resolve=True))


@pytest.mark.parametrize("system", ["GYDPETGTWG", "transferable"])
def test_launchers_require_checkpoint_and_share_multinode_run_directory(system):
    base = [sys.executable, str(ROOT / "scripts/nfmd/run.py")]
    result = subprocess.run(base + ["eval", system, "--dry-run"], capture_output=True, text=True)
    assert result.returncode != 0 and "requires --checkpoint" in result.stderr
    commands = []
    for rank in (0, 1):
        result = subprocess.run(
            base
            + [
                "train",
                system,
                "--nodes",
                "2",
                "--devices",
                "4",
                "--node-rank",
                str(rank),
                "--master-addr",
                "node0",
                "--run-id",
                "seed47-test",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        commands.append(result.stdout)
        assert "torch.distributed.run" in result.stdout and "--nproc_per_node 4" in result.stdout
        assert f"--node_rank {rank}" in result.stdout
        assert "${paths.log_dir}/${task_name}/seed47-test" in result.stdout
    assert commands[0] != commands[1]


def test_seed_zero_is_passed_to_lightning_before_instantiation(monkeypatch):
    from transferable_samplers import train as entrypoint

    called = []
    monkeypatch.setattr(entrypoint.lightning, "seed_everything", lambda seed, workers: called.append((seed, workers)))

    def stop(*args, **kwargs):
        raise RuntimeError("stop before data construction")

    monkeypatch.setattr(entrypoint.hydra.utils, "instantiate", stop)
    cfg = OmegaConf.create({"seed": 0, "torch_num_threads": 0, "data": {"_target_": "unused"}})
    with pytest.raises(RuntimeError, match="stop before data"):
        entrypoint.train.__wrapped__(cfg)
    assert called == [(0, True)]


def test_existing_rank_streams_are_repeatable_and_distinct(tmp_path):
    output = tmp_path / "distributed.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [sys.executable, str(ROOT / "tests/distributed_worker.py"), str(output)],
        cwd=ROOT,
        env=env,
        check=True,
        timeout=90,
    )
    result = json.loads(output.read_text())
    assert result["same_initial_weights"]
    assert result["repeatable"]
    assert result["different_rank_streams"]


def test_source_energy_copy_uses_full_molecular_density():
    from transferable_samplers.utils.dataclasses import SourceEnergyConfig

    model = make_model(False)
    model._trainer = SimpleNamespace(datamodule=SimpleNamespace(num_atoms=4))
    model.source_energy_config = SourceEnergyConfig(
        sample_batch_size=8, energy_batch_size=8, grad_batch_size=8, use_com_adjustment=False
    )
    condition = SystemCond(encodings={"atom_type": torch.zeros(4, dtype=torch.long)})
    source = model.build_source_energy(condition)
    x, energy = source.sample(16)
    expected = -torch.distributions.Normal(0.2, torch.exp(model.net.log_scale)).log_prob(x).sum(dim=(1, 2))
    torch.testing.assert_close(energy, expected)
    torch.testing.assert_close(source.energy(x), expected)
    assert model.net.logdet_sum is False
