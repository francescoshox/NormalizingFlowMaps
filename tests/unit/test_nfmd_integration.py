from pathlib import Path

import hydra
import lightning
import pytest
import torch
from hydra import compose, initialize_config_dir
from lightning import LightningDataModule, Trainer
from torch.utils.data import DataLoader

from transferable_samplers.callbacks.ema_weight_averaging import EMAWeightAveraging
from transferable_samplers.callbacks.loss_evaluation_callback import LossEvaluationCallback
from transferable_samplers.callbacks.sampling_evaluation_callback import SamplingEvaluationCallback
from transferable_samplers.evaluation.metrics.ess import normalized_ess
from transferable_samplers.samplers.snis_sampler import SNISSampler
from transferable_samplers.utils.dataclasses import EvalContext, SamplesData, SystemCond, TargetEnergy
from transferable_samplers.utils.checkpoint_utils import load_state_dict_from_ckpt

ROOT = Path(__file__).resolve().parents[2]


def small_model(system):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"experiment=nfmd/{system}_train"])
        cfg.model.net.channels = 64
        cfg.model.net.num_blocks = 2
        if system == "GYDPETGTWG":
            cfg.model.net.num_layers = 1
            cfg.model.net.img_size = 12
        else:
            cfg.model.net.layers_per_block = 1
            cfg.model.net.max_num_tokens = 4
        return hydra.utils.instantiate(cfg.model)


def condition():
    return SystemCond(
        encodings={
            "atom_type": torch.zeros(4, dtype=torch.long),
            "aa_type": torch.ones(4, dtype=torch.long),
            "aa_pos": torch.zeros(4, dtype=torch.long),
            "seq_len": torch.tensor([2]),
        },
        permutations={
            "n2c_residue-by-residue_standard": torch.arange(4),
            "n2c_residue-by-residue_standard_flip": torch.arange(4).flip(0),
        },
    )


@pytest.mark.parametrize("system", ["GYDPETGTWG", "transferable"])
def test_real_nfmd_network_forward_and_generated_densities_agree(system):
    torch.set_num_threads(2)
    lightning.seed_everything(47, workers=True)
    model = small_model(system).eval()
    with torch.no_grad():
        for block in model.net.blocks:
            torch.nn.init.normal_(block.proj_out.proj_out.weight, std=0.01)
            block.proj_out.proj_out.bias.fill_(0.05)
    x, energy = model.generate_proposal(model.net, 4, condition())
    torch.testing.assert_close(energy, model.proposal_energy(model.net, x, condition()), rtol=1e-4, atol=1e-4)
    assert torch.isfinite(energy).all()


class SyntheticData(LightningDataModule):
    num_atoms = 4
    batch_size = 4
    num_workers = 0
    temperature = 310
    val_sequences = ["AL"]
    test_sequences = ["AL"]

    def __init__(self):
        super().__init__()
        self.coords = torch.randn(8, 4, 3)

    def train_dataloader(self):
        cond = condition()
        return DataLoader(
            [{"x": x, "encodings": cond.encodings, "permutations": cond.permutations} for x in self.coords],
            batch_size=4,
        )

    def test_dataloader(self):
        return DataLoader([torch.tensor(0)])

    def prepare_eval(self, sequence, stage):
        energy_fn = lambda x: 0.5 * x.square().sum(dim=(1, 2))
        return EvalContext(
            true_data=SamplesData(self.coords, energy_fn(self.coords)),
            target_energy=TargetEnergy(energy_fn, torch.tensor(1.0)),
            normalization_std=torch.tensor(1.0),
            system_cond=condition(),
            topology=None,
            tica_model=None,
        )


class NumericalEvaluator:
    def evaluate(self, samples, context, prefix):
        return {
            f"{prefix}/resampled/mean-energy": samples["resampled"].E_target.mean(),
            f"{prefix}/resampled/effective-sample-size": normalized_ess(samples["resampled"].logw),
        }


@pytest.mark.parametrize("system", ["GYDPETGTWG", "transferable"])
def test_cpu_train_ema_checkpoint_and_eval_export_include_losses(system, tmp_path):
    import json

    torch.set_num_threads(2)
    lightning.seed_everything(47, workers=True)
    model = small_model(system)
    data = SyntheticData()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        limit_test_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[EMAWeightAveraging()],
    )
    trainer.fit(model, datamodule=data)
    assert trainer.global_step == 2
    path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(path)
    restored = small_model(system)
    restored.load_state_dict(load_state_dict_from_ckpt(str(path)))
    restored._run_seed = 50
    restored._checkpoint_path = str(path)
    restored._checkpoint_epoch = 0
    callbacks = [
        SamplingEvaluationCallback(NumericalEvaluator(), SNISSampler(64, 0.002), str(tmp_path / "evaluation")),
        LossEvaluationCallback(batch_size=4),
    ]
    tester = Trainer(
        accelerator="cpu",
        devices=1,
        limit_val_batches=1,
        limit_test_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=callbacks,
        inference_mode=False,
    )
    lightning.seed_everything(50, workers=True)
    tester.test(restored, datamodule=data)
    result = json.loads((tmp_path / "evaluation/test/epoch_0000/metrics.json").read_text())
    assert result["metadata"]["seed"] == 50
    assert result["metadata"]["num_proposals"] == 64
    assert result["metadata"]["proposal_temperature"] == 1.0
    assert result["metrics"]["test/AL/resampled/effective-sample-size"] > 0
    assert result["metrics"]["test/AL/eval-loss"] is not None
    assert result["metrics"]["test/mean/eval-loss"] is not None
    assert (tmp_path / "evaluation/test/epoch_0000/metrics.csv").exists()
