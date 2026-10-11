# NFMD: Boltzmann generators

Training and numerical evaluation of NFMD for GYDPETGTWG and transferable
peptide Boltzmann generators. See the [paper](https://openreview.net/forum?id=guBm2gNlxU)
and the [Boltzmann generator guide](docs/bgs.md) for configurations and reproduction commands.

The main entry points are in `scripts/nfmd/`. Training defaults to seed 47;
evaluation defaults to seed 50 and proposal temperature T=1.00.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
# Fill WANDB_PROJECT and WANDB_ENTITY in .env.
wandb login
bash scripts/nfmd/train_GYDPETGTWG.sh
bash scripts/nfmd/eval_GYDPETGTWG.sh --checkpoint /path/to/epoch_2699.ckpt
```

Python 3.11+ and CUDA GPUs are required for the full molecular experiments.
The scripts run continuously and support one or multiple nodes. Dataset downloads,
normalization, molecular energies, EMA checkpoints, SNIS/SMC, and quantitative
metrics are included. W&B records the resolved configuration and metrics;
JSON/CSV results are also saved locally.

Standard MLE and flow matching recipes are available under
`configs/experiment/baselines/`; see [the baseline guide](docs/baselines.md).
The GYDPETGTWG MLE training launcher is `scripts/baselines/train_mle_GYDPETGTWG.sh`.

This release currently covers molecular experiments. The image experiments will
be documented with their code when added.

## Attribution and license

This Boltzmann-generator implementation branches from
[Transferable Samplers](https://github.com/transferable-samplers/transferable-samplers)
and builds on its molecular sampling and training code.

Original NFM contributions are licensed under MIT; see [LICENSE](LICENSE).
Upstream MIT copyright and permission notices are retained. Inherited code
also includes Apple, NVIDIA and CC BY 4.0 terms; those files retain their
original licenses. See [NOTICE](NOTICE) and individual file headers for the
applicable terms.
