# Boltzmann generator reproduction

NFMD recipes live in `configs/experiment/nfmd/`. All four scripts in
`scripts/nfmd/` select those recipes explicitly. Activate the installed Python
environment before running them. Additional Hydra `key=value` overrides can be
appended to any command; `--dry-run` prints the command without executing it.

## Setup and data

Install with `pip install -e .`, copy `.env.example` to `.env`, set
`WANDB_PROJECT` and `WANDB_ENTITY`, and authenticate with `wandb login`.
W&B is enabled in every default recipe, with no fixed project, entity, or run ID.
The resolved configuration includes the seed, architecture, data, optimizer,
training schedule, sampler, and device layout.

The datamodules download their public Hugging Face data and prepare evaluation
references. `GYDPETGTWG` has 138 atoms and uses data at 310 K. Transferable training
uses ManyPeptidesMD peptides of lengths 2–8, with the validation/test sequences
listed in `configs/data/transferable/up_to_8aa.yaml`.

`NFMD_DATA_DIR` defaults to `./data`: GYDPETGTWG is stored beneath
`sequential-boltzmann-generators-data/`, and transferable data beneath
`many-peptides-md/`. `NFMD_CACHE_DIR` defaults to `./.cache` and
`NFMD_OUTPUT_DIR` to `./outputs`. For multiple nodes, use absolute shared paths
accessible with the same environment on every node. Transferable data preparation runs on
global rank zero before other ranks start loading the data. GYDPETGTWG uses
Lightning’s per-node preparation with Hugging Face download caching.

## Training

| Setting | GYDPETGTWG | Transferable |
|---|---|---|
| Training seed | 47 | 47 |
| Blocks × layers | 6 × 6 | 8 × 8 |
| Width | 384 | 384 |
| Clipping range | 3 | 5 |
| NFMD interpolation maximum | 0.05 | 0.05 |
| Optimizer / maximum learning rate | Muon / 2e-4 | Muon / 2e-4 |
| Global batch size | 2048 | 2048 |
| Epochs | 3000 | 1000 |
| Batches per epoch | Full training dataset | 250 |
| Validation interval | 100 epochs | 50 epochs |
| Validation proposals per peptide | 200,000 | 10,000 |
| Sampling / energy batch | 10,000 | 10,240 |
| Energy-gradient batch | 2048 | 10,240 |
| EMA decay | 0.999 | 0.999 |
| Reference training layout | 2 nodes × 4 GPUs | 4 nodes × 4 GPUs |

The learning rate follows the training-step OneCycle schedule (5% warmup,
initial and final learning rate equal to the maximum divided by 500).
Training uses centered/normalized coordinates with COM augmentation.

```bash
bash scripts/nfmd/train_GYDPETGTWG.sh --devices 4
bash scripts/nfmd/train_transferable.sh --devices 4
```

Scripts default to one node and one GPU. The global batch remains 2048 and is
divided across processes. Matching the reference hardware layout is the closest
reproduction; a different layout changes random-stream consumption and can
change numerical results. Memory requirements depend on the per-GPU batch.

For two-node GYDPETGTWG training, launch one command per allocated node:

```bash
# Node 0
bash scripts/nfmd/train_GYDPETGTWG.sh --nodes 2 --devices 4 --node-rank 0 \
  --master-addr FIRST_NODE_HOSTNAME --run-id GYDPETGTWG_seed47
# Node 1
bash scripts/nfmd/train_GYDPETGTWG.sh --nodes 2 --devices 4 --node-rank 1 \
  --master-addr FIRST_NODE_HOSTNAME --run-id GYDPETGTWG_seed47
```

Use the same run ID, master address, port, configuration, and shared directories
on all nodes. The reference transferable layout uses `--nodes 4 --devices 4`
with node ranks 0–3. Launch one script per node inside your allocation.

Seeding uses the existing `lightning.seed_everything(seed, workers=True)` and
rank-dependent RNG advancement in `BaseLightningModule.setup`. Global rank
identifies streams across nodes; it is not the local GPU index. Streaming shard
resampling and shuffling also receive the run seed and global rank explicitly.
Repeatability requires the same seed, world size, workers, software, and hardware.

Checkpoints include EMA evaluation weights under `state_dict`. Keep the
checkpoint you intend to report; evaluation always takes an explicit path.
The reference GYDPETGTWG checkpoint was epoch 2699, and the reference transferable
checkpoint was epoch 949. Training longer does not automatically make the final
checkpoint the reference result.

## Evaluation

```bash
bash scripts/nfmd/eval_GYDPETGTWG.sh --checkpoint /path/to/GYDPETGTWG.ckpt
bash scripts/nfmd/eval_transferable.sh --checkpoint /path/to/transferable.ckpt
```

Both default to seed **50**, SNIS, and proposal temperature **T=1.00**
(`model.prior.std=1.0`). GYDPETGTWG generates **200,000** proposals and transferable
**10,000 per peptide**. Both filter the top 0.2% of importance weights before
resampling. T here is the proposal temperature ratio; the molecular target
energy remains at the dataset temperature. It is separate from the NFMD
training interpolation parameter.

The evaluator compares up to 10,000 conformations against each reference set.
The proposal count is the sampling budget, not the metric sample count.
Quantitative results include energy, torus and TICA Wasserstein distances,
KMeans JSD, chirality statistics, and effective sample size.

Optional SMC recipes:

```bash
bash scripts/nfmd/eval_GYDPETGTWG.sh --checkpoint /path/to/GYDPETGTWG.ckpt --sampler smc
bash scripts/nfmd/eval_transferable.sh --checkpoint /path/to/transferable.ckpt --sampler smc
bash scripts/nfmd/eval_transferable.sh --checkpoint /path/to/transferable.ckpt --sampler mala
```

| Setting | GYDPETGTWG SMC | Transferable SMC | Transferable MALA |
|---|---|---|---|
| Annealing steps | 500 | 100 | 50 |
| Initial epsilon | 1e-5 | 1e-5 | 5e-4 |
| ESS threshold | 0.5 | 1.0 | 1.0 |
| Resampling | Multinomial | Systematic | Systematic |
| Energy cutoff | -300 | 100 | 100 |
| Adaptive step size | No | No | Yes |

These are retained evaluation settings; this guide does not claim SMC outperforms
SNIS or that the seeded training recipes reproduce historical weights exactly.
The historical transferable run did not record a training seed. The release fixes
it at 47 for reproducible new training; evaluating that run's original checkpoint
is the way to reproduce its historical model. Changing GYDPETGTWG evaluation from
seed 47 to 50 also requires recomputing its reported numbers.

## Outputs

Each run has a unique directory beneath `NFMD_OUTPUT_DIR`. Hydra saves the
configuration, Lightning saves checkpoints and CSV logs, and W&B logs the run.
Evaluation writes `evaluation/{val,test}/epoch_XXXX/metrics.json` and
`metrics.csv`. JSON contains all metric keys plus the seed, checkpoint epoch,
sampler settings, proposal temperature, proposal budget, and distributed layout.
CSV contains one metric/value row per per-peptide or aggregate metric; it can be
loaded without W&B. Undefined values appear as JSON null and empty CSV cells.

## Export to a new repository

From a committed release branch, run:

```bash
python scripts/export_release.py /path/to/empty/public-code
cd /path/to/empty/public-code
git init
```

The exporter copies the committed release tree without Git history. Use this
fresh directory for the new code repository. Paper analyses and plots remain in
the research repository and are excluded from the release tree.
