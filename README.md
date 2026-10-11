# Normalizing Flow Maps

Minimal JAX implementation of the image experiments in
[**Normalizing Flow Maps**](https://openreview.net/pdf?id=guBm2gNlxU).
The models combine flow-map training with autoregressive normalizing flows.
This implementation branches from the public [BiFlow](https://github.com/Lyy-iiis/BiFlow)
codebase; upstream licenses and attribution are retained in [NOTICE](NOTICE).

| Branch | Contents |
| --- | --- |
| `main` | JAX: CIFAR-10 NFMD, ImageNet-64 NFMD and two-stage ImageNet-64 NFMR |
| `boltzmann-generators` | PyTorch Boltzmann-generator release, with its original README |
| `reflow-couplings` | Empty branch reserved for coupling-generation code |

NFM-D and NFM-R in the paper are named **NFMD** and **NFMR** in this code.
NFMR contains both **flow matching** and **mean flow**. CIFAR-10 NFMR is outside
this release's scope.

## Install

Use Python 3.12. With `uv`, the included lockfile pins the dependencies:

```bash
uv sync --frozen --python 3.12 --extra cuda --extra eval --extra nfmr
source .venv/bin/activate
```

Alternatively, use `pip` in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[cuda]'
# CPU PyTorch is used for the reference FID resize, not image-model training.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[eval]'
# Also install these for the NFMR perceptual losses:
pip install -e '.[nfmr]'
```

For CPU development, replace `.[cuda]` with `.[dev]`. All image models and
training objectives use JAX/Flax. Training defaults to BF16 computation with
FP32 model weights, residuals and likelihood reductions. The release uses
portable XLA attention and native JAX directional derivatives; the original
cluster-specific FA4 kernels are not bundled. This preserves the architecture
and objectives, but is not a bit-for-bit replay of those kernels and may require
more accelerator memory. Run-specific provenance and outstanding checks are
listed in [docs/provenance.md](docs/provenance.md).

## Data

Prepare the training pixels:

```bash
python -m nfm.prepare cifar10 --output data/cifar10/train
python -m nfm.prepare imagenet64 --input /path/to/imagenet/train --output data/imagenet64/train
```

ImageNet must be obtained separately and arranged as `train/<class>/*`.
Preparation uses the original progressive BOX/BICUBIC center crop and writes
an alphabetical class mapping. Existing preprocessed pixel safetensors files
can be used directly.

A dataset is either a safetensors file or a directory of memory-mapped NumPy
arrays:

| Key/file | Shape and type | Meaning |
| --- | --- | --- |
| `x1` / `x1.npy` | `[N,H,W,3]`, uint8 | RGB image |
| `label` / `label.npy` | `[N]`, integer | Class ID |
| `x0` / `x0.npy` | `[N,H,W,3]`, floating point | Paired Gaussian source, required for NFMR |

NFMR uses the existing EDM2 coupling pool, with 1,280,512 globally shuffled
examples per epoch. It requires stored paired `x0`; Gaussian sources are not
silently regenerated. **TODO:** Add the public coupling download or exact
generation instructions once the collaborator's assets are available.

Compute a compatible FID reference from the real training pixels:

```bash
python -m nfm.prepare fid-reference --data data/cifar10/train \
  --image-size 32 --output data/cifar10_fid.npz
python -m nfm.prepare fid-reference --data data/imagenet64/train \
  --image-size 64 --output data/imagenet64_fid.npz
```

The default uses every real image, the inherited Inception FID weights, and the
original `grid_sample` resize. For comparing to a historical run, supply its
original reference statistics instead; preprocessing and reference selection
affect FID. `--weights /path/to/inception_v3_weights_fid.pickle` supports an
offline weights file. Otherwise the weights are downloaded into
`JAX_FID_CACHE_DIR` (default: the system temporary directory). Set
`JAX_FID_OFFLINE=1` to require already cached weights.

## One script per run

Each launcher trains, resumes automatically from its latest complete
checkpoint, and evaluates the final checkpoint with 50,000 samples:

```bash
bash scripts/cifar10_nfmd.sh data/cifar10/train data/cifar10_fid.npz
bash scripts/imagenet64_nfmd.sh data/imagenet64/train data/imagenet64_fid.npz
bash scripts/imagenet64_nfmr.sh /path/to/couplings.safetensors data/imagenet64_fid.npz
```

An optional third argument sets the output directory. Keep large datasets and
training outputs on storage suitable for the experiment. Launchers resolve
their configs from the repository root. Relative data paths are therefore
relative to that root. Set `NFM_PYTHON` to select a Python executable.

| Recipe | Model | Global batch | Epochs | Learning rate | EMAs | Final evaluation |
| --- | --- | --- | --- | --- | --- | --- |
| CIFAR-10 NFMD | 256 channels, 8 blocks, 6 layers, patch 2; `s_max=0.10` | 256 | 960 | `4e-4`, 1,954-step warmup from `1e-6` | `0.9995`, `0.9999` | CFG 0 and 2.8 |
| ImageNet-64 NFMD | B/2; `s_max=0.15` | 256 | 300 | 25,000-step warmup to `4e-4`; cosine from epoch 150 to 200 down to `1e-4`; then constant | `0.9995`, `0.9999` | **CFG 3.3 only** |
| ImageNet-64 NFMR: flow matching | B/2 | **512** | 200 | `4e-4`, constant | `0.999`, `0.9995`, `0.9999` | Optional 50-step Euler evaluation |
| ImageNet-64 NFMR: mean flow | B/2, initialized from flow-matching EMA | **512** | 25 | `1e-4`, constant | `0.999`, `0.9995`, `0.9999` | One-step generation, CFG 4.5 on interval `[0.5,0.9]` (provisional) |

All recipes use adjusted Muon: momentum 0.95, Nesterov updates, five
Newton–Schulz iterations and matrix weight decay 0.1. Embeddings use AdamW
with decay `1e-4`; scalar/vector parameters use AdamW without decay. AdamW
betas are `(0.9,0.95)`. NFMD uses class dropout 0.1 and the single-time
distributional objective; there is no denoising step in its sampler.

NFMR's one launcher runs `flow_matching` first and then `mean_flow`, each in
its own output directory. The handoff selects EMA `0.999` provisionally,
resets the optimizer, step count and destination EMAs, and starts the
25-epoch schedule. A destination recovery checkpoint takes precedence over
fresh initialization. The remaining NFMR details are explicit **TODOs** in
[the provenance notes](docs/provenance.md).

The train CLI supports `--max-steps` for short probes. A probe checkpoint
records `complete: false`; it cannot be mistaken for a finished paper run or
used automatically as a completed NFMR first stage. Checkpoints contain the
optimizer, online weights, all EMAs, step, seed, config and data identity.
Restarting uses the same shuffled epoch and resumes at the next global batch.
`SIGTERM`/`SIGUSR1` or a positive `NFM_MAX_SECONDS` request a synchronized
checkpoint and exit status 75 when training is unfinished. Set the time budget
below the scheduler's limit to allow checkpoint writing, then rerun the same
launcher to resume. An unfinished stage does not advance or evaluate.

## Sampling and evaluation

Use the checkpoint directory reported in `<workdir>/latest.json`:

```bash
python -m nfm.evaluate --config configs/imagenet64_nfmd.yaml \
  --checkpoint /path/to/checkpoint --reference data/imagenet64_fid.npz \
  --output runs/imagenet64_nfmd/evaluation
python -m nfm.evaluate --config configs/imagenet64_nfmd.yaml \
  --checkpoint /path/to/checkpoint --sample-only --samples 64 \
  --ema 0.9999 --output runs/imagenet64_nfmd/samples
python -m nfm.evaluate --config configs/cifar10_nfmd.yaml \
  --checkpoint /path/to/checkpoint --likelihood-data /path/to/cifar10/test \
  --samples 10000 --output runs/cifar10_nfmd/likelihood
```

FID records include the reference and Inception-weight SHA256 hashes.
Completed evaluations with identical settings are skipped on restart.
Evaluation runs on one process/device, independently of distributed training.
Likelihood uses uniform dequantization and unguided conditional density.
NFMR additionally requires a sequential inverse, which can be numerically
unstable for the paper's non-invertible-in-finite-precision recipe; its model
module exposes round-trip diagnostics.

The best completed epoch-300 ImageNet NFMD evaluation reported for the source
run was **FID 4.5947366**, EMA **0.9999**, CFG **3.3**. This is historical evidence,
not a new measurement of this extracted release. No guidance sweep is included.

## Distributed training and logging

The batch sizes above are global, across all devices and processes. For
multiple hosts, set `JAX_COORDINATOR_ADDRESS`, `JAX_PROCESS_COUNT`,
`JAX_PROCESS_INDEX`, and optionally `JAX_LOCAL_DEVICE_IDS`. Slurm allocations
can use JAX's automatic discovery or `MASTER_ADDR`/`MASTER_PORT` together with
`SLURM_NTASKS`/`SLURM_PROCID`. Every rank must see the same dataset and output
directory and invoke the same launcher. Final evaluation runs only on rank 0.

Local JSONL logging is always enabled. W&B is optional:

```bash
pip install -e '.[wandb]'
WANDB_PROJECT=your-project WANDB_ENTITY=your-entity bash scripts/cifar10_nfmd.sh DATA REFERENCE
```

No entity, personal project, cluster path or credential is embedded in the
release. W&B uses neutral recipe names. Authentication comes from your own
environment or W&B login.

## Development

```bash
pip install -e '.[dev]'
JAX_PLATFORMS=cpu pytest -q
ruff check .
```

Tests exercise nontrivial inverses and Jacobian determinants, the NFMR
directional derivative in FP32/BF16, EMA updates, deterministic resume,
coupling sharding, stage handoff and streaming FID statistics. Full training,
multi-host accelerator performance and the paper's FIDs require the datasets
and accelerator runs.

## License

Original NFM contributions are licensed under MIT; see [LICENSE](LICENSE).
This JAX implementation derives from [BiFlow](https://github.com/Lyy-iiis/BiFlow).
Inherited code retains its upstream copyrights and license terms, including
the Apache 2.0 and BSD terms described in [NOTICE](NOTICE).
The `boltzmann-generators` branch derives from
[Transferable Samplers](https://github.com/transferable-samplers/transferable-samplers)
and includes additional third-party terms documented in that branch's NOTICE.
