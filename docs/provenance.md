# Source branches and remaining checks

The public implementation ancestry is [BiFlow](https://github.com/Lyy-iiis/BiFlow).
The branch names below identify the experiment snapshots used for extraction;
they are not dependencies of the release.

| Release component | Source branch / checkout | Source commit | What is retained or changed |
| --- | --- | --- | --- |
| NFMD architecture, single-time objective, adjusted Muon and ImageNet cooldown | `jupiter/fkl-cooldown150-to300` | `3a547c5b97461b4c8f5c116955c0fb81ff3129a6` | Time-conditioned model and objective retained; renamed NFMD; fresh warmup + 150–200 cooldown implemented explicitly |
| CIFAR JAX recipe cross-check | `cifar-jupiter` | `e16a992247c0e42b499c553f662eff114ac0c6a6` | Small model shape and pixel pipeline checked; final NFMD hyperparameters follow the PyTorch paper recipe |
| CIFAR PyTorch paper settings | `nf-flow-map`: `francesco/fkl-images`, plus the supplied `c10_smax010_paper` run | `1b1d1401d244189864271ab630b15b7e139f1758` | Batch 256, `s_max=0.10`, 960 epochs, LR `4e-4`, warmup 1,954, adjusted Muon; only EMAs 0.9995/0.9999 retained |
| NFMR model, flow-matching step, mean-flow step and perceptual losses | `mn/fm-pretrain-probes` | `e2a8d49bb4d921eb9d2c2e72963c38a4e150f1c1` | Both stages under NFMR; durations/LRs follow paper; batch 512 in both stages confirmed by the user |
| Historical NFMR probe evidence | MareNostrum jobs `45852649` / `45880802` | `bef8f1c` / `463f456` | Completed 100-update probes at batches 512 / 256; not completed paper training, no checkpoints/evaluation |
| Historical ImageNet NFMD best completed evaluation | User-provided epoch-300 result | Checkpoint `1501200` | EMA 0.9999, CFG 3.3, FID 4.5947366; release evaluates only CFG 3.3 |
| `boltzmann-generators` branch | `flows/nfmd-release`, branch `release/nfmd` | `74b962f3f0da36704ade20643070d0d6d99a50c7` | Code and file modes preserved; README, LICENSE and NOTICE updated to credit Transferable Samplers and identify MIT-licensed NFM contributions while retaining upstream terms |
| `reflow-couplings` branch | New empty tree | New release commit | No tracked files |

The evaluation-sweep branch `jupiter/fkl-from150-e300-wide-sweep` (`6cbbf58`)
is **not used**. Fixed evaluation at CFG 3.3 satisfies the requested release.
Historical CFG 3.8 scored 4.7324512 at EMA 0.9999. Four planned historical
evaluations were missing (CFG 3.0/4.1 at both EMAs), so CFG 3.3 is described
as the best completed result, without claiming an exhaustive optimum.

## Confirmed release choices

- CIFAR NFMD: single-time `s_max=0.10`, 960 epochs, global batch 256,
  adjusted Muon, EMAs 0.9995 and 0.9999. CIFAR NFMR omitted.
- ImageNet NFMD: B/2, `s_max=0.15`, 300 epochs, batch 256; fresh warmup to
  `4e-4`, cosine decay from epochs 150 to 200 down to `1e-4`, then hold.
  Two EMAs, evaluated at CFG 3.3 only.
- ImageNet NFMR: B/2; **global batch 512 in both stages**; 200 epochs of
  flow matching at `4e-4`, then 25 epochs of mean flow at `1e-4`.
- BF16 computation retained. Portable XLA attention replaces the unbundled
  cluster FA4 implementation; NFMR uses native JAX JVP through that attention.
- W&B optional, with account/project supplied by the user at runtime.

## Checks still needed from the other clusters / collaborator

**TODO:** Retrieve the collaborator's full NFMR source revision and resolved
configs. A W&B export references `727725a3121e2b72bbea59ab0ca3d729374f6bca`,
but that Git object was unavailable and the exact associated completed run
could not be verified. The 25-epoch mean-flow duration is accepted from the
paper and the user's confirmation; no completed paper run is inferred from
the short probes.

**TODO:** Resolve the flow-matching LR discrepancy in historical metadata:
available branch/run settings use `0.000566`, whereas the paper specifies
`4e-4`. This release uses **`4e-4` at the confirmed batch 512**. Confirm the
collaborator's actual paper checkpoint settings and update the provenance
if the paper description or recorded recipe needs correction.

**TODO:** Confirm the flow-matching EMA chosen for the handoff. `0.999` is
the current explicit provisional default. The destination optimizer, step
and EMAs reset at the handoff; a recovery checkpoint preserves all of them.

**TODO:** Confirm NFMR affine clip limit `4.0`, denoiser denominator floor
`0.05`, clipped velocity target, time sampling, adaptive loss reweighting,
LPIPS weight `0.4`, ConvNeXt weight `0.1`, and the final evaluation guidance
`4.5` with interval `[0.5,0.9]`. These match available recorded settings,
but cannot establish the unavailable collaborator revision.

**TODO:** Publish or identify the precise EDM2 coupling pool and its license,
class mapping, checksums and generation procedure. Available metadata describes
a 3.9-million-pair pool and 1,280,512 examples per shuffled epoch. The current
implementation supports that format and epoch size directly. The reserved
`reflow-couplings` branch remains empty as requested.

**TODO:** Supply the exact FID reference file/hash and released model
checkpoints if the release should reproduce the historical measured FIDs
without retraining. Freshly computed reference statistics are supported,
but are not asserted to be identical to the historical reference.

**TODO:** Validate full accelerator training and multi-host launch/resume
with the portable attention implementation. CPU tests check the mathematical
and checkpoint invariants; no full paper experiment has been rerun here.
