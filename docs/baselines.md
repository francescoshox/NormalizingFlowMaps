# Baselines

NFMD reproduction uses `scripts/nfmd/`. Standard MLE (TARFlow, PROSE) and flow
matching (ECNF/ECNF++) recipes are retained separately under
`configs/experiment/baselines/`.

The GYDPETGTWG TARFlow MLE launcher uses the standard baseline recipe, with
training seed 47, W&B logging, and numerical evaluation exports. Append Hydra
`key=value` overrides to change settings. Other baselines use the Python entry points:

```bash
bash scripts/baselines/train_mle_GYDPETGTWG.sh
python -m transferable_samplers.train \
  experiment=baselines/transferable/train/prose_up_to_8aa seed=47
python -m transferable_samplers.eval \
  experiment=baselines/transferable/eval/prose_up_to_8aa_snis \
  ckpt_path=/path/to/prose.ckpt hf_state_dict_path=null seed=50
```

Use the same W&B and data setup as the [molecular guide](bgs.md). These baseline
architectures and schedules are separate from the NFMD recipes. The release
keeps the network code, datamodules, normalizing-flow/flow-matching modules,
optimizers, numerical evaluators, and sampling algorithms they need.
