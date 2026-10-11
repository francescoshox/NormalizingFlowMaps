"""Exercise the existing rank RNG handling in a two-process Gloo group."""

import json
import sys
from pathlib import Path

import lightning
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transferable_samplers.utils.dist_utils import all_gather_cat, drift_rng_state


def worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    streams = []
    same_initial_weights = True
    for _ in range(2):
        lightning.seed_everything(47, workers=True)
        weights = torch.randn(1, 12)
        gathered = all_gather_cat(weights)
        same_initial_weights = same_initial_weights and torch.equal(gathered[0], gathered[1])
        drift_rng_state()
        streams.append(all_gather_cat(torch.randn(1, 12)))
    if rank == 0:
        Path(output).write_text(
            json.dumps(
                {
                    "same_initial_weights": same_initial_weights,
                    "repeatable": torch.equal(streams[0], streams[1]),
                    "different_rank_streams": not torch.equal(streams[0][0], streams[0][1]),
                }
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    output = str(Path(sys.argv[1]).resolve())
    mp.spawn(worker, args=(output + ".rendezvous", output), nprocs=2, join=True)
