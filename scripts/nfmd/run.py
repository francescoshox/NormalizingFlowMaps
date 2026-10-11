"""Portable NFMD entry point; one torchrun launcher per node."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]


def build_command(args, overrides):
    if args.nodes < 1 or args.devices < 1:
        raise ValueError("--nodes and --devices must be positive")
    if not 0 <= args.node_rank < args.nodes:
        raise ValueError("--node-rank must be between 0 and --nodes minus one")
    if args.nodes > 1 and (not args.run_id or not args.master_addr):
        raise ValueError("Multinode runs require the same --run-id and --master-addr on every node")
    if args.mode == "eval" and not args.checkpoint:
        raise ValueError("Evaluation requires --checkpoint PATH")
    if args.mode == "train" and args.checkpoint:
        raise ValueError("Use --checkpoint with evaluation")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("--run-id may contain letters, digits, underscores, dots and dashes")
    command = [sys.executable]
    if args.nodes * args.devices > 1:
        command += [
            "-m",
            "torch.distributed.run",
            "--nnodes",
            str(args.nodes),
            "--nproc_per_node",
            str(args.devices),
            "--node_rank",
            str(args.node_rank),
            "--master_addr",
            args.master_addr or "127.0.0.1",
            "--master_port",
            str(args.master_port),
        ]
    command += [
        "-m",
        "transferable_samplers." + args.mode,
        f"experiment=nfmd/{args.system}_{args.mode}",
        f"trainer.devices={args.devices}",
        f"trainer.num_nodes={args.nodes}",
        "hydra.run.dir=${paths.log_dir}/${task_name}/" + run_id,
    ]
    if args.seed is not None:
        command += [f"seed={args.seed}"]
    if args.mode == "eval":
        command += ["ckpt_path=" + json.dumps(str(Path(args.checkpoint).expanduser().resolve()))]
    if args.sampler != "snis":
        command += [f"sampler={args.system}_smc" if args.sampler == "smc" else "sampler=transferable_mala"]
    if args.sampler == "mala" and args.system != "transferable":
        raise ValueError("The MALA recipe is provided for transferable evaluation")
    command += overrides
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("system", choices=["GYDPETGTWG", "transferable"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int, help="Default: training 47, evaluation 50")
    parser.add_argument("--sampler", choices=["snis", "smc", "mala"], default="snis")
    parser.add_argument("--devices", type=int, default=1, help="GPUs per node")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--run-id", help="Use an identical, unique identifier on all nodes")
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    if any("=" not in item or item.startswith("--") for item in overrides):
        parser.error("Additional arguments must be Hydra key=value overrides")
    try:
        command = build_command(args, overrides)
    except ValueError as error:
        parser.error(str(error))
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
