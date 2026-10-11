import io
import itertools
import tarfile

import numpy as np
import torch

from transferable_samplers.data.datasets.webdataset import build_webdataset


def test_streaming_resampling_repeats_with_seed_and_differs_by_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    urls = []
    for shard in range(3):
        filename = tmp_path / f"shard{shard}.tar"
        with tarfile.open(filename, "w") as archive:
            for index in range(8):
                values = np.full((2, 3), shard * 8 + index, dtype=np.float32).tobytes()
                info = tarfile.TarInfo(f"AL_{shard}_{index}.bin")
                info.size = len(values)
                archive.addfile(info, io.BytesIO(values))
        urls.append(str(filename))

    def draw(seed, rank):
        dataset = build_webdataset(urls, seed=seed, rank=rank, shuffle_buffer=7)
        return torch.stack([sample["x"] for sample in itertools.islice(dataset, 32)])

    first = draw(47, 0)
    torch.testing.assert_close(first, draw(47, 0))
    assert not torch.equal(first, draw(50, 0))
    assert not torch.equal(first, draw(47, 1))


def test_worker_streams_change_between_epochs_and_repeat_across_runs(tmp_path):
    import webdataset as wds

    urls = []
    for shard in range(3):
        filename = tmp_path / f"workers{shard}.tar"
        with tarfile.open(filename, "w") as archive:
            for index in range(8):
                values = np.full((2, 3), shard * 8 + index, dtype=np.float32).tobytes()
                info = tarfile.TarInfo(f"AL_{shard}_{index}.bin")
                info.size = len(values)
                archive.addfile(info, io.BytesIO(values))
        urls.append(str(filename))

    def run():
        torch.manual_seed(47)
        dataset = build_webdataset(urls, seed=47, rank=0, shuffle_buffer=7)
        loader = wds.WebLoader(dataset, batch_size=None, num_workers=2, timeout=20)
        return [torch.stack([sample["x"] for sample in itertools.islice(loader, 32)]) for _ in range(2)]

    first = run()
    second = run()
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    assert not torch.equal(first[0], first[1])
