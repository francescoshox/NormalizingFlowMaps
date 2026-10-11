import gc

import pytest
import torch


@pytest.fixture(autouse=True)
def free_memory_after_test():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
