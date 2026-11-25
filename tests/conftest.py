import os

import pytest
import torch


def pytest_configure(config):
    dev = os.environ.get("NANOASK_TEST_DEVICE", "cpu")
    if dev.startswith("cuda") and not torch.cuda.is_available():
        pytest.exit(
            f"NANOASK_TEST_DEVICE={dev} but no CUDA GPU is available to PyTorch "
            f"(torch {torch.__version__}). On Colab: Runtime -> Change runtime type -> T4 GPU, "
            "then Runtime -> Disconnect and delete runtime, and run all cells again.",
            returncode=2,
        )
