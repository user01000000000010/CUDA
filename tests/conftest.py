"""Shared pytest fixtures.

Tests are skipped (not failed) when:
  * no CUDA device is available (e.g. on macOS), or
  * the extension is not built, or
  * the specific kernel was not included in KERNELS=... at build time.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")


@pytest.fixture(scope="session")
def ext():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine")
    try:
        import fused_online_softmax_cuda as module
    except ImportError as e:
        pytest.skip(f"Extension not built: {e}")
    return module


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
