"""The base Anaconda env must stay exactly as it was.

Not a test of this project's code -- a test of a hazard this project creates.
`N:\\ML\\Anaconda\\Install` pins numpy 1.24.3 for numba 0.57.1, and an unrelated
biophysics solver runs against it as a live service with a job queue. Installing
torch into that env would very likely upgrade numpy and break a running system,
which is why this project's env lives at its own prefix in `.venv/`.

The check is cheap and the failure it guards against is expensive and delayed:
numba would keep importing and only fail at JIT time, hours into a solve. Skips
cleanly if the base interpreter is absent, so the suite still runs elsewhere.
"""
import os
import subprocess

import pytest

BASE_PYTHON = r"N:\ML\Anaconda\Install\python.exe"
EXPECTED = {"numba": "0.57.1", "numpy": "1.24.3"}


@pytest.mark.skipif(not os.path.exists(BASE_PYTHON),
                    reason="base Anaconda interpreter not present")
def test_base_env_pins_are_intact():
    out = subprocess.run(
        [BASE_PYTHON, "-c",
         "import numba, numpy; print(numba.__version__, numpy.__version__)"],
        capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, (
        f"the base env no longer imports numba/numpy -- this project may have "
        f"contaminated it:\n{out.stderr}")
    numba_v, numpy_v = out.stdout.split()
    assert numba_v == EXPECTED["numba"], (
        f"numba moved from {EXPECTED['numba']} to {numba_v} in the base env")
    assert numpy_v == EXPECTED["numpy"], (
        f"numpy moved from {EXPECTED['numpy']} to {numpy_v} in the base env; "
        f"numba 0.57.1 requires numpy < 1.25 and the solver will fail at JIT time")


def test_this_project_is_not_running_from_the_base_env():
    """Guard against someone running the suite with the wrong interpreter."""
    import sys
    assert "Anaconda" not in sys.executable or ".venv" in sys.executable, (
        f"tests are running under {sys.executable}; use .venv\\python.exe so "
        f"the base env is never asked to import torch")
