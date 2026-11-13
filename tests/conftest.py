import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def data():
    """The full FEA dataset with all physics features added."""
    from wpt_physics import add_physics_features
    return add_physics_features(pd.read_csv(ROOT / "data" / "raw" / "ANNCoilsFEA.csv"))
