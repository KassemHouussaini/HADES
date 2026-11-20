"""Shared fixtures. system_builder imports bw2data at module level; the pure
functions tested here never touch Brightway, so a mock module is enough."""
import importlib.util, os, sys, types
import pytest

ADM1_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ADM1_DIR)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ADM1_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def sb():
    if "bw2data" not in sys.modules:
        bd = types.ModuleType("bw2data")
        bd.databases = {}
        bd.Database = lambda name: []
        sys.modules["bw2data"] = bd
    return _load("p1_system_builder_test", "system_builder.py")


@pytest.fixture(scope="session")
def hce():
    return _load("p1_hce_test", "hydrochar_enduse.py")


@pytest.fixture(scope="session")
def ac():
    return _load("p1_ac_test", "ad_characterization.py")


@pytest.fixture(scope="session")
def adm1():
    return _load("p1_adm1_test", "ADM1.py")
