import numpy as np
from scipy import stats
from datagenerator.Horizons import RandomHorizonStack
from datagenerator.Parameters import Parameters
import pytest


@pytest.fixture
def parameters(parameters_path: str = "../config/example.json") -> Parameters:
    parameters = Parameters(user_config=parameters_path)

    return parameters


def test_generate_lookup_tables(cfg: Parameters):
    cfg = {
        "num_lyr_lut": 10,
        "verbose": True,
        "include_channels": True,
        "cube_shape": (100, 100, 100),
        "dip_factor_max": 1.0
    }

    random_state = np.random.RandomState(0)
    stats.gamma.rvs = random_state.gamma
    
    thicknesses, onlaps, dips, azimuths, channels = \
        RandomHorizonStack.generate_lookup_tables(cfg)
    
    assert len(thicknesses) == cfg["num_lyr_lut"]
    assert len(onlaps) == int(500 / 1250 * cfg["cube_shape"][2])
    assert len(dips) == cfg["num_lyr_lut"]
    assert len(azimuths) == cfg["num_lyr_lut"]
    assert len(channels) == cfg["num_lyr_lut"]
    
    assert np.all(thicknesses >= 0)
    assert np.all(onlaps >= 0)
    assert np.all(dips >= 0)
    assert np.all(dips <= 7.0 * cfg["dip_factor_max"])
    assert np.all(azimuths >= 0.0)
    assert np.all(azimuths <= 360.0)
    assert np.all(channels >= 0)
    assert np.all(channels <= 1)


test_generate_lookup_tables(parameters)
