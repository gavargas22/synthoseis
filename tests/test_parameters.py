import pytest
import json
import numpy as np
from pathlib import Path

from datagenerator.Parameters import Parameters
from synthoseis.storage import StorageClient

def test_parameters_storage_flow(tmp_path):
    config_path = Path(__file__).parent.parent / "config" / "test_config.json"
    p = Parameters(str(config_path), test_mode=32)
    p.project_folder = str(tmp_path / "project")
    p.work_folder = str(tmp_path)
    p.setup_model()
    assert hasattr(p, 'storage')
    assert isinstance(p.storage, StorageClient)
    test_shape = (10, 10, 10)
    test_arr = p.storage_init('test_dset', test_shape)
    assert test_arr.shape == test_shape
    retrieved = p.storage.get_dataset('test_dset')
    np.testing.assert_array_equal(retrieved, 0)
    p.storage.close()