import os
import tempfile
from Main_App.settings_manager import SettingsManager


def test_run_in_pipeline_default():
    with tempfile.TemporaryDirectory() as tmp:
        ini = os.path.join(tmp, 'settings.ini')
        manager = SettingsManager(ini_path=ini)
        assert manager.get('loreta', 'run_in_pipeline', 'False').lower() == 'true'

