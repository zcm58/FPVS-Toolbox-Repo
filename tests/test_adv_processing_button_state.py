import importlib.util
import os
import sys
import types


def _import_module(monkeypatch):
    dummy_core = types.SimpleNamespace(QObject=object, QThread=object, Signal=lambda *a, **k: object())
    dummy_widgets = types.SimpleNamespace(QMessageBox=types.SimpleNamespace(critical=lambda *a, **k: None))
    dummy_pyside = types.SimpleNamespace(QtCore=dummy_core, QtWidgets=dummy_widgets)
    monkeypatch.setitem(sys.modules, "PySide6", dummy_pyside)
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", dummy_core)
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", dummy_widgets)

    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "Tools",
        "Average_Preprocessing",
        "advanced_analysis_qt_processing.py",
    )

    # Create minimal package structure so relative imports work
    if "Tools" not in sys.modules:
        sys.modules["Tools"] = types.ModuleType("Tools")
    if "Tools.Average_Preprocessing" not in sys.modules:
        pkg = types.ModuleType("Tools.Average_Preprocessing")
        pkg.__path__ = []
        sys.modules["Tools.Average_Preprocessing"] = pkg

    if "Tools.Average_Preprocessing.advanced_analysis_core" not in sys.modules:
        core_mod = types.ModuleType("Tools.Average_Preprocessing.advanced_analysis_core")
        core_mod.run_advanced_averaging_processing = lambda *a, **k: None
        sys.modules["Tools.Average_Preprocessing.advanced_analysis_core"] = core_mod

    if "Main_App.post_process" not in sys.modules:
        pp_mod = types.ModuleType("Main_App.post_process")
        pp_mod.post_process = lambda *a, **k: None
        sys.modules["Main_App.post_process"] = pp_mod

    spec = importlib.util.spec_from_file_location(
        "Tools.Average_Preprocessing.adv_qt_proc",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




class DummyButton:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, val):
        self.enabled = bool(val)


class DummyVar:
    def __init__(self, value=""):
        self._value = value

    def get(self):
        return self._value

    def set(self, v):
        self._value = v


class DummyMaster:
    def __init__(self):
        self.validated_params = None
        self.save_folder_path = DummyVar()


def test_start_processing_button_state(monkeypatch):
    module = _import_module(monkeypatch)
    AdvancedAnalysisProcessingMixin = module.AdvancedAnalysisProcessingMixin

    class DummyWin(AdvancedAnalysisProcessingMixin):
        def __init__(self):
            self.master_app = DummyMaster()
            self.defined_groups = []
            self.start_btn = DummyButton()

    win = DummyWin()

    # No params or output dir
    win._update_start_processing_button_state()
    assert win.start_btn.enabled is False

    # Groups and output dir, but no params
    win.defined_groups = [
        {"config_saved": True, "file_paths": ["a"], "condition_mappings": [1]}
    ]
    win.master_app.save_folder_path.set("/tmp")
    win._update_start_processing_button_state()
    assert win.start_btn.enabled is False

    # Params but missing output dir
    win.master_app.validated_params = {"a": 1}
    win.master_app.save_folder_path.set("")
    win._update_start_processing_button_state()
    assert win.start_btn.enabled is False

    # All requirements met
    win.master_app.save_folder_path.set("/tmp")
    win._update_start_processing_button_state()
    assert win.start_btn.enabled is True
