# Compatibility shim: route legacy imports to the new PySide6 SettingsManager
from Main_App.PySide6_App.Backend.settings_manager import SettingsManager

__all__ = ["SettingsManager"]
