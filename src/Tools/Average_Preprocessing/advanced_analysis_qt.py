"""Aggregate Qt Advanced Analysis components into a single class."""

from .advanced_analysis_qt_base import AdvancedAnalysisWindowBase
from .advanced_analysis_qt_file_ops import AdvancedAnalysisFileOpsMixin
from .advanced_analysis_qt_group_ops import AdvancedAnalysisGroupOpsMixin
from .advanced_analysis_qt_processing import AdvancedAnalysisProcessingMixin
from .advanced_analysis_qt_post import AdvancedAnalysisPostMixin


class AdvancedAnalysisWindow(
    AdvancedAnalysisFileOpsMixin,
    AdvancedAnalysisGroupOpsMixin,
    AdvancedAnalysisProcessingMixin,
    AdvancedAnalysisPostMixin,
    AdvancedAnalysisWindowBase,
):
    """Combined Qt-based window class composed from mixins."""

    pass
