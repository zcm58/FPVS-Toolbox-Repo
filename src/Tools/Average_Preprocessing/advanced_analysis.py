"""Aggregate advanced analysis logic mixins."""

from .advanced_analysis_base import AdvancedAnalysisBase
from .advanced_analysis_file_ops import AdvancedAnalysisFileOpsMixin
from .advanced_analysis_group_ops import AdvancedAnalysisGroupOpsMixin
from .advanced_analysis_processing import AdvancedAnalysisProcessingMixin
from .advanced_analysis_post import AdvancedAnalysisPostMixin


class AdvancedAnalysis(
    AdvancedAnalysisFileOpsMixin,
    AdvancedAnalysisGroupOpsMixin,
    AdvancedAnalysisProcessingMixin,
    AdvancedAnalysisPostMixin,
    AdvancedAnalysisBase,
):
    """Concrete class combining all advanced analysis logic."""

    pass

