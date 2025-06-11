"""Python interface for the fast C++ processing extension."""

try:
    from .downsample_filter import downsample, apply_fir_filter
    EXTENSION_AVAILABLE = True
except Exception:  # pragma: no cover - extension may not be built
    downsample = None
    apply_fir_filter = None
    EXTENSION_AVAILABLE = False

__all__ = ["downsample", "apply_fir_filter", "EXTENSION_AVAILABLE"]
