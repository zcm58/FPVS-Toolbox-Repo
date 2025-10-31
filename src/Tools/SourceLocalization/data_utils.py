# src/Tools/SourceLocalization/data_utils.py
""""Utility functions for preparing data for source localization."""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import mne
from Main_App import SettingsManager

logger = logging.getLogger(__name__)


class MissingTemplateError(Exception):
    """Raised when the required MRI template is unavailable."""



def _load_data(fif_path: str) -> mne.Evoked:
    """Load epochs or evoked data and return an Evoked instance.

    Note
    ----
    This loader does not filter events. If the input is an Epochs file
    (-epo.fif), it averages **all** contained epochs as-is.
    """
    if fif_path.endswith("-epo.fif"):
        epochs = mne.read_epochs(fif_path, preload=True)
        return epochs.average()
    return mne.read_evokeds(fif_path, condition=0, baseline=(None, 0))


def _default_template_location() -> Path:
    """Return the directory where the template MRI should reside (legacy default)."""
    base_dir = Path(__file__).resolve().parents[2]
    return base_dir / "fsaverage"


def _wrap_fsaverage(subjects_dir: Path, subject: str = "fsaverage") -> None:
    """Ensure ``subjects_dir/subject`` contains a nested ``subject`` folder."""
    flat = subjects_dir / subject
    nested = flat / subject
    if flat.is_dir() and not nested.exists():
        nested.mkdir()
        for child in flat.iterdir():
            if child.name == subject:
                continue
            shutil.move(str(child), str(nested / child.name))


def _resolve_subjects_dir(path: Path | None, subject: str) -> Path:
    """Normalize ``path`` so that it points to the subjects directory."""
    if path is None:
        return _default_template_location().parent
    path = path.resolve()
    if (path / subject).is_dir():
        return path
    if len(path.parts) >= 2 and path.parts[-2:] == (subject, subject):
        return path.parent
    if path.name.lower() == subject.lower():
        return path.parent
    return path


def _dir_size_mb(path: str | Path) -> float:
    """Return the size of ``path`` in megabytes."""
    path = Path(path)
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def _threshold_stc(stc: mne.SourceEstimate, thr: float) -> mne.SourceEstimate:
    """Return a copy of ``stc`` with values below ``thr`` zeroed (abs-based).

    Semantics
    ---------
    - ``0 < thr < 1`` → fraction of **max |STC|** (e.g., 0.05 = 5% of max amp).
    - ``thr >= 1``    → absolute amplitude threshold in the STC unit.
    """
    stc = stc.copy()
    if 0 < thr < 1:
        thr_val = thr * np.max(np.abs(stc.data))
    else:
        thr_val = thr
    stc.data[np.abs(stc.data) < thr_val] = 0
    if SettingsManager().debug_enabled():
        active = int(np.count_nonzero(np.any(stc.data != 0, axis=1)))
        logger.debug(
            "threshold_stc",
            extra={"thr": thr, "cutoff": float(thr_val), "active_vertices": active},
        )
    return stc


def prepare_fsaverage(
    target_dir: Path, progress_cb: Optional[Callable[[int], None]] = None
) -> Path:
    """Download the ``fsaverage`` template into ``target_dir``.

    Parameters
    ----------
    target_dir
        Destination directory where the ``fsaverage`` dataset should be
        installed. Must be writable and outside of the application install
        directory.
    progress_cb
        Optional callable receiving integer progress units (approximate MB
        downloaded) during the transfer.

    Returns
    -------
    Path
        The path to the nested ``fsaverage`` directory (``<target>/fsaverage/fsaverage``).

    Raises
    ------
    ValueError
        If ``target_dir`` is not writable or resides within the application
        install directory.
    """

    resolved_target = target_dir.expanduser().resolve()
    source_file = Path(__file__).resolve()
    install_root = source_file.parents[3] if len(source_file.parents) >= 4 else source_file.parent
    if resolved_target == install_root or install_root in resolved_target.parents:
        raise ValueError("Target directory must be outside the application install tree.")

    resolved_target.mkdir(parents=True, exist_ok=True)

    probe = resolved_target / f".fpvs_write_test_{uuid.uuid4().hex}"
    try:
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok")
    except OSError as exc:  # pragma: no cover - depends on filesystem perms
        raise ValueError("Target directory is not writable.") from exc
    finally:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass

    dest = resolved_target / "fsaverage"
    nested = dest / "fsaverage"
    if (nested / "surf" / "lh.white").exists() and (nested / "bem" / "fsaverage-5120-bem.fif").exists():
        _wrap_fsaverage(resolved_target, subject="fsaverage")
        return nested.resolve()

    stop_event = threading.Event()

    def _report_progress() -> None:
        last = -1
        while not stop_event.is_set():
            if dest.is_dir():
                size = int(_dir_size_mb(dest))
                if size != last and progress_cb is not None:
                    progress_cb(size)
                last = size
            time.sleep(1)

    thread = threading.Thread(target=_report_progress, daemon=True)
    thread.start()
    try:
        path = Path(
            mne.datasets.fetch_fsaverage(subjects_dir=str(resolved_target), verbose=True)
        )
    finally:
        stop_event.set()
        thread.join()

    if progress_cb is not None:
        progress_cb(int(_dir_size_mb(path)))

    _wrap_fsaverage(resolved_target, subject="fsaverage")
    final_path = path.resolve()
    logger.info("prepare_fsaverage complete", extra={"op": "prepare_fsaverage", "path": str(final_path)})
    return final_path


def _estimate_epochs_covariance(
    epochs: mne.Epochs,
    log_func: Callable[[str], None] = logger.info,
    baseline: Optional[Tuple[float | None, float | None]] = None,
) -> mne.Covariance:
    """Return a noise covariance estimated from ``epochs``.

    Uses pre-stim baseline when provided; otherwise baseline ends at 0 s.
    Falls back to ad-hoc when there is only a single epoch.
    """
    if len(epochs) > 1:
        if baseline is not None:
            tmin, tmax = baseline
            return mne.compute_covariance(epochs, tmin=tmin, tmax=tmax)
        return mne.compute_covariance(epochs, tmax=0.0)
    log_func("Only one epoch available. Using ad-hoc covariance.")
    return mne.make_ad_hoc_cov(epochs.info)


def fetch_fsaverage_with_progress(
    subjects_dir: str | Path, log_func: Callable[[str], None] = logger.info
) -> str:
    """Return the path to ``fsaverage`` if present or raise :class:`MissingTemplateError`."""

    subjects_dir = Path(subjects_dir).resolve()
    if subjects_dir.name.lower() == "fsaverage" and (subjects_dir / "surf").exists():
        dest = subjects_dir.parent
    else:
        dest = subjects_dir / "fsaverage"
    nested = dest / "fsaverage"

    if (nested / "surf" / "lh.white").exists() and (
        nested / "bem" / "fsaverage-5120-bem.fif"
    ).exists():
        _wrap_fsaverage(dest, subject="fsaverage")
        log_func(f"Using existing fsaverage template at {nested}")
        return str(nested)

    raise MissingTemplateError(
        f"fsaverage template not found under '{nested}'. "
        "Download it using prepare_fsaverage() before continuing."
    )


def _project_cache_dir(settings: SettingsManager, fallback_base: Path) -> Path:
    """Choose a cache directory under the active project when available.

    Priority:
    1) settings.get('loreta','cache_dir')
    2) settings.get('paths','project_root') / '.cache' / 'loreta'
    3) fallback_base / 'fpvs_cache'  (legacy behavior)
    """
    # 1) explicit cache dir
    try:
        explicit = settings.get("loreta", "cache_dir", "")
        if explicit:
            p = Path(explicit).resolve()
            p.mkdir(parents=True, exist_ok=True)
            return p
    except Exception:
        pass

    # 2) project root
    try:
        proj_root = settings.get("paths", "project_root", "")
        if proj_root:
            p = Path(proj_root).resolve() / ".cache" / "loreta"
            p.mkdir(parents=True, exist_ok=True)
            return p
    except Exception:
        pass

    # 3) legacy: next to subjects_dir
    legacy = fallback_base / "fpvs_cache"
    legacy.mkdir(exist_ok=True)
    return legacy


def _prepare_forward(
    evoked: mne.Evoked,
    settings: SettingsManager,
    log_func: Callable[[str], None],
) -> tuple[mne.Forward, str, str]:
    """Construct a forward model using MRI info from settings or fsaverage.

    Returns
    -------
    fwd : mne.Forward
    subject : str
    subjects_dir : str
    """
    stored_dir = settings.get("loreta", "mri_path", "")
    stored_dir_path: Path | None = Path(stored_dir).resolve() if stored_dir else None
    subject = "fsaverage"

    log_func(f"Initial stored MRI directory: {stored_dir_path or stored_dir}")

    if stored_dir_path is None or not stored_dir_path.is_dir():
        raise MissingTemplateError(
            "The fsaverage template is not configured. Download it using prepare_fsaverage() "
            "and update the Source Localization settings."
        )

    log_func(f"Using existing MRI directory: {stored_dir_path}")
    subjects_dir_path = _resolve_subjects_dir(stored_dir_path, subject)

    trans = settings.get("paths", "trans_file", "fsaverage")
    log_func(f"Using trans file: {trans}")

    def _cache_path(current_subjects_dir: Path) -> Path:
        cache_base = _project_cache_dir(settings, current_subjects_dir)
        return cache_base / f"forward-{subject}.fif"

    fwd_file = _cache_path(subjects_dir_path)

    if fwd_file.is_file():
        log_func(f"Loading cached forward model from {fwd_file}")
        fwd = mne.read_forward_solution(str(fwd_file))
        return fwd, subject, str(subjects_dir_path)

    try:
        fs_path = Path(fetch_fsaverage_with_progress(stored_dir_path, log_func))
    except MissingTemplateError as exc:
        raise MissingTemplateError(str(exc)) from exc
    stored_dir_path = fs_path
    settings.set("loreta", "mri_path", str(stored_dir_path))
    try:
        settings.save()
    except Exception:
        pass

    subjects_dir_path = _resolve_subjects_dir(stored_dir_path, subject)
    log_func(f"Subjects directory resolved to: {subjects_dir_path}")

    fwd_file = _cache_path(subjects_dir_path)

    log_func(
        f"Building source space using subjects_dir={subjects_dir_path}, subject={subject}"
    )
    src = mne.setup_source_space(
        subject, spacing="oct6", subjects_dir=str(subjects_dir_path), add_dist=False
    )

    log_func("Creating BEM model …")
    model = mne.make_bem_model(subject=subject, subjects_dir=str(subjects_dir_path), ico=4)
    bem = mne.make_bem_solution(model)

    log_func("Computing forward solution …")
    fwd = mne.make_forward_solution(
        evoked.info, trans=trans, src=src, bem=bem, eeg=True
    )
    mne.write_forward_solution(str(fwd_file), fwd, overwrite=True)
    return fwd, subject, str(subjects_dir_path)
