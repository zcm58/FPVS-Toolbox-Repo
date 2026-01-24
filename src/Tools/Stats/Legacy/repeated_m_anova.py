# repeated_m_anova.py
# -*- coding: utf-8 -*-
"""
Within-subject (repeated measures) ANOVA on long-format data.

Exports
-------
run_repeated_measures_anova(data, dv_col, within_cols, subject_col) -> pandas.DataFrame

Behavior
--------
- Prefers Pingouin's rm_anova (if installed) to obtain detailed RM-ANOVA
  results (and, when available, GG/HF corrected p-values).
- Falls back to statsmodels.AnovaRM if Pingouin is not available.
- Always returns a tidy DataFrame with columns:
    'Effect', 'F Value', 'Num DF', 'Den DF', 'Pr > F', 'partial eta squared'
  and, when available from Pingouin:
    'Pr > F (GG)', 'Pr > F (HF)'

Notes
-----
- This function requires a *balanced* within-subject design: each subject must
  have exactly one observation per combination of the within factors.
- For inference in the presence of potential sphericity violations and/or
  missing data, prefer a Linear Mixed-Effects model; this RM-ANOVA can be
  used as a complementary, descriptive analysis.

Dependencies
------------
- pandas (required)
- numpy (required)
- pingouin (optional; if present, used for the primary path)
- statsmodels (required for fallback path)
"""

from __future__ import annotations

from itertools import product
from typing import Optional, List

import numpy as np
import pandas as pd

DEBUG_UNBALANCED = True


# ----------------------------- helpers --------------------------------- #

def _dbg(log_func, msg: str) -> None:
    try:
        if log_func:
            log_func(msg)
        else:
            print(msg)
    except Exception:
        print(msg)


def _preview_labels(values: List[object], max_items: int = 50, head_items: int = 30) -> str:
    values_repr = sorted((repr(v) for v in values))
    if len(values_repr) <= max_items:
        return f"[{', '.join(values_repr)}]"
    preview = ", ".join(values_repr[:head_items])
    return f"[{preview}, ... ({len(values_repr)} total)]"


def _check_balance(
    df: pd.DataFrame,
    subject_col: str,
    within_cols: List[str],
    dv_col: Optional[str] = None,
    raw_df: Optional[pd.DataFrame] = None,
    log_func=None,
) -> None:
    """
    Validate that each subject has exactly one observation per within-factor combo.
    Raises ValueError with a detailed message if unbalanced.
    """
    # Enumerate all expected combinations (order-invariant for comparison)
    levels = [sorted(df[col].dropna().unique()) for col in within_cols]
    expected_combos = list(product(*levels))

    issues = []
    # Faster check using a subject × within combo count
    key_cols = [subject_col] + within_cols
    counts = (
        df
        .assign(_one=1)
        .groupby(key_cols, dropna=False)["_one"]
        .count()
        .reset_index()
    )

    # Build a lookup: for each subject, the set of combos present and any duplicates
    missing_logged = 0
    for subject, grp in counts.groupby(subject_col, dropna=False):
        present = [tuple(row[within_cols].tolist()) for _, row in grp.iterrows()]
        dup_mask = grp["_one"] > 1
        dups = grp.loc[dup_mask, within_cols].to_records(index=False).tolist()

        # Missing combos
        for combo in expected_combos:
            if combo not in present:
                cond_str = " ".join(f"{col} {val}" for col, val in zip(within_cols, combo))
                issues.append(f"Subject {subject} missing {cond_str}")
                if DEBUG_UNBALANCED and missing_logged < 10:
                    # RM_ANOVA DEBUG: unbalanced diagnostics
                    base_df = raw_df if raw_df is not None else df
                    mask_raw = pd.Series(True, index=base_df.index)
                    mask_raw = mask_raw & (base_df[subject_col] == subject)
                    for col, val in zip(within_cols, combo):
                        mask_raw = mask_raw & (base_df[col] == val)
                    raw_count = int(mask_raw.sum())
                    valid_count = raw_count
                    if dv_col and dv_col in base_df.columns:
                        dv = base_df[dv_col]
                        mask_valid = mask_raw & dv.notna()
                        try:
                            mask_valid = mask_valid & np.isfinite(dv.astype(float))
                        except Exception:
                            pass
                        valid_count = int(mask_valid.sum())
                    combo_repr = " ".join(
                        f"{col} {repr(val)}" for col, val in zip(within_cols, combo)
                    )
                    _dbg(
                        log_func,
                        "[RM_ANOVA DEBUG] missing cell check: "
                        f"subject={repr(subject)} {combo_repr} raw_rows={raw_count} valid_rows={valid_count}",
                    )
                    if raw_count > 0 and valid_count == 0 and dv_col and dv_col in base_df.columns:
                        dv_vals = base_df.loc[mask_raw, dv_col].head(5).tolist()
                        _dbg(
                            log_func,
                            "[RM_ANOVA DEBUG] missing cell check: "
                            f"dv sample (first 5)={dv_vals!r}",
                        )
                    if raw_count == 0:
                        subj_present = subject in df[subject_col].unique()
                        cond_presence = {
                            col: (val in df[col].unique())
                            for col, val in zip(within_cols, combo)
                        }
                        _dbg(
                            log_func,
                            "[RM_ANOVA DEBUG] missing cell hint: "
                            f"subject_present={subj_present} "
                            f"condition_present={cond_presence}",
                        )
                        if not subj_present:
                            subj_preview = _preview_labels(df[subject_col].unique().tolist())
                            _dbg(log_func, f"[RM_ANOVA DEBUG] subjects preview={subj_preview}")
                        for col, val in zip(within_cols, combo):
                            if val not in df[col].unique():
                                col_preview = _preview_labels(df[col].unique().tolist())
                                _dbg(log_func, f"[RM_ANOVA DEBUG] {col} preview={col_preview}")
                    missing_logged += 1

        # Duplicate combos
        for combo in dups:
            cond_str = " ".join(f"{col} {val}" for col, val in zip(within_cols, combo))
            ndup = int(grp.loc[(grp[within_cols] == pd.Series(combo, index=within_cols)).all(axis=1), "_one"])
            issues.append(f"Subject {subject} has {ndup} duplicates for {cond_str}")

    if dv_col:
        for subject, g in df.groupby(subject_col, dropna=False):
            # Near-zero variance across conditions can produce unstable F
            if g[dv_col].var(ddof=1) is not None and g[dv_col].var(ddof=1) < np.finfo(float).eps:
                issues.append(f"Subject {subject} has nearly zero variance across conditions")

    if issues:
        raise ValueError("Data is unbalanced. " + "; ".join(issues))


def _partial_eta_squared(F: float, df_num: float, df_den: float) -> float:
    """
    Compute partial eta squared from F and dfs:
        ηp² = (F * df_num) / (F * df_num + df_den)
    Guard against division-by-zero.
    """
    try:
        num = F * df_num
        den = num + df_den
        return float(num / den) if den > 0 else np.nan
    except Exception:
        return np.nan


def _tidy_from_pingouin(pg_table: pd.DataFrame) -> pd.DataFrame:
    """
    Map Pingouin rm_anova output to the standardized table.
    Handles presence/absence of corrected p-value columns.
    """
    df = pg_table.copy()
    # Common columns in Pingouin rm_anova(detailed=True):
    # 'Source', 'ddof1', 'ddof2', 'F', 'p-unc', 'np2', possibly 'p-GG-corr', 'p-HF-corr'
    colmap = {
        "Source": "Effect",
        "F": "F Value",
        "ddof1": "Num DF",
        "ddof2": "Den DF",
        "p-unc": "Pr > F",
        "np2": "partial eta squared",
    }
    out = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})

    # Compute partial eta squared if Pingouin did not provide it
    if "partial eta squared" not in out.columns and {"F Value", "Num DF", "Den DF"}.issubset(out.columns):
        out["partial eta squared"] = out.apply(
            lambda r: _partial_eta_squared(r["F Value"], r["Num DF"], r["Den DF"]), axis=1
        )

    # If corrected p-values are available, surface them with consistent names
    if "p-GG-corr" in df.columns and "Pr > F (GG)" not in out.columns:
        out["Pr > F (GG)"] = df["p-GG-corr"]
    if "p-HF-corr" in df.columns and "Pr > F (HF)" not in out.columns:
        out["Pr > F (HF)"] = df["p-HF-corr"]

    # Reorder columns nicely if they exist
    cols = ["Effect", "F Value", "Num DF", "Den DF", "Pr > F"]
    if "Pr > F (GG)" in out.columns:
        cols.append("Pr > F (GG)")
    if "Pr > F (HF)" in out.columns:
        cols.append("Pr > F (HF)")
    cols.append("partial eta squared")

    # Keep only present columns, in desired order
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index(drop=True)


def _tidy_from_statsmodels(sm_table: pd.DataFrame) -> pd.DataFrame:
    """
    Map statsmodels AnovaRM output to the standardized table and compute partial η².
    statsmodels columns typically include: 'F Value', 'Num DF', 'Den DF', 'Pr > F'
    with the effect names in the index.
    """
    if not {"F Value", "Num DF", "Den DF", "Pr > F"}.issubset(sm_table.columns):
        raise RuntimeError("Unexpected AnovaRM table format. Got columns: " + ", ".join(sm_table.columns))

    out = sm_table.copy()
    out = out.reset_index().rename(columns={"index": "Effect"})
    out["partial eta squared"] = out.apply(
        lambda r: _partial_eta_squared(r["F Value"], r["Num DF"], r["Den DF"]), axis=1
    )

    cols = ["Effect", "F Value", "Num DF", "Den DF", "Pr > F", "partial eta squared"]
    return out[cols].reset_index(drop=True)


# ------------------------------- API ----------------------------------- #

def run_repeated_measures_anova(
    data: pd.DataFrame,
    dv_col: str,
    within_cols: List[str],
    subject_col: str,
) -> pd.DataFrame:
    """
    Run a repeated measures ANOVA on long-format data.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format table. Must contain columns: `subject_col`, `dv_col`, and all `within_cols`.
    dv_col : str
        Name of the dependent variable column.
    within_cols : list of str
        Names of within-subject factor columns (e.g., ['condition', 'roi']).
    subject_col : str
        Name of the subject identifier column.

    Returns
    -------
    pd.DataFrame
        Tidy ANOVA table with:
        ['Effect', 'F Value', 'Num DF', 'Den DF', 'Pr > F', 'partial eta squared']
        and when available from Pingouin:
        ['Pr > F (GG)', 'Pr > F (HF)']

    Raises
    ------
    ValueError
        If required columns are missing or the design is unbalanced.
    RuntimeError
        If the underlying statistical routine fails unexpectedly.
    """
    # ----- validate inputs -----
    if not isinstance(within_cols, (list, tuple)) or len(within_cols) == 0:
        raise ValueError("`within_cols` must be a non-empty list of within-subject factor names.")

    required_cols = [subject_col, dv_col] + list(within_cols)
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in data for ANOVA: {missing}")

    # Drop NAs in all required columns
    df = data.dropna(subset=required_cols).copy()
    if df.empty:
        raise ValueError("After dropping missing values, no data remain for ANOVA.")

    # Ensure balanced design
    if DEBUG_UNBALANCED:
        # RM_ANOVA DEBUG: unbalanced diagnostics
        subjects = df[subject_col].unique().tolist()
        within_levels = {
            col: df[col].dropna().unique().tolist() for col in within_cols
        }
        _dbg(None, f"[RM_ANOVA DEBUG] df_long shape={df.shape}")
        _dbg(None, f"[RM_ANOVA DEBUG] subjects({len(subjects)}): {_preview_labels(subjects)}")
        for col, vals in within_levels.items():
            _dbg(None, f"[RM_ANOVA DEBUG] {col}({len(vals)}): {_preview_labels(vals)}")
        expected_subjects = sorted(df[subject_col].dropna().unique().tolist())
        _dbg(
            None,
            f"[RM_ANOVA DEBUG] expected subjects({len(expected_subjects)}): "
            f"{_preview_labels(expected_subjects)}",
        )
        for col in within_cols:
            expected_vals = sorted(df[col].dropna().unique().tolist())
            _dbg(
                None,
                f"[RM_ANOVA DEBUG] expected {col}({len(expected_vals)}): "
                f"{_preview_labels(expected_vals)}",
            )

    _check_balance(
        df,
        subject_col=subject_col,
        within_cols=list(within_cols),
        dv_col=dv_col,
        raw_df=data,
        log_func=None,
    )

    # ----- primary path: Pingouin (if available) -----
    try:
        import pingouin as pg  # type: ignore

        # Pingouin accepts a list for multi-factor designs or a string for one factor
        within_arg = list(within_cols) if len(within_cols) > 1 else within_cols[0]
        # Use detailed output; let Pingouin decide on corrections (if supported by version)
        # We call with the safest signature; not all versions expose 'correction' kwarg.
        try:
            pg_table = pg.rm_anova(
                data=df,
                dv=dv_col,
                within=within_arg,
                subject=subject_col,
                detailed=True,
            )
        except TypeError:
            # Fallback in case an older pingouin version has a different signature
            pg_table = pg.rm_anova(
                data=df,
                dv=dv_col,
                within=within_arg,
                subject=subject_col,
                detailed=True,
            )

        out = _tidy_from_pingouin(pg_table)

        # Sanity: ensure numeric columns are numeric
        for col in ["F Value", "Num DF", "Den DF", "Pr > F", "partial eta squared"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        return out

    except ImportError:
        # Pingouin not installed; proceed to statsmodels fallback
        pass
    except Exception as e:
        # If Pingouin is installed but failed unexpectedly, fall back to statsmodels
        # while preserving the error context in case of repeated failure.
        pingouin_error = e
    else:
        pingouin_error = None  # type: ignore

    # ----- fallback path: statsmodels.AnovaRM -----
    try:
        from statsmodels.stats.anova import AnovaRM  # type: ignore
    except ImportError as e:
        # If both pingouin and statsmodels are unavailable, raise a clear error.
        raise ImportError(
            "Neither Pingouin nor statsmodels are available. "
            "Install one of them to run repeated measures ANOVA "
            "(e.g., `pip install pingouin` or `pip install statsmodels`)."
        ) from e

    try:
        aov = AnovaRM(
            data=df,
            depvar=dv_col,
            subject=subject_col,
            within=list(within_cols),
        )
        res = aov.fit()
        sm_table = res.anova_table.copy()
        out = _tidy_from_statsmodels(sm_table)
        return out
    except Exception as e:
        # If we previously caught a Pingouin error, include it for context.
        ctx = f" Pingouin error: {pingouin_error}" if 'pingouin_error' in locals() and pingouin_error else ""
        raise RuntimeError(f"Failed to run repeated measures ANOVA via statsmodels. Original error: {e}.{ctx}")
