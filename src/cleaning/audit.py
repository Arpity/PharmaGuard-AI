"""Cleaning audit log: one row per change (or per documented decision)."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

AUDIT_COLUMNS = ["timestamp", "step", "issue", "source_row", "batch_id", "column",
                 "original_value", "new_value", "action", "reason"]


def _fmt(v) -> str:
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return "<missing>"
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v)) if abs(v) < 1e15 else str(v)
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return str(v)


class AuditLog:
    def __init__(self) -> None:
        self._frames: list[pd.DataFrame] = []

    def log(self, df: pd.DataFrame, mask: pd.Series, column: str, original: pd.Series, new,
            step: str, issue: str, action: str, reason: str) -> int:
        """Record one entry per row where `mask` is True. `df` supplies row ids / Batch_ID;
        `new` is a Series aligned to df or a scalar."""
        idx = df.index[mask.to_numpy(dtype=bool)]
        if len(idx) == 0:
            return 0
        new_vals = new.loc[idx] if isinstance(new, pd.Series) else pd.Series([new] * len(idx), index=idx)
        self._frames.append(pd.DataFrame({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "step": step, "issue": issue, "source_row": idx,
            "batch_id": df.loc[idx, "Batch_ID"].map(_fmt).to_numpy() if "Batch_ID" in df else "",
            "column": column,
            "original_value": [_fmt(v) for v in original.loc[idx]],
            "new_value": [_fmt(v) for v in new_vals],
            "action": action, "reason": reason}))
        return len(idx)

    def note(self, step: str, issue: str, column: str, original: str, new: str, action: str,
             reason: str) -> None:
        """Column-level decision that has no single row (e.g. 'not imputed')."""
        self._frames.append(pd.DataFrame([{
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "step": step,
            "issue": issue, "source_row": "", "batch_id": "", "column": column,
            "original_value": original, "new_value": new, "action": action, "reason": reason}]))

    def to_frame(self) -> pd.DataFrame:
        if not self._frames:
            return pd.DataFrame(columns=AUDIT_COLUMNS)
        return pd.concat(self._frames, ignore_index=True)[AUDIT_COLUMNS]
