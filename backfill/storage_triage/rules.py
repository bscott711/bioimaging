"""Per-dataset verdicts. Pure functions over `Facts` -- no I/O -- so every rule
can be tested against a fixture that reproduces the incident it guards against.
Rule ids are cited verbatim in the report; docs/STORAGE_TRIAGE_RULES.md lists
the incident behind each.

Evaluated in order, first match wins:
  P2  active        something is running/pending/reprocessing or the dataset is new -> KEEP
  P1  protected     a downstream consumer needs it, or a human said keep -> KEEP
  D1  human delete  PI said delete AND every cross-check passes -> SUGGEST_DELETE
  R1  review        anything that needs the PI's eyes (reasons say which check failed)
  K0  default       KEEP raw + final (intermediates handled separately, rule I1)
Automated detector output alone is NEVER a reason to suggest a delete.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

ACTIVE_WINDOW = timedelta(days=14)
KEEP, SUGGEST_DELETE, REVIEW = "KEEP", "SUGGEST_DELETE", "REVIEW"
_BEAD_PSF = re.compile(r"bead|psf|focalcheck", re.I)
BAD_SIGNAL = ("dud", "dead", "corrupt")


@dataclass(frozen=True)
class Label:
    label: str                 # keep | delete | interrogate
    source: str                # human | accepted_suggestion
    updated_at: str | None


@dataclass
class Facts:
    key: str
    leaf_dir: str
    status: str                          # registry overall status
    signal_flag: str | None
    discovered_at: str | None
    reprocessed_at: str | None           # latest deskew/mip_encode finished_at
    label: Label | None
    sibling_labels: dict[str, Label] = field(default_factory=dict)  # other candidate copies
    protected: list[str] = field(default_factory=list)
    auto: dict | None = None             # auto_triage row (reason, suggested_label, ...)
    now: datetime | None = None


@dataclass
class Verdict:
    rule: str
    action: str
    reasons: list[str] = field(default_factory=list)


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def is_bead_or_psf(leaf_dir: str) -> bool:
    return bool(_BEAD_PSF.search(leaf_dir))


def decide(f: Facts) -> Verdict:
    now = f.now or datetime.now().astimezone()

    # P2 -- never touch anything still in motion
    if f.status in ("running", "in_progress", "reprocessing", "pending"):
        return Verdict("P2", KEEP, [f"pipeline status is {f.status}"])
    disc = _parse(f.discovered_at)
    if disc and now - disc < ACTIVE_WINDOW:
        return Verdict("P2", KEEP, [f"discovered {disc:%Y-%m-%d}, inside the {ACTIVE_WINDOW.days}-day settle window"])

    # P1 -- a downstream consumer or an explicit keep
    reasons = list(f.protected)
    if f.label and f.label.label == "keep":
        reasons.append("labelled keep")
    reasons += [f"copy {k.rsplit('/', 2)[-2]}/{k.rsplit('/', 1)[-1]} labelled keep" for k, l in f.sibling_labels.items() if l.label == "keep"]
    if reasons:
        # a delete label colliding with protection is surfaced, not silently kept
        if f.label and f.label.label == "delete":
            return Verdict("R1", REVIEW, ["label is delete but dataset is protected"] + reasons)
        return Verdict("P1", KEEP, reasons)

    lab = f.label
    if lab and lab.label == "delete":
        problems: list[str] = []
        if lab.source != "human":
            problems.append(f"label came from {lab.source}, not a human decision")
        conflicts = [k for k, l in f.sibling_labels.items() if l.label != "delete"]
        if conflicts:
            problems.append(f"another copy is labelled differently ({len(conflicts)})")
        lab_t, rep_t = _parse(lab.updated_at), _parse(f.reprocessed_at)
        if lab_t and rep_t and lab_t < rep_t:
            problems.append(
                f"label ({lab_t:%Y-%m-%d}) predates the latest reprocess ({rep_t:%Y-%m-%d}); re-confirm on the current movie"
            )
        if f.auto and f.auto.get("suggested_label") == "keep" and f.auto.get("reason") == "motion_detected":
            problems.append(
                f"detector sees a moving cell (activity {f.auto.get('max_activity_score', 0):.3f})"
            )
        if is_bead_or_psf(f.leaf_dir):
            problems.append("bead/PSF acquisition: averaged PSFs on disk list their source scans only partly")
        if problems:
            return Verdict("R1", REVIEW, problems)
        return Verdict("D1", SUGGEST_DELETE, ["human delete label, all cross-checks pass"])

    if lab and lab.label == "interrogate":
        return Verdict("R1", REVIEW, ["labelled interrogate (unresolved)"])
    if f.signal_flag in BAD_SIGNAL and not lab:
        return Verdict("R1", REVIEW, [f"pipeline signal flag is {f.signal_flag}; needs a human look before any delete"])

    return Verdict("K0", KEEP, ["no evidence for deletion"])
