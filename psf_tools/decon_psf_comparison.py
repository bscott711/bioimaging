#!/usr/bin/env python3
"""A/B/C decon comparison across candidate PSFs on the same dataset -- the
end-to-end confirmation that a new PSF (workstream-1's robustly-averaged
bead PSF, or workstream-2's phase-retrieved PSF) actually reduces
deconvolution artifacts (e.g. streaking) compared to whatever PSF is
currently in use, rather than just looking better in isolation.

Modeled on decon_iteration_sweep.py's submit/poll pattern, but sweeps
`psf_paths` at a single fixed iteration count/method instead of sweeping
iterations -- decon happens before deskew, so results land in
"<target_dir>/Decon_<label>", then "<target_dir>/Decon_<label>/DSR" once
deskewed.

Usage:
  uv run python -m psf_tools.decon_psf_comparison \\
      --target-dir /path/to/cell_MMStack_Pos0_test \\
      --psf old_master /path/to/Master_PSF_Final_Strict.tif \\
      --psf workstream1 /path/to/averaged_psf.tif \\
      --psf workstream2 /path/to/averaged_psf_phase_retrieved.tif
"""

import argparse
import re
import sys
import time
from pathlib import Path

from opym.petakit import submit_remote_decon_job, submit_remote_deskew_job

BASE_DIR = Path.home() / "petakit_jobs"


def _detect_channel_pattern(target_dir):
    tif_files = list(target_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No TIF files found in {target_dir}")
    first_tif = tif_files[0].name
    match = re.match(r"(.*?)_T\d+\.tif", first_tif)
    return match.group(1) if match else first_tif.replace(".tif", "")


def _wait_for_tickets(tickets, label):
    start_time = time.time()
    while True:
        elapsed = int(time.time() - start_time)
        completed = sum(1 for t, _ in tickets if (BASE_DIR / "completed" / t.name).exists())
        failed = sum(1 for t, _ in tickets if (BASE_DIR / "failed" / t.name).exists())
        sys.stdout.write(
            f"\r{label} elapsed: {elapsed}s | Completed: {completed}/{len(tickets)} "
            f"| Failed: {failed}/{len(tickets)}{' ' * 5}"
        )
        sys.stdout.flush()
        if failed > 0:
            print(f"\nSome {label} jobs failed. Check logs for details.")
            return False
        if completed == len(tickets):
            print(f"\nAll {len(tickets)} {label} jobs completed in {elapsed}s.")
            return True
        time.sleep(2)


def run_comparison(target_dir, psf_candidates, iterations=25, rl_method="simplified",
                    sheet_angle_deg=60.0, z_step_um=0.3):
    """``psf_candidates``: dict of {label: psf_path}, e.g.
    {"old_master": ..., "workstream1": ..., "workstream2": ...}."""
    target_dir = Path(target_dir)
    channel_pattern = _detect_channel_pattern(target_dir)
    print(f"Auto-detected channel pattern: {channel_pattern}")

    decon_tickets = []
    for label, psf_path in psf_candidates.items():
        result_dir = f"Decon_{label}"
        ticket = submit_remote_decon_job(
            input_target=target_dir,
            psf_paths=Path(psf_path),
            iterations=iterations,
            gpu_job=True,
            skewed=True,
            result_dir_name=result_dir,
            channel_patterns=[channel_pattern],
            rl_method=rl_method,
        )
        decon_tickets.append((ticket, result_dir))
        time.sleep(0.005)  # guarantee unique ticket-file timestamps

    print(f"Dispatched {len(decon_tickets)} decon jobs across candidate PSFs: {list(psf_candidates)}")
    if not _wait_for_tickets(decon_tickets, "Decon"):
        return

    deskew_tickets = []
    for ticket, result_dir in decon_tickets:
        decon_output_dir = target_dir / result_dir
        deskew_ticket = submit_remote_deskew_job(
            input_target=decon_output_dir,
            sheet_angle_deg=sheet_angle_deg,
            z_step_um=z_step_um,
            rotate=True,
            reverse=True,
        )
        deskew_tickets.append((deskew_ticket, result_dir))

    print(f"Dispatched {len(deskew_tickets)} deskew jobs.")
    if not _wait_for_tickets(deskew_tickets, "Deskew"):
        return

    print("Done. Compare deskewed MIPs across:")
    for label, (_ticket, result_dir) in zip(psf_candidates, deskew_tickets):
        print(f"  {label}: {target_dir / result_dir}")


def _build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument(
        "--psf", action="append", nargs=2, metavar=("LABEL", "PATH"),
        required=True, dest="psf_candidates",
        help="Repeatable: --psf old_master /path/a.tif --psf workstream2 /path/b.tif",
    )
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--rl-method", default="simplified")
    parser.add_argument("--sheet-angle-deg", type=float, default=60.0)
    parser.add_argument("--z-step-um", type=float, default=0.3)
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    psf_candidates = {label: path for label, path in args.psf_candidates}
    run_comparison(
        args.target_dir, psf_candidates,
        iterations=args.iterations, rl_method=args.rl_method,
        sheet_angle_deg=args.sheet_angle_deg, z_step_um=args.z_step_um,
    )


if __name__ == "__main__":
    main()
