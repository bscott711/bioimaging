"""Bulk no-decon backfill: crop -> channel-remap -> zarr -> deskew/rotate ->
MIP, across every raw OPM acquisition on the lab's data roots. Deliberately
skips deconvolution -- see run_backfill_cli.py for the full pipeline
description and this package's README-equivalent, the top-level plan doc.
"""
