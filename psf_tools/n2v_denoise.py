#!/usr/bin/env python3
"""Self-supervised (Noise2Void) denoise of one channel's staged skewed volume, trained on that cell's own frames.

Runs in its OWN env (CAREamics + torch; kept out of the pipeline's env):

  /mmfs2/scratch/SDSMT.LOCAL/bscott/decon_sweeps/envs/careamics/bin/python -m psf_tools.n2v_denoise \
      --train T000.tif T010.tif ... --target T023.tif --out stage_n2v/Cell_004_C0_T023.tif --work DIR

Inputs are the staged decon TIFFs, (nz, ny, nx) uint16, so the model sees the decon input's orientation and noise. N2V needs
noise that is independent between voxels given the signal: camera shot + read noise are; the PSF-correlated signal is not
touched by the blind-spot loss. The camera offset (100) is preserved by training in float and clipping on the way out.
Photon-starved data: the output is an estimate of the expectation, not extra photons.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=120, help="batches per epoch")
    ap.add_argument("--patch", type=int, nargs=3, default=(16, 96, 96))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args(argv)

    import torch
    from careamics import CAREamist
    from careamics.config import create_n2v_config

    torch.set_float32_matmul_precision("medium")
    vols = np.stack([tifffile.imread(f).astype(np.float32) for f in a.train])  # (S, Z, Y, X)
    print(f"training on {vols.shape} from {[f.name for f in a.train]}", flush=True)
    cfg = create_n2v_config(experiment_name=a.out.stem, data_type="array", axes="SZYX", patch_size=list(a.patch),
                            batch_size=a.batch, num_epochs=a.epochs, num_steps=a.steps, n_val_patches=16)
    cfg.training_config.trainer_params = {**(cfg.training_config.trainer_params or {}), "accelerator": "gpu", "devices": [a.gpu]}
    a.work.mkdir(parents=True, exist_ok=True)
    cm = CAREamist(cfg, work_dir=a.work)
    cm.train(train_data=vols)
    tgt = tifffile.imread(a.target).astype(np.float32)
    pred, _ = cm.predict(tgt, data_type="array", axes="ZYX", tile_size=(32, 256, 256), tile_overlap=(8, 48, 48))
    out = np.squeeze(np.asarray(pred[0]))
    if out.shape != tgt.shape:
        raise SystemExit(f"denoised shape {out.shape} != target {tgt.shape}")
    out = np.clip(np.rint(out), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(a.out, out, compression="zlib")
    d = out.astype(np.float32) - tgt
    print(f"wrote {a.out}: median in {np.median(tgt):.1f} out {np.median(out):.1f}, mean |change| {np.abs(d).mean():.2f}, "
          f"p99.9 in {np.percentile(tgt, 99.9):.0f} out {np.percentile(out, 99.9):.0f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
