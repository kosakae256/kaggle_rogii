# -*- coding: utf-8 -*-
"""v93 L3 image builder -- STRUCT-FREE 2D evidence-image renderer.

Renders each well as a 2D "evidence image" EV of shape (C_img, nb, ND):
  - ND = 181 delta-bins along a last_known-relative delta axis:
        OFF = (arange(181) - 90) * 0.5  ->  +/- 45 ft around last_known_tvt.
  - nb  = number of segment-bins (STR=16), taken verbatim from seqfeat_v93.pkl.
  - C_img = 14 struct-free trajectories, each drawn as a soft Gaussian ridge per bin:
        ridge(b, :) = exp(-0.5 * ((OFF - D[b,c]) / S[b,c])^2)
    where D[b,c] is the trajectory delta (already relative to last_known_tvt) and
    S[b,c] its width. This mirrors v62 build_field_v62.py's stripe() idea but the
    band center is 0 = last_known (NOT the v38 struct prior `st`).

CRITICAL struct-free guarantee:
  v62's imagesF/imagesR carry a struct-contaminated band center `st` (traced to
  v38 struct_prior via v61 -> nnpf_v51 -> stageA_v50). This builder reads ONLY
  seqfeat_v93.pkl (which is itself built struct-free from the v93 parquet, re-anchored
  to last_known_tvt; see build_seqfeat_v93.py). It never touches
  gf / v50b / struct / v38 / v41 / imagesR / imagesF / st. An explicit assert enforces
  that no banned source path is opened.

Output: v93/artifacts_v93/images_v93.pkl
  = {well: {EV:(14,nb,181) float16, dtrue:(nb,) f32, wgt:(nb,) f32,
            bid:(T,) int32, rows:(T,) int64, T:int}}
  EV is float16 to stay memory-lean (~1.5 MB/well at mean nb).

cp932-safe output ('->' not em-dash).
"""
import io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import os, pickle, time
from pathlib import Path
import numpy as np

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = P0 / "v93" / "artifacts_v93"
SF_FP = ART / "seqfeat_v93.pkl"
OUT = ART / "images_v93.pkl"

ND = 181
NBc = 90                      # center index of the delta grid
STEP = 0.5
OFF = (np.arange(ND) - NBc) * STEP   # (181,) delta axis, +/-45 ft, 0 = last_known
K_IMG = 14                    # 14 struct-free trajectories = seqfeat D channels

# --- struct-free assert: this file must only read seqfeat_v93.pkl -------------
_BANNED = ("gf_feature", "v50b", "struct", "v38", "v41", "imagesr", "imagesf",
           "stagea", "nnpf")
assert not any(b in str(SF_FP).lower() for b in _BANNED), "banned source path"


def render(D, S):
    """D,S: (nb,14) -> EV (14,nb,181). Soft Gaussian ridge per (channel,bin).

    D is the trajectory delta relative to last_known (band center = 0), so the
    ridge for channel c sits at OFF == D[b,c]. S is the clipped width from seqfeat.
    """
    nb = D.shape[0]
    # (nb,14,181): exp(-0.5*((OFF - D)/S)^2)
    z = (OFF[None, None, :] - D[:, :, None]) / np.maximum(S[:, :, None], 1e-3)
    img = np.exp(-0.5 * z * z)                       # (nb,14,181) float32
    ev = np.transpose(img, (1, 0, 2))                # (14,nb,181)
    return ev.astype(np.float16)


def main():
    t0 = time.time()
    SF = pickle.loads(SF_FP.read_bytes())
    wells = sorted(SF.keys())
    K = SF[wells[0]]["D"].shape[1]
    assert K == K_IMG, f"expected 14 struct-free channels, got {K}"

    OUTD = {}
    tot_bytes = 0
    for wi, wn in enumerate(wells):
        f = SF[wn]
        D = np.asarray(f["D"], np.float32)           # (nb,14) delta vs last_known
        S = np.asarray(f["S"], np.float32)           # (nb,14) width
        EV = render(D, S)                            # (14,nb,181) f16
        tot_bytes += EV.nbytes
        OUTD[wn] = dict(
            EV=EV,
            dtrue=np.asarray(f["dtrue"], np.float32),
            wgt=np.asarray(f["wgt"], np.float32),
            bid=np.asarray(f["bid"], np.int32),
            rows=np.asarray(f["rows"], np.int64),
            T=int(f["T"]),
        )
        if (wi + 1) % 150 == 0:
            print(f"  {wi+1}/{len(wells)} ({time.time()-t0:.0f}s)", flush=True)

    pickle.dump(OUTD, open(OUT, "wb"), protocol=4)
    k0 = next(iter(OUTD.values()))
    nbs = [v["EV"].shape[1] for v in OUTD.values()]
    print(f"saved {OUT.name}: {len(OUTD)} wells  EV=(C={k0['EV'].shape[0]}, nb, "
          f"ND={k0['EV'].shape[2]})  [struct-free, last_known-anchored]")
    print(f"  nb: min={min(nbs)} max={max(nbs)} mean={np.mean(nbs):.1f}  "
          f"EV total={tot_bytes/1e6:.0f} MB (float16)")
    print(f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
