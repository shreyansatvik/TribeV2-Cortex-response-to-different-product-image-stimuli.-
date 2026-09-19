"""Run TRIBE v2 on a flashed-image stimulus and summarise the predicted
response per image.

Outputs a JSON bundle consumed by the comparison web page.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SNAP = "hf_cache/models--facebook--tribev2/snapshots/f894e783020944dcd96e5568550afe2aa9743f9f"
ATLAS = Path("viz/atlas")


def load_atlas():
    """Schaefer-400 / 17-network labels concatenated lh+rh on fsaverage5."""
    import nibabel as nb
    labs, names = [], None
    for h in ("lh", "rh"):
        lab, _, nm = nb.freesurfer.read_annot(ATLAS / f"{h}.Schaefer400.annot")
        nm = [n.decode() for n in nm]
        if names is None:
            names = nm
        labs.append(lab)
    full = np.concatenate(labs)                      # (20484,)
    net_of_parcel = {}
    for i, n in enumerate(names):
        parts = n.split("_")
        # names[0] is "Background+FreeSurfer_Defined_Medial_Wall" -> not a network
        bad = n.startswith("Background") or "Medial_Wall" in n
        net_of_parcel[i] = "None" if bad else (parts[2] if len(parts) > 2 else "None")
    nets = sorted({v for v in net_of_parcel.values() if v != "None"})
    masks = {net: np.isin(full, [i for i, v in net_of_parcel.items() if v == net])
             for net in nets}
    return full, masks


def main(a):
    import torch
    from tribev2 import TribeModel
    from tribev2.demo_utils import get_audio_and_text_events

    meta = json.loads(Path(a.trials).read_text())
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)

    model = TribeModel.from_pretrained(
        SNAP, cache_folder="./cache", device="cpu",
        config_update={"data.video_feature.image.device": a.device,
                       "data.video_feature.image.batch_size": 1,
                       "data.audio_feature.device": "cpu",
                       "data.text_feature.device": "cpu"},
    )
    ev = pd.DataFrame([{"type": "Video", "filepath": a.video, "start": 0,
                        "timeline": "default", "subject": "default"}])
    ev = get_audio_and_text_events(ev, audio_only=True)
    preds, segments = model.predict(events=ev)
    times = np.array([s.start for s in segments], float)
    order = np.argsort(times)
    preds, times = preds[order], times[order]
    np.save(out_dir / "preds.npy", preds)
    np.save(out_dir / "times.npy", times)

    # peri-stimulus window, 0..WIN seconds after each onset
    WIN = int(a.window)
    images = meta["images"]
    per_img = {im: [] for im in images}
    for tr in meta["trials"]:
        idx = [int(np.argmin(np.abs(times - (tr["onset"] + dt)))) for dt in range(WIN + 1)]
        if np.abs(times[idx[0]] - tr["onset"]) > 1.5:
            continue                                  # onset not covered
        per_img[tr["image"]].append(preds[idx])       # (WIN+1, V)
    curves = {im: np.mean(v, 0) for im, v in per_img.items() if v}
    if not curves:
        raise SystemExit("no trials aligned to predictions")

    _, masks = load_atlas()
    stack = np.stack([curves[im] for im in images])   # (I, T, V)
    # global peak timing, from the data rather than assumed
    gmag = np.abs(stack).mean(axis=(0, 2))
    peak = int(np.argmax(gmag))
    at_peak = stack[:, peak, :]                       # (I, V)
    # paper-style contrast: image vs mean of the others
    contrast = np.stack([at_peak[i] - np.delete(at_peak, i, 0).mean(0)
                         for i in range(len(images))])

    res = {"images": images, "peak_s": peak, "window": WIN,
           "n_trials_per_image": {im: len(v) for im, v in per_img.items()},
           "timecourse_mag": {im: [float(x) for x in np.abs(curves[im]).mean(1)]
                              for im in images},
           "networks": sorted(masks), "per_image": {}}
    for i, im in enumerate(images):
        nets = {n: float(contrast[i][m].mean()) for n, m in masks.items()}
        res["per_image"][im] = {
            "networks": nets,
            "response_magnitude": float(np.abs(at_peak[i]).mean()),
            "spatial_extent": float((np.abs(contrast[i]) >
                                     np.percentile(np.abs(contrast), 90)).mean()),
        }
    # pairwise dissimilarity of contrast maps (1 - corr)
    C = contrast - contrast.mean(1, keepdims=True)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    res["dissimilarity"] = (1 - C @ C.T).tolist()
    np.save(out_dir / "contrast.npy", contrast)
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    print("peak at t=%ds; wrote %s" % (peak, out_dir / "results.json"))
    for im in images:
        print(f"  {im}: mag={res['per_image'][im]['response_magnitude']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="viz/out/stimulus.mp4")
    ap.add_argument("--trials", default="viz/out/trials.json")
    ap.add_argument("--out", default="viz/out")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--device", default="mps")
    main(ap.parse_args())
