"""Build a flashed-image stimulus video following TRIBE v2 paper section 5.9.

Images are shown for `on_dur` seconds every `soa` seconds against a mid-grey
background, in randomised order, repeated `reps` times.  This converts still
images into the static-video form the model expects, and records the onset
time of every trial so responses can be recovered per image.
"""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

FPS = 16          # matches V-JEPA-2 sampling of 64 frames per 4 s
SIZE = (512, 512)
GREY = 128


def _load(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    im.thumbnail(SIZE, Image.LANCZOS)
    canvas = Image.new("RGB", SIZE, (GREY, GREY, GREY))
    canvas.paste(im, ((SIZE[0] - im.width) // 2, (SIZE[1] - im.height) // 2))
    return np.asarray(canvas)


def build(image_dir, out_video, out_events, reps=3, on_dur=1.0, soa=8.0, seed=0):
    image_dir = Path(image_dir)
    paths = sorted(p for p in image_dir.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not paths:
        raise SystemExit(f"no images found in {image_dir}")
    frames_by_name = {p.stem: _load(p) for p in paths}
    grey = np.full((*SIZE[::-1], 3), GREY, np.uint8)

    rng = random.Random(seed)
    order = []
    for _ in range(reps):
        block = [p.stem for p in paths]
        rng.shuffle(block)
        order.extend(block)

    n_on, n_off = int(on_dur * FPS), int((soa - on_dur) * FPS)
    frames, trials = [], []
    for i, name in enumerate(order):
        trials.append({"index": i, "image": name, "onset": round(i * soa, 3)})
        frames.extend([frames_by_name[name]] * n_on)
        frames.extend([grey] * n_off)

    import imageio.v2 as imageio
    Path(out_video).parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(out_video, fps=FPS, codec="libx264",
                           macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    for f in frames:
        w.append_data(f)
    w.close()

    meta = {"images": [p.stem for p in paths], "reps": reps, "on_dur": on_dur,
            "soa": soa, "fps": FPS, "duration": len(frames) / FPS, "trials": trials}
    Path(out_events).write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_video} ({meta['duration']:.1f}s, {len(trials)} trials, "
          f"{len(frames_by_name)} images x {reps} reps)")
    return meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", default="viz/stimuli")
    ap.add_argument("--out-video", default="viz/out/stimulus.mp4")
    ap.add_argument("--out-events", default="viz/out/trials.json")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--soa", type=float, default=8.0)
    a = ap.parse_args()
    build(a.image_dir, a.out_video, a.out_events, reps=a.reps, soa=a.soa)
