"""Render per-image cortical contrast maps to PNG for the comparison page."""
import argparse, base64, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(a):
    from tribev2.plotting import PlotBrainNilearn
    out = Path(a.out)
    res = json.loads((out / "results.json").read_text())
    contrast = np.load(out / "contrast.npy")
    images = res["images"]

    plotter = PlotBrainNilearn(mesh="fsaverage5")
    vmax = float(np.percentile(np.abs(contrast), 99))
    assets = {}
    for i, im in enumerate(images):
        fig, axarr = plotter.get_fig_axes(views=["left", "posterior", "ventral"])
        plotter.plot_surf(
            contrast[i], axes=axarr, views=["left", "posterior", "ventral"],
            cmap="cold_hot", vmin=-vmax, vmax=vmax, symmetric_cbar=True,
            threshold=vmax * 0.25, alpha_cmap=(0, 0.25),
        )
        fig.patch.set_alpha(0)
        p = out / f"map_{im}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight", transparent=True)
        plt.close(fig)
        assets[im] = p.name
        print("rendered", p)
    (out / "maps.json").write_text(json.dumps({"vmax": vmax, "files": assets}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="viz/out")
    main(ap.parse_args())
