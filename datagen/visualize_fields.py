"""Visualize log10(ks), log10(mu / ks), and log10(beta) for one sample."""

import matplotlib.pyplot as plt
import numpy as np

from .fft_ma import fft_ma_2d


RESOLUTION = 64
length = 1.0
LOG10_KS_MEAN = -8
LOG10_KS_STDEV = 1
CORRELATION_LENGTHS = (0.05, 0.275)
ANGLE_DEGREES = 45
VISCOSITY = 1.0
FLUID_QUANTILE = 0.8


def main():
    log10_ks = fft_ma_2d(
        nx=RESOLUTION,
        ny=RESOLUTION,
        dx=length / RESOLUTION,
        dy=length / RESOLUTION,
        mean_value=LOG10_KS_MEAN,
        stdev=LOG10_KS_STDEV,
        scale=CORRELATION_LENGTHS,
        angle=ANGLE_DEGREES,
    )

    ks = 10**log10_ks
    threshold = np.quantile(
        ks, FLUID_QUANTILE
    )
    is_fluid = ks > threshold
    beta = VISCOSITY / ks
    beta[is_fluid] = np.nan

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    log10_mu_over_ks = np.log10(VISCOSITY / ks)
    log10_beta = np.log10(beta)

    panels = (
        (log10_ks, r"$\log_{10}(k_s)$"),
        (log10_mu_over_ks, r"$\log_{10}(\mu/k_s)$"),
        (log10_beta, r"$\log_{10}(\beta)$ (solid only)"),
    )
    colormap = plt.colormaps["viridis"].copy()
    colormap.set_bad("white")

    for axis, (field, title) in zip(axes, panels):
        image = axis.imshow(field, origin="lower", cmap=colormap)
        axis.set_title(title)
        axis.set_xlabel("x (pixels)")
        axis.set_ylabel("y (pixels)")
        figure.colorbar(image, ax=axis, label="log10 value")

    plt.show()


if __name__ == "__main__":
    main()
