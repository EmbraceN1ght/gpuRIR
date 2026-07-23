#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt

GPU_RIR_SRC = ""
if GPU_RIR_SRC and (GPU_RIR_SRC not in sys.path):
    sys.path.append(GPU_RIR_SRC)

import gpuRIR

gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(False)

FS = 16000
TMAX = 0.500
C = 343.0

L_ROOM = [10.0, 10.0, 10.0]
SRC = np.array([[3.05, 3.37, 1.70]], dtype=np.float32)
RCV_CENTER = np.array([2.60, 4.05, 1.70], dtype=np.float32)

EM32_RADIUS = 0.042
EM32_ANGLES_DEG = [
    (69, 0), (90, 0), (111, 0), (90, 45),
    (0, 45), (180, 45), (32, 69), (0, 90),
    (180, 90), (328, 69), (0, 135), (180, 135),
    (69, 180 - 32), (90, 180 - 32), (111, 180 - 32), (90, 180 - 0),
    (69, 32), (90, 32), (111, 32), (148, 0),
    (212, 0), (148, 45), (212, 45), (180, 69),
    (0, 69), (32, 90), (148, 90), (212, 90),
    (270, 69), (328, 90), (32, 111), (328, 111)
]

BETA = np.ones(6, np.float32)
NB_IMG = [3, 3, 3]

def wrap_elevation_deg(elev_deg: float) -> float:
    return elev_deg if elev_deg <= 90.0 else (elev_deg - 180.0)

def spherical_to_cartesian(radius: float, azimuth_deg: float, elevation_deg_encoded: float):
    az = math.radians(azimuth_deg)
    el = math.radians(wrap_elevation_deg(elevation_deg_encoded))
    x = radius * math.cos(el) * math.cos(az)
    y = radius * math.cos(el) * math.sin(az)
    z = radius * math.sin(el)
    return x, y, z

def build_em32_positions():
    pos = [spherical_to_cartesian(EM32_RADIUS, az, el) for az, el in EM32_ANGLES_DEG]
    return np.array(pos, dtype=np.float32)

def simulate_rir(L_room, beta, src, rcv_positions, nb_img, Tmax, fs):
    rir = gpuRIR.simulateRIR(
        L_room, beta, src, rcv_positions, nb_img, Tmax, fs,
        orV_src=None, orV_rcv=None,
        spkr_pattern="omni",
        mic_pattern="omni",
    )
    rir = rir[0]
    ns = int(round(Tmax * fs))
    rir = rir[:, :ns].astype(np.float32)
    rir[~np.isfinite(rir)] = 0.0
    return rir

def add_info_box(ax, text: str, loc="upper right"):
    if loc == "upper right":
        xy = (0.99, 0.99)
        ha, va = "right", "top"
    elif loc == "upper left":
        xy = (0.01, 0.99)
        ha, va = "left", "top"
    elif loc == "lower right":
        xy = (0.99, 0.01)
        ha, va = "right", "bottom"
    else:
        xy = (0.01, 0.01)
        ha, va = "left", "bottom"

    ax.text(
        xy[0], xy[1], text,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black", alpha=0.85),
    )

def main():
    t_total0 = time.time()

    em32_positions = build_em32_positions()
    rcv_positions = (RCV_CENTER[None, :] + em32_positions).astype(np.float32)

    nsample = int(round(TMAX * FS))
    t_ms = np.arange(nsample, dtype=np.float64) / FS * 1000.0

    d_center = float(np.linalg.norm(SRC[0].astype(np.float64) - RCV_CENTER.astype(np.float64)))
    toa_ms = (d_center / C) * 1000.0

    t_sim0 = time.time()
    rir_all = simulate_rir(L_ROOM, BETA, SRC, rcv_positions, NB_IMG, TMAX, FS)
    t_sim_ms = (time.time() - t_sim0) * 1000.0

    t_total_ms = (time.time() - t_total0) * 1000.0

    n_mics = int(rir_all.shape[0])
    per_mic_ms = t_sim_ms / max(n_mics, 1)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1)

    for ch_idx in range(n_mics):
        h = rir_all[ch_idx].astype(np.float64)
        s = max(np.max(np.abs(h)), 1e-12)
        ax.plot(t_ms, h / s, linewidth=0.7, alpha=0.8)

    ax.axvline(x=toa_ms, linestyle="--", linewidth=1.0)
    ax.set_xlim(0, 100)
    ax.set_title("gpuRIR EM32 RIR (32 channels)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

    add_info_box(
        ax,
        f"Total runtime: {t_total_ms:.2f} ms\nPer-channel: {per_mic_ms:.3f} ms/ch",
        loc="upper right",
    )

    plt.tight_layout()
    plt.savefig("gpuRIR_EM32_32channels_overlay.png", dpi=150, bbox_inches="tight")
    print("Saved: gpuRIR_EM32_32channels_overlay.png")
    plt.show()

if __name__ == "__main__":
    main()