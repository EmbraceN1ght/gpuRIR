#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, spherical_jn, spherical_yn

GPU_RIR_SRC = ""
if GPU_RIR_SRC and (GPU_RIR_SRC not in sys.path):
    sys.path.append(GPU_RIR_SRC)

import gpuRIR

gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(False)

FS = 16000
C = 343.0
TMAX = 0.05

L_ROOM = [10.0, 10.0, 10.0]
SRC = np.array([[3.05, 3.37, 1.70]], dtype=np.float32)
RCV_CENTER = np.array([2.60, 4.05, 1.70], dtype=np.float32)

EM32_RADIUS = 0.042
EM32_ANGLES_DEG = [
    (69, 0), (90, 0), (111, 0), (90, 45),
    (0, 45), (180, 45), (32, 69), (0, 90),
    (180, 90), (328, 69), (0, 135), (180, 135),
    (69, 180-32), (90, 180-32), (111, 180-32), (90, 180-0),
    (69, 32), (90, 32), (111, 32), (148, 0),
    (212, 0), (148, 45), (212, 45), (180, 69),
    (0, 69), (32, 90), (148, 90), (212, 90),
    (270, 69), (328, 90), (32, 111), (328, 111)
]

BETA = np.zeros(6, np.float32)
NB_IMG = [1, 1, 1]

N_SH = 3
F_MAX = 4000.0
REG_LAMBDA = 1e-6

PLOT_MS = 50.0

def wrap_elevation_deg(elev_deg: float) -> float:
    return elev_deg if elev_deg <= 90.0 else (elev_deg - 180.0)

def spherical_to_cartesian(radius: float, azimuth_deg: float, elevation_deg_encoded: float):
    az = math.radians(azimuth_deg)
    el = math.radians(wrap_elevation_deg(elevation_deg_encoded))
    x = radius * math.cos(el) * math.cos(az)
    y = radius * math.cos(el) * math.sin(az)
    z = radius * math.sin(el)
    return x, y, z

def build_em32_positions_and_angles():
    pos = []
    phi = []
    theta = []
    for az_deg, el_enc_deg in EM32_ANGLES_DEG:
        x, y, z = spherical_to_cartesian(EM32_RADIUS, az_deg, el_enc_deg)
        pos.append([x, y, z])
        el_deg = wrap_elevation_deg(el_enc_deg)
        phi.append((math.radians(az_deg) % (2.0 * math.pi)))
        theta.append((math.pi / 2.0) - math.radians(el_deg))
    return (np.array(pos, dtype=np.float64),
            np.array(phi, dtype=np.float64),
            np.array(theta, dtype=np.float64))

def simulate_em32_direct_only(rcv_positions: np.ndarray, nsample: int):
    rir = gpuRIR.simulateRIR(
        L_ROOM, BETA, SRC, rcv_positions, NB_IMG, TMAX, FS,
        orV_src=None, orV_rcv=None,
        spkr_pattern="omni",
        mic_pattern="omni",
    )
    rir = rir[0]
    rir = rir[:, :nsample].astype(np.float64)
    rir[~np.isfinite(rir)] = 0.0
    return rir

def build_Y_matrix(phi: np.ndarray, theta: np.ndarray, N: int):
    M = phi.shape[0]
    Q = (N + 1) ** 2
    Y = np.zeros((M, Q), dtype=np.complex128)
    order_idx = np.zeros(Q, dtype=np.int32)
    for n in range(N + 1):
        for m in range(-n, n + 1):
            q = n * (n + 1) + m
            Y[:, q] = sph_harm(m, n, phi, theta)
            order_idx[q] = n
    return Y, order_idx

def rigid_scattering_factor_Sn(n: int, x: np.ndarray):
    S = np.ones_like(x, dtype=np.complex128)
    mask = x > 1e-9
    if not np.any(mask):
        return S
    xx = x[mask]
    j = spherical_jn(n, xx)
    jp = spherical_jn(n, xx, derivative=True)
    y = spherical_yn(n, xx)
    yp = spherical_yn(n, xx, derivative=True)
    h2 = j - 1j * y
    h2p = jp - 1j * yp
    eps0 = 1e-12
    valid = np.abs(j) >= eps0
    S_tmp = np.ones_like(j, dtype=np.complex128)
    S_tmp[valid] = 1.0 - (jp[valid] / h2p[valid]) * (h2[valid] / j[valid])
    S[mask] = S_tmp
    return S

def apply_rigid_sphere(rir_open: np.ndarray, phi: np.ndarray, theta: np.ndarray, radius_a: float):
    M, T = rir_open.shape
    Nfft = int(2 ** np.ceil(np.log2(T)))
    K = Nfft // 2 + 1
    freqs = np.fft.rfftfreq(Nfft, d=1.0 / FS)
    x = (2.0 * np.pi * freqs / C) * radius_a

    H_open = np.fft.rfft(rir_open, n=Nfft, axis=1)

    if F_MAX is not None:
        band = freqs <= float(F_MAX)
        H_open[:, ~band] = 0.0

    Y, order_idx = build_Y_matrix(phi, theta, N_SH)
    Q = Y.shape[1]
    Yh = Y.conj().T
    G = Yh @ Y
    Aop = np.linalg.solve(G + REG_LAMBDA * np.eye(Q), Yh)
    Acoef = Aop @ H_open

    S_order = np.zeros((N_SH + 1, K), dtype=np.complex128)
    for n in range(N_SH + 1):
        S_order[n, :] = rigid_scattering_factor_Sn(n, x)

    S_qk = S_order[order_idx, :]
    H_closed = Y @ (Acoef * S_qk)

    if F_MAX is not None:
        band = freqs <= float(F_MAX)
        H_closed[:, ~band] = 0.0

    rir_closed = np.fft.irfft(H_closed, n=Nfft, axis=1)
    rir_closed = np.real(rir_closed[:, :T])
    return rir_closed

def add_info_box(ax, text: str):
    ax.text(
        0.99, 0.99, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black", alpha=0.85),
    )

def main():
    t_total0 = time.time()

    em32_positions, phi_arr, theta_arr = build_em32_positions_and_angles()
    rcv_positions = (RCV_CENTER[None, :].astype(np.float64) + em32_positions).astype(np.float32)

    nsample = int(round(TMAX * FS))
    plotN = int(round(min(PLOT_MS / 1000.0, TMAX) * FS))
    t_ms = np.arange(plotN, dtype=np.float64) / FS * 1000.0

    t_sim0 = time.time()
    rir_open = simulate_em32_direct_only(rcv_positions, nsample)
    t_sim_ms = (time.time() - t_sim0) * 1000.0

    t_plan0 = time.time()
    rir_rigid = apply_rigid_sphere(rir_open, phi_arr, theta_arr, EM32_RADIUS)
    t_plan_ms = (time.time() - t_plan0) * 1000.0

    t_total_ms = (time.time() - t_total0) * 1000.0

    n_mics = int(rir_rigid.shape[0])
    per_ch_ms = t_total_ms / max(n_mics, 1)

    for ch in range(n_mics):
        np.savetxt(f"gpuRIR_EM32_direct_rigid_ch{ch+1}.txt", rir_rigid[ch])
    print("Saved: gpuRIR_EM32_direct_rigid_ch1-32.txt")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1)

    for ch in range(n_mics):
        h = rir_rigid[ch, :plotN].astype(np.float64)
        s = max(np.max(np.abs(h)), 1e-12)
        ax.plot(t_ms, h / s, linewidth=0.7, alpha=0.8)

    ax.set_xlim(0, float(PLOT_MS))
    ax.set_title("Rigid RIR (32 channels)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

    add_info_box(
        ax,
        f"Total runtime: {t_total_ms:.2f} ms\nPer-channel: {per_ch_ms:.3f} ms/ch"
    )

    plt.tight_layout()
    plt.savefig("gpuRIR_EM32_direct_rigid_32ch_overlay.png", dpi=150, bbox_inches="tight")
    print("Saved: gpuRIR_EM32_direct_rigid_32ch_overlay.png")

    print(f"Timing breakdown: simulate={t_sim_ms:.2f} ms, planB={t_plan_ms:.2f} ms, total={t_total_ms:.2f} ms")
    print(f"Per-channel total: {per_ch_ms:.3f} ms/ch")

    plt.show()

if __name__ == "__main__":
    main()