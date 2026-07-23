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

N_SH = 3
F_MAX = 4000.0
TAPER_HZ = 600.0
PAD_FACTOR = 32
REG_LAMBDA = 1e-6
CAUSAL_SHIFT_MS = 0.0

FS = 16000
C = 343.0

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

L_ROOM = [10.0, 10.0, 10.0]
SRC = np.array([[3.05, 3.37, 1.70]], dtype=np.float32)
RCV_CENTER = np.array([2.60, 4.05, 1.70], dtype=np.float32)

ORDER = 6
NB_IMG = [2 * ORDER + 1, 2 * ORDER + 1, 2 * ORDER + 1]
BETA_WALLS = 0.8 * np.ones(6, np.float32)

TMAX = 0.50
PLOT_MS = 200.0


def wrap_elevation_deg(elev_deg: float) -> float:
    return elev_deg if elev_deg <= 90.0 else (elev_deg - 180.0)


def spherical_to_cartesian(radius: float, azimuth_deg: float, elevation_deg_encoded: float):
    az = math.radians(azimuth_deg)
    el = math.radians(wrap_elevation_deg(elevation_deg_encoded))
    x = radius * math.cos(el) * math.cos(az)
    y = radius * math.cos(el) * math.sin(az)
    z = radius * math.sin(el)
    return x, y, z


def build_em32_geometry():
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


def simulate_em32_with_reflections(rcv_positions: np.ndarray, nsample: int):
    rir_all = gpuRIR.simulateRIR(
        L_ROOM, BETA_WALLS, SRC, rcv_positions, NB_IMG, TMAX, FS,
        orV_src=None, orV_rcv=None,
        spkr_pattern="omni",
        mic_pattern="omni",
    )
    rir_all = rir_all[0]
    rir_all = rir_all[:, :nsample].astype(np.float64)
    rir_all[~np.isfinite(rir_all)] = 0.0
    return rir_all


def next_pow2(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def raised_cosine_taper(freqs: np.ndarray, fmax: float, taper_hz: float):
    w = np.ones_like(freqs, dtype=np.float64)
    if fmax is None or fmax <= 0:
        return w
    f1 = max(0.0, float(fmax) - float(taper_hz))
    f2 = float(fmax)
    w[freqs >= f2] = 0.0
    idx = (freqs >= f1) & (freqs < f2)
    if np.any(idx):
        x = (freqs[idx] - f1) / (f2 - f1)
        w[idx] = 0.5 * (1.0 + np.cos(np.pi * x))
    return w


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


def apply_rigid_sphere_sh(
    rir_open: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    radius_a: float,
    fs: int,
    c: float,
    N_sh: int,
    reg_lambda: float,
    f_max: float,
    taper_hz: float,
    pad_factor: int,
    causal_shift_ms: float,
):
    M, T = rir_open.shape
    Nfft = next_pow2(int(T * pad_factor))
    K = Nfft // 2 + 1
    freqs = np.fft.rfftfreq(Nfft, d=1.0 / fs)
    x = (2.0 * np.pi * freqs / c) * radius_a

    W = raised_cosine_taper(freqs, f_max, taper_hz).astype(np.float64)

    H_open = np.fft.rfft(rir_open, n=Nfft, axis=1)
    H_open *= W[None, :]

    Y, order_idx = build_Y_matrix(phi, theta, N_sh)
    Q = Y.shape[1]
    Yh = Y.conj().T
    G = Yh @ Y
    Aop = np.linalg.solve(G + reg_lambda * np.eye(Q), Yh)
    Acoef = Aop @ H_open

    S_order = np.zeros((N_sh + 1, K), dtype=np.complex128)
    for n in range(N_sh + 1):
        S_order[n, :] = rigid_scattering_factor_Sn(n, x)
    S_order *= W[None, :]

    if causal_shift_ms and causal_shift_ms > 0:
        delay_sec = float(causal_shift_ms) / 1000.0
        phase = np.exp(-1j * 2.0 * np.pi * freqs * delay_sec)
        S_order *= phase[None, :]

    S_qk = S_order[order_idx, :]
    H_closed = Y @ (Acoef * S_qk)
    H_closed *= W[None, :]

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

    em32_positions, phi_arr, theta_arr = build_em32_geometry()
    rcv_positions = (RCV_CENTER[None, :].astype(np.float64) + em32_positions).astype(np.float32)

    nsample = int(round(TMAX * FS))
    plotN = int(round(min(TMAX, PLOT_MS / 1000.0) * FS))
    t_ms = np.arange(plotN, dtype=np.float64) / FS * 1000.0

    t_sim0 = time.time()
    rir_open = simulate_em32_with_reflections(rcv_positions, nsample)
    t_sim_ms = (time.time() - t_sim0) * 1000.0

    t_sh0 = time.time()
    rir_rigid = apply_rigid_sphere_sh(
        rir_open, phi_arr, theta_arr, EM32_RADIUS,
        fs=FS, c=C,
        N_sh=N_SH, reg_lambda=REG_LAMBDA,
        f_max=F_MAX, taper_hz=TAPER_HZ,
        pad_factor=PAD_FACTOR, causal_shift_ms=CAUSAL_SHIFT_MS
    )
    t_sh_ms = (time.time() - t_sh0) * 1000.0

    t_total_ms = (time.time() - t_total0) * 1000.0

    n_mics = int(rir_rigid.shape[0])
    per_ch_total_ms = t_total_ms / max(n_mics, 1)

    np.savez_compressed(
        "gpuRIR_EM32_reverb_rigid_32ch.npz",
        rir_rigid=rir_rigid.astype(np.float32),
        fs=np.int32(FS),
        tmax=np.float32(TMAX),
        rcv_center=RCV_CENTER.astype(np.float32),
        em32_positions=em32_positions.astype(np.float32),
        src=SRC.astype(np.float32),
        beta=BETA_WALLS.astype(np.float32),
        nb_img=np.array(NB_IMG, dtype=np.int32),
        timings_ms=np.array([t_sim_ms, t_sh_ms, t_total_ms], dtype=np.float64),
    )
    print("Saved: gpuRIR_EM32_reverb_rigid_32ch.npz")
    print(f"Timings (ms): simulate={t_sim_ms:.2f}, sh_rigid={t_sh_ms:.2f}, total={t_total_ms:.2f}")
    print(f"Per-channel (avg, total): {per_ch_total_ms:.3f} ms/ch")

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
        f"Total runtime: {t_total_ms:.2f} ms\nPer-channel (avg): {per_ch_total_ms:.3f} ms/ch"
    )

    plt.tight_layout()
    plt.savefig("gpuRIR_EM32_reverb_rigid_32ch_overlay.png", dpi=150, bbox_inches="tight")
    print("Saved: gpuRIR_EM32_reverb_rigid_32ch_overlay.png")
    plt.show()


if __name__ == "__main__":
    main()