#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import time
import math
import shutil
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.io import savemat
from scipy.signal import hilbert
from scipy.special import spherical_jn, spherical_yn

try:
    from scipy.special import sph_harm_y

    def eval_sph_harm(m, n, phi, theta):
        return sph_harm_y(n, m, theta, phi)

except ImportError:
    from scipy.special import sph_harm

    def eval_sph_harm(m, n, phi, theta):
        return sph_harm(m, n, phi, theta)


# gpuRIR import

GPU_RIR_SRC = ""
if GPU_RIR_SRC and (GPU_RIR_SRC not in sys.path):
    sys.path.append(GPU_RIR_SRC)

import gpuRIR

gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(False)


# Output

OUT_DIR = Path("em32_rigid_from_gpurir_open")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_MAT_FOR_MATLAB = True
COPY_MAT_TO_CURRENT_DIR = True

MAT_FILENAME = "gpuRIR_EM32_open_and_rigid.mat"


# Basic simulation parameters

FS = 16000
TMAX = 0.500
C = 343.0

NSAMPLE = int(round(TMAX * FS))

L_ROOM = [10.0, 10.0, 10.0]

SRC = np.array([[3.05, 3.37, 1.70]], dtype=np.float32)
RCV_CENTER = np.array([2.60, 4.05, 1.70], dtype=np.float32)

EM32_RADIUS = 0.042

BETA = 0.8 * np.ones(6, np.float32)

NB_IMG = [40, 40, 40]

ORDER_REF_FOR_SMIR = int((NB_IMG[0] - 1) / 2)


# EM32 official theta / phi table
# theta = colatitude in degrees
# phi   = azimuth in degrees

EM32_THETA_PHI_DEG = [
    (69, 0),
    (90, 32),
    (111, 0),
    (90, 328),
    (32, 0),
    (55, 45),
    (90, 69),
    (125, 45),
    (148, 0),
    (125, 315),
    (90, 291),
    (55, 315),
    (21, 91),
    (58, 90),
    (121, 90),
    (159, 89),
    (69, 180),
    (90, 212),
    (111, 180),
    (90, 148),
    (32, 180),
    (55, 225),
    (90, 249),
    (125, 225),
    (148, 180),
    (125, 135),
    (90, 111),
    (55, 135),
    (21, 269),
    (58, 270),
    (122, 270),
    (159, 271),
]


# Paper Eq.(4) rigid-sphere correction settings

N_SH_REQUESTED = 4
F_MAX_HZ = 3800.0
TAPER_HZ = 700.0
PAD_FACTOR = 16
REG_LAMBDA_SH = 1e-5
HANKEL_KIND = "h2"

INV_REG_REL = 2e-2
INV_REG_ABS = 1e-8
RATIO_MAG_CLIP = 8.0

PLOT_TIME_MS = 100.0
DIRECT_HALF_MS = 2.0

SELECTED_CHANNELS_1BASED = [1, 8, 16, 17, 24, 32]

SAVE_EACH_CHANNEL_TXT = True


# Plot style

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# Geometry

def theta_phi_to_cartesian(radius, theta_deg, phi_deg):
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)

    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * math.cos(theta)

    return x, y, z


def build_em32_geometry():
    local_positions = []
    theta_list = []
    phi_list = []
    theta_rad_list = []
    phi_rad_list = []

    for theta_deg, phi_deg in EM32_THETA_PHI_DEG:
        x, y, z = theta_phi_to_cartesian(EM32_RADIUS, theta_deg, phi_deg)

        local_positions.append([x, y, z])
        theta_list.append(theta_deg)
        phi_list.append(phi_deg)
        theta_rad_list.append(math.radians(theta_deg))
        phi_rad_list.append(math.radians(phi_deg) % (2.0 * math.pi))

    local_positions = np.array(local_positions, dtype=np.float64)
    theta_list = np.array(theta_list, dtype=np.float64)
    phi_list = np.array(phi_list, dtype=np.float64)
    theta_rad_list = np.array(theta_rad_list, dtype=np.float64)
    phi_rad_list = np.array(phi_rad_list, dtype=np.float64)

    return local_positions, theta_list, phi_list, theta_rad_list, phi_rad_list


def save_geometry_csv(path, local_positions, rcv_positions, theta_list, phi_list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "channel",
            "theta_deg",
            "phi_deg",
            "local_x_m",
            "local_y_m",
            "local_z_m",
            "world_x_m",
            "world_y_m",
            "world_z_m",
        ])

        for i in range(local_positions.shape[0]):
            writer.writerow([
                i + 1,
                float(theta_list[i]),
                float(phi_list[i]),
                float(local_positions[i, 0]),
                float(local_positions[i, 1]),
                float(local_positions[i, 2]),
                float(rcv_positions[i, 0]),
                float(rcv_positions[i, 1]),
                float(rcv_positions[i, 2]),
            ])


# gpuRIR open / omni simulation

def simulate_omni_rir(fs, tmax, rcv_positions):
    nsample = int(round(tmax * fs))

    rir = gpuRIR.simulateRIR(
        L_ROOM,
        BETA,
        SRC,
        rcv_positions,
        NB_IMG,
        tmax,
        fs,
        orV_src=None,
        orV_rcv=None,
        spkr_pattern="omni",
        mic_pattern="omni",
    )

    rir = rir[0]
    rir = rir[:, :nsample].astype(np.float64)
    rir[~np.isfinite(rir)] = 0.0

    return rir


# rigid correction implementation

def next_pow2(n):
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def raised_cosine_weight(freqs, f_max_hz, taper_hz):
    w = np.ones_like(freqs, dtype=np.float64)

    f2 = float(f_max_hz)
    f1 = max(0.0, f2 - float(taper_hz))

    w[freqs >= f2] = 0.0

    idx = (freqs >= f1) & (freqs < f2)
    if np.any(idx):
        x = (freqs[idx] - f1) / max(f2 - f1, 1e-12)
        w[idx] = 0.5 * (1.0 + np.cos(np.pi * x))

    return w


def build_Y_matrix(phi, theta, N):
    """
    Complex spherical harmonic matrix.

    Y shape:
        M microphones x Q SH coefficients

    Index:
        q = n*(n+1) + m, for m = -n ... n
    """
    M = phi.shape[0]
    Q = (N + 1) ** 2

    Y = np.zeros((M, Q), dtype=np.complex128)
    order_idx = np.zeros(Q, dtype=np.int32)
    degree_idx = np.zeros(Q, dtype=np.int32)

    for n in range(N + 1):
        for m in range(-n, n + 1):
            q = n * (n + 1) + m
            Y[:, q] = eval_sph_harm(m, n, phi, theta)
            order_idx[q] = n
            degree_idx[q] = m

    return Y, order_idx, degree_idx


def hankel_and_derivative(n, x, hankel_kind):
    """
    h_l^(1) = j_l + i y_l
    h_l^(2) = j_l - i y_l

    Paper Eq.(4) writes h_l^(1) under physics Fourier convention.
    For NumPy / signal-processing convention, h2 is the equivalent default.
    """
    j = spherical_jn(n, x)
    jp = spherical_jn(n, x, derivative=True)

    y = spherical_yn(n, x)
    yp = spherical_yn(n, x, derivative=True)

    if hankel_kind.lower() == "h1":
        h = j + 1j * y
        hp = jp + 1j * yp
    elif hankel_kind.lower() == "h2":
        h = j - 1j * y
        hp = jp - 1j * yp
    else:
        raise ValueError("HANKEL_KIND must be 'h1' or 'h2'.")

    return j, jp, h, hp


def paper_formula4_mode_strength(n, ka, hankel_kind):
    """
    Exact paper Eq.(4) mode strength for a rigid sphere:

        b_l(ka) = j_l(ka) - [j_l'(ka) / h_l'(ka)] h_l(ka)

    where h_l is the spherical Hankel function.
    """
    b_open = np.ones_like(ka, dtype=np.complex128)
    b_rigid = np.ones_like(ka, dtype=np.complex128)

    valid = ka > 1e-9

    if not np.any(valid):
        return b_open, b_rigid

    x = ka[valid]

    j, jp, h, hp = hankel_and_derivative(n, x, hankel_kind)

    b_open_x = j.astype(np.complex128)

    hp_safe = hp.copy()
    tiny = np.abs(hp_safe) < 1e-14
    hp_safe[tiny] = 1e-14 + 0.0j

    b_rigid_x = b_open_x - (jp / hp_safe) * h

    b_open[valid] = b_open_x
    b_rigid[valid] = b_rigid_x

    b_open[~np.isfinite(b_open)] = 1.0 + 0.0j
    b_rigid[~np.isfinite(b_rigid)] = 1.0 + 0.0j

    return b_open, b_rigid


def stable_formula4_ratio(n, ka, hankel_kind):
    """
    The open gpuRIR surface response already contains the open-sphere mode
    strength b_open = j_l(ka). To convert it to rigid Eq.(4), we apply:

        ratio_l(f) = b_rigid_l(ka) / b_open_l(ka)

    Near zeros of j_l(ka), this division is regularized.
    """
    b_open, b_rigid = paper_formula4_mode_strength(n, ka, hankel_kind)

    ratio_raw = np.ones_like(ka, dtype=np.complex128)

    valid = np.abs(b_open) > 1e-12
    ratio_raw[valid] = b_rigid[valid] / b_open[valid]

    max_open = max(float(np.max(np.abs(b_open))), 1e-12)
    lam = INV_REG_ABS + INV_REG_REL * max_open

    ratio_reg = b_rigid * np.conj(b_open) / (np.abs(b_open) ** 2 + lam ** 2)

    mag = np.abs(ratio_reg)
    too_large = mag > RATIO_MAG_CLIP

    ratio_reg[too_large] = (
        ratio_reg[too_large] /
        (mag[too_large] + 1e-20) *
        RATIO_MAG_CLIP
    )

    ratio_raw[~np.isfinite(ratio_raw)] = 1.0 + 0.0j
    ratio_reg[~np.isfinite(ratio_reg)] = 1.0 + 0.0j

    return ratio_reg, ratio_raw, b_open, b_rigid


def apply_paper_formula4_rigid_correction(rir_open, phi_rad, theta_rad):
    """
    Main SH-domain Eq.(4) application.

    Input:
        rir_open: M x T open / omni RIR generated by gpuRIR.

    Output:
        rir_rigid: M x T rigid RIR using paper Eq.(4) mode strength.
        rir_open_bandlimited: open RIR in the same effective frequency band.
    """
    M, T = rir_open.shape

    n_sh_max = int(math.floor(math.sqrt(M) - 1))
    n_sh_used = min(N_SH_REQUESTED, n_sh_max)

    Nfft = next_pow2(T * PAD_FACTOR)

    freqs = np.fft.rfftfreq(Nfft, d=1.0 / FS)
    ka = 2.0 * np.pi * freqs * EM32_RADIUS / C

    W = raised_cosine_weight(freqs, F_MAX_HZ, TAPER_HZ)

    H_open = np.fft.rfft(rir_open, n=Nfft, axis=1)

    # Same-band open RIR for fair comparison
    H_open_band = H_open * W[None, :]
    rir_open_bandlimited = np.fft.irfft(H_open_band, n=Nfft, axis=1)
    rir_open_bandlimited = np.real(rir_open_bandlimited[:, :T])
    rir_open_bandlimited[~np.isfinite(rir_open_bandlimited)] = 0.0

    Y, order_idx, degree_idx = build_Y_matrix(phi_rad, theta_rad, n_sh_used)

    Yh = Y.conj().T
    G = Yh @ Y
    G_reg = G + REG_LAMBDA_SH * np.eye(G.shape[0])
    Aop = np.linalg.solve(G_reg, Yh)

    cond_number = float(np.linalg.cond(G_reg))

    # Project band-limited open field to SH coefficients
    A_open = Aop @ H_open_band

    ratio_reg_order = np.ones((n_sh_used + 1, freqs.size), dtype=np.complex128)
    ratio_raw_order = np.ones((n_sh_used + 1, freqs.size), dtype=np.complex128)
    b_open_order = np.ones((n_sh_used + 1, freqs.size), dtype=np.complex128)
    b_rigid_order = np.ones((n_sh_used + 1, freqs.size), dtype=np.complex128)

    for n in range(n_sh_used + 1):
        ratio_reg, ratio_raw, b_open, b_rigid = stable_formula4_ratio(
            n,
            ka,
            HANKEL_KIND
        )

        ratio_reg_order[n, :] = ratio_reg
        ratio_raw_order[n, :] = ratio_raw
        b_open_order[n, :] = b_open
        b_rigid_order[n, :] = b_rigid

    ratio_qf = ratio_reg_order[order_idx, :]

    # Correct SH coefficients order-by-order using paper Eq.(4)
    A_rigid = A_open * ratio_qf

    # Reconstruct microphone signals
    H_rigid = Y @ A_rigid

    rir_rigid = np.fft.irfft(H_rigid, n=Nfft, axis=1)
    rir_rigid = np.real(rir_rigid[:, :T])
    rir_rigid[~np.isfinite(rir_rigid)] = 0.0

    info = {
        "n_sh_used": n_sh_used,
        "nfft": Nfft,
        "freqs": freqs,
        "ka": ka,
        "W": W,
        "Y": Y,
        "order_idx": order_idx,
        "degree_idx": degree_idx,
        "cond_number": cond_number,
        "ratio_reg_order": ratio_reg_order,
        "ratio_raw_order": ratio_raw_order,
        "b_open_order": b_open_order,
        "b_rigid_order": b_rigid_order,
    }

    return rir_rigid, rir_open_bandlimited, info


# Metrics

def compute_direct_toa(rcv_positions):
    src0 = SRC[0].astype(np.float64)
    rcvp = rcv_positions.astype(np.float64)

    dist = np.linalg.norm(rcvp - src0[None, :], axis=1)
    toa_sec = dist / C
    toa_samples_float = toa_sec * FS
    toa_samples_round = np.round(toa_samples_float).astype(int)

    return dist, toa_sec, toa_samples_float, toa_samples_round


def compute_pair_metrics(rir_open_bl, rir_rigid, toa_samples_round):
    M, T = rir_open_bl.shape

    direct_half = int(round(DIRECT_HALF_MS / 1000.0 * FS))
    early_n = min(T, int(round(PLOT_TIME_MS / 1000.0 * FS)))

    rows = []

    for ch in range(M):
        o = rir_open_bl[ch].astype(np.float64)
        r = rir_rigid[ch].astype(np.float64)

        err = r - o

        rms_o = float(np.sqrt(np.mean(o ** 2)) + 1e-20)
        rms_r = float(np.sqrt(np.mean(r ** 2)) + 1e-20)
        rms_e = float(np.sqrt(np.mean(err ** 2)) + 1e-20)

        center = int(toa_samples_round[ch])
        a = max(0, center - direct_half)
        b = min(T, center + direct_half + 1)

        o_dir = o[a:b]
        r_dir = r[a:b]

        po = float(np.max(np.abs(o_dir)) + 1e-20)
        pr = float(np.max(np.abs(r_dir)) + 1e-20)

        if np.std(o_dir) > 1e-20 and np.std(r_dir) > 1e-20:
            corr_dir = float(np.corrcoef(o_dir, r_dir)[0, 1])
        else:
            corr_dir = np.nan

        env_o = np.abs(hilbert(o[:early_n]))
        env_r = np.abs(hilbert(r[:early_n]))

        nmse_env = float(np.sum((env_r - env_o) ** 2) / (np.sum(env_o ** 2) + 1e-20))
        nmse_env_db = float(10.0 * np.log10(nmse_env + 1e-20))

        rows.append({
            "channel": ch + 1,
            "rms_open_bl": rms_o,
            "rms_rigid": rms_r,
            "rms_diff": rms_e,
            "diff_over_open_bl_db": float(20.0 * np.log10(rms_e / rms_o + 1e-20)),
            "rigid_over_open_bl_rms_db": float(20.0 * np.log10(rms_r / rms_o + 1e-20)),
            "direct_peak_open_bl": po,
            "direct_peak_rigid": pr,
            "direct_peak_rigid_over_open_bl_db": float(20.0 * np.log10(pr / po + 1e-20)),
            "direct_corr_open_bl_rigid": corr_dir,
            "early_envelope_nmse_db": nmse_env_db,
        })

    return rows


def write_csv(path, rows):
    if not rows:
        return

    keys = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def median_from_rows(rows, key):
    vals = np.array([float(r[key]) for r in rows if np.isfinite(float(r[key]))])
    if vals.size == 0:
        return np.nan
    return float(np.median(vals))


# Plotting

def save_fig(fig, name):
    png_path = OUT_DIR / f"{name}.png"
    pdf_path = OUT_DIR / f"{name}.pdf"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_formula4_ratio(info):
    freqs = info["freqs"]
    ratio = info["ratio_reg_order"]

    max_f = min(F_MAX_HZ + 500.0, FS / 2)
    mask = freqs <= max_f

    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    for n in range(ratio.shape[0]):
        mag_db = 20.0 * np.log10(np.abs(ratio[n, mask]) + 1e-20)
        ax.plot(freqs[mask], mag_db, label=f"n={n}")

    ax.axvline(F_MAX_HZ - TAPER_HZ, linestyle=":", linewidth=1.0, color="black", alpha=0.8)
    ax.axvline(F_MAX_HZ, linestyle="--", linewidth=1.0, color="black", alpha=0.8)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|b_rigid / b_open| (dB)")
    ax.set_title("Paper Eq.(4) radial correction magnitude")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False)

    save_fig(fig, "fig00_formula4_ratio_magnitude")


def plot_selected_direct_windows(rir_open_bl, rir_rigid, toa_samples_round):
    selected = [ch for ch in SELECTED_CHANNELS_1BASED if 1 <= ch <= rir_open_bl.shape[0]]

    direct_half = int(round(DIRECT_HALF_MS / 1000.0 * FS))

    fig, axes = plt.subplots(len(selected), 1, figsize=(7.5, 1.65 * len(selected)), sharex=False)

    if len(selected) == 1:
        axes = [axes]

    for ax, ch1 in zip(axes, selected):
        ch = ch1 - 1
        center = int(toa_samples_round[ch])

        a = max(0, center - direct_half)
        b = min(rir_open_bl.shape[1], center + direct_half + 1)

        t_ms = (np.arange(a, b) - center) / FS * 1000.0

        o = rir_open_bl[ch, a:b]
        r = rir_rigid[ch, a:b]

        scale = max(np.max(np.abs(o)), np.max(np.abs(r)), 1e-20)

        ax.plot(t_ms, o / scale, label="gpuRIR open, band-limited")
        ax.plot(t_ms, r / scale, label="Eq.(4) rigid")
        ax.axvline(0.0, linestyle="--", linewidth=0.8, color="black", alpha=0.8)
        ax.set_ylabel(f"Ch {ch1}")
        ax.grid(True, linewidth=0.4, alpha=0.35)

        if ch1 == selected[0]:
            ax.set_title("Direct window: open vs paper Eq.(4) rigid")
            ax.legend(frameon=False, loc="best")

    axes[-1].set_xlabel("Time relative to theoretical TOA (ms)")

    save_fig(fig, "fig01_direct_window_openBL_vs_formula4_rigid")


def plot_selected_early_rirs(rir_open_bl, rir_rigid):
    selected = [ch for ch in SELECTED_CHANNELS_1BASED if 1 <= ch <= rir_open_bl.shape[0]]

    n_show = min(rir_open_bl.shape[1], int(round(PLOT_TIME_MS / 1000.0 * FS)))
    t_ms = np.arange(n_show) / FS * 1000.0

    fig, axes = plt.subplots(len(selected), 1, figsize=(7.5, 1.7 * len(selected)), sharex=True)

    if len(selected) == 1:
        axes = [axes]

    for ax, ch1 in zip(axes, selected):
        ch = ch1 - 1

        o = rir_open_bl[ch, :n_show]
        r = rir_rigid[ch, :n_show]

        scale = max(np.max(np.abs(o)), np.max(np.abs(r)), 1e-20)

        ax.plot(t_ms, o / scale, label="gpuRIR open, band-limited")
        ax.plot(t_ms, r / scale, label="Eq.(4) rigid")
        ax.plot(t_ms, (r - o) / scale, linewidth=0.8, alpha=0.75, label="rigid - open")

        ax.set_ylabel(f"Ch {ch1}")
        ax.grid(True, linewidth=0.4, alpha=0.35)

        if ch1 == selected[0]:
            ax.set_title(f"First {PLOT_TIME_MS:.0f} ms: open vs paper Eq.(4) rigid")
            ax.legend(frameon=False, loc="best")

    axes[-1].set_xlabel("Time (ms)")

    save_fig(fig, "fig02_early_rir_openBL_vs_formula4_rigid")


def plot_overlay_all_channels(rir_open_bl, rir_rigid):
    M, T = rir_open_bl.shape

    n_show = min(T, int(round(PLOT_TIME_MS / 1000.0 * FS)))
    t_ms = np.arange(n_show) / FS * 1000.0

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    for ch in range(M):
        o = rir_open_bl[ch, :n_show]
        r = rir_rigid[ch, :n_show]

        scale = max(np.max(np.abs(o)), np.max(np.abs(r)), 1e-20)

        ax.plot(t_ms, o / scale, color="0.75", linewidth=0.45, alpha=0.55)
        ax.plot(t_ms, r / scale, color="0.10", linewidth=0.45, alpha=0.45)

    ax.set_xlim(0, PLOT_TIME_MS)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Locally normalized amplitude")
    ax.set_title("All 32 channels: gray=open band-limited, black=Eq.(4) rigid")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    save_fig(fig, "fig03_all_channels_openBL_vs_formula4_rigid")


# Saving

def save_each_channel_txt(rir_open, rir_open_bl, rir_rigid):
    channel_dir = OUT_DIR / "channels_txt"
    channel_dir.mkdir(parents=True, exist_ok=True)

    t = np.arange(rir_open.shape[1]) / FS
    np.savetxt(channel_dir / "time_s.txt", t, fmt="%.10e")

    for ch in range(rir_open.shape[0]):
        np.savetxt(channel_dir / f"open_raw_ch{ch + 1:02d}.txt", rir_open[ch], fmt="%.10e")
        np.savetxt(channel_dir / f"open_bandlimited_ch{ch + 1:02d}.txt", rir_open_bl[ch], fmt="%.10e")
        np.savetxt(channel_dir / f"formula4_rigid_ch{ch + 1:02d}.txt", rir_rigid[ch], fmt="%.10e")


def save_mat_for_matlab(
    mat_path,
    rir_open,
    rir_open_bl,
    rir_rigid,
    local_positions,
    rcv_positions,
    theta_deg,
    phi_deg,
    theta_rad,
    phi_rad,
    distance_each,
    toa_sec,
    toa_samples_float,
    toa_samples_round,
    runtime_info,
    formula4_info
):
    mat_dict = {
        "rir_open": rir_open.astype(np.float64),
        "rir_omni": rir_open.astype(np.float64),
        "rir_open_bandlimited": rir_open_bl.astype(np.float64),
        "rir_rigid_formula4": rir_rigid.astype(np.float64),
        "rir_rigid": rir_rigid.astype(np.float64),

        "fs": np.array([[FS]], dtype=np.float64),
        "tmax": np.array([[TMAX]], dtype=np.float64),
        "c": np.array([[C]], dtype=np.float64),

        "L_room": np.array(L_ROOM, dtype=np.float64),
        "src": SRC.astype(np.float64),
        "rcv_center": RCV_CENTER.astype(np.float64),

        "rcv_positions": rcv_positions.astype(np.float64),
        "local_positions": local_positions.astype(np.float64),

        "theta_deg": theta_deg.reshape(1, -1).astype(np.float64),
        "phi_deg": phi_deg.reshape(1, -1).astype(np.float64),
        "theta_rad": theta_rad.reshape(1, -1).astype(np.float64),
        "phi_rad": phi_rad.reshape(1, -1).astype(np.float64),

        "em32_radius": np.array([[EM32_RADIUS]], dtype=np.float64),

        "beta": BETA.reshape(1, -1).astype(np.float64),
        "nb_img": np.array(NB_IMG, dtype=np.float64).reshape(1, -1),
        "order_ref": np.array([[ORDER_REF_FOR_SMIR]], dtype=np.float64),

        "n_sh_requested": np.array([[N_SH_REQUESTED]], dtype=np.float64),
        "n_sh_used": np.array([[formula4_info["n_sh_used"]]], dtype=np.float64),

        "f_max_hz": np.array([[F_MAX_HZ]], dtype=np.float64),
        "taper_hz": np.array([[TAPER_HZ]], dtype=np.float64),

        "hankel_kind_code": np.array([[1 if HANKEL_KIND.lower() == "h1" else 2]], dtype=np.float64),
        "pad_factor": np.array([[PAD_FACTOR]], dtype=np.float64),
        "reg_lambda_sh": np.array([[REG_LAMBDA_SH]], dtype=np.float64),
        "inv_reg_rel": np.array([[INV_REG_REL]], dtype=np.float64),
        "inv_reg_abs": np.array([[INV_REG_ABS]], dtype=np.float64),
        "ratio_mag_clip": np.array([[RATIO_MAG_CLIP]], dtype=np.float64),

        "distance_each_m": distance_each.reshape(1, -1).astype(np.float64),
        "toa_each_sec": toa_sec.reshape(1, -1).astype(np.float64),
        "toa_each_samples_float": toa_samples_float.reshape(1, -1).astype(np.float64),
        "toa_each_samples_round": toa_samples_round.reshape(1, -1).astype(np.float64),

        "freqs": formula4_info["freqs"].reshape(1, -1).astype(np.float64),
        "ka": formula4_info["ka"].reshape(1, -1).astype(np.float64),
        "W": formula4_info["W"].reshape(1, -1).astype(np.float64),

        "ratio_reg_order": formula4_info["ratio_reg_order"],
        "ratio_raw_order": formula4_info["ratio_raw_order"],
        "b_open_order": formula4_info["b_open_order"],
        "b_rigid_order": formula4_info["b_rigid_order"],

        "sh_condition_number": np.array([[formula4_info["cond_number"]]], dtype=np.float64),

        "runtime_ms": np.array(
            [[runtime_info["open_ms"], runtime_info["formula4_ms"], runtime_info["total_ms"]]],
            dtype=np.float64
        ),

        "rigid_correction_enabled": np.array([[1]], dtype=np.float64),
        "formula4_enabled": np.array([[1]], dtype=np.float64),
    }

    savemat(mat_path, mat_dict, do_compression=True)


def write_summary(
    summary_path,
    runtime_info,
    metrics_rows,
    formula4_info,
    mat_path
):
    diff_db = median_from_rows(metrics_rows, "diff_over_open_bl_db")
    peak_db = median_from_rows(metrics_rows, "direct_peak_rigid_over_open_bl_db")
    corr = median_from_rows(metrics_rows, "direct_corr_open_bl_rigid")
    env_nmse = median_from_rows(metrics_rows, "early_envelope_nmse_db")

    text = []

    text.append("Step 2: Paper Eq.(4) rigid-sphere correction applied to gpuRIR open RIR")
    text.append("=" * 80)
    text.append("")
    text.append("What this script does")
    text.append("-" * 80)
    text.append("1. Generate EM32 32-channel open / omni RIR with gpuRIR.")
    text.append("2. Project the band-limited open field to the SH domain.")
    text.append("3. Compute the rigid-sphere mode strength using paper Eq.(4):")
    text.append("       b_l(ka) = j_l(ka) - [j_l'(ka) / h_l'(ka)] h_l(ka)")
    text.append("4. Convert open mode strength j_l(ka) to rigid mode strength using b_rigid / b_open.")
    text.append("5. Reconstruct 32-channel Eq.(4) rigid RIR.")
    text.append("")
    text.append("Important limitation")
    text.append("-" * 80)
    text.append("This is a correct application of paper Eq.(4) as an SH-domain radial correction")
    text.append("to the existing gpuRIR open RIR. It is not the full paper Eq.(10) image-source")
    text.append("summation, because each reflection path is not processed separately here.")
    text.append("")
    text.append("Simulation setup")
    text.append("-" * 80)
    text.append(f"Room: {L_ROOM} m")
    text.append(f"Source: {SRC[0].tolist()} m")
    text.append(f"Receiver center: {RCV_CENTER.tolist()} m")
    text.append(f"EM32 radius: {EM32_RADIUS} m")
    text.append(f"FS: {FS} Hz")
    text.append(f"TMAX: {TMAX} s")
    text.append(f"Beta: {BETA.tolist()}")
    text.append(f"NB_IMG: {NB_IMG}")
    text.append("")
    text.append("Eq.(4) settings")
    text.append("-" * 80)
    text.append(f"N_SH requested: {N_SH_REQUESTED}")
    text.append(f"N_SH used: {formula4_info['n_sh_used']}")
    text.append(f"F_MAX_HZ: {F_MAX_HZ}")
    text.append(f"TAPER_HZ: {TAPER_HZ}")
    text.append(f"HANKEL_KIND: {HANKEL_KIND}")
    text.append(f"SH condition number: {formula4_info['cond_number']:.6e}")
    text.append(f"REG_LAMBDA_SH: {REG_LAMBDA_SH}")
    text.append(f"INV_REG_REL: {INV_REG_REL}")
    text.append(f"RATIO_MAG_CLIP: {RATIO_MAG_CLIP}")
    text.append("")
    text.append("Open vs Eq.(4) rigid metrics, median")
    text.append("-" * 80)
    text.append(f"RMS difference rigid-openBL over openBL: {diff_db:.3f} dB")
    text.append(f"Direct peak gain rigid/openBL: {peak_db:.3f} dB")
    text.append(f"Direct-window correlation: {corr:.4f}")
    text.append(f"Early envelope NMSE: {env_nmse:.3f} dB")
    text.append("")
    text.append("Runtime")
    text.append("-" * 80)
    text.append(f"gpuRIR open simulation: {runtime_info['open_ms']:.3f} ms")
    text.append(f"Paper Eq.(4) correction: {runtime_info['formula4_ms']:.3f} ms")
    text.append(f"Total: {runtime_info['total_ms']:.3f} ms")
    text.append("")
    text.append("Files")
    text.append("-" * 80)
    text.append(f"MAT file: {mat_path}")
    text.append("Important fields:")
    text.append("  rir_open")
    text.append("  rir_open_bandlimited")
    text.append("  rir_rigid_formula4 / rir_rigid")
    text.append("")

    summary = "\n".join(text)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print("")
    print(summary)
    print("")
    print(f"Saved: {summary_path}")


# Main

def main():
    total_t0 = time.time()

    local_positions, theta_deg, phi_deg, theta_rad, phi_rad = build_em32_geometry()
    rcv_positions = (RCV_CENTER[None, :].astype(np.float64) + local_positions).astype(np.float32)

    geometry_csv = OUT_DIR / "em32_geometry_from_theta_phi_table.csv"
    save_geometry_csv(geometry_csv, local_positions, rcv_positions, theta_deg, phi_deg)
    print(f"Saved: {geometry_csv}")

    print("=" * 80)
    print("Step 2: gpuRIR open + paper Eq.(4) rigid-sphere correction")
    print("=" * 80)
    print(f"FS = {FS}")
    print(f"TMAX = {TMAX}")
    print(f"Room = {L_ROOM}")
    print(f"Source = {SRC[0].tolist()}")
    print(f"Receiver center = {RCV_CENTER.tolist()}")
    print(f"EM32 radius = {EM32_RADIUS}")
    print(f"Beta = {BETA.tolist()}")
    print(f"NB_IMG = {NB_IMG}")
    print(f"N_SH_REQUESTED = {N_SH_REQUESTED}")
    print(f"F_MAX_HZ = {F_MAX_HZ}")
    print(f"TAPER_HZ = {TAPER_HZ}")
    print(f"HANKEL_KIND = {HANKEL_KIND}")
    print("")

    t_open0 = time.time()
    rir_open = simulate_omni_rir(FS, TMAX, rcv_positions)
    open_ms = (time.time() - t_open0) * 1000.0

    t_f4_0 = time.time()
    rir_rigid, rir_open_bl, formula4_info = apply_paper_formula4_rigid_correction(
        rir_open,
        phi_rad,
        theta_rad
    )
    formula4_ms = (time.time() - t_f4_0) * 1000.0

    total_ms = (time.time() - total_t0) * 1000.0

    runtime_info = {
        "open_ms": float(open_ms),
        "formula4_ms": float(formula4_ms),
        "total_ms": float(total_ms),
    }

    print("Runtime")
    print("-" * 80)
    print(f"gpuRIR open simulation: {open_ms:.3f} ms")
    print(f"Paper Eq.(4) correction: {formula4_ms:.3f} ms")
    print(f"Total: {total_ms:.3f} ms")
    print("")

    distance_each, toa_sec, toa_samples_float, toa_samples_round = compute_direct_toa(rcv_positions)

    metrics_rows = compute_pair_metrics(rir_open_bl, rir_rigid, toa_samples_round)
    metrics_csv = OUT_DIR / "open_bandlimited_vs_formula4_rigid_metrics.csv"
    write_csv(metrics_csv, metrics_rows)
    print(f"Saved: {metrics_csv}")

    if SAVE_MAT_FOR_MATLAB:
        mat_path = OUT_DIR / MAT_FILENAME

        save_mat_for_matlab(
            mat_path,
            rir_open,
            rir_open_bl,
            rir_rigid,
            local_positions,
            rcv_positions,
            theta_deg,
            phi_deg,
            theta_rad,
            phi_rad,
            distance_each,
            toa_sec,
            toa_samples_float,
            toa_samples_round,
            runtime_info,
            formula4_info
        )

        print(f"Saved MATLAB MAT: {mat_path}")

        if COPY_MAT_TO_CURRENT_DIR:
            shutil.copyfile(mat_path, MAT_FILENAME)
            print(f"Copied MAT file to current directory: {MAT_FILENAME}")

    if SAVE_EACH_CHANNEL_TXT:
        save_each_channel_txt(rir_open, rir_open_bl, rir_rigid)

    plot_formula4_ratio(formula4_info)
    plot_selected_direct_windows(rir_open_bl, rir_rigid, toa_samples_round)
    plot_selected_early_rirs(rir_open_bl, rir_rigid)
    plot_overlay_all_channels(rir_open_bl, rir_rigid)

    npz_path = OUT_DIR / "open_and_formula4_rigid_data.npz"
    np.savez(
        npz_path,
        rir_open=rir_open,
        rir_open_bandlimited=rir_open_bl,
        rir_rigid_formula4=rir_rigid,
        fs=FS,
        tmax=TMAX,
        c=C,
        L_room=np.array(L_ROOM, dtype=np.float64),
        src=SRC.astype(np.float64),
        rcv_center=RCV_CENTER.astype(np.float64),
        rcv_positions=rcv_positions.astype(np.float64),
        local_positions=local_positions.astype(np.float64),
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        theta_rad=theta_rad,
        phi_rad=phi_rad,
        em32_radius=EM32_RADIUS,
        beta=BETA.astype(np.float64),
        nb_img=np.array(NB_IMG, dtype=np.int32),
        f_max_hz=F_MAX_HZ,
        taper_hz=TAPER_HZ,
        n_sh_used=formula4_info["n_sh_used"],
        freqs=formula4_info["freqs"],
        ka=formula4_info["ka"],
        W=formula4_info["W"],
        ratio_reg_order=formula4_info["ratio_reg_order"],
        ratio_raw_order=formula4_info["ratio_raw_order"],
        b_open_order=formula4_info["b_open_order"],
        b_rigid_order=formula4_info["b_rigid_order"],
        runtime_open_ms=open_ms,
        runtime_formula4_ms=formula4_ms,
        runtime_total_ms=total_ms,
    )
    print(f"Saved: {npz_path}")

    summary_path = OUT_DIR / "formula4_rigid_summary.txt"
    write_summary(
        summary_path,
        runtime_info,
        metrics_rows,
        formula4_info,
        OUT_DIR / MAT_FILENAME
    )

    print("")
    print("=" * 80)
    print("Done.")
    print(f"Results saved in: {OUT_DIR}")
    print(f"Main MAT file: {OUT_DIR / MAT_FILENAME}")
    print("=" * 80)


if __name__ == "__main__":
    main()