"""SPSA (simultaneous-perturbation stochastic approximation) of N-joint direct_torque friction scales (e.g. UR, OpenArm).

Chirp metric: free-space response_pose (off-axis, quality, gain, delay).
Path metric: figure-eight PathRecorder summary (RMS, lag) plus y→rx and rotation as vibration proxies.

Scales live in [0, 1]; SPSA steps in unconstrained logit space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gafro_system_identification.analysis import _low_frequency_delay
from gafro_system_identification.response import fit_second_order
from gafro_system_identification.trace import coupling

N_JOINTS = 6
N_PARAMS = 2 * N_JOINTS  # viscous then coulomb (default for 6-DOF callers; pass n_joints to unpack for N-DOF)


def pack(viscous, coulomb):
    """Concatenate equal-length 1-D arrays `viscous` and `coulomb` (in [0, 1]) into a single 2N-element parameter vector."""
    v = np.asarray(viscous, dtype=float)
    c = np.asarray(coulomb, dtype=float)
    if v.shape != c.shape or v.ndim != 1:
        raise ValueError(f"viscous and coulomb must be 1-D arrays of equal length, got {v.shape} and {c.shape}")
    return np.concatenate([v, c])


def unpack(theta, n_joints=None):
    """Split a 2N-element parameter vector `theta` into `(viscous, coulomb)` 1-D arrays of length `n_joints` (inferred as `len(theta) // 2` if None)."""
    theta = np.asarray(theta, dtype=float).ravel()
    if n_joints is None:
        if theta.size % 2 != 0:
            raise ValueError(f"theta length must be even when n_joints is omitted, got {theta.size}")
        n_joints = theta.size // 2
    elif theta.size != 2 * n_joints:
        raise ValueError(f"expected {2 * n_joints} parameters for {n_joints} joints, got {theta.size}")
    return theta[:n_joints].copy(), theta[n_joints:].copy()


def clip_scales(theta, lo=0.0, hi=1.0):
    """Elementwise clip `theta` (array-like of friction scales) to `[lo, hi]` (default `[0.0, 1.0]`)."""
    return np.clip(np.asarray(theta, dtype=float), lo, hi)


def clip_around(theta, center, radius=0.25):
    """Keep each scale in `theta` within `±radius` (>= 0) of `center` (same shape), clamped to `[0, 1]`."""
    center = np.asarray(center, dtype=float)
    lo = np.maximum(0.0, center - radius)
    hi = np.minimum(1.0, center + radius)
    return np.clip(np.asarray(theta, dtype=float), lo, hi)


def to_logit(theta, eps=1e-4):
    """Map `[0, 1]` scales `theta` to unconstrained `R` for SPSA, clamping away from `{0, 1}` by `eps` (`0 < eps < 0.5`)."""
    u = np.clip(np.asarray(theta, dtype=float), eps, 1.0 - eps)
    return np.log(u / (1.0 - u))


def from_logit(z):
    """Inverse of `to_logit`: sigmoid mapping unconstrained logit array `z` back to `(0, 1)`."""
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class TrialMetrics:
    gain: float
    wn: float
    zeta: float
    delay_ms: float
    low_f_delay_ms: float
    bandwidth_hz: float | None
    quality: float
    off_axis: float
    off_axis_dof: str | None
    cost: float


@dataclass
class PathTrialMetrics:
    position_rms_mm: float
    position_peak_mm: float
    lag_ms: float
    rotation_rms_deg: float
    rotation_peak_deg: float
    y_to_rx: float
    cost: float
    rejected: bool


def metrics_from_results(results, dof="z",
                         w_off=1.0, w_quality=0.25, w_gain=0.1, w_delay=0.1, delay_ref_ms=50.0):
    """Scalar cost from an `analysis` result list (`SegmentAnalysis` items) containing a sweep of `dof` (in `trace.DOFS`, default `"z"`).

    Weights `w_*` are >= 0 and `delay_ref_ms` > 0."""
    for result in results:
        if result.tones:
            continue
        if dof not in result.inputs:
            continue
        responses = result.inputs[dof]
        own = responses[dof]
        if own.frequency.size == 0:
            raise ValueError(f"no frequencies analysed for {dof}")
        try:
            fit = fit_second_order(own)
        except ValueError:
            fit = None
        cross = coupling(responses, dof)
        worst = max(cross, key=cross.get) if cross else None
        off = float(cross[worst]) if worst is not None else 0.0
        quality = float(own.quality.mean())
        gain = float(fit.gain) if fit else 1.0
        low_f = _low_frequency_delay(own)
        cost = (w_off * off ** 2
                + w_quality * (1.0 - quality) ** 2
                + w_gain * (gain - 1.0) ** 2
                + w_delay * (low_f / delay_ref_ms) ** 2)
        return TrialMetrics(
            gain=gain,
            wn=float(fit.wn) if fit else float("nan"),
            zeta=float(fit.zeta) if fit else float("nan"),
            delay_ms=float(fit.delay * 1e3) if fit else float("nan"),
            low_f_delay_ms=low_f,
            bandwidth_hz=own.bandwidth(),
            quality=quality,
            off_axis=off,
            off_axis_dof=worst,
            cost=float(cost),
        )
    raise ValueError(f"no sweep of {dof!r} in the analysis results")


def y_to_rx_from_results(results):
    """Median |H| of the figure-eight y tone into rx, or 0 if tones are missing."""
    for result in results:
        if not result.tones or "y" not in result.inputs:
            continue
        cross = coupling(result.inputs["y"], "y")
        return float(cross.get("rx", 0.0))
    return 0.0


def path_cost(summary, y_to_rx=0.0,
              pos_ref_mm=2.0, lag_ref_ms=100.0, rot_ref_deg=0.5, coup_ref=0.35,
              w_rot=1.5, w_coup=1.0,
              rot_peak_reject_deg=1.2, pos_peak_reject_mm=8.0):
    """Cost from a `PathRecorder` figures `summary` dict (keys `position_rms_mm`, `position_peak_mm`, `lag_ms`, `rotation_rms_deg`,
    `rotation_peak_deg`; or empty/falsy for a failed run) plus `y_to_rx` (>= 0, rad/m). Positive reference scales `*_ref_*`; hard-rejects vibrating trials."""
    if not summary:
        return PathTrialMetrics(
            position_rms_mm=float("nan"), position_peak_mm=float("nan"), lag_ms=float("nan"),
            rotation_rms_deg=float("nan"), rotation_peak_deg=float("nan"), y_to_rx=float(y_to_rx),
            cost=1e3, rejected=True,
        )
    pos_rms = float(summary["position_rms_mm"])
    pos_peak = float(summary["position_peak_mm"])
    lag = float(summary["lag_ms"])
    rot_rms = float(summary["rotation_rms_deg"])
    rot_peak = float(summary["rotation_peak_deg"])
    y_to_rx = float(y_to_rx)
    rejected = rot_peak > rot_peak_reject_deg or pos_peak > pos_peak_reject_mm
    pos = pos_rms / pos_ref_mm
    lag_n = max(lag, 0.0) / lag_ref_ms
    rot = rot_rms / rot_ref_deg
    coup = y_to_rx / coup_ref
    cost = pos ** 2 + lag_n ** 2 + w_rot * rot ** 2 + w_coup * coup ** 2
    if rejected:
        cost += 100.0
    return PathTrialMetrics(
        position_rms_mm=pos_rms, position_peak_mm=pos_peak, lag_ms=lag,
        rotation_rms_deg=rot_rms, rotation_peak_deg=rot_peak, y_to_rx=y_to_rx,
        cost=float(cost), rejected=rejected,
    )


def spsa_perturb(z, k, c=0.08, gamma=0.101, seed=None):
    """Draw Rademacher perturbation `delta` in {-1, +1}^d at iteration `k` (>= 0) and return `(z_plus, z_minus, ck, delta)` in logit space."""
    z = np.asarray(z, dtype=float)
    rng = np.random.default_rng(seed)
    ck = c / ((k + 1) ** gamma)
    delta = rng.choice([-1.0, 1.0], size=z.shape)
    return z + ck * delta, z - ck * delta, ck, delta


def spsa_step(z, cost_plus, cost_minus, k, a=0.15, c=0.08, A=5.0, alpha=0.602, gamma=0.101, seed=None, delta=None):
    """One SPSA update in logit space on 1-D vector `z` at iteration `k` (>= 0). `cost_plus` and `cost_minus` are scalar losses `L(z + c_k Δ)` and
    `L(z - c_k Δ)`; pass the `delta` array from `spsa_perturb` (or the same `seed` used when perturbing). `a, c, A, alpha, gamma` > 0 are the
    standard SPSA gain-schedule hyperparameters."""
    z = np.asarray(z, dtype=float)
    ck = c / ((k + 1) ** gamma)
    ak = a / ((k + 1 + A) ** alpha)
    if delta is None:
        rng = np.random.default_rng(seed)
        delta = rng.choice([-1.0, 1.0], size=z.shape)
    else:
        delta = np.asarray(delta, dtype=float)
    grad = (cost_plus - cost_minus) / (2.0 * ck * delta)
    return z - ak * grad, ck, ak, delta


def format_scales(viscous, coulomb):
    v = ", ".join(f"{x:.3f}" for x in viscous)
    c = ", ".join(f"{x:.3f}" for x in coulomb)
    return f"viscous: [{v}]\n    coulomb: [{c}]"


def yaml_friction_block(viscous, coulomb, indent=4):
    pad = " " * indent
    lines = [f"{pad}friction_compensation:"]
    lines.append(f"{pad}  viscous: [{', '.join(f'{x:.4f}' for x in viscous)}]")
    lines.append(f"{pad}  coulomb: [{', '.join(f'{x:.4f}' for x in coulomb)}]")
    return "\n".join(lines)
