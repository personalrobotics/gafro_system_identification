"""The pose loop as a six-degree-of-freedom plant, identified from the sweeps' frequency responses, and what the identified plant says about stability.

The pose stage is a PD law whose output is an *acceleration* the controller resolves through its model. If the model were exact and the actuation
instantaneous, the arm's pose would accelerate exactly as commanded and each axis would be the textbook second-order system of `omega_n` and `zeta`.
It is not: the model's inertia is not the arm's (the payload, the CoM of the tool, the link inertials), the torque takes time to act, and the joints have
friction. The fit lumps these into

    pose'' = A u(t - tau) - C pose',        u = Omega^2 (r - pose) - 2 Zeta Omega pose'   (+ the reference's twist and rate with feedforward)

with `A` a full 6x6 matrix (roughly the arm's inverse task-space inertia times the model's: the identity when the model is right; its off-diagonal terms are
the coupling between axes, e.g. y into rx when the tool's centre of mass is off), `tau` the delay of the command (one number, or one per axis), and
`C` a viscous damping per axis. `Omega` and `Zeta` are the stage's gains as they were commanded. The pose is the one the sweeps measure (`trace.DOFS`:
world-frame translations and rotations about world axes through the end effector).

What it is fitted to is the 6x6 closed-loop response of `analysis` -- reference to pose, every output for every excited input (the sweep of each axis gives
one column) -- so it needs a full sweep of all six axes to identify `A`; with fewer, the columns that were not excited stay the identity and the fit says so.
Fitted on the amplitude-scaled response (each axis in units of its own sweep's amplitude), so that metres and radians weigh alike.

The identified plant then gives the closed loop's poles for any gains, with the delay as a Pade approximation: `stability_scan`, `stability_limit`.
"""

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy import signal
from scipy.optimize import least_squares

from gafro_system_identification.trace import DOFS

FEEDFORWARD = ("off", "twist", "twist+rate")
PADE_ORDER = 4
ANGULAR = np.array([False, False, False, True, True, True])


def _expand(value, what):
    """A gain as the configuration states it -- one number, {linear, angular}, or {x, y, z, rx, ry, rz} -- as six numbers in the order of DOFS."""
    if isinstance(value, (int, float)):
        return np.full(6, float(value))
    if isinstance(value, dict) and set(value) == {"linear", "angular"}:
        return np.array([value["linear"]] * 3 + [value["angular"]] * 3, dtype=float)
    if isinstance(value, dict) and set(value) == set(DOFS):
        return np.array([value[d] for d in DOFS], dtype=float)
    raise ValueError(f"{what}: expected a number, {{linear, angular}} or {{{', '.join(DOFS)}}}, got {value!r}")


@dataclass(frozen=True)
class PoseGains:
    """The pose stage's PD gains per degree of freedom: `omega` ((6,) rad/s, > 0) and `zeta` ((6,), > 0), in the order of `trace.DOFS`."""

    omega: np.ndarray
    zeta: np.ndarray

    @classmethod
    def uniform(cls, omega, zeta):
        return cls(np.full(6, float(omega)), np.full(6, float(zeta)))

    @classmethod
    def from_controller(cls, controller, stage="pose"):
        """The gains of stage `stage` of a controller description (`controllers.free` of the configuration), a `pd` law."""
        for entry in controller["stages"]:
            if entry.get("name") == stage:
                pd = entry["law"]["pd"]
                return cls(_expand(pd["natural_frequency"], f"{stage}.natural_frequency"), _expand(pd["damping_ratio"], f"{stage}.damping_ratio"))
        raise ValueError(f"the controller has no stage {stage!r}")

    def with_omega(self, omega):
        return replace(self, omega=np.full(6, float(omega)))

    def __str__(self):
        if np.ptp(self.omega) == 0 and np.ptp(self.zeta) == 0:
            return f"omega {self.omega[0]:g} rad/s, zeta {self.zeta[0]:g}"
        return "omega " + ", ".join(f"{d} {w:g}" for d, w in zip(DOFS, self.omega)) + "; zeta " + ", ".join(f"{d} {z:g}" for d, z in zip(DOFS, self.zeta))


@dataclass
class Excitation:
    """The response to one sweep: of axis `dof` (amplitude `amplitude`, m or rad) over `frequency` (Hz). `response[k, j]` is the complex response of output
    `j` (the order of DOFS) at `frequency[k]`; `quality[k, j]` the share of that output's power the estimate explains."""

    dof: str
    amplitude: float
    frequency: np.ndarray
    response: np.ndarray
    quality: np.ndarray


@dataclass
class ResponseSet:
    """What an experiment measured of the loop, with the feedforward it ran under: the data the fit takes. Saved and loaded as an .npz, so the fit can be
    redone without the cycle log."""

    excitations: list
    feedforward: str = "off"
    reference_period: float = 0.0  # s: the chunks the reference was published in (a first-order hold); 0: none

    @classmethod
    def from_analysis(cls, results, feedforward="off"):
        """From `analysis.analyse` results (`SegmentAnalysis` sequence): every sweep segment with a measured response is one `Excitation` (figure
        eights are ignored). Reads the sweep amplitude from `segment.parameters["sweep"]["amplitude"]` or `segment.path().sweep.amplitude`."""
        excitations, period = [], 0.0
        for result in results:
            if result.segment.kind != "sweep":
                continue
            period = float(result.segment.period)
            params = getattr(result.segment, "parameters", None)
            if isinstance(params, dict) and "sweep" in params and "amplitude" in params["sweep"]:
                amplitude = float(params["sweep"]["amplitude"])
            else:
                amplitude = float(result.segment.path().sweep.amplitude)
            for dof, responses in result.inputs.items():
                own = responses[dof]
                if not own.frequency.size:
                    continue
                excitations.append(Excitation(dof, amplitude, own.frequency.copy(),
                                              np.column_stack([responses[o].response for o in DOFS]),
                                              np.column_stack([responses[o].quality for o in DOFS])))
        if not excitations:
            raise ValueError("no sweep with a measured response in the analysis: nothing to identify the loop from")
        return cls(excitations, feedforward, period)

    def save(self, path):
        arrays = {"feedforward": np.array(self.feedforward), "count": np.array(len(self.excitations)), "reference_period": np.array(self.reference_period)}
        for k, e in enumerate(self.excitations):
            arrays.update({f"dof{k}": np.array(e.dof), f"amplitude{k}": np.array(e.amplitude), f"frequency{k}": e.frequency,
                           f"response{k}": e.response, f"quality{k}": e.quality})
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls([Excitation(str(data[f"dof{k}"]), float(data[f"amplitude{k}"]), data[f"frequency{k}"], data[f"response{k}"], data[f"quality{k}"])
                        for k in range(int(data["count"]))], str(data["feedforward"]),
                       float(data["reference_period"]) if "reference_period" in data else 0.0)

    @property
    def swept(self):
        return sorted({e.dof for e in self.excitations}, key=DOFS.index)

    def scales(self):
        """The amplitude of each degree of freedom (six): the sweeps' own where there is one, else the median of its kind (linear or angular)."""
        amplitudes = np.full(6, np.nan)
        for k, dof in enumerate(DOFS):
            found = [e.amplitude for e in self.excitations if e.dof == dof]
            if found:
                amplitudes[k] = np.median(found)
        for kind in (ANGULAR, ~ANGULAR):
            known = amplitudes[kind][np.isfinite(amplitudes[kind])]
            amplitudes[kind & ~np.isfinite(amplitudes)] = np.median(known) if known.size else 1.0
        return amplitudes


def closed_loop_response(A, tau, C, gains, frequency, feedforward="off", hold=0.0):
    """The model's closed-loop response, reference to pose, at `frequency` (1-D Hz): (n, 6, 6), `[k, output, input]`.

    `A` is (6, 6) in `trace.DOFS` order; `tau` is a scalar or (6,) delay per input channel (s, >= 0); `C` is (6,) viscous damping (1/s);
    `gains` is `PoseGains`; `feedforward` is in `FEEDFORWARD` (`"off"`, `"twist"`, `"twist+rate"`).
    T = (s^2 I + C s + A e^{-s tau} (Omega^2 + 2 Zeta Omega s))^-1 A e^{-s tau} N, with N = Omega^2 for no feedforward, plus 2 Zeta Omega s for the
    reference's twist and s^2 for its rate. `hold` (>= 0, s): the reference reaches the controller as straight chunks of this length between samples of the
    nominal path (one device period), a first-order hold of gain sinc^2(f hold) -- 3% down at 5 Hz for 20 ms, 18% at 12 Hz -- and no phase; the nominal
    path is what the lock-in estimate takes as the input, so the response it measures carries that gain."""
    s = 2j * np.pi * np.asarray(frequency, dtype=float)[:, None, None]
    tau = np.broadcast_to(np.asarray(tau, dtype=float), (6,))
    omega, zeta = gains.omega, gains.zeta
    G = np.asarray(A)[None] * np.exp(-s[:, :, 0] * tau[None, :])[:, None, :]  # the delay on each input channel (column) of A
    feedback = (omega**2)[None, None, :] + 2.0 * (zeta * omega)[None, None, :] * s
    numerator = np.broadcast_to((omega**2)[None, None, :], G.shape).astype(complex)
    if feedforward in ("twist", "twist+rate"):
        numerator = numerator + 2.0 * (zeta * omega)[None, None, :] * s
    if feedforward == "twist+rate":
        numerator = numerator + s**2
    M = s**2 * np.eye(6)[None] + s * np.diag(np.asarray(C, dtype=float))[None] + G * feedback
    response = np.linalg.solve(M, G * numerator)
    if hold:
        response = response * (np.sinc(np.asarray(frequency, dtype=float) * hold) ** 2)[:, None, None]
    return response


@dataclass
class PlantFit:
    """The identified plant: `A` (6x6, with standard errors `A_se`), `tau` (six: the delay of each input channel), `C` (six), the gains and feedforward it was
    fitted under, and how well it fits: `residual`, the RMS of the model's error over the RMS of the measured response (unweighted, in units of each
    sweep's amplitude), `per_input` of it by sweep, `points` the number of complex values that count in the fit."""

    A: np.ndarray
    A_se: np.ndarray
    tau: np.ndarray
    C: np.ndarray
    gains: PoseGains
    feedforward: str
    residual: float
    per_input: dict
    points: int
    swept: list
    diagonal: bool
    hold: float = 0.0

    def response(self, frequency, gains=None):
        return closed_loop_response(self.A, self.tau, self.C, self.gains if gains is None else gains, frequency, self.feedforward, self.hold)

    @property
    def coupling(self):
        """The off-diagonal part of A relative to its diagonal: the largest |A_ji| / sqrt(|A_jj A_ii|), and where."""
        scaled = np.abs(self.A) / np.sqrt(np.abs(np.outer(np.diag(self.A), np.diag(self.A))) + 1e-12)
        np.fill_diagonal(scaled, 0.0)
        j, i = np.unravel_index(np.argmax(scaled), scaled.shape)
        return float(scaled[j, i]), DOFS[j], DOFS[i]


def _uncertainty(measured, quality, floor):
    """The standard deviation of each lock-in estimate, in the units of `measured` (the response scaled by the sweeps' amplitudes).

    The quality q is the share of the output's power the sinusoid explains, so the unexplained power over the explained is (1 - q) / q and the estimate's
    error is about |H| sqrt((1 - q) / q): the noise floor where there is no signal (a channel with no coupling holds only noise: its response is
    *known to be small*, which is what it tells the fit), large where a big response is not clean (a stick-slip, a contact). `floor` bounds it from
    below: the model error and the estimator's bias are never smaller."""
    q = np.clip(quality, 1e-6, 1.0)
    return np.maximum(np.abs(measured) * np.sqrt((1.0 - q) / q), floor)


def fit_plant(data, gains, diagonal=False, per_axis_delay=False, noise_floor=0.02, loss="soft_l1", delay_starts=(0.005, 0.02, 0.05)):
    """Fits the plant to `data` (a `ResponseSet` with `feedforward` in `FEEDFORWARD`) for the pose stage's commanded `gains` (`PoseGains`).

    `diagonal`: no coupling between axes (A diagonal), to show what it is worth. `per_axis_delay`: a delay per axis, not one for all. Every point is
    weighted by the inverse of its uncertainty (`_uncertainty`, at least `noise_floor` > 0 in units of the sweep's amplitude); `loss` is
    `scipy.optimize.least_squares`'s (robust to outliers by default); `delay_starts` are initial delay guesses in seconds (bounded to [0, 0.3] s).
    A column of A whose axis was not swept stays the identity's and its damping zero. The standard errors come from the fit's Jacobian, scaled by the
    fit's own residual, and are optimistic where the errors are not white."""
    ff = data.feedforward
    if ff not in FEEDFORWARD:
        raise ValueError(f"feedforward {ff!r}: expected one of {FEEDFORWARD}")
    scale = data.scales()
    swept = np.array([d in data.swept for d in DOFS])
    free = np.zeros((6, 6), bool)
    for i in range(6):
        if swept[i]:
            free[:, i] = np.eye(6, dtype=bool)[:, i] if diagonal else True
    free_c = swept.copy()
    n_a, n_c, n_t = int(free.sum()), int(free_c.sum()), 6 if per_axis_delay else 1
    ratio = scale[None, :] / scale[:, None]  # (S)_ji = a_i / a_j: the amplitude scaling of a response, output j to input i

    def unpack(p):
        A = np.eye(6)
        A[free] = p[:n_a]
        C = np.zeros(6)
        C[free_c] = p[n_a:n_a + n_c]
        tau = np.broadcast_to(p[n_a + n_c:], (6,))
        return A, C, tau  # A here is in the scaled coordinates

    prepared = []
    for e in data.excitations:
        i = DOFS.index(e.dof)
        measured = e.response * ratio[:, i][None, :]  # (n, 6): output j scaled by a_i / a_j
        prepared.append((i, e.frequency, measured, 1.0 / _uncertainty(measured, e.quality, noise_floor)))

    def residuals(p, parts=False):
        A, C, tau = unpack(p)
        out = []
        for i, frequency, measured, weight in prepared:
            model = closed_loop_response(A, tau, C, gains, frequency, ff, data.reference_period)[:, :, i]
            error = (model - measured) * weight
            out.append(np.concatenate([error.real.ravel(), error.imag.ravel()]))
        return out if parts else np.concatenate(out)

    lower = np.concatenate([np.full(n_a, -10.0), np.zeros(n_c), np.zeros(n_t)])
    upper = np.concatenate([np.full(n_a, 10.0), np.full(n_c, 100.0), np.full(n_t, 0.3)])
    identity = np.eye(6)[free]
    best = None
    for tau0 in delay_starts:
        start = np.concatenate([identity, np.zeros(n_c), np.full(n_t, tau0)])
        result = least_squares(residuals, start, bounds=(lower, upper), loss=loss, f_scale=3.0)
        if best is None or result.cost < best.cost:
            best = result
    A_s, C, tau = unpack(best.x)
    A = A_s * (scale[:, None] / scale[None, :])  # back to SI: A = D A_s D^-1
    raw = best.fun  # in standard deviations
    points = int(sum(np.count_nonzero(w > 0.1 / noise_floor) for _, _, _, w in prepared))  # complex values that count: not drowned in their own uncertainty
    dof = max(len(raw) - len(best.x), 1)
    cov = (raw @ raw / dof) * np.linalg.pinv(best.jac.T @ best.jac)
    se_s = np.zeros((6, 6))
    se_s[free] = np.sqrt(np.maximum(np.diag(cov)[:n_a], 0.0))
    A_se = se_s * (scale[:, None] / scale[None, :])
    errors, powers, per_input = [], [], {}
    for i, frequency, measured, _ in prepared:
        error = closed_loop_response(A_s, tau, C, gains, frequency, ff, data.reference_period)[:, :, i] - measured
        errors.append(np.abs(error).ravel() ** 2)
        powers.append(np.abs(measured).ravel() ** 2)
        per_input.setdefault(DOFS[i], []).append(float(np.sqrt(errors[-1].mean() / max(powers[-1].mean(), 1e-300))))
    per_input = {d: float(np.mean(v)) for d, v in per_input.items()}
    residual = float(np.sqrt(np.concatenate(errors).mean() / max(np.concatenate(powers).mean(), 1e-300)))
    return PlantFit(A, A_se, np.array(tau, dtype=float), C, gains, ff, residual, per_input, points, [d for d, s in zip(DOFS, swept) if s], diagonal, data.reference_period)


def _pade_delay(tau, order=PADE_ORDER):
    """State space (Ap, Bp, Cp, Dp) of the Pade approximation of e^{-s tau}."""
    # e^{-x} ~ P(-x) / P(x), P(x) = sum_k (2n-k)! n! / ((2n)! k! (n-k)!) x^k, with x = s tau
    p = np.array([math.factorial(2 * order - k) * math.factorial(order) / (math.factorial(2 * order) * math.factorial(k) * math.factorial(order - k))
                  for k in range(order + 1)]) * tau ** np.arange(order + 1)
    return signal.tf2ss((p * (-1.0) ** np.arange(order + 1))[::-1], p[::-1])


def closed_loop_poles(A, tau, C, gains):
    """The poles of the loop: states [pose, pose', the delay's Pade states per channel], command u = -Omega^2 pose - 2 Zeta Omega pose' (the reference
    does not move the poles)."""
    tau = np.broadcast_to(np.asarray(tau, dtype=float), (6,))
    sections = [(_pade_delay(t) if t > 1e-6 else None) for t in tau]
    sizes = [0 if s is None else s[0].shape[0] for s in sections]
    n = 12 + sum(sizes)
    F = np.zeros((n, n))
    F[:6, 6:12] = np.eye(6)
    K_x, K_v = -np.diag(gains.omega**2), -np.diag(2.0 * gains.zeta * gains.omega)  # u = K_x x + K_v v
    F[6:12, 6:12] = -np.diag(C)
    offset = 12
    for k, section in enumerate(sections):
        if section is None:  # no delay: the command reaches the acceleration at once
            F[6:12, :6] += np.outer(A[:, k], K_x[k])
            F[6:12, 6:12] += np.outer(A[:, k], K_v[k])
            continue
        Ap, Bp, Cp, Dp = section
        m = Ap.shape[0]
        sl = slice(offset, offset + m)
        F[sl, sl] = Ap
        F[sl, :6] += Bp @ K_x[k][None, :]
        F[sl, 6:12] += Bp @ K_v[k][None, :]
        # ud_k = Cp z + Dp u_k
        F[6:12, sl] += np.outer(A[:, k], Cp[0])
        F[6:12, :6] += float(Dp[0, 0]) * np.outer(A[:, k], K_x[k])
        F[6:12, 6:12] += float(Dp[0, 0]) * np.outer(A[:, k], K_v[k])
        offset += m
    return np.linalg.eigvals(F)


@dataclass(frozen=True)
class Stability:
    """The loop at one set of gains: the largest real part of a pole (negative: stable), the lowest damping ratio among the oscillatory poles and the
    frequency (Hz) of that pole."""

    omega: float
    max_real: float
    zeta: float
    frequency: float

    @property
    def stable(self):
        return self.max_real < 0.0


def stability(plant, gains):
    """Evaluate `Stability` of `plant` (`PlantFit`) at `gains` (`PoseGains`), filtering out stable Padé-approximation artifact poles above `2.5 / max(tau)`."""
    poles = closed_loop_poles(plant.A, plant.tau, plant.C, gains)
    # The Pade approximation of the delay brings fast poles of its own, around 6 / tau and well damped (about 0.5): not the loop's, and the least damped of a
    # well damped loop. The loop's own oscillation is at about pi / (2 tau) at most (1 / (4 tau) Hz): keep what is below 2.5 / tau.
    # An unstable pole is the loop's wherever it is, so only the stable fast poles are dropped (and nothing is, if that leaves none).
    tau = float(np.max(plant.tau))
    keep = (np.abs(poles) < (2.5 / tau if tau > 1e-6 else 2.0 * np.pi * 200.0)) | (poles.real >= 0.0)
    poles = poles[keep] if keep.any() else poles
    oscillatory = poles[(poles.imag > 1e-6)]
    if oscillatory.size:
        zeta = -oscillatory.real / np.abs(oscillatory)
        k = int(np.argmin(zeta))
        z, f = float(zeta[k]), float(oscillatory[k].imag / (2.0 * np.pi))
    else:
        z, f = 1.0, 0.0
    return Stability(float(np.max(gains.omega)), float(np.max(poles.real)), z, f)


def stability_scan(plant, omegas, zeta=None):
    """The loop's stability for each natural frequency in `omegas` (rad/s, the same on every axis) at the damping ratio `zeta` (default: the fitted
    gains')."""
    base = plant.gains if zeta is None else PoseGains.uniform(1.0, zeta)
    return [stability(plant, base.with_omega(w)) for w in omegas]


def stability_limit(plant, zeta=None, omega_max=400.0, step=0.5):
    """The natural frequency (rad/s) above which the loop is unstable at damping ratio `zeta`: the first crossing found scanning up from `step`
    and refined by bisection; None if it is stable up to `omega_max`."""
    scan = np.arange(step, omega_max, step)
    rows = stability_scan(plant, scan, zeta)
    for k, row in enumerate(rows):
        if not row.stable:
            low, high = scan[max(k - 1, 0)], scan[k]  # stable at low, unstable at high
            for _ in range(30):
                middle = 0.5 * (low + high)
                if stability_scan(plant, [middle], zeta)[0].stable:
                    low = middle
                else:
                    high = middle
            return float(high)
    return None


def modes(plant):
    """The eigenvalues and eigenvectors of A, by increasing real part. A mode's eigenvalue is how much of the commanded acceleration it gets (1: as
    designed; the loop's natural frequency in that mode is sqrt(eigenvalue) times the commanded); the vector says which axes move together in it."""
    values, vectors = np.linalg.eig(plant.A)
    order = np.argsort(values.real)
    return values[order], vectors[:, order]


def plot_fit(fit, data, path):
    """Writes the 36 closed-loop responses -- measured (dots) and fitted (line), gain over frequency, one panel per output (row) and swept input
    (column) -- to the PNG `path`. matplotlib is imported here: it is optional."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(6, 6, figsize=(16, 12), sharex=True)
    for e in data.excitations:
        i = DOFS.index(e.dof)
        grid = np.geomspace(e.frequency.min(), e.frequency.max(), 120)
        model = fit.response(grid)[:, :, i]
        for j in range(6):
            axis = axes[j, i]
            axis.loglog(e.frequency, np.maximum(np.abs(e.response[:, j]), 1e-6), ".", color="C0")
            axis.loglog(grid, np.maximum(np.abs(model[:, j]), 1e-6), "-", color="C1")
            axis.set_ylim(1e-3, 3.0)
            axis.grid(True, which="both", alpha=0.25)
    for k, dof in enumerate(DOFS):
        axes[0, k].set_title(f"{dof} swept")
        axes[k, 0].set_ylabel(f"into {dof}")
        axes[5, k].set_xlabel("Hz")
    figure.suptitle(f"measured (dots) and fitted (line) response, |output / reference|; fit error {fit.residual:.1%}")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path
