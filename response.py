"""Frequency response of a closed loop, measured: a swept sine in, a lock-in estimate of gain and phase out.

Nothing here knows about a robot. `Sweep` is the excitation (a log chirp, eased in and out); `frequency_response` takes
two time-aligned signals sampled together -- the reference as it was commanded and the output as it was measured, a
control cycle's samples -- and estimates the response between them at a grid of frequencies; `fit_second_order` fits the
textbook second-order system with a delay to what was measured.

The estimate is a lock-in analysis on the chirp's own phase. For each analysis frequency, a window around the time the
chirp passes it, a few cycles wide and at most half an octave of the sweep, has a cosine and a sine of the *chirp's
phase* (not of one fixed frequency, which the sweep leaves within the window), a constant and a drift fitted to the
input and to the output by weighted least squares; the ratio of the two complex amplitudes is the response there. The
share of the output's power that the fit explains says how much to believe it: noise and distortion -- a stick-slip, a
contact that lets go -- are what it does not explain.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Sweep:
    """`amplitude` times a sine whose frequency rises geometrically from `f_start` to `f_end` Hz over `duration` seconds,
    with `ramp` seconds easing in before and out after (the frequency holds at its end value there, the amplitude goes
    to zero), so the signal starts and ends at rest. Time is the signal's own, from 0 to `total`."""

    amplitude: float
    f_start: float = 0.2
    f_end: float = 5.0
    duration: float = 120.0
    ramp: float = 3.0

    def __post_init__(self):
        if self.amplitude <= 0.0 or not 0.0 < self.f_start < self.f_end or self.duration <= 0.0 or self.ramp <= 0.0:
            raise ValueError(f"need positive amplitude, 0 < f_start < f_end, and positive duration and ramp: {self}")

    @property
    def total(self):
        return self.duration + 2.0 * self.ramp

    @property
    def _ratio(self):
        return self.f_end / self.f_start

    def _chirp_time(self, t):
        return np.clip(np.asarray(t, dtype=float) - self.ramp, 0.0, self.duration)

    def frequency(self, t):
        """The chirp's instantaneous frequency in Hz at time `t`."""
        return self.f_start * self._ratio ** (self._chirp_time(t) / self.duration)

    def phase(self, t):
        """The chirp's phase in radians at time `t`, the integral of 2 pi times its frequency: zero at the start, turning at
        `f_start` through the easing in and at `f_end` through the easing out."""
        t = np.asarray(t, dtype=float)
        rate = np.log(self._ratio) / self.duration
        tau = self._chirp_time(t)
        head = 2.0 * np.pi * self.f_start * np.minimum(t, self.ramp)
        chirp = 2.0 * np.pi * self.f_start * (self._ratio ** (tau / self.duration) - 1.0) / rate
        tail = 2.0 * np.pi * self.f_end * np.maximum(t - self.ramp - self.duration, 0.0)
        return head + chirp + tail

    def time_at_frequency(self, f):
        """The time at which the chirp passes `f` Hz."""
        return self.ramp + self.duration * np.log(np.asarray(f, dtype=float) / self.f_start) / np.log(self._ratio)

    def envelope(self, t):
        t = np.asarray(t, dtype=float)
        rising = np.clip(t / self.ramp, 0.0, 1.0)
        falling = np.clip((self.total - t) / self.ramp, 0.0, 1.0)
        return (3.0 * rising**2 - 2.0 * rising**3) * (3.0 * falling**2 - 2.0 * falling**3)

    def value(self, t):
        """The signal at time `t`; zero outside [0, total]."""
        t = np.asarray(t, dtype=float)
        inside = (t > 0.0) & (t < self.total)
        return np.where(inside, self.amplitude * self.envelope(t) * np.sin(self.phase(t)), 0.0)

    def peak_speed(self):
        """An upper bound on the signal's rate of change: amplitude * 2 pi * f_end."""
        return self.amplitude * 2.0 * np.pi * self.f_end


@dataclass(frozen=True)
class FrequencyResponse:
    """The response at `frequency` (Hz): complex `response` (output over input) and `quality` in [0, 1], the share of the
    output's power in the analysis window that the fit explains (1: a clean sinusoid at the chirp's phase)."""

    frequency: np.ndarray
    response: np.ndarray
    quality: np.ndarray

    @property
    def gain(self):
        return np.abs(self.response)

    @property
    def gain_db(self):
        return 20.0 * np.log10(np.maximum(self.gain, 1e-12))

    @property
    def phase(self):
        """The phase in radians, unwrapped along the frequency from the first point (a lag is negative)."""
        return np.unwrap(np.angle(self.response))

    @property
    def phase_deg(self):
        return np.degrees(self.phase)

    def delay(self):
        """The delay in seconds that the phase alone would give at each frequency: -phase / (2 pi f)."""
        return -self.phase / (2.0 * np.pi * self.frequency)

    def bandwidth(self, level_db=-3.0, reference=None):
        """The first frequency (Hz) at which the gain has fallen `level_db` (< 0, dB) from `reference` (default: the gain at the lowest
        frequencies, the median of the first three points), found by interpolating the gain in dB; None if `frequency` is empty or never falls below."""
        if self.frequency.size == 0:
            return None
        reference = float(np.median(self.gain[:3])) if reference is None else reference
        below = self.gain_db < 20.0 * np.log10(reference) + level_db
        if not below.any():
            return None
        k = int(np.argmax(below))
        if k == 0:
            return float(self.frequency[0])
        f0, f1 = np.log(self.frequency[k - 1]), np.log(self.frequency[k])
        g0, g1 = self.gain_db[k - 1], self.gain_db[k]
        target = 20.0 * np.log10(reference) + level_db
        return float(np.exp(f0 + (f1 - f0) * (target - g0) / (g1 - g0)))

    def table(self):
        lines = [f"{'Hz':>7s} {'gain dB':>8s} {'phase deg':>10s} {'delay ms':>9s} {'quality':>8s}"]
        for f, g, p, d, q in zip(self.frequency, self.gain_db, self.phase_deg, self.delay() * 1e3, self.quality):
            lines.append(f"{f:7.3f} {g:8.2f} {p:10.1f} {d:9.1f} {q:8.2f}")
        return "\n".join(lines)


def _phasor(t, x, phase, weights):
    """x ~ a cos(phase) + b sin(phase) + c + d (t - t_mean), weighted least squares: the complex amplitude C = a - j b
    (so x ~ Re{C e^{j phase}}) and the share of the weighted power that the sinusoid explains."""
    centre = np.sum(weights * t) / np.sum(weights)
    basis = np.column_stack([np.cos(phase), np.sin(phase), np.ones_like(t), t - centre])
    root = np.sqrt(weights)
    coefficients, *_ = np.linalg.lstsq(basis * root[:, None], x * root, rcond=None)
    sinusoid = basis[:, :2] @ coefficients[:2]
    residual = x - basis @ coefficients
    explained = float(np.sum(weights * sinusoid**2))
    unexplained = float(np.sum(weights * residual**2))
    return complex(coefficients[0], -coefficients[1]), explained / max(explained + unexplained, 1e-300)


def frequency_response(t, u, y, sweep, t0=0.0, frequencies=None, points=24, cycles=4.0, octave_span=0.5, minimum_samples=20,
                       sweep_time=None):
    """The response of `y` to `u` at a grid of frequencies, from samples taken together at times `t` (1-D monotonic seconds, any clock).

    `u` (excitation as received) and `y` (output) are 1-D arrays with the same shape as `t`; `sweep` is a `Sweep` instance (`0 < f_start < f_end`,
    positive `duration` and `ramp`), whose time is `t - t0` when `sweep_time` is None. `frequencies` (Hz, > 0): where to estimate, default `points`
    geometrically spaced points a quarter-octave in from the sweep's ends. Each window is `cycles` (> 0) cycles wide at its frequency but at most
    `octave_span` (> 0) octaves of the sweep, and holds at least `minimum_samples` (>= 4 for the 4-parameter `_phasor` fit) or the point is dropped.
    The samples need not be evenly spaced. `sweep_time`: 1-D array with the shape of `t` giving the sweep's own time at each sample when known better
    than `t - t0` (e.g. a reference that ran late: see PathFollower.log); the windows and the chirp's phase then follow it."""
    t, u, y = (np.asarray(a, dtype=float) for a in (t, u, y))
    if not t.shape == u.shape == y.shape:
        raise ValueError("t, u and y must have the same length")
    s = t - t0 if sweep_time is None else np.asarray(sweep_time, dtype=float)
    if s.shape != t.shape:
        raise ValueError("sweep_time must have the length of t")
    if frequencies is None:
        margin = min(0.25, 0.2 * float(np.log2(sweep.f_end / sweep.f_start)))
        frequencies = np.geomspace(sweep.f_start * 2 ** margin, sweep.f_end / 2 ** margin, points)
    seconds_per_octave = sweep.duration * np.log(2.0) / np.log(sweep.f_end / sweep.f_start)
    kept, responses, qualities = [], [], []
    for f in np.asarray(frequencies, dtype=float):
        centre = float(sweep.time_at_frequency(f))
        width = min(cycles / f, octave_span * seconds_per_octave)
        window = np.abs(s - centre) <= 0.5 * width
        if window.sum() < minimum_samples:
            continue
        tw = s[window]
        weights = 0.5 * (1.0 + np.cos(2.0 * np.pi * (tw - centre) / (width + 1e-12)))  # Hann
        phase = sweep.phase(tw)
        c_in, _ = _phasor(tw, u[window], phase, weights)
        c_out, quality = _phasor(tw, y[window], phase, weights)
        if abs(c_in) < 1e-12:
            continue
        kept.append(f)
        responses.append(c_out / c_in)
        qualities.append(quality)
    return FrequencyResponse(np.array(kept), np.array(responses, dtype=complex), np.array(qualities))


def _tone_fit(s, x, frequencies):
    """x ~ sum over `frequencies` of a_k cos(2 pi f_k s) + b_k sin(2 pi f_k s), plus a constant and a drift, by least squares: the complex
    amplitudes C_k = a_k - j b_k (x ~ Re{sum C_k e^{j 2 pi f_k s}}) and the share of the power the tones explain. All the tones are fitted
    together: a drift term is not orthogonal to a tone, so fitting one tone with the others left in the signal leaks them into it."""
    s, x = np.asarray(s, dtype=float), np.asarray(x, dtype=float)
    centre = s.mean()
    columns = []
    for f in frequencies:
        columns += [np.cos(2.0 * np.pi * f * s), np.sin(2.0 * np.pi * f * s)]
    basis = np.column_stack(columns + [np.ones_like(s), s - centre])
    coefficients, *_ = np.linalg.lstsq(basis, x, rcond=None)
    tones = basis[:, : 2 * len(frequencies)] @ coefficients[: 2 * len(frequencies)]
    residual = x - basis @ coefficients
    explained, unexplained = float(np.sum(tones**2)), float(np.sum(residual**2))
    amplitudes = np.array([complex(coefficients[2 * k], -coefficients[2 * k + 1]) for k in range(len(frequencies))])
    return amplitudes, explained / max(explained + unexplained, 1e-300)


def tone_response(s, u, y, frequency, others=()):
    """The response of `y` to `u` at one known `frequency` (> 0, Hz), from equal-length 1-D samples at times `s` (s): the ratio of their complex
    amplitudes at that frequency in a joint least-squares fit of it and of `others` (iterable of distinct non-zero Hz frequencies present in the signals,
    e.g. the figure eight's other tones) with a constant and a drift. Returns (response, quality), the quality in [0, 1] as in FrequencyResponse.
    For a reference made of tones over a whole number of their cycles."""
    s, u, y = (np.asarray(a, dtype=float) for a in (s, u, y))
    if not s.shape == u.shape == y.shape:
        raise ValueError("s, u and y must have the same length")
    frequencies = [frequency, *others]
    c_in, _ = _tone_fit(s, u, frequencies)
    c_out, quality = _tone_fit(s, y, frequencies)
    if abs(c_in[0]) < 1e-12:
        raise ValueError(f"no input at {frequency} Hz to take a response to")
    return c_out[0] / c_in[0], quality


def plot_bode(response, path, fit=None, title=None):
    """Writes the response as a Bode plot (gain in dB over frequency, phase in degrees) to the PNG `path`, with the fitted
    model over it and the points of poor quality (under 0.5) hollow. matplotlib is imported here: it is optional."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    good = response.quality >= 0.5
    for axis, values, model in ((top, response.gain_db, None if fit is None else 20 * np.log10(np.abs(fit.model(response.frequency)))),
                                (bottom, response.phase_deg, None if fit is None else np.degrees(np.unwrap(np.angle(fit.model(response.frequency)))))):
        axis.semilogx(response.frequency[good], values[good], "o", color="C0", label="measured")
        axis.semilogx(response.frequency[~good], values[~good], "o", mfc="none", color="C0")
        if model is not None:
            axis.semilogx(response.frequency, model, "-", color="C1", label="second order + delay")
        axis.grid(True, which="both", alpha=0.3)
    top.set_ylabel("gain (dB)")
    bottom.set_ylabel("phase (deg)")
    bottom.set_xlabel("frequency (Hz)")
    top.legend()
    if title:
        top.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


@dataclass(frozen=True)
class SecondOrderFit:
    """H(s) = gain wn^2 / (s^2 + 2 zeta wn s + wn^2) e^{-s delay}: `wn` in rad/s, `delay` in seconds; `residual` the RMS of
    the fit's error in the complex plane, relative to the response."""

    gain: float
    wn: float
    zeta: float
    delay: float
    residual: float

    def model(self, frequency):
        w = 2.0 * np.pi * np.asarray(frequency, dtype=float)
        return (self.gain * self.wn**2 / (self.wn**2 - w**2 + 2j * self.zeta * self.wn * w)) * np.exp(-1j * w * self.delay)

    def __str__(self):
        return (f"second order + delay: gain {self.gain:.3f}, wn {self.wn:.1f} rad/s ({self.wn / (2 * np.pi):.2f} Hz), "
                f"zeta {self.zeta:.2f}, delay {self.delay * 1e3:.1f} ms (rms residual {self.residual:.3f})")


def fit_second_order(response, minimum_quality=0.5, delay=True):
    """Fits a second-order system with a delay to a measured `response` (`FrequencyResponse` with >= 5 points having `quality >= minimum_quality` in
    [0, 1]), weighting each frequency by `sqrt(quality)` and leaving out those under `minimum_quality`. The error is the log of the ratio of model
    to measurement, so gain and phase count alike. `delay=False` fixes it at zero."""
    use = response.quality >= minimum_quality
    if use.sum() < 5:
        raise ValueError(f"only {int(use.sum())} frequencies with quality >= {minimum_quality}: too few to fit")
    f, h, weight = response.frequency[use], response.response[use], np.sqrt(response.quality[use])
    w = 2.0 * np.pi * f
    k0 = float(np.median(np.abs(h[:3])))
    bandwidth = response.bandwidth() or float(f[len(f) // 2])

    def unpack(p):
        return p[0], p[1], p[2], (p[3] if delay else 0.0)

    def residuals(p):
        gain, wn, zeta, tau = unpack(p)
        model = (gain * wn**2 / (wn**2 - w**2 + 2j * zeta * wn * w)) * np.exp(-1j * w * tau)
        error = np.log(model / h)  # log-magnitude and phase together
        error = error.real + 1j * np.angle(np.exp(1j * error.imag))
        return np.concatenate([weight * error.real, weight * error.imag])

    best = None
    for zeta0 in (0.5, 1.0, 2.0):  # a few starts: the damping is the flat direction of the fit
        start = [k0, 2.0 * np.pi * bandwidth, zeta0, 0.01]
        result = least_squares(residuals, start[: 4 if delay else 3], bounds=([0.0, 0.1, 0.05, 0.0][: 4 if delay else 3],
                                                                                 [np.inf, 2e3, 20.0, 0.5][: 4 if delay else 3]))
        if best is None or result.cost < best.cost:
            best = result
    gain, wn, zeta, tau = unpack(best.x)
    rms = float(np.sqrt(np.mean(best.fun**2)))
    return SecondOrderFit(float(gain), float(wn), float(zeta), float(tau), rms)
