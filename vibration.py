"""Where a vibration starts and what it is tied to, from the robot interface's control cycle log and the teleop's tick log.

Two logs, one clock (CLOCK_MONOTONIC). The cycle log (`record_diagnostics(trace=True)`) has every control cycle's `wall_time`, its `period`
and `compute` (the control loop's own timing: a loop that stalls delays the command *inside* the feedback loop) and the arm's `qd_<joint>`. The tick log
has every device tick of the teleop: when it began, how late, how long its Python work took, and the command it acted on.

What it tells apart:
  * Where the vibration is: the band of frequencies (slow hunting, a mid band, a fast one), its dominant frequency and the joints that carry it, and when
    it began.
  * Whether the device is behind it: the device's commands (active: the input device is deflected, idle: it is not) against the onset, and whether the
    vibration goes on after the device has been idle. Nothing the device or the reference path does after it goes idle can keep a vibration going: the
    reference does not move (the leash acts only on a lagging arm), so what continues is the loop itself.
  * Whether the loop's timing changes with the device: the control cycles' period and compute while it is active against while it is idle. A control loop
    that slows or stalls when the input device is read (a Python law in the loop, contending for the interpreter lock) is a command delay inside the
    feedback loop; the reference path's own jitter (the tick log's lateness) is not, since it is outside it: it excites, it cannot destabilise.
"""

from dataclasses import dataclass, field

import numpy as np

from gafro_system_identification.trace import read_trace

BANDS = {"slow": (0.5, 3.0), "mid": (3.0, 10.0), "fast": (10.0, 40.0), "high": (40.0, 120.0), "nyquist": (120.0, 250.0)}
# Hz: hunting and the commanded motion itself, the loop's own, what rings above it, what is audible (a buzz), and up to the 500 Hz cycle log's Nyquist (a period-2
# oscillation of the command is at 250 Hz). A band above the log's own Nyquist is empty; sounds above it alias into these bands.


@dataclass
class Cycles:
    """The control cycles of one arm: `wall` (s), the cycle's `period` and `compute` (s) and the joints' velocity `qd` (n, joints), with the joint names."""

    wall: np.ndarray
    period: np.ndarray
    compute: np.ndarray
    qd: np.ndarray
    joints: list
    command: np.ndarray = None  # (n, joints): what the controller commanded (Nm in torque mode), NaN for a joint it does not command; None if not logged

    @property
    def rate(self):
        return 1.0 / float(np.median(np.diff(self.wall)))


@dataclass
class Ticks:
    """The teleop's device ticks: `stamp` (s, the cycle log's clock), `lateness` and `work` (s) and the `velocity` command it acted on, (n, 6)."""

    stamp: np.ndarray
    lateness: np.ndarray
    work: np.ndarray
    velocity: np.ndarray

    def active(self, times, hold=0.1):
        """Whether the device was commanding at each of `times`: the latest tick before it acted on a non-zero command, and was no more than `hold` s old."""
        k = np.searchsorted(self.stamp, times, side="right") - 1
        valid = (k >= 0) & (times - self.stamp[np.clip(k, 0, None)] <= hold)
        moving = np.any(self.velocity != 0.0, axis=1)
        return valid & moving[np.clip(k, 0, None)]


def read_cycles(path, velocity_prefix):
    """The cycle log at `path`, for the joints of one arm: the joints whose velocity columns start with `velocity_prefix` (e.g. `qd_right_ur5e/`: the
    column name template of the robot's cycle log, `qd_<arm's group>/<joint>`)."""
    with open(path) as handle:
        header = handle.readline().strip().split(",")
    joints = [name[3:] for name in header if name.startswith(velocity_prefix)]
    if not joints:
        raise ValueError(f"{path}: no {velocity_prefix}... columns: was it recorded with trace=True, and for that arm?")
    trace = read_trace(path, ["wall_time", "period", "compute"] + [f"qd_{j}" for j in joints])
    command = None
    if all(f"command_{j}" in header for j in joints):
        command = read_trace(path, [f"command_{j}" for j in joints])
        command = np.column_stack([command[f"command_{j}"] for j in joints])
    return Cycles(trace["wall_time"], trace["period"], trace["compute"], np.column_stack([trace[f"qd_{j}"] for j in joints]), joints, command)


class TickLog:
    """Writes the teleop's device ticks as TeleopLoop's `on_tick` gives them (stamp, lateness, velocity, work), in the format `read_ticks` reads, flushed every
    `flush_every` ticks (a second at the device's 50 Hz): a run that ends badly leaves what it did."""

    HEADER = "stamp,lateness,work,v0,v1,v2,v3,v4,v5"

    def __init__(self, path, flush_every=50):
        self._file = open(path, "w")
        self._file.write(self.HEADER + "\n")
        self._flush_every, self._count = flush_every, 0

    def __call__(self, stamp, lateness, velocity, work):
        self._file.write(f"{stamp:.6f},{lateness:.6f},{work:.6f}," + ",".join(f"{v:.6g}" for v in velocity) + "\n")
        self._count += 1
        if self._count % self._flush_every == 0:
            self._file.flush()

    def close(self):
        self._file.close()


def read_ticks(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return Ticks(data[:, 0], data[:, 1], data[:, 2], data[:, 3:9])


def band_levels(signal, rate, window=0.5, hop=0.25, bands=BANDS):
    """The `(n, k)` `signal` (e.g. joint velocities) in windows of `window` (> 0, s) every `hop` (> 0, s) at sample `rate` (> 0, Hz): returns
    centre times `(m,)`, rms in each band `(m, bands, k)`, and in each band `(low_hz, high_hz)` of `bands` the dominant frequency over all `k`, `(m, bands)`.

    Each window is Hann-weighted and its one-sided power summed over the band, scaled so that a sinusoid of amplitude a reads a / sqrt(2) and a signal that
    flips sign every sample (all of its power at the Nyquist frequency, a bin counted once) reads its amplitude. A band reaching the Nyquist frequency
    includes it."""
    size, step = int(round(window * rate)), int(round(hop * rate))
    weights = np.hanning(size)
    one_sided = np.full(size // 2 + 1, 2.0)  # an interior bin stands for its mirror image too; the Nyquist bin (even sizes) and DC do not
    one_sided[0] = 1.0
    if size % 2 == 0:
        one_sided[-1] = 1.0
    scale = one_sided[:, None] / (size * np.sum(weights**2))
    frequencies = np.fft.rfftfreq(size, 1.0 / rate)
    starts = np.arange(0, len(signal) - size + 1, step)
    rms = np.zeros((len(starts), len(bands), signal.shape[1]))
    dominant = np.zeros((len(starts), len(bands)))
    for m, start in enumerate(starts):
        segment = signal[start:start + size]
        power = scale * np.abs(np.fft.rfft((segment - segment.mean(0)) * weights[:, None], axis=0)) ** 2
        for b, (low, high) in enumerate(bands.values()):
            reaches = high >= frequencies[-1] * (1.0 - 1e-6)  # a band that reaches the Nyquist frequency includes its bin (the rate is measured: no exact comparison)
            inside = (frequencies >= low) & (np.ones_like(frequencies, bool) if reaches else (frequencies < high))
            if inside.any():  # a band above the signal's Nyquist has nothing
                rms[m, b] = np.sqrt(power[inside].sum(0))
                dominant[m, b] = frequencies[inside][int(np.argmax(power[inside].sum(1)))]
    return (starts + size / 2) / rate, rms, dominant


def find_onset(times, level, baseline=3.0, factor=4.0, floor=0.005, persist=3):
    """The first time in `times` ((m,) monotonic s) where `level` ((m,) non-negative) rises above `factor` (> 0) times its median over the first
    `baseline` (> 0, s) seconds -- and above `floor` (>= 0) -- and stays there for `persist` (>= 1) consecutive windows; None if it does not.
    Returns (time, threshold)."""
    quiet = times <= times[0] + baseline
    threshold = max(factor * float(np.median(level[quiet])), floor)
    above = level > threshold
    for k in range(len(above) - persist + 1):
        if above[k:k + persist].all():
            return float(times[k]), threshold
    return None, threshold


def lag_one(x, smooth=11):
    """The lag-1 autocorrelation of `x` (1-D array, length > `2 * smooth`) after its slow part is taken out (a centred moving average of `smooth` >= 1
    samples): +1 for a slow signal, 0 for white noise, -1 for one that alternates sign every sample -- a period-2 oscillation at the Nyquist frequency,
    the buzz a discrete-time loop falls into when its velocity feedback and its delay do not agree."""
    x = np.asarray(x, dtype=float)
    fast = x - np.convolve(x, np.ones(smooth) / smooth, mode="same")
    fast = fast[smooth:-smooth]
    return float(np.sum(fast[:-1] * fast[1:]) / max(np.sum(fast**2), 1e-300))


def percentiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return dict(median=np.nan, p99=np.nan, max=np.nan)
    return dict(median=float(np.median(values)), p99=float(np.percentile(values, 99)), max=float(values.max()))


@dataclass
class BandReport:
    name: str
    onset: float
    threshold: float
    final: float  # rms over the last windows (rad/s for the velocity, the command's units for the command)
    baseline: float
    frequency: float  # dominant frequency in the last windows, Hz
    joints: list  # (joint, rms) in the last windows, largest first
    persists: float  # rms after the device's last command (the idle tail), as a multiple of the threshold; None when there is none


@dataclass
class VibrationReport:
    duration: float
    rate: float
    bands: list = field(default_factory=list)  # BandReports of the joints' velocity
    command_bands: list = field(default_factory=list)  # BandReports of the controller's command, when it was logged
    alternation: dict = field(default_factory=dict)  # {"velocity"|"command": (joint, lag-1 autocorrelation in the quiet, in the last seconds)}
    rest: dict = field(default_factory=dict)  # {joint: {"qd_std", "qd_step", "command_std"}} over the quiet at the start
    cycle_timing: dict = field(default_factory=dict)  # {"all"|"active"|"idle"|"before onset": {"period": {...}, "compute": {...}}}
    tick_timing: dict = field(default_factory=dict)  # likewise, with "lateness" and "work"
    device_last_active: float = None
    device_active_fraction: float = None
    late_cycles: int = 0
    notes: list = field(default_factory=list)


def _band_reports(times, signal, rate, joints, bands, window, baseline, factor, floor, tail, device_last_active):
    """The BandReports of one signal (n, k): per band when it rose, at what frequency, on which joints, and whether it goes on after the device's last command."""
    times_, rms, dominant = band_levels(signal, rate, window, window / 2, bands)
    times_ = times_ + times
    last = times_ >= times_[-1] - tail
    reports = []
    for b, name in enumerate(bands):
        level = rms[:, b].max(axis=1)  # the joint that vibrates most, per window
        onset, threshold = find_onset(times_, level, baseline, factor, floor)
        order = np.argsort(rms[last, b].mean(0))[::-1]
        persists = None
        if device_last_active is not None and device_last_active <= times_[-1] - tail:
            idle_tail = times_ > device_last_active + 0.5
            persists = float(level[idle_tail].mean() / threshold) if idle_tail.any() else None
        reports.append(BandReport(name, onset, threshold, float(level[last].mean()), float(np.median(level[times_ <= times_[0] + baseline])),
                                  float(np.median(dominant[last, b])), [(joints[j], float(rms[last, b].mean(0)[j])) for j in order[:3]], persists))
    return reports


def _alternation(signal, wall, joints, baseline, tail, rate):
    """The joint whose samples carry most of their power above 40 Hz at the end, and its lag-1 autocorrelation (lag_one) in the quiet and at the end."""
    quiet, end = wall <= wall[0] + baseline, wall >= wall[-1] - tail
    fast = [float(np.sum(np.abs(np.fft.rfft(signal[end, j] - signal[end, j].mean())[np.fft.rfftfreq(int(end.sum()), 1.0 / rate) > 40.0]) ** 2))
            for j in range(signal.shape[1])]
    j = int(np.argmax(fast))
    return joints[j], lag_one(signal[quiet, j]), lag_one(signal[end, j])


def analyse(cycles, ticks=None, window=0.5, baseline=3.0, factor=4.0, floor=0.005, tail=2.0, bands=BANDS, command_floor=0.02):
    """The report on a run from `cycles` (`Cycles`) and optional `ticks` (`Ticks` on the same `CLOCK_MONOTONIC` clock): the vibration per band of
    the joints' velocity and, when logged, of the controller's command (when it began, at which frequency, on which joints, whether it goes on after
    the device is idle for the last `tail` > 0 s); whether the last seconds alternate sign every cycle (a period-2 buzz); the noise at rest (first
    `baseline` > 0 s); and the timing of the control loop and the ticks, split by the device's state when there is a tick log. `window` > 0 (s);
    `factor` > 0; `floor`, `command_floor` >= 0. Bands above the log's Nyquist frequency are left out."""
    wall = cycles.wall
    report = VibrationReport(float(wall[-1] - wall[0]), cycles.rate)
    bands = {name: edges for name, edges in bands.items() if edges[0] < cycles.rate / 2}
    if ticks is not None:
        moving = ticks.stamp[np.any(ticks.velocity != 0.0, axis=1)]
        report.device_last_active = float(moving[-1]) if moving.size else None
        report.device_active_fraction = float(np.mean(ticks.active(wall)))
    args = (cycles.rate, cycles.joints, bands, window, baseline, factor)
    report.bands = _band_reports(wall[0], cycles.qd, args[0], args[1], args[2], args[3], args[4], args[5], floor, tail, report.device_last_active)
    report.alternation["velocity"] = _alternation(cycles.qd, wall, cycles.joints, baseline, tail, cycles.rate)
    quiet = wall <= wall[0] + baseline
    for j, name in enumerate(cycles.joints):
        steps = np.abs(np.diff(cycles.qd[quiet, j]))
        steps = steps[steps > 1e-9]
        report.rest[name] = {"qd_std": float(np.std(cycles.qd[quiet, j])), "qd_step": float(steps.min()) if steps.size else float("nan"),
                             "command_std": float("nan")}
    if cycles.command is not None:
        commanded = np.all(np.isfinite(cycles.command), axis=0)
        if commanded.any():
            names = [name for name, used in zip(cycles.joints, commanded) if used]
            command = cycles.command[:, commanded]
            report.command_bands = _band_reports(wall[0], command, args[0], names, args[2], args[3], args[4], args[5], command_floor, tail, report.device_last_active)
            report.alternation["command"] = _alternation(command, wall, names, baseline, tail, cycles.rate)
            for k, name in enumerate(names):
                ripple = command[quiet, k] - np.convolve(command[quiet, k], np.ones(11) / 11, mode="same")
                report.rest[name]["command_std"] = float(np.std(ripple[11:-11]))

    def timing(selected):
        return {"period": percentiles(cycles.period[selected]), "compute": percentiles(cycles.compute[selected])}

    everything = np.ones(len(wall), bool)
    report.cycle_timing["all"] = timing(everything)
    typical = report.cycle_timing["all"]["period"]["median"]
    report.late_cycles = int(np.sum(cycles.period > 1.5 * typical))
    if ticks is not None:
        device = ticks.active(wall)
        report.cycle_timing["active"], report.cycle_timing["idle"] = timing(device), timing(~device)
        report.tick_timing["all"] = {"lateness": percentiles(ticks.lateness), "work": percentiles(ticks.work)}
        tick_active = np.any(ticks.velocity != 0.0, axis=1)
        report.tick_timing["active"] = {"lateness": percentiles(ticks.lateness[tick_active]), "work": percentiles(ticks.work[tick_active])}
        report.tick_timing["idle"] = {"lateness": percentiles(ticks.lateness[~tick_active]), "work": percentiles(ticks.work[~tick_active])}
    onsets = [band.onset for band in report.bands if band.name != "slow" and band.onset is not None]
    if onsets:
        first = min(onsets)
        before = (wall > first - 2.0) & (wall <= first)
        if before.sum() > 10:
            report.cycle_timing["before onset"] = timing(before)
        if ticks is not None:
            near = (ticks.stamp > first - 2.0) & (ticks.stamp <= first)
            if near.any():
                report.tick_timing["before onset"] = {"lateness": percentiles(ticks.lateness[near]), "work": percentiles(ticks.work[near])}
            report.notes.append("onset with the device " + ("ACTIVE" if bool(ticks.active(np.array([first]))[0]) else "IDLE") + f" (t = {first - wall[0]:.1f} s)")
    return report
