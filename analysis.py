"""The analysis of an experiment run: the response of the end effector's pose to the reference that was run, from the control cycle log and the
run's record (a record object with `segments`, `origin()`, `model`, `arm`). The same functions are called at the end of an experiment and by CLI
analysis tools on data recorded earlier.

Each segment of a record is analysed by what it is. A sweep (a chirp along or about one axis) gives the response over a band of frequencies by the
lock-in estimate of response.py. A figure eight is three tones -- x at 1/period, y at twice that, z at `heave_cycles` times it -- so its response is
estimated at those frequencies by a least-squares fit over the steady part (the whole figures, easing excluded), which holds a whole number of
cycles of each. Either way every output of the pose is read, not only the excited one: the others are the coupling.
"""

import pathlib
import time
from dataclasses import dataclass

import numpy as np

from gafro_system_identification.response import FrequencyResponse, fit_second_order, frequency_response, plot_bode, tone_response
from gafro_system_identification.trace import (
    DOFS,
    coupling,
    joint_columns,
    measured_displacements,
    measured_displacements_with,
    read_trace,
    reference_times,
)

FIGURE_DOFS = ("x", "y", "z")


@dataclass
class SegmentAnalysis:
    """The response to one segment: `inputs` maps each excited degree of freedom to {output degree of freedom: FrequencyResponse}, the entry of
    its own dof being the loop and the others the coupling. A sweep excites one dof over a band; a figure eight three, one tone each (a
    FrequencyResponse of a single frequency)."""

    segment: object
    inputs: dict
    samples: int

    @property
    def tones(self):
        return self.segment.kind == "figure8"


def figure_tones(path):
    """{dof: frequency in Hz} of a Figure8's tones: x once per figure, y twice, z `heave_cycles` times; a dof with no amplitude has none. The tones
    must differ, or their responses cannot be told apart."""
    tones = {dof: m / path.period for dof, m, amplitude in (("x", 1, path.width), ("y", 2, path.depth), ("z", path.heave_cycles, path.heave))
             if amplitude != 0.0}
    frequencies = sorted(tones.values())
    if any(b - a < 1e-9 for a, b in zip(frequencies, frequencies[1:])):
        raise ValueError(f"the figure's tones coincide ({tones}): the responses at one frequency cannot be told apart; change --heave-cycles")
    return tones


def analyse_arrays(record, wall, displacements, **options):
    """The analysis of every segment of `record`, from the cycle log's stamps `wall` ((N,) monotonic s) and the end effector's displacement at each,
    (N, 6) in the order of `trace.DOFS` (`x, y, z, rx, ry, rz` in m and rad).

    `record.segments` is an iterable of segment objects with `.kind` (`"sweep"` or `"figure8"`), `.log` (`(stamp_s, path_time_s)` sequence),
    `.period` (> 0, s), `.label`, and `.path()`. `options` go to `response.frequency_response`."""
    wall = np.asarray(wall, dtype=float)
    displacements = np.asarray(displacements, dtype=float)
    if displacements.shape != (len(wall), len(DOFS)):
        raise ValueError(f"displacements must have shape ({len(wall)}, {len(DOFS)}) matching wall and DOFS, got {displacements.shape}")
    results = []
    for segment in record.segments:
        path = segment.path()
        tau = reference_times(segment.log, segment.period, wall)
        inside = np.isfinite(tau)
        if inside.sum() < 100:
            raise ValueError(f"{segment.label}: only {int(inside.sum())} cycles of the log fall inside it: is this the log of the run?")
        if segment.kind == "sweep":
            u = path.sweep.value(tau[inside])
            inputs = {path.dof: {out: frequency_response(wall[inside], u, displacements[inside, j], path.sweep, sweep_time=tau[inside], **options)
                                 for j, out in enumerate(DOFS)}}
        elif segment.kind == "figure8":
            # the whole figures that were traced (a run stopped early has fewer than it set out to): each tone holds whole cycles of them
            whole = min(path.loops, int(np.floor((np.max(tau[inside]) - path.ramp) / path.period + 1e-9)))
            if whole < 1:
                raise ValueError("the figure eight stopped before one whole figure was traced: nothing to analyse at its tones")
            steady = inside & (tau >= path.ramp) & (tau <= path.ramp + whole * path.period)
            offsets = np.array([path.offset(t) for t in tau[steady]])
            inputs = {}
            tones = figure_tones(path)
            for dof, frequency in tones.items():
                u = offsets[:, FIGURE_DOFS.index(dof)]
                others = [f for d, f in tones.items() if d != dof]
                inputs[dof] = {}
                for j, out in enumerate(DOFS):
                    h, quality = tone_response(tau[steady], u, displacements[steady, j], frequency, others)
                    inputs[dof][out] = FrequencyResponse(np.array([frequency]), np.array([h], dtype=complex), np.array([quality]))
        else:
            raise ValueError(f"{segment.label}: unknown segment kind {segment.kind!r}; expected 'sweep' or 'figure8'")
        results.append(SegmentAnalysis(segment, inputs, int(inside.sum())))
    return results


def analyse_with_fk(record, cycles_path, columns, fk, origin_position, origin_matrix, **options):
    """The analysis of `record` from the cycle log at `cycles_path` using a generic `fk(q) -> (position, rotation_matrix)` callable and origin pose,
    without requiring a gafro System or task space.

    `columns` must list the `q_<joint>` column names in the exact joint order expected by `fk(q)`; `origin_position` is (3,) in m and
    `origin_matrix` is a (3, 3) world-frame rotation matrix."""
    trace = read_trace(cycles_path, ["wall_time"] + list(columns))
    displacements = measured_displacements_with(trace, columns, fk, origin_position, origin_matrix)
    return analyse_arrays(record, trace["wall_time"], displacements, **options)


def analyse(record, cycles_path, system, task_space, **options):
    """The analysis of `record` from the cycle log at `cycles_path`: the logged joint positions of the arm of `task_space` are put through its forward
    kinematics, as displacements from the record's origin, then analysed (analyse_arrays)."""
    columns = joint_columns(system, task_space)
    trace = read_trace(cycles_path, ["wall_time"] + columns)
    displacements = measured_displacements(trace, columns, task_space, record.origin())
    return analyse_arrays(record, trace["wall_time"], displacements, **options)


def analyse_files(record, cycles_path, model=None, **options):
    """`analyse`, with the kinematics from a model loaded for the purpose (a kinematic RobotInterface, no robot, no simulation): the record's
    model unless `model` is given. How an experiment analyses its run once the robot is stopped, and how CLI tools analyse a recorded
    one -- one path for both."""
    import gafro_robot_interface as ri

    interface = ri.RobotInterface(ri.Configuration(str(model or record.model), "kinematic"))
    interface.initialize()
    try:
        return analyse(record, cycles_path, interface.get_system(), interface.get_task_space(f"{record.arm}_ee"), **options)
    finally:
        interface.shutdown()


def _low_frequency_delay(response):
    if response.frequency.size == 0:
        return float("nan")
    return float(np.median(response.delay()[:3])) * 1e3


def report(results, out=None, plot=False, verbose=False, name="analysis", title=""):
    """Prints what `analyse` found, and with `out` (a folder path) writes it as CSV, and with `plot` the sweeps as Bode plots.

    `results` is a sequence of `SegmentAnalysis` (where each `.segment` exposes `.complete` and `.label`).
    Sweeps: per degree of freedom the fit (gain, wn, zeta, delay), the delay the phase implies at low frequency, the -3 dB bandwidth, the mean quality and
    the largest off-axis gain. Figure eights: per tone the gain, phase, the delay the phase implies at that frequency, the quality and the largest
    off-axis gain. Off-axis gains are median |H| in SI units of output per unit of input (m/m, rad/rad, m/rad, rad/m)."""
    out = None if out is None else pathlib.Path(out)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        if not result.segment.complete:
            bands = [float(r[dof].frequency.max()) for dof, r in result.inputs.items() if r[dof].frequency.size] if not result.tones else []
            if result.tones:
                note = f"analysed over the {len(result.inputs)} tones of its whole figures"
            elif bands:
                note = f"its response is for the band it got to, up to {max(bands):.2f} Hz"
            else:
                note = "it had not swept far enough to analyse any frequency"
            print(f"NOTE: {result.segment.label} stopped early; {note}")
    results = [r for r in results if r.tones or any(responses[dof].frequency.size for dof, responses in r.inputs.items())]
    sweeps = [r for r in results if not r.tones]
    figures = [r for r in results if r.tones]
    for result in sweeps:
        for dof, responses in result.inputs.items():
            own = responses[dof]
            if verbose:
                print(f"--- {dof} ---\n{own.table()}\n")
            try:
                fit = fit_second_order(own)
            except ValueError as error:
                fit = None
                print(f"{dof}: no fit: {error}")
            rows.append(("sweep", dof, own, fit, coupling(responses, dof)))
            if out is not None:
                np.savetxt(out / f"{name}_{dof}_{stamp}.csv", np.column_stack([own.frequency, own.gain_db, own.phase_deg, own.delay() * 1e3, own.quality]),
                           delimiter=",", header="hz,gain_db,phase_deg,delay_ms,quality", comments="")
                if plot:
                    plot_bode(own, out / f"{name}_{dof}_{stamp}.png", fit, f"pose loop, {dof}{', ' + title if title else ''}")
    if sweeps:
        print(f"{'dof':>4s} {'gain':>6s} {'wn rad/s':>9s} {'zeta':>5s} {'delay ms':>9s} {'low-f delay ms':>15s} {'-3 dB Hz':>9s} {'quality':>8s}   off-axis (median |H|, SI)")
        for _, dof, own, fit, cross in rows:
            bandwidth = own.bandwidth()
            worst = max(cross, key=cross.get) if cross else None
            print(f"{dof:>4s} " + (f"{fit.gain:6.3f} {fit.wn:9.1f} {fit.zeta:5.2f} {fit.delay * 1e3:9.1f}" if fit else f"{'':6s} {'':9s} {'':5s} {'':9s}")
                  + f" {_low_frequency_delay(own):15.1f} {'-' if bandwidth is None else f'{bandwidth:9.2f}':>9s} {own.quality.mean():8.2f}   "
                  + ("" if worst is None else f"largest into {worst}: {cross[worst]:.3f}"))
    for result in figures:
        print(f"\n{result.segment.label}, at its tones:")
        print(f"{'dof':>4s} {'Hz':>7s} {'gain':>7s} {'phase deg':>10s} {'delay ms':>9s} {'quality':>8s}   off-axis (median |H|, SI)")
        table = []
        for dof, responses in result.inputs.items():
            own = responses[dof]
            cross = coupling(responses, dof)
            worst = max(cross, key=cross.get) if cross else None
            print(f"{dof:>4s} {own.frequency[0]:7.3f} {own.gain[0]:7.3f} {own.phase_deg[0]:10.2f} {own.delay()[0] * 1e3:9.1f} {own.quality[0]:8.2f}   "
                  + ("" if worst is None else f"largest into {worst}: {cross[worst]:.4f}"))
            table.append([DOFS.index(dof), own.frequency[0], own.gain_db[0], own.phase_deg[0], own.delay()[0] * 1e3, own.quality[0]]
                         + [responses[o].gain[0] for o in DOFS])
            rows.append(("tone", dof, own, None, cross))
        if out is not None:
            np.savetxt(out / f"{name}_tones_{stamp}.csv", np.array(table), delimiter=",", comments="",
                       header="dof_index,hz,gain_db,phase_deg,delay_ms,quality," + ",".join(f"gain_into_{o}" for o in DOFS))
    if out is not None:
        print(f"\nanalysis in {out}")
    return rows
