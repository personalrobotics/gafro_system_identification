"""Joint friction of any N-joint arm from constant-speed passes at the MEASURED joint torque: the schedule, the path, the reader and the fit.

`joint_friction` fits the UR's commanded torque. An arm that commands position and reports the torque it is giving (a MIT-frame motor: the OpenArm)
has no commanded torque to read, but the argument is the same. At a steady speed the joint does not accelerate, so the torque it gives balances
friction plus gravity; forward it is `g + f(v)`, back it is `g - f(v)`, so `f = (forward - back) / 2` and `g = (forward + back) / 2`, and `f` is
fitted as Coulomb plus viscous. This module reuses that pairing and fit (`joint_friction.friction_table_with`) and adds what a robot-agnostic
package can: where the passes go, the path of each, how a log is read, and the table as plain numbers.

Nothing here moves a robot or knows one. A pass is data (`pass_path`: the times, positions and velocities of a joint path, the shape
`gafro_control_nodes.RobotNode.follow_path` takes); whatever plays it owns the gates and the permit, and the log it records is read back here.

The profile is the one `geodude identify joint-friction` has always used (a ramp of `ramp` seconds up, `constant` seconds at the speed, the same ramp
down), so a run of either reads the same way.
"""

import csv
import dataclasses

import numpy as np

from gafro_system_identification.joint_friction import Pass, friction_table_with

RAMP = 0.5  # s of acceleration at each end of a pass


@dataclasses.dataclass(frozen=True)
class PassPlan:
    """One pass of one joint, at rest at both ends: from `start`, `ramp` seconds up to `speed`, `constant` seconds at it, `ramp` seconds down.

    It covers `speed * (constant + ramp)` rad in `direction`, of which `speed * constant` is at the steady speed (`travel`)."""

    joint: int
    direction: int  # +1 or -1
    speed: float  # rad/s, positive
    start: float  # rad, where the pass begins, at rest
    constant: float  # s at the steady speed
    ramp: float = RAMP  # s of acceleration, and of braking

    def __post_init__(self):
        if self.direction not in (-1, 1) or self.speed <= 0.0 or self.constant < 0.0 or self.ramp <= 0.0:
            raise ValueError(f"need direction in (-1, 1), speed > 0, constant >= 0, and ramp > 0: {self}")

    @property
    def travel(self):
        """The rad covered at the steady speed."""
        return self.speed * self.constant

    @property
    def displacement(self):
        """The rad the whole pass covers, ramps included, signed."""
        return self.direction * self.speed * (self.constant + self.ramp)

    @property
    def duration(self):
        return 2.0 * self.ramp + self.constant


def stretch_profile(speed, constant, ramp=RAMP):
    """`(position(t), velocity(t), total_s)` callables for a pass of positive `speed` (rad/s): `ramp` (> 0, s) up, `constant` (>= 0, s) at `speed`,
    `ramp` down. `position(t)` is the unsigned distance covered (rad, >= 0) and `total` is `2 * ramp + constant` (s)."""
    total = 2.0 * ramp + constant

    def velocity(time_):
        if time_ <= 0.0 or time_ >= total:
            return 0.0
        return speed * min(time_ / ramp, 1.0, (total - time_) / ramp)

    def position(time_):
        time_ = float(np.clip(time_, 0.0, total))
        accelerating = 0.5 * speed * min(time_, ramp) ** 2 / ramp
        cruising = speed * max(0.0, min(time_, ramp + constant) - ramp)
        braking = 0.0
        if time_ > ramp + constant:
            remaining = time_ - ramp - constant
            braking = speed * remaining - 0.5 * speed * remaining**2 / ramp
        return accelerating + cruising + braking

    return position, velocity, total


def plan_stretches(speeds, travel_limit, min_duration, ramp=RAMP):
    """`([(speed, constant_s, covered_rad)], [skipped_speed])`: the positive `speeds` (rad/s) whose pass (`min_duration` >= 0 s at steady speed,
    `ramp` > 0 s at each end) covers at most `travel_limit` (> 0, rad), and those that would exceed it."""
    kept, skipped = [], []
    for speed in speeds:
        constant = max(min_duration, 0.0)
        covered = speed * (constant + ramp)
        if covered > travel_limit + 1e-9:
            skipped.append(speed)
        else:
            kept.append((speed, constant, covered))
    return kept, skipped


def reach(plan):
    """`(lowest_rad, highest_rad)` joint positions visited by `plan` (`PassPlan`)."""
    end = plan.start + plan.displacement
    return min(plan.start, end), max(plan.start, end)


def schedule(joints, speeds, centre, travel_limit, min_duration, ramp=RAMP):
    """(plans, skipped): for each 0-based joint index in `joints`, one at a time, at each positive speed in `speeds` (rad/s) ascending, a forward
    pass and the back pass over the same stretch.

    `centre` is a sequence indexed by joint (rad) giving the middle of the stretch; a pass covers `speed * (min_duration + ramp)` rad, at most
    `travel_limit` (> 0, rad), with `min_duration` >= 0 (s) and `ramp` > 0 (s); speeds that would cover more are in `skipped`. A joint's passes finish
    where they began, so the arm is not walked away from where it was put."""
    kept, skipped = plan_stretches(sorted(speeds), travel_limit, min_duration, ramp)
    plans = []
    for joint in joints:
        for speed, constant, covered in kept:
            low = centre[joint] - 0.5 * covered
            plans.append(PassPlan(joint, +1, float(speed), low, constant, ramp))
            plans.append(PassPlan(joint, -1, float(speed), low + covered, constant, ramp))
    return plans, skipped


def check_limits(plans, limits, margin=0.0):
    """Raise ValueError naming every pass in `plans` that leaves `limits[plan.joint]` = (lower, upper) (rad) shrunk by `margin` (>= 0, rad);
    nothing otherwise."""
    bad = []
    for plan in plans:
        lower, upper = limits[plan.joint]
        low, high = reach(plan)
        if low < lower + margin or high > upper - margin:
            bad.append(f"joint {plan.joint} {plan.direction:+d} {plan.speed} rad/s reaches [{low:.3f}, {high:.3f}], limits [{lower:.3f}, {upper:.3f}]")
    if bad:
        raise ValueError("passes outside the joint limits:\n  " + "\n  ".join(bad))


def pass_path(plan, hold, period=0.01):
    """The pass as a joint path a robot plays: (times, positions, velocities, window).

    `hold` is the 1-D joint vector (rad, length > `plan.joint`) the other joints stay at; the driven joint follows `stretch_profile` from
    `plan.start` sampled every `period` (> 0, s). `positions` and `velocities` have one row per time and one column per entry of `hold`: the shape
    `RobotNode.follow_path` takes. `window` is (begin, end) of the steady-speed part in the path's own time (s from its first row). The path's
    first row is where the arm must be, so a caller moves there first by whatever it uses to go places."""
    hold = np.asarray(hold, dtype=float)
    position, velocity, total = stretch_profile(plan.speed, plan.constant, plan.ramp)
    times = np.arange(0.0, total + 0.5 * period, period)
    positions = np.tile(hold, (len(times), 1))
    velocities = np.zeros_like(positions)
    positions[:, plan.joint] = plan.start + plan.direction * np.array([position(t) for t in times])
    velocities[:, plan.joint] = plan.direction * np.array([velocity(t) for t in times])
    return times, positions, velocities, (plan.ramp, plan.ramp + plan.constant)


def as_pass(plan, window, origin):
    """The `Pass` of `plan` once played: `window` from `pass_path`, `origin` the cycle log's clock at the path's time zero."""
    return Pass(plan.joint, plan.direction, plan.speed, origin + window[0], origin + window[1])


def steady_window(times, velocity, plan, tolerance=0.1, minimum=0.5):
    """(begin, end) of the longest stretch of `times` (1-D s) where the measured `velocity` (1-D rad/s, same length) held `plan`'s speed in its
    direction, or None.

    "Held" is within relative `tolerance` (> 0) of `plan.speed`, and the stretch must last at least `minimum` (in (0, 1]) of `plan.constant`. The
    window is read from what the joint did, not from a clock offset between the path and the recording, so a pass whose joint never held its speed
    (stiction at the slowest, a stop that fired) has no window and the caller says so instead of fitting a ramp."""
    times, velocity = np.asarray(times, dtype=float), np.asarray(velocity, dtype=float)
    holding = np.abs(velocity - plan.direction * plan.speed) <= tolerance * plan.speed
    best, start = None, None
    for i in range(len(times) + 1):
        if i < len(times) and holding[i]:
            start = i if start is None else start
            continue
        if start is not None:
            span = (times[start], times[i - 1])
            if span[1] - span[0] >= minimum * plan.constant and (best is None or span[1] - span[0] > best[1] - best[0]):
                best = span
            start = None
    return best


def torque_reader(names, cycles, joint_names, torque="tau_{}", velocity="qd_{}", stamp="wall_time", middle=0.5):
    """The `read(stretch)` of `friction_table_with` for a log (`names` column headers, `cycles` (N, cols) array) with a measured torque column and a
    velocity column per joint.

    `joint_names` is a sequence indexed by `stretch.joint`; `torque` and `velocity` are column templates formatted with `joint_names[stretch.joint]`,
    and `stamp` is the time column name (s). The window is the middle `middle` (in (0, 1]) share of the pass."""
    clock = cycles[:, names.index(stamp)]

    def read(stretch):
        joint = joint_names[stretch.joint]
        torque_column, velocity_column = names.index(torque.format(joint)), names.index(velocity.format(joint))
        length = stretch.end - stretch.begin
        keep = (clock >= stretch.begin + 0.5 * (1.0 - middle) * length) & (clock <= stretch.end - 0.5 * (1.0 - middle) * length)
        if keep.sum() < 10:
            raise ValueError(f"joint {joint}, {stretch.direction:+d} {stretch.speed} rad/s: {int(keep.sum())} cycles in the window")
        return (float(cycles[keep, torque_column].mean()), float(cycles[keep, velocity_column].mean()),
                float(cycles[keep, torque_column].std()), int(keep.sum()))

    return read


def identify(names, cycles, joint_names, passes, tolerance=0.25, **columns):
    """`friction_table` of the passes on a log read through `torque_reader`; `columns` are its `torque`, `velocity`, `stamp`, `middle`."""
    return friction_table_with(passes, torque_reader(names, cycles, joint_names, **columns), tolerance)


@dataclasses.dataclass(frozen=True)
class Recording:
    """One played pass's own log: its column `names` and the `cycles` array (one row per sample)."""

    names: list
    cycles: object


def read_recording(path):
    """A per-goal recording CSV at `path` (header row, then numeric sample rows) as a `Recording`."""
    with open(path, newline="", encoding="utf-8") as handle:
        names = [n.strip() for n in next(csv.reader(handle))]
    return Recording(names, np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2))


def identify_recordings(played, joint_names, tolerance=0.25, **columns):
    """`friction_table` of passes that each have their own recording: `played` is [(Pass, Recording)].

    A service that records one file per goal (the OpenArm's: `t` in seconds from the goal's start, columns `q1..`, `qd1..`, `tau1..`) gives every
    pass the origin 0 and its own clock, so the reader is chosen per pass. `columns` are `torque_reader`'s: for that log
    `torque="tau{}", velocity="qd{}", stamp="t"` with `joint_names` ("1", ..., "7"). Samples that repeat a row (a driver loop slower than the log's
    rate) are read as they are: a mean over a steady stretch does not care."""
    readers = {id(played_pass): torque_reader(recording.names, recording.cycles, joint_names, **columns) for played_pass, recording in played}
    return friction_table_with([played_pass for played_pass, _ in played], lambda stretch: readers[id(stretch)](stretch), tolerance)


def as_numbers(table, joint_names):
    """The table as {joint name: {coulomb, viscous, gravity_error, rms, coulomb_se, viscous_se, passes, dropped}}: plain floats, ready for YAML or a controller.

    A joint with fewer than two usable speeds has no fit; it is listed with its `dropped` reasons and no coefficients, never with zeros."""
    out = {}
    for joint, entry in sorted(table.items()):
        row = dict(passes=len(entry["rows"]), dropped=[f"{speed} rad/s: {why}" for speed, why in entry["dropped"]])
        if "coulomb" in entry:
            row.update(coulomb=entry["coulomb"], viscous=entry["viscous"], gravity_error=entry["gravity"], rms=entry["residual"],
                       coulomb_se=entry["coulomb_se"], viscous_se=entry["viscous_se"])
        out[joint_names[joint]] = row
    return out


def write_passes(path, passes):
    """The passes as a CSV (joint, direction, speed, begin, end): what the run must keep next to the cycle log."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["joint", "direction", "speed", "begin", "end"])
        for p in passes:
            writer.writerow([p.joint, p.direction, repr(p.speed), repr(p.begin), repr(p.end)])


def read_passes(path):
    with open(path, newline="") as handle:
        return [Pass(int(r["joint"]), int(r["direction"]), float(r["speed"]), float(r["begin"]), float(r["end"])) for r in csv.DictReader(handle)]
