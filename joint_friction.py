"""Joint friction of a UR arm from constant-speed passes, and the compensation scales that follow from two such runs.

What is measured. One joint at a time is driven at a constant speed, forwards and back over the same stretch, by a joint-space controller
that tracks a moving reference. At a steady speed the joint does not accelerate, so the torque the controller commands (the cycle log's
`command_<joint>`; the UR adds gravity itself) is what balances the joint's friction, plus whatever is wrong with the payload's gravity.
The two directions tell them apart: forwards the command is `g + f(v)`, back it is `g - f(v)`, so `f = (forward - back) / 2` and
`g = (forward + back) / 2`. `f(v)` is fitted as Coulomb plus viscous, `f = c + b v`. Both coefficients may be negative:
a negative line is a compensation that pushes past the friction, so the controller has to hold the joint back.

What it is for. The UR's own compensation subtracts a scale times its own friction estimate from that torque. The scales are what
`direct_torque` takes (`viscous_scale`, `coulomb_scale`), each in [0, 1]; the estimate itself is not published and is not the `c + b v` line.
A run with the compensation off measures the arm's residual, `c` and `b`. A second run measures what is left. The difference is torque.
A cancellation scale follows only when that second run commanded its scales in the configuration, so the contribution per unit scale is
`(off - on) / scale` and the cancelling scale is `c / c_UR` and `b / b_UR`, kept in [0, 1] (see `recommend`). `friction_compensation: robot`
leaves the scales to the controller. Those scales were not commanded and are not known, so that pair of runs reports the torque removed
and no scale.
"""

import csv
import dataclasses

import numpy as np

# Clochiatti et al., Robotica 2024 (doi 10.1017/S0263574724000833), Table VI: one UR5e's friction on the MOTOR side, identified from the motor currents,
# friction = k_v w_m + tanh(w_m / 1e-3) T_c, with T_c = T_dir while the power flows from the motor to the load and T_rev when the load drives it.
# k_v in N m s/rad, T_c in N m; the 100:1 harmonic drive puts the joint side at gear * T_c (N m) and gear**2 * k_v (N m s/rad).
GEAR = 100.0
PUBLISHED_VISCOUS_MOTOR = (0.475e-3, 1.073e-3, 0.382e-3, 0.295e-3, 0.114e-3, 0.188e-3)
PUBLISHED_COULOMB_DIR_MOTOR = (0.0938, 0.0973, 0.0750, 0.0206, 0.0201, 0.0276)
PUBLISHED_COULOMB_REV_MOTOR = (0.071, 0.113, 0.083, 0.008, 0.001, 0.005)


def published_joint_side(joint):
    """(Coulomb driven by the motor, Coulomb driven by the load, viscous) of one joint on the joint side: N m, N m, N m s/rad. One arm of one
    lab: what to expect to within a factor of two or so, not a calibration of this arm."""
    return (GEAR * PUBLISHED_COULOMB_DIR_MOTOR[joint], GEAR * PUBLISHED_COULOMB_REV_MOTOR[joint], GEAR**2 * PUBLISHED_VISCOUS_MOTOR[joint])


JOINTS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")


def scales_of(entry, n_joints=6):
    """(viscous, coulomb) per joint that the configuration commanded: `off`, or {viscous: [n_joints], coulomb: [n_joints]}.

    `robot` has none. The controller chose them, and they are not a measured parameter."""
    if isinstance(entry, dict):
        return np.asarray(entry["viscous"], dtype=float), np.asarray(entry["coulomb"], dtype=float)
    if entry == "off" or entry is False:  # YAML reads a bare `off` as false; the driver treats that as off too
        return np.zeros(n_joints), np.zeros(n_joints)
    if entry == "robot":
        raise ValueError("friction_compensation robot: the controller chose the scales; they were not commanded and are not known")
    raise ValueError(f"friction_compensation {entry!r}: expected off, robot, or {{viscous: [...], coulomb: [...]}}")


def commanded_scales(entry, n_joints=6):
    """(viscous, coulomb) the configuration sent, or None for `robot`."""
    if entry == "robot":
        return None
    return scales_of(entry, n_joints=n_joints)


@dataclasses.dataclass(frozen=True)
class Pass:
    """One constant-speed stretch of one joint: where it began and ended on the cycle log's clock (CLOCK_MONOTONIC), and what was commanded."""

    joint: int  # 0-based index into joint_names
    direction: int  # +1 or -1
    speed: float  # rad/s, strictly positive
    begin: float  # start of the constant-speed part, s (< end)
    end: float  # end of the constant-speed part, s


def read_cycles(path):
    """The cycle log as (column names, array): rows are control cycles."""
    with open(path, newline="") as handle:
        names = next(csv.reader(handle))
    return names, np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def steady_state(names, cycles, group, stretch, middle=0.5, joint_names=JOINTS):
    """(mean command, mean velocity, torque std, samples) of `stretch`'s middle `middle` (in (0, 1]) share, for the joint it drove.

    `names` is the list of CSV column names, `cycles` is the (N, cols) array, `group` is the control group's name (e.g. `right_ur5e`; columns
    `command_<group>/<joint>_joint` and `qd_<...>`), `stretch` is a `Pass`, and `joint_names` is indexed by `stretch.joint`."""
    suffix = f"{group}/{joint_names[stretch.joint]}_joint"
    command, velocity = names.index(f"command_{suffix}"), names.index(f"qd_{suffix}")
    stamp = cycles[:, names.index("wall_time")]
    length = stretch.end - stretch.begin
    keep = (stamp >= stretch.begin + 0.5 * (1.0 - middle) * length) & (stamp <= stretch.end - 0.5 * (1.0 - middle) * length)
    if keep.sum() < 10:
        raise ValueError(f"joint {joint_names[stretch.joint]}, {stretch.direction:+d} {stretch.speed} rad/s: {int(keep.sum())} cycles in the window")
    return float(cycles[keep, command].mean()), float(cycles[keep, velocity].mean()), float(cycles[keep, command].std()), int(keep.sum())


def fit_coulomb_viscous(speeds, friction):
    """(coulomb, viscous, rms, coulomb standard error, viscous standard error) of `friction = c + b speed`.

    `speeds` (rad/s, > 0) and `friction` (N m) are 1-D arrays of equal length >= 2. `c` and `b` are free, so an over-compensated joint comes out
    negative instead of being clamped to zero. The standard errors are from the least-squares covariance; with only two speeds there is no spare
    point and they are NaN."""
    speeds, friction = np.asarray(speeds, dtype=float), np.asarray(friction, dtype=float)
    design = np.column_stack([np.ones_like(speeds), speeds])
    coulomb, viscous = np.linalg.lstsq(design, friction, rcond=None)[0]
    error = friction - design @ [coulomb, viscous]
    dof = len(friction) - 2
    if dof > 0:
        variance = float(error @ error) / dof
        spread = np.sqrt(np.maximum(np.diag(variance * np.linalg.inv(design.T @ design)), 0.0))
    else:
        spread = np.full(2, np.nan)
    rms = float(np.sqrt(np.mean(error ** 2)))
    return float(coulomb), float(viscous), rms, float(spread[0]), float(spread[1])


def friction_table(names, cycles, group, passes, tolerance=0.25, joint_names=JOINTS):
    """Per joint: the measurements and the fit. `passes` are Pass objects, forwards and back at each speed.

    A forward and a back pass are matched by (joint, speed). A pair whose measured speed is more than `tolerance` off the commanded one, in
    either direction, is dropped and listed: the joint did not move at the speed the model assumes there."""
    return friction_table_with(passes, lambda stretch: steady_state(names, cycles, group, stretch, joint_names=joint_names), tolerance)


def friction_table_with(passes, read, tolerance=0.25):
    """`friction_table` with the reader left to the caller: `read(stretch)` is (mean torque, mean velocity, torque std, samples) of one pass.

    A robot whose cycle log names its columns differently, or reports the measured torque instead of a commanded one, uses the same pairing and fit."""
    by_key = {}
    for stretch in passes:
        by_key.setdefault((stretch.joint, stretch.speed), {})[stretch.direction] = read(stretch)
    table = {}
    for joint in sorted({key[0] for key in by_key}):
        rows, dropped = [], []
        for (j, speed), pair in sorted(by_key.items()):
            if j != joint:
                continue
            if +1 not in pair or -1 not in pair:
                dropped.append((speed, "one direction missing"))
                continue
            (forward, forward_speed, forward_std, _), (back, back_speed, back_std, _) = pair[+1], pair[-1]
            if abs(forward_speed / speed - 1.0) > tolerance or abs(-back_speed / speed - 1.0) > tolerance:
                dropped.append((speed, f"measured {forward_speed:+.3f} and {back_speed:+.3f} rad/s"))
                continue
            rows.append(dict(speed=0.5 * (forward_speed - back_speed), friction=0.5 * (forward - back), gravity=0.5 * (forward + back),
                             noise=0.5 * (forward_std + back_std)))
        entry = dict(rows=rows, dropped=dropped)
        if len(rows) >= 2:
            coulomb, viscous, residual, coulomb_se, viscous_se = fit_coulomb_viscous(
                [r["speed"] for r in rows], [r["friction"] for r in rows])
            entry.update(coulomb=coulomb, viscous=viscous, residual=residual, coulomb_se=coulomb_se, viscous_se=viscous_se,
                         gravity=float(np.mean([r["gravity"] for r in rows])))
        table[joint] = entry
    return table


def removed(off, on):
    """Per joint, the Coulomb and viscous torque the on run took off the off run: N m and N m s/rad.

    This is the difference of the two fits. It does not divide by a scale."""
    out = {}
    for joint, measured in off.items():
        if "coulomb" not in measured or joint not in on or "coulomb" not in on[joint]:
            continue
        out[joint] = dict(coulomb=float(measured["coulomb"] - on[joint]["coulomb"]),
                          viscous=float(measured["viscous"] - on[joint]["viscous"]),
                          coulomb_off=float(measured["coulomb"]), viscous_off=float(measured["viscous"]))
    return out


def recommend(off, on, on_scales, margin=1.0):
    """The compensation scales from a run with it off (`off`: a `friction_table`) and one with it on at commanded `on_scales` = (viscous, coulomb)
    (each a sequence indexed by joint index, values in [0, 1]).

    `on_scales` are the numbers written in the configuration for that run. Per joint, c_UR and b_UR are what one unit of that scale removed,
    `(off - on) / scale`, with the on-run coefficients signed, so a negative residual counts as compensation that went past the friction.
    The scale that cancels it is `c / c_UR` and `b / b_UR`, stored as `coulomb_ideal` and `viscous_ideal`. The returned `coulomb` and `viscous`
    are that scale times `margin` (> 0), kept in [0, 1]: that is the range `direct_torque` accepts. A joint without a fit in both runs is left out."""
    viscous_on, coulomb_on = on_scales
    out = {}
    for joint, measured in off.items():
        if "coulomb" not in measured or joint not in on or "coulomb" not in on[joint]:
            continue
        notes = []
        result = {}
        for name, scale in (("coulomb", coulomb_on[joint]), ("viscous", viscous_on[joint])):
            contribution = (measured[name] - on[joint][name]) / scale if scale > 0.0 else 0.0
            if contribution <= 1e-6:
                result[name] = 0.0
                result[f"{name}_ideal"] = 0.0
                notes.append(f"{name}: the UR's compensation removed nothing measurable ({measured[name]:.3g} -> {on[joint][name]:.3g}); left at 0")
                continue
            ideal = float(measured[name] / contribution)
            result[f"{name}_ideal"] = ideal
            result[name] = float(np.clip(margin * ideal, 0.0, 1.0))
            if margin * ideal > 1.0:
                notes.append(f"{name}: cancellation {ideal:.2f} is past the controller's scale of 1; one full unit of the UR's friction "
                             f"model is short of this residual, so the scale stays at 1")
            if on[joint][name] < 0.0:
                notes.append(f"{name}: compensation overshot, residual {on[joint][name]:+.3g}; cancellation is at {ideal:.2f}, "
                             f"below the {scale:.2f} that was on")
        out[joint] = dict(result, notes=notes)
    return out
