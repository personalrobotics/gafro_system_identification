"""Reading a run's trace: the cycle log of a robot interface, the pose read off it, and the reference's time rebuilt from a publish log.

The six degrees of freedom of the pose, in the order of the displacements below, are a translation along a world axis (m) or a rotation about a world axis
through the end effector's origin (rad). Nothing here needs a robot: `measured_displacements_with` takes any `fk(q) -> (position, rotation_matrix)`
callable, and `joint_columns` / `measured_displacements` wrap a gafro System and task space when present.
"""

import numpy as np
from scipy.spatial.transform import Rotation

DOFS = ("x", "y", "z", "rx", "ry", "rz")


def reference_times(log, period, wall):
    """The path time the reference was at, at each controller-clock stamp in `wall` (1-D array-like, s), from a PathFollower's `log` (non-empty
    sequence of `(stamp_s, path_time_s)` pairs sorted by `stamp_s` on the same clock as `wall`): the tick that published the latest chunk before the
    stamp, plus the time since it (a chunk lasts one `period` > 0 s, then the reference holds). NaN before the first chunk and after the last one's
    period."""
    stamps, times = np.array([s for s, _ in log]), np.array([t for _, t in log])
    wall = np.asarray(wall, dtype=float)
    k = np.searchsorted(stamps, wall, side="right") - 1
    tau = times[np.clip(k, 0, None)] + np.minimum(wall - stamps[np.clip(k, 0, None)], period)
    tau = np.where((k < 0) | (wall > stamps[-1] + period), np.nan, tau)
    return tau


def read_trace(path, names):
    """The columns `names` of a cycle log CSV (RobotInterface.record_diagnostics) as a dict of arrays; the file is large, so only these are read."""
    with open(path) as handle:
        header = handle.readline().strip().split(",")
    missing = [n for n in names if n not in header]
    if missing:
        raise KeyError(f"the cycle log has no column {missing[:3]}: was it recorded with trace=True?")
    columns = [header.index(n) for n in names]
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=columns, ndmin=2)
    return {name: data[:, k] for k, name in enumerate(names)}


def joint_columns(system, task_space):
    """The cycle log's `q_<joint>` column names for the joints of `task_space`, in the order compute_ee_motor takes them."""
    names = {joint.get_index(): joint.get_name() for joint in system.get_joints() if joint.get_index() >= 0}
    return ["q_" + names[int(i)] for i in task_space.get_joint_indices()]


def measured_displacements_with(trace, columns, fk, origin_position, origin_matrix):
    """The end effector's displacement from `(origin_position, origin_matrix)` at every sample of a trace read by `read_trace`, as (N, 6) in the order
    of DOFS: the position difference in the world frame (m), then the rotation vector of R R_origin^T (rad), also in the world frame.

    `trace` maps each name in `columns` to an equal-length 1-D array (rad); `columns` must be in the exact joint order expected by `fk(q)`.
    `fk(q)` returns `(position (3,), rotation_matrix (3, 3))` in the world frame (same signature as `payload.ArmModel.fk`).
    `origin_position` is (3,) in metres and `origin_matrix` is a (3, 3) proper rotation matrix."""
    origin_position = np.asarray(origin_position, dtype=float)
    origin_matrix = np.asarray(origin_matrix, dtype=float)
    joints = np.column_stack([trace[c] for c in columns])
    poses = [fk(q) for q in joints]
    position = np.array([p for p, _ in poses], dtype=float)
    matrices = np.array([r for _, r in poses], dtype=float)
    rotation = Rotation.from_matrix(matrices @ origin_matrix.T).as_rotvec()
    return np.hstack([position - origin_position, rotation])


def measured_displacements(trace, columns, task_space, origin):
    """The end effector's displacement from `origin` at every sample of a trace read by `read_trace`, using a gafro `task_space` and `origin` motor."""
    def fk(q):
        motor = task_space.compute_ee_motor(q)
        return motor.get_translator().to_translation_vector(), motor.get_rotor().to_rotation_matrix()

    return measured_displacements_with(
        trace,
        columns,
        fk,
        origin.get_translator().to_translation_vector(),
        origin.get_rotor().to_rotation_matrix(),
    )


def coupling(responses, dof, minimum_quality=0.5):
    """The off-axis gains of a run that excited `dof`: for every other DOF in `responses` (a `{dof: FrequencyResponse}` mapping), the median |H| over
    the frequencies where `responses[dof].quality >= minimum_quality` (in [0, 1]), in SI units of output per unit of input (m/m, rad/rad, m/rad, rad/m)."""
    own = responses[dof]
    good = own.quality >= minimum_quality
    return {other: float(np.median(r.gain[good])) for other, r in responses.items() if other != dof and good.any()}
