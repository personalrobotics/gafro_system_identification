"""Reading a run's trace: the cycle log of a robot interface, the pose read off it, and the reference's time rebuilt from a publish log.

The six degrees of freedom of the pose, in the order of the displacements below, are a translation along a world axis (m) or a rotation about a world axis
through the end effector's origin (rad). Nothing here needs a robot: `joint_columns` and `measured_displacements` take a gafro System and task space (or anything
with their methods) as arguments.
"""

import numpy as np
from scipy.spatial.transform import Rotation

DOFS = ("x", "y", "z", "rx", "ry", "rz")


def reference_times(log, period, wall):
    """The path time the reference was at, at each controller-clock stamp in `wall`, from a PathFollower's `log`: the tick that
    published the latest chunk before the stamp, plus the time since it (a chunk lasts one `period`, then the reference
    holds). NaN before the first chunk and after the last one's period."""
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


def measured_displacements(trace, columns, task_space, origin):
    """The end effector's displacement from `origin` at every sample of a trace read by `read_trace`, as (N, 6) in the order of DOFS: the
    position difference in the world frame (m), then the rotation vector of R R_origin^T (rad), also in the world frame."""
    origin_position = np.asarray(origin.get_translator().to_translation_vector(), dtype=float)
    origin_matrix = np.asarray(origin.get_rotor().to_rotation_matrix(), dtype=float)
    joints = np.column_stack([trace[c] for c in columns])
    motors = [task_space.compute_ee_motor(q) for q in joints]
    position = np.array([m.get_translator().to_translation_vector() for m in motors], dtype=float)
    matrices = np.array([m.get_rotor().to_rotation_matrix() for m in motors], dtype=float)
    rotation = Rotation.from_matrix(matrices @ origin_matrix.T).as_rotvec()
    return np.hstack([position - origin_position, rotation])


def coupling(responses, dof, minimum_quality=0.5):
    """The off-axis gains of a run that excited `dof`: for every other DOF, the median |H| over the frequencies where the run's own response is
    clean, in SI units of output per unit of input (m/m, rad/rad, m/rad, rad/m)."""
    own = responses[dof]
    good = own.quality >= minimum_quality
    return {other: float(np.median(r.gain[good])) for other, r in responses.items() if other != dof and good.any()}
