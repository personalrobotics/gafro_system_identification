"""What an error in the tool's mass and centre of mass does to the pose loop -- and, run backwards, what an identified loop says about the tool.

The pose stage commands a task-space acceleration `a`; the controller turns it into joint torques with its model's operational-space inertia, so the wrench is
Lambda_model a. The real arm answers with its own inertia: pose'' = J M_real^-1 J^T Lambda_model a = A a, so

    A = (J M_real^-1 J^T) (J M_model^-1 J^T)^-1  =  Lambda_real^-1 Lambda_model

with `J` the task Jacobian in the sweeps' coordinates (`sweep_run.DOFS`: world translation of the end effector, rotation about world axes) and `M` the mass
matrix of the arm's joints at the posture. A is the identity for a model that is right. A tool whose mass or centre of mass is not where the model has it
changes M by the tool's own term, m J_c^T J_c + J_w^T I_w J_w (its centre of mass moving with the end effector, and its inertia about it), so `A` becomes
non-diagonal in a way that says where: an offset along the tool's x couples y into rx, and so on. `loop_gain` computes it, `fit_tool_error` finds the mass and
the centre-of-mass error that reproduces an identified A (identify.py): the tool's own error only -- friction, the other links and the arm's own joint control
produce coupling too, so a good fit is consistent with a tool error, not proof of one.

The arm is two callables, so that nothing here needs the robot: `fk(q) -> (position, rotation matrix)` of the end effector for the arm's controlled joints,
and `mass_matrix(q)` for the same joints (`ArmModel.from_gafro` makes them from a gafro System and task space). The controller's regularisation (a small
damping of the inertia's inversion) is not modelled: it changes A by about its own square over the inertia, a percent or two on the translations.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

STEP = 1e-6  # rad: the finite differences of the kinematics


@dataclass(frozen=True)
class Tool:
    """What hangs on the arm's flange: `mass` (kg), `com` (m) and `inertia` about it (kg m^2, 3x3), in the end effector's frame (axes and origin)."""

    mass: float
    com: np.ndarray
    inertia: np.ndarray

    def __str__(self):
        return f"{self.mass * 1e3:.0f} g, CoM {np.round(self.com * 1e3, 1).tolist()} mm from the end effector"


@dataclass
class ArmModel:
    """The arm as the analysis needs it: `fk(q)` -> (position (3,), rotation matrix (3, 3)) in the world frame (m, proper rotation),
    `mass_matrix(q)` -> (n, n) symmetric positive-definite joint-space inertia (kg m^2 / kg), both for the n joints the controller moves.

    `q` (n,) and `armature` (n,) (reflected rotor inertia in kg m^2, or None) must be ordered **base-to-tip** along the arm; `Posture.gain` and
    `fit_tool_error` treat the last `min(3, n)` entries as the wrist joints and the preceding `n - min(3, n)` entries as the base joints."""

    fk: object
    mass_matrix: object
    armature: np.ndarray = None  # (n,) the model's armature (reflected rotor inertia, kg m^2) of each joint, when known: it is a part of the mass matrix's diagonal

    @classmethod
    def from_gafro(cls, system, task_space, armature=None):
        """From a gafro System and the arm's task space: the joints it controls, the others (a rail, the other arm) at the system's default. `armature`: the
        model's armature of those joints (`joint_armature`)."""
        joints = np.asarray(task_space.get_joint_indices())
        controlled = np.asarray(task_space.get_controlled_joint_indices())
        base = np.asarray(system.get_default_configuration(), dtype=float)

        def full(q):
            configuration = base.copy()
            configuration[controlled] = q
            return configuration

        def fk(q):
            motor = task_space.compute_ee_motor(full(q)[joints])
            return (np.asarray(motor.get_translator().to_translation_vector(), dtype=float),
                    np.asarray(motor.get_rotor().to_rotation_matrix(), dtype=float))

        def mass_matrix(q):
            return np.asarray(system.compute_mass_matrix(full(q)), dtype=float)[np.ix_(controlled, controlled)]

        return cls(fk, mass_matrix, None if armature is None else np.asarray(armature, dtype=float))


def task_jacobian(fk, q, step=STEP):
    """(6, n): how the end effector's pose moves with the n joints `q` (rad), in the coordinates of the sweeps (`trace.DOFS`: world-frame
    displacement of its origin in m and world-frame rotation vector in rad), by central differences of `fk(q)` with finite-difference `step` (> 0, rad)."""
    q = np.asarray(q, dtype=float)
    columns = []
    for k in range(len(q)):
        up, down = q.copy(), q.copy()
        up[k] += step
        down[k] -= step
        (p_up, r_up), (p_down, r_down) = fk(up), fk(down)
        columns.append(np.concatenate([(p_up - p_down) / (2 * step), Rotation.from_matrix(r_up @ r_down.T).as_rotvec() / (2 * step)]))
    return np.column_stack(columns)


def _centre_jacobian(fk, q, offset, step=STEP):
    """(3, n): the world-frame translation Jacobian of the point `offset` ((3,) m, in the end effector's frame) at joint vector `q` ((n,) rad)."""
    q = np.asarray(q, dtype=float)
    columns = []
    for k in range(len(q)):
        up, down = q.copy(), q.copy()
        up[k] += step
        down[k] -= step
        (p_up, r_up), (p_down, r_down) = fk(up), fk(down)
        columns.append(((p_up + r_up @ offset) - (p_down + r_down @ offset)) / (2 * step))
    return np.column_stack(columns)


def tool_mass_matrix(arm, q, tool):
    """The `tool`'s (`Tool`) own (n, n) contribution to `arm`'s (`ArmModel`) joint-space mass matrix at posture `q` ((n,) rad):
    m J_c^T J_c + J_w^T (R I R^T) J_w."""
    posture = Posture.__new__(Posture)
    posture.arm, posture.q = arm, np.asarray(q, dtype=float)
    posture.jacobian, posture.rotation = task_jacobian(arm.fk, posture.q), arm.fk(posture.q)[1]
    return posture._tool_term(tool)


class Posture:
    """What `loop_gain` needs of `arm` (`ArmModel`) at one non-singular joint posture `q` ((n,) rad, base-to-tip), computed once: the (6, n) task
    Jacobian, the (n, n) mass matrix of the arm *without* its tool (`model_tool` subtracted) and the (6, 6) inverse of the model's task-space
    admittance J M^-1 J^T. `gain(tool)` is then A for an arm that carries `tool`."""

    def __init__(self, arm, q, model_tool):
        self.arm, self.q = arm, np.asarray(q, dtype=float)
        self.jacobian = task_jacobian(arm.fk, self.q)
        self.rotation = arm.fk(self.q)[1]
        model = arm.mass_matrix(self.q)
        self.without_tool = model - self._tool_term(model_tool)
        admittance = self._admittance(model)
        if np.linalg.cond(admittance) > 1e10:
            raise ValueError("posture q is singular or near-singular (task-space admittance condition number > 1e10); choose a non-singular posture")
        self.model_inverse = np.linalg.inv(admittance)

    def _admittance(self, mass):
        return self.jacobian @ np.linalg.solve(mass, self.jacobian.T)

    def _tool_term(self, tool):
        centre = _centre_jacobian(self.arm.fk, self.q, tool.com)
        inertia = self.rotation @ tool.inertia @ self.rotation.T
        return tool.mass * centre.T @ centre + self.jacobian[3:].T @ inertia @ self.jacobian[3:]

    def gain(self, tool, wrist_scale=1.0, base_scale=1.0, armature=None):
        """A (6x6, in `trace.DOFS` order) for an arm carrying `tool` whose joints have a different armature than the model's.

        `base_scale` and `wrist_scale` (> 0) multiply `self.arm.armature`, assuming `self.q` and `self.arm.armature` are ordered **base-to-tip**:
        the last `min(3, n)` joints are scaled by `wrist_scale` and the preceding `n - min(3, n)` joints by `base_scale`. `armature` instead gives
        the arm's own (n,) array (kg m^2, same joint order as `q`; treats `self.arm.armature is None` as zero) and overrides the two scales."""
        mass = self.without_tool + self._tool_term(tool)
        if armature is not None:
            model_armature = 0.0 if self.arm.armature is None else np.asarray(self.arm.armature, dtype=float)
            mass = mass + np.diag(np.asarray(armature, dtype=float) - model_armature)
        elif (wrist_scale != 1.0 or base_scale != 1.0) and self.arm.armature is not None:
            n = len(self.q)
            n_wrist = min(3, n)
            scales = np.array([base_scale] * (n - n_wrist) + [wrist_scale] * n_wrist, dtype=float)
            mass = mass + np.diag((scales - 1.0) * self.arm.armature)
        return self._admittance(mass) @ self.model_inverse


def loop_gain(arm, q, model_tool, real_tool):
    """A (6x6, in the order of `trace.DOFS`): the pose's acceleration per commanded acceleration when the controller's model (`arm`: `ArmModel`) has
    `model_tool` (`Tool`) and the arm carries `real_tool` (`Tool`), at the non-singular joint posture `q` ((n,) rad, base-to-tip). Rows are the
    pose's axes (outputs), columns the command's."""
    return Posture(arm, q, model_tool).gain(real_tool)


@dataclass
class ToolError:
    """What `fit_tool_error` found: the tool's mass is `mass_error` kg more than the model's and its centre of mass `com_error` ((3,) m, end effector
    frame) further, each with its standard error; `predicted` is the (6, 6) A that tool gives, `explained` the share of A's departure from the
    identity it accounts for (1: all of it, 0: none), `residual` the RMS error of the prediction in units of A's standard errors."""

    mass_error: float
    com_error: np.ndarray
    mass_se: float
    com_se: np.ndarray
    predicted: np.ndarray
    explained: float
    residual: float
    tool: Tool
    inertia_scale: float = 1.0
    inertia_se: float = 0.0
    wrist_scale: float = 1.0
    wrist_se: float = 0.0
    base_scale: float = 1.0
    base_se: float = 0.0

    def __str__(self):
        c, s = self.com_error * 1e3, self.com_se * 1e3
        return (f"tool {self.mass_error * 1e3:+.0f} +- {self.mass_se * 1e3:.0f} g, CoM moved by ({c[0]:+.1f} +- {s[0]:.1f}, {c[1]:+.1f} +- {s[1]:.1f}, "
                f"{c[2]:+.1f} +- {s[2]:.1f}) mm, tool inertia x{self.inertia_scale:.2f} +- {self.inertia_se:.2f}, wrist joints' armature x{self.wrist_scale:.2f} "
                f"+- {self.wrist_se:.2f}, base joints' armature x{self.base_scale:.2f} +- {self.base_se:.2f}: explains {self.explained:.0%} of A's departure "
                f"from the identity")


def fit_tool_error(arm, q, model_tool, A, A_se=None, scale=None, floor=0.02, fit_inertia=False, fit_armature=True, armature=None):
    """The mass and centre-of-mass error of the tool, the scale of the wrist/base joints' armature and, with `fit_inertia`, of the tool's, from an
    identified A (`identify.PlantFit.A`, (6, 6) in `trace.DOFS` order, with optional (6, 6) standard errors `A_se`).

    `arm` is an `ArmModel` and `q` is the (n,) joint posture (rad) ordered **base-to-tip** (when `fit_armature` is True and `armature` is None, the
    last `min(3, n)` joints are treated as the wrist and the preceding `n - min(3, n)` as the base). The model's tool is `model_tool`; the arm's is
    the same with a different mass and centre of mass. `fit_armature` (when `arm.armature` is known): one factor on the armature of the wrist joints
    and one on that of the base joints -- the armature is nearly all of a wrist joint's inertia (the generic 0.1 kg m^2 a model without the published
    values has is well below a published armature), so they set how much acceleration the arm gets per acceleration commanded. Their effect on A dwarfs
    the tool's: with the armature that far off, the tool's mass and centre of mass are not identifiable from A (the fit absorbs what the armature's two
    factors leave in them). With `armature` ((n,) kg m^2 in the same joint order as `q`, e.g. a published table) the arm's own is taken as known, not
    fitted, and the tool's error is found on top of it. `fit_inertia`: the tool's own inertia about its centre scaled, off by default: next to the
    wrist's armature it moves A by under 1%. `scale` is (6,) positive sweep amplitudes in SI units `(m, m, m, rad, rad, rad)` (default linear 1,
    angular 4), so m/rad and rad/m terms weigh alike, each over its standard error floored at `floor` (> 0)."""
    A = np.asarray(A, dtype=float)
    scale = np.array([1.0] * 3 + [4.0] * 3) if scale is None else np.asarray(scale, dtype=float)
    ratio = scale[None, :] / scale[:, None]
    sigma = np.maximum(ratio * (np.zeros((6, 6)) if A_se is None else np.asarray(A_se, dtype=float)), floor)

    def tool_of(p):
        return replace(model_tool, mass=model_tool.mass + p[0], com=model_tool.com + p[1:4], inertia=model_tool.inertia * p[4])

    use_armature = fit_armature and arm.armature is not None and armature is None

    posture = Posture(arm, q, model_tool)

    def residuals(p):
        return ((posture.gain(tool_of(p), p[5], p[6], armature) - A) * ratio / sigma).ravel()

    lower = np.array([-model_tool.mass + 0.02, -0.2, -0.2, -0.2, 0.05 if fit_inertia else 1.0 - 1e-9, 0.02 if use_armature else 1.0 - 1e-9, 0.02 if use_armature else 1.0 - 1e-9])
    upper = np.array([2.0, 0.2, 0.2, 0.2, 5.0 if fit_inertia else 1.0 + 1e-9, 12.0 if use_armature else 1.0 + 1e-9, 40.0 if use_armature else 1.0 + 1e-9])
    best = None
    for start in ([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [0.1, 0.02, 0.0, 0.0, 1.0, 0.3, 0.3], [-0.1, 0.0, 0.02, 0.0, 0.6, 0.1, 3.0], [0.0, -0.02, -0.02, 0.03, 1.6, 2.0, 10.0], [0.0, 0.0, 0.0, 0.0, 1.0, 2.1, 15.0]):
        result = least_squares(residuals, np.clip(start, lower, upper), bounds=(lower, upper), x_scale=[0.1, 0.02, 0.02, 0.02, 0.3, 0.5, 2.0])
        if best is None or result.cost < best.cost:
            best = result
    p = best.x
    free = np.array([True, True, True, True, fit_inertia, use_armature, use_armature])
    cov = np.zeros((7, 7))
    cov[np.ix_(free, free)] = np.linalg.pinv(best.jac[:, free].T @ best.jac[:, free]) * max(2.0 * best.cost / max(A.size - int(free.sum()), 1), 1.0)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    predicted = posture.gain(tool_of(p), p[5], p[6], armature)
    departure = np.sum(((A - np.eye(6)) * ratio / sigma) ** 2)
    explained = 1.0 - np.sum(((A - predicted) * ratio / sigma) ** 2) / max(departure, 1e-12)
    return ToolError(float(p[0]), p[1:4].copy(), float(se[0]), se[1:4].copy(), predicted, float(explained),
                     float(np.sqrt(np.mean(best.fun**2))), tool_of(p), float(p[4]), float(se[4]), float(p[5]), float(se[5]), float(p[6]), float(se[6]))


def sensitivity(arm, q, tool, com_step=0.01, mass_step=0.1, wrist_scales=(0.5, 0.25, 0.1), reference_armature=None, reference_label="reference armature"):
    """The (6, 6) A of `tool` (`Tool`) at posture `q` ((n,) rad, base-to-tip on `arm`: `ArmModel`) when shifted by `com_step` (> 0, m) along each
    axis, made `mass_step` (> 0, kg) heavier, with inertia scaled by 0.5 and 0.25, and -- when `arm.armature` is known -- with the distal wrist
    joints' (`min(3, n)`) armature scaled by each factor in `wrist_scales`: returns `{label: A}`. With `reference_armature` ((n,) kg m^2) adds one
    more case labelled `reference_label`."""
    cases = {f"CoM {sign}{axis} {com_step * 1e3:.0f} mm": replace(tool, com=tool.com + sign_value * com_step * np.eye(3)[k])
             for k, axis in enumerate("xyz") for sign, sign_value in (("+", 1.0), ("-", -1.0))}
    cases[f"mass +{mass_step * 1e3:.0f} g"] = replace(tool, mass=tool.mass + mass_step)
    cases["tool inertia x0.5"] = replace(tool, inertia=tool.inertia * 0.5)
    cases["tool inertia x0.25"] = replace(tool, inertia=tool.inertia * 0.25)
    posture = Posture(arm, q, tool)
    gains = {label: posture.gain(real) for label, real in cases.items()}
    if arm.armature is not None:
        for wrist_scale in wrist_scales:
            gains[f"wrist armature x{wrist_scale:g}"] = posture.gain(tool, wrist_scale)
        if reference_armature is not None and len(arm.armature) == len(reference_armature):
            gains[reference_label] = posture.gain(tool, armature=reference_armature)
    return gains


def _quaternion_matrix(orientation):
    return Rotation.from_quat([orientation["x"], orientation["y"], orientation["z"], orientation["w"]]).as_matrix()


def _pose(origin):
    position, orientation = origin["position"], origin["orientation"]
    matrix = np.eye(4)
    matrix[:3, :3] = _quaternion_matrix(orientation)
    matrix[:3, 3] = [position["x"], position["y"], position["z"]]
    return matrix


def _urdf_origin(element):
    xyz = [float(x) for x in (element.get("xyz") if element is not None and element.get("xyz") else "0 0 0").split()]
    rpy = [float(x) for x in (element.get("rpy") if element is not None and element.get("rpy") else "0 0 0").split()]
    qx, qy, qz, qw = Rotation.from_euler("xyz", rpy).as_quat()
    return {"position": {"x": xyz[0], "y": xyz[1], "z": xyz[2]}, "orientation": {"x": float(qx), "y": float(qy), "z": float(qz), "w": float(qw)}}


def _load_system(model_path):
    """Load `{"links": ..., "joints": ...}` from either a gafro model YAML or a URDF XML file at `model_path`."""
    path_str = str(model_path)
    with open(model_path, encoding="utf-8") as handle:
        head = handle.read(256).lstrip()
        handle.seek(0)
        if path_str.endswith((".urdf", ".xml")) or head.startswith("<"):
            import xml.etree.ElementTree as ET

            root = ET.parse(handle).getroot()
            links, joints = {}, {}
            for link in root.findall("link"):
                lname = link.get("name")
                entry = {"joints": []}
                inertial = link.find("inertial")
                if inertial is not None:
                    mass_el, inertia_el = inertial.find("mass"), inertial.find("inertia")
                    mass_val = float(mass_el.get("value", 0.0)) if mass_el is not None else 0.0
                    if mass_val > 0.0 and inertia_el is not None:
                        entry["inertial"] = {
                            "mass": mass_val,
                            "origin": _urdf_origin(inertial.find("origin")),
                            "inertia": {k: float(inertia_el.get(f"i{k}", 0.0)) for k in ("xx", "xy", "xz", "yy", "yz", "zz")},
                        }
                links[lname] = entry
            for joint in root.findall("joint"):
                jname = joint.get("name")
                parent = joint.find("parent").get("link")
                child = joint.find("child").get("link")
                dyn = joint.find("dynamics")
                arm_el = joint.find("armature")
                arm_val = float(dyn.get("armature", 0.0)) if dyn is not None and dyn.get("armature") else (
                    float(arm_el.get("value", 0.0)) if arm_el is not None else 0.0
                )
                joints[jname] = {"name": jname, "parent_link": parent, "child_link": child, "origin": _urdf_origin(joint.find("origin")),
                                 "dynamics": {"armature": arm_val}}
                links.setdefault(parent, {"joints": []})["joints"].append(jname)
            return {"links": links, "joints": joints}
        import yaml

        return yaml.safe_load(handle)["system"]


def distal_tool(model_path, wrist_link, end_effector_link):
    """The tool of an arm in a gafro model YAML or URDF XML at `model_path`: every link beyond the wrist (`wrist_link`, a link name, itself excluded)
    -- the gripper, or a camera and its mount -- lumped into one `Tool` (mass, centre of mass and inertia about it), in the frame of
    `end_effector_link` (must lie in the subtree rooted at `wrist_link`), at the zero configuration (the tool does not move relative to the end
    effector, unless it has joints of its own, which the zero configuration freezes)."""
    system = _load_system(model_path)
    links, joints = system["links"], {j["name"]: j for j in system["joints"].values()}
    wrist, end_effector = wrist_link, end_effector_link
    if wrist not in links:
        raise ValueError(f"{model_path}: no link {wrist!r}: is it the right arm's link name?")
    poses = {}

    def walk(name, pose):
        poses[name] = pose
        for joint in links[name].get("joints", []):
            walk(joints[joint]["child_link"], pose @ _pose(joints[joint]["origin"]))

    walk(wrist, np.eye(4))
    if end_effector not in poses:
        raise ValueError(f"{model_path}: no link {end_effector!r} beyond {wrist!r}")
    to_ee = np.linalg.inv(poses[end_effector])
    pieces = []
    for name, pose in poses.items():
        inertial = links[name].get("inertial")
        if name == wrist or not inertial or not inertial.get("mass"):
            continue
        frame = to_ee @ pose @ _pose(inertial["origin"])  # the inertial frame in the end effector's frame
        t = inertial["inertia"]
        tensor = np.array([[t["xx"], t["xy"], t["xz"]], [t["xy"], t["yy"], t["yz"]], [t["xz"], t["yz"], t["zz"]]], dtype=float)
        pieces.append((float(inertial["mass"]), frame[:3, 3], frame[:3, :3] @ tensor @ frame[:3, :3].T))
    if not pieces:
        raise ValueError(f"{model_path}: nothing with mass beyond {wrist!r}")
    mass = sum(m for m, _, _ in pieces)
    com = sum(m * c for m, c, _ in pieces) / mass
    inertia = sum(I + m * ((c - com) @ (c - com) * np.eye(3) - np.outer(c - com, c - com)) for m, c, I in pieces)
    return Tool(mass, com, inertia)


def joint_armature(model_path, arm_or_joint_names, joint_names=None):
    """The armature (kg m^2, `dynamics: armature`) the model YAML or URDF XML at `model_path` gives each joint in `joint_names` (full names, e.g.
    `'openarm_left_joint1'` or `'right_ur5e/wrist_1_joint'`, ordered base-to-tip to match `ArmModel.q`): the reflected rotor inertia that is part of
    every joint's mass-matrix diagonal. 0 for a joint that states none.

    Accepts either `joint_armature(model_path, joint_names)` or legacy `joint_armature(model_path, arm, joint_names)`."""
    if joint_names is None:
        joint_names = arm_or_joint_names
    joints = _load_system(model_path)["joints"]
    return np.array([float((joints[name].get("dynamics") or {}).get("armature", 0.0)) for name in joint_names])
