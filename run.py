"""Friction passes played through a transport-agnostic `RobotAdapter`, each recorded from the robot's own state.

This is the glue, not a second client: the paths are `actuator_friction`'s, the playing, refusals and outcome are the adapter's `follow_path`,
the state is `JointState` (position, velocity and torque), and the fit is `identify_recordings`. Nothing here imports gafro, control-nodes,
ROS 2, or Dora: any transport plugs in by implementing `RobotAdapter` (two methods: `get_state` and `follow_path`).

`GafroNodeAdapter` wraps an in-process `gafro_control_nodes.RobotNode` for one arm. An out-of-process client (e.g. OpenArm over Dora or ROS 2
channels `<arm>/<input>/trajectory`, `<arm>/mode`, `<arm>/state`) implements the same `RobotAdapter` protocol. What is shared across all
transports is the part that matters: a pass is played from its first pose, the state is recorded as `t, q1.., qd1.., tau1..` with `t` from
the start of the recording, and the steady window is found in the recording from the joint's own velocity (`steady_window`), so no clock
has to be shared with the player.
"""

from dataclasses import dataclass
import threading
import time
from typing import Protocol, runtime_checkable

import numpy as np

from gafro_system_identification import actuator_friction as af


@dataclass(frozen=True)
class JointState:
    """One sample of an arm's N joints: position (rad), velocity (rad/s), and measured torque (N m)."""

    position: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray


@dataclass(frozen=True)
class FollowResult:
    """Outcome of a trajectory playback request."""

    ok: bool
    status: str = ""
    message: str = ""


@runtime_checkable
class RobotAdapter(Protocol):
    """Minimal interface a robot or middleware transport implements to run friction passes."""

    def get_state(self) -> JointState:
        """Current `(position, velocity, torque)` of the controlled arm's N joints."""
        ...

    def follow_path(self, times, positions, velocities=None, **play) -> FollowResult:
        """Play a joint-space trajectory (`times` (T,) s, `positions` (T, N) rad[, `velocities` (T, N) rad/s], `**play` transport options) and return its `FollowResult`."""
        ...


class GafroNodeAdapter:
    """Wraps an in-process `gafro_control_nodes.RobotNode` and arm name as a `RobotAdapter`."""

    def __init__(self, node, arm):
        self.node = node
        self.arm = arm
        self.indices = np.asarray(node.controlled[arm])

    def get_state(self) -> JointState:
        state = self.node.interface.get_state()
        take = lambda vector: np.asarray(vector, dtype=float).ravel()[self.indices]  # noqa: E731
        return JointState(take(state.get_position()), take(state.get_velocity()), take(state.get_torque()))

    def follow_path(self, times, positions, velocities=None, **play) -> FollowResult:
        if velocities is None:
            result = self.node.follow_path(self.arm, times, positions, **play)
        else:
            result = self.node.follow_path(self.arm, times, positions, velocities, **play)
        return FollowResult(ok=bool(result.ok), status=str(result.status), message=str(result.message))


def _as_adapter(robot, arm=None):
    if isinstance(robot, RobotAdapter):
        return robot
    if arm is None:
        raise TypeError("pass a RobotAdapter instance, or (node, arm) for a gafro_control_nodes.RobotNode")
    return GafroNodeAdapter(robot, arm)


class Sampler:
    """Reads state every `period` (> 0, s) seconds in a thread: rows of (t, q, qd, tau), `t` from `start()`.

    Accepts either `Sampler(adapter, period=0.002)` for a `RobotAdapter` or the legacy `Sampler(interface, indices, period=0.002)`
    where `indices` is a sequence of 0-based joint indices into `interface.get_state()` vectors."""

    def __init__(self, adapter_or_interface, indices_or_period=0.002, period=0.002):
        if isinstance(adapter_or_interface, RobotAdapter):
            self.adapter = adapter_or_interface
            self.interface = None
            self.indices = None
            self.period = float(indices_or_period)
        else:
            self.adapter = None
            self.interface = adapter_or_interface
            self.indices = np.asarray(indices_or_period)
            self.period = float(period)
        self.rows = []
        self._n_joints = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._origin = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            if self.adapter is not None:
                state = self.adapter.get_state()
                q = np.asarray(state.position, dtype=float).ravel()
                qd = np.asarray(state.velocity, dtype=float).ravel()
                tau = np.asarray(state.torque, dtype=float).ravel()
            else:
                state = self.interface.get_state()
                take = lambda vector: np.asarray(vector, dtype=float).ravel()[self.indices]  # noqa: E731
                q, qd, tau = take(state.get_position()), take(state.get_velocity()), take(state.get_torque())
            self._n_joints = q.size
            self.rows.append(np.concatenate([[time.monotonic() - self._origin], q, qd, tau]))
            time.sleep(self.period)

    def stop(self):
        self._stop.set()
        self._thread.join()
        if self._n_joints is not None:
            count = self._n_joints
        elif self.adapter is not None:
            count = np.asarray(self.adapter.get_state().position).size
        else:
            count = len(self.indices)
        return recording(self.rows, count)


def recording(rows, count):
    """The rows as an `actuator_friction.Recording` with the columns `t`, `q1..`, `qd1..`, `tau1..`."""
    names = ["t"] + [f"{kind}{j + 1}" for kind in ("q", "qd", "tau") for j in range(count)]
    return af.Recording(names, np.asarray(rows, dtype=float).reshape(-1, 1 + 3 * count))


def joint_names(count):
    return tuple(str(j + 1) for j in range(count))


def go_to(robot, *args, seconds=None, speed=0.3, minimum=2.0):
    """Move to the joint vector `target` ((N,) rad) along a straight line at about `speed` (> 0, rad/s) (at least `minimum` > 0 s, or exact
    `seconds` > 0), forwarding `play` (dict or None) kwargs to `follow_path`.

    Call forms (`target` and `play` are positional-only via `*args` and cannot be passed by keyword):
      - `go_to(robot, target[, play], seconds=..., speed=..., minimum=...)` where `robot` is a `RobotAdapter`
      - legacy `go_to(robot, arm, target[, play], ...)` where `robot` is a `gafro_control_nodes.RobotNode`"""
    if isinstance(robot, RobotAdapter):
        adapter = robot
        target = args[0]
        play = args[1] if len(args) > 1 else None
    else:
        arm, target = args[0], args[1]
        play = args[2] if len(args) > 2 else None
        adapter = GafroNodeAdapter(robot, arm)
    here = np.asarray(adapter.get_state().position, dtype=float).ravel()
    seconds = seconds or max(minimum, float(np.abs(np.asarray(target) - here).max()) / speed)
    return adapter.follow_path([0.0, seconds], [here, np.asarray(target, dtype=float)], **(play or {}))


def play_pass(robot, *args, period=0.002, tolerance=0.1, minimum_move=2.0):
    """Play one `actuator_friction.PassPlan`, returning `(Pass or None, Recording, FollowResult, why)`: go to its first pose, play it while
    sampling at `period` (> 0, s), and read its steady window with relative `tolerance` (> 0).

    `hold` is the (N,) joint vector (rad) for un-driven joints (None samples current position); `play` is a dict or None of `follow_path` kwargs;
    `minimum_move` (> 0, s) is the minimum duration for moving to the start pose. `why` is None when there is a window and otherwise says what
    happened.

    Call forms (`arm`, `plan`, `play`, and `hold` are positional-only via `*args` and cannot be passed by keyword):
      - `play_pass(robot, plan[, play[, hold]], period=..., tolerance=..., minimum_move=...)` where `robot` is a `RobotAdapter`
      - legacy `play_pass(robot, arm, plan[, play[, hold]], ...)` where `robot` is a `gafro_control_nodes.RobotNode`"""
    if isinstance(robot, RobotAdapter):
        adapter = robot
        plan = args[0]
        play = args[1] if len(args) > 1 else None
        hold = args[2] if len(args) > 2 else None
    else:
        arm, plan = args[0], args[1]
        play = args[2] if len(args) > 2 else None
        hold = args[3] if len(args) > 3 else None
        adapter = GafroNodeAdapter(robot, arm)
    if hold is None:
        hold = np.asarray(adapter.get_state().position, dtype=float).ravel()
    first = np.array(hold, dtype=float)
    first[plan.joint] = plan.start
    moved = go_to(adapter, first, play, minimum=minimum_move)
    if not moved.ok:
        return None, None, moved, f"going to the first pose: {moved.status} {moved.message}"
    times, positions, rates, _ = af.pass_path(plan, hold)
    sampler = Sampler(adapter, period).start()
    try:
        result = adapter.follow_path(times, positions, rates, **(play or {}))
    finally:
        taken = sampler.stop()
    column = taken.names.index(f"qd{plan.joint + 1}")
    window = af.steady_window(taken.cycles[:, 0], taken.cycles[:, column], plan, tolerance)
    if window is None:
        return None, taken, result, f"joint {plan.joint + 1} never held {plan.direction * plan.speed:+.3f} rad/s ({result.status})"
    return af.Pass(plan.joint, plan.direction, plan.speed, *window), taken, result, None


def run(robot, *args, hold=None, period=0.002, on_pass=None, minimum_move=2.0):
    """Play every `actuator_friction.PassPlan` in `plans`; returns `(played, dropped)` with `played` = [(Pass, Recording)] for
    `identify_recordings` and `dropped` = [(plan, why)].

    The other joints stay where they were when the run began (`hold` (N,) rad, or the measured pose if None). `play` is a dict or None of
    `follow_path` kwargs; `period` (> 0, s) is the sampler interval; `on_pass(plan, stretch_or_none, why_or_none)` is an optional callback after
    each pass. A move the adapter refuses or stops is not retried: it is dropped with its reason, and the arm has held where it is, so the next
    pass starts from there.

    Call forms (`arm`, `plans`, and `play` are positional-only via `*args` and cannot be passed by keyword):
      - `run(robot, plans[, play], hold=..., period=..., on_pass=..., minimum_move=...)` where `robot` is a `RobotAdapter`
      - legacy `run(robot, arm, plans[, play], ...)` where `robot` is a `gafro_control_nodes.RobotNode`"""
    if isinstance(robot, RobotAdapter):
        adapter = robot
        plans = args[0]
        play = args[1] if len(args) > 1 else None
    else:
        arm, plans = args[0], args[1]
        play = args[2] if len(args) > 2 else None
        adapter = GafroNodeAdapter(robot, arm)
    if hold is None:
        hold = np.asarray(adapter.get_state().position, dtype=float).ravel()
    played, dropped = [], []
    for plan in plans:
        stretch, taken, result, why = play_pass(adapter, plan, play, hold, period=period, minimum_move=minimum_move)
        (played.append((stretch, taken)) if stretch is not None else dropped.append((plan, why)))
        if on_pass:
            on_pass(plan, stretch, why)
    return played, dropped
