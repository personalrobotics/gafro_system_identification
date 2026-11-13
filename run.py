"""Friction passes played through a `gafro_control_nodes.RobotNode` in this process, each recorded from the robot's own state.

This is the glue, not a second client: the paths are `actuator_friction`'s, the playing, refusals and outcome are `RobotNode.follow_path`'s, the
state is the interface's `SystemState` (position, velocity and torque, the same one `<arm>/state` carries), and the fit is `identify_recordings`.
Nothing here imports gafro or control-nodes at module level: the node is handed in, so the package's analysis still imports without them.

A node in another process (the OpenArm's service) is reached through the same channels, `<arm>/<input>/trajectory`, `<arm>/mode` and
`<arm>/state`; that client is not here yet. What is shared with it is the part that matters: a pass is played from its first pose, the state is
recorded as `t, q1.., qd1.., tau1..` with `t` from the start of the recording, and the steady window is found in the recording from the joint's
own velocity (`steady_window`), so no clock has to be shared with the player.
"""

import threading
import time

import numpy as np

from gafro_system_identification import actuator_friction as af


class Sampler:
    """Reads the interface's state every `period` seconds in a thread: rows of (t, q, qd, tau) of the `indices` joints, `t` from `start()`."""

    def __init__(self, interface, indices, period=0.002):
        self.interface, self.indices, self.period = interface, np.asarray(indices), period
        self.rows = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._origin = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            state = self.interface.get_state()
            take = lambda vector: np.asarray(vector, dtype=float).ravel()[self.indices]  # noqa: E731
            self.rows.append(np.concatenate([[time.monotonic() - self._origin], take(state.get_position()), take(state.get_velocity()),
                                             take(state.get_torque())]))
            time.sleep(self.period)

    def stop(self):
        self._stop.set()
        self._thread.join()
        return recording(self.rows, len(self.indices))


def recording(rows, count):
    """The rows as an `actuator_friction.Recording` with the columns `t`, `q1..`, `qd1..`, `tau1..` (the OpenArm service's names)."""
    names = ["t"] + [f"{kind}{j + 1}" for kind in ("q", "qd", "tau") for j in range(count)]
    return af.Recording(names, np.asarray(rows, dtype=float).reshape(-1, 1 + 3 * count))


def joint_names(count):
    return tuple(str(j + 1) for j in range(count))


def go_to(node, arm, target, play, seconds=None, speed=0.3, minimum=2.0):
    """Move `arm` to the joint vector `target` along a straight line at about `speed` rad/s (at least `minimum` s), by the same `play` arguments."""
    here = np.asarray(node.interface.get_state().get_position(), dtype=float).ravel()[node.controlled[arm]]
    seconds = seconds or max(minimum, float(np.abs(np.asarray(target) - here).max()) / speed)
    return node.follow_path(arm, [0.0, seconds], [here, np.asarray(target, dtype=float)], **play)


def play_pass(node, arm, plan, play, hold, period=0.002, tolerance=0.1, minimum_move=2.0):
    """Play one pass, `(Pass or None, Recording, FollowResult, why)`: go to its first pose, play it, and read its window from the recording.

    `why` is None when there is a window and otherwise says what happened (what the move reported, or that the joint never held the speed)."""
    first = np.array(hold, dtype=float)
    first[plan.joint] = plan.start
    moved = go_to(node, arm, first, play, minimum=minimum_move)
    if not moved.ok:
        return None, None, moved, f"going to the first pose: {moved.status} {moved.message}"
    times, positions, rates, _ = af.pass_path(plan, hold)
    sampler = Sampler(node.interface, node.controlled[arm], period).start()
    try:
        result = node.follow_path(arm, times, positions, rates, **play)
    finally:
        taken = sampler.stop()
    column = taken.names.index(f"qd{plan.joint + 1}")
    window = af.steady_window(taken.cycles[:, 0], taken.cycles[:, column], plan, tolerance)
    if window is None:
        return None, taken, result, f"joint {plan.joint + 1} never held {plan.direction * plan.speed:+.3f} rad/s ({result.status})"
    return af.Pass(plan.joint, plan.direction, plan.speed, *window), taken, result, None


def run(node, arm, plans, play, hold=None, period=0.002, on_pass=None, minimum_move=2.0):
    """Play every plan; `(played, dropped)` with `played` = [(Pass, Recording)] for `identify_recordings` and `dropped` = [(plan, why)].

    The other joints stay where they were when the run began (`hold`, or the measured pose). A move the node refuses or stops is not retried:
    it is dropped with its reason, and the arm has held where it is, so the next pass starts from there."""
    count = len(node.controlled[arm])
    if hold is None:
        hold = np.asarray(node.interface.get_state().get_position(), dtype=float).ravel()[node.controlled[arm]]
    played, dropped = [], []
    for plan in plans:
        stretch, taken, result, why = play_pass(node, arm, plan, play, hold, period, minimum_move=minimum_move)
        (played.append((stretch, taken)) if stretch is not None else dropped.append((plan, why)))
        if on_pass:
            on_pass(plan, stretch, why)
    return played, dropped
