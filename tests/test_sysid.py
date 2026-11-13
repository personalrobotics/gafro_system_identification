"""Synthetic unit tests for robot-agnostic `gafro_system_identification`."""

import numpy as np
import pytest
from scipy import signal

from gafro_system_identification import (
    actuator_friction as af,
    friction_opt,
    identify,
    response,
    run,
    trace,
)


def test_second_order_sweep_identification():
    wn_true, zeta_true, delay_true = 18.0, 0.7, 0.012
    fs = 500.0
    sweep = response.Sweep(amplitude=0.02, f_start=0.5, f_end=8.0, duration=8.0, ramp=0.5)
    t = np.arange(0.0, sweep.total, 1.0 / fs)
    r = sweep.amplitude * sweep.envelope(t) * np.sin(sweep.phase(t))

    sys = signal.TransferFunction([wn_true**2], [1.0, 2.0 * zeta_true * wn_true, wn_true**2])
    _, y_nodelay, _ = signal.lsim(sys, U=r, T=t)
    delay_samples = int(round(delay_true * fs))
    y = np.concatenate([np.zeros(delay_samples), y_nodelay[:-delay_samples]])

    fr = response.frequency_response(t, r, y, sweep, points=24)
    fit = response.fit_second_order(fr)

    assert fit.wn == pytest.approx(wn_true, rel=0.08)
    assert fit.zeta == pytest.approx(zeta_true, rel=0.10)
    assert fit.delay == pytest.approx(delay_true, abs=0.004)
    assert np.median(fr.quality) > 0.98


def test_actuator_friction_7dof_recovery():
    coulomb_true, viscous_true, gravity_true = 0.85, 0.14, -1.2
    n_joints = 7
    joint = 3
    plans = [
        af.PassPlan(joint=joint, direction=d, speed=v, start=-0.2 * d, constant=0.6, ramp=0.2)
        for v in (0.1, 0.3, 0.5)
        for d in (1, -1)
    ]

    class MockArm:
        def __init__(self):
            self.q = np.zeros(n_joints)
            self.qd = np.zeros(n_joints)

        def get_state(self):
            tau = np.zeros(n_joints)
            v = self.qd[joint]
            if abs(v) > 1e-6:
                tau[joint] = gravity_true + np.sign(v) * (coulomb_true + viscous_true * abs(v))
            return run.JointState(self.q.copy(), self.qd.copy(), tau)

        def follow_path(self, times, positions, velocities=None, **play):
            for idx in range(len(times)):
                self.q = np.asarray(positions[idx], dtype=float)
                self.qd = np.asarray(velocities[idx], dtype=float) if velocities is not None else np.zeros(n_joints)
                if idx + 1 < len(times):
                    import time
                    time.sleep(max(0.0, float(times[idx + 1] - times[idx])))
            return run.FollowResult(ok=True, status="OK")

    adapter = MockArm()
    fast_plans = [
        af.PassPlan(joint=p.joint, direction=p.direction, speed=p.speed, start=p.start, constant=0.04, ramp=0.01)
        for p in plans[:2]
    ]
    played, dropped = run.run(adapter, fast_plans, period=0.001, minimum_move=0.01)
    assert not dropped
    assert len(played) == len(fast_plans)

    # Also test exact steady-state recording directly with identify_recordings
    records = []
    hold = np.zeros(n_joints)
    for p in plans:
        t, positions, rates, _ = af.pass_path(p, hold, period=0.005)
        tau = np.zeros_like(positions)
        qd = rates[:, joint]
        tau[:, joint] = gravity_true + np.sign(qd) * (coulomb_true + viscous_true * np.abs(qd))
        cols = np.column_stack([t, positions, rates, tau])
        rec = run.recording(cols, n_joints)
        w = af.steady_window(t, qd, p, tolerance=0.1)
        assert w is not None
        records.append((af.Pass(joint, p.direction, p.speed, *w), rec))

    table = af.identify_recordings(
        records, joint_names=run.joint_names(n_joints), torque="tau{}", velocity="qd{}", stamp="t"
    )
    row = table[joint]
    assert row["coulomb"] == pytest.approx(coulomb_true, abs=1e-3)
    assert row["viscous"] == pytest.approx(viscous_true, abs=1e-3)
    assert row["gravity"] == pytest.approx(gravity_true, abs=1e-3)


def test_friction_opt_supports_7dof():
    viscous = np.linspace(0.1, 0.7, 7)
    coulomb = np.linspace(0.2, 0.8, 7)
    theta = friction_opt.pack(viscous, coulomb)
    assert theta.shape == (14,)
    v_out, c_out = friction_opt.unpack(theta)
    assert np.allclose(v_out, viscous)
    assert np.allclose(c_out, coulomb)

    v_out_explicit, c_out_explicit = friction_opt.unpack(theta, n_joints=7)
    assert np.allclose(v_out_explicit, viscous)
    assert np.allclose(c_out_explicit, coulomb)


def test_measured_displacements_with_generic_fk():
    t_trace = {"q1": np.array([0.0, 0.1, 0.2]), "q2": np.array([0.0, 0.0, 0.0])}

    def fk(q):
        return np.array([q[0], 0.0, 0.5]), np.eye(3)

    disp = trace.measured_displacements_with(t_trace, ["q1", "q2"], fk, [0.0, 0.0, 0.5], np.eye(3))
    assert disp.shape == (3, 6)
    assert np.allclose(disp[:, 0], [0.0, 0.1, 0.2])
    assert np.allclose(disp[:, 1:], 0.0)


def test_identify_plant_diagonal():
    gains = identify.PoseGains.uniform(omega=15.0, zeta=0.7)
    freqs = np.geomspace(0.5, 10.0, 20)
    h_model = identify.closed_loop_response(np.eye(6), tau=0.01, C=np.zeros(6), gains=gains, frequency=freqs)
    excitations = [
        identify.Excitation(dof=d, amplitude=0.01, frequency=freqs, response=h_model[:, :, i], quality=np.ones((len(freqs), 6)))
        for i, d in enumerate(trace.DOFS)
    ]
    rset = identify.ResponseSet(excitations=excitations, feedforward="off")
    plant = identify.fit_plant(rset, gains, diagonal=True)
    assert np.allclose(plant.A, np.eye(6), atol=1e-2)
    assert plant.tau[0] == pytest.approx(0.01, abs=1e-3)
