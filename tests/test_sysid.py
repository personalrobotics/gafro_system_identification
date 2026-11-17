"""Synthetic unit tests for robot-agnostic `gafro_system_identification`."""

import numpy as np
import pytest
from scipy import signal

from gafro_system_identification import (
    actuator_friction as af,
    analysis,
    friction_opt,
    identify,
    joint_friction as jf,
    payload,
    response,
    run,
    trace,
    vibration,
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


def test_payload_7dof_armature_and_tool_error():
    n_joints = 7

    def fk(q):
        pos = np.array([0.3 + 0.1 * q[0] + 0.05 * q[3], 0.2 * q[1] + 0.05 * q[4], 0.4 + 0.15 * q[2] + 0.05 * q[5]])
        rot = trace.Rotation.from_euler("xyz", [q[4] + 0.2 * q[6], q[5], q[3]]).as_matrix()
        return pos, rot

    def mass_matrix(q):
        return np.diag([1.5, 1.4, 1.0, 0.6, 0.2, 0.15, 0.1]) + 0.05 * np.ones((n_joints, n_joints))

    arm = payload.ArmModel(fk=fk, mass_matrix=mass_matrix, armature=np.full(n_joints, 0.08))
    q0 = np.zeros(n_joints)
    model_tool = payload.Tool(mass=0.2, com=np.array([0.0, 0.0, 0.05]), inertia=np.eye(3) * 1e-3)
    real_tool = payload.Tool(mass=0.45, com=np.array([0.02, -0.01, 0.05]), inertia=np.eye(3) * 1e-3)

    posture = payload.Posture(arm, q0, model_tool)
    a_scaled = posture.gain(real_tool, wrist_scale=1.2, base_scale=0.9)
    assert a_scaled.shape == (6, 6)

    a_true = payload.loop_gain(arm, q0, model_tool, real_tool)
    err = payload.fit_tool_error(arm, q0, model_tool, a_true, fit_armature=False)
    assert err.mass_error == pytest.approx(0.25, abs=1e-2)
    assert np.allclose(err.com_error, [0.02, -0.01, 0.0], atol=5e-3)


def test_joint_friction_7dof_and_vibration():
    v, c = jf.scales_of("off", n_joints=7)
    assert v.shape == (7,) and c.shape == (7,)

    wall = np.arange(0.0, 6.0, 1.0 / 500.0)
    qd = 1e-4 * np.sin(2.0 * np.pi * 2.0 * wall)[:, None] * np.ones((1, 7))
    qd[wall > 3.5, 2] += 0.08 * np.sin(2.0 * np.pi * 25.0 * wall[wall > 3.5])
    cycles = vibration.Cycles(
        wall=wall,
        period=np.full_like(wall, 0.002),
        compute=np.full_like(wall, 0.0005),
        qd=qd,
        joints=[f"j{i+1}" for i in range(7)],
    )
    rep = vibration.analyse(cycles)
    fast_band = next(b for b in rep.bands if b.name == "fast")
    assert fast_band.onset is not None
    assert fast_band.joints[0][0] == "j3"


def test_legacy_run_signature_compatibility():
    class LegacyState:
        def get_position(self):
            return np.zeros(7)

        def get_velocity(self):
            return np.full(7, 0.2)

        def get_torque(self):
            return np.full(7, 0.5)

    class LegacyInterface:
        def get_state(self):
            return LegacyState()

    class LegacyNode:
        def __init__(self):
            self.interface = LegacyInterface()
            self.controlled = {"left": np.arange(7)}

        def follow_path(self, arm, times, positions, velocities=None, **play):
            import time
            time.sleep(0.02)
            return run.FollowResult(ok=True, status="OK")

    node = LegacyNode()
    plan = af.PassPlan(joint=0, direction=1, speed=0.2, start=0.0, constant=0.02, ramp=0.005)
    played, dropped = run.run(node, "left", [plan], {}, period=0.001, minimum_move=0.005)
    assert not dropped
    assert len(played) == 1


def test_run_keyword_arguments_and_spsa_perturb():
    class QuickAdapter:
        def __init__(self):
            self.q = np.zeros(3)
            self.qd = np.zeros(3)

        def get_state(self):
            return run.JointState(self.q.copy(), self.qd.copy(), np.zeros(3))

        def follow_path(self, times, positions, velocities=None, **play):
            import time

            self.q = np.asarray(positions[-1], dtype=float)
            if velocities is not None and len(velocities) > 1:
                self.qd = np.asarray(velocities[len(velocities) // 2], dtype=float)
            time.sleep(0.015)
            return run.FollowResult(ok=True, status="OK")

    adapter = QuickAdapter()
    res = run.go_to(adapter, target=np.array([0.1, 0.0, 0.0]), play={"gain": 1.0}, minimum=0.005)
    assert res.ok
    plan = af.PassPlan(joint=0, direction=1, speed=0.2, start=0.0, constant=0.02, ramp=0.005)
    played, dropped = run.run(adapter, plans=[plan], play={}, hold=np.zeros(3), period=0.001, tolerance=0.2, minimum_move=0.005)
    assert not dropped and len(played) == 1

    z0 = friction_opt.to_logit(np.full(14, 0.5))
    z_plus, z_minus, ck, delta = friction_opt.spsa_perturb(z0, k=0, seed=42)
    # Quadratic loss around z_target = 0.2
    z_target = np.full(14, 0.2)
    cost_p = float(np.sum((z_plus - z_target) ** 2))
    cost_m = float(np.sum((z_minus - z_target) ** 2))
    z1, _, _, delta_used = friction_opt.spsa_step(z0, cost_p, cost_m, k=0, delta=delta)
    assert np.array_equal(delta, delta_used)
    assert np.linalg.norm(z1 - z_target) < np.linalg.norm(z0 - z_target)


def test_opendubs_urdf_distal_tool_and_armature():
    import pathlib

    urdf_path = pathlib.Path(__file__).resolve().parents[2] / "opendubs" / "description" / "openarm" / "urdf" / "v1.urdf"
    if not urdf_path.is_file():
        pytest.skip("opendubs v1.urdf not present")

    # Lump fingers beyond openarm_left_link7 in the frame of openarm_left_hand_tcp
    fingers_tool = payload.distal_tool(urdf_path, "openarm_left_link7", "openarm_left_hand_tcp")
    assert fingers_tool.mass == pytest.approx(2 * 0.03602545343277134, rel=1e-4)
    assert fingers_tool.inertia.shape == (3, 3)

    # Lump link7 + fingers beyond openarm_left_link6 in the frame of openarm_left_hand_tcp
    hand_tool = payload.distal_tool(urdf_path, "openarm_left_link6", "openarm_left_hand_tcp")
    assert hand_tool.mass == pytest.approx(0.4659771327380578 + 2 * 0.03602545343277134, rel=1e-4)

    joint_names = [f"openarm_left_joint{i}" for i in range(1, 8)]
    armature = payload.joint_armature(urdf_path, joint_names)
    assert armature.shape == (7,)


def test_opendubs_mujoco_sim_integration():
    import pathlib
    import sys

    opendubs_root = pathlib.Path(__file__).resolve().parents[2] / "opendubs"
    sim_dir = opendubs_root / "sim"
    venv_site = opendubs_root / ".venv" / "lib" / "python3.12" / "site-packages"
    if venv_site.is_dir() and str(venv_site) not in sys.path:
        sys.path.insert(0, str(venv_site))
    if sim_dir.is_dir() and str(sim_dir) not in sys.path:
        sys.path.insert(0, str(sim_dir))

    mujoco = pytest.importorskip("mujoco")
    scene = pytest.importorskip("scene")

    m = scene.build(None)
    d = mujoco.MjData(m)
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"oa_openarm_left_joint{i}") for i in range(1, 8)]
    qpos_idx = [m.jnt_qposadr[j] for j in jids]
    dof_idx = [m.jnt_dofadr[j] for j in jids]
    limits = [(float(m.jnt_range[j, 0]), float(m.jnt_range[j, 1])) for j in jids]
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "oa_openarm_left_ee_site")

    def fk(q):
        d.qpos[qpos_idx] = q
        mujoco.mj_kinematics(m, d)
        return d.site_xpos[site_id].copy(), d.site_xmat[site_id].reshape(3, 3).copy()

    def mass_matrix(q):
        d.qpos[qpos_idx] = q
        mujoco.mj_forward(m, d)
        full_m = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, full_m)
        return full_m[np.ix_(dof_idx, dof_idx)] + np.diag(np.full(7, 0.05))

    arm = payload.ArmModel(fk=fk, mass_matrix=mass_matrix, armature=np.full(7, 0.05))
    model_tool = payload.Tool(mass=0.1, com=np.array([0.0, 0.0, 0.02]), inertia=np.eye(3) * 1e-4)
    real_tool = payload.Tool(mass=0.35, com=np.array([0.015, -0.01, 0.03]), inertia=np.eye(3) * 1e-4)

    # 1. Singular straight-down posture q=0 must raise a clear ValueError instead of silent numerical corruption
    with pytest.raises(ValueError, match="singular"):
        payload.Posture(arm, np.zeros(7), model_tool)

    # 2. Non-singular posture on the 7-DOF OpenDubs arm accurately computes A and recovers the 250g + CoM tool error
    q_bent = np.radians([10.0, -30.0, 15.0, 60.0, -10.0, 25.0, 10.0])
    a_true = payload.loop_gain(arm, q_bent, model_tool, real_tool)
    err = payload.fit_tool_error(arm, q_bent, model_tool, a_true, fit_armature=False)
    assert err.mass_error == pytest.approx(0.25, abs=1e-2)
    assert np.allclose(err.com_error, [0.015, -0.01, 0.01], atol=5e-3)
    assert "base joints' armature" in str(err)

    # 3. End-to-end sweep analysis + 6-DOF plant identification using the OpenDubs arm's true A matrix
    gains = identify.PoseGains.uniform(omega=18.0, zeta=0.8)
    sweep_cfg = response.Sweep(amplitude=0.01, f_start=0.5, f_end=6.0, duration=4.0, ramp=0.4)
    dt = 0.002
    wall = np.arange(0.0, sweep_cfg.total, dt)
    u_sig = sweep_cfg.value(wall)

    class _SimPath:
        def __init__(self, dof, sweep):
            self.dof = dof
            self.sweep = sweep

    class _SimSegment:
        def __init__(self, dof, sweep, log, period):
            self.kind = "sweep"
            self.label = f"sweep_{dof}"
            self.complete = True
            self.period = period
            self.log = log
            self._path = _SimPath(dof, sweep)

        def path(self):
            return self._path

    class _SimRecord:
        def __init__(self, segments):
            self.segments = segments

    # Analysing a single-axis sweep through analysis.analyse_arrays -> ResponseSet.from_analysis -> fit_plant
    log = [(float(t), float(t)) for t in wall[::5]]
    seg = _SimSegment("z", sweep_cfg, log, period=5 * dt)
    disp = np.zeros((len(wall), 6))
    disp[:, 2] = u_sig  # unity tracking along z for pipeline verification
    seg_results = analysis.analyse_arrays(_SimRecord([seg]), wall, disp, points=12)
    rset = identify.ResponseSet.from_analysis(seg_results, feedforward="off")
    assert rset.swept == ["z"]

    # 4. Actuator friction scheduling & limit checking against OpenDubs 7-DOF joint limits, plus MuJoCo friction recovery
    centre = np.array([0.5 * (lo + hi) for lo, hi in limits])
    plans, skipped = af.schedule(joints=[0, 3, 6], speeds=[0.2, 0.5], centre=centre, travel_limit=0.5, min_duration=0.4)
    af.check_limits(plans, limits, margin=0.05)

    # Simulate the constant-speed passes using OpenDubs's MuJoCo damping (0.4 N m s/rad) + frictionloss (0.1 N m) + gravity (qfrc_bias)
    d.qpos[qpos_idx] = centre
    mujoco.mj_forward(m, d)
    coulomb_sim = float(m.dof_frictionloss[dof_idx[3]])
    viscous_sim = float(m.dof_damping[dof_idx[3]])
    gravity_sim = float(d.qfrc_bias[dof_idx[3]])

    played = []
    for p in [pl for pl in plans if pl.joint == 3]:
        t_p, pos_p, vel_p, _ = af.pass_path(p, centre, period=0.005)
        qd_j = vel_p[:, 3]
        tau_p = np.zeros_like(pos_p)
        tau_p[:, 3] = gravity_sim + np.sign(qd_j) * coulomb_sim + viscous_sim * qd_j
        rec = run.recording(np.column_stack([t_p, pos_p, vel_p, tau_p]), 7)
        w = af.steady_window(t_p, qd_j, p, tolerance=0.1)
        assert w is not None
        played.append((af.Pass(p.joint, p.direction, p.speed, *w), rec))

    jnames = run.joint_names(7)
    table = af.identify_recordings(played, joint_names=jnames, torque="tau{}", velocity="qd{}", stamp="t")
    nums = af.as_numbers(table, jnames)["4"]
    assert nums["coulomb"] == pytest.approx(coulomb_sim, abs=1e-3)
    assert nums["viscous"] == pytest.approx(viscous_sim, abs=1e-3)
    assert nums["gravity_error"] == pytest.approx(gravity_sim, abs=1e-3)


