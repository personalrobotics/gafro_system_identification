# gafro_system_identification

Robot-agnostic system identification for N-DOF manipulators. Pure `numpy` + `scipy` core with no middleware (ROS 2, Dora) or robot-specific dependencies required.

## Modules

- **`response.py`** — Log-chirp sweep excitation (`Sweep`), lock-in frequency response estimation (`frequency_response`), tone response, and 2nd-order + delay fitting (`fit_second_order`).
- **`analysis.py`** — Segment-level analysis of sweep and figure-eight runs against a control-cycle trace.
- **`identify.py`** — 6-DOF closed-loop plant fit (`PoseGains`, `ResponseSet`, inertia mismatch `A`, delay `tau`, damping `C`) and Padé pole stability scans.
- **`payload.py`** — Tool mass and center-of-mass error estimation from identified inertia mismatch `A`, using generic `fk(q)` and `mass_matrix(q)` callables.
- **`actuator_friction.py`** — Joint friction (Coulomb + viscous) and gravity estimation for N-joint arms that report measured torque (e.g. MIT-frame actuators).
- **`joint_friction.py`** — Joint friction and compensation-scale identification from commanded torque.
- **`friction_opt.py`** — N-joint SPSA optimization of viscous/coulomb friction compensation scales in logit space.
- **`vibration.py`** — Spectral band analysis aligned against control-loop period/compute jitter and teleop device ticks.
- **`trace.py`** — Cycle-log reading and end-effector displacement extraction via generic FK callables or a `gafro` task space.
- **`run.py`** — Live friction-pass runner using a transport-agnostic `RobotAdapter` protocol (plus `GafroNodeAdapter` for in-process `gafro_control_nodes`).

## Install & Test

```bash
pip install -e ".[dev]"
pytest
```

## Basic Usage Examples

### 1. Frequency Response & 2nd-Order Fit (Chirp Sweep)

Estimate the closed-loop Bode response and fit natural frequency (`wn`), damping ratio (`zeta`), and loop delay (`delay`) from time-aligned reference (`u`) and measured (`y`) signals:

```python
import numpy as np
from gafro_system_identification import response

sweep = response.Sweep(amplitude=0.02, f_start=0.5, f_end=8.0, duration=8.0, ramp=0.5)
t = np.arange(0.0, sweep.total, 1.0 / 500.0)
u = sweep.amplitude * sweep.envelope(t) * np.sin(sweep.phase(t))

# `y` is your measured 1-D displacement along the excited axis at timestamps `t`
fr = response.frequency_response(t, u, y, sweep, points=24)
fit = response.fit_second_order(fr)
print(f"wn={fit.wn:.2f} rad/s, zeta={fit.zeta:.2f}, delay={fit.delay * 1e3:.1f} ms")
```

### 2. 6-DOF Plant Identification & Stability Limit

Fit the 6-DOF closed-loop plant (`A`, `tau`, `C`) from sweep excitations and find the maximum stable `omega_n` before the poles cross into the right-half plane:

```python
from gafro_system_identification import identify

gains = identify.PoseGains.uniform(omega=15.0, zeta=0.7)
# `results` comes from analysis.analyse(...) or a loaded ResponseSet(.npz)
rset = identify.ResponseSet.from_analysis(results, feedforward="off")
plant = identify.fit_plant(rset, gains)
limit = identify.stability_limit(plant, zeta=0.7)
print(f"estimated delay={plant.tau[0] * 1e3:.1f} ms, max stable omega={limit:.1f} rad/s")
```

### 3. Live Joint-Friction Identification (Any N-DOF Arm)

Implement the two-method `RobotAdapter` protocol for your transport (Dora, ROS 2, MuJoCo, or `GafroNodeAdapter`), play constant-speed passes, and fit per-joint Coulomb + viscous friction:

```python
import numpy as np
from gafro_system_identification import actuator_friction as af
from gafro_system_identification import run

class MyArmAdapter:
    def get_state(self) -> run.JointState:
        # Return (position, velocity, measured_torque) arrays of length N (e.g. 7 for OpenArm)
        return run.JointState(position=..., velocity=..., torque=...)

    def follow_path(self, times, positions, velocities=None, **play) -> run.FollowResult:
        # Execute trajectory on your robot/middleware and return FollowResult(ok=True/False)
        return run.FollowResult(ok=True, status="OK")

adapter = MyArmAdapter()
n_joints = 7
centre = np.zeros(n_joints)
plans, skipped = af.schedule(
    joints=range(n_joints),
    speeds=[0.1, 0.3, 0.5],
    centre=centre,
    travel_limit=0.6,
    min_duration=1.0,
)
played, dropped = run.run(adapter, plans, period=0.002)
table = af.identify_recordings(
    played,
    joint_names=run.joint_names(n_joints),
    torque="tau{}",
    velocity="qd{}",
    stamp="t",
)
print(af.as_numbers(table, run.joint_names(n_joints)))
```

### 4. End-Effector Displacements Without `gafro`

Extract 6-DOF world-frame displacements `(x, y, z, rx, ry, rz)` from a cycle log using any forward-kinematics callable `fk(q) -> (position, rotation_matrix)`:

```python
from gafro_system_identification import trace

disp = trace.measured_displacements_with(
    log_dict,
    columns=[f"q_{i}" for i in range(7)],
    fk=my_fk_fn,
    origin_position=p0,
    origin_matrix=R0,
)
```
