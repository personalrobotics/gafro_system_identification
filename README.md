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
