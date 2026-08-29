# Gripping Parameter Estimation

A MuJoCo-based simulation framework for estimating, evaluating, and optimizing gripping parameters for robotic manipulation. This repository contains tools for simulating a UR5 robotic arm with a parallel gripper, evaluating multiple grasp candidates, and tuning control parameters (friction, stiffness, contact geometry, etc.) for stable and reliable object gripping.

## What It Does

This project simulates a **UR5 robotic arm with a parallel-jaw gripper** performing pick-and-place tasks on objects like batteries. The system:

- **Simulates physics-based gripping** using MuJoCo, a fast physics engine
- **Evaluates grasp candidates** by testing different grip points, approach angles, and parameters
- **Optimizes gripper control** parameters including friction coefficients, squeeze forces, and timing
- **Integrates with FreeCAD** for CAD-based grasp planning and visualization
- **Generates detailed metrics** on grasp success/failure, contact forces, and slip detection

Key capability: given an object CAD model and target grasp points, the system predicts success likelihood and identifies optimal control parameters.

## Features

- **MuJoCo-based physics simulation** with configurable XML model definitions for arm, gripper, and environment
- **Parallel grasp evaluation** system to test multiple candidate grasps and rank by success metrics
- **FreeCAD macro integration** for importing CAD models, computing grasp points, and visualizing results
- **Configurable control parameters**: grip point location, pre-position, post-position, squeeze force, timing, friction, etc.
- **Contact-based success detection** using MuJoCo contact data to evaluate grip stability
- **JSON-based data pipelines** for storing grasp candidates, parameters, and evaluation results
- **Inverse kinematics (IK) solver** for computing arm joint angles to reach desired end-effector positions

## Repository Layout

```
├── Grip.py                          # Main gripper control and single-grasp simulation
├── iteration.py                     # Batch evaluation of multiple grasp candidates
├── FreeCAD macro.py                 # FreeCAD integration for CAD-based grasp planning
├── UR5.xml                          # MuJoCo robot model (UR5 arm + gripper)
├── scene.xml / scene_test.xml       # MuJoCo scene definitions with object and table
├── battery_grip_data.json           # Example grasp parameters and force estimates
├── candidates_reduced.json          # Example set of grasp candidates for evaluation
├── candidate_eval_results.json      # Evaluation results from batch runs
├── best_candidate_single_grasp.json # Best-performing grasp parameters
├── xml_files/                       # Alternative MuJoCo model variants
├── macros/                          # FreeCAD macro scripts for grasp planning
├── mesh/                            # 3D mesh files for simulation
└── textures/                        # Texture files for visualization
```

## Requirements

- **Python 3.8+**
- **MuJoCo** (physics simulation engine)
- **NumPy** (numerical computation)
- **FreeCAD** (optional, for CAD-based grasp planning)

## Getting Started

### 1. Install Dependencies

```bash
pip install mujoco numpy
```

For FreeCAD-based grasp planning, install FreeCAD via your system package manager or download from https://www.freecadweb.org/.

### 2. Run a Single Grasp Simulation

The `Grip.py` script simulates a complete grasp sequence on a single object:

```bash
python Grip.py
```

This will:
1. Load the UR5 robot model and scene
2. Move the arm to a safe home position
3. Approach a pre-defined grip point
4. Close the gripper with configurable squeeze force
5. Lift the object and hold it
6. Log contact forces and grip stability metrics

**Configuration**: Edit the constants at the top of `Grip.py` to change grip points, timing, friction, and gripper force.

### 3. Batch Evaluate Grasp Candidates

The `iteration.py` script evaluates multiple grasp candidates in sequence:

```bash
python iteration.py
```

This script:
1. Loads a list of grasp candidates from `candidates_reduced.json`
2. Runs each grasp through the full simulation
3. Records contact forces, slip detection, and success/failure
4. Saves detailed results to `candidate_eval_results.json`
5. Identifies the best-performing grasp and saves it to `best_candidate_single_grasp.json`

### 4. FreeCAD-Based Grasp Planning (Optional)

Use FreeCAD macros to automatically generate grasp candidates from CAD models:

1. Open your object model in FreeCAD
2. Run one of the macros in the `macros/` directory (e.g., `grip_iteration.FCMacro`)
3. The macro computes geometric properties (center of mass, bounding box, principal axes)
4. Generates candidate grasp points and exports to JSON format

## Configuration Guide

Key parameters in `Grip.py` and `iteration.py`:

| Parameter | Description |
|-----------|-------------|
| `T_HOME_SETTLE` | Time for arm to reach safe home position (seconds) |
| `T_TO_PRE` | Time for arm to move to pre-grip position (seconds) |
| `T_TO_GRIP` | Time for arm to reach final grip point (seconds) |
| `CLOSE_MAG` | Gripper closing distance per cycle (meters) |
| `CLOSE_RAMP_TIME` | Duration of squeeze ramp phase (seconds) |
| `FRICTION_MULT` | Multiplier on friction coefficient for grip stability check |
| `TARGET_PINCH` | Target gripper opening width (meters) |
| `GRIPPER_STRENGTH_MULT` | Force multiplier for grip strength |
| `TABLE_CLEARANCE` | Minimum height above table surface (meters) |

## Output Files

- **`candidate_eval_results.json`**: Detailed results for each grasp candidate including success rate, contact forces, and slip events
- **`best_candidate_single_grasp.json`**: Best-performing candidate with all parameters and metrics
- **`MUJOCO_LOG.TXT`**: Detailed timestep-by-timestep log from MuJoCo simulation

## Troubleshooting

**Issue**: MuJoCo license error or missing model files  
**Solution**: Ensure `UR5.xml` and `scene.xml` are in the same directory as the Python scripts. Set `XML_PATH` correctly.

**Issue**: Grasp candidate fails to load  
**Solution**: Verify JSON format matches the expected schema (see `battery_grip_data.json` for reference). Check that grip points are within object bounds.

**Issue**: FreeCAD macro not running  
**Solution**: Ensure FreeCAD is installed and the macro file is in the correct directory. Some macros require specific FreeCAD workbenches (e.g., Part Design).

## Contributing

Contributions welcome! Please:
- Report issues via GitHub Issues with detailed simulation parameters and expected vs. actual behavior
- Submit pull requests for bug fixes or new features (e.g., new gripper models, optimization algorithms)
- Add comments explaining control logic and parameter tuning rationale

## License

[Specify your license here, e.g., MIT, Apache 2.0]

## Contact & Support

**Author**: milton-stark  
**Repository**: [GitHub link to repository]

For questions or collaboration inquiries:
- Open a GitHub Issue with your question or feature request
- Contact the repository owner via GitHub

## References

- **MuJoCo**: https://github.com/deepmind/mujoco
- **UR Robotics**: https://www.universal-robots.com/
- **FreeCAD**: https://www.freecadweb.org/
