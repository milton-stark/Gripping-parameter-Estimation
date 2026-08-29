# Gripping Parameter Estimation

A compact, MuJoCo-based simulation framework for generating, evaluating, and validating robotic grasp candidates for a UR5 arm with a parallel-jaw gripper. The repo automates candidate generation from CAD, batch simulation-based evaluation, and single-case validation so you can find robust gripping parameters quickly.

---

## Why this is useful

- Reduce physical testing by validating grasps in simulation.
- Automate candidate generation from CAD models (FreeCAD macros).
- Quantify grip stability (contact forces, slip detection) to pick reliable grasps.
- Tune control parameters (squeeze, timing, friction) and reproduce experiments.

---

## What it does

1. Use FreeCAD macros to generate candidate grasps from an object CAD model.
2. Simulate every candidate in MuJoCo and score success metrics.
3. Select the best candidate(s) and re-run a focused single-case simulation for verification.

---

## Quick start

1. Install dependencies:

```bash
pip install mujoco numpy
```

(Install FreeCAD separately if you plan to use the macros.)

2. Generate candidates (in FreeCAD):

- Open your model and run a macro from `macros/` (e.g. `grip_iteration.FCMacro` or `grip_new.FCMacro`).
- The macro writes a JSON of candidates (default: `candidates_reduced.json`).

3. Batch-evaluate candidates:

```bash
python Candidates_iteration.py
```

- Input: `candidates_reduced.json`
- Outputs: `candidate_eval_results.json` and `best_candidate_single_grasp.json`

4. Validate the top candidate:

```bash
python Candidate_Grip_test.py
```

- Uses `best_candidate_single_grasp.json` as input; runs a single simulation and logs contact traces.

---

## Repository layout (important files)

- `Candidate_Grip_test.py` — single-case simulation and debug runner
- `Candidates_iteration.py` — batch evaluator and filter; produces best candidate JSON
- `macros/` — FreeCAD macros to produce candidate JSON files
- `UR5.xml` — primary MuJoCo robot model used by scripts
- `candidates_reduced.json` — canonical example input for the batch runner
- `candidate_eval_results.json` — batch-run output (per-candidate metrics)
- `best_candidate_single_grasp.json` — top candidate selected by the batch run
- `mesh/`, `textures/` — visual meshes and textures used by simulations
- `xml_files/` — optional alternate MuJoCo model variants

Note: `battery_grip_data.json` is legacy/example data and not required for normal operation (it is kept in `macros/`).

---

## Configuration

Edit top-level constants in `Candidates_iteration.py` and `Candidate_Grip_test.py` to change timing, friction, and gripper parameters. Key parameters:

- `T_HOME_SETTLE`, `T_TO_PRE`, `T_TO_GRIP` — motion timing
- `CLOSE_MAG`, `CLOSE_RAMP_TIME` — gripper closing behavior
- `FRICTION_MULT`, `GRIPPER_STRENGTH_MULT` — stability multipliers
- `TARGET_PINCH` — desired pinch width

---

## Troubleshooting

- MuJoCo errors: ensure `UR5.xml` and required scene files are accessible; set `XML_PATH` in scripts.
- Candidate JSON issues: validate against `candidates_reduced.json` format (array of candidate objects with `grip_point`, `pre_grip`, etc.).
- FreeCAD macros: run macros from within FreeCAD; some require specific workbenches.

---

## How to contribute

- Open issues for bugs or feature requests.
- Send PRs that include tests or reproducible examples. Keep changes small and focused.

---

## Maintainers & support

Maintained by milton-stark. For support or questions, open a GitHub Issue on this repository or contact the maintainer via GitHub.

---

## Workflow diagram

```mermaid
graph LR
  CAD[CAD/ (3D models)] --> A[FreeCAD macros\n(macros/*.FCMacro)]
  A --> B[candidates_reduced.json]
  B --> C[Candidates_iteration.py\n(batch simulate & score)]
  C --> D[candidate_eval_results.json]
  C --> E[best_candidate_single_grasp.json]
  E --> F[Candidate_Grip_test.py\n(single-case validate)]
  F --> G[Logs / visual check\n(MuJoCo viewer, MUJOCO_LOG.TXT)]
  C -.->|uses| X[UR5.xml, mesh/, textures/]
  F -.->|uses| X
```

## Thesis & assets

This repository contains your CAD models and thesis artifacts; quick links for reviewers:

- CAD models: `CAD/` — examples: [CAD/battery.stl](CAD/battery.stl), [CAD/casing_new.stl](CAD/casing_new.stl)
- 3D meshes used by simulations: `mesh/visual/` (battery.stl, casing.stl, robot links)
- Thesis report (PDF): [Report/Sepasthiyammal,Milton,1702059_Thesis.pdf](Report/Sepasthiyammal,Milton,1702059_Thesis.pdf)
- Presentation: [Report/Thesis Presentation.pptx](Report/Thesis Presentation.pptx)

Include these files when preparing artefacts for review or submission. The thesis contains methodology, experimental results, and evaluation plots derived from `candidate_eval_results.json`.

---

## References

- MuJoCo: https://github.com/deepmind/mujoco
- FreeCAD: https://www.freecadweb.org/
- Universal Robots (UR5): https://www.universal-robots.com/
