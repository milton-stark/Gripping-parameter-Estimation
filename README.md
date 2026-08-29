# Gripping Parameter Estimation

Simulation framework for finding reliable grasp parameters for a UR5 with a parallel-jaw gripper. Grasp candidates are generated from a CAD model, scored in bulk in MuJoCo, and the best one is re-run on its own for verification, so grip points and forces are settled before anything is tried on hardware.

---

## About the project

In modular battery assembly, the parts a cell has to handle change often, a different cell diameter, a revised housing plate, a new locking element. Each change means someone has to decide again how the part gets picked up: which surface the gripper closes on, from which direction it approaches, and how much force it applies. Too little force and the part slips during transfer; too much and a cell casing deforms.

That decision is normally made by hand, from the CAD model and a gripper datasheet, and then confirmed by trial on the actual cell. It is slow, it depends on the experience of whoever makes it, and it has to be repeated for every variant.

This project replaces the trial part with simulation. Candidate grasps are generated directly from the object's CAD geometry, every candidate is simulated on a UR5 with a parallel-jaw gripper in MuJoCo, and each one is scored on how stably it holds during approach, closing, and lift. The best candidate is then re-run on its own so the result can be inspected rather than trusted on a number alone.

Developed as a Master's thesis in M.Sc. Mechatronics at the University of Siegen (Chair of Interconnected Automation Systems), in cooperation with Fraunhofer IGCV, Augsburg, on a battery module assembly use case.

## Why it's useful

- **Fewer hardware trials.** Weak grasps are found in simulation, where a failure costs a few seconds instead of a damaged cell and a stopped line.
- **Candidates come from the CAD model.** Nothing is hand-labelled, so a new part variant goes through the same pipeline without new manual setup.
- **Grasps are scored, not guessed.** Contact forces and slip give a comparable number across candidates, instead of relying on which one looks reasonable.
- **Experiments are reproducible.** Timing, friction, and gripper parameters are explicit constants, so a result can be re-run and the effect of a change isolated.
- **The output is a parameter set.** What comes out is the grip point, approach direction, and closing parameters, the same values that would go into a robot program.

## How it works

The project runs as three stages, each handing a JSON file to the next.

**1 — Generate candidates (FreeCAD).** A macro from `macros/` reads the object's CAD model and writes out a set of candidate grasps: for each one, where the gripper approaches from (`pre_grip`) and where it closes (`grip_point`). Output: `candidates_reduced.json`.

**2 — Score them all (MuJoCo).** `Candidates_iteration.py` loads that file and simulates every candidate against the `UR5.xml` scene: approach, close, lift, and measure. Each candidate gets contact-force and slip metrics. All results land in `candidate_eval_results.json`; the winner is written separately to `best_candidate_single_grasp.json`.

**3 — Verify the winner.** `Candidate_Grip_test.py` re-runs that single grasp on its own, with the viewer available and full contact traces logged. This is where you watch the grip and confirm the batch score wasn't an artefact.

```
                 ┌──────────────────────────────┐
                 │  CAD/*.stl                   │
                 │  object geometry             │
                 └───────────────┬──────────────┘
                                 │
                 ┌───────────────▼──────────────┐
                 │  macros/*.FCMacro            │   run inside FreeCAD
                 │  generate grasp candidates   │
                 └───────────────┬──────────────┘
                                 │
                 ┌───────────────▼──────────────┐
                 │  candidates_reduced.json     │
                 └───────────────┬──────────────┘
                                 │
                 ┌───────────────▼──────────────┐      ┌──────────────┐
                 │  Candidates_iteration.py     │◀─────┤  UR5.xml     │
                 │  simulate all · score · rank │      │  mesh/       │
                 └───────┬─────────────┬────────┘      │  textures/   │
                         │             │               └──────┬───────┘
      ┌──────────────────▼──┐   ┌──────▼───────────────────┐  │
      │ candidate_eval_     │   │ best_candidate_          │  │
      │ results.json        │   │ single_grasp.json        │  │
      │ (all metrics)       │   └──────┬───────────────────┘  │
      └─────────────────────┘          │                      │
                                ┌──────▼───────────────────┐  │
                                │  Candidate_Grip_test.py  │◀─┘
                                │  re-run winner · viewer  │
                                └──────┬───────────────────┘
                                       │
                                ┌──────▼───────────────────┐
                                │  contact traces          │
                                │  visual verification     │
                                └──────────────────────────┘
```

## Files

| File / folder | Role |
|---|---|
| `FreeCAD macro_candidates.FCMacro` | FreeCAD macro that turn a CAD model into grasp candidates. Run from inside FreeCAD, not the terminal. |
| `candidates_reduced.json` | The candidate list produced by the macro is the input to the batch run. Also serves as the reference for the expected format. |
| `Candidates_iteration.py` | Batch evaluator. Simulates every candidate, scores stability, filters, and picks the best. The main script of the project. |
| `candidate_eval_results.json` | Per-candidate metrics from the batch run. The evaluation plots in the thesis come from this file. |
| `best_candidate_single_grasp.json` | The single top candidate, in the same format as the input list. |
| `Candidate_Grip_test.py` | Single-case runner. Loads the best candidate, simulates it with the viewer and detailed logging. Used for debugging and for the final visual check. |
| `UR5.xml` | The MuJoCo scene: UR5 arm, parallel-jaw gripper, object, and ground. Both scripts load this. |
| `xml_files/` | Alternative scene variants and different objects or gripper settings. Point `XML_PATH` at one of these to swap scenes. |
| `mesh/`, `textures/` | Collision and visual assets referenced by the MuJoCo models. |
| `CAD/` | Source CAD models (`battery.stl`, `casing_new.stl`) that the macros work from. |
| `Report/` | Thesis PDF and presentation. |

## Running it

```bash
pip install mujoco numpy
```

FreeCAD is installed separately and is only needed for stage 1.

```bash
python Candidates_iteration.py    # stage 2 - batch evaluation
python Candidate_Grip_test.py     # stage 3 - verify the winner
```

Both scripts read paths from constants at the top of the file. If MuJoCo fails to load the scene, check that `XML_PATH` points at `UR5.xml` and that `mesh/` and `textures/` are reachable from it.

## Parameters

Both scripts expose their tuning constants at the top:

| Constant | Controls |
|---|---|
| `T_HOME_SETTLE`, `T_TO_PRE`, `T_TO_GRIP` | Motion timing between phases |
| `CLOSE_MAG`, `CLOSE_RAMP_TIME` | How hard and how fast the gripper closes |
| `TARGET_PINCH` | Commanded jaw width at the grip |
| `FRICTION_MULT`, `GRIPPER_STRENGTH_MULT` | Scaling on contact friction and actuator strength |

These change what counts as a successful grasp, so keep them consistent between the batch run and the verification run, otherwise the winner won't reproduce.

## Thesis and assets

- Thesis: [`Report/Sepasthiyammal,Milton,1702059_Thesis.pdf`](Report/Sepasthiyammal,Milton,1702059_Thesis.pdf)
- Presentation: [`Report/Thesis Presentation.pptx`](Report/Thesis%20Presentation.pptx)
- Methodology, experimental results, and the evaluation plots derived from `candidate_eval_results.json` are all in the thesis.

## Built with

[MuJoCo](https://github.com/google-deepmind/mujoco) · [FreeCAD](https://www.freecad.org/) · Universal Robots UR5
