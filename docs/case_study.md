# Case Study — Battery Grasp Evaluation

This case study summarizes a representative batch evaluation run included in this repository.

Summary
- Input candidate set: `candidates_reduced.json`
- Simulation model: `UR5.xml` (scene: `scene.xml`)
- Number of candidates evaluated: 15
- Number of valid candidates: 6

Best candidate (ID: 10)
- Accepted: true
- Grip point (mm): [0.0, 0.0, 18.0]
- Pre-grip position (mm): [0.0, 0.0, 35.7]
- Post-grip position (mm): [0.0, 0.0, 35.7]
- Clearance (mm): 0.43498
- Gripper open width (mm): 15.369308
- Release height (mm): 20.0
- Contact counts: left pad 4, right pad 3

Notes
- Selection rule: "best valid candidate = lowest grip_point_mm[2], tie-break by candidate_id" (see `candidate_eval_results.json`).
- The case study run is saved in `candidate_eval_results.json` and the top candidate in `best_candidate_single_grasp.json`.

How this file was produced
1. Generate candidates via FreeCAD macro (`macros/`).
2. Run `Candidates_iteration.py` to simulate and evaluate each candidate.
3. The script writes per-candidate metrics and selects the best candidate.

Use this case study as a reproducible example when writing methods sections for reports or embedding figures in your thesis.
