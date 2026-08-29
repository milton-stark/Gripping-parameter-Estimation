import time
import json
import numpy as np
import mujoco

# =========================
# FILES
# =========================
XML_PATH = "UR5.xml"
JSON_PATH = "candidates_reduced.json"

RESULTS_PATH = "candidate_eval_results.json"
BEST_CANDIDATE_JSON_PATH = "best_candidate_single_grasp.json"

# =========================
# USER SETTINGS
# =========================
SAFE_HOME_Q = np.array([0.88, -2.01, 1.88, -1.51, -1.51, -0.628], dtype=float)
SITE_TCP = "tcp_tip"

# Timing
T_HOME_SETTLE = 0.6
T_TO_PRE = 2.0
WAIT_AT_PRE = 0.5
T_TO_GRIP = 2.0

DT_SLEEP = 0.0
SUBSTEPS = 4

# Table safety
TABLE_CLEARANCE = 0.000

# Gate tolerances
POS_TOL_PRE = 0.0010
POS_TOL_GRIP = 0.0005
ROT_TOL = 0.04
SETTLE_TIME = 0.15
GATE_TIMEOUT = 8.0

# IK tuning
IK_ITERS_PER_CYCLE = 8
IK_DAMP = 0.02
IK_STEP = 0.55

# Close / hold
CLOSE_MAG = 0.010
CLOSE_RAMP_TIME = 0.6
CLOSE_HOLD_TIME = 0.8
POST_CLOSE_HOLD = 0.20

# Friction / strength
FRICTION_MULT = 3.0
FRICTION_SLIDE_CAP = 25.0
SQUEEZE_EXTRA = 0.0015
USE_PINCH_VISUAL_CLOSE = True
TARGET_PINCH = 0.0160
PINCH_GAIN = 0.35
PINCH_MAX_EXTRA = 0.0015
GRIPPER_STRENGTH_MULT = 2.0

# Actuators
ARM_ACT_NAMES = ["shoulder_pan", "shoulder_lift", "forearm", "wrist_1", "wrist_2", "wrist_3"]

# Finger collision geoms to disable during approach
FINGER_COL_GEOMS = ["ee_left_col", "ee_right_col"]

# Success / failure logic
LEFT_PAD_NAME = "ee_left_pad"
RIGHT_PAD_NAME = "ee_right_pad"
TARGET_BODY_NAME = "battery"
FORBIDDEN_BODY_NAMES = {"casing", "battery2"}
GRIPPER_BODY_NAMES = {"ee_link", "ee_gripper_left_link", "ee_gripper_right_link"}


# =========================
# NAME HELPERS
# =========================
def _id(model, objtype, name: str) -> int:
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx == -1:
        raise RuntimeError(f"Missing in model: {name} ({objtype})")
    return idx

def body_id(model, name):     return _id(model, mujoco.mjtObj.mjOBJ_BODY, name)
def joint_id(model, name):    return _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
def actuator_id(model, name): return _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
def site_id(model, name):     return _id(model, mujoco.mjtObj.mjOBJ_SITE, name)
def geom_id(model, name):     return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
def geom_id_strict(model, name): return _id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

def body_geom_ids(model, body_name: str):
    bid = body_id(model, body_name)
    gadr = int(model.body_geomadr[bid])
    gnum = int(model.body_geomnum[bid])
    return list(range(gadr, gadr + gnum))


# =========================
# JSON LOADING
# =========================
def load_candidate_root(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        root = json.load(f)
    return root

def load_candidate_set(json_path: str):
    root = load_candidate_root(json_path)
    rel_tcp = np.array(root.get("relative_to_tcp", [0.0, 0.0, 0.0]), dtype=float).reshape(3) * 1e-3
    candidates = root["candidates"]
    return root, rel_tcp, candidates


# =========================
# TABLE TOP + CLAMP
# =========================
def get_table_top_z(model, data):
    candidates = ["table_block", "table_block_2"]
    tops = []
    for name in candidates:
        gid = geom_id(model, name)
        if gid == -1:
            continue
        zc = float(data.geom_xpos[gid][2])
        sz = float(model.geom_size[gid][2])
        tops.append(zc + sz)
    return max(tops) if tops else None

def clamp_above_table(p, table_top_z):
    if table_top_z is None or TABLE_CLEARANCE <= 0:
        return p
    q = p.copy()
    q[2] = max(q[2], table_top_z + TABLE_CLEARANCE)
    return q


# =========================
# CONTACT HELPERS
# =========================
def geom_body_name(model, geom_id_):
    body_idx = int(model.geom_bodyid[geom_id_])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_idx)

def count_specific_contacts(data, geom_ids_a, geom_ids_b):
    n = int(data.ncon)
    cnt = 0
    for i in range(n):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        if (g1 in geom_ids_a and g2 in geom_ids_b) or (g2 in geom_ids_a and g1 in geom_ids_b):
            cnt += 1
    return cnt

def has_forbidden_collision(model, data, gripper_body_names, forbidden_body_names):
    n = int(data.ncon)
    for i in range(n):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        b1 = geom_body_name(model, g1)
        b2 = geom_body_name(model, g2)
        if b1 is None or b2 is None:
            continue
        if (b1 in gripper_body_names and b2 in forbidden_body_names) or \
           (b2 in gripper_body_names and b1 in forbidden_body_names):
            return True, {"contact_index": i, "body1": b1, "body2": b2}
    return False, None


# =================
# FINGER COLLISION 
# =================
def set_finger_collision(model, enable: bool):
    contype = 1 if enable else 0
    conaff = 1 if enable else 0
    for gname in FINGER_COL_GEOMS:
        gid = geom_id(model, gname)
        if gid == -1:
            continue
        model.geom_contype[gid] = contype
        model.geom_conaffinity[gid] = conaff


def boost_friction(model, geom_names, mult=3.0, slide_cap=25.0):
    for name in geom_names:
        gid = geom_id(model, name)
        if gid == -1:
            continue
        fr = model.geom_friction[gid].copy()
        fr *= mult
        fr[0] = min(fr[0], slide_cap)
        model.geom_friction[gid] = fr

def boost_friction_ids(model, geom_ids, mult=3.0, slide_cap=25.0):
    for gid in geom_ids:
        if gid < 0 or gid >= model.ngeom:
            continue
        fr = model.geom_friction[gid].copy()
        fr *= mult
        fr[0] = min(fr[0], slide_cap)
        model.geom_friction[gid] = fr

def boost_gripper_strength(model, mult=2.0):
    for name in ["ee_gripper_left", "ee_gripper_right"]:
        aid = actuator_id(model, name)
        model.actuator_gear[aid, :] *= mult
        model.actuator_forcerange[aid, 0] *= mult
        model.actuator_forcerange[aid, 1] *= mult


# ========
# ARM IDS
# ========
def get_arm_actuator_ids_strict(model):
    return [actuator_id(model, n) for n in ARM_ACT_NAMES]

def get_arm_joint_ids_from_actuators(model, arm_act_ids):
    jids = []
    for aid in arm_act_ids:
        jid = int(model.actuator_trnid[aid, 0])
        if jid < 0:
            raise RuntimeError("Actuator has no joint transmission.")
        jids.append(jid)
    if len(jids) != 6:
        raise RuntimeError("Expected 6 arm joints from actuators.")
    return jids

def sync_arm_ctrl_to_qpos(model, data, arm_act_ids):
    for aid in arm_act_ids:
        jid = int(model.actuator_trnid[aid, 0])
        qadr = model.jnt_qposadr[jid]
        data.ctrl[aid] = float(data.qpos[qadr])


# ================
# GRIPPER CONTROL
# ================
def set_gripper_close_targets(model, data, closeL, closeR):
    aL = actuator_id(model, "ee_gripper_left")
    aR = actuator_id(model, "ee_gripper_right")
    data.ctrl[aL] = float(closeL)
    data.ctrl[aR] = float(closeR)

def set_gripper_open_targets(model, data):
    jL = joint_id(model, "ee_gripper_left_joint")
    jR = joint_id(model, "ee_gripper_right_joint")
    rL = model.jnt_range[jL]
    rR = model.jnt_range[jR]
    aL = actuator_id(model, "ee_gripper_left")
    aR = actuator_id(model, "ee_gripper_right")
    data.ctrl[aL] = float(rL[1])
    data.ctrl[aR] = float(rR[0])

def pinch_distance(model, data):
    sL = site_id(model, "pinch_L")
    sR = site_id(model, "pinch_R")
    pL = data.site_xpos[sL].copy()
    pR = data.site_xpos[sR].copy()
    return float(np.linalg.norm(pL - pR))

def autocalibrate_close_targets(model, data):
    jL = joint_id(model, "ee_gripper_left_joint")
    jR = joint_id(model, "ee_gripper_right_joint")
    qadrL = model.jnt_qposadr[jL]
    qadrR = model.jnt_qposadr[jR]

    qL0 = float(data.qpos[qadrL])
    qR0 = float(data.qpos[qadrR])
    eps = 0.001

    data.qpos[qadrL] = qL0 + eps
    mujoco.mj_forward(model, data)
    dL_plus = pinch_distance(model, data)

    data.qpos[qadrL] = qL0 - eps
    mujoco.mj_forward(model, data)
    dL_minus = pinch_distance(model, data)

    data.qpos[qadrL] = qL0
    mujoco.mj_forward(model, data)

    data.qpos[qadrR] = qR0 + eps
    mujoco.mj_forward(model, data)
    dR_plus = pinch_distance(model, data)

    data.qpos[qadrR] = qR0 - eps
    mujoco.mj_forward(model, data)
    dR_minus = pinch_distance(model, data)

    data.qpos[qadrR] = qR0
    mujoco.mj_forward(model, data)

    dirL = +1.0 if dL_plus < dL_minus else -1.0
    dirR = +1.0 if dR_plus < dR_minus else -1.0

    closeL = qL0 + dirL * CLOSE_MAG
    closeR = qR0 + dirR * CLOSE_MAG

    rL = model.jnt_range[jL]
    rR = model.jnt_range[jR]
    closeL = float(np.clip(closeL, rL[0], rL[1]))
    closeR = float(np.clip(closeR, rR[0], rR[1]))

    return closeL, closeR



def desired_tcp_R_vertical():
    z = np.array([0.0, 0.0, 1.0])
    y = np.array([0.0, 1.0, 0.0])
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-12)
    return np.column_stack([x, y, z])

def rotvec_from_R(Rerr):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Rerr.reshape(9))
    if q[0] < 0:
        q *= -1.0
    rv = np.zeros(3)
    mujoco.mju_quat2Vel(rv, q, 1.0)
    return rv



def ik_step_pose_qpos(model, data, site_name, target_pos, target_R,
                      arm_joint_ids, damp=IK_DAMP, step=IK_STEP):
    sid = site_id(model, site_name)

    p = data.site_xpos[sid].copy()
    M = data.site_xmat[sid].reshape(3, 3).copy()

    e_p = (target_pos - p)
    Rerr = target_R @ M.T
    e_r = rotvec_from_R(Rerr)

    pos_err = float(np.linalg.norm(e_p))
    rot_err = float(np.linalg.norm(e_r))

    Jp = np.zeros((3, model.nv))
    Jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, sid)

    J = np.zeros((6, 6))
    for k, jid in enumerate(arm_joint_ids):
        dofadr = model.jnt_dofadr[jid]
        J[0:3, k] = Jp[:, dofadr]
        J[3:6, k] = Jr[:, dofadr]

    e = np.hstack([e_p, e_r])
    A = J @ J.T + damp * np.eye(6)
    dq = J.T @ np.linalg.solve(A, e)
    dq *= step

    for k, jid in enumerate(arm_joint_ids):
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr] = float(data.qpos[qadr] + dq[k])
        r = model.jnt_range[jid]
        data.qpos[qadr] = float(np.clip(data.qpos[qadr], r[0], r[1]))

    return pos_err, rot_err

def gate_reach_pose(model, data, site_name, target_pos, target_R,
                    arm_joint_ids, arm_act_ids, pos_tol, rot_tol,
                    settle=SETTLE_TIME, timeout=GATE_TIMEOUT):
    t0 = time.time()
    stable_since = None
    last_pos_err = None
    last_rot_err = None

    while True:
        mujoco.mj_forward(model, data)

        for _ in range(IK_ITERS_PER_CYCLE):
            last_pos_err, last_rot_err = ik_step_pose_qpos(
                model, data, site_name, target_pos, target_R, arm_joint_ids
            )
            mujoco.mj_forward(model, data)

        sync_arm_ctrl_to_qpos(model, data, arm_act_ids)

        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

        ok = (last_pos_err < pos_tol) and (last_rot_err < rot_tol)
        if ok:
            if stable_since is None:
                stable_since = time.time()
            if (time.time() - stable_since) > settle:
                return True, last_pos_err, last_rot_err
        else:
            stable_since = None

        if (time.time() - t0) > timeout:
            return False, last_pos_err, last_rot_err

        if DT_SLEEP > 0:
            time.sleep(DT_SLEEP)

def move_interp_pose(model, data, site_name, p_start, p_goal, target_R,
                     arm_joint_ids, arm_act_ids, duration, table_top_z):
    t0 = time.time()
    while True:
        t = (time.time() - t0) / max(duration, 1e-6)
        t = float(np.clip(t, 0.0, 1.0))
        s = t * t * (3.0 - 2.0 * t)

        p = (1 - s) * p_start + s * p_goal
        p = clamp_above_table(p, table_top_z)

        mujoco.mj_forward(model, data)
        for _ in range(IK_ITERS_PER_CYCLE):
            _ = ik_step_pose_qpos(model, data, site_name, p, target_R, arm_joint_ids)
            mujoco.mj_forward(model, data)
        sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

        if t >= 1.0:
            break
        if DT_SLEEP > 0:
            time.sleep(DT_SLEEP)

def close_gripper_ramp_hold(model, data, hold_site, hold_pos, hold_R,
                            arm_joint_ids, arm_act_ids, closeL, closeR):
    aL = actuator_id(model, "ee_gripper_left")
    aR = actuator_id(model, "ee_gripper_right")

    cL0 = float(data.ctrl[aL])
    cR0 = float(data.ctrl[aR])

    t0 = time.time()
    while (time.time() - t0) < CLOSE_RAMP_TIME:
        alpha = (time.time() - t0) / max(CLOSE_RAMP_TIME, 1e-6)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        cL = (1 - alpha) * cL0 + alpha * closeL
        cR = (1 - alpha) * cR0 + alpha * closeR

        data.ctrl[aL] = float(cL)
        data.ctrl[aR] = float(cR)

        mujoco.mj_forward(model, data)
        for _ in range(IK_ITERS_PER_CYCLE):
            _ = ik_step_pose_qpos(model, data, hold_site, hold_pos, hold_R, arm_joint_ids)
            mujoco.mj_forward(model, data)
        sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

    t1 = time.time()
    while (time.time() - t1) < CLOSE_HOLD_TIME:
        data.ctrl[aL] = float(closeL)
        data.ctrl[aR] = float(closeR)

        mujoco.mj_forward(model, data)
        for _ in range(IK_ITERS_PER_CYCLE):
            _ = ik_step_pose_qpos(model, data, hold_site, hold_pos, hold_R, arm_joint_ids)
            mujoco.mj_forward(model, data)
        sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)


def compute_tcp_targets_live(model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off):
    batt_pos = data.xpos[bid_batt].copy()
    rel_tcp_world = R_des @ rel_tcp

    obj_pre = batt_pos + pre_off
    obj_grip = batt_pos + grip_off
    obj_post = batt_pos + post_off

    tcp_pre = obj_pre - rel_tcp_world
    tcp_grip = obj_grip - rel_tcp_world
    tcp_post = obj_post - rel_tcp_world
    return tcp_pre, tcp_grip, tcp_post


# =========================
# SINGLE-GRASP EXPORT FORMAT
# =========================
def build_single_grasp_json(candidates_root, best_candidate):
    geometry = candidates_root.get("geometry", {})
    force_estimation = candidates_root.get("force_estimation", {})
    iteration_logic = candidates_root.get("iteration_logic", {})

    zmax = iteration_logic.get("zmax_of_object_mm", None)

    grip_point = best_candidate["grip_point_mm"]
    pre_grip = best_candidate["pre_grip_mm"]
    post_grip = best_candidate["post_grip_mm"]

    release_height_mm = best_candidate.get("release_height_mm", 25.0)

    pre_grip_offset_mm = None
    final_post_z_mm = None
    post_clearance_mm = None

    if pre_grip is not None:
        final_post_z_mm = float(post_grip[2])
    if zmax is not None and pre_grip is not None:
        pre_grip_offset_mm = float(pre_grip[2] - zmax)
    if zmax is not None and post_grip is not None:
        post_clearance_mm = float(post_grip[2] - zmax)

    return {
        "object_id": "battery_grip_COM",
        "relative_to_tcp": candidates_root.get("relative_to_tcp", [0.0, 0.0, 0.0]),
        "source": "freecad_grip_macro_COM_weighted",
        "units": candidates_root.get("units", "mm, N"),
        "geometry": {
            "bounding_box_mm": geometry.get("bounding_box_mm", [0.0, 0.0, 0.0]),
            "center_mm": geometry.get("center_mm", [0.0, 0.0, 0.0]),
            "principal_axis_world": geometry.get(
                "principal_axis_world",
                {"axis_dir_unit": [0.0, 0.0, 1.0]}
            )
        },
        "grasp_parameters": {
            "grip_point_mm": grip_point,
            "pre_grip_mm": pre_grip,
            "post_grip_mm": post_grip,
            "release_height_mm": release_height_mm
        },
        "force_estimation": {
            "mass_kg": force_estimation.get("mass_kg", 0.0),
            "friction_coefficient": force_estimation.get("friction_coefficient", 0.5),
            "safety_factor": force_estimation.get("safety_factor", 2.0),
            "normal_force_per_jaw_N": force_estimation.get("normal_force_per_jaw_N", 0.0),
            "total_normal_force_both_jaws_N": force_estimation.get("total_normal_force_both_jaws_N", 0.0)
        },
        "post_grip_logic": {
            "zmax_of_object": zmax,
            "pre_grip_offset_mm": pre_grip_offset_mm,
            "post_clearance_mm": post_clearance_mm,
            "final_post_z_mm": final_post_z_mm
        }
    }


# =========================
# CANDIDATE EVALUATION
# =========================
def evaluate_candidate(model, candidate, rel_tcp):
    data = mujoco.MjData(model)

    R_des = desired_tcp_R_vertical()
    arm_act_ids = get_arm_actuator_ids_strict(model)
    arm_joint_ids = get_arm_joint_ids_from_actuators(model, arm_act_ids)

    left_pad_gid = geom_id_strict(model, LEFT_PAD_NAME)
    right_pad_gid = geom_id_strict(model, RIGHT_PAD_NAME)
    left_pad_set = {left_pad_gid}
    right_pad_set = {right_pad_gid}

    battery_geom_ids = set(body_geom_ids(model, TARGET_BODY_NAME))
    bid_batt = body_id(model, TARGET_BODY_NAME)

    pre_off = np.array(candidate["pre_grip_mm"], dtype=float) * 1e-3
    grip_off = np.array(candidate["grip_point_mm"], dtype=float) * 1e-3
    post_off = np.array(candidate["post_grip_mm"], dtype=float) * 1e-3

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    pan = joint_id(model, "shoulder_pan_joint")
    arm_qpos_adr = model.jnt_qposadr[pan]
    data.qpos[arm_qpos_adr:arm_qpos_adr + 6] = SAFE_HOME_Q
    mujoco.mj_forward(model, data)
    sync_arm_ctrl_to_qpos(model, data, arm_act_ids)

    set_finger_collision(model, enable=False)
    set_gripper_open_targets(model, data)

    boost_friction(model, ["ee_left_col", "ee_right_col"], mult=FRICTION_MULT, slide_cap=FRICTION_SLIDE_CAP)
    boost_friction_ids(model, body_geom_ids(model, "battery"), mult=FRICTION_MULT, slide_cap=FRICTION_SLIDE_CAP)
    boost_gripper_strength(model, mult=GRIPPER_STRENGTH_MULT)

    mujoco.mj_forward(model, data)
    table_top_z = get_table_top_z(model, data)

    for _ in range(int(T_HOME_SETTLE / 0.002)):
        mujoco.mj_step(model, data)

    tcp_pre, tcp_grip, _ = compute_tcp_targets_live(
        model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off
    )
    tcp_pre = clamp_above_table(tcp_pre, table_top_z)
    tcp_grip = clamp_above_table(tcp_grip, table_top_z)

    p0 = data.site_xpos[site_id(model, SITE_TCP)].copy()

    move_interp_pose(model, data, SITE_TCP, p0, tcp_pre, R_des,
                     arm_joint_ids, arm_act_ids, T_TO_PRE, table_top_z)

    ok_pre, pre_pos_err, pre_rot_err = gate_reach_pose(
        model, data, SITE_TCP, tcp_pre, R_des,
        arm_joint_ids, arm_act_ids, POS_TOL_PRE, ROT_TOL
    )

    if not ok_pre:
        return {
            "candidate_id": candidate["candidate_id"],
            "accepted": False,
            "reason": "PRE pose not reached",
            "pre_pos_err": pre_pos_err,
            "pre_rot_err": pre_rot_err,
            "grip_point_mm": candidate["grip_point_mm"],
            "pre_grip_mm": candidate["pre_grip_mm"],
            "post_grip_mm": candidate["post_grip_mm"],
            "release_height_mm": candidate.get("release_height_mm", 25.0),
        }

    t_wait = time.time()
    while time.time() - t_wait < WAIT_AT_PRE:
        mujoco.mj_forward(model, data)
        tcp_pre_live, _, _ = compute_tcp_targets_live(
            model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off
        )
        tcp_pre_live = clamp_above_table(tcp_pre_live, table_top_z)
        for _ in range(IK_ITERS_PER_CYCLE):
            _ = ik_step_pose_qpos(model, data, SITE_TCP, tcp_pre_live, R_des, arm_joint_ids)
            mujoco.mj_forward(model, data)
        sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)
    _, tcp_grip, _ = compute_tcp_targets_live(
        model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off
    )
    tcp_grip = clamp_above_table(tcp_grip, table_top_z)

    tcp_near = tcp_grip.copy()
    tcp_near[2] += 0.015

    p1 = data.site_xpos[site_id(model, SITE_TCP)].copy()
    move_interp_pose(model, data, SITE_TCP, p1, tcp_near, R_des,
                     arm_joint_ids, arm_act_ids, T_TO_GRIP, table_top_z)

    set_finger_collision(model, enable=True)
    mujoco.mj_forward(model, data)

    t0 = time.time()
    ok_grip = False
    grip_pos_err = None
    grip_rot_err = None

    while time.time() - t0 < 8.0:
        mujoco.mj_forward(model, data)
        _, tcp_grip_live, _ = compute_tcp_targets_live(
            model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off
        )
        tcp_grip_live = clamp_above_table(tcp_grip_live, table_top_z)
        p_cur = data.site_xpos[site_id(model, SITE_TCP)].copy()

        p_step = p_cur + 0.25 * (tcp_grip_live - p_cur)
        move_interp_pose(model, data, SITE_TCP, p_cur, p_step, R_des,
                         arm_joint_ids, arm_act_ids, 0.15, table_top_z)

        ok_grip, grip_pos_err, grip_rot_err = gate_reach_pose(
            model, data, SITE_TCP, tcp_grip_live, R_des,
            arm_joint_ids, arm_act_ids, POS_TOL_GRIP, ROT_TOL
        )
        if ok_grip:
            break

    if not ok_grip:
        return {
            "candidate_id": candidate["candidate_id"],
            "accepted": False,
            "reason": "GRIP pose not reached",
            "pre_pos_err": pre_pos_err,
            "pre_rot_err": pre_rot_err,
            "grip_pos_err": grip_pos_err,
            "grip_rot_err": grip_rot_err,
            "grip_point_mm": candidate["grip_point_mm"],
            "pre_grip_mm": candidate["pre_grip_mm"],
            "post_grip_mm": candidate["post_grip_mm"],
            "release_height_mm": candidate.get("release_height_mm", 25.0),
        }

    mujoco.mj_forward(model, data)
    closeL, closeR = autocalibrate_close_targets(model, data)

    closeL -= SQUEEZE_EXTRA
    closeR += SQUEEZE_EXTRA

    if USE_PINCH_VISUAL_CLOSE:
        pd = pinch_distance(model, data)
        err = pd - TARGET_PINCH
        delta = float(np.clip(PINCH_GAIN * err, 0.0, PINCH_MAX_EXTRA))
        closeL -= delta
        closeR += delta

    jL = joint_id(model, "ee_gripper_left_joint")
    jR = joint_id(model, "ee_gripper_right_joint")
    rL = model.jnt_range[jL]
    rR = model.jnt_range[jR]
    closeL = float(np.clip(closeL, rL[0], rL[1]))
    closeR = float(np.clip(closeR, rR[0], rR[1]))

    _, tcp_grip_hold, _ = compute_tcp_targets_live(
        model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off
    )
    tcp_grip_hold = clamp_above_table(tcp_grip_hold, table_top_z)

    close_gripper_ramp_hold(model, data, SITE_TCP, tcp_grip_hold, R_des,
                            arm_joint_ids, arm_act_ids, closeL, closeR)

    t_hold = time.time()
    while time.time() - t_hold < POST_CLOSE_HOLD:
        mujoco.mj_forward(model, data)
        _, tcp_grip_live, _ = compute_tcp_targets_live(
            model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off
        )
        tcp_grip_live = clamp_above_table(tcp_grip_live, table_top_z)

        set_gripper_close_targets(model, data, closeL, closeR)
        for _ in range(IK_ITERS_PER_CYCLE):
            _ = ik_step_pose_qpos(model, data, SITE_TCP, tcp_grip_live, R_des, arm_joint_ids)
            mujoco.mj_forward(model, data)
        sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

    left_contacts = count_specific_contacts(data, left_pad_set, battery_geom_ids)
    right_contacts = count_specific_contacts(data, right_pad_set, battery_geom_ids)
    both_touch = (left_contacts > 0) and (right_contacts > 0)

    forbidden, forbidden_info = has_forbidden_collision(
        model, data, GRIPPER_BODY_NAMES, FORBIDDEN_BODY_NAMES
    )

    accepted = both_touch and (not forbidden)
    reason = "valid"
    if not both_touch and forbidden:
        reason = "no two-pad battery contact and forbidden collision"
    elif not both_touch:
        reason = "both pads did not contact battery"
    elif forbidden:
        reason = "collision with casing or battery2"

    return {
        "candidate_id": candidate["candidate_id"],
        "accepted": accepted,
        "reason": reason,
        "pre_pos_err": pre_pos_err,
        "pre_rot_err": pre_rot_err,
        "grip_pos_err": grip_pos_err,
        "grip_rot_err": grip_rot_err,
        "left_pad_battery_contacts": int(left_contacts),
        "right_pad_battery_contacts": int(right_contacts),
        "both_pads_touch_battery": bool(both_touch),
        "forbidden_collision": bool(forbidden),
        "forbidden_collision_info": forbidden_info,
        "grip_point_mm": candidate["grip_point_mm"],
        "pre_grip_mm": candidate["pre_grip_mm"],
        "post_grip_mm": candidate["post_grip_mm"],
        "clearance_mm": candidate.get("clearance_mm"),
        "gripper_open_width_mm": candidate.get("gripper_open_width_mm"),
        "release_height_mm": candidate.get("release_height_mm", 25.0),
    }


# =========================
# MAIN
# =========================
def main():
    candidates_root, rel_tcp, candidates = load_candidate_set(JSON_PATH)
    model = mujoco.MjModel.from_xml_path(XML_PATH)

    valid = []

    print(f"Loaded {len(candidates)} candidates from {JSON_PATH}")

    for cand in candidates:
        print(f"\n=== Evaluating candidate {cand['candidate_id']} ===")
        try:
            res = evaluate_candidate(model, cand, rel_tcp)
        except Exception as e:
            res = {
                "candidate_id": cand["candidate_id"],
                "accepted": False,
                "reason": f"exception: {str(e)}",
                "grip_point_mm": cand.get("grip_point_mm"),
                "pre_grip_mm": cand.get("pre_grip_mm"),
                "post_grip_mm": cand.get("post_grip_mm"),
                "release_height_mm": cand.get("release_height_mm", 25.0),
            }

        if res["accepted"]:
            valid.append(res)
            print(
                f"VALID  | candidate {cand['candidate_id']} | "
                f"grip_z={res['grip_point_mm'][2]} | {res['reason']}"
            )
        else:
            print(f"REJECT | candidate {cand['candidate_id']} | {res['reason']}")

    # Best valid candidate = lowest grip point z, tie-break by candidate_id
    best = None
    if valid:
        best = min(valid, key=lambda x: (x["grip_point_mm"][2], x["candidate_id"]))

    # Evaluation JSON contains ONLY the optimal candidate
    summary = {
        "xml_path": XML_PATH,
        "json_path": JSON_PATH,
        "selection_rule": "best valid candidate = lowest grip_point_mm[2], tie-break by candidate_id",
        "num_candidates": len(candidates),
        "num_valid_candidates": len(valid),
        "best_valid_candidate": best
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if best is not None:
        best_single = build_single_grasp_json(candidates_root, best)
        with open(BEST_CANDIDATE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(best_single, f, indent=4)

    print("\n==============================")
    print(f"Saved evaluation summary to: {RESULTS_PATH}")
    if best is not None:
        print(f"Saved best-candidate JSON to: {BEST_CANDIDATE_JSON_PATH}")
        print(
            f"Best valid candidate: {best['candidate_id']} "
            f"(lowest grip z = {best['grip_point_mm'][2]})"
        )
    else:
        print("No valid candidates found.")
    print("==============================")


if __name__ == "__main__":
    main()