import time
import json
import threading
import numpy as np
import mujoco
import mujoco.viewer

# =========================
# FILES
# =========================
XML_PATH = "scene_new.xml"
JSON_PATH = "battery_grip_data.json"

# =========================
# USER SETTINGS
# =========================
SAFE_HOME_Q = np.array([0.88, -2.01, 1.88, -1.51, -1.51, -0.628], dtype=float)
SITE_TCP = "tcp_tip"

# Timing
T_HOME_SETTLE = 0.6
T_TO_PRE  = 2.0
WAIT_AT_PRE = 0.5         # short; we pre-close here
T_TO_GRIP = 2.0
T_TO_POST = 2.5

DT_SLEEP = 0.004
SUBSTEPS = 4

# Table safety
TABLE_CLEARANCE = 0.000  # meters; 0 disables

# Gate tolerances
POS_TOL_PRE  = 0.0010    # 1.0 mm
POS_TOL_GRIP = 0.0005    # 0.5 mm
ROT_TOL      = 0.04      # rad
SETTLE_TIME  = 0.20
GATE_TIMEOUT = 12.0

# IK tuning
IK_ITERS_PER_CYCLE = 8
IK_DAMP = 0.02
IK_STEP = 0.55

# =========================
# BATTERY + GRIPPER WIDTHS
# =========================
BATTERY_HALF_WIDTH = 0.00725
BATTERY_DIAMETER = 2.0 * BATTERY_HALF_WIDTH

# NOTE: your comment said 0.5mm/side, but you had 0.002 (2mm) in code.
# If you truly want 0.5mm per side, use 0.0005.
PRE_CLEAR_PER_SIDE = 0.001  # 0.5 mm per side
PRE_PINCH_TARGET = BATTERY_DIAMETER + 2.0 * PRE_CLEAR_PER_SIDE

# Final pinch limit (don’t crush below diameter minus small squeeze allowance)
SQUEEZE_ALLOW = 0.0005   # allow 0.5mm total squeeze
MIN_PINCH = max(0.001, BATTERY_DIAMETER - SQUEEZE_ALLOW)

# =========================
# CONTACT + SQUEEZE REQUIREMENTS
# =========================
# “Squeeze” measured by normal force on each finger contact (Newtons).
FN_SQUEEZE_N = 1.5         # start with 1.0~3.0; increase if still “loose”
STABLE_STEPS = 8           # require squeeze stable for N consecutive sim steps
SQUEEZE_STEP = 0.0008      # how much to tighten per iteration during squeeze stage

# Gripper “servo” behavior
PRE_CLOSE_SECONDS = 2.0
PRE_CLOSE_TOL = 0.0005

CLOSE_SECONDS = 1.0
CLOSE_TOL = 0.0004

# Grasp robustness
FRICTION_MULT = 3.0
FRICTION_SLIDE_CAP = 25.0
GRIPPER_STRENGTH_MULT = 2.0

HOLD_AFTER_DONE_SECONDS = 3.0

# Actuator names (match your XML)
ARM_ACT_NAMES = ["shoulder_pan", "shoulder_lift", "forearm", "wrist_1", "wrist_2", "wrist_3"]

# Finger collision geoms to disable during approach
FINGER_COL_GEOMS = ["ee_left_col", "ee_right_col"]

# "Magnets" (soft attach)
ENABLE_SOFT_ATTACH = True

# Require BOTH col boxes contact before attaching
COL_GEOMS = ["ee_left_col", "ee_right_col"]


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

def body_geom_ids(model, body_name: str):
    bid = body_id(model, body_name)
    gadr = int(model.body_geomadr[bid])
    gnum = int(model.body_geomnum[bid])
    return list(range(gadr, gadr + gnum))


# =========================
# JSON LOADER
# =========================
def load_battery_params(json_path: str):
    with open(json_path, "r") as f:
        j = json.load(f)

    rel_tcp = np.array(j.get("relative_to_tcp", [0.0, 0.0, 0.0]), dtype=float).reshape(3) * 1e-3
    gp = j["grasp_parameters"]
    pre  = np.array(gp["pre_grip_mm"], dtype=float).reshape(3) * 1e-3
    grip = np.array(gp["grip_point_mm"], dtype=float).reshape(3) * 1e-3
    post = np.array(gp["post_grip_mm"], dtype=float).reshape(3) * 1e-3
    return rel_tcp, pre, grip, post


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
# COLLISION TOGGLE
# =========================
def set_finger_collision(model, enable: bool):
    contype = 1 if enable else 0
    conaff  = 1 if enable else 0
    for gname in FINGER_COL_GEOMS:
        gid = geom_id(model, gname)
        if gid == -1:
            continue
        model.geom_contype[gid] = contype
        model.geom_conaffinity[gid] = conaff


# =========================
# FRICTION + STRENGTH
# =========================
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
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            continue
        model.actuator_gear[aid, :] *= mult
        model.actuator_forcerange[aid, 0] *= mult
        model.actuator_forcerange[aid, 1] *= mult


# =========================
# ARM IDS
# =========================
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


# =========================
# GRIPPER (pinch-based, sign-safe)
# =========================
def pinch_distance(model, data):
    sL = site_id(model, "pinch_L")
    sR = site_id(model, "pinch_R")
    pL = data.site_xpos[sL].copy()
    pR = data.site_xpos[sR].copy()
    return float(np.linalg.norm(pL - pR))

def detect_close_dirs(model, data):
    """Return dirL, dirR such that increasing ctrl by dir*+eps tends to CLOSE (reduce pinch)."""
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
    mujoco.mj_forward(model, data)
    return dirL, dirR

def set_gripper_ctrl(model, data, cL, cR):
    aL = actuator_id(model, "ee_gripper_left")
    aR = actuator_id(model, "ee_gripper_right")
    cL = float(np.clip(cL, model.actuator_ctrlrange[aL, 0], model.actuator_ctrlrange[aL, 1]))
    cR = float(np.clip(cR, model.actuator_ctrlrange[aR, 0], model.actuator_ctrlrange[aR, 1]))
    data.ctrl[aL] = cL
    data.ctrl[aR] = cR

def drive_pinch_to_target(model, data, lock, target_pd, hold_site, hold_pos, hold_R,
                          arm_joint_ids, arm_act_ids, seconds=0.8, tol=0.0003):
    """Hold TCP pose and servo gripper until pinch ~= target_pd (sign-safe)."""
    aL = actuator_id(model, "ee_gripper_left")
    aR = actuator_id(model, "ee_gripper_right")

    with lock:
        mujoco.mj_forward(model, data)
        dirL, dirR = detect_close_dirs(model, data)

    K = 0.75
    MAX_STEP = 0.0030

    t0 = time.time()
    while time.time() - t0 < seconds:
        with lock:
            mujoco.mj_forward(model, data)
            pd = pinch_distance(model, data)
            err = pd - float(target_pd)
            if abs(err) <= tol:
                return True, pd

            step = float(np.clip(K * 0.5 * err, -MAX_STEP, MAX_STEP))
            cL = float(data.ctrl[aL] + dirL * step)
            cR = float(data.ctrl[aR] + dirR * step)
            set_gripper_ctrl(model, data, cL, cR)

            for _ in range(IK_ITERS_PER_CYCLE):
                _ = ik_step_pose_qpos(model, data, hold_site, hold_pos, hold_R, arm_joint_ids)
                mujoco.mj_forward(model, data)
            sync_arm_ctrl_to_qpos(model, data, arm_act_ids)

            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)

        time.sleep(DT_SLEEP)

    with lock:
        mujoco.mj_forward(model, data)
        return False, pinch_distance(model, data)


# =========================
# CONTACTS + FORCE (SQUEEZE)
# =========================
def finger_contact_forces(model, data, battery_geom_ids):
    """
    Returns:
      left_ok, right_ok: bool (contact exists)
      fnL, fnR: summed normal forces (N) for contacts between each col geom and battery
    """
    gL = geom_id(model, "ee_left_col")
    gR = geom_id(model, "ee_right_col")

    left_ok = False
    right_ok = False
    fnL = 0.0
    fnR = 0.0

    cf = np.zeros(6, dtype=float)

    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)

        # only care about contacts involving battery
        if not ((g1 in battery_geom_ids) or (g2 in battery_geom_ids)):
            continue

        mujoco.mj_contactForce(model, data, i, cf)
        fn = float(cf[0])  # normal force

        if (g1 == gL and g2 in battery_geom_ids) or (g2 == gL and g1 in battery_geom_ids):
            left_ok = True
            fnL += fn
        if (g1 == gR and g2 in battery_geom_ids) or (g2 == gR and g1 in battery_geom_ids):
            right_ok = True
            fnR += fn

    return left_ok, right_ok, fnL, fnR


def tighten_until_both_cols_touch(model, data, lock, battery_geom_ids,
                                 hold_site, hold_pos, hold_R,
                                 arm_joint_ids, arm_act_ids,
                                 max_iters=40, step_joint=0.0010):
    """Tighten symmetrically until BOTH col boxes TOUCH the battery (contact exists)."""
    aL = actuator_id(model, "ee_gripper_left")
    aR = actuator_id(model, "ee_gripper_right")

    with lock:
        mujoco.mj_forward(model, data)
        dirL, dirR = detect_close_dirs(model, data)

    for _ in range(max_iters):
        with lock:
            mujoco.mj_forward(model, data)
            left_ok, right_ok, fnL, fnR = finger_contact_forces(model, data, battery_geom_ids)
            if left_ok and right_ok:
                return True, left_ok, right_ok, fnL, fnR, pinch_distance(model, data)

            # tighten a bit
            cL = float(data.ctrl[aL] + dirL * step_joint)
            cR = float(data.ctrl[aR] + dirR * step_joint)
            set_gripper_ctrl(model, data, cL, cR)

            for _ in range(IK_ITERS_PER_CYCLE):
                _ = ik_step_pose_qpos(model, data, hold_site, hold_pos, hold_R, arm_joint_ids)
                mujoco.mj_forward(model, data)
            sync_arm_ctrl_to_qpos(model, data, arm_act_ids)

            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)

        time.sleep(DT_SLEEP)

    with lock:
        mujoco.mj_forward(model, data)
        left_ok, right_ok, fnL, fnR = finger_contact_forces(model, data, battery_geom_ids)
        return False, left_ok, right_ok, fnL, fnR, pinch_distance(model, data)


def squeeze_until_force(model, data, lock, battery_geom_ids,
                        hold_site, hold_pos, hold_R,
                        arm_joint_ids, arm_act_ids,
                        fn_target=1.5, stable_steps=8,
                        step_joint=0.0008, max_iters=80):
    """
    After BOTH contacts exist, keep tightening until both normal forces exceed fn_target
    for 'stable_steps' consecutive checks, OR until pinch hits MIN_PINCH.
    """
    aL = actuator_id(model, "ee_gripper_left")
    aR = actuator_id(model, "ee_gripper_right")

    with lock:
        mujoco.mj_forward(model, data)
        dirL, dirR = detect_close_dirs(model, data)

    stable = 0
    for _ in range(max_iters):
        with lock:
            mujoco.mj_forward(model, data)

            pd = pinch_distance(model, data)
            left_ok, right_ok, fnL, fnR = finger_contact_forces(model, data, battery_geom_ids)

            # Must keep both contacts
            if not (left_ok and right_ok):
                stable = 0
            else:
                if (fnL >= fn_target) and (fnR >= fn_target):
                    stable += 1
                else:
                    stable = 0

            if stable >= stable_steps:
                return True, left_ok, right_ok, fnL, fnR, pd, True  # squeezed_ok

            # stop if pinch too small
            if pd <= MIN_PINCH + 1e-6:
                return False, left_ok, right_ok, fnL, fnR, pd, False

            # tighten a bit
            cL = float(data.ctrl[aL] + dirL * step_joint)
            cR = float(data.ctrl[aR] + dirR * step_joint)
            set_gripper_ctrl(model, data, cL, cR)

            # hold pose
            for _ in range(IK_ITERS_PER_CYCLE):
                _ = ik_step_pose_qpos(model, data, hold_site, hold_pos, hold_R, arm_joint_ids)
                mujoco.mj_forward(model, data)
            sync_arm_ctrl_to_qpos(model, data, arm_act_ids)

            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)

        time.sleep(DT_SLEEP)

    with lock:
        mujoco.mj_forward(model, data)
        pd = pinch_distance(model, data)
        left_ok, right_ok, fnL, fnR = finger_contact_forces(model, data, battery_geom_ids)
        return False, left_ok, right_ok, fnL, fnR, pd, False


# =========================
# ORIENTATION + IK
# =========================
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

    return float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))

def gate_reach_pose(model, data, lock, site_name, target_pos, target_R,
                    arm_joint_ids, arm_act_ids, label,
                    pos_tol, rot_tol,
                    settle=SETTLE_TIME, timeout=GATE_TIMEOUT):
    t0 = time.time()
    stable_since = None
    last_pos_err = None
    last_rot_err = None

    while True:
        with lock:
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
                print(f"[GATE {label}] ok=True pos_err={last_pos_err:.6f} rot_err={last_rot_err:.6f}")
                return True
        else:
            stable_since = None

        if (time.time() - t0) > timeout:
            print(f"[GATE {label}] ok=False pos_err={last_pos_err:.6f} rot_err={last_rot_err:.6f}")
            return False

        time.sleep(DT_SLEEP)

def move_interp_pose(model, data, lock, site_name, p_start, p_goal, target_R,
                     arm_joint_ids, arm_act_ids, duration, table_top_z):
    t0 = time.time()
    while True:
        t = (time.time() - t0) / max(duration, 1e-6)
        t = float(np.clip(t, 0.0, 1.0))
        s = t * t * (3.0 - 2.0 * t)

        p = (1 - s) * p_start + s * p_goal
        p = clamp_above_table(p, table_top_z)

        with lock:
            mujoco.mj_forward(model, data)
            for _ in range(IK_ITERS_PER_CYCLE):
                _ = ik_step_pose_qpos(model, data, site_name, p, target_R, arm_joint_ids)
                mujoco.mj_forward(model, data)
            sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)

        if t >= 1.0:
            break
        time.sleep(DT_SLEEP)


# =========================
# SOFT ATTACH (“MAGNETS”)
# =========================
def get_freejoint_qadr(model, body_name):
    bid = body_id(model, body_name)
    jnum = model.body_jntnum[bid]
    if jnum < 1:
        raise RuntimeError(f"Body {body_name} has no joint.")
    jid = int(model.body_jntadr[bid])
    if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError(f"Body {body_name} is not freejoint.")
    return model.jnt_qposadr[jid], model.jnt_dofadr[jid]

def mat_to_quat(Rm):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Rm.reshape(9))
    if q[0] < 0:
        q *= -1.0
    return q

def quat_mul(q1, q2):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q1, q2)
    return out

def quat_inv(q):
    out = q.copy()
    mujoco.mju_negQuat(out, out)
    return out

def quat_rot(q, v):
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, v, q)
    return out


# =========================
# LIVE TARGETS
# =========================
def compute_tcp_targets_live(model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off):
    batt_pos = data.xpos[bid_batt].copy()
    rel_tcp_world = R_des @ rel_tcp

    obj_pre  = batt_pos + pre_off
    obj_grip = batt_pos + grip_off
    obj_post = batt_pos + post_off

    tcp_pre  = obj_pre  - rel_tcp_world
    tcp_grip = obj_grip - rel_tcp_world
    tcp_post = obj_post - rel_tcp_world
    return tcp_pre, tcp_grip, tcp_post


# =========================
# MAIN SEQUENCE
# =========================
def run_sequence(model, data, lock):
    rel_tcp, pre_off, grip_off, post_off = load_battery_params(JSON_PATH)
    R_des = desired_tcp_R_vertical()
    rel_tcp_world = R_des @ rel_tcp

    arm_act_ids = get_arm_actuator_ids_strict(model)
    arm_joint_ids = get_arm_joint_ids_from_actuators(model, arm_act_ids)

    bid_batt = body_id(model, "battery")
    batt_geoms = set(body_geom_ids(model, "battery"))
    batt_qadr, batt_dofadr = get_freejoint_qadr(model, "battery")

    with lock:
        pan = joint_id(model, "shoulder_pan_joint")
        arm_qpos_adr = model.jnt_qposadr[pan]
        data.qpos[arm_qpos_adr:arm_qpos_adr + 6] = SAFE_HOME_Q

        mujoco.mj_forward(model, data)
        sync_arm_ctrl_to_qpos(model, data, arm_act_ids)

        # Start gripper neutral (IMPORTANT: avoids opening to limit unexpectedly)
        set_gripper_ctrl(model, data, 0.0, 0.0)

        set_finger_collision(model, enable=False)

        boost_friction(model, ["ee_left_col", "ee_right_col"], mult=FRICTION_MULT, slide_cap=FRICTION_SLIDE_CAP)
        boost_friction_ids(model, list(batt_geoms), mult=FRICTION_MULT, slide_cap=FRICTION_SLIDE_CAP)
        boost_gripper_strength(model, mult=GRIPPER_STRENGTH_MULT)

        mujoco.mj_forward(model, data)

        table_top_z = get_table_top_z(model, data)
        if table_top_z is not None:
            print(f"[INFO] Table top z = {table_top_z:.4f} m, clearance = {TABLE_CLEARANCE:.3f} m")
        else:
            print("[WARN] Could not detect table top. No Z clamp.")

        print("[OK] Initialized home, finger collision OFF")

    time.sleep(T_HOME_SETTLE)

    with lock:
        mujoco.mj_forward(model, data)
        tcp_pre, tcp_grip, tcp_post = compute_tcp_targets_live(model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off)
        tcp_pre  = clamp_above_table(tcp_pre, table_top_z)
        tcp_grip = clamp_above_table(tcp_grip, table_top_z)
        tcp_post = clamp_above_table(tcp_post, table_top_z)
        p0 = data.site_xpos[site_id(model, SITE_TCP)].copy()

    # PRE
    print("[STEP] tcp_tip -> PRE_GRIP (vertical)")
    move_interp_pose(model, data, lock, SITE_TCP, p0, tcp_pre, R_des,
                     arm_joint_ids, arm_act_ids, T_TO_PRE, table_top_z)

    if not gate_reach_pose(model, data, lock, SITE_TCP, tcp_pre, R_des,
                           arm_joint_ids, arm_act_ids, label="PRE",
                           pos_tol=POS_TOL_PRE, rot_tol=ROT_TOL):
        print("[STOP] Could not reach PRE.")
        return

    # PRE-CLOSE to battery diameter + clearance (reduces “wide open” near object)
    print(f"[PRE-CLOSE] Driving pinch to {PRE_PINCH_TARGET:.6f} m")
    ok, pd = drive_pinch_to_target(
        model, data, lock,
        target_pd=PRE_PINCH_TARGET,
        hold_site=SITE_TCP, hold_pos=tcp_pre, hold_R=R_des,
        arm_joint_ids=arm_joint_ids, arm_act_ids=arm_act_ids,
        seconds=PRE_CLOSE_SECONDS, tol=PRE_CLOSE_TOL
    )
    print(f"[PRE-CLOSE] ok={ok} pinch={pd:.6f}")

    if WAIT_AT_PRE > 0:
        time.sleep(WAIT_AT_PRE)

    # GRIP approach
    with lock:
        mujoco.mj_forward(model, data)
        _, tcp_grip, _ = compute_tcp_targets_live(model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off)
        tcp_grip = clamp_above_table(tcp_grip, table_top_z)

    print("[STEP] tcp_tip -> NEAR_GRIP (collision OFF)")
    tcp_near = tcp_grip.copy()
    tcp_near[2] += 0.015

    with lock:
        mujoco.mj_forward(model, data)
        p1 = data.site_xpos[site_id(model, SITE_TCP)].copy()

    move_interp_pose(model, data, lock, SITE_TCP, p1, tcp_near, R_des,
                     arm_joint_ids, arm_act_ids, T_TO_GRIP, table_top_z)

    print("[STEP] Enable finger col geoms BEFORE final approach")
    with lock:
        set_finger_collision(model, enable=True)
        mujoco.mj_forward(model, data)

    print("[STEP] tcp_tip -> GRIP_POINT (collision ON)")
    move_interp_pose(model, data, lock, SITE_TCP, tcp_near, tcp_grip, R_des,
                     arm_joint_ids, arm_act_ids, 0.35, table_top_z)

    if not gate_reach_pose(model, data, lock, SITE_TCP, tcp_grip, R_des,
                           arm_joint_ids, arm_act_ids, label="GRIP",
                           pos_tol=POS_TOL_GRIP, rot_tol=ROT_TOL):
        print("[STOP] Could not reach GRIP.")
        return

    # Close toward diameter (but never below MIN_PINCH)
    print("[STEP] CLOSE gripper to near battery diameter (pinch servo)")
    target_close = max(MIN_PINCH, BATTERY_DIAMETER)
    ok_close, pd_close = drive_pinch_to_target(
        model, data, lock,
        target_pd=target_close,
        hold_site=SITE_TCP, hold_pos=tcp_grip, hold_R=R_des,
        arm_joint_ids=arm_joint_ids, arm_act_ids=arm_act_ids,
        seconds=CLOSE_SECONDS, tol=CLOSE_TOL
    )
    with lock:
        mujoco.mj_forward(model, data)
        pd_now = pinch_distance(model, data)
    print(f"[PINCH AFTER RAMP] ok={ok_close} pinch={pd_now:.6f}  MIN_PINCH={MIN_PINCH:.6f}")

    # Stage A: tighten until BOTH contacts exist
    print("[STEP] Tighten until BOTH col boxes touch the battery (contact-only)")
    cols_ok, left_ok, right_ok, fnL, fnR, pd_now = tighten_until_both_cols_touch(
        model, data, lock,
        battery_geom_ids=batt_geoms,
        hold_site=SITE_TCP, hold_pos=tcp_grip, hold_R=R_des,
        arm_joint_ids=arm_joint_ids, arm_act_ids=arm_act_ids,
        max_iters=50, step_joint=0.0010
    )
    print(f"[CONTACTS] cols_ok={cols_ok} left={left_ok} right={right_ok}  fnL={fnL:.3f}N fnR={fnR:.3f}N  pinch={pd_now:.6f}")

    # Stage B: ONLY if both contacts exist, squeeze until BOTH forces exceed FN_SQUEEZE_N stably
    squeezed_ok = False
    if left_ok and right_ok:
        print(f"[STEP] SQUEEZE until both fingers have fn>={FN_SQUEEZE_N:.2f}N for {STABLE_STEPS} steps (or pinch limit)")
        ok_sq, left_ok, right_ok, fnL, fnR, pd_now, squeezed_ok = squeeze_until_force(
            model, data, lock,
            battery_geom_ids=batt_geoms,
            hold_site=SITE_TCP, hold_pos=tcp_grip, hold_R=R_des,
            arm_joint_ids=arm_joint_ids, arm_act_ids=arm_act_ids,
            fn_target=FN_SQUEEZE_N, stable_steps=STABLE_STEPS,
            step_joint=SQUEEZE_STEP, max_iters=120
        )
        print(f"[SQUEEZE] ok={ok_sq} left={left_ok} right={right_ok}  fnL={fnL:.3f}N fnR={fnR:.3f}N  pinch={pd_now:.6f}  squeezed_ok={squeezed_ok}")
    else:
        print("[SQUEEZE] skipped (missing initial both contacts).")

    # Magnet attach ONLY after both contacts + squeeze
    attached = False
    tcp_to_batt_pos = None
    tcp_to_batt_quat = None

    if ENABLE_SOFT_ATTACH and left_ok and right_ok and squeezed_ok and (pd_now >= MIN_PINCH - 1e-6):
        with lock:
            mujoco.mj_forward(model, data)
            sid = site_id(model, SITE_TCP)
            tcp_pos = data.site_xpos[sid].copy()
            tcp_R = data.site_xmat[sid].reshape(3, 3).copy()
            tcp_q = mat_to_quat(tcp_R)

            batt_pos_w = data.xpos[bid_batt].copy()
            batt_q = data.qpos[batt_qadr + 3: batt_qadr + 7].copy()

            tcp_q_inv = quat_inv(tcp_q)
            rel_pos = batt_pos_w - tcp_pos
            rel_pos_tcp = quat_rot(tcp_q_inv, rel_pos)
            rel_q = quat_mul(tcp_q_inv, batt_q)

            tcp_to_batt_pos = rel_pos_tcp
            tcp_to_batt_quat = rel_q
            attached = True
            print("[MAGNET] Attached (BOTH contacts + squeeze stable).")
    else:
        print("[MAGNET] Not attaching (waiting for BOTH contacts + squeeze).")

    # Freeze post target
    with lock:
        mujoco.mj_forward(model, data)
        batt_pos_ref = data.xpos[bid_batt].copy()
    obj_post_ref = batt_pos_ref + post_off
    tcp_post_frozen = clamp_above_table(obj_post_ref - rel_tcp_world, table_top_z)

    print("[STEP] tcp_tip -> POST_GRIP (vertical lift)")
    with lock:
        mujoco.mj_forward(model, data)
        p2 = data.site_xpos[site_id(model, SITE_TCP)].copy()

    move_interp_pose(model, data, lock, SITE_TCP, p2, tcp_post_frozen, R_des,
                     arm_joint_ids, arm_act_ids, T_TO_POST, table_top_z)

    # enforce attach briefly
    t_hold = time.time()
    while time.time() - t_hold < 0.25:
        with lock:
            mujoco.mj_forward(model, data)
            sync_arm_ctrl_to_qpos(model, data, arm_act_ids)

            if attached and tcp_to_batt_pos is not None:
                sid = site_id(model, SITE_TCP)
                tcp_pos = data.site_xpos[sid].copy()
                tcp_R = data.site_xmat[sid].reshape(3, 3).copy()
                tcp_q = mat_to_quat(tcp_R)

                batt_pos_w = tcp_pos + quat_rot(tcp_q, tcp_to_batt_pos)
                batt_q = quat_mul(tcp_q, tcp_to_batt_quat)

                data.qpos[batt_qadr + 0:batt_qadr + 3] = batt_pos_w
                data.qpos[batt_qadr + 3:batt_qadr + 7] = batt_q
                data.qvel[batt_dofadr:batt_dofadr + 6] = 0.0

            mujoco.mj_step(model, data)

        time.sleep(DT_SLEEP)

    with lock:
        mujoco.mj_forward(model, data)
        print("[BATTERY Z]", float(data.xpos[bid_batt][2]))

    print("[DONE] Pick sequence finished.")

    if HOLD_AFTER_DONE_SECONDS > 0:
        print(f"[HOLD] Maintaining grasp for {HOLD_AFTER_DONE_SECONDS:.2f}s")
        t_end = time.time() + HOLD_AFTER_DONE_SECONDS
        while time.time() < t_end:
            with lock:
                sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
                if attached and tcp_to_batt_pos is not None:
                    sid = site_id(model, SITE_TCP)
                    tcp_pos = data.site_xpos[sid].copy()
                    tcp_R = data.site_xmat[sid].reshape(3, 3).copy()
                    tcp_q = mat_to_quat(tcp_R)

                    batt_pos_w = tcp_pos + quat_rot(tcp_q, tcp_to_batt_pos)
                    batt_q = quat_mul(tcp_q, tcp_to_batt_quat)

                    data.qpos[batt_qadr + 0:batt_qadr + 3] = batt_pos_w
                    data.qpos[batt_qadr + 3:batt_qadr + 7] = batt_q
                    data.qvel[batt_dofadr:batt_dofadr + 6] = 0.0

                mujoco.mj_forward(model, data)
            time.sleep(DT_SLEEP)


# =========================
# ENTRYPOINT
# =========================
def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    lock = threading.Lock()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.7, 0.25, 0.95]
        viewer.cam.distance = 0.8

        th = threading.Thread(target=run_sequence, args=(model, data, lock), daemon=True)
        th.start()

        while viewer.is_running():
            with lock:
                viewer.sync()
            time.sleep(0.02)

if __name__ == "__main__":
    main()