import time
import json
import threading
import numpy as np
import mujoco
import mujoco.viewer

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
WAIT_AT_PRE = 3.0
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

# Gripper open commands (actuator ctrl)
OPEN_L_CMD = 0.00875
OPEN_R_CMD = -0.00875

# Closing
CLOSE_MAG = 0.016
CLOSE_RAMP_TIME = 0.6
CLOSE_HOLD_TIME = 0.8
POST_CLOSE_HOLD = 0.25

# ===== Grasp robustness tuning =====
FRICTION_MULT = 3.0       # 2..6
FRICTION_SLIDE_CAP = 25.0 # clamp slide to avoid solver instability
SQUEEZE_EXTRA = 0.010     # extra close in joint units (tune 0.002..0.010)
GRIPPER_STRENGTH_MULT = 2.0  # 1..4

HOLD_AFTER_DONE_SECONDS = 2.0  # keep holding/attaching after pick to prevent end-slip

# Actuator names (match your XML)
ARM_ACT_NAMES = ["shoulder_pan", "shoulder_lift", "forearm", "wrist_1", "wrist_2", "wrist_3"]

# Finger collision geoms to disable during approach
FINGER_COL_GEOMS = ["ee_left_col", "ee_right_col"]

# Soft attach (optional)
ENABLE_SOFT_ATTACH = True
ATTACH_PINCH_DIST_THRESH = 0.030
ATTACH_MIN_CONTACTS = 1


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
    """Return list of geom ids belonging to a body (including unnamed geoms)."""
    bid = body_id(model, body_name)
    gadr = int(model.body_geomadr[bid])
    gnum = int(model.body_geomnum[bid])
    return list(range(gadr, gadr + gnum))


# =========================
# JSON LOADER (your uploaded schema)
# =========================
def load_battery_params(json_path: str):
    """
    Expected schema:
      relative_to_tcp: [x,y,z] mm (in TCP frame)
      grasp_parameters: pre_grip_mm, grip_point_mm, post_grip_mm  (mm, in world axes)
    Returns meters.
    """
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
# DEBUG WORLD
# =========================
def dbg_world(model, data):
    sid_tip = site_id(model, "tcp_tip")
    sid_L = site_id(model, "pinch_L")
    sid_R = site_id(model, "pinch_R")

    gid_padL = geom_id_strict(model, "ee_left_pad")
    gid_padR = geom_id_strict(model, "ee_right_pad")

    tcp_tip_w = data.site_xpos[sid_tip].copy()
    pinch_L_w = data.site_xpos[sid_L].copy()
    pinch_R_w = data.site_xpos[sid_R].copy()

    padL_w = data.geom_xpos[gid_padL].copy()
    padR_w = data.geom_xpos[gid_padR].copy()

    mid = 0.5 * (pinch_L_w + pinch_R_w)
    off = tcp_tip_w - mid

    dL = float(np.linalg.norm(tcp_tip_w - padL_w))
    dR = float(np.linalg.norm(tcp_tip_w - padR_w))

    print("----- DEBUG WORLD -----")
    print("tcp_tip   world:", tcp_tip_w)
    print("pinch_L   world:", pinch_L_w)
    print("pinch_R   world:", pinch_R_w)
    print("pinch_mid world:", mid)
    print("tcp_tip - mid  :", off)
    print("left_pad  world:", padL_w)
    print("right_pad world:", padR_w)
    print("dist tcp_tip->left_pad :", dL)
    print("dist tcp_tip->right_pad:", dR)
    print("-----------------------")

def print_contacts(model, data, max_lines=12):
    n = int(data.ncon)
    if n == 0:
        print("[CONTACTS] ncon=0")
        return
    print(f"[CONTACTS] ncon={n}")
    for i in range(min(n, max_lines)):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom{g1}"
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom{g2}"
        print(f"  {i:02d}: {n1}({g1}) <-> {n2}({g2})")
    if n > max_lines:
        print(f"  ... ({n-max_lines} more)")


# =========================
# FINGER COLLISION TOGGLE
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
# FRICTION + STRENGTH BOOSTERS (Python-only)
# =========================
def boost_friction(model, geom_names, mult=3.0, slide_cap=25.0):
    """
    Multiply geom friction (slide/torsion/roll). Clamp slide to avoid solver instability.
    """
    for name in geom_names:
        gid = geom_id(model, name)
        if gid == -1:
            continue
        fr = model.geom_friction[gid].copy()
        fr *= mult
        fr[0] = min(fr[0], slide_cap)
        model.geom_friction[gid] = fr
def boost_friction_ids(model, geom_ids, mult=3.0, slide_cap=25.0):
    """Same as boost_friction but uses numeric geom ids (works for unnamed geoms)."""
    for gid in geom_ids:
        if gid < 0 or gid >= model.ngeom:
            continue
        fr = model.geom_friction[gid].copy()
        fr *= mult
        fr[0] = min(fr[0], slide_cap)
        model.geom_friction[gid] = fr



def boost_gripper_strength(model, mult=2.0):
    """
    Increase effective actuator strength for position servos by scaling gear + forcerange.
    """
    for name in ["ee_gripper_left", "ee_gripper_right"]:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            continue
        model.actuator_gear[aid, :] *= mult
        model.actuator_forcerange[aid, 0] *= mult
        model.actuator_forcerange[aid, 1] *= mult


# =========================
# ARM IDS (STRICT)
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
# GRIPPER CONTROL
# =========================
def set_gripper_open(model, data):
    aL = actuator_id(model, "ee_gripper_left")
    aR = actuator_id(model, "ee_gripper_right")
    data.ctrl[aL] = float(OPEN_L_CMD)
    data.ctrl[aR] = float(OPEN_R_CMD)

def set_gripper_close_targets(model, data, closeL, closeR):
    aL = actuator_id(model, "ee_gripper_left")
    aR = actuator_id(model, "ee_gripper_right")
    data.ctrl[aL] = float(closeL)
    data.ctrl[aR] = float(closeR)

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

    d0 = pinch_distance(model, data)
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

    print("[AUTO-CAL] pinch distance d0:", d0)
    print("[AUTO-CAL] left  d(+eps):", dL_plus, " d(-eps):", dL_minus, " chosen dir:", dirL)
    print("[AUTO-CAL] right d(+eps):", dR_plus, " d(-eps):", dR_minus, " chosen dir:", dirR)
    print("[AUTO-CAL] close targets -> left:", closeL, " right:", closeR)
    return closeL, closeR


# =========================
# DESIRED ORIENTATION (FIXED YAW, VERTICAL TOOL)
# =========================
def desired_tcp_R_vertical():
    z = np.array([0.0, 0.0, 1.0])   # tool Z up
    y = np.array([0.0, 1.0, 0.0])   # fix yaw
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


# =========================
# JOINT-SPACE IK STEP (POSE)
# =========================
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


def gate_reach_pose(model, data, lock, site_name, target_pos, target_R,
                    arm_joint_ids, arm_act_ids, label,
                    pos_tol, rot_tol,
                    settle=SETTLE_TIME, timeout=GATE_TIMEOUT):
    t0 = time.time()
    stable_since = None
    last_pos_err = None
    last_rot_err = None
    last_p = None

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

            sid = site_id(model, site_name)
            last_p = data.site_xpos[sid].copy()

        ok = (last_pos_err < pos_tol) and (last_rot_err < rot_tol)
        if ok:
            if stable_since is None:
                stable_since = time.time()
            if (time.time() - stable_since) > settle:
                print(f"[GATE {label}] ok=True pos_err={last_pos_err:.6f} rot_err={last_rot_err:.6f}")
                return True, last_pos_err, last_rot_err
        else:
            stable_since = None

        if (time.time() - t0) > timeout:
            print(f"[GATE {label}] ok=False pos_err={last_pos_err:.6f} rot_err={last_rot_err:.6f}")
            print(f"[TCP] pos={last_p}  target={target_pos}")
            return False, last_pos_err, last_rot_err

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


def close_gripper_ramp_hold(model, data, lock, hold_site, hold_pos, hold_R,
                            arm_joint_ids, arm_act_ids, closeL, closeR):
    aL = actuator_id(model, "ee_gripper_left")
    aR = actuator_id(model, "ee_gripper_right")

    with lock:
        cL0 = float(data.ctrl[aL])
        cR0 = float(data.ctrl[aR])

    t0 = time.time()
    while (time.time() - t0) < CLOSE_RAMP_TIME:
        alpha = (time.time() - t0) / max(CLOSE_RAMP_TIME, 1e-6)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        cL = (1 - alpha) * cL0 + alpha * closeL
        cR = (1 - alpha) * cR0 + alpha * closeR

        with lock:
            data.ctrl[aL] = float(cL)
            data.ctrl[aR] = float(cR)

            mujoco.mj_forward(model, data)
            for _ in range(IK_ITERS_PER_CYCLE):
                _ = ik_step_pose_qpos(model, data, hold_site, hold_pos, hold_R, arm_joint_ids)
                mujoco.mj_forward(model, data)
            sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)

        time.sleep(DT_SLEEP)

    t1 = time.time()
    while (time.time() - t1) < CLOSE_HOLD_TIME:
        with lock:
            data.ctrl[aL] = float(closeL)
            data.ctrl[aR] = float(closeR)

            mujoco.mj_forward(model, data)
            for _ in range(IK_ITERS_PER_CYCLE):
                _ = ik_step_pose_qpos(model, data, hold_site, hold_pos, hold_R, arm_joint_ids)
                mujoco.mj_forward(model, data)
            sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)

        time.sleep(DT_SLEEP)

    with lock:
        jL = joint_id(model, "ee_gripper_left_joint")
        jR = joint_id(model, "ee_gripper_right_joint")
        qL = float(data.qpos[model.jnt_qposadr[jL]])
        qR = float(data.qpos[model.jnt_qposadr[jR]])
        print(f"[GRIPPER QPOS] left={qL:.6f} right={qR:.6f}")
        print(f"[PINCH DIST] {pinch_distance(model, data):.6f} m")


# =========================
# SOFT ATTACH HELPERS
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

def count_finger_battery_contacts(model, data, battery_geom_ids, finger_geom_ids):
    n = int(data.ncon)
    cnt = 0
    for i in range(n):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        if (g1 in finger_geom_ids and g2 in battery_geom_ids) or (g2 in finger_geom_ids and g1 in battery_geom_ids):
            cnt += 1
    return cnt

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
# LIVE TARGETS (battery may drift on table)
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

    # Finger & battery geom sets for contact detection
    finger_geom_ids = set()
    for gn in ["ee_left_col", "ee_right_col", "ee_left_pad", "ee_right_pad"]:
        gid = geom_id(model, gn)
        if gid != -1:
            finger_geom_ids.add(gid)

    bid_batt = body_id(model, "battery")
    gadr = model.body_geomadr[bid_batt]
    gnum = model.body_geomnum[bid_batt]
    battery_geom_ids = set(range(gadr, gadr + gnum))

    batt_qadr, batt_dofadr = get_freejoint_qadr(model, "battery")

    with lock:
        # Initial state
        pan = joint_id(model, "shoulder_pan_joint")
        arm_qpos_adr = model.jnt_qposadr[pan]
        data.qpos[arm_qpos_adr:arm_qpos_adr + 6] = SAFE_HOME_Q

        mujoco.mj_forward(model, data)
        sync_arm_ctrl_to_qpos(model, data, arm_act_ids)

        set_gripper_open(model, data)
        set_finger_collision(model, enable=False)

        # Friction + strength boosts
        boost_friction(model, ["ee_left_col", "ee_right_col"], mult=FRICTION_MULT, slide_cap=FRICTION_SLIDE_CAP)
        # battery collision geom in XML may be unnamed -> boost all battery body geoms
        boost_friction_ids(model, body_geom_ids(model, "battery"), mult=FRICTION_MULT, slide_cap=FRICTION_SLIDE_CAP)
        boost_gripper_strength(model, mult=GRIPPER_STRENGTH_MULT)

        mujoco.mj_forward(model, data)

        table_top_z = get_table_top_z(model, data)
        if table_top_z is not None:
            print(f"[INFO] Table top z = {table_top_z:.4f} m, clearance = {TABLE_CLEARANCE:.3f} m")
        else:
            print("[WARN] Could not detect table top. No Z clamp.")

        print("[OK] Initialized home, gripper open, finger collision OFF")
        dbg_world(model, data)

    time.sleep(T_HOME_SETTLE)

    # Initial targets
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

    ok_pre, *_ = gate_reach_pose(model, data, lock, SITE_TCP, tcp_pre, R_des,
                                 arm_joint_ids, arm_act_ids, label="PRE",
                                 pos_tol=POS_TOL_PRE, rot_tol=ROT_TOL)

    with lock:
        mujoco.mj_forward(model, data)
        dbg_world(model, data)

    print(f"[STEP] WAIT at PRE_GRIP for {WAIT_AT_PRE:.1f}s")
    t_wait = time.time()
    while time.time() - t_wait < WAIT_AT_PRE:
        with lock:
            mujoco.mj_forward(model, data)
            tcp_pre_live, _, _ = compute_tcp_targets_live(model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off)
            tcp_pre_live = clamp_above_table(tcp_pre_live, table_top_z)
            for _ in range(IK_ITERS_PER_CYCLE):
                _ = ik_step_pose_qpos(model, data, SITE_TCP, tcp_pre_live, R_des, arm_joint_ids)
                mujoco.mj_forward(model, data)
            sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
        time.sleep(DT_SLEEP)

    # GRIP (split approach to avoid penetration "pop")
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

    print("[STEP] Enable finger collision BEFORE final approach")
    with lock:
        set_finger_collision(model, enable=True)
        mujoco.mj_forward(model, data)

    print("[STEP] tcp_tip -> GRIP_POINT (collision ON, slow, LIVE target)")
    t0 = time.time()
    ok_grip = False
    while time.time() - t0 < 8.0:
        with lock:
            mujoco.mj_forward(model, data)
            _, tcp_grip_live, _ = compute_tcp_targets_live(model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off)
            tcp_grip_live = clamp_above_table(tcp_grip_live, table_top_z)
            p_cur = data.site_xpos[site_id(model, SITE_TCP)].copy()

        # short incremental move towards live grip
        p_step = p_cur + 0.25 * (tcp_grip_live - p_cur)
        move_interp_pose(model, data, lock, SITE_TCP, p_cur, p_step, R_des,
                         arm_joint_ids, arm_act_ids, 0.15, table_top_z)

        ok_grip, _, _ = gate_reach_pose(model, data, lock, SITE_TCP, tcp_grip_live, R_des,
                                        arm_joint_ids, arm_act_ids, label="GRIP",
                                        pos_tol=POS_TOL_GRIP, rot_tol=ROT_TOL)
        if ok_grip:
            break

    with lock:
        mujoco.mj_forward(model, data)
        dbg_world(model, data)

    if not ok_grip:
        print("[STOP] GRIP pose not reached closely enough -> not closing.")
        return

    # Close targets + extra squeeze (more normal force)
    with lock:
        mujoco.mj_forward(model, data)
        closeL, closeR = autocalibrate_close_targets(model, data)

        closeL -= SQUEEZE_EXTRA
        closeR += SQUEEZE_EXTRA

        jL = joint_id(model, "ee_gripper_left_joint")
        jR = joint_id(model, "ee_gripper_right_joint")
        rL = model.jnt_range[jL]
        rR = model.jnt_range[jR]
        closeL = float(np.clip(closeL, rL[0], rL[1]))
        closeR = float(np.clip(closeR, rR[0], rR[1]))
        print("[SQUEEZE] close targets -> left:", closeL, " right:", closeR)

        # refresh live grip target (in case object moved)
        _, tcp_grip, _ = compute_tcp_targets_live(model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off)
        tcp_grip = clamp_above_table(tcp_grip, table_top_z)

    print("[STEP] CLOSE gripper (ramp) while HOLDING GRIP pose")
    close_gripper_ramp_hold(model, data, lock, SITE_TCP, tcp_grip, R_des,
                            arm_joint_ids, arm_act_ids, closeL, closeR)

    with lock:
        mujoco.mj_forward(model, data)
        print_contacts(model, data, max_lines=25)

    print(f"[STEP] HOLD after close for {POST_CLOSE_HOLD:.2f}s (LIVE target)")
    t_hold = time.time()
    while time.time() - t_hold < POST_CLOSE_HOLD:
        with lock:
            mujoco.mj_forward(model, data)
            _, tcp_grip_live, _ = compute_tcp_targets_live(model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off)
            tcp_grip_live = clamp_above_table(tcp_grip_live, table_top_z)

            set_gripper_close_targets(model, data, closeL, closeR)
            for _ in range(IK_ITERS_PER_CYCLE):
                _ = ik_step_pose_qpos(model, data, SITE_TCP, tcp_grip_live, R_des, arm_joint_ids)
                mujoco.mj_forward(model, data)
            sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
        time.sleep(DT_SLEEP)

    # ========= PATCH: Freeze post-grip target after grasp =========
    # This prevents "runaway" where post target is recomputed from an object pose
    # that you are also driving via soft-attach.
    with lock:
        mujoco.mj_forward(model, data)
        batt_pos_ref = data.xpos[bid_batt].copy()  # freeze reference
    obj_post_ref = batt_pos_ref + post_off
    tcp_post_frozen = clamp_above_table(obj_post_ref - rel_tcp_world, table_top_z)

    # Soft attach decision AFTER settle
    attached = False
    tcp_to_batt_pos = None
    tcp_to_batt_quat = None

    if ENABLE_SOFT_ATTACH:
        with lock:
            mujoco.mj_forward(model, data)
            pd = pinch_distance(model, data)
            cnum = count_finger_battery_contacts(model, data, battery_geom_ids, finger_geom_ids)
            print(f"[ATTACH CHECK AFTER HOLD] pinch_dist={pd:.6f}  finger-battery-contacts={cnum}")

            if pd < ATTACH_PINCH_DIST_THRESH and cnum >= ATTACH_MIN_CONTACTS:
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
                print("[ATTACH] Soft-attached battery to tcp_tip for lift.")
            else:
                print("[ATTACH] Not attaching (insufficient squeeze/contact).")

    # Lift (POST) using FROZEN target to avoid going above the planned post_grip
    print("[STEP] tcp_tip -> POST_GRIP (vertical lift, FROZEN target)")
    with lock:
        mujoco.mj_forward(model, data)
        p2 = data.site_xpos[site_id(model, SITE_TCP)].copy()

    t0 = time.time()
    while True:
        t = (time.time() - t0) / max(T_TO_POST, 1e-6)
        t = float(np.clip(t, 0.0, 1.0))
        s = t * t * (3.0 - 2.0 * t)

        p = (1 - s) * p2 + s * tcp_post_frozen
        p = clamp_above_table(p, table_top_z)

        with lock:
            mujoco.mj_forward(model, data)
            for _ in range(IK_ITERS_PER_CYCLE):
                _ = ik_step_pose_qpos(model, data, SITE_TCP, p, R_des, arm_joint_ids)
                mujoco.mj_forward(model, data)
            sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
            set_gripper_close_targets(model, data, closeL, closeR)

            # If attached, enforce follow
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

            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)

        if t >= 1.0:
            break
        time.sleep(DT_SLEEP)

    # Gate to frozen post
    ok_post, *_ = gate_reach_pose(model, data, lock, SITE_TCP, tcp_post_frozen, R_des,
                                  arm_joint_ids, arm_act_ids, label="POST",
                                  pos_tol=POS_TOL_PRE, rot_tol=ROT_TOL)

    with lock:
        mujoco.mj_forward(model, data)
        dbg_world(model, data)
        print("[BATTERY Z]", float(data.xpos[bid_batt][2]))
        print_contacts(model, data, max_lines=25)

    print("[DONE] Pick sequence finished.")


    # Keep holding after sequence to prevent the small end-slip that can happen
    # when the control thread exits but physics keeps running in the viewer loop.
    if HOLD_AFTER_DONE_SECONDS and HOLD_AFTER_DONE_SECONDS > 0:
        print(f"[HOLD] Maintaining grasp for {HOLD_AFTER_DONE_SECONDS:.2f}s to prevent settling slip.")
        t_hold = time.time()
        while time.time() - t_hold < HOLD_AFTER_DONE_SECONDS:
            with lock:
                # keep arm ctrl synced (hold pose) and keep gripper closed
                sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
                set_gripper_close_targets(model, data, closeL, closeR)

                # keep soft-attach active if it was engaged
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
