import time
import json
import threading
import numpy as np
import mujoco
import mujoco.viewer

# =========================
# FILES
# =========================
XML_PATH  = "scene.xml"
JSON_PATH = "battery_params.json"

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
T_TO_POST = 2.8

DT_SLEEP = 0.004
SUBSTEPS = 4

# Table safety:
# - Apply clamp ONLY for PRE and POST.
# - Do NOT clamp GRIP or you will grip above the target.
TABLE_CLEARANCE_PREPOST = 0.002  # 2mm
# Small "push down" bias at grip to ensure contact (tune: 0..3mm)
GRIP_BIAS_Z = -0.0015  # -1.5mm

# Gate tolerances
POS_TOL = 0.0006     # 0.6 mm
ROT_TOL = 0.05       # rad (~2.9 deg)
SETTLE_TIME = 0.12
GATE_TIMEOUT = 10.0

# IK tuning (stable)
IK_ITERS_PER_CYCLE = 5
IK_DAMP = 0.02       # increase if jitter
IK_STEP = 0.45       # decrease if jitter

# Gripper open commands (actuator ctrl)
OPEN_L_CMD = 0.012
OPEN_R_CMD = -0.012

# Closing
CLOSE_MAG_START = 0.016     # initial attempt
CLOSE_MAG_MAX   = 0.030     # allow more if needed
CLOSE_MAG_STEP  = 0.004     # increase if grasp is weak
CLOSE_RAMP_TIME = 0.8
CLOSE_HOLD_TIME = 0.9
POST_CLOSE_HOLD = 0.35

# Lift behavior (important for no-adhesion)
LIFT_SLOW_TIME = 0.6   # first part of lift is slow
LIFT_SLOW_ALPHA = 0.25 # slow fraction of lift

# Actuator names (match your XML)
ARM_ACT_NAMES = ["shoulder_pan", "shoulder_lift", "forearm", "wrist_1", "wrist_2", "wrist_3"]

# Finger collision geoms to disable during approach (your names)
FINGER_COL_GEOMS = ["ee_left_col", "ee_right_col", "ee_left_pad", "ee_right_pad"]

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

def clamp_above_table_prepost(p, table_top_z):
    if table_top_z is None or TABLE_CLEARANCE_PREPOST <= 0:
        return p
    q = p.copy()
    q[2] = max(q[2], table_top_z + TABLE_CLEARANCE_PREPOST)
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

def print_contacts(model, data, max_lines=20):
    n = int(data.ncon)
    if n == 0:
        print("[CONTACTS] ncon=0")
        return
    print(f"[CONTACTS] ncon={n}")
    for i in range(min(n, max_lines)):
        c = data.contact[i]
        g1 = int(c.geom1); g2 = int(c.geom2)
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
# GRIPPER
# =========================
def set_gripper_open(model, data):
    aL = actuator_id(model, "ee_gripper_left")
    aR = actuator_id(model, "ee_gripper_right")
    data.ctrl[aL] = float(OPEN_L_CMD)
    data.ctrl[aR] = float(OPEN_R_CMD)

def set_gripper_targets(model, data, closeL, closeR):
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

def autocal_close_targets(model, data, close_mag):
    jL = joint_id(model, "ee_gripper_left_joint")
    jR = joint_id(model, "ee_gripper_right_joint")
    qadrL = model.jnt_qposadr[jL]
    qadrR = model.jnt_qposadr[jR]
    qL0 = float(data.qpos[qadrL])
    qR0 = float(data.qpos[qadrR])

    d0 = pinch_distance(model, data)
    eps = 0.001

    # left + / -
    data.qpos[qadrL] = qL0 + eps; mujoco.mj_forward(model, data); dLp = pinch_distance(model, data)
    data.qpos[qadrL] = qL0 - eps; mujoco.mj_forward(model, data); dLm = pinch_distance(model, data)
    data.qpos[qadrL] = qL0;       mujoco.mj_forward(model, data)

    # right + / -
    data.qpos[qadrR] = qR0 + eps; mujoco.mj_forward(model, data); dRp = pinch_distance(model, data)
    data.qpos[qadrR] = qR0 - eps; mujoco.mj_forward(model, data); dRm = pinch_distance(model, data)
    data.qpos[qadrR] = qR0;       mujoco.mj_forward(model, data)

    dirL = +1.0 if dLp < dLm else -1.0
    dirR = +1.0 if dRp < dRm else -1.0

    closeL = qL0 + dirL * close_mag
    closeR = qR0 + dirR * close_mag

    rL = model.jnt_range[jL]; rR = model.jnt_range[jR]
    closeL = float(np.clip(closeL, rL[0], rL[1]))
    closeR = float(np.clip(closeR, rR[0], rR[1]))

    print(f"[AUTO-CAL] d0={d0:.6f}  mag={close_mag:.3f}  dirL={dirL:+.0f} dirR={dirR:+.0f}  -> closeL={closeL:.6f} closeR={closeR:.6f}")
    return closeL, closeR

# =========================
# VERTICAL ORIENTATION (RIGHT-HANDED, NO FLIPS)
# tool Z axis = world Z (down)
# tool Y axis = world +Y (keeps pinch direction stable)
# =========================
def desired_tcp_R_vertical():
    z = np.array([0.0, 0.0, 1.0])  # DOWN
    y = np.array([0.0, 1.0,  0.0])  # keep pinch along +Y
    x = np.cross(y, z)             # right-handed
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
# IK STEP (POSE) applied to qpos, then we sync ctrl to qpos
# =========================
def ik_step_pose_qpos(model, data, site_name, target_pos, target_R, arm_joint_ids):
    sid = site_id(model, site_name)

    p = data.site_xpos[sid].copy()
    M = data.site_xmat[sid].reshape(3, 3).copy()

    e_p = target_pos - p
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
    A = J @ J.T + IK_DAMP * np.eye(6)
    dq = J.T @ np.linalg.solve(A, e)
    dq = np.clip(dq, -0.10, 0.10)  # prevent spikes
    dq *= IK_STEP

    for k, jid in enumerate(arm_joint_ids):
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr] = float(data.qpos[qadr] + dq[k])
        r = model.jnt_range[jid]
        data.qpos[qadr] = float(np.clip(data.qpos[qadr], r[0], r[1]))

    return pos_err, rot_err

def gate_pose(model, data, lock, site_name, target_pos, target_R,
              arm_joint_ids, arm_act_ids, label):
    t0 = time.time()
    stable = None
    last_pe = 1e9
    last_re = 1e9

    while True:
        with lock:
            mujoco.mj_forward(model, data)
            for _ in range(IK_ITERS_PER_CYCLE):
                last_pe, last_re = ik_step_pose_qpos(model, data, site_name, target_pos, target_R, arm_joint_ids)
                mujoco.mj_forward(model, data)

            sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)

        ok = (last_pe < POS_TOL) and (last_re < ROT_TOL)
        if ok:
            if stable is None:
                stable = time.time()
            if (time.time() - stable) > SETTLE_TIME:
                print(f"[GATE {label}] ok=True pos_err={last_pe:.6f} rot_err={last_re:.6f}")
                return True, last_pe, last_re
        else:
            stable = None

        if (time.time() - t0) > GATE_TIMEOUT:
            print(f"[GATE {label}] ok=False pos_err={last_pe:.6f} rot_err={last_re:.6f}")
            return False, last_pe, last_re

        time.sleep(DT_SLEEP)

def move_interp_pose(model, data, lock, site_name, p_start, p_goal, R_des,
                     arm_joint_ids, arm_act_ids, duration, clamp_mode, table_top_z):
    """
    clamp_mode: "prepost" or "none"
    """
    t0 = time.time()
    while True:
        t = (time.time() - t0) / max(duration, 1e-6)
        t = float(np.clip(t, 0.0, 1.0))
        s = t * t * (3.0 - 2.0 * t)

        p = (1 - s) * p_start + s * p_goal
        if clamp_mode == "prepost":
            p = clamp_above_table_prepost(p, table_top_z)

        with lock:
            mujoco.mj_forward(model, data)
            for _ in range(IK_ITERS_PER_CYCLE):
                _ = ik_step_pose_qpos(model, data, site_name, p, R_des, arm_joint_ids)
                mujoco.mj_forward(model, data)
            sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)

        if t >= 1.0:
            break
        time.sleep(DT_SLEEP)

def contacts_with_battery(model, data):
    """
    Returns (left_touch, right_touch) based on ee_left_col/ee_right_col touching battery geom(s).
    """
    left = False
    right = False
    # battery body geoms: detect by body name "battery"
    b_id = body_id(model, "battery")
    battery_geom_ids = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == b_id:
            battery_geom_ids.append(gid)

    gL = geom_id(model, "ee_left_col")
    gR = geom_id(model, "ee_right_col")

    for i in range(int(data.ncon)):
        c = data.contact[i]
        a = int(c.geom1); b = int(c.geom2)
        if a in battery_geom_ids or b in battery_geom_ids:
            other = b if a in battery_geom_ids else a
            if other == gL:
                left = True
            if other == gR:
                right = True
    return left, right

def close_gripper_progressive(model, data, lock, hold_pos, hold_R,
                              arm_joint_ids, arm_act_ids):
    """
    Close ramp -> check contacts -> if not holding both sides, increase close magnitude.
    """
    close_mag = CLOSE_MAG_START
    closeL = closeR = 0.0

    while close_mag <= CLOSE_MAG_MAX + 1e-9:
        with lock:
            mujoco.mj_forward(model, data)
            closeL, closeR = autocal_close_targets(model, data, close_mag)

        # Ramp to targets while holding pose
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
                    _ = ik_step_pose_qpos(model, data, SITE_TCP, hold_pos, hold_R, arm_joint_ids)
                    mujoco.mj_forward(model, data)

                sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
                for _ in range(SUBSTEPS):
                    mujoco.mj_step(model, data)

            time.sleep(DT_SLEEP)

        # Hold closed briefly
        t1 = time.time()
        while (time.time() - t1) < CLOSE_HOLD_TIME:
            with lock:
                set_gripper_targets(model, data, closeL, closeR)
                mujoco.mj_forward(model, data)
                for _ in range(IK_ITERS_PER_CYCLE):
                    _ = ik_step_pose_qpos(model, data, SITE_TCP, hold_pos, hold_R, arm_joint_ids)
                    mujoco.mj_forward(model, data)
                sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
                for _ in range(SUBSTEPS):
                    mujoco.mj_step(model, data)
            time.sleep(DT_SLEEP)

        with lock:
            mujoco.mj_forward(model, data)
            Lc, Rc = contacts_with_battery(model, data)
            jL = joint_id(model, "ee_gripper_left_joint")
            jR = joint_id(model, "ee_gripper_right_joint")
            qL = float(data.qpos[model.jnt_qposadr[jL]])
            qR = float(data.qpos[model.jnt_qposadr[jR]])
            print(f"[GRIP] mag={close_mag:.3f}  qL={qL:.6f} qR={qR:.6f}  touchL={Lc} touchR={Rc}  pinch={pinch_distance(model,data):.6f}")

        if Lc and Rc:
            return closeL, closeR, True

        close_mag += CLOSE_MAG_STEP

    return closeL, closeR, False

# =========================
# MAIN SEQUENCE
# =========================
def run_sequence(model, data, lock):
    rel_tcp, pre_off, grip_off, post_off = load_battery_params(JSON_PATH)

    arm_act_ids = get_arm_actuator_ids_strict(model)
    arm_joint_ids = get_arm_joint_ids_from_actuators(model, arm_act_ids)
    R_des = desired_tcp_R_vertical()

    with lock:
        # set arm to home (qpos)
        pan = joint_id(model, "shoulder_pan_joint")
        arm_qpos_adr = model.jnt_qposadr[pan]
        data.qpos[arm_qpos_adr:arm_qpos_adr + 6] = SAFE_HOME_Q
        mujoco.mj_forward(model, data)

        sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
        set_gripper_open(model, data)

        # collisions off for approach
        set_finger_collision(model, enable=False)
        mujoco.mj_forward(model, data)

        table_top_z = get_table_top_z(model, data)
        if table_top_z is not None:
            print(f"[INFO] Table top z = {table_top_z:.4f} m (PRE/POST clamp {TABLE_CLEARANCE_PREPOST:.3f} m)")
        else:
            print("[WARN] Could not detect table top.")

        print("[OK] Home, gripper open, finger collision OFF")
        dbg_world(model, data)

    time.sleep(T_HOME_SETTLE)

    # battery world pos
    bid_batt = body_id(model, "battery")
    with lock:
        mujoco.mj_forward(model, data)
        batt_pos = data.xpos[bid_batt].copy()

    # object targets in world
    obj_pre  = batt_pos + pre_off
    obj_grip = batt_pos + grip_off
    obj_post = batt_pos + post_off

    # tcp targets
    tcp_pre  = obj_pre  - rel_tcp
    tcp_grip = obj_grip - rel_tcp
    tcp_post = obj_post - rel_tcp

    # apply clamp only to PRE/POST
    tcp_pre  = clamp_above_table_prepost(tcp_pre, table_top_z)
    tcp_post = clamp_above_table_prepost(tcp_post, table_top_z)

    # IMPORTANT: do NOT clamp grip; instead apply small bias downward
    tcp_grip = tcp_grip.copy()
    tcp_grip[2] += GRIP_BIAS_Z

    with lock:
        mujoco.mj_forward(model, data)
        p0 = data.site_xpos[site_id(model, SITE_TCP)].copy()

    # PRE
    print("[STEP] tcp_tip -> PRE_GRIP (vertical, clamp ON)")
    move_interp_pose(model, data, lock, SITE_TCP, p0, tcp_pre, R_des,
                     arm_joint_ids, arm_act_ids, T_TO_PRE, clamp_mode="prepost", table_top_z=table_top_z)
    gate_pose(model, data, lock, SITE_TCP, tcp_pre, R_des, arm_joint_ids, arm_act_ids, "PRE")

    with lock:
        mujoco.mj_forward(model, data)
        dbg_world(model, data)

    # WAIT
    print(f"[STEP] WAIT at PRE_GRIP for {WAIT_AT_PRE:.1f}s")
    t_wait = time.time()
    while time.time() - t_wait < WAIT_AT_PRE:
        with lock:
            mujoco.mj_forward(model, data)
            for _ in range(IK_ITERS_PER_CYCLE):
                _ = ik_step_pose_qpos(model, data, SITE_TCP, tcp_pre, R_des, arm_joint_ids)
                mujoco.mj_forward(model, data)
            sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
        time.sleep(DT_SLEEP)

    # GRIP (no clamp)
    print("[STEP] tcp_tip -> GRIP_POINT (vertical, clamp OFF, collision OFF)")
    with lock:
        mujoco.mj_forward(model, data)
        p1 = data.site_xpos[site_id(model, SITE_TCP)].copy()

    move_interp_pose(model, data, lock, SITE_TCP, p1, tcp_grip, R_des,
                     arm_joint_ids, arm_act_ids, T_TO_GRIP, clamp_mode="none", table_top_z=table_top_z)
    ok_grip, pe, re = gate_pose(model, data, lock, SITE_TCP, tcp_grip, R_des, arm_joint_ids, arm_act_ids, "GRIP")

    with lock:
        mujoco.mj_forward(model, data)
        dbg_world(model, data)

    if not ok_grip:
        print("[STOP] Could not reach GRIP pose tightly -> not closing (prevents glitch).")
        return

    # Enable finger collisions only now
    print("[STEP] Enable finger collision (contact allowed)")
    with lock:
        set_finger_collision(model, enable=True)
        mujoco.mj_forward(model, data)

    # Close progressively until BOTH fingers touch battery
    print("[STEP] CLOSE gripper progressively until both fingers contact battery")
    closeL, closeR, ok_touch = close_gripper_progressive(model, data, lock, tcp_grip, R_des,
                                                         arm_joint_ids, arm_act_ids)

    with lock:
        mujoco.mj_forward(model, data)
        print_contacts(model, data, max_lines=20)

    if not ok_touch:
        print("[WARN] Did not achieve two-sided contact. Lift may fail (no adhesion). Proceeding anyway...")

    # Hold after close
    print(f"[STEP] HOLD after close for {POST_CLOSE_HOLD:.2f}s")
    t_hold = time.time()
    while time.time() - t_hold < POST_CLOSE_HOLD:
        with lock:
            set_gripper_targets(model, data, closeL, closeR)
            mujoco.mj_forward(model, data)
            for _ in range(IK_ITERS_PER_CYCLE):
                _ = ik_step_pose_qpos(model, data, SITE_TCP, tcp_grip, R_des, arm_joint_ids)
                mujoco.mj_forward(model, data)
            sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
        time.sleep(DT_SLEEP)

    # POST lift (keep squeezing during lift)
    print("[STEP] tcp_tip -> POST_GRIP (vertical lift, keep closing)")
    with lock:
        mujoco.mj_forward(model, data)
        p2 = data.site_xpos[site_id(model, SITE_TCP)].copy()

    # two-phase lift: slow first, then normal
    tcp_post_slow = p2 + (tcp_post - p2) * LIFT_SLOW_ALPHA

    move_interp_pose(model, data, lock, SITE_TCP, p2, tcp_post_slow, R_des,
                     arm_joint_ids, arm_act_ids, LIFT_SLOW_TIME, clamp_mode="prepost", table_top_z=table_top_z)
    with lock:
        # keep close while transitioning
        set_gripper_targets(model, data, closeL, closeR)

    move_interp_pose(model, data, lock, SITE_TCP, tcp_post_slow, tcp_post, R_des,
                     arm_joint_ids, arm_act_ids, max(T_TO_POST - LIFT_SLOW_TIME, 0.5),
                     clamp_mode="prepost", table_top_z=table_top_z)

    gate_pose(model, data, lock, SITE_TCP, tcp_post, R_des, arm_joint_ids, arm_act_ids, "POST")

    with lock:
        mujoco.mj_forward(model, data)
        dbg_world(model, data)

    # Keep holding close at post (helps prevent slip)
    t_post_hold = time.time()
    while time.time() - t_post_hold < 0.6:
        with lock:
            set_gripper_targets(model, data, closeL, closeR)
            mujoco.mj_forward(model, data)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
        time.sleep(DT_SLEEP)

    print("[DONE] Pick sequence finished.")

# =========================
# ENTRYPOINT
# =========================
def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    lock  = threading.Lock()

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
