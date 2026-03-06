import time
import json
import threading
import numpy as np
import mujoco
import mujoco.viewer

# =========================
# FILES
# =========================
XML_PATH = "scene_friction.xml"
JSON_PATH = "battery_grip_data.json"

# =========================
# USER SETTINGS
# =========================
SAFE_HOME_Q = np.array([0.88, -2.01, 1.88, -1.51, -1.51, -0.628], dtype=float)
SITE_TCP = "tcp_tip"

# Timing
T_HOME_SETTLE = 0.6
T_TO_PRE  = 2.0
WAIT_AT_PRE = 2.0
T_TO_GRIP = 2.0
T_TO_POST = 2.5

DT_SLEEP = 0.004
SUBSTEPS = 4

# Gate tolerances
POS_TOL_PRE  = 0.0010
POS_TOL_GRIP = 0.0006
ROT_TOL      = 0.04
SETTLE_TIME  = 0.20
GATE_TIMEOUT = 12.0

# IK tuning
IK_ITERS_PER_CYCLE = 8
IK_DAMP = 0.02
IK_STEP = 0.55

# Gripper closing profile
CLOSE_MAG = 0.016          # base close motion magnitude (joint units)
CLOSE_RAMP_TIME = 0.6
CLOSE_HOLD_TIME = 0.8
POST_CLOSE_HOLD = 0.25

# Gripper open limit (battery width + margin)
BATTERY_HALF_WIDTH = 0.00725
BATTERY_DIAMETER = 2.0 * BATTERY_HALF_WIDTH
MAX_GRIPPER_OPENING = BATTERY_DIAMETER + 0.003      # meters
MAX_FINGER_OPEN_CMD = MAX_GRIPPER_OPENING / 2.0     # meters per finger

OPEN_L_CMD = float(MAX_FINGER_OPEN_CMD)
OPEN_R_CMD = float(-MAX_FINGER_OPEN_CMD)

# Actuator names (must match XML)
ARM_ACT_NAMES = ["shoulder_pan", "shoulder_lift", "forearm", "wrist_1", "wrist_2", "wrist_3"]

# Finger collision geoms to disable during approach (optional, but helps avoid “pop-in”)
FINGER_COL_GEOMS = ["ee_left_col", "ee_right_col"]


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


# =========================
# JSON LOADER
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
# GRIPPER CONTROL
# =========================
def set_gripper_open(model, data):
    aL = actuator_id(model, "ee_gripper_left")
    aR = actuator_id(model, "ee_gripper_right")

    maxL = min(MAX_FINGER_OPEN_CMD, float(model.actuator_ctrlrange[aL, 1]))
    minR = max(-MAX_FINGER_OPEN_CMD, float(model.actuator_ctrlrange[aR, 0]))

    data.ctrl[aL] = float(np.clip(OPEN_L_CMD, float(model.actuator_ctrlrange[aL, 0]), maxL))
    data.ctrl[aR] = float(np.clip(OPEN_R_CMD, minR, float(model.actuator_ctrlrange[aR, 1])))

def set_gripper_ctrl(model, data, left, right):
    aL = actuator_id(model, "ee_gripper_left")
    aR = actuator_id(model, "ee_gripper_right")
    data.ctrl[aL] = float(left)
    data.ctrl[aR] = float(right)

def pinch_distance(model, data):
    sL = site_id(model, "pinch_L")
    sR = site_id(model, "pinch_R")
    pL = data.site_xpos[sL].copy()
    pR = data.site_xpos[sR].copy()
    return float(np.linalg.norm(pL - pR))

def autocalibrate_close_targets(model, data):
    """
    Uses a small finite-difference check to decide which direction closes the pinch.
    Then closes by CLOSE_MAG in joint space (clamped to joint range).
    """
    jL = joint_id(model, "ee_gripper_left_joint")
    jR = joint_id(model, "ee_gripper_right_joint")
    qadrL = model.jnt_qposadr[jL]
    qadrR = model.jnt_qposadr[jR]

    qL0 = float(data.qpos[qadrL])
    qR0 = float(data.qpos[qadrR])

    d0 = pinch_distance(model, data)
    eps = 0.001

    # left direction
    data.qpos[qadrL] = qL0 + eps
    mujoco.mj_forward(model, data)
    dL_plus = pinch_distance(model, data)

    data.qpos[qadrL] = qL0 - eps
    mujoco.mj_forward(model, data)
    dL_minus = pinch_distance(model, data)

    data.qpos[qadrL] = qL0
    mujoco.mj_forward(model, data)

    # right direction
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

    print("[AUTO-CAL] pinch d0:", d0)
    print("[AUTO-CAL] close targets -> left:", closeL, " right:", closeR)
    return closeL, closeR


# =========================
# SUCTION / MAGNET CONTROL
# =========================
def suction_ids(model):
    aL = actuator_id(model, "magnet_left")
    aR = actuator_id(model, "magnet_right")
    return aL, aR

def set_suction(model, data, on: bool, strength: float = 1.0):
    """
    strength in [0,1] because ctrlrange="0 1"
    """
    aL, aR = suction_ids(model)
    val = float(np.clip(strength, 0.0, 1.0)) if on else 0.0
    data.ctrl[aL] = val
    data.ctrl[aR] = val

def battery_in_contact(model, data) -> bool:
    """
    Returns True if any contact involves the battery body.
    """
    bid = body_id(model, "battery")
    for i in range(data.ncon):
        con = data.contact[i]
        b1 = model.geom_bodyid[con.geom1]
        b2 = model.geom_bodyid[con.geom2]
        if b1 == bid or b2 == bid:
            return True
    return False


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
# DESIRED ORIENTATION (VERTICAL TOOL)
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
                     arm_joint_ids, arm_act_ids, duration):
    t0 = time.time()
    while True:
        t = (time.time() - t0) / max(duration, 1e-6)
        t = float(np.clip(t, 0.0, 1.0))
        s = t * t * (3.0 - 2.0 * t)

        p = (1 - s) * p_start + s * p_goal

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

    # ramp
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

    # hold
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
        print(f"[PINCH DIST AFTER CLOSE] {pinch_distance(model, data):.6f} m")


# =========================
# LIVE TARGETS (battery can drift)
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

    arm_act_ids = get_arm_actuator_ids_strict(model)
    arm_joint_ids = get_arm_joint_ids_from_actuators(model, arm_act_ids)

    bid_batt = body_id(model, "battery")

    with lock:
        # set arm to home
        pan = joint_id(model, "shoulder_pan_joint")
        adr = model.jnt_qposadr[pan]
        data.qpos[adr:adr + 6] = SAFE_HOME_Q

        mujoco.mj_forward(model, data)
        sync_arm_ctrl_to_qpos(model, data, arm_act_ids)

        set_gripper_open(model, data)

        # suction OFF at start
        set_suction(model, data, on=False)

        # optional: avoid early collisions during approach
        set_finger_collision(model, enable=False)

        mujoco.mj_forward(model, data)

    time.sleep(T_HOME_SETTLE)

    # compute initial targets
    with lock:
        mujoco.mj_forward(model, data)
        tcp_pre, tcp_grip, tcp_post = compute_tcp_targets_live(model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off)
        p0 = data.site_xpos[site_id(model, SITE_TCP)].copy()

    # PRE
    print("[STEP] Move -> PRE")
    move_interp_pose(model, data, lock, SITE_TCP, p0, tcp_pre, R_des, arm_joint_ids, arm_act_ids, T_TO_PRE)
    if not gate_reach_pose(model, data, lock, SITE_TCP, tcp_pre, R_des, arm_joint_ids, arm_act_ids,
                           label="PRE", pos_tol=POS_TOL_PRE, rot_tol=ROT_TOL):
        return

    print(f"[STEP] Wait at PRE for {WAIT_AT_PRE:.1f}s")
    t_wait = time.time()
    while time.time() - t_wait < WAIT_AT_PRE:
        with lock:
            mujoco.mj_forward(model, data)
            tcp_pre_live, _, _ = compute_tcp_targets_live(model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off)
            for _ in range(IK_ITERS_PER_CYCLE):
                _ = ik_step_pose_qpos(model, data, SITE_TCP, tcp_pre_live, R_des, arm_joint_ids)
                mujoco.mj_forward(model, data)
            sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
        time.sleep(DT_SLEEP)

    # GRIP (split approach)
    with lock:
        mujoco.mj_forward(model, data)
        _, tcp_grip, _ = compute_tcp_targets_live(model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off)

    tcp_near = tcp_grip.copy()
    tcp_near[2] += 0.015

    print("[STEP] Move -> NEAR_GRIP (collision OFF)")
    with lock:
        mujoco.mj_forward(model, data)
        p1 = data.site_xpos[site_id(model, SITE_TCP)].copy()
    move_interp_pose(model, data, lock, SITE_TCP, p1, tcp_near, R_des, arm_joint_ids, arm_act_ids, T_TO_GRIP)

    print("[STEP] Enable finger collision for final approach")
    with lock:
        set_finger_collision(model, enable=True)
        mujoco.mj_forward(model, data)

    print("[STEP] Move -> GRIP (collision ON, live target)")
    t0 = time.time()
    ok_grip = False
    while time.time() - t0 < 8.0:
        with lock:
            mujoco.mj_forward(model, data)
            _, tcp_grip_live, _ = compute_tcp_targets_live(model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off)

        ok_grip = gate_reach_pose(model, data, lock, SITE_TCP, tcp_grip_live, R_des,
                                 arm_joint_ids, arm_act_ids, label="GRIP",
                                 pos_tol=POS_TOL_GRIP, rot_tol=ROT_TOL)
        if ok_grip:
            break

    if not ok_grip:
        print("[STOP] Could not reach GRIP precisely enough.")
        return

    # CLOSE
    with lock:
        mujoco.mj_forward(model, data)
        closeL, closeR = autocalibrate_close_targets(model, data)
        _, tcp_grip_hold, _ = compute_tcp_targets_live(model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off)

    print("[STEP] Close gripper while holding GRIP pose")
    close_gripper_ramp_hold(model, data, lock, SITE_TCP, tcp_grip_hold, R_des,
                            arm_joint_ids, arm_act_ids, closeL, closeR)

    # SUCTION ON (after close, only if contact)
    with lock:
        mujoco.mj_forward(model, data)
        if battery_in_contact(model, data):
            print("[SUCTION] Battery contact detected -> suction ON")
            set_suction(model, data, on=True, strength=1.0)
        else:
            print("[SUCTION] No contact detected -> suction stays OFF")
            set_suction(model, data, on=False)

    print(f"[STEP] Hold after close for {POST_CLOSE_HOLD:.2f}s")
    t_hold = time.time()
    while time.time() - t_hold < POST_CLOSE_HOLD:
        with lock:
            mujoco.mj_forward(model, data)
            _, tcp_grip_live, _ = compute_tcp_targets_live(model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off)

            set_gripper_ctrl(model, data, closeL, closeR)
            for _ in range(IK_ITERS_PER_CYCLE):
                _ = ik_step_pose_qpos(model, data, SITE_TCP, tcp_grip_live, R_des, arm_joint_ids)
                mujoco.mj_forward(model, data)
            sync_arm_ctrl_to_qpos(model, data, arm_act_ids)
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
        time.sleep(DT_SLEEP)

    # POST (lift)
    print("[STEP] Lift -> POST")
    with lock:
        mujoco.mj_forward(model, data)
        p2 = data.site_xpos[site_id(model, SITE_TCP)].copy()
        _, _, tcp_post = compute_tcp_targets_live(model, data, bid_batt, R_des, rel_tcp, pre_off, grip_off, post_off)

    move_interp_pose(model, data, lock, SITE_TCP, p2, tcp_post, R_des, arm_joint_ids, arm_act_ids, T_TO_POST)
    gate_reach_pose(model, data, lock, SITE_TCP, tcp_post, R_des, arm_joint_ids, arm_act_ids,
                    label="POST", pos_tol=POS_TOL_PRE, rot_tol=ROT_TOL)

    print("[DONE] Pick sequence finished (friction + suction).")


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