import mujoco
import mujoco.viewer
import time
import numpy as np

# --- CONFIGURATION ---
XML_FILE = "scene.xml"

# NEW RANGES:
# Left:  -0.01 to 0.04
# Right: -0.04 to 0.01

# fully OPEN (Wide gap)
CMD_OPEN  = [0.04, -0.04] 

# fully CLOSED (Squeeze past center)
# We command them to cross the center line to ensure a tight seal
CMD_CLOSE = [-0.01, 0.01] 

def main():
    print(f"Loading {XML_FILE}...")
    model = mujoco.MjModel.from_xml_path(XML_FILE)
    data = mujoco.MjData(model)

    try:
        left_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'ee_gripper_left')
        right_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'ee_gripper_right')
        left_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'ee_gripper_left_joint')
        right_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'ee_gripper_right_joint')
    except Exception as e:
        print(f"Error finding grippers: {e}")
        return

    # Initialize
    mujoco.mj_resetData(model, data) 
    mujoco.mj_forward(model, data)

    print("\n------------------------------------------------")
    print(" GRIPPER CALIBRATION TEST")
    print("------------------------------------------------")
    print(" Cycle: SQUEEZE (-0.01/0.01) <--> OPEN (0.04/-0.04)")
    print("------------------------------------------------\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            now = time.time() - start_time
            cycle = now % 4.0
            
            if cycle < 2.0:
                # CLOSE (Squeeze)
                target_vals = CMD_CLOSE
                status = ">>> SQUEEZING <<<"
            else:
                # OPEN
                target_vals = CMD_OPEN
                status = "<   OPENING   >"

            # Apply Control
            data.ctrl[left_act] = target_vals[0]
            data.ctrl[right_act] = target_vals[1]

            # Physics Step
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Measure actual position
            l_pos = data.qpos[model.jnt_qposadr[left_joint]]
            r_pos = data.qpos[model.jnt_qposadr[right_joint]]
            
            # Gap calculation: 
            # If fingers overlap, this number might be negative or near zero depending on math
            # Perfect touch is usually around 0.0
            gap = l_pos - r_pos # (Left - Right). Since Right is neg, this sums the distance.
            
            print(f"Status: {status} | Gap: {gap:.4f} m | L: {l_pos:.3f} R: {r_pos:.3f}", end="\r")

            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()