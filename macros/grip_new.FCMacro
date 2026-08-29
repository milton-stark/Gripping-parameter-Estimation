import FreeCAD as App
import FreeCADGui as Gui
import Part
import json
import os
from PySide import QtGui

# =========================
# CONFIGURATION
# =========================
DENSITY_KG_PER_M3 = 2700   # Aluminium
MU = 0.5
SAFETY = 2.0
G = 9.80665

PRE_GRIP_OFFSET_MM = 20.0   # mm above the top surface (world ZMax)
POST_CLEARANCE_MM = 15.0
RELEASE_HEIGHT_MM = 25.0

# Grip point near top (battery-in-casing: only a small part is exposed)
GRIP_ZONE_FROM_TOP_MM = 5.0    # Allowed gripping zone measured downward from the top surface
GRIP_DEPTH_BELOW_TOP_MM = 2.5  # Pick a point this far below the top (must be <= GRIP_ZONE_FROM_TOP_MM)

SPHERE_RADIUS = 0.5
AXIS_LENGTH = 50.0
ROUND = 6

VIS_PREFIX = "GRIP_VIZ_"

# =========================
# HELPERS
# =========================
def r(x): return round(x, ROUND)

def vec(v):
    return [r(v.x), r(v.y), r(v.z)]

def clear_visuals():
    doc = App.ActiveDocument
    for o in doc.Objects:
        if o.Name.startswith(VIS_PREFIX):
            doc.removeObject(o.Name)

def make_sphere(name, pos, color):
    doc = App.ActiveDocument
    s = doc.addObject("Part::Sphere", VIS_PREFIX + name)
    s.Radius = SPHERE_RADIUS
    s.Placement.Base = pos
    s.ViewObject.ShapeColor = color
    s.ViewObject.Transparency = 30
    return s

def make_axis_line(name, start, direction, color=(1.0, 1.0, 0.0)):
    doc = App.ActiveDocument
    end = start + direction.multiply(AXIS_LENGTH)
    line = Part.makeLine(start, end)
    obj = doc.addObject("Part::Feature", VIS_PREFIX + name)
    obj.Shape = line
    obj.ViewObject.LineColor = color
    obj.ViewObject.LineWidth = 2
    return obj

# =========================
# MAIN
# =========================
def main():
    doc = App.ActiveDocument
    if not doc:
        App.Console.PrintError("No active document.\n")
        return

    sel = Gui.Selection.getSelection()
    if not sel:
        App.Console.PrintError("Please select the Mesh or Object in the tree.\n")
        return

    obj = sel[0]

    # --- FILE DIALOG ---
    default_path = os.path.join(os.path.expanduser("~"), f"{obj.Label}_grip_data.json")
    file_path, _ = QtGui.QFileDialog.getSaveFileName(None, "Save JSON", default_path, "JSON (*.json)")
    if not file_path:
        return

    clear_visuals()

    # --- GEOMETRY DATA EXTRACTION ---
    if hasattr(obj, "Mesh"):
        mesh = obj.Mesh
        bb = mesh.BoundBox
        com = bb.Center
        volume_mm3 = mesh.Volume
    elif hasattr(obj, "Shape"):
        shape = obj.Shape
        bb = shape.BoundBox
        com = shape.CenterOfMass
        volume_mm3 = shape.Volume
    else:
        App.Console.PrintError("Object type not supported.\n")
        return

    # --- CALCULATION LOGIC ---
    # Top Surface Z coordinate
    z_max = bb.ZMax

    # --- GRIP POINT (near top) ---
    # Choose XY center, and Z within GRIP_ZONE_FROM_TOP_MM below the top surface
    grip_depth = min(max(GRIP_DEPTH_BELOW_TOP_MM, 0.0), GRIP_ZONE_FROM_TOP_MM)
    x_c = 0.5 * (bb.XMin + bb.XMax)
    y_c = 0.5 * (bb.YMin + bb.YMax)
    grip_guess = App.Vector(x_c, y_c, z_max - grip_depth)

    # If we have a solid shape, project the guess onto the actual surface for robustness.
    grip_world = grip_guess
    if "shape" in locals():
        try:
            dist, pts, info = shape.distToShape(Part.Vertex(grip_guess))
            # pts[0][0] is the closest point on the shape
            if pts and pts[0] and hasattr(pts[0][0], "x"):
                grip_world = pts[0][0]
        except Exception:
            grip_world = grip_guess

    # Store grip point relative to COM (matches the JSON convention used in this macro)
    grip_rel = grip_world - com

    # Pre-Grip: above top surface (world)
    pre_grip_world_z = z_max + PRE_GRIP_OFFSET_MM
    pre_z_relative = pre_grip_world_z - com.z

    # Post-Grip: SAME AS PRE-GRIP (as requested)
    post_z_relative = pre_z_relative  # Post-grip = same as pre-grip

    # Mass & Force
    volume_m3 = abs(volume_mm3) * 1e-9
    mass = volume_m3 * DENSITY_KG_PER_M3
    f_per_jaw = (mass * G * SAFETY) / (2 * MU)

    # --- JSON DATA ---
    data = {
        "object_id": f"{obj.Label}_grip_COM",
        "relative_to_tcp": vec(com),
        "source": "freecad_grip_macro_COM_weighted",
        "units": "mm, N",
        "geometry": {
            "bounding_box_mm": [r(bb.XLength), r(bb.YLength), r(bb.ZLength)],
            "center_mm": [0.0, 0.0, 0.0],
            "principal_axis_world": {
                "axis_dir_unit": [0.0, 0.0, 1.0]
            }
        },
        "grasp_parameters": {
            "grip_point_mm": vec(grip_rel),
            "pre_grip_mm": [0.0, 0.0, r(pre_z_relative)],
            "post_grip_mm": [0.0, 0.0, r(post_z_relative)],
            "release_height_mm": RELEASE_HEIGHT_MM
        },
        "force_estimation": {
            "mass_kg": r(mass),
            "friction_coefficient": MU,
            "safety_factor": SAFETY,
            "normal_force_per_jaw_N": r(f_per_jaw),
            "total_normal_force_both_jaws_N": r(2 * f_per_jaw)
        },
        "post_grip_logic": {
            "zmax_of_object": r(z_max),
            "pre_grip_offset_mm": PRE_GRIP_OFFSET_MM,
            "post_clearance_mm": POST_CLEARANCE_MM,
            "final_post_z_mm": r(pre_grip_world_z)  # post == pre
        }
    }

    # --- SAVE ---
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    # --- VISUALIZATION ---
    make_sphere("GRIP_POINT", grip_world, (1.0, 0.0, 0.0))  # Red
    make_sphere("PRE_GRIP", App.Vector(com.x, com.y, pre_grip_world_z), (0.2, 0.4, 1.0))  # Blue
    make_sphere("POST_GRIP", App.Vector(com.x, com.y, pre_grip_world_z), (0.2, 0.8, 0.2))  # Green (same as pre)
    make_axis_line("MOTION_AXIS_Z", com, App.Vector(0, 0, 1))

    doc.recompute()
    App.Console.PrintMessage(f"Exported to: {file_path}\n")
    App.Console.PrintMessage(f"Pre/Post-Grip set to {PRE_GRIP_OFFSET_MM}mm above Z-Max ({r(z_max)})\n")

if __name__ == "__main__":
    main()
