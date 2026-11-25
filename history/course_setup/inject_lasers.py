import gymnasium as gym
import os
import shutil

gym_dir = os.path.dirname(gym.__file__)
found_target = False

for root, dirs, files in os.walk(gym_dir):
    if "humanoid.xml" in files or "humanoid_v4.xml" in files:
        target = "humanoid_v4.xml" if "humanoid_v4.xml" in files else "humanoid.xml"
        asset_path = os.path.join(root, target)
        backup_path = asset_path + ".backup"
        
        # 1. Restore from clean backup to wipe out the broken lasers
        if os.path.exists(backup_path):
            shutil.copy(backup_path, asset_path)
            print(f"Restored clean {target} from backup.")
        
        with open(asset_path, "r") as f:
            content = f.read()

        # 2. Inject the Obstacle Course Terrain
        terrain_xml = """
    <geom name="bump1" type="box" pos="1.5 0 0.025" size="0.1 2 0.025" rgba=".7 .3 .3 1"/>
    <geom name="bump2" type="box" pos="3.0 0 0.04" size="0.15 2 0.04" rgba=".8 .4 .2 1"/>

    <geom name="ramp_up" type="box" pos="6 0 0.25" size="2 2 0.25" euler="0 -8 0" rgba=".5 .5 .5 1"/>

    <geom name="landing" type="box" pos="10 0 0.5" size="2 2 0.5" rgba=".4 .6 .4 1"/>
        """
        if "bump1" not in content:
            content = content.replace("</worldbody>", terrain_xml + "\n    </worldbody>")

        # 3. THE FIX: Attach 5 Angled Laser Mounts using 'zaxis'
        sites = """
        <site name="laser_down" pos="0 0 0.1" size="0.01" rgba="1 0 0 1" zaxis="0 0 -1"/>
        <site name="laser_steep" pos="0 0 0.1" size="0.01" rgba="1 0 0 1" zaxis="0.5 0 -1"/>
        <site name="laser_45" pos="0 0 0.1" size="0.01" rgba="1 0 0 1" zaxis="1 0 -1"/>
        <site name="laser_shallow" pos="0 0 0.1" size="0.01" rgba="1 0 0 1" zaxis="1 0 -0.5"/>
        <site name="laser_fwd" pos="0 0 0.1" size="0.01" rgba="1 0 0 1" zaxis="1 0 0"/>
        """
        if 'name="laser_down"' not in content:
            content = content.replace(
                '<body name="torso" pos="0 0 1.4">',
                '<body name="torso" pos="0 0 1.4">\n' + sites
            )

        # 4. Wire up the 5 LIDAR Rangefinders to the angled mounts
        sensors = """
    <sensor>
        <rangefinder name="ray1_down" site="laser_down"/>
        <rangefinder name="ray2_steep" site="laser_steep"/>
        <rangefinder name="ray3_45deg" site="laser_45"/>
        <rangefinder name="ray4_shallow" site="laser_shallow"/>
        <rangefinder name="ray5_fwd" site="laser_fwd"/>
    </sensor>
</mujoco>
        """
        if '<rangefinder name="ray1_down"' not in content:
            content = content.replace('</mujoco>', sensors)

        with open(asset_path, "w") as f:
            f.write(content)
        
        print(f"✅ TERRAIN AND CYBERNETICS INSTALLED IN {target}")
        found_target = True
        break

if not found_target:
    print("Could not find the core humanoid XML file.")