import gymnasium as gym
import os

gym_dir = os.path.dirname(gym.__file__)
asset_path = None

# Smart search for either naming convention
for root, dirs, files in os.walk(gym_dir):
    if "humanoid.xml" in files or "humanoid_v4.xml" in files:
        target = "humanoid_v4.xml" if "humanoid_v4.xml" in files else "humanoid.xml"
        asset_path = os.path.join(root, target)
        break

if asset_path:
    with open(asset_path, "r") as f:
        content = f.read()

    # 1. Replace the sharp Box bumps with smooth Cylinders
    # We use euler="90 0 0" to lay the cylinders flat across the track
    content = content.replace(
        '<geom name="bump1" type="box" pos="1.5 0 0.025" size="0.1 2 0.025" rgba=".7 .3 .3 1"/>',
        '<geom name="bump1" type="cylinder" pos="1.5 0 0.025" size="0.05 2" euler="90 0 0" rgba=".7 .3 .3 1"/>'
    )
    content = content.replace(
        '<geom name="bump2" type="box" pos="3.0 0 0.04" size="0.15 2 0.04" rgba=".8 .4 .2 1"/>',
        '<geom name="bump2" type="cylinder" pos="3.0 0 0.04" size="0.075 2" euler="90 0 0" rgba=".8 .4 .2 1"/>'
    )

    with open(asset_path, "w") as f:
        f.write(content)
    print(f"✅ The speed bumps in {os.path.basename(asset_path)} have been smoothed out into cylinders!")
else:
    print("Could not find the XML file.")