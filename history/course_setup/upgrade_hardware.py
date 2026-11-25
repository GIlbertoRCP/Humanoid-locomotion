import gymnasium as gym
import os

gym_dir = os.path.dirname(gym.__file__)
asset_path = None

for root, dirs, files in os.walk(gym_dir):
    if "humanoid.xml" in files or "humanoid_v4.xml" in files:
        target = "humanoid_v4.xml" if "humanoid_v4.xml" in files else "humanoid.xml"
        asset_path = os.path.join(root, target)
        break

if asset_path:
    with open(asset_path, "r") as f:
        content = f.read()

    # We are drastically reducing the density/mass of the lower legs and feet!
    # By making the 'geom' tags smaller or less dense, the physics engine will 
    # require significantly less torque to swing them, unlocking the sprint!
    
    content = content.replace(
        '<geom name="right_shin1" type="capsule" fromto="0 0 0 0 0 -.3"  size="0.049"/>',
        '<geom name="right_shin1" type="capsule" fromto="0 0 0 0 0 -.3"  size="0.035" density="500"/>'
    )
    content = content.replace(
        '<geom name="left_shin1"  type="capsule" fromto="0 0 0 0 0 -.3"  size="0.049"/>',
        '<geom name="left_shin1"  type="capsule" fromto="0 0 0 0 0 -.3"  size="0.035" density="500"/>'
    )
    
    content = content.replace(
        '<geom name="right_foot" type="capsule" fromto="-0.07 0 0 0.14 0 0" size="0.046"/>',
        '<geom name="right_foot" type="capsule" fromto="-0.07 0 0 0.14 0 0" size="0.035" density="500"/>'
    )
    content = content.replace(
        '<geom name="left_foot" type="capsule" fromto="-0.07 0 0 0.14 0 0" size="0.046"/>',
        '<geom name="left_foot" type="capsule" fromto="-0.07 0 0 0.14 0 0" size="0.035" density="500"/>'
    )

    with open(asset_path, "w") as f:
        f.write(content)
    print(f"✅ The bionic legs have been installed in {os.path.basename(asset_path)}!")
else:
    print("Could not find the XML file.")