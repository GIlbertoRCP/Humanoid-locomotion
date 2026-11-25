"""Build the obstacle-course XMLs from Gymnasium's stock humanoid model.

Writes two files into assets/ and never touches the installed Gymnasium package:

    assets/course.xml       the course the v30 policy was trained on
    assets/course_demo.xml  the same course with the landing pad lowered
                            (top at z = 0.6 instead of 1.0), used for the demo video

Both add the same three things to the stock humanoid:
  * five rangefinder "LiDAR" rays on the torso, fanned out vertically from straight
    down to straight ahead,
  * two cylindrical bumps, an 8-degree ramp and a landing pad,
  * the rangefinder sensors, which the policy reads as 5 extra observation values.
"""

import argparse
import os

import gymnasium

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
STOCK_XML = os.path.join(os.path.dirname(gymnasium.__file__), "envs", "mujoco", "assets", "humanoid.xml")

SITES = """
        <site name="laser_down" pos="0 0 0.1" size="0.01" rgba="1 0 0 1" zaxis="0 0 -1"/>
        <site name="laser_steep" pos="0 0 0.1" size="0.01" rgba="1 0 0 1" zaxis="0.5 0 -1"/>
        <site name="laser_45" pos="0 0 0.1" size="0.01" rgba="1 0 0 1" zaxis="1 0 -1"/>
        <site name="laser_shallow" pos="0 0 0.1" size="0.01" rgba="1 0 0 1" zaxis="1 0 -0.5"/>
        <site name="laser_fwd" pos="0 0 0.1" size="0.01" rgba="1 0 0 1" zaxis="1 0 0"/>"""

SENSORS = """
    <sensor>
        <rangefinder name="ray1_down" site="laser_down"/>
        <rangefinder name="ray2_steep" site="laser_steep"/>
        <rangefinder name="ray3_45deg" site="laser_45"/>
        <rangefinder name="ray4_shallow" site="laser_shallow"/>
        <rangefinder name="ray5_fwd" site="laser_fwd"/>
    </sensor>
"""


def terrain(landing_z):
    return f"""
        <geom name="bump1" type="cylinder" pos="1.5 0 0.025" size="0.05 2" euler="90 0 0" rgba=".7 .3 .3 1"/>
        <geom name="bump2" type="cylinder" pos="3.0 0 0.04" size="0.075 2" euler="90 0 0" rgba=".8 .4 .2 1"/>
        <geom name="ramp_up" type="box" pos="6 0 0.25" size="2 2 0.25" euler="0 -8 0" rgba=".5 .5 .5 1"/>
        <geom name="landing" type="box" pos="10 0 {landing_z}" size="2 2 0.5" rgba=".4 .6 .4 1"/>
"""


def build(landing_z):
    with open(STOCK_XML) as f:
        xml = f.read()

    torso = '<body name="torso" pos="0 0 1.4">'
    for needle in (torso, "</worldbody>", "</mujoco>"):
        if xml.count(needle) != 1:
            raise RuntimeError(f"Unexpected humanoid.xml layout: {needle!r} not found exactly once")

    xml = xml.replace(torso, torso + SITES)
    xml = xml.replace("</worldbody>", terrain(landing_z) + "    </worldbody>")
    xml = xml.replace("</mujoco>", SENSORS + "</mujoco>")
    return xml


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    os.makedirs(ASSETS, exist_ok=True)
    for name, landing_z in (("course.xml", 0.5), ("course_demo.xml", 0.1)):
        path = os.path.join(ASSETS, name)
        with open(path, "w") as f:
            f.write(build(landing_z))
        print(f"wrote {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    main()
