# History

Earlier training stages, kept exactly as they were run. They are here to show how the project got to v30, not to be run: most of them expect model and stats files that aren't in this repo, and the course scripts overwrite the XML inside the installed Gymnasium package. Use `build_course.py` and `course_env.py` in the repo root instead.

| File | What changed |
|---|---|
| `01_walker2d_baseline.py` | Started in 2D: PPO on Walker2d for 100k steps. |
| `02_humanoid_3d.py` | Moved to the 3D Humanoid-v4, 10M steps on flat ground. |
| `03_humanoid_strict.py` | The agent learned to dive to end episodes early. This run raised the "still standing" height so diving counts as falling (the comment says 1.3 m; the code uses 1.15 m). |
| `04_humanoid_goldilocks.py` | Same 1.15 m limit, longer 10M-step run: low enough to allow bent knees, high enough to stop diving. |
| `05_obstacle_course.py` | First obstacle course (bumps, ramp, landing), no sensing yet. |
| `06_lidar_v1.py` | Added the five rangefinder rays so the policy can see the terrain ahead. |
| `07_lidar_v29.py` | The stage right before v30 (`train.py`), which resumed from it. |

`course_setup/` has the original scripts that built the course by editing Gymnasium's installed XML (`inject_lasers.py`, `pave_terrain.py`, `upgrade_hardware.py`) and the original presentation playback script. `upgrade_hardware.py`'s replacements never matched the XML, so the "lighter legs" change it describes never took effect. `presentation_playback.py` sets a minimum forward speed on the ramp; see the main README.
