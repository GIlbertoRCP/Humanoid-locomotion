"""Watch the trained policy, or measure how far it gets over many episodes.

    python play.py                      # watch v33 on the demo course
    python play.py --eval 50            # headless: 50 episodes, print distance stats
    python play.py --policy v30 --eval 50

Each policy runs with the settings it was trained under (course, minimum torso height,
episode length); any of them can be overridden with the flags below.

Course layout along x (meters): bumps at 1.5 and 3.0, ramp from 4 to 8, landing pad
from 8 to 12. No forces or velocities are applied to the robot; every movement comes
from the policy. (The original presentation script set a minimum forward speed between
x = 5 and 9.5. That is left out here; `--presentation-assist` turns it back on, only so
the two can be compared.)
"""

import argparse
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from course_env import LidarObservation, make_course_env

POLICIES = {
    # v33: 5M-step fine-tune of v30 that gets up the ramp (finetune_v33.py)
    "v33": dict(model="models/ppo_humanoid_v33_surge.zip", stats="models/vec_normalize_v33.pkl",
                course="demo", healthy_z_min=0.5, max_steps=2000),
    # v30: end of the ~160M-step main training chain (train.py)
    "v30": dict(model="models/ppo_humanoid_vision_v30.zip", stats="models/vec_normalize_v30.pkl",
                course="train", healthy_z_min=0.9, max_steps=3000),
}
RAMP_START, LANDING_START = 4.0, 8.0


class PresentationAssist(LidarObservation):
    """Reproduces the original presentation script: forward speed >= 1.8 m/s on x in (5, 9.5)."""

    def step(self, action):
        data = self.env.unwrapped.data
        if 5.0 < data.qpos[0] < 9.5 and data.qvel[0] < 1.8:
            data.qvel[0] = 1.8
        return super().step(action)


def build(course, render, healthy_z_min, assist, max_steps, stats):
    def make():
        env = make_course_env(course, max_episode_steps=max_steps,
                              healthy_z_range=(healthy_z_min, 3.0),
                              render_mode="human" if render else None)
        return (PresentationAssist if assist else LidarObservation)(env)

    venv = VecFrameStack(DummyVecEnv([make]), n_stack=4)
    venv = VecNormalize.load(stats, venv)
    venv.training = False
    venv.norm_reward = False
    return venv


def run_episode(model, venv, deterministic, render):
    data = venv.venv.venv.envs[0].unwrapped.data
    obs = venv.reset()
    max_x, steps, min_z_on_ramp = 0.0, 0, np.inf
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, done, info = venv.step(action)
        steps += 1
        if not done[0]:
            x, z = float(data.qpos[0]), float(data.qpos[2])
        else:  # the env auto-resets; read the final position from terminal info
            x = float(info[0].get("x_position", x))
        max_x = max(max_x, x)
        if x > RAMP_START and not done[0]:
            min_z_on_ramp = min(min_z_on_ramp, z)
        if render:
            time.sleep(0.02)
        if done[0]:
            timed_out = info[0].get("TimeLimit.truncated", False)
            return {"max_x": max_x, "steps": steps, "timed_out": timed_out, "min_z_on_ramp": min_z_on_ramp}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=list(POLICIES), default="v33")
    p.add_argument("--eval", type=int, metavar="N", help="run N headless episodes and print stats")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--course", choices=["demo", "train"])
    p.add_argument("--healthy-z-min", type=float, help="episode ends if the torso drops below this height")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--model")
    p.add_argument("--stats", help="VecNormalize .pkl saved with --model")
    p.add_argument("--presentation-assist", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    for key, value in POLICIES[args.policy].items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    render = args.eval is None
    venv = build(args.course, render, args.healthy_z_min, args.presentation_assist, args.max_steps, args.stats)
    venv.seed(args.seed)
    model = PPO.load(args.model, env=venv)

    if render:
        while True:
            r = run_episode(model, venv, not args.stochastic, render=True)
            print(f"episode ended at x = {r['max_x']:.2f} m after {r['steps']} steps")

    results = [run_episode(model, venv, not args.stochastic, render=False) for _ in range(args.eval)]
    xs = np.array([r["max_x"] for r in results])
    print(f"policy={args.policy}  course={args.course}  episodes={len(xs)}  deterministic={not args.stochastic}  "
          f"healthy_z_min={args.healthy_z_min}  assist={args.presentation_assist}")
    print(f"furthest x: median {np.median(xs):.2f} m, min {xs.min():.2f}, max {xs.max():.2f}")
    print(f"reached ramp (x>{RAMP_START}): {np.mean(xs > RAMP_START):.0%}   "
          f"reached landing (x>{LANDING_START}): {np.mean(xs > LANDING_START):.0%}   "
          f"ran out the clock: {np.mean([r['timed_out'] for r in results]):.0%}")
    ramp_z = [r["min_z_on_ramp"] for r in results if np.isfinite(r["min_z_on_ramp"])]
    if ramp_z:
        print(f"lowest torso height once past x={RAMP_START}: median {np.median(ramp_z):.2f} m")


if __name__ == "__main__":
    main()
