"""Record a policy run to a GIF or MP4 (offscreen, no window needed).

    python record.py --out docs/v33_ramp.gif
    python record.py --policy v30 --out docs/v30.mp4 --any

By default it keeps trying episodes until one reaches the landing pad (x > 8 m) and
saves that one; --any saves the first episode whatever happens. No forces are applied
to the robot, same as play.py.
"""

import argparse

import imageio.v2 as imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from course_env import LidarObservation, make_course_env
from play import LANDING_START, POLICIES


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=list(POLICIES), default="v33")
    p.add_argument("--out", required=True, help=".gif or .mp4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--any", action="store_true", help="save the first episode even if it falls short")
    p.add_argument("--max-tries", type=int, default=20)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=270)
    p.add_argument("--every", type=int, default=3, help="keep every Nth frame (sim runs at ~67 fps)")
    args = p.parse_args()
    cfg = POLICIES[args.policy]

    def make():
        env = make_course_env(cfg["course"], max_episode_steps=cfg["max_steps"],
                              healthy_z_range=(cfg["healthy_z_min"], 3.0), render_mode="rgb_array",
                              width=args.width, height=args.height)
        return LidarObservation(env)

    venv = VecNormalize.load(cfg["stats"], VecFrameStack(DummyVecEnv([make]), n_stack=4))
    venv.training = False
    venv.norm_reward = False
    venv.seed(args.seed)
    model = PPO.load(cfg["model"], env=venv)
    env = venv.venv.venv.envs[0]

    for attempt in range(1, args.max_tries + 1):
        obs = venv.reset()
        frames, max_x, step = [env.render()], 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = venv.step(action)
            step += 1
            if done[0]:
                max_x = max(max_x, float(info[0].get("x_position", max_x)))
                break
            max_x = max(max_x, float(env.unwrapped.data.qpos[0]))
            if step % args.every == 0:
                frames.append(env.render())
        print(f"attempt {attempt}: reached x = {max_x:.2f} m")
        if args.any or max_x > LANDING_START:
            fps = round(env.unwrapped.metadata["render_fps"] / args.every)
            imageio.mimsave(args.out, frames, fps=fps) if args.out.endswith(".mp4") else \
                imageio.mimsave(args.out, frames, duration=1000 / fps, loop=0)
            print(f"saved {args.out} ({len(frames)} frames, {fps} fps)")
            return
    print("no episode reached the landing pad; try --any or another --seed")


if __name__ == "__main__":
    main()
