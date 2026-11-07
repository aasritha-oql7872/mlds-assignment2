"""Roll out 500 episodes with a trained network; plot histogram of episode rewards, report mean and std.

    python evaluate.py --env cartpole --weights results/cartpole_seed0_best.pt
    python evaluate.py --env pacman   --weights results/pacman_seed0_best.pt --eps 0.05
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dqn import CNN, MLP, evaluate
from envs import make_env

p = argparse.ArgumentParser()
p.add_argument("--env", choices=["cartpole", "pacman"], required=True)
p.add_argument("--weights", required=True)
p.add_argument("--episodes", type=int, default=500)
p.add_argument("--eps", type=float, default=0.0, help="exploration during rollout (Pacman: 0.05 avoids getting stuck)")
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = make_env(args.env)
n_actions, shape = env.action_space.n, env.observation_space.shape
q_net = (MLP(shape[0], n_actions) if args.env == "cartpole" else CNN(shape[0], n_actions)).to(device)
q_net.load_state_dict(torch.load(args.weights, map_location=device))

rewards = np.array(evaluate(env, q_net, args.episodes, device, eps=args.eps))
mean, std = rewards.mean(), rewards.std()
print(f"{args.env}: {args.episodes} episodes | mean reward = {mean:.2f} | std = {std:.2f} | "
      f"min = {rewards.min():.0f} | max = {rewards.max():.0f}")

plt.figure(figsize=(6, 4))
plt.hist(rewards, bins=30, edgecolor="black")
plt.axvline(mean, color="red", linestyle="--", label=f"mean = {mean:.1f}, std = {std:.1f}")
plt.xlabel("episode reward"); plt.ylabel("number of episodes")
plt.title(f"{args.env}: reward histogram over {args.episodes} rollout episodes")
plt.legend(); plt.tight_layout()
out = args.weights.replace(".pt", "") + "_eval_hist.png"
plt.savefig(out, dpi=150)
np.save(args.weights.replace(".pt", "") + "_eval_rewards.npy", rewards)
print("saved", out)
