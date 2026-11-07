"""Training plots from a *_logs.npz file.

    python plot_results.py --logs results/cartpole_seed0_logs.npz --env cartpole
Makes: (i) max Q vs episodes, (ii) episode reward + moving average of last 100, and the non-standard plot:
(iii) mean age of the replay-buffer tuples sampled for training, vs episodes.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--logs", required=True)
p.add_argument("--env", choices=["cartpole", "pacman"], required=True)
args = p.parse_args()
d = np.load(args.logs)
prefix = args.logs.replace("_logs.npz", "")
eps = np.arange(1, len(d["rewards"]) + 1)

# (i) maximum Q value per episode
plt.figure(figsize=(7, 4))
plt.plot(eps, d["max_q"], label="max Q(s,a) over the episode")
if args.env == "cartpole":      # +1 reward per step and gamma = 0.95  ->  Q can never exceed 1/(1-gamma) = 20
    plt.axhline(1 / (1 - 0.95), color="red", linestyle="--", label="theoretical bound 1/(1-γ) = 20")
plt.xlabel("training episode"); plt.ylabel("max Q-value"); plt.title(f"{args.env}: max Q-value vs episodes")
plt.legend(); plt.tight_layout(); plt.savefig(prefix + "_maxq.png", dpi=150); plt.close()

# (ii) episode rewards + moving average of the last 100 episodes
ma = np.array([d["rewards"][max(0, i - 99): i + 1].mean() for i in range(len(eps))])
plt.figure(figsize=(7, 4))
plt.plot(eps, d["rewards"], alpha=0.35, label="episode reward")
plt.plot(eps, ma, color="red", label="moving average (last 100)")
plt.xlabel("training episode"); plt.ylabel("episode reward"); plt.title(f"{args.env}: episode rewards")
plt.legend(); plt.tight_layout(); plt.savefig(prefix + "_rewards.png", dpi=150); plt.close()

# (iii) non-standard: how old are the transitions the network learns from?
plt.figure(figsize=(7, 4))
plt.plot(eps, d["mean_age"])
plt.xlabel("training episode"); plt.ylabel("mean age of sampled tuples (env steps)")
plt.title(f"{args.env}: age of replay-buffer tuples used in updates")
plt.tight_layout(); plt.savefig(prefix + "_buffer_age.png", dpi=150); plt.close()
print("saved plots with prefix", prefix)
