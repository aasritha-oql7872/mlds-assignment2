"""Train DQN on CartPole-v1 or MsPacman-v5.

    python train.py --env cartpole
    python train.py --env pacman --episodes 5000
"""
import argparse
import os
import torch
from dqn import CNN, MLP, train
from envs import make_env

# Per-environment hyper-parameters. gamma values come from the assignment (0.95 / 0.99).
CONFIGS = {
    "cartpole": dict(gamma=0.95, lr=5e-4, lr_decay=0.9997, batch_size=64, buffer_size=50_000, learning_starts=1_000,
                     train_every=1, tau=0.9995, eps_end=0.01, eps_decay_steps=10_000, n_episodes=2500,
                     log_every=10, eval_every=20, eval_episodes=10),
    "pacman": dict(gamma=0.99, lr=1e-4, lr_decay=0.9995, batch_size=32, buffer_size=50_000, learning_starts=10_000,
                   train_every=4, target_update=2_000, eps_end=0.05, eps_decay_steps=1_000_000, n_episodes=5000,
                   clip_rewards=True, log_every=10),
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=list(CONFIGS), required=True)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    cfg = dict(CONFIGS[args.env], seed=args.seed)
    if args.episodes:
        cfg["n_episodes"] = args.episodes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| config:", cfg, flush=True)

    env = make_env(args.env)
    n_actions = env.action_space.n
    obs_shape, obs_dtype = env.observation_space.shape, env.observation_space.dtype
    if args.env == "cartpole":
        q_net = MLP(obs_shape[0], n_actions)
    else:
        q_net = CNN(obs_shape[0], n_actions)
    q_net.to(device)

    os.makedirs(args.out, exist_ok=True)
    train(env, q_net, obs_shape, obs_dtype, n_actions, device, cfg,
          save_path=os.path.join(args.out, f"{args.env}_seed{args.seed}"),
          log_fn=lambda m: print(m, flush=True))
