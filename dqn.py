"""Core DQN: Q-networks, replay buffer, target network, training loop.

Algorithm = "classic" DQN from the Week 6 slides:
  1. take an eps-greedy action, store (s, a, r, s', done) in the replay buffer
  2. sample a mini-batch uniformly from the buffer
  3. y_j = r_j + gamma * max_a' Q_target(s'_j, a')      (y_j = r_j if s'_j is terminal)
  4. phi <- phi - alpha * grad (Q_phi(s_j, a_j) - y_j)^2   (Huber loss instead of squared loss)
  5. every N steps copy phi into the target network phi'
"""
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    """Q-network for vector states (CartPole, toy MDPs)."""

    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x.float())


class CNN(nn.Module):
    """Q-network for images (MsPacman). Input: int8 frames in [-128, 127], shape (B, C, 88, 80)."""

    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),   # 88x80 -> 21x19
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),            # -> 9x8
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU())            # -> 7x6
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 6, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, x):
        x = (x.float() + 128.0) / 255.0                            # back to [0, 1]
        return self.fc(self.conv(x).flatten(1))


# ----------------------------------------------------------------------------
# Replay buffer (uniform sampling)
# ----------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity, obs_shape, obs_dtype):
        self.capacity = capacity
        self.s = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.s2 = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)     # 1.0 only for true termination
        self.birth = np.zeros(capacity, dtype=np.int64)      # env step at which tuple was stored (for the "age" plot)
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, s2, done, step):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.done[i], self.birth[i] = s, a, r, s2, done, step
        self.ptr = (self.ptr + 1) % self.capacity           # overwrite oldest when full
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return idx, self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.done[idx]


# ----------------------------------------------------------------------------
# DQN loss / target (kept as separate functions so the tests can call them directly)
# ----------------------------------------------------------------------------
@torch.no_grad()
def compute_targets(target_net, r, s2, done, gamma):
    """y = r + gamma * max_a' Q_target(s', a') * (1 - done)."""
    max_next_q = target_net(s2).max(dim=1).values
    return r + gamma * max_next_q * (1.0 - done)


def dqn_loss(q_net, target_net, batch, gamma, device, loss="huber"):
    s, a, r, s2, done = [torch.as_tensor(x, device=device) for x in batch]
    q_sa = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)       # Q_phi(s_j, a_j)
    y = compute_targets(target_net, r, s2, done, gamma)
    if loss == "mse":
        return F.mse_loss(q_sa, y), y                          # (Q - y)^2, as in the basic algorithm
    return F.smooth_l1_loss(q_sa, y), y                        # Huber loss (slide: "Advanced Tips")


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
DEFAULTS = dict(
    gamma=0.95, lr=1e-3, lr_decay=1.0,          # lr_decay: multiplied into the lr after every episode
    batch_size=64, buffer_size=50_000, learning_starts=1_000,
    train_every=1,                              # one gradient step every `train_every` env steps
    target_update=500,                          # copy phi -> phi' every N env steps
    tau=None,                                   # if set: Polyak target phi' <- tau*phi' + (1-tau)*phi every step (slide 14)
    eps_start=1.0, eps_end=0.01, eps_decay_steps=10_000,   # linear epsilon schedule (per env step)
    loss="huber", grad_clip=10.0, clip_rewards=False,
    treat_truncation_as_done=False,             # False = correct handling (time-limit is NOT a terminal state)
    use_target_network=True,
    n_episodes=500, seed=0, log_every=10,
    eval_every=None, eval_episodes=10,          # if set: greedy validation rollouts; best network is saved by eval score
)


def select_action(q_net, obs, eps, n_actions, device):
    """eps-greedy. Returns (action, max_a Q(s, a))."""
    with torch.no_grad():
        q = q_net(torch.as_tensor(obs, device=device).unsqueeze(0))[0]
    a = np.random.randint(n_actions) if np.random.rand() < eps else int(q.argmax())
    return a, float(q.max())


def train(env, q_net, obs_shape, obs_dtype, n_actions, device, cfg, save_path=None, log_fn=print):
    """Train q_net on env. Returns a dict of per-episode logs."""
    cfg = {**DEFAULTS, **cfg}
    np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    target_net = copy.deepcopy(q_net)
    target_net.eval()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg["lr_decay"])
    buffer = ReplayBuffer(cfg["buffer_size"], obs_shape, obs_dtype)

    logs = dict(rewards=[], max_q=[], mean_age=[], lengths=[], loss=[], eval_episode=[], eval_reward=[])
    step, best_ma, best_eval, t0 = 0, -np.inf, -np.inf, time.time()

    for ep in range(cfg["n_episodes"]):
        obs, _ = env.reset(seed=cfg["seed"] + ep)
        ep_reward, ep_max_q, ep_len, ep_losses, ep_ages = 0.0, -np.inf, 0, [], []
        done = False
        while not done:
            eps = max(cfg["eps_end"], cfg["eps_start"] - (cfg["eps_start"] - cfg["eps_end"]) * step / cfg["eps_decay_steps"])
            a, q_max = select_action(q_net, obs, eps, n_actions, device)
            obs2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_reward += r                                   # log the raw reward
            ep_max_q = max(ep_max_q, q_max)
            train_r = float(np.sign(r)) if cfg["clip_rewards"] else r
            # Edge case: only a TRUE terminal state has value 0. A time-limit truncation must still bootstrap.
            stored_done = float(terminated or (truncated and cfg["treat_truncation_as_done"]))
            buffer.add(obs, a, train_r, obs2, stored_done, step)
            obs = obs2
            step += 1; ep_len += 1

            if buffer.size >= cfg["learning_starts"] and step % cfg["train_every"] == 0:
                idx, *batch = buffer.sample(cfg["batch_size"])
                loss, _ = dqn_loss(q_net, target_net, batch, cfg["gamma"], device, cfg["loss"])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), cfg["grad_clip"])
                optimizer.step()
                ep_losses.append(loss.item())
                ep_ages.append(step - buffer.birth[idx].mean())
            # target network: hard copy every N steps (with use_target_network=False it is copied every step)
            if cfg["tau"] is not None:
                with torch.no_grad():
                    for tp, p in zip(target_net.parameters(), q_net.parameters()):
                        tp.mul_(cfg["tau"]).add_((1 - cfg["tau"]) * p)
            elif step % (cfg["target_update"] if cfg["use_target_network"] else 1) == 0:
                target_net.load_state_dict(q_net.state_dict())

        scheduler.step()
        logs["rewards"].append(ep_reward); logs["max_q"].append(ep_max_q); logs["lengths"].append(ep_len)
        logs["loss"].append(np.mean(ep_losses) if ep_losses else np.nan)
        logs["mean_age"].append(np.mean(ep_ages) if ep_ages else np.nan)

        ma = np.mean(logs["rewards"][-100:])
        if cfg["eval_every"] is None:
            # best network = highest moving average of the last 100 training episodes
            if save_path and len(logs["rewards"]) >= min(100, cfg["n_episodes"]) and ma > best_ma:
                best_ma = ma
                torch.save(q_net.state_dict(), save_path + "_best.pt")
        elif (ep + 1) % cfg["eval_every"] == 0:
            # best network = highest mean reward of greedy validation rollouts (seeds differ from training seeds)
            ev = np.mean(evaluate(env, q_net, cfg["eval_episodes"], device, eps=0.0, seed=10**6 + ep, log_fn=lambda m: None))
            q_net.train()
            logs["eval_episode"].append(ep + 1); logs["eval_reward"].append(ev)
            if save_path and ev >= best_eval:
                best_eval = ev
                torch.save(q_net.state_dict(), save_path + "_best.pt")
        if (ep + 1) % cfg["log_every"] == 0:
            log_fn(f"ep {ep+1:5d} | reward {ep_reward:8.1f} | MA100 {ma:8.1f} | maxQ {ep_max_q:8.2f} | "
                   f"eps {eps:.3f} | lr {scheduler.get_last_lr()[0]:.2e} | steps {step} | {time.time()-t0:6.0f}s")
        if save_path and (ep + 1) % 50 == 0:                  # periodic save so a dropped SSH session loses little
            torch.save(q_net.state_dict(), save_path + "_last.pt")
            np.savez(save_path + "_logs.npz", **{k: np.array(v) for k, v in logs.items()})

    if save_path:
        torch.save(q_net.state_dict(), save_path + "_last.pt")
        np.savez(save_path + "_logs.npz", **{k: np.array(v) for k, v in logs.items()})
    return logs


def evaluate(env, q_net, n_episodes, device, eps=0.0, seed=10_000, log_fn=print):
    """Roll out the (almost) greedy policy for n_episodes, return the list of episode rewards."""
    q_net.eval()
    n_actions = q_net(torch.as_tensor(env.reset(seed=seed)[0], device=device).unsqueeze(0)).shape[1]
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            a, _ = select_action(q_net, obs, eps, n_actions, device)
            obs, r, terminated, truncated, _ = env.step(a)
            total += r
            done = terminated or truncated
        rewards.append(total)
        if (ep + 1) % 50 == 0:
            log_fn(f"eval episode {ep+1}/{n_episodes} | mean so far {np.mean(rewards):.1f}")
    return rewards
