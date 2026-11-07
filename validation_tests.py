"""Validation experiments for the DQN implementation (not unit tests: each one checks that the algorithm
behaves the way theory predicts when we change the problem).

    python validation_tests.py                                   # tests 1-4, 7 (fast, ~2 min)
    python validation_tests.py --weights results/cartpole_seed0_best.pt   # + test 5 (policy sanity map)
    python validation_tests.py --ablation                        # + test 6 (ablations, ~15 min)

Results (text + png) go to results/validation/.
"""
import argparse
import os
import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dqn import MLP, ReplayBuffer, compute_targets, dqn_loss, select_action, train

OUT = "results/validation"
DEVICE = torch.device("cpu")
report = []


def log(msg):
    print(msg, flush=True)
    report.append(msg)


def check(name, ok, detail):
    log(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ----------------------------------------------------------------------------
# Toy environments with known analytic answers
# ----------------------------------------------------------------------------
class ChainEnv:
    """Deterministic chain of n states (one-hot observations). Action 1 = right, 0 = left.
    Reaching the last state gives reward 1 and terminates. Time limit -> truncation."""

    def __init__(self, n=5, limit=50):
        self.n, self.limit = n, limit

    def _obs(self):
        o = np.zeros(self.n, dtype=np.float32); o[self.s] = 1.0
        return o

    def reset(self, seed=None):
        self.s, self.t = 0, 0
        return self._obs(), {}

    def step(self, a):
        self.s = min(self.s + 1, self.n - 1) if a == 1 else max(self.s - 1, 0)
        self.t += 1
        terminated = self.s == self.n - 1
        return self._obs(), float(terminated), terminated, (not terminated) and self.t >= self.limit, {}


class ConstantRewardEnv:
    """One state, reward 1 every step, never terminates, time limit T (truncation). True Q = 1/(1-gamma)."""

    def __init__(self, T=20):
        self.T = T

    def reset(self, seed=None):
        self.t = 0
        return np.ones(1, dtype=np.float32), {}

    def step(self, a):
        self.t += 1
        return np.ones(1, dtype=np.float32), 1.0, False, self.t >= self.T, {}


class OneStepEnv(ConstantRewardEnv):
    """One state, reward 1, episode ends (terminates) after the first action. True Q = 1 (not 1/(1-gamma))."""

    def step(self, a):
        return np.ones(1, dtype=np.float32), 1.0, True, False, {}


def train_toy(env, obs_dim, n_actions, **cfg):
    net = MLP(obs_dim, n_actions, hidden=32)
    base = dict(lr=3e-3, batch_size=32, buffer_size=5000, learning_starts=200, target_update=100,
                eps_decay_steps=2000, eps_end=0.1, n_episodes=300, seed=0)
    train(env, net, (obs_dim,), np.float32, n_actions, DEVICE, {**base, **cfg}, log_fn=lambda m: None)
    return net


# ----------------------------------------------------------------------------
# Test 1: tabular ground truth.  DQN's Q must match value iteration on a chain MDP.
# ----------------------------------------------------------------------------
def test_chain_matches_value_iteration():
    n, gamma = 5, 0.9
    Q = np.zeros((n, 2))                                    # value iteration = the exact Q* (Q-iteration slide)
    for _ in range(200):
        for s in range(n - 1):
            for a in range(2):
                s2 = min(s + 1, n - 1) if a == 1 else max(s - 1, 0)
                Q[s, a] = float(s2 == n - 1) + gamma * (0 if s2 == n - 1 else Q[s2].max())
    net = train_toy(ChainEnv(n), n, 2, gamma=gamma)
    with torch.no_grad():
        Qdqn = net(torch.eye(n)).numpy()
    err = np.abs(Qdqn[:n - 1] - Q[:n - 1]).max()
    check("1. chain MDP vs value iteration", err < 0.08, f"max |Q_dqn - Q*| = {err:.3f} (threshold 0.08)")
    log("   Q* (value iteration) rows s=0..3: " + str(np.round(Q[:n - 1], 3).tolist()))
    log("   Q  (DQN)             rows s=0..3: " + str(np.round(Qdqn[:n - 1], 3).tolist()))
    # greedy policy must be 'always right'
    check("1b. greedy policy goes right", bool((Qdqn[:n - 1].argmax(1) == 1).all()), str(Qdqn[:n - 1].argmax(1).tolist()))


# ----------------------------------------------------------------------------
# Test 2: terminal-state masking (edge case).  Q must be r, not r/(1-gamma).
# ----------------------------------------------------------------------------
def test_terminal_masking():
    gamma = 0.9
    net = train_toy(OneStepEnv(), 1, 2, gamma=gamma, n_episodes=500)
    q = net(torch.ones(1, 1)).max().item()
    check("2. terminal masking", abs(q - 1.0) < 0.05,
          f"Q = {q:.3f}; expected 1.0 (without the (1-done) mask it would drift to {1/(1-gamma):.1f})")


# ----------------------------------------------------------------------------
# Test 3: time-limit truncation is NOT terminal (edge case; CartPole-v1 truncates at 500 steps).
# Uses the squared loss so that the fixed point of the regression is the mean target, which we can predict exactly.
# ----------------------------------------------------------------------------
def test_truncation_handling():
    gamma, T = 0.9, 20
    right = 1 / (1 - gamma)                                 # infinite-horizon value = 10
    wrong = 1 / (1 - gamma * (T - 1) / T)                   # fixed point if every truncation is stored as done=1
    one = torch.ones(1, 1)
    kw = dict(gamma=gamma, n_episodes=400, loss="mse")
    q_ok = train_toy(ConstantRewardEnv(T), 1, 2, **kw)(one).max().item()
    q_bad = train_toy(ConstantRewardEnv(T), 1, 2, treat_truncation_as_done=True, **kw)(one).max().item()
    check("3. truncation handled as non-terminal (MSE loss)", abs(q_ok - right) < 0.6 and abs(q_bad - wrong) < 0.6,
          f"correct handling Q={q_ok:.2f} (true {right:.1f}); truncation-as-terminal Q={q_bad:.2f} (predicted {wrong:.2f})")
    # Finding: with the Huber loss the same bug is hidden, because the rare terminal targets (error > 1) are treated as
    # outliers and Q fits the median target instead of the mean.
    q_huber = train_toy(ConstantRewardEnv(T), 1, 2, gamma=gamma, n_episodes=400,
                        treat_truncation_as_done=True)(one).max().item()
    log(f"   note: truncation-as-terminal with Huber loss gives Q={q_huber:.2f} (close to {right:.1f}): Huber hides rare terminal targets")


# ----------------------------------------------------------------------------
# Test 4: isolate the first 10 steps on CartPole and verify every number by hand.
# ----------------------------------------------------------------------------
def test_first_10_steps():
    gamma = 0.95
    torch.manual_seed(0); np.random.seed(0)
    env = gym.make("CartPole-v1")
    q_net, target_net = MLP(4, 2), MLP(4, 2)
    target_net.load_state_dict(q_net.state_dict())
    obs, _ = env.reset(seed=0)
    buf = ReplayBuffer(100, (4,), np.float32)
    log("   step | Q(s,left)  Q(s,right) | a | r | done | y by hand | y from code")
    ok = True
    for t in range(10):
        a, _ = select_action(q_net, obs, 0.0, 2, DEVICE)                      # greedy
        obs2, r, term, trunc, _ = env.step(a)
        with torch.no_grad():
            q = q_net(torch.as_tensor(obs)[None])[0]
            y_hand = r + gamma * (0.0 if term else target_net(torch.as_tensor(obs2)[None])[0].max().item())
        y_code = compute_targets(target_net, torch.tensor([r]), torch.as_tensor(obs2)[None], torch.tensor([float(term)]), gamma).item()
        ok &= abs(y_hand - y_code) < 1e-5
        log(f"   {t:4d} | {q[0]:9.4f} {q[1]:10.4f} | {a} | {r:.0f} | {int(term)} | {y_hand:9.4f} | {y_code:9.4f}")
        buf.add(obs, a, r, obs2, float(term), t)
        obs = obs2
    # Huber loss on this batch by hand
    idx, s, a, r, s2, d = buf.sample(10)
    loss, y = dqn_loss(q_net, target_net, (s, a, r, s2, d), gamma, DEVICE)
    q_sa = q_net(torch.as_tensor(s)).gather(1, torch.as_tensor(a)[:, None])[:, 0]
    diff = (q_sa - y).abs().detach().numpy()
    hand = np.mean(np.where(diff < 1, 0.5 * diff ** 2, diff - 0.5))
    ok &= abs(hand - loss.item()) < 1e-5
    check("4. first 10 steps: targets and Huber loss match hand computation", ok, f"loss code={loss.item():.5f}, by hand={hand:.5f}")


# ----------------------------------------------------------------------------
# Test 5: policy sanity map for a trained CartPole network (also the non-standard plot).
# ----------------------------------------------------------------------------
def test_cartpole_policy_map(weights):
    net = MLP(4, 2); net.load_state_dict(torch.load(weights, map_location=DEVICE)); net.eval()
    # heat map: Q(right) - Q(left) over (theta, omega) with the cart at the centre and at rest
    theta = np.linspace(-0.2, 0.2, 81)
    omega = np.linspace(-2.0, 2.0, 81)
    T, W = np.meshgrid(theta, omega)
    states = np.stack([np.zeros_like(T), np.zeros_like(T), T, W], axis=-1).reshape(-1, 4).astype(np.float32)
    with torch.no_grad():
        adv = (lambda q: q[:, 1] - q[:, 0])(net(torch.as_tensor(states))).numpy().reshape(T.shape)
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(T, W, adv, cmap="coolwarm", shading="auto", vmin=-np.abs(adv).max(), vmax=np.abs(adv).max())
    plt.colorbar(label="Q(s, push right) - Q(s, push left)")
    plt.plot(theta, -2 * theta, "k--", label="θ + 0.5·ω = 0")
    plt.xlabel("pole angle θ (rad)"); plt.ylabel("pole angular velocity ω"); plt.legend(loc="upper right")
    plt.title("CartPole learned policy map (cart x = 0, ẋ = 0)"); plt.tight_layout()
    plt.savefig(f"{OUT}/cartpole_policy_map.png", dpi=150); plt.close()

    # quantitative check on the states the greedy policy actually visits: does it push toward the lean?
    env, S = gym.make("CartPole-v1"), []
    for ep in range(20):
        o, _ = env.reset(seed=500 + ep); d = False
        while not d:
            S.append(o); a, _ = select_action(net, o, 0.0, 2, DEVICE)
            o, _, t, tr, _ = env.step(a); d = t or tr
    S = np.array(S, dtype=np.float32)
    with torch.no_grad():
        q = net(torch.as_tensor(S)).numpy()
    agree = ((q[:, 1] > q[:, 0]) == ((S[:, 2] + 0.5 * S[:, 3]) > 0)).mean()
    # the rule ignores cart position/velocity, so we only require agreement far above the 50% chance level
    check("5. learned policy pushes toward the lean", agree > 0.70, f"agreement with rule on {len(S)} visited states = {agree:.1%} (chance = 50%, threshold 70%)")
    log(f"   observation: median cart position in greedy rollouts = {np.median(S[:, 0]):.2f} (track limit is +-2.4): the policy also drifts the cart")


# ----------------------------------------------------------------------------
# Test 6: ablations on CartPole (each switch is predicted to hurt or change something measurable).
# ----------------------------------------------------------------------------
def test_ablations(seeds, episodes):
    base = dict(gamma=0.95, lr=1e-3, lr_decay=0.999, batch_size=64, buffer_size=50_000, learning_starts=1_000,
                target_update=200, eps_end=0.01, eps_decay_steps=10_000, n_episodes=episodes)
    variants = {
        "full DQN (gamma=0.95)": base,
        "no target network": dict(base, use_target_network=False),
        "tiny replay buffer (=batch size)": dict(base, buffer_size=64, learning_starts=64),
        "gamma=0.99": dict(base, gamma=0.99),
    }
    results = {}
    for name, cfg in variants.items():
        runs = []
        for sd in seeds:
            env = gym.make("CartPole-v1")
            net = MLP(4, 2)
            runs.append(train(env, net, (4,), np.float32, 2, DEVICE, dict(cfg, seed=sd), log_fn=lambda m: None))
            log(f"   ablation '{name}' seed {sd}: last-100 mean reward {np.mean(runs[-1]['rewards'][-100:]):.1f}")
        results[name] = runs
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name, runs in results.items():
        for k, key in enumerate(["rewards", "max_q"]):
            arr = np.array([r[key] for r in runs])
            if key == "rewards":
                arr = np.array([[a[max(0, i - 99):i + 1].mean() for i in range(len(a))] for a in arr])
            ax[k].plot(arr.mean(0), label=name); ax[k].fill_between(range(arr.shape[1]), arr.min(0), arr.max(0), alpha=0.15)
    ax[0].set_title("moving average (100) of episode reward"); ax[1].set_title("max Q per episode")
    ax[1].axhline(20, color="gray", ls="--"); ax[1].axhline(100, color="gray", ls=":")
    for a in ax: a.set_xlabel("training episode"); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/ablations.png", dpi=150); plt.close()

    final = {n: np.mean([np.mean(r["rewards"][-100:]) for r in runs]) for n, runs in results.items()}
    maxq = {n: np.mean([np.max(r["max_q"][-50:]) for r in runs]) for n, runs in results.items()}
    for n in results:
        log(f"   {n:35s} final MA100 reward {final[n]:7.1f} | max Q (last 50 eps) {maxq[n]:6.1f}")
    check("6a. gamma=0.95: max Q stays near 1/(1-gamma)=20", maxq["full DQN (gamma=0.95)"] < 25, f"{maxq['full DQN (gamma=0.95)']:.1f}")
    check("6b. gamma=0.99 raises the Q ceiling toward 100", maxq["gamma=0.99"] > 2 * maxq["full DQN (gamma=0.95)"], f"{maxq['gamma=0.99']:.1f}")
    check("6c. full DQN >= no-target-network", final["full DQN (gamma=0.95)"] >= final["no target network"],
          f"{final['full DQN (gamma=0.95)']:.1f} vs {final['no target network']:.1f}")
    check("6d. full DQN >= tiny replay buffer", final["full DQN (gamma=0.95)"] >= final["tiny replay buffer (=batch size)"],
          f"{final['full DQN (gamma=0.95)']:.1f} vs {final['tiny replay buffer (=batch size)']:.1f}")


# ----------------------------------------------------------------------------
# Test 7: small edge cases (buffer wrap-around, preprocessing, frame stack, action count)
# ----------------------------------------------------------------------------
def test_edge_cases():
    buf = ReplayBuffer(5, (1,), np.float32)
    for i in range(8):
        buf.add(np.array([i]), 0, 0.0, np.array([i]), 0.0, i)
    check("7a. replay buffer overwrites oldest", buf.size == 5 and sorted(buf.s[:, 0].tolist()) == [3, 4, 5, 6, 7], str(sorted(buf.s[:, 0].tolist())))
    try:
        from envs import make_env, preprocess_observation
        env = make_env("pacman")
        obs, _ = env.reset(seed=0)
        frame = preprocess_observation(env.env.unwrapped.ale.getScreenRGB())
        check("7b. pacman preprocessing", frame.shape == (88, 80, 1) and frame.dtype == np.int8 and frame.min() >= -128,
              f"shape {frame.shape}, dtype {frame.dtype}, range [{frame.min()}, {frame.max()}]")
        check("7c. frame stack at reset repeats first frame", all((obs[0] == obs[i]).all() for i in range(1, 4)), f"stack shape {obs.shape}")
        obs2, *_ = env.step(env.action_space.sample())
        check("7d. frame stack shifts by one frame per step", (obs2[:3] == obs[1:]).all(), "obs'[0:3] == obs[1:4]")
        check("7e. MsPacman has 9 actions", env.action_space.n == 9, str(env.action_space.n))
    except Exception as e:                                   # ale-py not installed on this machine
        log(f"[SKIP] 7b-e pacman checks ({e})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=None, help="trained CartPole weights for the policy-map test")
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--ablation_episodes", type=int, default=400)
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    test_chain_matches_value_iteration()
    test_terminal_masking()
    test_truncation_handling()
    test_first_10_steps()
    test_edge_cases()
    if args.weights:
        test_cartpole_policy_map(args.weights)
    if args.ablation:
        test_ablations(range(args.seeds), args.ablation_episodes)
    with open(f"{OUT}/validation_report.txt", "w") as f:
        f.write("\n".join(report))
    print("\nwrote", f"{OUT}/validation_report.txt")
