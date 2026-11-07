# MLDS 490 – Assignment 2: Deep Q-Network (CartPole-v1, MsPacman-v5)

Simple PyTorch DQN following the Week 6 slides: neural network + replay buffer + target network, Huber loss,
epsilon-greedy with a linear schedule, learning-rate decay, gradient clipping.

    y_j = r_j + gamma * max_a' Q_target(s'_j, a')      (y_j = r_j if s'_j is a terminal state)
    loss = Huber(Q_phi(s_j, a_j) - y_j),  target network updated every N steps (or Polyak average, slide 14)

| file | purpose |
|---|---|
| `dqn.py` | networks (MLP / CNN), replay buffer, target + loss, training loop, evaluation rollout |
| `envs.py` | CartPole-v1, MsPacman-v5 with the assignment's `preprocess_observation` + 4-frame stack |
| `train.py` | `python train.py --env cartpole|pacman` (hyper-parameters at the top of the file) |
| `evaluate.py` | 500-episode rollout, reward histogram, mean and std |
| `plot_results.py` | max-Q plot, reward + moving-average plot, replay-buffer-age plot |
| `validation_tests.py` | validation experiments (see below) |

Hyper-parameters: CartPole gamma = 0.95, MsPacman gamma = 0.99 (as required). MsPacman rewards are clipped to
{-1, 0, 1} **for training only**; all reported/plotted rewards are the raw game score. Max-Q for MsPacman is therefore
in clipped-reward units.

## Setup (deepdish)

    ssh <netid>@mlds-deepdishX.ads.northwestern.edu
    git clone <repo-url> && cd <repo>/dqn_project
    python3 -m venv ~/dqn_env && source ~/dqn_env/bin/activate      # or use conda
    pip install -r requirements.txt
    nvidia-smi                                                       # pick a GPU nobody is using
    export CUDA_VISIBLE_DEVICES=0                                    # single GPU only (assignment rule)

`ale-py >= 0.9` ships the Atari ROMs, so no separate ROM license step is needed.

## Run

    # CartPole (about 5-10 min)
    python train.py --env cartpole
    python evaluate.py --env cartpole --weights results/cartpole_seed0_best.pt
    python plot_results.py --env cartpole --logs results/cartpole_seed0_logs.npz

    # MsPacman (hours) - run detached so it survives a dropped SSH session
    nohup python train.py --env pacman --episodes 5000 > pacman_train.log 2>&1 &
    tail -f pacman_train.log
    python evaluate.py --env pacman --weights results/pacman_seed0_best.pt --eps 0.05
    python plot_results.py --env pacman --logs results/pacman_seed0_logs.npz

Outputs go to `results/`: `*_maxq.png` (plot i), `*_rewards.png` (plot ii), `*_eval_hist.png` (plot iii),
`*_buffer_age.png` (non-standard plot).

CartPole checkpoint selection: every 20 episodes the current network plays 10 greedy validation episodes (separate seeds)
and the best one is saved as `*_best.pt`. DQN on CartPole oscillates a lot (see the reward plot), so the final network
is often much worse than the best one; we evaluate the validation-selected one and also report `*_last.pt`.
MsPacman keeps the checkpoint with the best 100-episode moving average.

## Validation / edge-case experiments

    python validation_tests.py                                              # tests 1-4, 7, ~1 min
    python validation_tests.py --weights results/cartpole_seed0_best.pt     # + test 5
    python validation_tests.py --ablation --seeds 3 --ablation_episodes 400 # + test 6, ~15 min

| # | what it does | what we expect / why it validates |
|---|---|---|
| 1 | Train on a 5-state deterministic chain MDP and compare Q with exact value iteration | DQN's Q must equal Q* (max error < 0.08) and the greedy policy must go right |
| 2 | MDP that terminates after one step, reward 1 | Q must be 1, not 1/(1-gamma): checks the `(1 - done)` mask |
| 3 | MDP with a time limit (truncation), reward 1/step | correct handling gives Q = 1/(1-gamma) = 10; treating truncation as terminal gives the analytically predicted 6.9 |
| 4 | First 10 CartPole steps: print Q, action, reward, target; recompute target and Huber loss by hand | code and hand computation agree |
| 5 | CartPole policy map Q(right)-Q(left) over (theta, omega) + agreement with "push toward the lean" on visited states | agreement well above 50% chance |
| 6 | Ablations (no target net, tiny replay buffer, gamma 0.99), 3 seeds x 400 episodes | max Q bounded by ~1/(1-gamma) (20 vs 100); full DQN expected to beat ablated variants (see findings) |
| 7 | Buffer wrap-around, preprocessing shape/dtype/range, frame-stack reset/shift, 9 actions | edge cases |

**Findings from the local run (CPU, 3 seeds, 400 episodes; rerun on deepdish for the report):**
* 6a/6b PASS: max Q is about 24.7 for gamma = 0.95 (bound 20) and about 131.6 for gamma = 0.99 (bound 100); the ceiling moves with gamma as theory predicts, with some overestimation.
* 6c PASS: without a target network Q **diverges** (max Q about 4.7 million) and reward collapses to about 12; with it, Q stays near the bound.
* 6d **FAIL** (kept on purpose): the tiny replay buffer scored higher at 400 episodes (184 vs 125). Our hypothesis (a big buffer helps) was not
  supported at this short horizon. A plausible but untested explanation: a small buffer is more on-policy and learns faster early, while a large
  buffer's benefit (stability, remembering failures) only shows over longer training. A longer, more-seeds comparison is left as follow-up.

Mutation check (done during development): deliberately removing the done-mask, removing gamma, or using min instead of
max in the target makes tests 1/2 fail, so they can detect real bugs.

### Edge cases considered
* **Terminal vs time-limit.** CartPole-v1 truncates at 500 steps. Only `terminated` stores `done = 1`; truncation still bootstraps (test 3).
* **Huber hides that bug.** With Huber loss, rare terminal targets are "outliers" and Q fits the median target, so a wrong
  truncation flag barely changes Q in test 3 (9.97 vs 10); with MSE the error is visible (7.12 vs 10). Handling it correctly matters regardless of the loss.
* **Q-value upper bound.** With +1 reward per step, Q <= 1/(1-gamma) = 20 for gamma = 0.95. Observed max-Q slightly above 20 (about 20.2-20.7) = mild overestimation from the max operator.
* **Cart drift.** The greedy CartPole policy drifts toward the track edge (median x about -2.05, limit 2.4), which leaves little margin. We have not yet measured how many failed episodes end by cart-out-of-bounds vs pole-fall (worth logging the termination cause). gamma = 0.95 gives only a ~20-step horizon, which is short for anticipating drift.
* **Frame stack at reset** repeats the first frame (no history exists); frames are shifted by exactly one per step (test 7).
* **Replay buffer** overwrites oldest entries when full; learning starts only after `learning_starts` transitions.
* **MsPacman sticky actions / lives.** v5 uses 25% sticky actions; the episode ends only at game over (not on losing a life).
  Rollouts use eps = 0.05 so the agent does not get stuck repeating one action.

### Non-standard plot
`*_buffer_age.png`: mean age (in env steps) of the replay tuples sampled in each episode. It shows how stale the data is
that the network trains on (grows until the buffer is full, then plateaus around half the buffer size).
`results/validation/cartpole_policy_map.png`: heat map of Q(right) - Q(left) over pole angle and angular velocity.

## If we had 3 additional months
* 10+ seeds with confidence intervals for every claim; hyper-parameter sweeps (lr, target update, buffer size, epsilon schedule).
* Double DQN, dueling heads, prioritized replay, n-step returns, Rainbow-style ablations (the slides name these as helpful).
* Tabular-vs-network comparison on larger gridworlds with exact Q*; measure Q-overestimation bias directly against Monte-Carlo returns.
* Study the Huber vs MSE interaction with terminal/truncated targets and reward clipping/scaling on MsPacman.
* Replay-buffer diagnostics: composition by episode quality, forgetting of early failure states (the cause of CartPole oscillations); try reserving buffer capacity for failure states.
* Robustness: evaluate under different sticky-action probabilities, frame-skip values, and random starts; check generalisation across game levels.
* Interpretability for MsPacman: saliency maps of the CNN, Q-value traces around ghost encounters, human-score baseline.
