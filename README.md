# MDP Optimization: Policy Iteration vs Value Iteration

## 📌 Overview
This project implements and compares two fundamental dynamic programming algorithms for solving Markov Decision Processes (MDPs):

- Policy Iteration (PI)
- Value Iteration (VI)

The goal is to analyze how different discount factors (γ) affect:
- Optimal value functions
- Policy decisions
- Convergence behavior

---

## 🧠 Problem Description

We model a simplified decision-making problem as an MDP with:

- Finite state space
- Deterministic and stochastic transitions
- Step penalties and terminal rewards

The agent must choose optimal actions to maximize long-term rewards.

---

## ⚙️ Algorithms Implemented

### 🔵 Policy Iteration
- Alternates between:
  - Policy Evaluation
  - Policy Improvement
- Converges when policy becomes stable

### 🟣 Value Iteration
- Iteratively updates value function using Bellman optimality
- Extracts optimal policy after convergence

---

## 📊 Experiments

We evaluate both algorithms across a range of discount factors:

```text
γ ∈ [0.2, 0.9]
Metrics:
Optimal value at initial state
Number of iterations to converge
Policy stability
Evaluation sweeps (for PI)
📈 Results
1. Value Function vs Discount Factor
Higher γ → higher value
Future rewards are weighted more heavily
2. Convergence Behavior
Larger γ → slower convergence
Value Iteration requires more iterations
Policy Iteration needs fewer outer loops but heavier evaluation
3. Policy Stability
Optimal policy stabilizes quickly
Small state space leads to fast convergence
🧪 Sample Output

The following plots are generated:

pi_gamma_vs_value.png
pi_gamma_vs_policy_iters.png
pi_gamma_vs_eval_sweeps.png
vi_gamma_vs_value.png
vi_gamma_vs_iters.png
🏗️ Project Structure
.
├── assignment1a_policy_iteration.py
├── assignment1a_value_iteration.py
├── results/
│   ├── pi_gamma_vs_value.png
│   ├── pi_gamma_vs_policy_iters.png
│   ├── pi_gamma_vs_eval_sweeps.png
│   ├── vi_gamma_vs_value.png
│   └── vi_gamma_vs_iters.png
└── README.md
🚀 How to Run
Option 1: Local Python
python assignment1a_policy_iteration.py
python assignment1a_value_iteration.py
Option 2: Google Colab
Upload .py files
Run directly
📚 Key Insights
Both PI and VI converge to the same optimal policy
Discount factor (γ) significantly affects:
Value magnitude
Convergence speed
PI is computationally efficient for small problems
VI is simpler but may require more iterations
🎯 Learning Outcomes

This project demonstrates:

Fundamentals of MDPs
Bellman equations
Trade-offs between dynamic programming algorithms
Impact of discount factors on decision-making
👤 Author

Fengyuan Liu
Purdue University Northwest
Computer Science / AI

📎 Keywords

MDP, Reinforcement Learning, Policy Iteration, Value Iteration, Dynamic Programming
