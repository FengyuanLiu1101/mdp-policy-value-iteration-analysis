# MDP Optimization with Dynamic Programming

## 📌 Overview
This project implements and analyzes two fundamental algorithms for solving Markov Decision Processes (MDPs):

- Policy Iteration (PI)
- Value Iteration (VI)

The project includes two environments of increasing complexity:

- **Assignment 1a**: Simple navigation problem (path selection)
- **Assignment 1b**: Energy management system with stochastic dynamics

The goal is to study:
- Optimal policy behavior
- Convergence properties
- Impact of discount factor (γ)

---

## 🧠 Environments

### 🔵 Assignment 1a — Path Planning MDP
- Small state space
- Deterministic + stochastic transitions
- Trade-off: safe vs risky path

### 🔴 Assignment 1b — Energy Management MDP
- Multi-factor state (battery + price)
- Stochastic transitions
- Real-world inspired decision problem

---

## ⚙️ Algorithms

### Policy Iteration
- Policy Evaluation + Policy Improvement
- Fast convergence in small environments

### Value Iteration
- Bellman optimality updates
- Simpler but may require more iterations

---

## 📊 Key Results

### 1. Discount Factor Impact
- Higher γ → higher value function
- Future rewards are weighted more heavily

### 2. Convergence Behavior
- Higher γ → slower convergence
- VI requires more iterations than PI

### 3. Policy Behavior
- Assignment 1a: policy may change depending on γ
- Assignment 1b: policy is stable across γ (cost-dominated decisions)

---

## 🏗️ Project Structure

```text
.
├── assignment1a/
├── assignment1b/
└── docs/
