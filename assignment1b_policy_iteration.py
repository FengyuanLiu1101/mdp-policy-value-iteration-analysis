import numpy as np
import matplotlib.pyplot as plt


EPSILON = 1e-6
MAX_EVAL_SWEEPS = 500000
MAX_POLICY_ITERS = 10000

S_HL, S_HH, S_LL, S_LH = 0, 1, 2, 3
STATE_NAMES = ["H,L", "H,H", "L,L", "L,H"]

A_GRID = "UseGrid"
A_BATT = "UseBattery"
A_SHED = "ShedLoad"
ACTIONS = [A_GRID, A_BATT, A_SHED]

P_SWITCH = 0.3
P_STAY = 0.7

P_HIGH_TO_LOW_IF_BATT = 0.7
P_LOW_TO_HIGH_IF_GRID = 0.4


def battery_level(state: int) -> str:
    return "H" if state in (S_HL, S_HH) else "L"


def price_level(state: int) -> str:
    return "L" if state in (S_HL, S_LL) else "H"


def next_price_probs(curr_price: str):
    if curr_price == "L":
        return {"L": P_STAY, "H": P_SWITCH}
    else:
        return {"H": P_STAY, "L": P_SWITCH}


def state_from(batt: str, price: str) -> int:
    if batt == "H" and price == "L":
        return S_HL
    if batt == "H" and price == "H":
        return S_HH
    if batt == "L" and price == "L":
        return S_LL
    if batt == "L" and price == "H":
        return S_LH
    raise ValueError("Invalid (batt, price)")


def action_cost(curr_state: int, action: str) -> float:
    p = price_level(curr_state)
    if action == A_GRID:
        return -1.0 if p == "L" else -5.0
    if action == A_BATT:
        return -2.0
    if action == A_SHED:
        return -8.0
    raise ValueError("Unknown action")


def battery_transition_probs(curr_batt: str, action: str):
    if action == A_SHED:
        return {curr_batt: 1.0}

    if action == A_BATT:
        if curr_batt == "H":
            return {"L": P_HIGH_TO_LOW_IF_BATT, "H": 1.0 - P_HIGH_TO_LOW_IF_BATT}
        else:
            return {"L": 1.0}

    if action == A_GRID:
        if curr_batt == "L":
            return {"H": P_LOW_TO_HIGH_IF_GRID, "L": 1.0 - P_LOW_TO_HIGH_IF_GRID}
        else:
            return {"H": 1.0}

    raise ValueError("Unknown action")


def transitions(curr_state: int, action: str):
    curr_batt = battery_level(curr_state)
    curr_price = price_level(curr_state)

    price_probs = next_price_probs(curr_price)
    batt_probs = battery_transition_probs(curr_batt, action)

    base_cost = action_cost(curr_state, action)

    out = []
    for nb, pb in batt_probs.items():
        for np_, pp in price_probs.items():
            ns = state_from(nb, np_)
            extra_low_penalty = -1.0 if nb == "L" else 0.0
            rew = base_cost + extra_low_penalty
            out.append((pb * pp, ns, rew))
    return out


def q_value(V: np.ndarray, state: int, action: str, gamma: float) -> float:
    return sum(p * (r + gamma * V[ns]) for p, ns, r in transitions(state, action))


def policy_evaluation(pi: dict, gamma: float, epsilon: float = EPSILON, max_sweeps: int = MAX_EVAL_SWEEPS):
    """
    Iterative policy evaluation:
      V_{k+1}(s) = sum_{s'} P(s'|s,pi(s)) [ R + gamma V_k(s') ]
    Stop when max change < epsilon
    """
    V = np.zeros(4, dtype=float)
    for sweep in range(1, max_sweeps + 1):
        delta = 0.0
        V_new = V.copy()
        for s in range(4):
            a = pi[s]
            V_new[s] = q_value(V, s, a, gamma)
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < epsilon:
            return V, sweep
    return V, max_sweeps


def policy_iteration(gamma: float, epsilon: float = EPSILON, max_policy_iters: int = MAX_POLICY_ITERS):
    """
    Returns:
      V*, pi*, outer_iters, total_eval_sweeps
    """

    pi = {
        S_HL: A_GRID,
        S_HH: A_BATT,
        S_LL: A_GRID,
        S_LH: A_GRID,
    }

    total_eval_sweeps = 0

    for outer in range(1, max_policy_iters + 1):
        V, sweeps = policy_evaluation(pi, gamma, epsilon=epsilon)
        total_eval_sweeps += sweeps

        stable = True
        for s in range(4):
            old_a = pi[s]
            best_a, best_q = None, -float("inf")
            for a in ACTIONS:
                q = q_value(V, s, a, gamma)
                if q > best_q:
                    best_q, best_a = q, a
            pi[s] = best_a
            if best_a != old_a:
                stable = False

        if stable:
            return V, pi, outer, total_eval_sweeps

    return V, pi, max_policy_iters, total_eval_sweeps


def plot_policy_map(gammas, policy_actions, filename="b_gamma_vs_policy_map.png"):
    """
    Visualize policy changes across gammas as a heatmap-like grid.
    Encode actions to integers for plotting.
    """
    action_to_id = {A_GRID: 0, A_BATT: 1, A_SHED: 2}
    id_to_action = {v: k for k, v in action_to_id.items()}

    mat = np.array([[action_to_id[a] for a in acts] for acts in policy_actions])  # shape: [len(gamma), 4]

    plt.figure(figsize=(10, 4))
    plt.imshow(mat.T, aspect="auto")  # states on y-axis, gammas on x-axis
    plt.yticks(range(4), STATE_NAMES)
    plt.xticks(range(len(gammas)), [str(g) for g in gammas], rotation=45)
    plt.xlabel("Discount factor (gamma)")
    plt.title("Assignment 1b: Optimal Action per State across gamma")
    cbar = plt.colorbar()
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels([id_to_action[0], id_to_action[1], id_to_action[2]])
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")


def run_sweep():
    gammas = np.round(np.arange(0.2, 0.951, 0.05), 2)

    tracked_state = S_LH
    tracked_label = "V*(L,H)"

    values = []
    outer_iters = []
    eval_sweeps = []
    policy_actions = []

    for g in gammas:
        V, pi, out_it, total_sweeps = policy_iteration(g)
        values.append(V[tracked_state])
        outer_iters.append(out_it)
        eval_sweeps.append(total_sweeps)
        policy_actions.append([pi[s] for s in range(4)])

    # Plot 1: gamma vs tracked V*
    plt.figure()
    plt.plot(gammas, values, marker="o")
    plt.xlabel("Discount factor (gamma)")
    plt.ylabel(tracked_label)
    plt.title(f"Policy Iteration (1b): gamma vs {tracked_label}")
    plt.grid(True, alpha=0.3)
    plt.savefig("pi_b_gamma_vs_value.png", dpi=200, bbox_inches="tight")

    # Plot 2: gamma vs policy improvement iterations (outer loop)
    plt.figure()
    plt.plot(gammas, outer_iters, marker="o")
    plt.xlabel("Discount factor (gamma)")
    plt.ylabel("Policy improvement iterations (outer loop)")
    plt.title("Policy Iteration (1b): gamma vs Policy Improvement Iterations")
    plt.grid(True, alpha=0.3)
    plt.savefig("pi_b_gamma_vs_policy_iters.png", dpi=200, bbox_inches="tight")

    # Plot 3: gamma vs total policy evaluation sweeps
    plt.figure()
    plt.plot(gammas, eval_sweeps, marker="o")
    plt.xlabel("Discount factor (gamma)")
    plt.ylabel("Total policy evaluation sweeps")
    plt.title("Policy Iteration (1b): gamma vs Total Evaluation Sweeps")
    plt.grid(True, alpha=0.3)
    plt.savefig("pi_b_gamma_vs_eval_sweeps.png", dpi=200, bbox_inches="tight")

    # Extra credit: policy map across gammas
    plot_policy_map(gammas, policy_actions, filename="b_gamma_vs_policy_map.png")

    # Print summary
    print("=== Assignment 1b: Policy Iteration Summary ===")
    for g, v, out_it, sweeps, acts in zip(gammas, values, outer_iters, eval_sweeps, policy_actions):
        act_str = ", ".join([f"{STATE_NAMES[s]}->{acts[s]}" for s in range(4)])
        print(f"gamma={g:>4} | {tracked_label}={v:>10.4f} | outer_iters={out_it:>4} | eval_sweeps={sweeps:>6} | {act_str}")

    print("\nSaved: pi_b_gamma_vs_value.png, pi_b_gamma_vs_policy_iters.png, pi_b_gamma_vs_eval_sweeps.png, b_gamma_vs_policy_map.png")


if __name__ == "__main__":
    run_sweep()