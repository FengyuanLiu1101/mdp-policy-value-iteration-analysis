import numpy as np
import matplotlib.pyplot as plt


EPSILON = 1e-6  
MAX_EVAL_SWEEPS = 200000
MAX_POLICY_ITERS = 10000

S_OFFICE, S_MAIN, S_SHORT, S_DEST = 0, 1, 2, 3
STATE_NAMES = ["Office", "MainPath", "ShortcutPath", "Destination"]

A_MAINPATH = "MainPath"
A_SHORTCUT = "Shortcut"
A_FORWARD = "MoveForward"
A_NOOP = "NoOp"


def actions(state: int):
    if state == S_OFFICE:
        return [A_MAINPATH, A_SHORTCUT]
    if state == S_MAIN:
        return [A_FORWARD]
    if state == S_SHORT:
        return [A_FORWARD]
    if state == S_DEST:
        return [A_NOOP]
    raise ValueError("Unknown state")


def transitions(state: int, action: str):
    if state == S_DEST:
        return [(1.0, S_DEST, 0.0)]

    def r(next_state):
        return -1.0 + (10.0 if next_state == S_DEST else 0.0)

    if state == S_OFFICE and action == A_MAINPATH:
        return [(1.0, S_MAIN, r(S_MAIN))]
    if state == S_OFFICE and action == A_SHORTCUT:
        return [(1.0, S_SHORT, r(S_SHORT))]

    if state == S_MAIN and action == A_FORWARD:
        return [
            (0.9, S_DEST, r(S_DEST)),
            (0.1, S_MAIN, r(S_MAIN)),
        ]

    if state == S_SHORT and action == A_FORWARD:
        return [
            (0.6, S_DEST, r(S_DEST)),
            (0.4, S_OFFICE, r(S_OFFICE)),
        ]

    raise ValueError(f"Invalid (state, action): {state}, {action}")


def q_value(V: np.ndarray, state: int, action: str, gamma: float) -> float:
    q = 0.0
    for p, ns, rew in transitions(state, action):
        q += p * (rew + gamma * V[ns])
    return q


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
            if s == S_DEST:
                V_new[s] = 0.0
                continue
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
      V*: value under optimal policy
      pi*: optimal policy
      policy_iters: number of outer improvement steps
      total_eval_sweeps: total sweeps used across evaluation phases
    """
    # Initialize a deterministic policy:
    # Office -> MainPath (reasonable starting point)
    pi = {
        S_OFFICE: A_MAINPATH,
        S_MAIN: A_FORWARD,
        S_SHORT: A_FORWARD,
        S_DEST: A_NOOP,
    }

    total_eval_sweeps = 0

    for outer_it in range(1, max_policy_iters + 1):
        V, eval_sweeps = policy_evaluation(pi, gamma, epsilon=epsilon)
        total_eval_sweeps += eval_sweeps

        policy_stable = True

        for s in range(4):
            if s == S_DEST:
                continue

            old_a = pi[s]
            best_a, best_q = None, -float("inf")
            for a in actions(s):
                q = q_value(V, s, a, gamma)
                if q > best_q:
                    best_q, best_a = q, a

            pi[s] = best_a
            if best_a != old_a:
                policy_stable = False

        if policy_stable:
            return V, pi, outer_it, total_eval_sweeps

    return V, pi, max_policy_iters, total_eval_sweeps


def run_sweep():
    gammas = np.round(np.arange(0.2, 0.91, 0.05), 2)

    office_values = []
    outer_iters_list = []
    eval_sweeps_list = []
    chosen_action_at_office = []

    for g in gammas:
        V, pi, outer_iters, total_eval_sweeps = policy_iteration(g)
        office_values.append(V[S_OFFICE])
        outer_iters_list.append(outer_iters)
        eval_sweeps_list.append(total_eval_sweeps)
        chosen_action_at_office.append(pi[S_OFFICE])

    # Plot 1: gamma vs V*(Office)
    plt.figure()
    plt.plot(gammas, office_values, marker="o")
    plt.xlabel("Discount factor (gamma)")
    plt.ylabel("Optimal Value V*(Office)")
    plt.title("Policy Iteration: gamma vs V*(Office)")
    plt.grid(True, alpha=0.3)
    plt.savefig("pi_gamma_vs_value.png", dpi=200, bbox_inches="tight")

    # Plot 2A: gamma vs outer policy improvement iterations
    plt.figure()
    plt.plot(gammas, outer_iters_list, marker="o")
    plt.xlabel("Discount factor (gamma)")
    plt.ylabel("Policy improvement iterations (outer loop)")
    plt.title("Policy Iteration: gamma vs Policy Improvement Iterations")
    plt.grid(True, alpha=0.3)
    plt.savefig("pi_gamma_vs_policy_iters.png", dpi=200, bbox_inches="tight")

    # Plot 2B : gamma vs total evaluation sweeps
    plt.figure()
    plt.plot(gammas, eval_sweeps_list, marker="o")
    plt.xlabel("Discount factor (gamma)")
    plt.ylabel("Total policy evaluation sweeps")
    plt.title("Policy Iteration: gamma vs Total Evaluation Sweeps")
    plt.grid(True, alpha=0.3)
    plt.savefig("pi_gamma_vs_eval_sweeps.png", dpi=200, bbox_inches="tight")

    # Print summary
    print("=== Policy Iteration Summary ===")
    for g, v0, o_it, e_sw, act in zip(gammas, office_values, outer_iters_list, eval_sweeps_list, chosen_action_at_office):
        print(f"gamma={g:>4} | V*(Office)={v0:>8.4f} | outer_iters={o_it:>3} | eval_sweeps={e_sw:>6} | pi(Office)={act}")


if __name__ == "__main__":
    run_sweep()
    print("\nSaved: pi_gamma_vs_value.png, pi_gamma_vs_policy_iters.png, pi_gamma_vs_eval_sweeps.png")