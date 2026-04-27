import numpy as np
import matplotlib.pyplot as plt


EPSILON = 1e-6  # be consistent across all gammas and algorithms
MAX_ITERS = 100000

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
    """
    Returns list of (prob, next_state, reward).
    """
    if state == S_DEST:
        # terminal
        return [(1.0, S_DEST, 0.0)]

    # reward function for non-terminal actions:
    # base step cost -1, and +10 if landing in Destination
    def r(next_state):
        return -1.0 + (10.0 if next_state == S_DEST else 0.0)

    if state == S_OFFICE and action == A_MAINPATH:
        # go to Main Path deterministically
        return [(1.0, S_MAIN, r(S_MAIN))]
    if state == S_OFFICE and action == A_SHORTCUT:
        # go to Shortcut deterministically
        return [(1.0, S_SHORT, r(S_SHORT))]

    if state == S_MAIN and action == A_FORWARD:
        # reaches Destination 0.9, stays on Main 0.1
        return [
            (0.9, S_DEST, r(S_DEST)),
            (0.1, S_MAIN, r(S_MAIN)),
        ]

    if state == S_SHORT and action == A_FORWARD:
        # reaches Destination 0.6, returns to Office 0.4
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


def value_iteration(gamma: float, epsilon: float = EPSILON, max_iters: int = MAX_ITERS):
    """
    Returns:
      V*: optimal state values
      pi*: greedy policy derived from V*
      iters: number of VI sweeps until convergence
    """
    V = np.zeros(4, dtype=float)

    if gamma > 0:
        threshold = epsilon * (1.0 - gamma) / gamma
    else:
        threshold = epsilon

    for it in range(1, max_iters + 1):
        delta = 0.0
        V_new = V.copy()

        for s in range(4):
            if s == S_DEST:
                V_new[s] = 0.0
                continue

            best = -float("inf")
            for a in actions(s):
                best = max(best, q_value(V, s, a, gamma))
            V_new[s] = best
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new

        if delta < threshold:
            break

    # derive greedy policy
    pi = {}
    for s in range(4):
        if s == S_DEST:
            pi[s] = A_NOOP
            continue
        best_a, best_q = None, -float("inf")
        for a in actions(s):
            q = q_value(V, s, a, gamma)
            if q > best_q:
                best_q, best_a = q, a
        pi[s] = best_a

    return V, pi, it


def run_sweep():
    gammas = np.round(np.arange(0.2, 0.91, 0.05), 2)

    office_values = []
    iterations = []
    chosen_action_at_office = []

    for g in gammas:
        V, pi, iters = value_iteration(g)
        office_values.append(V[S_OFFICE])
        iterations.append(iters)
        chosen_action_at_office.append(pi[S_OFFICE])

    # Plot 1: gamma vs V*(Office)
    plt.figure()
    plt.plot(gammas, office_values, marker="o")
    plt.xlabel("Discount factor (gamma)")
    plt.ylabel("Optimal Value V*(Office)")
    plt.title("Value Iteration: gamma vs V*(Office)")
    plt.grid(True, alpha=0.3)
    plt.savefig("vi_gamma_vs_value.png", dpi=200, bbox_inches="tight")

    # Plot 2: gamma vs iterations
    plt.figure()
    plt.plot(gammas, iterations, marker="o")
    plt.xlabel("Discount factor (gamma)")
    plt.ylabel("Iterations to converge (sweeps)")
    plt.title("Value Iteration: gamma vs Convergence Iterations")
    plt.grid(True, alpha=0.3)
    plt.savefig("vi_gamma_vs_iters.png", dpi=200, bbox_inches="tight")


    print("=== Value Iteration Summary ===")
    for g, v0, iters, act in zip(gammas, office_values, iterations, chosen_action_at_office):
        print(f"gamma={g:>4} | V*(Office)={v0:>8.4f} | iters={iters:>5} | pi(Office)={act}")


if __name__ == "__main__":
    run_sweep()
    print("\nSaved: vi_gamma_vs_value.png, vi_gamma_vs_iters.png")