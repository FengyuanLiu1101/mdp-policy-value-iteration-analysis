import numpy as np
import matplotlib.pyplot as plt



EPSILON = 1e-6
MAX_ITERS = 200000

# State encoding
S_HL, S_HH, S_LL, S_LH = 0, 1, 2, 3
STATE_NAMES = ["H,L", "H,H", "L,L", "L,H"]

# Action names
A_GRID = "UseGrid"
A_BATT = "UseBattery"
A_SHED = "ShedLoad"
ACTIONS = [A_GRID, A_BATT, A_SHED]

# Price switch probability
P_SWITCH = 0.3
P_STAY = 1.0 - P_SWITCH

# Battery transition parameters
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
    # Costs depend on action, and for Grid also on current price
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
            # already low; stays low
            return {"L": 1.0}

    if action == A_GRID:
        if curr_batt == "L":
            return {"H": P_LOW_TO_HIGH_IF_GRID, "L": 1.0 - P_LOW_TO_HIGH_IF_GRID}
        else:
            # already high; stays high
            return {"H": 1.0}

    raise ValueError("Unknown action")


def transitions(curr_state: int, action: str):
    """
    Return list of (prob, next_state, reward).
    We assume independence: P(next) = P(next_batt)*P(next_price)
    Reward = action_cost(curr_state, action) + (-1 if next_batt == 'L' else 0)
    """
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


def value_iteration(gamma: float, epsilon: float = EPSILON, max_iters: int = MAX_ITERS):
    V = np.zeros(4, dtype=float)

    # scaled threshold (common VI stopping criterion)
    threshold = epsilon * (1.0 - gamma) / gamma if gamma > 0 else epsilon

    for it in range(1, max_iters + 1):
        delta = 0.0
        V_new = V.copy()
        for s in range(4):
            best = -float("inf")
            for a in ACTIONS:
                best = max(best, q_value(V, s, a, gamma))
            V_new[s] = best
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < threshold:
            break

    # greedy policy from V
    pi = {}
    for s in range(4):
        best_a, best_q = None, -float("inf")
        for a in ACTIONS:
            q = q_value(V, s, a, gamma)
            if q > best_q:
                best_q, best_a = q, a
        pi[s] = best_a

    return V, pi, it


def run_sweep():
    gammas = np.round(np.arange(0.2, 0.951, 0.05), 2)

    tracked_state = S_LH
    tracked_label = "V*(L,H)"

    values = []
    iters = []
    policy_actions = []  # action per state per gamma

    for g in gammas:
        V, pi, n_it = value_iteration(g)
        values.append(V[tracked_state])
        iters.append(n_it)
        policy_actions.append([pi[s] for s in range(4)])

    # Plot 1: gamma vs tracked V*
    plt.figure()
    plt.plot(gammas, values, marker="o")
    plt.xlabel("Discount factor (gamma)")
    plt.ylabel(tracked_label)
    plt.title(f"Value Iteration (1b): gamma vs {tracked_label}")
    plt.grid(True, alpha=0.3)
    plt.savefig("vi_b_gamma_vs_value.png", dpi=200, bbox_inches="tight")

    # Plot 2: gamma vs VI iterations
    plt.figure()
    plt.plot(gammas, iters, marker="o")
    plt.xlabel("Discount factor (gamma)")
    plt.ylabel("Iterations to converge (sweeps)")
    plt.title("Value Iteration (1b): gamma vs Convergence Iterations")
    plt.grid(True, alpha=0.3)
    plt.savefig("vi_b_gamma_vs_iters.png", dpi=200, bbox_inches="tight")

    # Print summary table
    print("=== Assignment 1b: Value Iteration Summary ===")
    for g, v, n_it, acts in zip(gammas, values, iters, policy_actions):
        act_str = ", ".join([f"{STATE_NAMES[s]}->{acts[s]}" for s in range(4)])
        print(f"gamma={g:>4} | {tracked_label}={v:>10.4f} | iters={n_it:>6} | {act_str}")

    print("\nSaved: vi_b_gamma_vs_value.png, vi_b_gamma_vs_iters.png")


if __name__ == "__main__":
    run_sweep()