import random
from concurrent.futures import ProcessPoolExecutor

# 1. Power System & Generator Data (3-Generator Test System)
# Format per generator: (P_min_MW, P_max_MW, a, b, c)
# Fuel Cost C(P) = a*P^2 + b*P + c  ($/hr)
GENERATORS = [
    (100.0, 600.0, 0.001562, 7.92, 561.0),  # Generator 1
    (100.0, 400.0, 0.001940, 7.85, 310.0),  # Generator 2
    (50.0,  200.0, 0.004820, 7.97, 78.0),   # Generator 3
]
TOTAL_DEMAND_MW = 850.0


# 2. Fitness Function f(x): Minimize Fuel Cost + Power Balance Penalty
def evaluate_dispatch(powers):
    total_cost = 0.0
    total_power = 0.0

    for p, (p_min, p_max, a, b, c) in zip(powers, GENERATORS):
        # Enforce generator operating limits
        if p < p_min or p > p_max:
            return -1e9
        total_cost += a * (p ** 2) + b * p + c
        total_power += p

    # Penalty for mismatch between generated power and grid demand (850 MW)
    power_mismatch = abs(total_power - TOTAL_DEMAND_MW)
    penalty = 1000.0 * (power_mismatch ** 2)

    return -(total_cost + penalty)


# 3. Helper Functions
def repair_power_balance(powers):
    """Adjusts generator outputs proportionally so P1 + P2 + P3 == TOTAL_DEMAND_MW."""
    adjusted = list(powers)
    for _ in range(10):
        diff = TOTAL_DEMAND_MW - sum(adjusted)
        if abs(diff) < 1e-6:
            break
        share = diff / len(adjusted)
        for i, (p_min, p_max, _, _, _) in enumerate(GENERATORS):
            adjusted[i] = max(p_min, min(p_max, adjusted[i] + share))
    return adjusted


def get_neighbors(grid, r, c, neighborhood_type="Moore"):
    rows = len(grid)
    cols = len(grid)
    neighbors = []

    if neighborhood_type == "VonNeumann":
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    for dr, dc in offsets:
        nr = (r + dr) % rows
        nc = (c + dc) % cols
        neighbors.append(grid[nr][nc])

    return neighbors


def update_rule(current_state, best_neighbor_state):
    new_state = []
    for cur, best_nb, (p_min, p_max, _, _, _) in zip(
        current_state, best_neighbor_state, GENERATORS
    ):
        alpha = random.uniform(0.3, 0.7)
        mutation = random.gauss(0.0, 2.0)
        val = cur + alpha * (best_nb - cur) + mutation
        val = max(p_min, min(p_max, val))
        new_state.append(val)

    return repair_power_balance(new_state)


# 4. Main Parallel Cellular Algorithm
def parallel_cellular_eld(rows=5, cols=5, max_iter=30, neighborhood="Moore", seed=42):
    random.seed(seed)

    # Initialize Grid with valid random power schedules
    grid = [[None for _ in range(cols)] for _ in range(rows)]
    initial_states = []
    for _ in range(rows * cols):
        raw = [random.uniform(p_min, p_max) for (p_min, p_max, _, _, _) in GENERATORS]
        initial_states.append(repair_power_balance(raw))

    with ProcessPoolExecutor() as executor:
        initial_fitness = list(executor.map(evaluate_dispatch, initial_states))

        idx = 0
        for r in range(rows):
            for c in range(cols):
                grid[r][c] = {
                    "state": initial_states[idx],
                    "fitness": initial_fitness[idx],
                }
                idx += 1

        best_global = max(
            (cell for row in grid for cell in row), key=lambda x: x["fitness"]
        ).copy()

        # Main Evolution Loop
        for t in range(max_iter):
            candidate_states = []
            for r in range(rows):
                for c in range(cols):
                    neighbors = get_neighbors(grid, r, c, neighborhood)
                    best_neighbor = max(neighbors, key=lambda x: x["fitness"])
                    new_state = update_rule(
                        grid[r][c]["state"], best_neighbor["state"]
                    )
                    candidate_states.append(new_state)

            candidate_fitness = list(executor.map(evaluate_dispatch, candidate_states))

            # Synchronous Grid Update
            next_grid = [[None for _ in range(cols)] for _ in range(rows)]
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if candidate_fitness[idx] > grid[r][c]["fitness"]:
                        next_grid[r][c] = {
                            "state": candidate_states[idx],
                            "fitness": candidate_fitness[idx],
                        }
                    else:
                        next_grid[r][c] = grid[r][c].copy()
                    idx += 1

            grid = next_grid
            current_best = max(
                (cell for row in grid for cell in row), key=lambda x: x["fitness"]
            )
            if current_best["fitness"] > best_global["fitness"]:
                best_global = current_best.copy()

            p1, p2, p3 = best_global["state"]
            print(
                f"Iter {t + 1:02d} | Fuel Cost: ${-best_global['fitness']:8.2f}/hr | "
                f"Dispatch (MW) -> G1: {p1:6.2f}, G2: {p2:6.2f}, G3: {p3:6.2f}"
            )

    return best_global


if __name__ == "__main__":
    best = parallel_cellular_eld(rows=5, cols=5, max_iter=30)
    p1, p2, p3 = best["state"]
    print("\n--- Optimal Smart Grid Dispatch ---")
    print(f"Generator 1: {p1:.2f} MW")
    print(f"Generator 2: {p2:.2f} MW")
    print(f"Generator 3: {p3:.2f} MW")
    print(f"Total Power: {p1 + p2 + p3:.2f} MW (Demand: {TOTAL_DEMAND_MW} MW)")
    print(f"Minimum Fuel Cost: ${-best['fitness']:.2f}/hr")
