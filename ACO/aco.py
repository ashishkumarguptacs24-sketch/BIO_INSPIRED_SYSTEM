import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ==========================================
# 1. SETUP TELECOMMUNICATION NETWORK TOPOLOGY
# ==========================================
# Defining a sample 6-router mesh topology
# Nodes represent routers, edges represent physical fiber links with latencies (distance)
nodes = [0, 1, 2, 3, 4, 5]
start_node = 0
destination_node = 5

# Distance matrix representing latency (ms) between routers
# infinity (inf) implies no direct link
inf = float("inf")
distance_matrix = [
    [0, 10, 15, inf, inf, inf],  # Router 0
    [10, 0, inf, 12, 25, inf],  # Router 1
    [15, inf, 0, 10, inf, inf],  # Router 2
    [inf, 12, 10, 0, 8, 14],  # Router 3
    [inf, 25, inf, 8, 0, 5],  # Router 4
    [inf, inf, inf, 14, 5, 0],  # Router 5
]

num_nodes = len(nodes)

# ==========================================
# 2. ALGORITHM HYPERPARAMETERS (From Notes)
# ==========================================
alpha = 1.0  # Pheromone importance
beta = 2.0  # Heuristic importance (visibility)
Q = 100.0  # Pheromone deposit factor
evaporation_rate = 0.3  # Rho (evaporation coefficient)
tau_0 = 0.1  # Initial small pheromone value
num_ants = 15
max_iterations = 50

# Initialize pheromone matrix uniformly across all valid edges
pheromone_matrix = [
    [
        (tau_0 if distance_matrix[i][j] != inf and i != j else 0.0)
        for j in range(num_nodes)
    ]
    for i in range(num_nodes)
]


# Helper to calculate path latency cost
def get_cost(path):
    if not path or path[-1] != destination_node:
        return float("inf")
    total_cost = 0
    for i in range(len(path) - 1):
        total_cost += distance_matrix[path[i]][path[i + 1]]
    return total_cost


# Helper for roulette wheel selection
def select_via_roulette_wheel(probabilities):
    r = random.random()
    cumulative = 0.0
    for idx, prob in enumerate(probabilities):
        cumulative += prob
        if r <= cumulative:
            return idx
    return len(probabilities) - 1


# ==========================================
# 3. CORE ACO ROUTING PROCEDURE
# ==========================================
best_path = None
best_cost = float("inf")
convergence_history = []

print("Starting Ant Colony Optimization Routing Protocol...\n")

for iteration in range(1, max_iterations + 1):
    solutions = []

    for ant in range(num_ants):
        current = start_node
        path = [current]
        visited = {current}

        # Keep routing until packet reaches destination
        while current != destination_node:
            # Get unvisited adjacent neighbors
            neighbors = []
            for n in range(num_nodes):
                if (
                    distance_matrix[current][n] != inf
                    and current != n
                    and n not in visited
                ):
                    neighbors.append(n)

            # Dead-end check (Drop packet/Deadlock state)
            if not neighbors:
                break

            scores = []
            for n in neighbors:
                # visibility = 1 / distance (latency)
                visibility = 1.0 / distance_matrix[current][n]
                # score = pheromone^alpha * visibility^beta
                score = (pheromone_matrix[current][n] ** alpha) * (
                    visibility**beta
                )
                scores.append(score)

            sum_scores = sum(scores)

            # If scores are zero, distribute probabilities equally
            if sum_scores == 0:
                probabilities = [1.0 / len(neighbors)] * len(neighbors)
            else:
                probabilities = [score / sum_scores for score in scores]

            # Select next hop using roulette wheel selection
            selected_index = select_via_roulette_wheel(probabilities)
            next_node = neighbors[selected_index]

            path.append(next_node)
            visited.add(next_node)
            current = next_node

        # Only count successful deliveries to destination
        if path[-1] == destination_node:
            solutions.append(path)
            path_cost = get_cost(path)

            if path_cost < best_cost:
                best_path = path
                best_cost = path_cost

    # --- Pheromone Evaporation ---
    for i in range(num_nodes):
        for j in range(num_nodes):
            if distance_matrix[i][j] != inf:
                pheromone_matrix[i][j] *= 1 - evaporation_rate

    # --- Pheromone Reinforcement ---
    for path in solutions:
        path_cost = get_cost(path)
        if path_cost > 0:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                # Pheromone accumulation
                pheromone_matrix[u][v] += Q / path_cost
                pheromone_matrix[v][u] += Q / path_cost  # Symmetric update

    convergence_history.append(best_cost)
    if iteration % 10 == 0 or iteration == 1:
        print(
            f"Iteration {iteration:02d} | Best Path Found: {best_path} | Cost: {best_cost}ms"
        )

# ==========================================
# 4. PLOT VISUALIZATION RESULTS
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Optimization Convergence Curve
axes[0].plot(
    range(1, max_iterations + 1),
    convergence_history,
    color="darkblue",
    linewidth=2,
    marker="o",
    markersize=4,
)
axes[0].set_title("ACO Routing Convergence Curve", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Iteration Sequence", fontsize=11)
axes[0].set_ylabel("Optimal Network Latency Cost (ms)", fontsize=11)
axes[0].grid(True, linestyle="--", alpha=0.6)

# Plot 2: Telecommunications Node Mesh Network Graph
G = nx.Graph()
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        if distance_matrix[i][j] != inf:
            G.add_edge(i, j, weight=distance_matrix[i][j])

pos = nx.spring_layout(G, seed=42)

# Draw base background topology
nx.draw_networkx_nodes(
    G, pos, ax=axes[1], node_size=700, node_color="lightblue", edgecolors="black"
)
nx.draw_networkx_edges(
    G, pos, ax=axes[1], width=1.5, alpha=0.5, edge_color="gray", style="dashed"
)

# Highlight calculated best data packet path in solid red lines
if best_path:
    best_edges = [(best_path[i], best_path[i + 1]) for i in range(len(best_path) - 1)]
    nx.draw_networkx_edges(
        G, pos, ax=axes[1], edgelist=best_edges, width=3.5, edge_color="crimson"
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=axes[1],
        nodelist=[start_node, destination_node],
        node_size=850,
        node_color="salmon",
        edgecolors="darkred",
    )

nx.draw_networkx_labels(G, pos, ax=axes[1], font_size=12, font_weight="bold")
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, ax=axes[1], edge_labels=edge_labels, font_size=10)

axes[1].set_title(
    f"Optimized Packet Route Link Highlighted (Latency: {best_cost}ms)",
    fontsize=13,
    fontweight="bold",
)
axes[1].axis("off")

plt.tight_layout()
plt.show()
