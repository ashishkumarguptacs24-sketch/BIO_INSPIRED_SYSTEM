import numpy as np
import math
import matplotlib.pyplot as plt

# =============================================================================
# 1. OPTIMIZATION PROBLEM & LÉVY FLIGHT UTILITIES
# =============================================================================
BOUNDS = np.array([[200.0, 500.0], [140.0, 220.0], [800.0, 1200.0], [600.0, 900.0]])

def objective_function(x):
    cement, water, coarse, fine = x[0], x[1], x[2], x[3]
    if not (2200 <= sum(x) <= 2500) or not (0.35 <= water / cement <= 0.60):
        return 1e6
    strength = (25 + 0.1 * (cement - 200) - 0.25 * (water - 140) 
                - 0.0005 * (coarse - 1000)**2 - 0.0008 * (fine - 750)**2)
    return -strength 

def levy_flight(Lambda, size):
    num = math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2)
    den = math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2)
    sigma = (num / den) ** (1 / Lambda)
    return np.random.normal(0, sigma, size) / (np.abs(np.random.normal(0, 1, size)) ** (1 / Lambda))

# =============================================================================
# 2. CUCKOO SEARCH IMPLEMENTATION
# =============================================================================
def cuckoo_search(pop_size=30, max_iter=150, pa=0.25, beta=1.5):
    dim = len(BOUNDS)
    nests = np.random.uniform(BOUNDS[:, 0], BOUNDS[:, 1], (pop_size, dim))
    fitness = np.array([objective_function(nest) for nest in nests])
    best_nest = nests[np.argmin(fitness)].copy()
    best_fitness = np.min(fitness)
    
    history = []  # Tracks maximum concrete strength over time

    for t in range(max_iter):
        # Step 1: Lévy Flight Exploration
        new_nests = nests.copy()
        for i in range(pop_size):
            step_size = 0.01 * levy_flight(beta, dim) * (nests[i] - best_nest)
            new_nests[i] += np.random.normal(0, 1, dim) * step_size
            new_nests[i] = np.clip(new_nests[i], BOUNDS[:, 0], BOUNDS[:, 1])
            
        # Step 2: Greedy Local Selection
        for i in range(pop_size):
            f_new = objective_function(new_nests[i])
            r_idx = np.random.randint(0, pop_size)
            if f_new < fitness[r_idx]:
                fitness[r_idx], nests[r_idx] = f_new, new_nests[i].copy()

        # Step 3: Nest Abandonment (pa)
        idx_sorted = np.argsort(fitness)
        nests, fitness = nests[idx_sorted], fitness[idx_sorted]
        for i in range(pop_size - int(pa * pop_size), pop_size):
            r1, r2 = np.random.randint(0, pop_size, 2)
            nests[i] = np.clip(nests[i] + np.random.rand() * (nests[r1] - nests[r2]), BOUNDS[:, 0], BOUNDS[:, 1])
            fitness[i] = objective_function(nests[i])
            
        # Update Global Best
        if np.min(fitness) < best_fitness:
            best_fitness, best_nest = np.min(fitness), nests[np.argmin(fitness)].copy()
            
        history.append(-best_fitness) 

    return best_nest, -best_fitness, history

# =============================================================================
# 3. RUN & GENERATE PLOT
# =============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    _, max_str, convergence_history = cuckoo_search()

    plt.figure(figsize=(8, 5))
    plt.plot(convergence_history, color='darkorange', linewidth=2.5, label='Best Nest Performance')
    plt.title('Cuckoo Search Convergence Curve (Concrete Mixture Optimization)', fontsize=12, fontweight='bold')
    plt.xlabel('Iteration / Generation Count', fontsize=10)
    plt.ylabel('Concrete Compressive Strength (MPa)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(y=max_str, color='green', linestyle=':', label=f'Optimal Peak Strength: {max_str:.2f} MPa')
    plt.legend(loc='lower right')
    plt.show()
