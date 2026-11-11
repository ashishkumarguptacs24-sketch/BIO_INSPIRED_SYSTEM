import numpy as np
import math

# =============================================================================
# 1. DEFINE THE OPTIMIZATION PROBLEM (Concrete Mix Design)
# =============================================================================
# Objective: Maximize 28-day Compressive Strength (or Minimize Cost/CO2 emission)
# Here, we use a regression-based proxy for Compressive Strength (MPa) based on:
# x[0]: Cement (kg/m³)      -> Bounds: [200, 500]
# x[1]: Water (kg/m³)       -> Bounds: [140, 220]
# x[2]: Coarse Agg (kg/m³)  -> Bounds: [800, 1200]
# x[3]: Fine Agg (kg/m³)    -> Bounds: [600, 900]

BOUNDS = np.array([
    [200.0, 500.0],  # Cement
    [140.0, 220.0],  # Water
    [800.0, 1200.0], # Coarse Aggregate
    [600.0, 900.0]   # Fine Aggregate
])

def objective_function(x):
    """
    Evaluates the concrete mix. We return a negative strength because 
    Cuckoo Search is written below as a MINIMIZER. Minimizing negative strength 
    is equivalent to maximizing the actual strength.
    """
    cement, water, coarse, fine = x[0], x[1], x[2], x[3]
    
    # 1. Structural Constraints (Hard Penalties)
    # Total volume of concrete mix should ideally be close to 2400 kg/m³ density
    total_mass = cement + water + coarse + fine
    if not (2200 <= total_mass <= 2500):
        return 1e6 # Large penalty for unrealistic mixes
        
    # Water-to-Cement ratio constraint (typically 0.35 to 0.60 for structural concrete)
    w_c_ratio = water / cement
    if not (0.35 <= w_c_ratio <= 0.60):
        return 1e6

    # 2. Mock Strength Empirical Model (Simplified non-linear relationship)
    # Strength increases with cement, decreases with excess water, and relies on proper aggregate packing
    strength = (25 + 0.1 * (cement - 200) 
                - 0.25 * (water - 140) 
                - 0.0005 * (coarse - 1000)**2 
                - 0.0008 * (fine - 750)**2)
    
    return -strength # Minimize negative strength to maximize actual strength

# =============================================================================
# 2. CUCKOO SEARCH ALGORITHM IMPLEMENTATION
# =============================================================================

def levy_flight(Lambda, size):
    """Generates steps from a Lévy distribution using Mantegna's algorithm."""
    # Calculate sigma based on analytical gamma function approximation
    num = math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2)
    den = math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2)
    sigma = (num / den) ** (1 / Lambda)
    
    u = np.random.normal(0, sigma, size)
    v = np.random.normal(0, 1, size)
    step = u / (np.abs(v) ** (1 / Lambda))
    return step

def cuckoo_search(pop_size=25, max_iter=100, pa=0.25, beta=1.5):
    dim = len(BOUNDS)
    lower_bounds = BOUNDS[:, 0]
    upper_bounds = BOUNDS[:, 1]
    
    # Initialize population of host nests randomly within bounds
    nests = np.random.uniform(lower_bounds, upper_bounds, (pop_size, dim))
    
    # Evaluate initial fitness
    fitness = np.array([objective_function(nest) for nest in nests])
    
    # Find the current best solution
    best_idx = np.argmin(fitness)
    best_nest = nests[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # Track performance across iterations
    history = []

    for t in range(max_iter):
        # Step 1: Generate new solutions via Lévy Flights
        new_nests = nests.copy()
        for i in range(pop_size):
            step_size = 0.01 * levy_flight(beta, dim) * (nests[i] - best_nest)
            new_nests[i] += np.random.normal(0, 1, dim) * step_size
            
            # Clip to boundary limits
            new_nests[i] = np.clip(new_nests[i], lower_bounds, upper_bounds)
            
        # Step 2: Random Choice & Greedy Replacement
        for i in range(pop_size):
            f_new = objective_function(new_nests[i])
            random_partner = np.random.randint(0, pop_size)
            
            if f_new < fitness[random_partner]:
                fitness[random_partner] = f_new
                nests[random_partner] = new_nests[i].copy()

        # Step 3: Abandon a fraction (pa) of the worst nests
        # Sort to easily find the worst performers
        idx_sorted = np.argsort(fitness)
        nests = nests[idx_sorted]
        fitness = fitness[idx_sorted]
        
        # Discover and replace poor nests using a random walk step
        num_abandoned = int(pa * pop_size)
        for i in range(pop_size - num_abandoned, pop_size):
            # Select two random nests to define a step vector
            r1, r2 = np.random.randint(0, pop_size, 2)
            step_factor = np.random.rand()
            nests[i] = nests[i] + step_factor * (nests[r1] - nests[r2])
            nests[i] = np.clip(nests[i], lower_bounds, upper_bounds)
            fitness[i] = objective_function(nests[i])
            
        # Step 4: Update Global Best
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_nest = nests[current_best_idx].copy()
            
        history.append(-best_fitness) # Record actual maximized strength

    return best_nest, -best_fitness, history

# =============================================================================
# 3. RUN THE OPTIMIZER & PRINT RESULTS
# =============================================================================
if __name__ == "__main__":
    np.random.seed(42) # Fixed seed for reproducible output
    
    best_mix, max_strength, convergence = cuckoo_search(pop_size=30, max_iter=150, pa=0.25)
    
    print("="*50)
    print("OPTIMIZED CONCRETE MIXTURE COMPOSITION")
    print("="*50)
    print(f"Cement:            {best_mix[0]:.2f} kg/m³")
    print(f"Water:             {best_mix[1]:.2f} kg/m³")
    print(f"Coarse Aggregate:  {best_mix[2]:.2f} kg/m³")
    print(f"Fine Aggregate:    {best_mix[3]:.2f} kg/m³")
    print("-"*50)
    print(f"Total Mixture Mass:{sum(best_mix):.2f} kg/m³")
    print(f"Water/Cement Ratio:{best_mix[1]/best_mix[0]:.3f}")
    print(f"Estimated 28-day Compressive Strength: {max_strength:.2f} MPa")
    print("="*50)
