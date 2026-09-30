import random
import numpy as np

CITIES = {
    'A': (0, 0), 'B': (1, 5), 'C': (5, 6), 'D': (6, 1), 'E': (3, 3)
}
CITY_NAMES = list(CITIES.keys())

def calculate_distance(city1, city2):
    p1, p2 = np.array(CITIES[city1]), np.array(CITIES[city2])
    return np.sqrt(np.sum((p1 - p2) ** 2))

def total_route_distance(route):
    dist = 0
    for i in range(len(route)):
        dist += calculate_distance(route[i], route[(i + 1) % len(route)])
    return dist

def calculate_fitness(route):
    return 1 / float(total_route_distance(route))

def create_initial_population(pop_size):
    population = []
    for _ in range(pop_size):
        route = random.sample(CITY_NAMES, len(CITY_NAMES))
        population.append(route)
    return population

def tournament_selection(population, k=3):
    tournament = random.sample(population, k)
    return max(tournament, key=calculate_fitness)

def ordered_crossover(parent1, parent2):
    child = [None] * len(parent1)
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child[start:end] = parent1[start:end]
    
    p2_pointer = 0
    for i in range(len(child)):
        if child[i] is None:
            while parent2[p2_pointer] in child:
                p2_pointer += 1
            child[i] = parent2[p2_pointer]
            
    return child

def mutate(route, mutation_rate=0.1):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

def genetic_algorithm(pop_size=20, generations=50, mutation_rate=0.15):
    population = create_initial_population(pop_size)
    
    best_route = min(population, key=total_route_distance)
    print(f"Initial Best Distance: {total_route_distance(best_route):.2f}")

    for gen in range(generations):
        new_population = []
        population.sort(key=calculate_fitness, reverse=True)
        new_population.append(population[0]) 
        
        while len(new_population) < pop_size:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)
            child = ordered_crossover(p1, p2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
            
        population = new_population
        best_route = min(population, key=total_route_distance)
        
    print(f"Final Optimized Route: {' -> '.join(best_route)} -> {best_route[0]}")
    print(f"Final Optimized Distance: {total_route_distance(best_route):.2f}")
    return best_route

optimized_path = genetic_algorithm()
