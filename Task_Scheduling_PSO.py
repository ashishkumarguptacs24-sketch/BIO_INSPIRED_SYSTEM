import numpy as np
import matplotlib.pyplot as plt

tasks = np.array([10, 5, 8, 12, 6, 7, 4, 9])

num_tasks = len(tasks)
num_machines = 3

def objective_function(x):
    assignment = np.clip(
        np.round(x).astype(int),
        0,
        num_machines - 1
    )

    machine_load = np.zeros(num_machines)

    for task in range(num_tasks):
        machine = assignment[task]
        machine_load[machine] += tasks[task]

    return np.max(machine_load)

N = 30
T = 100

w = 0.7
c1 = 1.5
c2 = 1.5

x = np.random.uniform(
    0,
    num_machines - 1,
    size=(N, num_tasks)
)

v = np.random.uniform(
    -1,
    1,
    size=(N, num_tasks)
)

fitness = np.zeros(N)

for i in range(N):
    fitness[i] = objective_function(x[i])

pBest = x.copy()
pBest_fitness = fitness.copy()

best_particle = np.argmin(pBest_fitness)

gBest = pBest[best_particle].copy()
gBest_fitness = pBest_fitness[best_particle]

best_history = []

for iteration in range(T):

    for i in range(N):

        r1 = np.random.random(num_tasks)
        r2 = np.random.random(num_tasks)

        v[i] = (
            w * v[i]
            + c1 * r1 * (pBest[i] - x[i])
            + c2 * r2 * (gBest - x[i])
        )

        v[i] = np.clip(v[i], -2, 2)

        x[i] = x[i] + v[i]

        x[i] = np.clip(
            x[i],
            0,
            num_machines - 1
        )

        current_fitness = objective_function(x[i])

        if current_fitness < pBest_fitness[i]:
            pBest[i] = x[i].copy()
            pBest_fitness[i] = current_fitness

    best_particle = np.argmin(pBest_fitness)

    if pBest_fitness[best_particle] < gBest_fitness:
        gBest = pBest[best_particle].copy()
        gBest_fitness = pBest_fitness[best_particle]

    best_history.append(gBest_fitness)

final_assignment = np.clip(
    np.round(gBest).astype(int),
    0,
    num_machines - 1
)

machine_load = np.zeros(num_machines)

for task in range(num_tasks):
    machine = final_assignment[task]
    machine_load[machine] += tasks[task]

print("=" * 50)
print("       TASK SCHEDULING USING PSO")
print("=" * 50)

print("\nTask Processing Times:")
print(tasks)

print("\nFinal Task Assignment:")

for task in range(num_tasks):
    print(
        f"Task {task + 1} "
        f"(Time = {tasks[task]}) "
        f"--> Machine {final_assignment[task] + 1}"
    )

print("\nMachine Loads:")

for machine in range(num_machines):
    print(
        f"Machine {machine + 1}: "
        f"{machine_load[machine]}"
    )

print("\nMinimum Makespan:", gBest_fitness)

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, T + 1),
    best_history,
    marker='o',
    markersize=2
)

plt.title("PSO Convergence for Task Scheduling")
plt.xlabel("Iteration")
plt.ylabel("Best Makespan")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))

plt.bar(
    range(1, num_machines + 1),
    machine_load
)

plt.title("Final Machine Loads")
plt.xlabel("Machine")
plt.ylabel("Total Processing Time")
plt.xticks(range(1, num_machines + 1))

plt.show()