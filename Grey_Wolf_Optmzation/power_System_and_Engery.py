import math
import random


# 1. Solar PV Array Model under Partial Shading (Multi-Peak P-V Curve)
# Simulates 3 series-connected PV modules receiving different sunlight (irradiance)
def pv_power_output(voltage):
    if voltage <= 0.0 or voltage >= 100.0:
        return 0.0

    # Irradiance levels (W/m^2) on the 3 modules (partial shading condition)
    irradiances = [1000.0, 600.0, 350.0]
    v_oc_module = 33.3   # Open-circuit voltage per module (V)
    i_sc_stc = 8.5       # Short-circuit current at 1000 W/m^2 (A)

    # Bypass diodes allow higher-current modules to bypass shaded ones
    possible_powers = []
    for idx, irr in enumerate(sorted(irradiances, reverse=True)):
        active_modules = idx + 1
        v_max_section = active_modules * v_oc_module
        if voltage < v_max_section:
            i_ph = i_sc_stc * (irr / 1000.0)
            # Diode exponential I-V characteristic
            current = i_ph * (1.0 - math.exp((voltage - v_max_section) / (2.5 * active_modules)))
            possible_powers.append(max(0.0, voltage * current))

    return max(possible_powers) if possible_powers else 0.0


# 2. Grey Wolf Optimizer (GWO) for Global MPPT
def grey_wolf_mppt(num_wolves=12, max_iter=30, seed=42):
    random.seed(seed)

    # Search space: Operating Voltage [V_min, V_max]
    v_min, v_max = 0.0, 100.0

    # Step 2: Initialize population of N wolves randomly
    wolves = [random.uniform(v_min, v_max) for _ in range(num_wolves)]

    # Initialize Alpha, Beta, Delta leaders
    alpha_pos, alpha_score = 0.0, -1.0
    beta_pos, beta_score = 0.0, -1.0
    delta_pos, delta_score = 0.0, -1.0

    # Step 6: Repeat until maximum iterations
    for t in range(max_iter):
        # Step 4 & 5: Evaluate fitness and update Alpha, Beta, Delta
        for i in range(num_wolves):
            # Enforce boundary conditions
            wolves[i] = max(v_min, min(v_max, wolves[i]))
            fitness = pv_power_output(wolves[i])

            if fitness > alpha_score:
                delta_score, delta_pos = beta_score, beta_pos
                beta_score, beta_pos = alpha_score, alpha_pos
                alpha_score, alpha_pos = fitness, wolves[i]
            elif fitness > beta_score:
                delta_score, delta_pos = beta_score, beta_pos
                beta_score, beta_pos = fitness, wolves[i]
            elif fitness > delta_score:
                delta_score, delta_pos = fitness, wolves[i]

        # Decrease 'a' linearly from 2 to 0
        a = 2.0 - t * (2.0 / max_iter)

        # Update position of each wolf using Alpha, Beta, and Delta
        for i in range(num_wolves):
            # Alpha guidance
            r1, r2 = random.random(), random.random()
            A1 = 2.0 * a * r1 - a
            C1 = 2.0 * r2
            D_alpha = abs(C1 * alpha_pos - wolves[i])
            X1 = alpha_pos - A1 * D_alpha

            # Beta guidance
            r1, r2 = random.random(), random.random()
            A2 = 2.0 * a * r1 - a
            C2 = 2.0 * r2
            D_beta = abs(C2 * beta_pos - wolves[i])
            X2 = beta_pos - A2 * D_beta

            # Delta guidance
            r1, r2 = random.random(), random.random()
            A3 = 2.0 * a * r1 - a
            C3 = 2.0 * r2
            D_delta = abs(C3 * delta_pos - wolves[i])
            X3 = delta_pos - A3 * D_delta

            # New wolf position (average of the three leaders)
            wolves[i] = (X1 + X2 + X3) / 3.0

        print(
            f"Iter {t + 1:02d} | Global Max Power (Alpha): {alpha_score:6.2f} W | "
            f"Optimal Voltage: {alpha_pos:6.2f} V"
        )

    return alpha_pos, alpha_score


if __name__ == "__main__":
    best_voltage, max_power = grey_wolf_mppt(num_wolves=12, max_iter=30)
    print("\n--- Optimal Solar PV MPPT Result ---")
    print(f"Global Maximum Power Point (GMPP): {max_power:.2f} W")
    print(f"Optimal Operating Voltage:         {best_voltage:.2f} V")
