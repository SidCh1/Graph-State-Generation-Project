import numpy as np
import matplotlib.pyplot as plt
import csv

# -------------------------------
# Case 1: Moment-of-success with lifetime
# -------------------------------
def simulate_case_1_moment_success(N, p, lifetime, trials=1000):
    """
    Case 1 with moment-of-success stopping:
    - Each Bell pair can be successful for 'lifetime' steps
    - Each failed pair tries to succeed with probability p
    - Stopping point: first time all pairs are simultaneously in success mode
    """
    total_times = []

    for _ in range(trials):
        time_remaining = np.zeros(N, dtype=int)
        time = 0

        while True:
            time += 1
            # Decrease lifetime for currently successful pairs
            time_remaining[time_remaining > 0] -= 1
            # Failed pairs attempt to succeed
            failed = (time_remaining == 0)
            successes = (np.random.rand(N) < p) & failed
            time_remaining[successes] = lifetime
            # Stop if all pairs are in success mode
            if np.all(time_remaining > 0):
                break

        total_times.append(time)

    return np.mean(total_times)

# -------------------------------
# Case 2: All pairs succeed simultaneously
# -------------------------------
def simulate_case_2(N, p, trials=1000):
    total_times = []

    for _ in range(trials):
        time = 0
        while True:
            time += 1
            if np.all(np.random.rand(N) < p):
                break
        total_times.append(time)

    return np.mean(total_times)

# -------------------------------
# Simulation Parameters
# -------------------------------
p = 0.4                     # success probability per pair per time step
Ns = range(3, 11)           # number of Bell pairs (3 to 10)
lifetimes = range(1,6, 1) # lifetimes: 5,10,...,50
trials = 1000

# -------------------------------
# Run Simulations
# -------------------------------
case2_times = []
# Run Case 2 once (doesn't depend on lifetime)
for N in Ns:
    print(f"Simulating Case 2, N = {N}")
    case2_times.append(simulate_case_2(N, p, trials))

# Case 1 for multiple lifetimes
case1_times_dict = {}  # store results for each lifetime

for lifetime in lifetimes:
    times = []
    for N in Ns:
        print(f"Simulating Case 1 , lifetime = {lifetime}, N = {N}")
        t = simulate_case_1_moment_success(N, p, lifetime, trials)
        times.append(t)
    case1_times_dict[lifetime] = times

# -------------------------------
# Plotting
# -------------------------------
plt.figure(figsize=(10, 6))

# Plot all Case 1 curves
for lifetime, times in case1_times_dict.items():
    plt.plot(Ns, times, marker='o', label=f'Bell Pairs, lifetime={lifetime}')

# Plot Case 2 curve
plt.plot(Ns, case2_times, marker='s', color='brown', linestyle='--', linewidth=2, label='GHZ State')

plt.yscale('log')
plt.xlabel('Degree of the node (d)')
plt.ylabel('Average Completion Time')
plt.title(f'Comparison of Bell Pair Distribution Times for Various Lifetimes (p={p})')
plt.legend()
plt.grid(True)
plt.savefig('my_plot_TEST.pdf')  # saves in current directory
plt.show()



# -------------------------------
# Save results to CSV
# -------------------------------

# Create descriptive filename
csv_filename = (
    f"simulation_results_"
    f"p{p}_"
    f"N{min(Ns)}to{max(Ns)}_"
    f"life{min(lifetimes)}to{max(lifetimes)}_"
    f"trials{trials}.csv"
)

with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)

    # --- Write simulation parameters at top ---
    writer.writerow(["# Simulation Parameters"])
    writer.writerow(["p", p])
    writer.writerow(["N_min", min(Ns)])
    writer.writerow(["N_max", max(Ns)])
    writer.writerow(["lifetimes", list(lifetimes)])
    writer.writerow(["trials", trials])
    writer.writerow([])  # empty line

    # --- Write data header ---
    header = ["N", "Case2_GHZ"] + [f"Case1_lifetime_{l}" for l in lifetimes]
    writer.writerow(header)

    # --- Write data rows ---
    for i, N in enumerate(Ns):
        row = [N, case2_times[i]]
        for lifetime in lifetimes:
            row.append(case1_times_dict[lifetime][i])
        writer.writerow(row)

print(f"Results saved to {csv_filename}")
