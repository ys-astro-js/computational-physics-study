import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POPULATION = 1000
INITIAL_INFECTED = 10
INITIAL_RECOVERED = 0
DT = 0.05
T_MAX = 160.0
ABM_RUNS = 300
SEED = 2026
OUTPUT_PATH = "example-18.png"
IMMUNITY_LOSS_DAYS = 60.0

DISEASES = [
    {
        "name": "Ebola virus disease",
        "r0": 2.0,
        "infectious_days": 10.0,
        "fatality_rate": 0.50,
    },
    {
        "name": "Measles",
        "r0": 15.0,
        "infectious_days": 8.0,
        "fatality_rate": 0.002,
    },
]


def disease_rates(disease):
    gamma = 1.0 / disease["infectious_days"]
    beta = disease["r0"] * gamma
    return beta, gamma


def simulate_compartment_sird(disease):
    beta, gamma = disease_rates(disease)
    omega = 1.0 / IMMUNITY_LOSS_DAYS
    fatality_rate = disease["fatality_rate"]
    time = np.arange(0.0, T_MAX + DT, DT)

    susceptible = np.empty(len(time))
    infected = np.empty(len(time))
    recovered = np.empty(len(time))
    dead = np.empty(len(time))
    cumulative_infected = np.empty(len(time))

    susceptible[0] = POPULATION - INITIAL_INFECTED - INITIAL_RECOVERED
    infected[0] = INITIAL_INFECTED
    recovered[0] = INITIAL_RECOVERED
    dead[0] = 0.0
    cumulative_infected[0] = INITIAL_INFECTED

    for step in range(len(time) - 1):
        s = susceptible[step]
        i = infected[step]
        r = recovered[step]
        d = dead[step]
        c = cumulative_infected[step]

        new_infections = beta * s * i / POPULATION * DT
        removals = gamma * i * DT
        new_deaths = fatality_rate * removals
        new_recoveries = removals - new_deaths
        immunity_losses = omega * r * DT

        susceptible[step + 1] = s - new_infections + immunity_losses
        infected[step + 1] = i + new_infections - removals
        recovered[step + 1] = r + new_recoveries - immunity_losses
        dead[step + 1] = d + new_deaths
        cumulative_infected[step + 1] = c + new_infections

    return time, susceptible, infected, recovered, dead, cumulative_infected


def simulate_actor_sird(disease, rng):
    beta, gamma = disease_rates(disease)
    omega = 1.0 / IMMUNITY_LOSS_DAYS
    fatality_rate = disease["fatality_rate"]
    time = np.arange(0.0, T_MAX + DT, DT)

    susceptible = np.empty(len(time), dtype=int)
    infected = np.empty(len(time), dtype=int)
    recovered = np.empty(len(time), dtype=int)
    dead = np.empty(len(time), dtype=int)
    cumulative_infected = np.empty(len(time), dtype=int)

    susceptible[0] = POPULATION - INITIAL_INFECTED - INITIAL_RECOVERED
    infected[0] = INITIAL_INFECTED
    recovered[0] = INITIAL_RECOVERED
    dead[0] = 0
    cumulative_infected[0] = INITIAL_INFECTED

    removal_probability = 1.0 - np.exp(-gamma * DT)
    immunity_loss_probability = 1.0 - np.exp(-omega * DT)

    for step in range(len(time) - 1):
        susceptible_count = susceptible[step]
        infected_count = infected[step]
        recovered_count = recovered[step]
        infection_probability = 1.0 - np.exp(-beta * infected_count / POPULATION * DT)

        new_infections = rng.binomial(susceptible_count, infection_probability)
        removals = rng.binomial(infected_count, removal_probability)
        new_deaths = rng.binomial(removals, fatality_rate)
        new_recoveries = removals - new_deaths
        immunity_losses = rng.binomial(recovered_count, immunity_loss_probability)

        susceptible[step + 1] = susceptible_count - new_infections + immunity_losses
        infected[step + 1] = infected_count + new_infections - removals
        recovered[step + 1] = recovered_count + new_recoveries - immunity_losses
        dead[step + 1] = dead[step] + new_deaths
        cumulative_infected[step + 1] = cumulative_infected[step] + new_infections

    return susceptible, infected, recovered, dead, cumulative_infected


def simulate_actor_ensemble(disease, seed):
    rng = np.random.default_rng(seed)
    steps = int(T_MAX / DT) + 1
    susceptible_runs = np.empty((ABM_RUNS, steps))
    infected_runs = np.empty_like(susceptible_runs)
    recovered_runs = np.empty_like(susceptible_runs)
    dead_runs = np.empty_like(susceptible_runs)
    cumulative_infected_runs = np.empty_like(susceptible_runs)

    for run in range(ABM_RUNS):
        susceptible, infected, recovered, dead, cumulative_infected = simulate_actor_sird(
            disease,
            rng,
        )
        susceptible_runs[run] = susceptible
        infected_runs[run] = infected
        recovered_runs[run] = recovered
        dead_runs[run] = dead
        cumulative_infected_runs[run] = cumulative_infected

    return susceptible_runs, infected_runs, recovered_runs, dead_runs, cumulative_infected_runs


def summarize(
    time,
    compartment_s,
    compartment_i,
    compartment_d,
    compartment_c,
    actor_s_runs,
    actor_i_runs,
    actor_d_runs,
    actor_c_runs,
):
    actor_i_mean = np.mean(actor_i_runs, axis=0)
    actor_i_low = np.percentile(actor_i_runs, 5, axis=0)
    actor_i_high = np.percentile(actor_i_runs, 95, axis=0)
    actor_d_mean = np.mean(actor_d_runs, axis=0)
    actor_d_low = np.percentile(actor_d_runs, 5, axis=0)
    actor_d_high = np.percentile(actor_d_runs, 95, axis=0)
    actor_c_mean = np.mean(actor_c_runs, axis=0)
    actor_c_low = np.percentile(actor_c_runs, 5, axis=0)
    actor_c_high = np.percentile(actor_c_runs, 95, axis=0)

    compartment_peak_index = np.argmax(compartment_i)
    actor_peak_indices = np.argmax(actor_i_runs, axis=1)

    return {
        "actor_i_mean": actor_i_mean,
        "actor_i_low": actor_i_low,
        "actor_i_high": actor_i_high,
        "actor_d_mean": actor_d_mean,
        "actor_d_low": actor_d_low,
        "actor_d_high": actor_d_high,
        "compartment_cumulative_infected": compartment_c,
        "actor_cumulative_infected_mean": actor_c_mean,
        "actor_cumulative_infected_low": actor_c_low,
        "actor_cumulative_infected_high": actor_c_high,
        "compartment_peak_day": time[compartment_peak_index],
        "compartment_peak_infected": compartment_i[compartment_peak_index],
        "compartment_cumulative_infected_final": compartment_c[-1],
        "compartment_dead_final": compartment_d[-1],
        "actor_peak_day_mean": np.mean(time[actor_peak_indices]),
        "actor_peak_day_std": np.std(time[actor_peak_indices]),
        "actor_peak_infected_mean": np.mean(np.max(actor_i_runs, axis=1)),
        "actor_peak_infected_std": np.std(np.max(actor_i_runs, axis=1)),
        "actor_cumulative_infected_final_mean": np.mean(actor_c_runs[:, -1]),
        "actor_cumulative_infected_final_std": np.std(actor_c_runs[:, -1]),
        "actor_dead_final_mean": np.mean(actor_d_runs[:, -1]),
        "actor_dead_final_std": np.std(actor_d_runs[:, -1]),
        "infected_rmse": np.sqrt(np.mean((compartment_i - actor_i_mean) ** 2)),
        "dead_rmse": np.sqrt(np.mean((compartment_d - actor_d_mean) ** 2)),
    }


def run_disease(disease, seed):
    time, compartment_s, compartment_i, compartment_r, compartment_d, compartment_c = (
        simulate_compartment_sird(disease)
    )
    (
        actor_s_runs,
        actor_i_runs,
        actor_r_runs,
        actor_d_runs,
        actor_c_runs,
    ) = simulate_actor_ensemble(disease, seed)
    summary = summarize(
        time,
        compartment_s,
        compartment_i,
        compartment_d,
        compartment_c,
        actor_s_runs,
        actor_i_runs,
        actor_d_runs,
        actor_c_runs,
    )

    return {
        "time": time,
        "compartment_s": compartment_s,
        "compartment_i": compartment_i,
        "compartment_r": compartment_r,
        "compartment_d": compartment_d,
        "compartment_c": compartment_c,
        "actor_s_runs": actor_s_runs,
        "actor_i_runs": actor_i_runs,
        "actor_r_runs": actor_r_runs,
        "actor_d_runs": actor_d_runs,
        "actor_c_runs": actor_c_runs,
        "summary": summary,
    }


def plot_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)

    for row, (disease, result) in enumerate(results):
        time = result["time"]
        summary = result["summary"]
        beta, gamma = disease_rates(disease)

        ax_infected = axes[row, 0]
        ax_infected.plot(
            time,
            result["compartment_i"],
            color="black",
            label="compartment infected",
        )
        ax_infected.plot(
            time,
            summary["actor_i_mean"],
            color="tab:red",
            label="actor mean infected",
        )
        ax_infected.fill_between(
            time,
            summary["actor_i_low"],
            summary["actor_i_high"],
            color="tab:red",
            alpha=0.18,
            label="actor 5-95%",
        )
        ax_infected.set_title(
            f"{disease['name']} infected (beta={beta:.3f}, gamma={gamma:.3f})"
        )
        ax_infected.set_xlabel("day")
        ax_infected.set_ylabel("infected people")
        ax_infected.grid(alpha=0.25)
        ax_infected.legend(fontsize=8)

        ax_total = axes[row, 1]
        ax_total.plot(
            time,
            summary["compartment_cumulative_infected"],
            color="tab:blue",
            label="compartment infection events",
        )
        ax_total.plot(
            time,
            summary["actor_cumulative_infected_mean"],
            color="tab:cyan",
            label="actor mean infection events",
        )
        ax_total.fill_between(
            time,
            summary["actor_cumulative_infected_low"],
            summary["actor_cumulative_infected_high"],
            color="tab:cyan",
            alpha=0.14,
            label="infection events 5-95%",
        )
        ax_total.plot(
            time,
            result["compartment_d"],
            color="black",
            label="compartment deaths",
        )
        ax_total.plot(
            time,
            summary["actor_d_mean"],
            color="tab:purple",
            label="actor mean deaths",
        )
        ax_total.fill_between(
            time,
            summary["actor_d_low"],
            summary["actor_d_high"],
            color="tab:purple",
            alpha=0.18,
            label="deaths 5-95%",
        )
        ax_total.set_title(
            f"{disease['name']} infection events vs deaths (CFR={disease['fatality_rate']:.3g})"
        )
        ax_total.set_xlabel("day")
        ax_total.set_ylabel("people")
        ax_total.grid(alpha=0.25)
        ax_total.legend(fontsize=8)

    fig.savefig(OUTPUT_PATH, dpi=200)


def print_summary(disease, result):
    beta, gamma = disease_rates(disease)
    summary = result["summary"]

    print(disease["name"])
    print(f"  R0 = {disease['r0']:.2f}")
    print(f"  infectious period = {disease['infectious_days']:.1f} days")
    print(f"  beta = {beta:.4f} per day")
    print(f"  gamma = {gamma:.4f} per day")
    print(f"  fatality rate = {disease['fatality_rate']:.4f}")
    print(
        "  peak infected, compartment = "
        f"day {summary['compartment_peak_day']:.1f}, "
        f"{summary['compartment_peak_infected']:.1f} people"
    )
    print(
        "  peak infected, actor-based = "
        f"day {summary['actor_peak_day_mean']:.1f} +/- "
        f"{summary['actor_peak_day_std']:.1f}, "
        f"{summary['actor_peak_infected_mean']:.1f} +/- "
        f"{summary['actor_peak_infected_std']:.1f} people"
    )
    print(
        "  final deaths, compartment = "
        f"{summary['compartment_dead_final']:.1f} people"
    )
    print(
        "  final deaths, actor-based = "
        f"{summary['actor_dead_final_mean']:.1f} +/- "
        f"{summary['actor_dead_final_std']:.1f} people"
    )
    print(
        "  cumulative infection events, compartment = "
        f"{summary['compartment_cumulative_infected_final']:.1f}"
    )
    print(
        "  cumulative infection events, actor-based = "
        f"{summary['actor_cumulative_infected_final_mean']:.1f} +/- "
        f"{summary['actor_cumulative_infected_final_std']:.1f}"
    )
    print(f"  infected RMSE = {summary['infected_rmse']:.2f} people")
    print(f"  deaths RMSE = {summary['dead_rmse']:.2f} people")


def main():
    results = []
    for index, disease in enumerate(DISEASES):
        results.append((disease, run_disease(disease, SEED + 1000 * index)))

    plot_results(results)

    print("SIRS+D infection prediction: compartment model vs actor-based model")
    print(f"population = {POPULATION}, initial infected = {INITIAL_INFECTED}")
    print(f"mean immunity duration after recovery = {IMMUNITY_LOSS_DAYS:.1f} days")
    print(f"dt = {DT}, t_max = {T_MAX}, actor runs = {ABM_RUNS}")
    print()

    for disease, result in results:
        print_summary(disease, result)
        print()

    print(
        "Note: values are illustrative literature-based parameters, not a "
        "location-specific public-health forecast."
    )
    print(f"Saved graph to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
