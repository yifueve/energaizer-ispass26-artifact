import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_agents(n_agents, rng):
    agents = []
    for i in range(n_agents):
        agents.append(
            {
                "perf": rng.uniform(0.85, 1.25),
                "idle": rng.uniform(35.0, 55.0),
                "dyn": rng.uniform(85.0, 130.0),
                "v0": rng.uniform(0.68, 0.76),
                "v_slope": rng.uniform(0.34, 0.46),
                "energy_price": rng.uniform(0.8, 1.25),
                "carbon": rng.uniform(0.6, 1.4),
            }
        )
    return agents


def project_strategy(x, n_agents, demand, f_min, f_max):
    x = x.copy()
    q = np.maximum(x[:n_agents], 1e-3)
    q *= demand / q.sum()
    f = np.clip(x[n_agents:], f_min, f_max)
    return np.concatenate([q, f])


def voltage(agent, f, f_min, f_max):
    z = (f - f_min) / (f_max - f_min)
    return agent["v0"] + agent["v_slope"] * z + 0.035 * z * z


def service_rate(agent, f):
    return agent["perf"] * np.power(f, 0.92)


def local_metrics(agent, q, f, f_min, f_max):
    service = service_rate(agent, f)
    latency = q / service
    utilization = np.clip(q / (service + 1e-9), 0.05, 1.35)
    v = voltage(agent, f, f_min, f_max)
    power = agent["idle"] + agent["dyn"] * (v * v) * f * utilization
    energy = power * latency
    return latency, power, energy


def potential_cost(x, agents, demand, power_budget, f_min, f_max):
    n_agents = len(agents)
    q = x[:n_agents]
    f = x[n_agents:]

    cost = 0.0
    total_power = 0.0
    for i, agent in enumerate(agents):
        _, power, energy = local_metrics(agent, q[i], f[i], f_min, f_max)
        total_power += power
        cost += agent["energy_price"] * energy
        cost += 0.35 * agent["carbon"] * energy

    cost += 150.0 * np.square(q.sum() - demand)
    cost += 0.20 * np.square(max(total_power - power_budget, 0.0))
    return float(cost)


def objective_value(x, agents, args):
    return potential_cost(
        x, agents, args.demand, args.power_budget, args.f_min, args.f_max
    )


def finite_difference_gradient(x, objective, eps=1e-4):
    grad = np.zeros_like(x)
    for k in range(len(x)):
        step = eps * max(1.0, abs(x[k]))
        xp = x.copy()
        xm = x.copy()
        xp[k] += step
        xm[k] -= step
        grad[k] = (objective(xp) - objective(xm)) / (2.0 * step)
    return grad


def clip_vector(v, tau):
    norm = np.linalg.norm(v)
    if norm <= tau or norm == 0.0:
        return v.copy()
    return v * (tau / norm)


class CenteredClipping:
    def __init__(self, tau, inner_iterations):
        self.tau = tau
        self.inner_iterations = inner_iterations
        self.center = None

    def __call__(self, updates):
        if self.center is None:
            self.center = np.zeros_like(updates[0])
        for _ in range(self.inner_iterations):
            clipped = [clip_vector(update - self.center, self.tau) for update in updates]
            self.center = self.center + np.mean(clipped, axis=0)
        return self.center.copy()


def corrupt_update(update, attack, rng):
    if attack == "sign_flip":
        return -8.0 * update
    if attack == "frequency_high":
        corrupted = update.copy()
        n = len(update) // 2
        corrupted[:n] = -5.0 * update[:n]
        corrupted[n:] = -35.0
        return corrupted
    if attack == "random":
        return rng.normal(loc=0.0, scale=25.0, size=update.shape)
    raise ValueError(f"Unknown attack: {attack}")


def agent_reported_gradients(x, agents, args, rng):
    objective = lambda z: potential_cost(
        z,
        agents,
        args.demand,
        args.power_budget,
        args.f_min,
        args.f_max,
    )
    true_gradient = finite_difference_gradient(x, objective)

    reports = []
    for i, _ in enumerate(agents):
        noise_scale = args.gradient_noise * (1.0 + 0.15 * i)
        report = true_gradient + rng.normal(0.0, noise_scale, size=true_gradient.shape)
        if i < args.byzantine_agents:
            report = corrupt_update(report, args.attack, rng)
        reports.append(report)
    return reports, true_gradient


def summarize_state(iteration, name, x, grad, accepted_step, agents, args):
    n_agents = len(agents)
    q = x[:n_agents]
    f = x[n_agents:]
    latencies = []
    powers = []
    energies = []
    for i, agent in enumerate(agents):
        latency, power, energy = local_metrics(agent, q[i], f[i], args.f_min, args.f_max)
        latencies.append(latency)
        powers.append(power)
        energies.append(energy)

    return {
        "iteration": iteration,
        "method": name,
        "cost": objective_value(x, agents, args),
        "demand_error": q.sum() - args.demand,
        "total_power": float(np.sum(powers)),
        "power_violation": max(float(np.sum(powers)) - args.power_budget, 0.0),
        "max_latency": float(np.max(latencies)),
        "mean_frequency": float(np.mean(f)),
        "mean_workload": float(np.mean(q)),
        "gradient_norm": float(np.linalg.norm(grad)),
        "accepted_step": accepted_step,
    }


def projected_backtracking_step(x, grad, agents, args):
    current_cost = objective_value(x, agents, args)
    grad_norm_sq = float(np.dot(grad, grad))
    step = args.step_size

    for _ in range(args.line_search_steps):
        candidate = project_strategy(
            x - step * grad, len(agents), args.demand, args.f_min, args.f_max
        )
        candidate_cost = objective_value(candidate, agents, args)
        sufficient_decrease = current_cost - args.armijo_c * step * grad_norm_sq
        if candidate_cost <= sufficient_decrease:
            return candidate, step
        step *= args.line_search_shrink

    candidate = project_strategy(
        x - step * grad, len(agents), args.demand, args.f_min, args.f_max
    )
    if objective_value(candidate, agents, args) <= current_cost:
        return candidate, step
    return x.copy(), 0.0


def run_method(name, x0, agents, args, rng):
    x = x0.copy()
    aggregator = CenteredClipping(args.clip_tau, args.inner_iterations)
    rows = []
    accepted_step = 0.0

    for iteration in range(args.steps + 1):
        reports, true_gradient = agent_reported_gradients(x, agents, args, rng)

        if name == "mean":
            grad = np.mean(reports, axis=0)
        elif name == "centered_clipping":
            grad = aggregator(reports)
        elif name == "oracle":
            grad = true_gradient
        else:
            raise ValueError(f"Unknown method: {name}")

        rows.append(summarize_state(iteration, name, x, grad, accepted_step, agents, args))
        if iteration == args.steps:
            break

        if args.no_line_search:
            x = x - args.step_size * grad
            x = project_strategy(x, len(agents), args.demand, args.f_min, args.f_max)
            accepted_step = args.step_size
        else:
            x, accepted_step = projected_backtracking_step(x, grad, agents, args)

    return rows


def plot_results(df, output_prefix):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9.5, 6.0))
    axes = axes.flatten()

    specs = [
        ("cost", "Potential cost"),
        ("total_power", "Total power"),
        ("max_latency", "Max latency"),
        ("mean_frequency", "Mean frequency"),
    ]

    for ax, (column, title) in zip(axes, specs):
        for method, part in df.groupby("method"):
            ax.plot(part["iteration"], part[column], label=method, linewidth=1.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_prefix + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix + ".pdf", dpi=300, bbox_inches="tight")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Simulate a Byzantine-robust Nash-style DVFS game with mean and "
            "centered-clipping gradient aggregation."
        )
    )
    parser.add_argument("--agents", type=int, default=9)
    parser.add_argument("--byzantine_agents", type=int, default=2)
    parser.add_argument(
        "--attack",
        choices=["sign_flip", "frequency_high", "random"],
        default="sign_flip",
    )
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--step_size", type=float, default=0.0025)
    parser.add_argument("--no_line_search", action="store_true")
    parser.add_argument("--line_search_steps", type=int, default=16)
    parser.add_argument("--line_search_shrink", type=float, default=0.5)
    parser.add_argument("--armijo_c", type=float, default=1e-4)
    parser.add_argument("--clip_tau", type=float, default=9.0)
    parser.add_argument("--inner_iterations", type=int, default=5)
    parser.add_argument("--gradient_noise", type=float, default=0.06)
    parser.add_argument("--demand", type=float, default=9.0)
    parser.add_argument("--power_budget", type=float, default=760.0)
    parser.add_argument("--f_min", type=float, default=0.55)
    parser.add_argument("--f_max", type=float, default=1.45)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments_endtoend/results/robust_dvfs_game",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    agents = make_agents(args.agents, rng)

    q0 = np.full(args.agents, args.demand / args.agents)
    f0 = np.full(args.agents, 1.0)
    x0 = np.concatenate([q0, f0])

    rows = []
    method_seeds = {
        "oracle": args.seed + 101,
        "mean": args.seed + 202,
        "centered_clipping": args.seed + 303,
    }
    for method in ["oracle", "mean", "centered_clipping"]:
        method_rng = np.random.default_rng(method_seeds[method])
        rows.extend(run_method(method, x0, agents, args, method_rng))

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "trajectory.csv")
    plot_prefix = os.path.join(args.output_dir, "trajectory")
    df.to_csv(csv_path, index=False)
    plot_results(df, plot_prefix)

    final = df[df["iteration"] == args.steps].sort_values("cost")
    print("Saved {}".format(csv_path))
    print("Saved {}.png".format(plot_prefix))
    print("Saved {}.pdf".format(plot_prefix))
    print("\nFinal state:")
    print(final[["method", "cost", "total_power", "power_violation", "max_latency", "mean_frequency"]].to_string(index=False))


if __name__ == "__main__":
    main()
