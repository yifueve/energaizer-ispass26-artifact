import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BYZANTINE_CODE_ROOT = os.path.join(REPO_ROOT, "byzantine-robust-optimizer-main")
sys.path.append(BYZANTINE_CODE_ROOT)

from codes.aggregator.base import Mean
from codes.aggregator.clipping import Clipping
from codes.aggregator.coordinatewise_median import CM
from codes.aggregator.krum import Krum
from codes.aggregator.trimmed_mean import TM


def parse_model_name(workload):
    return workload.split("_")[0]


def build_aggregator(name, n_workers, byzantine_workers, clipping_tau, inner_iterations):
    if name == "mean":
        return Mean()
    if name == "clipping":
        return Clipping(tau=clipping_tau, n_iter=inner_iterations)
    if name == "cm":
        return CM()
    if name == "tm":
        return TM(b=byzantine_workers)
    if name == "krum":
        f = byzantine_workers
        if 2 * f + 2 > n_workers:
            raise ValueError(
                f"Krum requires 2*f+2 <= n_workers, got f={f}, n={n_workers}."
            )
        return Krum(n=n_workers, f=f, m=1)
    raise ValueError(f"Unknown aggregator: {name}")


def choose_reference_freq(freqs, requested):
    freqs = sorted(freqs)
    if requested is not None:
        if requested not in freqs:
            raise ValueError(
                f"reference_freq={requested} is not in the result frequencies {freqs}."
            )
        return requested
    return freqs[len(freqs) // 2]


def vectorize_worker_curve(worker_df, freqs, reference_freq):
    by_freq = worker_df.set_index("target_freq")
    time_ref = by_freq.loc[reference_freq, "time_predicted"]
    energy_ref = by_freq.loc[reference_freq, "energy_predicted"]

    if time_ref == 0 or energy_ref == 0:
        raise ValueError("Reference predictions must be non-zero for normalization.")

    time_scale = by_freq.loc[freqs, "time_predicted"].to_numpy(dtype=np.float64) / time_ref
    energy_scale = (
        by_freq.loc[freqs, "energy_predicted"].to_numpy(dtype=np.float64) / energy_ref
    )
    return torch.tensor(
        np.concatenate([time_scale, energy_scale]), dtype=torch.float32
    )


def apply_attack(vectors, attack, count):
    if attack == "none" or count == 0:
        return vectors

    attacked = list(vectors)
    for i in range(min(count, len(attacked))):
        if attack == "bitflip":
            attacked[i] = -attacked[i]
        elif attack == "high":
            attacked[i] = attacked[i] * 10.0
        elif attack == "low":
            attacked[i] = attacked[i] * 0.1
        elif attack == "activity_high":
            pass
        else:
            raise ValueError(f"Unknown attack: {attack}")
    return attacked


def load_idle_power(path):
    if path is None:
        return None
    with open(path, "r") as f:
        return {int(k): float(v) for k, v in json.load(f).items()}


def get_idle_power(idle_power, freq):
    if idle_power is None:
        raise ValueError("--dvfs_idle_power_json is required for activity_high attack.")
    if freq in idle_power:
        return idle_power[freq]
    nearest = min(idle_power, key=lambda x: abs(x - freq))
    return idle_power[nearest]


def apply_worker_curve_attack(
    worker_df,
    attack,
    idle_power,
    activity_scale,
    reference_freq,
    max_freq,
):
    if attack != "activity_high":
        return worker_df

    attacked = worker_df.copy()
    power = attacked["energy_predicted"] / attacked["time_predicted"] * 1000.0
    idle = attacked["target_freq"].apply(lambda f: get_idle_power(idle_power, int(f)))
    dynamic = (power - idle).clip(lower=0.0)

    freq_gain = (
        (attacked["target_freq"] - reference_freq) / max(max_freq - reference_freq, 1)
    ).clip(lower=0.0)
    activity_gain = 1.0 + activity_scale * freq_gain
    attacked_power = idle + activity_gain * dynamic
    attacked["energy_predicted"] = attacked_power * attacked["time_predicted"] / 1000.0
    return attacked


def robust_group_predictions(
    group,
    aggregation,
    byzantine_workers,
    clipping_tau,
    inner_iterations,
    reference_freq,
    attack,
    idle_power,
    activity_scale,
):
    workloads = sorted(group["workload"].unique())
    freqs = sorted(group["target_freq"].unique())
    reference_freq = choose_reference_freq(freqs, reference_freq)

    worker_vectors = []
    for worker_idx, workload in enumerate(workloads):
        worker_df = group[group["workload"] == workload].sort_values("target_freq")
        missing = sorted(set(freqs) - set(worker_df["target_freq"]))
        if missing:
            raise ValueError(f"{workload} is missing frequencies {missing}.")
        if worker_idx < byzantine_workers:
            worker_df = apply_worker_curve_attack(
                worker_df=worker_df,
                attack=attack,
                idle_power=idle_power,
                activity_scale=activity_scale,
                reference_freq=reference_freq,
                max_freq=max(freqs),
            )
        worker_vectors.append(vectorize_worker_curve(worker_df, freqs, reference_freq))

    if attack != "activity_high":
        worker_vectors = apply_attack(worker_vectors, attack, byzantine_workers)
    aggregator = build_aggregator(
        aggregation,
        n_workers=len(worker_vectors),
        byzantine_workers=byzantine_workers,
        clipping_tau=clipping_tau,
        inner_iterations=inner_iterations,
    )

    start = time.time()
    robust_curve = aggregator(worker_vectors).detach().cpu().numpy()
    elapsed = time.time() - start

    n_freqs = len(freqs)
    robust_time_scale = robust_curve[:n_freqs]
    robust_energy_scale = robust_curve[n_freqs:]

    rows = []
    for workload in workloads:
        worker_df = group[group["workload"] == workload].set_index("target_freq")
        time_ref = worker_df.loc[reference_freq, "time_predicted"]
        energy_ref = worker_df.loc[reference_freq, "energy_predicted"]

        for freq_idx, freq in enumerate(freqs):
            rows.append(
                {
                    "workload": workload,
                    "target_freq": freq,
                    "time_predicted": float(time_ref * robust_time_scale[freq_idx]),
                    "energy_predicted": float(
                        energy_ref * robust_energy_scale[freq_idx]
                    ),
                    "walltime": elapsed / max(len(workloads) * len(freqs), 1),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Use workloads as simulated workers and aggregate DVFS prediction curves "
            "with Byzantine-robust optimizers."
        )
    )
    parser.add_argument("--estimated_result_path", type=str, required=True)
    parser.add_argument("--result_save_to", type=str, required=True)
    parser.add_argument("--result_filename", type=str, default="dvfs_byzantine.csv")
    parser.add_argument("--source_estimator", type=str, default="gee")
    parser.add_argument("--output_estimator", type=str, default="gee_bro")
    parser.add_argument(
        "--worker_grouping",
        choices=["all", "model"],
        default="all",
        help=(
            "Use all workloads as one simulated worker pool, or aggregate only "
            "within each model family."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model-name filter, e.g. gpt2model, optmodel, qwen2model.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["mean", "clipping", "cm", "tm", "krum"],
        default="clipping",
    )
    parser.add_argument("--byzantine_workers", type=int, default=0)
    parser.add_argument("--clipping_tau", type=float, default=10.0)
    parser.add_argument("--inner_iterations", type=int, default=5)
    parser.add_argument("--reference_freq", type=int, default=None)
    parser.add_argument("--include_source_estimator", action="store_true", default=False)
    parser.add_argument(
        "--attack",
        choices=["none", "bitflip", "high", "low", "activity_high"],
        default="none",
        help="Optional synthetic attack applied to the first Byzantine worker curves.",
    )
    parser.add_argument("--dvfs_idle_power_json", type=str, default=None)
    parser.add_argument("--attack_activity_scale", type=float, default=4.0)
    args = parser.parse_args()

    estimated = pd.read_csv(args.estimated_result_path)
    source = estimated[estimated["estimator"] == args.source_estimator].copy()
    if source.empty:
        raise ValueError(f"No rows found for estimator {args.source_estimator}.")

    source["model_name"] = source["workload"].apply(parse_model_name)
    if args.model is not None:
        source = source[source["model_name"] == args.model].copy()
        if source.empty:
            raise ValueError(f"No workloads matched model {args.model}.")

    group_key = np.zeros(len(source), dtype=np.int64)
    if args.worker_grouping == "model":
        group_key = source["model_name"]

    result_rows = []
    idle_power = load_idle_power(args.dvfs_idle_power_json)
    for _, group in source.groupby(group_key):
        result_rows.extend(
            robust_group_predictions(
                group=group,
                aggregation=args.aggregation,
                byzantine_workers=args.byzantine_workers,
                clipping_tau=args.clipping_tau,
                inner_iterations=args.inner_iterations,
                reference_freq=args.reference_freq,
                attack=args.attack,
                idle_power=idle_power,
                activity_scale=args.attack_activity_scale,
            )
        )

    robust = pd.DataFrame(result_rows)
    robust["estimator"] = args.output_estimator
    robust = robust[
        [
            "workload",
            "target_freq",
            "estimator",
            "time_predicted",
            "energy_predicted",
            "walltime",
        ]
    ]

    if args.include_source_estimator:
        source = source[
            [
                "workload",
                "target_freq",
                "estimator",
                "time_predicted",
                "energy_predicted",
                "walltime",
            ]
        ]
        robust = pd.concat([source, robust], ignore_index=True)

    os.makedirs(args.result_save_to, exist_ok=True)
    result_path = os.path.join(args.result_save_to, args.result_filename)
    robust.to_csv(result_path, index=False)
    print(f"Saved Byzantine-robust DVFS predictions to {result_path}")


if __name__ == "__main__":
    main()
