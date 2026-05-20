import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIGURATIONS = [
    ("gpt2model", "gpt2", 8, 128, "prefill"),
    ("gpt2model", "gpt2", 32, 512, "prefill"),
    ("optmodel", "opt-1p3b", 2, 128, "prefill"),
    ("optmodel", "opt-1p3b", 8, 2048, "prefill"),
    ("qwen2model", "qwen2-1p5b", 1, 128, "prefill"),
    ("qwen2model", "qwen2-1p5b", 1, 4096, "prefill"),
]

MODEL_NAME_MAP = {
    "gpt2model": "GPT2",
    "optmodel": "OPT",
    "qwen2model": "Qwen2",
}

KEY_COLUMNS = ["model", "batch", "seq", "mode", "config", "prec", "freq"]


def add_power_columns(df, prefix):
    df = df.copy()
    df[f"{prefix}_power"] = df[f"{prefix}_energy"] / df[f"{prefix}_time"] * 1000.0
    return df


def load_reports(baseline_path, robust_path):
    baseline = pd.read_csv(baseline_path)
    robust = pd.read_csv(robust_path)

    baseline["measured_power"] = (
        baseline["measured_energy"] / baseline["measured_time"] * 1000.0
    )
    baseline = add_power_columns(baseline, "gee")
    robust = add_power_columns(robust, "gee_bro")

    robust_cols = KEY_COLUMNS + ["gee_bro_time", "gee_bro_energy", "gee_bro_power"]
    return baseline.merge(robust[robust_cols], on=KEY_COLUMNS, how="left")


def plot_dvfs(df, output_path):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8.5, 8.0), sharex=True)
    axes = axes.flatten()

    for idx, (model, config, batch, seq, mode) in enumerate(CONFIGURATIONS):
        ax = axes[idx]
        subset = df[
            (df["model"] == model)
            & (df["batch"] == batch)
            & (df["seq"] == seq)
            & (df["mode"] == mode)
            & (df["config"] == config)
        ].copy()

        subset = subset[np.abs(subset["measured_freq"] - subset["freq"]) <= 90]
        subset = subset.sort_values("freq")

        ax.set_xlim(509, 1411)
        ax.set_ylim(50, 260)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

        if subset.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        baseline_mape = (
            np.abs(subset["gee_power"] - subset["measured_power"])
            / subset["measured_power"]
            * 100.0
        ).mean()
        robust_mape = (
            np.abs(subset["gee_bro_power"] - subset["measured_power"])
            / subset["measured_power"]
            * 100.0
        ).mean()

        ax.scatter(
            subset["freq"],
            subset["measured_power"],
            marker="o",
            color="#0173B2",
            s=48,
            label="Measured",
            zorder=3,
        )
        ax.plot(
            subset["freq"],
            subset["gee_power"],
            marker="^",
            color="#DE8F05",
            linewidth=1.8,
            markersize=6,
            label="GEE",
            zorder=3,
        )
        ax.plot(
            subset["freq"],
            subset["gee_bro_power"],
            marker="s",
            color="#029E73",
            linewidth=1.8,
            markersize=5,
            label="GEE + BRO",
            zorder=3,
        )

        max_freq = subset["freq"].max()
        ax.axvline(
            x=max_freq, color="gray", linestyle="--", linewidth=1.4, alpha=0.7, zorder=2
        )
        ax.axvspan(max_freq, 1411, color="gray", alpha=0.15, zorder=1)

        if idx % 2 == 0:
            ax.set_ylabel("Power (W)", fontsize=11)
        if idx // 2 == 2:
            ax.set_xlabel("Frequency (MHz)", fontsize=11)

        title = f"{MODEL_NAME_MAP[model]}, batch={batch}, seq={seq}"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(
            0.04,
            0.94,
            f"GEE {baseline_mape:.1f}%\nBRO {robust_mape:.1f}%",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(output_path + ".pdf", dpi=300, bbox_inches="tight")
    print(f"Saved {output_path}.png")
    print(f"Saved {output_path}.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_report",
        default="figures/fig11_dvfs_results/dvfs_report.csv",
    )
    parser.add_argument(
        "--robust_report",
        default="figures/fig11_dvfs_results/dvfs_byzantine_robust_report.csv",
    )
    parser.add_argument(
        "--output",
        default="figures/generated/fig11_dvfs_byzantine_robust",
    )
    args = parser.parse_args()

    df = load_reports(args.baseline_report, args.robust_report)
    plot_dvfs(df, args.output)


if __name__ == "__main__":
    main()
