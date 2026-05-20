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


def add_power_columns(df):
    df = df.copy()
    for prefix in ["gee", "gee_attack_mean", "gee_attack_bro"]:
        df[f"{prefix}_power"] = df[f"{prefix}_energy"] / df[f"{prefix}_time"] * 1000.0
    df["measured_power"] = df["measured_energy"] / df["measured_time"] * 1000.0
    return df


def mape(predicted, truth):
    return (np.abs(predicted - truth) / truth * 100.0).mean()


def plot(df, output_path):
    df = add_power_columns(df)
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
        ax.set_ylim(0, 420)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

        if subset.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        ax.scatter(
            subset["freq"],
            subset["measured_power"],
            marker="o",
            color="#0173B2",
            s=42,
            label="Measured",
            zorder=4,
        )
        ax.plot(
            subset["freq"],
            subset["gee_power"],
            marker="^",
            color="#DE8F05",
            linewidth=1.6,
            markersize=5,
            label="Clean GEE",
            zorder=3,
        )
        ax.plot(
            subset["freq"],
            subset["gee_attack_mean_power"],
            marker="x",
            color="#D55E00",
            linewidth=1.8,
            markersize=6,
            label="Attacked mean",
            zorder=2,
        )
        ax.plot(
            subset["freq"],
            subset["gee_attack_bro_power"],
            marker="s",
            color="#029E73",
            linewidth=1.8,
            markersize=5,
            label="Attacked BRO",
            zorder=3,
        )

        max_freq = subset["freq"].max()
        ax.axvline(x=max_freq, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.axvspan(max_freq, 1411, color="gray", alpha=0.12)

        if idx % 2 == 0:
            ax.set_ylabel("Power (W)", fontsize=11)
        if idx // 2 == 2:
            ax.set_xlabel("Frequency (MHz)", fontsize=11)

        ax.set_title(
            f"{MODEL_NAME_MAP[model]}, batch={batch}, seq={seq}",
            fontsize=11,
            fontweight="bold",
        )
        ax.text(
            0.04,
            0.94,
            "MAPE\n"
            f"GEE {mape(subset['gee_power'], subset['measured_power']):.1f}%\n"
            f"Mean {mape(subset['gee_attack_mean_power'], subset['measured_power']):.1f}%\n"
            f"BRO {mape(subset['gee_attack_bro_power'], subset['measured_power']):.1f}%",
            transform=ax.transAxes,
            fontsize=8.5,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=10.5,
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
        "--report",
        default="figures/fig11_dvfs_results/dvfs_byzantine_attack_demo_report.csv",
    )
    parser.add_argument(
        "--output",
        default="figures/generated/fig11_dvfs_byzantine_attack_demo",
    )
    args = parser.parse_args()

    plot(pd.read_csv(args.report), args.output)


if __name__ == "__main__":
    main()
