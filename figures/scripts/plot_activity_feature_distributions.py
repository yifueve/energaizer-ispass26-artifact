import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUTS = {
    "FP32 GEMM": "database/data/netsres117_sgemm_fp32_lut_v2_prepared.csv",
    "BF16 GEMM": "database/data/netsres117_gemm_bf16_bf16_lut_prepared.csv",
}

OUTPUT_PREFIX = "figures/generated/activity_feature_distributions"


def load_activity_data():
    frames = []
    for label, path in INPUTS.items():
        df = pd.read_csv(path)
        activity_cols = [c for c in df.columns if c.startswith("a_")]
        activity = df[activity_cols].copy()
        activity["dataset"] = label
        frames.append(activity)
    return pd.concat(frames, ignore_index=True)


def module_name(feature):
    parts = feature.split("_")
    if len(parts) < 2:
        return "other"
    return parts[1]


def plot_feature_histograms(df, output_prefix):
    features = [c for c in df.columns if c.startswith("a_")]
    ncols = 5
    nrows = int(np.ceil(len(features) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 2.2 * nrows))
    axes = axes.flatten()

    colors = {
        "FP32 GEMM": "#0173B2",
        "BF16 GEMM": "#DE8F05",
    }

    for idx, feature in enumerate(features):
        ax = axes[idx]
        values = df[feature].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            ax.set_axis_off()
            continue

        upper = values.quantile(0.995)
        if upper <= 0:
            upper = values.max()
        bins = np.linspace(0, upper, 35) if upper > 0 else 10

        for dataset, color in colors.items():
            subset = (
                df.loc[df["dataset"] == dataset, feature]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            subset = subset[(subset >= 0) & (subset <= upper)]
            ax.hist(
                subset,
                bins=bins,
                density=True,
                alpha=0.45,
                color=color,
                label=dataset,
            )

        ax.set_title(feature.replace("a_", ""), fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.6)

    for idx in range(len(features), len(axes)):
        axes[idx].set_axis_off()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Distribution of Prepared Activity Features", y=1.01, fontsize=14)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    fig.savefig(output_prefix + "_histograms.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix + "_histograms.pdf", dpi=300, bbox_inches="tight")
    print(f"Saved {output_prefix}_histograms.png")
    print(f"Saved {output_prefix}_histograms.pdf")


def plot_module_boxplot(df, output_prefix):
    features = [c for c in df.columns if c.startswith("a_")]
    long = df.melt(id_vars=["dataset"], value_vars=features, var_name="feature")
    long["module"] = long["feature"].apply(module_name)
    long = long.replace([np.inf, -np.inf], np.nan).dropna()
    long = long[long["value"] >= 0]

    modules = ["dram", "l2", "smem", "math"]
    datasets = list(INPUTS.keys())

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    positions = []
    labels = []
    data = []
    colors = []
    palette = {
        "FP32 GEMM": "#0173B2",
        "BF16 GEMM": "#DE8F05",
    }

    pos = 1
    for module in modules:
        for dataset in datasets:
            vals = long[(long["module"] == module) & (long["dataset"] == dataset)][
                "value"
            ].to_numpy()
            data.append(vals)
            positions.append(pos)
            labels.append(dataset)
            colors.append(palette[dataset])
            pos += 1
        pos += 0.7

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.65,
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    centers = [
        np.mean(positions[i * len(datasets) : (i + 1) * len(datasets)])
        for i in range(len(modules))
    ]
    ax.set_xticks(centers)
    ax.set_xticklabels([m.upper() if m == "l2" else m.upper() for m in modules])
    ax.set_ylabel("Activity factor value")
    ax.set_title("Activity Feature Distributions by GPU Module", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.8)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[d], alpha=0.65) for d in datasets
    ]
    ax.legend(legend_handles, datasets, frameon=False, loc="upper right")
    fig.tight_layout()

    fig.savefig(output_prefix + "_module_boxplot.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix + "_module_boxplot.pdf", dpi=300, bbox_inches="tight")
    print(f"Saved {output_prefix}_module_boxplot.png")
    print(f"Saved {output_prefix}_module_boxplot.pdf")


def plot_utilization_proxy_distribution(df, output_prefix):
    features = [c for c in df.columns if c.startswith("a_")]
    proxy = df[["dataset"]].copy()
    proxy["activity_intensity"] = (
        df[features].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0).sum(axis=1)
    )
    max_activity = proxy["activity_intensity"].max()
    proxy["activity_intensity_norm"] = proxy["activity_intensity"] / max_activity

    datasets = list(INPUTS.keys())
    palette = {
        "FP32 GEMM": "#0173B2",
        "BF16 GEMM": "#DE8F05",
    }

    bins = np.linspace(0, 1, 45)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9.0, 3.4))

    ax = axes[0]
    for dataset in datasets:
        vals = proxy.loc[proxy["dataset"] == dataset, "activity_intensity_norm"]
        vals = vals[(vals >= 0) & (vals <= 1)]
        ax.hist(
            vals,
            bins=bins,
            density=True,
            alpha=0.45,
            color=palette[dataset],
            label=dataset,
        )
    ax.set_xlabel(r"Normalized activity intensity $(\sum_r a_r)/\max_i\sum_r a_{i,r}$")
    ax.set_ylabel("Density")
    ax.set_title("Distribution")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
    ax.legend(frameon=False)

    ax = axes[1]
    data = [
        proxy.loc[proxy["dataset"] == dataset, "activity_intensity_norm"].to_numpy()
        for dataset in datasets
    ]
    bp = ax.boxplot(data, labels=datasets, patch_artist=True, showfliers=False)
    for patch, dataset in zip(bp["boxes"], datasets):
        patch.set_facecolor(palette[dataset])
        patch.set_alpha(0.65)
    ax.set_ylabel(r"Normalized activity intensity")
    ax.set_title("Summary")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.8)

    fig.suptitle("Module-Level Activity Intensity Proxy", fontsize=13, fontweight="bold")
    fig.tight_layout()

    fig.savefig(output_prefix + "_utilization_proxy.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix + "_utilization_proxy.pdf", dpi=300, bbox_inches="tight")
    proxy.to_csv(output_prefix + "_utilization_proxy.csv", index=False)
    print(f"Saved {output_prefix}_utilization_proxy.png")
    print(f"Saved {output_prefix}_utilization_proxy.pdf")
    print(f"Saved {output_prefix}_utilization_proxy.csv")


def save_summary(df, output_prefix):
    features = [c for c in df.columns if c.startswith("a_")]
    rows = []
    for dataset in INPUTS:
        part = df[df["dataset"] == dataset]
        for feature in features:
            vals = part[feature].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "dataset": dataset,
                    "feature": feature,
                    "module": module_name(feature),
                    "count": len(vals),
                    "mean": vals.mean(),
                    "std": vals.std(),
                    "p05": vals.quantile(0.05),
                    "p50": vals.quantile(0.50),
                    "p95": vals.quantile(0.95),
                    "max": vals.max(),
                }
            )
    summary = pd.DataFrame(rows)
    path = output_prefix + "_summary.csv"
    summary.to_csv(path, index=False)
    print(f"Saved {path}")


def main():
    df = load_activity_data()
    plot_feature_histograms(df, OUTPUT_PREFIX)
    plot_module_boxplot(df, OUTPUT_PREFIX)
    plot_utilization_proxy_distribution(df, OUTPUT_PREFIX)
    save_summary(df, OUTPUT_PREFIX)


if __name__ == "__main__":
    main()
