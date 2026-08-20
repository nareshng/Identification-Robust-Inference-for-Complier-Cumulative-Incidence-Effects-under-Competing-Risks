
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS = Path(
    "extended_results_final/extended_simulation_results.csv"
)
REPLICATES = Path(
    "extended_results_final/extended_replicate_results.csv"
)

OUT = Path("figures")


def setup():
    OUT.mkdir(exist_ok=True)
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    })


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


def local_coverage(results):
    dat = results.query("scenario == 'local_to_zero'").copy()
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    markers = ["o", "s", "^", "D"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75))
    for (n, group), color, marker in zip(dat.groupby("n"), colors, markers):
        group = group.sort_values("local_c")
        axes[0].errorbar(group.local_c, group.score_coverage,
                         yerr=1.96*group.score_coverage_mcse,
                         color=color, marker=marker, lw=1.3, capsize=2,
                         label=f"n={n:,}")
        axes[1].plot(group.local_c, group.score_full_set,
                     color=color, marker=marker, lw=1.3, label=f"n={n:,}")
    axes[0].axhline(.95, color="0.25", ls="--", lw=1, label="nominal 0.95")
    axes[0].set(xlabel=r"Local strength $c=\sqrt{n}\,\kappa_n$",
                ylabel="Score-set coverage", ylim=(.925, .975),
                xticks=[.5, 1, 2, 4])
    axes[0].grid(axis="y", color="0.9", lw=.6)
    axes[1].set(xlabel=r"Local strength $c=\sqrt{n}\,\kappa_n$",
                ylabel=r"Probability of full $[-1,1]$ set", ylim=(-.03, 1.0),
                xticks=[.5, 1, 2, 4])
    axes[1].grid(axis="y", color="0.9", lw=.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", frameon=False,
               bbox_to_anchor=(.5, 1.05))
    fig.subplots_adjust(top=.78, wspace=.34)
    save(fig, "local_coverage_informativeness")


def denominator_instability(results, replicates):
    dat = replicates.query("scenario == 'core' and n == 1000 and censoring == 0.30").copy()
    colors = {0.05:"#D55E00", 0.15:"#0072B2", 0.35:"#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75))
    for delta, group in dat.groupby("delta"):
        vals = np.sort(group.wald_length.replace([np.inf, -np.inf], np.nan).dropna().to_numpy())
        y = np.arange(1, len(vals)+1)/len(vals)
        axes[0].step(vals, y, where="post", lw=1.4, color=colors[delta],
                     label=rf"$\delta={delta:.2f}$")
    axes[0].axvline(2, color="0.25", ls="--", lw=1)
    axes[0].text(2.18, .18, "entire feasible\nscore range", fontsize=7.5, color="0.25")
    axes[0].set_xscale("log")
    axes[0].set(xlabel="Wald interval length (log scale)", ylabel="Empirical CDF",
                ylim=(0, 1.01))
    axes[0].grid(axis="y", color="0.9", lw=.6)
    axes[0].legend(frameon=False, loc="lower right")

    core = results.query("scenario == 'core' and n == 1000 and censoring == 0.30").sort_values("mean_first_stage")
    x = core.mean_first_stage.to_numpy()
    axes[1].plot(x, core.wald_median_length, "o-", color="#D55E00", lw=1.4,
                 label="Wald median")
    axes[1].plot(x, core.wald_p99_length, "o--", color="#D55E00", lw=1.4,
                 label="Wald 99th percentile")
    # Compute exact score p99 from the saved replicate-level data.
    score_q = dat.groupby("delta").score_length.quantile([.5,.99]).unstack().reindex(core.delta)
    axes[1].plot(x, score_q[.5], "s-", color="#0072B2", lw=1.4,
                 label="Score median")
    axes[1].plot(x, score_q[.99], "s--", color="#0072B2", lw=1.4,
                 label="Score 99th percentile")
    axes[1].set_yscale("log")
    axes[1].set(xlabel=r"Mean estimated first stage $\widehat\kappa$",
                ylabel="Length (log scale)")
    axes[1].grid(axis="y", color="0.9", lw=.6)
    axes[1].legend(frameon=False, ncol=2, loc="lower center",
                   bbox_to_anchor=(.5,1.01), columnspacing=.8)
    fig.subplots_adjust(wspace=.34, top=.82)
    save(fig, "denominator_instability")


def topology(results):
    dat = results.query("scenario == 'core'").copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), sharey=True)
    colors = {"Ordinary interval":"#56B4E9", "Disconnected":"#E69F00",
              "Full set":"#009E73", "Empty":"#D55E00"}
    for ax, censoring in zip(axes, [0.0, .30]):
        g = dat.query("censoring == @censoring").sort_values(["delta","n"])
        full = g.score_full_set.to_numpy()
        disc = g.score_disconnected.to_numpy()
        empty = g.score_empty.to_numpy()
        ordinary = np.maximum(1-full-disc-empty, 0)
        bottom = np.zeros(len(g))
        for label, values in [("Ordinary interval",ordinary),("Disconnected",disc),
                              ("Full set",full),("Empty",empty)]:
            ax.bar(np.arange(len(g)), values, bottom=bottom, width=.72,
                   color=colors[label], edgecolor="white", linewidth=.3, label=label)
            bottom += values
        labels=[rf"{d:.2f}"+"\n"+f"{int(n)}" for d,n in zip(g.delta,g.n)]
        ax.set_xticks(np.arange(len(g)), labels)
        ax.set_xlabel(r"Instrument increment $\delta$ / sample size $n$")
        ax.set_title(f"{int(censoring*100)}% censoring")
        ax.grid(axis="y", color="0.92", lw=.5)
    axes[0].set_ylabel("Frequency of score-set topology")
    axes[0].set_ylim(0,1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc="upper center",frameon=False,
               bbox_to_anchor=(.5,1.04))
    fig.subplots_adjust(top=.78,wspace=.12)
    save(fig,"score_set_topology")


def main():
    setup()
    results = pd.read_csv(RESULTS)
    replicates = pd.read_csv(REPLICATES)
    local_coverage(results)
    denominator_instability(results, replicates)
    topology(results)


if __name__ == "__main__":
    main()
