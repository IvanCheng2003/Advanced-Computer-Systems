import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "data/results.csv"
OUTDIR = "plots/out"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Aggregate reps
gcols = ["impl", "workload", "nkeys", "threads"]
agg = df.groupby(gcols)["ops_per_s"].agg(["mean", "std", "count"]).reset_index()
agg["std"] = agg["std"].fillna(0.0)

def savefig(name: str):
    path = os.path.join(OUTDIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"wrote {path}")

# --- Plot 1: Throughput vs Threads (per workload, per nkeys) ---
for workload in sorted(agg["workload"].unique()):
    for nkeys in sorted(agg["nkeys"].unique()):
        sub = agg[(agg["workload"] == workload) & (agg["nkeys"] == nkeys)].copy()
        if sub.empty:
            continue

        plt.figure()
        for impl in ["coarse", "striped"]:
            s2 = sub[sub["impl"] == impl].sort_values("threads")
            if s2.empty:
                continue
            plt.errorbar(
                s2["threads"], s2["mean"], yerr=s2["std"],
                marker="o", capsize=3, label=impl
            )

        plt.xlabel("Threads")
        plt.ylabel("Throughput (ops/s)")
        plt.title(f"Throughput vs Threads — workload={workload}, nkeys={nkeys}")
        plt.legend()
        savefig(f"throughput_threads_{workload}_nkeys{nkeys}.png")

# --- Plot 2: Speedup vs Threads (per workload, per nkeys, per impl) ---
# speedup(t) = mean_ops(t) / mean_ops(1)
for workload in sorted(agg["workload"].unique()):
    for nkeys in sorted(agg["nkeys"].unique()):
        for impl in ["coarse", "striped"]:
            sub = agg[(agg["workload"] == workload) & (agg["nkeys"] == nkeys) & (agg["impl"] == impl)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("threads")
            base = sub[sub["threads"] == 1]["mean"]
            if base.empty:
                continue
            base = float(base.iloc[0])
            sub["speedup"] = sub["mean"] / base

            plt.figure()
            plt.plot(sub["threads"], sub["speedup"], marker="o")
            plt.xlabel("Threads")
            plt.ylabel("Speedup vs 1 thread")
            plt.title(f"Speedup vs Threads — impl={impl}, workload={workload}, nkeys={nkeys}")
            savefig(f"speedup_threads_{impl}_{workload}_nkeys{nkeys}.png")

# --- Plot 3: Coarse vs Striped ratio (how much better is striped) ---
# ratio = striped_mean / coarse_mean at each (workload, nkeys, threads)
pivot = agg.pivot_table(index=["workload","nkeys","threads"], columns="impl", values="mean").reset_index()
if "coarse" in pivot.columns and "striped" in pivot.columns:
    pivot["striped_over_coarse"] = pivot["striped"] / pivot["coarse"]

    for workload in sorted(pivot["workload"].unique()):
        for nkeys in sorted(pivot["nkeys"].unique()):
            sub = pivot[(pivot["workload"] == workload) & (pivot["nkeys"] == nkeys)].sort_values("threads")
            if sub.empty:
                continue
            plt.figure()
            plt.plot(sub["threads"], sub["striped_over_coarse"], marker="o")
            plt.axhline(1.0, linestyle="--")
            plt.xlabel("Threads")
            plt.ylabel("Striped / Coarse throughput ratio")
            plt.title(f"Benefit of Striped Locking — workload={workload}, nkeys={nkeys}")
            savefig(f"ratio_striped_over_coarse_{workload}_nkeys{nkeys}.png")
else:
    print("Could not compute ratio plot (missing coarse/striped in data).")
