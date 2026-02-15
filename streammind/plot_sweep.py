# streammind/plot_sweep.py
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df["k"] = df["k"].astype(int)
    df["wall_ms"] = df["wall_ms"].astype(float)
    df["gpu_util_pct"] = df["gpu_util_pct"].astype(float)

    g = df.groupby("k").agg(
        wall_ms_mean=("wall_ms", "mean"),
        wall_ms_std=("wall_ms", "std"),
        gpu_util_mean=("gpu_util_pct", "mean"),
        gpu_util_std=("gpu_util_pct", "std"),
    ).reset_index().sort_values("k")

    # 1) latency vs K
    plt.figure()
    plt.errorbar(g["k"], g["wall_ms_mean"], yerr=g["wall_ms_std"])
    plt.xlabel("Retrieved tokens (K)")
    plt.ylabel("Latency per call (ms)")
    plt.title("Latency vs Retrieved tokens (K)")
    p1 = os.path.join(args.outdir, "latency_vs_k.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()

    # 2) gpu util vs K  (gpu_ms / wall_ms * 100)
    plt.figure()
    plt.errorbar(g["k"], g["gpu_util_mean"], yerr=g["gpu_util_std"])
    plt.xlabel("Retrieved tokens (K)")
    plt.ylabel("GPU active ratio (%)")
    plt.title("GPU active ratio vs Retrieved tokens (K)")
    p2 = os.path.join(args.outdir, "gpuutil_vs_k.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()

    # summary csv
    g.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)

    print("[OK] plots:")
    print(" -", p1)
    print(" -", p2)
    print("[OK] summary:", os.path.join(args.outdir, "summary.csv"))


if __name__ == "__main__":
    main()
