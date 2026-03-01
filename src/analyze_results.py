#!/usr/bin/env python3
"""Analyze experiment outputs and create summary metrics + plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def paired_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    denom = d.std(ddof=1)
    if denom == 0:
        return 0.0
    return float(d.mean() / denom)


def bootstrap_ci_mean_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 4000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    diffs = x - y
    boots = []
    n = len(diffs)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(np.mean(diffs[idx])))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


def summarize_task(df: pd.DataFrame, task: str, metric: str) -> pd.DataFrame:
    sub = df[df["task"] == task]
    out = (
        sub.groupby("condition")[metric]
        .agg(["mean", "std", "count", "min", "max"])
        .reset_index()
        .sort_values("condition")
    )
    return out


def paired_tests(df: pd.DataFrame, task: str, metric: str) -> list[dict[str, Any]]:
    sub = df[df["task"] == task]
    piv = sub.pivot_table(index="item_id", columns="condition", values=metric, aggfunc="first").dropna()

    pairs = [
        ("think_between_sentences", "direct"),
        ("think_between_sentences", "pause_control"),
        ("pause_control", "direct"),
    ]
    rows: list[dict[str, Any]] = []
    raw_p = []

    for a, b in pairs:
        x = piv[a].to_numpy(dtype=float)
        y = piv[b].to_numpy(dtype=float)
        diff = x - y

        if np.allclose(diff, 0.0):
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "comparison": f"{a} - {b}",
                    "n": len(x),
                    "mean_diff": 0.0,
                    "ci95_low": 0.0,
                    "ci95_high": 0.0,
                    "test": "constant_zero_difference",
                    "statistic": 0.0,
                    "p_value": 1.0,
                    "shapiro_p": 1.0,
                    "cohens_d_paired": 0.0,
                }
            )
            raw_p.append(1.0)
            continue

        shapiro_p = stats.shapiro(diff).pvalue if len(diff) >= 3 else 1.0
        normal = shapiro_p > 0.05

        if normal:
            stat, p = stats.ttest_rel(x, y)
            test_name = "paired_t_test"
        else:
            stat, p = stats.wilcoxon(x, y, zero_method="wilcox", correction=False)
            test_name = "wilcoxon_signed_rank"

        d = paired_cohens_d(x, y)
        ci_lo, ci_hi = bootstrap_ci_mean_diff(x, y)

        rows.append(
            {
                "task": task,
                "metric": metric,
                "comparison": f"{a} - {b}",
                "n": len(x),
                "mean_diff": float(np.mean(diff)),
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
                "test": test_name,
                "statistic": float(stat),
                "p_value": float(p),
                "shapiro_p": float(shapiro_p),
                "cohens_d_paired": d,
            }
        )
        raw_p.append(float(p))

    reject, p_corr, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
    for i, row in enumerate(rows):
        row["p_value_holm"] = float(p_corr[i])
        row["reject_h0_holm_0_05"] = bool(reject[i])

    return rows


def mcnemar_binary(df: pd.DataFrame, task: str, metric: str) -> list[dict[str, Any]]:
    sub = df[df["task"] == task]
    piv = sub.pivot_table(index="item_id", columns="condition", values=metric, aggfunc="first").dropna()

    pairs = [
        ("think_between_sentences", "direct"),
        ("think_between_sentences", "pause_control"),
        ("pause_control", "direct"),
    ]
    rows = []
    raw_p = []
    for a, b in pairs:
        a1 = piv[a].astype(int)
        b1 = piv[b].astype(int)
        n01 = int(((a1 == 0) & (b1 == 1)).sum())
        n10 = int(((a1 == 1) & (b1 == 0)).sum())
        table = [[0, n01], [n10, 0]]
        res = mcnemar(table, exact=True)
        rows.append(
            {
                "task": task,
                "metric": metric,
                "comparison": f"{a} vs {b}",
                "n": int(len(piv)),
                "n01": n01,
                "n10": n10,
                "p_value": float(res.pvalue),
            }
        )
        raw_p.append(float(res.pvalue))

    reject, p_corr, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
    for i, row in enumerate(rows):
        row["p_value_holm"] = float(p_corr[i])
        row["reject_h0_holm_0_05"] = bool(reject[i])
    return rows


def flatten_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    def numeric_equal(pred: Any, gold: Any) -> int:
        try:
            if pred is None or gold is None:
                return 0
            return int(abs(float(str(pred)) - float(str(gold))) < 1e-9)
        except (TypeError, ValueError):
            return 0

    out = []
    for r in rows:
        base = {k: v for k, v in r.items() if k != "moderation"}
        mod = r.get("moderation") or {}
        base["flagged"] = r.get("flagged", 0)
        base["max_category_score"] = r.get("max_category_score", 0.0)
        base["truthful"] = r.get("truthful")
        base["carefulness"] = r.get("carefulness")
        if r.get("task") == "gsm8k":
            base["exact_match"] = numeric_equal(r.get("prediction"), r.get("gold"))
        else:
            base["exact_match"] = r.get("exact_match")
        out.append(base)
    return pd.DataFrame(out)


def save_plot(df: pd.DataFrame, task: str, metric: str, out_dir: Path) -> None:
    sub = df[df["task"] == task]
    plt.figure(figsize=(8, 5))
    sns.barplot(data=sub, x="condition", y=metric, estimator=np.mean, errorbar=("ci", 95))
    plt.title(f"{task}: {metric} by condition")
    plt.xlabel("Condition")
    plt.ylabel(metric)
    plt.xticks(rotation=15)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{task}_{metric}.png", dpi=150)
    plt.close()


def main() -> None:
    raw_path = Path("results/raw_outputs.jsonl")
    rows = load_jsonl(raw_path)
    df = flatten_rows(rows)

    out_dir = Path("results")
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_tables = {}
    summary_tables["truthfulqa_truthful"] = summarize_task(df, "truthfulqa", "truthful").to_dict(orient="records")
    summary_tables["truthfulqa_carefulness"] = summarize_task(df, "truthfulqa", "carefulness").to_dict(orient="records")
    summary_tables["gsm8k_exact_match"] = summarize_task(df, "gsm8k", "exact_match").to_dict(orient="records")
    summary_tables["rtp_flagged"] = summarize_task(df, "rtp", "flagged").to_dict(orient="records")
    summary_tables["rtp_max_category_score"] = summarize_task(df, "rtp", "max_category_score").to_dict(orient="records")
    summary_tables["all_word_count"] = df.groupby(["task", "condition"])["word_count"].mean().reset_index().to_dict(orient="records")
    summary_tables["all_hedge_count"] = df.groupby(["task", "condition"])["hedge_count"].mean().reset_index().to_dict(orient="records")

    pairwise = []
    pairwise += paired_tests(df, "truthfulqa", "carefulness")
    pairwise += paired_tests(df, "rtp", "max_category_score")
    pairwise += paired_tests(df, "gsm8k", "exact_match")

    binary_tests = []
    binary_tests += mcnemar_binary(df, "truthfulqa", "truthful")
    binary_tests += mcnemar_binary(df, "gsm8k", "exact_match")
    binary_tests += mcnemar_binary(df, "rtp", "flagged")

    save_plot(df, "truthfulqa", "truthful", plot_dir)
    save_plot(df, "truthfulqa", "carefulness", plot_dir)
    save_plot(df, "gsm8k", "exact_match", plot_dir)
    save_plot(df, "rtp", "flagged", plot_dir)
    save_plot(df, "rtp", "max_category_score", plot_dir)
    save_plot(df, "truthfulqa", "hedge_count", plot_dir)

    summary = {
        "n_rows": int(len(df)),
        "conditions": sorted(df["condition"].dropna().unique().tolist()),
        "tasks": sorted(df["task"].dropna().unique().tolist()),
        "summary_tables": summary_tables,
        "pairwise_tests": pairwise,
        "binary_tests": binary_tests,
    }

    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df.to_csv(out_dir / "flat_outputs.csv", index=False)
    pd.DataFrame(pairwise).to_csv(out_dir / "pairwise_tests.csv", index=False)
    pd.DataFrame(binary_tests).to_csv(out_dir / "binary_tests.csv", index=False)

    print("Saved:")
    print("- results/metrics_summary.json")
    print("- results/flat_outputs.csv")
    print("- results/pairwise_tests.csv")
    print("- results/binary_tests.csv")
    print("- results/plots/*.png")


if __name__ == "__main__":
    main()
