"""Holm-Bonferroni correction for the paired bootstrap comparisons.

`bootstrap_step_metrics` and `bootstrap_causal_metrics` call a pair of heads
significantly different when its 95% percentile interval excludes zero. With six
heads that is 15 comparisons per metric, so at least one false positive is
expected under the null far more often than 5% of the time. This module turns
each comparison into a two-sided bootstrap p-value and applies Holm's step-down
correction within each metric, which controls the family-wise error rate at the
chosen alpha without assuming the comparisons are independent.

It reads the `*_pairwise.csv` files those scripts already wrote and needs no
model, cache or GPU. The original CSV is left untouched; the corrected table is
written next to it with a `_holm` suffix.

The p-value comes from the share of resamples in which head A beat head B:
    p = min(1, 2 * min(k + 1, B - k + 1) / (B + 1)),  k = round(P(A > B) * B)
The +1 terms keep p above zero when every resample agrees, so the smallest
reportable p-value with B = 2000 is about 0.001.
"""

import argparse
import csv
import os

PROB_SUFFIX = "_prob_a_better"


def bootstrap_p_value(prob_a_better, n_samples):
    """Two-sided p-value from the fraction of resamples in which A beat B."""
    if prob_a_better != prob_a_better:  # NaN: no finite resample
        return float("nan")
    wins = round(prob_a_better * n_samples)
    tail = min(wins + 1, n_samples - wins + 1)
    return min(1.0, 2.0 * tail / (n_samples + 1))


def holm_adjust(p_values):
    """Holm step-down adjusted p-values, returned in the input order.

    NaN p-values are left as NaN and do not count towards the family size.
    """
    indexed = [(p, i) for i, p in enumerate(p_values) if p == p]
    indexed.sort()
    m = len(indexed)
    adjusted = [float("nan")] * len(p_values)
    running_max = 0.0
    for rank, (p, i) in enumerate(indexed):
        running_max = max(running_max, min(1.0, (m - rank) * p))
        adjusted[i] = running_max
    return adjusted


def correct_pairwise_rows(rows, n_samples, alpha=0.05):
    """Add p-value, Holm-adjusted p-value and Holm significance per metric."""
    metrics = [c[: -len(PROB_SUFFIX)] for c in rows[0] if c.endswith(PROB_SUFFIX)]
    out = [dict(row) for row in rows]
    for metric in metrics:
        p_values = [
            bootstrap_p_value(float(row[metric + PROB_SUFFIX]), n_samples) for row in rows
        ]
        adjusted = holm_adjust(p_values)
        for row, p, p_adj in zip(out, p_values, adjusted):
            row[metric + "_p_value"] = p
            row[metric + "_p_holm"] = p_adj
            row[metric + "_significant_holm"] = bool(p_adj < alpha)
    return out, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pairwise_csv", nargs="+",
                        help="e.g. results_conv/step_metrics_ci_pairwise.csv")
    parser.add_argument("--bootstrap_samples", type=int, default=2000,
                        help="must match the run that wrote the CSV")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    for path in args.pairwise_csv:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print("[skip] " + path + " is empty")
            continue
        corrected, metrics = correct_pairwise_rows(rows, args.bootstrap_samples, args.alpha)
        out_path = os.path.splitext(path)[0] + "_holm.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(corrected[0].keys()))
            writer.writeheader()
            writer.writerows(corrected)

        print("[saved] " + out_path)
        for metric in metrics:
            before = sum(row[metric + "_significant"] == "True" for row in rows)
            after = sum(row[metric + "_significant_holm"] for row in corrected)
            print(f"  {metric}: {before}/{len(rows)} significant by CI, "
                  f"{after}/{len(rows)} after Holm at alpha={args.alpha}")
            for row in corrected:
                if (row[metric + "_significant"] == "True") != row[metric + "_significant_holm"]:
                    print(f"    changed: {row['head_a']} vs {row['head_b']} "
                          f"(p={row[metric + '_p_value']:.4f}, "
                          f"p_holm={row[metric + '_p_holm']:.4f})")


if __name__ == "__main__":
    main()
