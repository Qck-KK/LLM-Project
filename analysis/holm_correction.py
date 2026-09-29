"""Holm-Bonferroni correction for the paired bootstrap comparisons."""

import argparse
import csv
import os

PROB_SUFFIX = "_prob_a_better"
WORSE_SUFFIX = "_prob_a_worse"
FAMILIES = ("all", "prefix", "suffix")


def bootstrap_p_value(prob_a_better, n_samples, prob_a_worse=None):
    """Two-sided p-value from the shares of resamples in which A beat / lost to B."""
    if prob_a_better != prob_a_better:  # NaN: no finite resample
        return float("nan")
    wins = round(prob_a_better * n_samples)
    losses = (n_samples - wins if prob_a_worse is None or prob_a_worse != prob_a_worse
              else round(prob_a_worse * n_samples))
    at_least = n_samples - losses   # difference >= 0
    at_most = n_samples - wins      # difference <= 0
    return min(1.0, 2.0 * min(at_least + 1, at_most + 1) / (n_samples + 1))


def holm_adjust(p_values):
    """Holm step-down adjusted p-values, returned in the input order."""
    indexed = [(p, i) for i, p in enumerate(p_values) if p == p]
    indexed.sort()
    m = len(indexed)
    adjusted = [float("nan")] * len(p_values)
    running_max = 0.0
    for rank, (p, i) in enumerate(indexed):
        running_max = max(running_max, min(1.0, (m - rank) * p))
        adjusted[i] = running_max
    return adjusted


def pair_names(row):
    return row.get("head_a", row.get("arm_a")), row.get("head_b", row.get("arm_b"))


def in_family(row, family):
    if family == "all":
        return True
    a, b = pair_names(row)
    side = 0 if family == "prefix" else 1
    return (len(a.rsplit("_", 1)) == 2 and len(b.rsplit("_", 1)) == 2
            and a.rsplit("_", 1)[side] == b.rsplit("_", 1)[side])


def correct_pairwise_rows(rows, n_samples, alpha=0.05, family="all"):
    """Add p-value, Holm-adjusted p-value and Holm significance per metric."""
    metrics = [c[: -len(PROB_SUFFIX)] for c in rows[0] if c.endswith(PROB_SUFFIX)]
    out = [dict(row) for row in rows]
    members = [i for i, row in enumerate(rows) if in_family(row, family)]
    for metric in metrics:
        p_values = []
        for row in rows:
            worse = row.get(metric + WORSE_SUFFIX)
            p_values.append(bootstrap_p_value(
                float(row[metric + PROB_SUFFIX]), n_samples,
                None if worse in (None, "") else float(worse)))
        adjusted = holm_adjust([p_values[i] for i in members])
        for row, p in zip(out, p_values):
            row[metric + "_p_value"] = p
            row[metric + "_p_holm"] = ""
            row[metric + "_significant_holm"] = ""
        for i, p_adj in zip(members, adjusted):
            out[i][metric + "_p_holm"] = p_adj
            out[i][metric + "_significant_holm"] = bool(p_adj < alpha)
    return out, metrics, len(members)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pairwise_csv", nargs="+",
                        help="e.g. results_final/step_metrics_ci_pairwise.csv")
    parser.add_argument("--bootstrap_samples", type=int, default=2000,
                        help="must match the run that wrote the CSV")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--family", choices=FAMILIES, default="all")
    args = parser.parse_args()

    for path in args.pairwise_csv:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print("[skip] " + path + " is empty")
            continue
        corrected, metrics, n_family = correct_pairwise_rows(
            rows, args.bootstrap_samples, args.alpha, args.family)
        out_path = os.path.splitext(path)[0] + "_holm.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(corrected[0].keys()))
            writer.writeheader()
            writer.writerows(corrected)

        print("[saved] {0}  (family={1}: {2} of {3} pairs)".format(
            out_path, args.family, n_family, len(rows)))
        for metric in metrics:
            family_rows = [(r, c) for r, c in zip(rows, corrected)
                           if c[metric + "_significant_holm"] != ""]
            before = sum(r[metric + "_significant"] == "True" for r, _ in family_rows)
            after = sum(c[metric + "_significant_holm"] for _, c in family_rows)
            print(f"  {metric}: {before}/{len(family_rows)} significant by CI, "
                  f"{after}/{len(family_rows)} after Holm at alpha={args.alpha}")
            for r, c in family_rows:
                if (r[metric + "_significant"] == "True") != c[metric + "_significant_holm"]:
                    a, b = pair_names(r)
                    print(f"    changed: {a} vs {b} "
                          f"(p={c[metric + '_p_value']:.4f}, "
                          f"p_holm={c[metric + '_p_holm']:.4f})")


if __name__ == "__main__":
    main()
