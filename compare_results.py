import os
import re

RESULTS_DIR = "results"
BASELINE_LOG = os.path.join(RESULTS_DIR, "baseline_log.txt")
TRAIN_LOG    = os.path.join(RESULTS_DIR, "train_log.txt")
TRAIN_TS_LOG = os.path.join(RESULTS_DIR, "train_ts_log.txt")
OUT_PATH     = os.path.join(RESULTS_DIR, "comparison.txt")

_EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s*\|.*?Val AUC\s+([\d.nan]+)", re.IGNORECASE)


def _parse_log(path):
    """Return {epoch: auc} dict from a log file. Empty dict if file missing."""
    if not os.path.exists(path):
        return {}
    epochs = {}
    with open(path) as f:
        for line in f:
            m = _EPOCH_RE.search(line)
            if m:
                ep = int(m.group(1))
                auc_str = m.group(2)
                epochs[ep] = float("nan") if auc_str.lower() == "nan" else float(auc_str)
    return epochs


def _best(epoch_dict):
    vals = [v for v in epoch_dict.values() if v == v]  # exclude nan
    return max(vals) if vals else float("nan")


def main():
    baseline = _parse_log(BASELINE_LOG)
    hu       = _parse_log(TRAIN_LOG)
    ts       = _parse_log(TRAIN_TS_LOG)

    has_ts = bool(ts)
    all_epochs = sorted(set(list(baseline.keys()) + list(hu.keys()) + list(ts.keys())))

    if not all_epochs:
        print("No log files found. Run baseline.py, train.py, and train_ts.py first.")
        return

    # Build table
    header_ts = " TS Mask AUC" if has_ts else ""
    sep_ts    = "-------------" if has_ts else ""
    header = f"Epoch | Baseline AUC | HU Mask AUC |{header_ts}"
    sep    = f"------|-------------|-------------|{sep_ts}"

    lines = [
        "=== RESULTS COMPARISON ===",
        header,
        sep,
    ]

    def _fmt(d, ep):
        v = d.get(ep, float("nan"))
        return f"  {'nan' if v != v else f'{v:.4f}'}  "

    for ep in all_epochs:
        b = _fmt(baseline, ep)
        h = _fmt(hu, ep)
        t = f"  {'nan' if not has_ts else (_fmt(ts, ep).strip())}  " if has_ts else ""
        lines.append(f"  {ep:02d}  |{b}|{h}|{t}")

    lines.append(sep)

    best_b = _best(baseline)
    best_h = _best(hu)
    best_t = _best(ts) if has_ts else float("nan")

    def _fmt_best(v):
        return f"  {'nan' if v != v else f'{v:.4f}'}  "

    best_ts_col = f"|{_fmt_best(best_t)}" if has_ts else ""
    lines.append(f" Best |{_fmt_best(best_b)}|{_fmt_best(best_h)}{best_ts_col}")
    lines.append("")

    def _diff(a, b):
        if a != a or b != b:
            return "nan"
        return f"{a - b:+.4f}"

    lines.append(f"HU improvement over baseline:  {_diff(best_h, best_b)} AUC")
    if has_ts:
        lines.append(f"TS improvement over baseline:  {_diff(best_t, best_b)} AUC")
        lines.append(f"TS improvement over HU:        {_diff(best_t, best_h)} AUC")

    output = "\n".join(lines)
    print(output)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(output + "\n")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
