import os
import re
import argparse
import numpy as np

from config import RESULTS_DIR

_EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s*\|.*?Val AUC\s+([\d.nan]+)", re.IGNORECASE)
_ALPHA_BEST_RE = re.compile(r"BEST:\s+alpha=([\d.]+)\s*->\s*AUC\s+([\d.]+)", re.IGNORECASE)
_ALPHA_ROW_RE = re.compile(r"^\s*([\d.]+)\s*\|\s*([\d.nan]+)\s*\|", re.IGNORECASE)
_STATS_MEAN_RE = re.compile(r"Mean % activation inside mask:\s*([\d.nan]+)%", re.IGNORECASE)
_STATS_STD_RE = re.compile(r"Std:\s*([\d.nan]+)%", re.IGNORECASE)
_STATS_N_RE = re.compile(r"Val samples evaluated:\s*(\d+)", re.IGNORECASE)
_TEST_AUC_RE = re.compile(r"Test AUC:\s+([\d.nan]+)", re.IGNORECASE)
_TEST_ACC_RE = re.compile(r"Test Accuracy:\s+([\d.nan]+)", re.IGNORECASE)
_TEST_F1_RE = re.compile(r"Test F1:\s+([\d.nan]+)", re.IGNORECASE)
_TEST_SENS_RE = re.compile(r"Test Sensitivity:\s+([\d.nan]+)", re.IGNORECASE)
_TEST_SPEC_RE = re.compile(r"Test Specificity:\s+([\d.nan]+)", re.IGNORECASE)
_TEST_GCAM_RE = re.compile(r"Mean % inside mask:\s*([\d.nan]+)%", re.IGNORECASE)


def _parse_test_results(path):
    if not os.path.exists(path):
        return {}
    text = open(path).read()
    def _get(regex):
        m = regex.search(text)
        if not m:
            return float("nan")
        v = m.group(1)
        return float("nan") if v.lower() == "nan" else float(v)
    return {
        "auc":  _get(_TEST_AUC_RE),
        "acc":  _get(_TEST_ACC_RE),
        "f1":   _get(_TEST_F1_RE),
        "sens": _get(_TEST_SENS_RE),
        "spec": _get(_TEST_SPEC_RE),
        "gcam": _get(_TEST_GCAM_RE),
    }


def _parse_epoch_log(path):
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


def _parse_gradcam_stats(path):
    """Return (mean_pct, std_pct, n) from a gradcam_stats_*.txt file."""
    if not os.path.exists(path):
        return float("nan"), float("nan"), 0
    text = open(path).read()
    m_mean = _STATS_MEAN_RE.search(text)
    m_std = _STATS_STD_RE.search(text)
    m_n = _STATS_N_RE.search(text)
    mean_pct = float(m_mean.group(1)) if m_mean else float("nan")
    std_pct = float(m_std.group(1)) if m_std else float("nan")
    n = int(m_n.group(1)) if m_n else 0
    return mean_pct, std_pct, n


def _parse_alpha_log(path):
    if not os.path.exists(path):
        return None, {}
    best_alpha = None
    alpha_aucs = {}
    with open(path) as f:
        for line in f:
            m_best = _ALPHA_BEST_RE.search(line)
            if m_best:
                best_alpha = float(m_best.group(1))
            m_row = _ALPHA_ROW_RE.match(line)
            if m_row:
                a = float(m_row.group(1))
                auc_str = m_row.group(2)
                auc = float("nan") if auc_str.lower() == "nan" else float(auc_str)
                alpha_aucs[a] = auc
    return best_alpha, alpha_aucs


def _best(epoch_dict):
    vals = [v for v in epoch_dict.values() if v == v]
    return max(vals) if vals else float("nan")


def _fmt(d, ep):
    v = d.get(ep, float("nan"))
    return "  nan   " if v != v else f" {v:.4f} "


def _fmt_best(v):
    return "   nan  " if v != v else f" {v:.4f} "


def _diff(a, b):
    if a != a or b != b:
        return "  nan  "
    return f"{a - b:+.4f}"


def _fmt_pct(v):
    return "  nan  " if v != v else f"{v:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID to compare. Reads results/latest_run.txt if omitted.")
    args = parser.parse_args()

    # Resolve run directory
    if args.run_id:
        run_id = args.run_id
    else:
        latest_file = os.path.join(RESULTS_DIR, "latest_run.txt")
        if os.path.exists(latest_file):
            run_id = open(latest_file).read().strip()
        else:
            run_id = None

    if run_id:
        run_dir = os.path.join(RESULTS_DIR, run_id)
        print(f"[COMPARE] Run: {run_id}  ({run_dir})")
    else:
        run_dir = RESULTS_DIR
        print(f"[COMPARE] No run ID found — reading from {RESULTS_DIR} directly")

    baseline_log = os.path.join(run_dir, "baseline_log.txt")
    train_log = os.path.join(run_dir, "train_log.txt")
    ts_log = os.path.join(run_dir, "train_ts_log.txt")
    alpha_log = os.path.join(run_dir, "alpha_search_results.txt")
    alpha_bayes_log = os.path.join(run_dir, "alpha_bayes_results.txt")
    out_path = os.path.join(run_dir, "full_comparison.txt")

    stats_baseline = os.path.join(run_dir, "gradcam_stats_baseline.txt")
    stats_train = os.path.join(run_dir, "gradcam_stats_train.txt")
    stats_ts = os.path.join(run_dir, "gradcam_stats_ts.txt")

    baseline = _parse_epoch_log(baseline_log)
    hu = _parse_epoch_log(train_log)
    ts = _parse_epoch_log(ts_log)
    best_alpha, alpha_aucs = _parse_alpha_log(alpha_log)
    best_alpha_bayes, _ = _parse_alpha_log(alpha_bayes_log)

    test_baseline = _parse_test_results(os.path.join(run_dir, "test_results_baseline.txt"))
    test_hu = _parse_test_results(os.path.join(run_dir, "test_results_hu.txt"))
    test_ts = _parse_test_results(os.path.join(run_dir, "test_results_ts.txt"))

    has_ts = bool(ts)

    all_epochs = sorted(set(list(baseline.keys()) + list(hu.keys()) + list(ts.keys())))

    lines = []

    if not all_epochs:
        msg = (f"No training log files found in {run_dir}.\n"
               f"Run baseline.py, train.py (and optionally train_ts.py) with --run-id {run_id or '<id>'} first.")
        print(msg)
        return

    # ── Per-epoch AUC table ───────────────────────────────────────────────────
    ts_hdr = " TS Mask  " if has_ts else ""
    ts_sep = "----------" if has_ts else ""
    header = f"Epoch | Baseline | HU Mask  |{ts_hdr}"
    sep = f"------|----------|----------|{ts_sep}"
    lines += [f"=== AUC BY EPOCH  (run: {run_id or 'direct'}) ===", header, sep]

    for ep in all_epochs:
        b = _fmt(baseline, ep)
        h = _fmt(hu, ep)
        ts_col = _fmt(ts, ep) if has_ts else ""
        lines.append(f"  {ep:02d}  |{b}|{h}|{ts_col}")

    lines.append(sep)

    best_b = _best(baseline)
    best_h = _best(hu)
    best_t = _best(ts) if has_ts else float("nan")

    ts_best_col = _fmt_best(best_t) if has_ts else ""
    lines.append(f"  Best  |{_fmt_best(best_b)}|{_fmt_best(best_h)}|{ts_best_col}")
    lines.append("")

    # ── AUC improvement ───────────────────────────────────────────────────────
    lines.append("=== AUC IMPROVEMENT OVER BASELINE ===")
    lines.append(f"HU Mask:       {_diff(best_h, best_b)} AUC")
    if has_ts:
        lines.append(f"TS Mask:       {_diff(best_t, best_b)} AUC")
        lines.append(f"TS vs HU:      {_diff(best_t, best_h)} AUC")
    lines.append("")

    # ── Grad-CAM alignment stats ──────────────────────────────────────────────
    b_mean, b_std, b_n = _parse_gradcam_stats(stats_baseline)
    h_mean, h_std, h_n = _parse_gradcam_stats(stats_train)
    t_mean, t_std, t_n = _parse_gradcam_stats(stats_ts)

    any_stats = b_n > 0 or h_n > 0 or t_n > 0

    lines.append("=== GRAD-CAM MASK ALIGNMENT (% activation inside mask) ===")
    if any_stats:
        lines.append(f"{'Model':<12} {'Mean %':>8} {'Std':>7} {'N':>5}")
        lines.append(f"{'-'*12} {'-'*8} {'-'*7} {'-'*5}")
        if b_n > 0:
            lines.append(f"{'Baseline':<12} {_fmt_pct(b_mean):>8} {_fmt_pct(b_std):>7} {b_n:>5}")
        if h_n > 0:
            lines.append(f"{'HU Mask':<12} {_fmt_pct(h_mean):>8} {_fmt_pct(h_std):>7} {h_n:>5}")
        if t_n > 0:
            lines.append(f"{'TS Mask':<12} {_fmt_pct(t_mean):>8} {_fmt_pct(t_std):>7} {t_n:>5}")
        lines.append("")
        if h_n > 0 and b_n > 0 and h_mean == h_mean and b_mean == b_mean:
            lines.append(f"HU attention gain over baseline: {h_mean - b_mean:+.1f}pp")
        if t_n > 0 and b_n > 0 and t_mean == t_mean and b_mean == b_mean:
            lines.append(f"TS attention gain over baseline: {t_mean - b_mean:+.1f}pp")
    else:
        lines.append("No gradcam_stats_*.txt files found in run directory.")
        lines.append("These are generated automatically at end of each training script.")
    lines.append("")

    # ── Test set results ──────────────────────────────────────────────────────
    any_test = any([test_baseline, test_hu, test_ts])
    lines.append("=== TEST SET RESULTS (Held-out, Best Checkpoint) ===")
    if any_test:
        def _tr(v):
            return "  nan  " if np.isnan(v) else f" {v:.4f}"
        def _tp(v):
            return "  nan " if np.isnan(v) else f"{v:.1f}%"
        hdr_ts = "  TS Mask " if has_ts else ""
        sep_ts = "----------" if has_ts else ""
        lines.append(f"{'Metric':<14} | {'Baseline':>9} | {'HU Mask':>9} |{hdr_ts}")
        lines.append(f"{'-'*14}-+-{'-'*9}-+-{'-'*9}-+{sep_ts}")
        for metric, key in [("AUC", "auc"), ("Accuracy", "acc"), ("F1", "f1"),
                             ("Sensitivity", "sens"), ("Specificity", "spec")]:
            b = _tr(test_baseline.get(key, float("nan")))
            h = _tr(test_hu.get(key, float("nan")))
            t = _tr(test_ts.get(key, float("nan"))) if has_ts else ""
            lines.append(f"{metric:<14} |{b}|{h}|{t}")
        lines.append("")
        lines.append("Test Grad-CAM alignment (% inside mask):")
        for label, d in [("Baseline", test_baseline), ("HU Mask", test_hu), ("TS Mask", test_ts)]:
            if d:
                lines.append(f"  {label:<12} {_tp(d.get('gcam', float('nan')))}")
    else:
        lines.append("No test_results_*.txt files found — run training scripts first.")
    lines.append("")

    # ── Alpha sensitivity ─────────────────────────────────────────────────────
    lines.append("=== ALPHA SENSITIVITY ===")
    if alpha_aucs:
        valid_aucs = [v for v in alpha_aucs.values() if v == v]
        auc_min = min(valid_aucs) if valid_aucs else float("nan")
        auc_max = max(valid_aucs) if valid_aucs else float("nan")
        auc_std = float(np.std(valid_aucs)) if len(valid_aucs) > 1 else float("nan")
        ba_str = f"{best_alpha:.2f}" if best_alpha is not None else "nan"
        lines.append(f"Grid search  — best alpha: {ba_str}  |  AUC range: {auc_min:.4f}–{auc_max:.4f}  |  std: {auc_std:.4f}")
    else:
        lines.append("Grid:  alpha_search_results.txt not found — run alpha_search.py first.")

    if best_alpha_bayes is not None:
        lines.append(f"Bayes search — best alpha: {best_alpha_bayes:.4f}")
    else:
        lines.append("Bayes: alpha_bayes_results.txt not found — run alpha_search.py --mode bayes first.")

    output = "\n".join(lines)
    print("\n" + output)

    os.makedirs(run_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(output + "\n")
    print(f"\nSaved → {out_path}")

    # ── alignment_delta.txt ───────────────────────────────────────────────────
    def _pp(a, b):
        return f"{a - b:+.1f}pp" if not (np.isnan(a) or np.isnan(b)) else "  nan"

    delta_lines = [
        f"=== GRAD-CAM ALIGNMENT DELTA  (run: {run_id or 'direct'}) ===",
        "(Val set, best checkpoint — isolates attention penalty beyond classification alone)",
        "",
        f"{'Model':<12} {'Mean %':>8} {'Std':>7} {'N':>5}",
        f"{'-'*12} {'-'*8} {'-'*7} {'-'*5}",
    ]
    gcam_vals = {}
    for label, fname in [("Baseline", "gradcam_stats_baseline.txt"),
                          ("HU Mask",  "gradcam_stats_train.txt"),
                          ("TS Mask",  "gradcam_stats_ts.txt")]:
        m, s, n = _parse_gradcam_stats(os.path.join(run_dir, fname))
        gcam_vals[label] = m
        if n > 0:
            delta_lines.append(f"{label:<12} {_fmt_pct(m):>8} {_fmt_pct(s):>7} {n:>5}")

    delta_lines.append("")
    b_gcam = gcam_vals.get("Baseline", float("nan"))
    h_gcam = gcam_vals.get("HU Mask",  float("nan"))
    t_gcam = gcam_vals.get("TS Mask",  float("nan"))
    delta_lines.append(f"HU delta over baseline:  {_pp(h_gcam, b_gcam)}")
    if not np.isnan(t_gcam):
        delta_lines.append(f"TS delta over baseline:  {_pp(t_gcam, b_gcam)}")
        delta_lines.append(f"TS delta over HU:        {_pp(t_gcam, h_gcam)}")
    delta_lines.append("")
    delta_lines.append("(pp = percentage points; positive = more attention inside mask than baseline)")

    delta_path = os.path.join(run_dir, "alignment_delta.txt")
    with open(delta_path, "w") as f:
        f.write("\n".join(delta_lines) + "\n")
    print(f"Saved → {delta_path}")


if __name__ == "__main__":
    main()
