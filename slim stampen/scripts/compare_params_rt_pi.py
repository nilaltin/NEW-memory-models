import os
import json
import numpy as np
import pandas as pd

# Import activation functions from the parallel script (same equations)
from fit_model_parallel import (
    compute_activation_actr,
    compute_activation_fear,
)

# -----------------------
# USER CONFIG
# -----------------------
DATA_PATH = "../data/newdata.csv"
RESULTS_DIR = "../outputs/pi_filtered"   # change to ../outputs/pi_full for full run output
OUT_CSV = os.path.join(RESULTS_DIR, "params_rt_summary.csv")


def pearsonr_safe(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = int(len(x))
    if n < 3:
        return np.nan, n
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan, n
    return float(np.corrcoef(x, y)[0, 1]), n


def load_results(results_dir):
    actr_path = os.path.join(results_dir, "actr_results.json")
    fear_path = os.path.join(results_dir, "fear_results.json")

    with open(actr_path, "r") as f:
        actr = json.load(f)
    with open(fear_path, "r") as f:
        fear = json.load(f)

    # JSON keys may be strings depending on writer; normalize to int for safe lookup
    actr = {int(k): v for k, v in actr.items()}
    fear = {int(k): v for k, v in fear.items()}
    return actr, fear


def load_and_filter_data(csv_path):
    df = pd.read_csv(csv_path)

    # Match newdata filtering logic
    df["RT_s"] = df["RT"] / 1000.0
    test_mask = df["type"] == "test"
    rt_mask = (df["RT_s"] >= 0.2) & (df["RT_s"] <= 10.0)
    df = df[~test_mask | rt_mask].copy()

    return df


def rt_corr_subject_actr(df_subj, actr_params):
    c = float(actr_params["c"])
    t0 = float(actr_params["t0"])
    F = float(actr_params["F"])
    phi_map = {int(k): float(v) for k, v in actr_params["phi"].items()}

    rt_obs = []
    rt_pred = []

    # Correct-only test trials
    df_test = df_subj[(df_subj["type"] == "test") & (df_subj["isCorrect"] == True)].copy()

    # We need, for each test trial, its encoding times up to that moment for that fact
    # Use "time" and "trial" exactly like the fit script.
    for item, df_item in df_subj.groupby("item"):
        item = int(item)
        phi = float(phi_map.get(item, np.nan))
        if not np.isfinite(phi):
            continue

        df_item = df_item.sort_values("trial").reset_index(drop=True)
        enc_times = []

        for _, row in df_item.iterrows():
            t = float(row["time"])
            if row["type"] == "study":
                enc_times.append(t)
            else:
                # test
                if bool(row["isCorrect"]) is True:
                    A = compute_activation_actr(np.array(enc_times, float), t, c, phi)
                    if np.isfinite(A):
                        pred = t0 + F * np.exp(-A)  # predicted RT (scale) + nondecision
                        rt_obs.append(float(row["RT_s"]))
                        rt_pred.append(float(pred))
                enc_times.append(t)

    return pearsonr_safe(rt_obs, rt_pred)


def rt_corr_subject_fear(df_subj, fear_params):
    d = float(fear_params["d"])
    t0 = float(fear_params["t0"])
    F = float(fear_params["F"])
    w1_map = {int(k): float(v) for k, v in fear_params["w1"].items()}

    rt_obs = []
    rt_pred = []

    for item, df_item in df_subj.groupby("item"):
        item = int(item)
        w1 = float(w1_map.get(item, np.nan))
        if not np.isfinite(w1):
            continue

        df_item = df_item.sort_values("trial").reset_index(drop=True)
        enc_times = []

        for _, row in df_item.iterrows():
            t = float(row["time"])
            if row["type"] == "study":
                enc_times.append(t)
            else:
                if bool(row["isCorrect"]) is True:
                    A = compute_activation_fear(np.array(enc_times, float), t, d, w1)
                    if np.isfinite(A):
                        pred = t0 + F * np.exp(-A)
                        rt_obs.append(float(row["RT_s"]))
                        rt_pred.append(float(pred))
                enc_times.append(t)

    return pearsonr_safe(rt_obs, rt_pred)


def main():
    df = load_and_filter_data(DATA_PATH)
    actr, fear = load_results(RESULTS_DIR)

    rows = []
    for subj in sorted(df["subj"].unique()):
        subj = int(subj)
        df_subj = df[df["subj"] == subj].copy()
        n_test = int((df_subj["type"] == "test").sum())

        if subj not in actr or subj not in fear:
            continue

        a = actr[subj]
        f = fear[subj]

        # Shared parameters (same role in both models)
        t0_actr = float(a["t0"])
        t0_fear = float(f["t0"])
        F_actr = float(a["F"])
        F_fear = float(f["F"])

        # RT correlations (correct-only)
        r_a, n_a = rt_corr_subject_actr(df_subj, a)
        r_f, n_f = rt_corr_subject_fear(df_subj, f)

        # Helpful model-specific summaries
        c = float(a["c"])
        d = float(f["d"])
        phi_vals = np.array([float(v) for v in a["phi"].values()], float)
        w1_vals = np.array([float(v) for v in f["w1"].values()], float)

        rows.append({
            "subj": subj,
            "n_test": n_test,

            # shared params comparison
            "t0_actr": t0_actr,
            "t0_fear": t0_fear,
            "diff_t0_fear_minus_actr": t0_fear - t0_actr,

            "F_actr": F_actr,
            "F_fear": F_fear,
            "diff_F_fear_minus_actr": F_fear - F_actr,

            # RT fit
            "rt_r_actr_correct_only": r_a,
            "rt_n_actr_correct_only": n_a,
            "rt_r_fear_correct_only": r_f,
            "rt_n_fear_correct_only": n_f,

            # model-specific (still useful)
            "c_actr": c,
            "d_fear": d,
            "phi_mean_actr": float(np.nanmean(phi_vals)) if len(phi_vals) else np.nan,
            "w1_mean_fear": float(np.nanmean(w1_vals)) if len(w1_vals) else np.nan,
            "n_facts": int(a.get("n_params", 0)) - 3 if "n_params" in a else len(phi_vals),
        })

    out = pd.DataFrame(rows).sort_values("subj")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print("Saved:", OUT_CSV)
    print(out.head())

    # quick console summary
    print("\nRT correlation (correct-only):")
    print("  mean r ACT-R:", np.nanmean(out["rt_r_actr_correct_only"]))
    print("  mean r FE   :", np.nanmean(out["rt_r_fear_correct_only"]))


if __name__ == "__main__":
    main()
