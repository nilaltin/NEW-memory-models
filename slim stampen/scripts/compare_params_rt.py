import os
import numpy as np
import pandas as pd

from model_pavlik import replay_pavlik_actr
from modelmemorypractice1 import replay_weighted_actr


def _pearsonr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n = int(len(x))
    if n < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return (np.nan, n)
    return (float(np.corrcoef(x, y)[0, 1]), n)


def _rt_corr_pavlik(df_s, phi, tau, s, T0, F):
    pred = replay_pavlik_actr(
        df_s,
        phi=phi, tau=tau, s=s, T0=T0, F=F,
        learn_on=("study", "test_correct"),
        rt_in_ms=True
    )
    rt_obs = pred["rt_obs"].to_numpy(float)
    rt_pred = pred["rt_pred"].to_numpy(float)
    y = pred["isCorrect"].to_numpy(int)

    # correct-only + rt_obs > T0, consistent with your fitting
    valid = np.isfinite(rt_obs) & np.isfinite(rt_pred) & (rt_obs > T0) & (y == 1)
    return _pearsonr(rt_obs[valid], rt_pred[valid])


def _rt_corr_weighted(df_s, d, w1, tau, s, T0, F):
    pred = replay_weighted_actr(
        df_s,
        d=d, w1=w1, tau=tau, s=s, T0=T0, F=F,
        learn_on=("study", "test_correct"),
        rt_in_ms=True
    )
    rt_obs = pred["rt_obs"].to_numpy(float)
    rt_pred = pred["rt_pred"].to_numpy(float)
    y = pred["isCorrect"].to_numpy(int)

    valid = np.isfinite(rt_obs) & np.isfinite(rt_pred) & (rt_obs > T0) & (y == 1)
    return _pearsonr(rt_obs[valid], rt_pred[valid])


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    slim_dir = os.path.abspath(os.path.join(script_dir, ".."))

    data_csv = os.path.join(slim_dir, "data", "slimstampen_all_trials.csv")
    out_dir = os.path.join(slim_dir, "outputs")

    pav_csv = os.path.join(out_dir, "pavlik_per_subject_fits.csv")
    wgt_csv = os.path.join(out_dir, "weighted_per_subject_fits.csv")

    df = pd.read_csv(data_csv)
    pav = pd.read_csv(pav_csv)
    wgt = pd.read_csv(wgt_csv)

    # Minimal columns we need
    req_pav = {"subj", "phi", "tau", "s", "T0", "F"}
    req_wgt = {"subj", "d", "w1", "tau", "s", "T0", "F"}
    if not req_pav.issubset(pav.columns):
        raise ValueError(f"Pavlik fits missing: {sorted(req_pav - set(pav.columns))}")
    if not req_wgt.issubset(wgt.columns):
        raise ValueError(f"Weighted fits missing: {sorted(req_wgt - set(wgt.columns))}")

    m = pav.merge(wgt, on="subj", suffixes=("_pavlik", "_weighted"), how="inner").sort_values("subj")

    rows = []
    for _, row in m.iterrows():
        subj = int(row["subj"])
        df_s = df[df["subj"] == subj].copy()

        r_p, n_p = _rt_corr_pavlik(
            df_s,
            phi=float(row["phi"]),
            tau=float(row["tau_pavlik"]),
            s=float(row["s_pavlik"]),
            T0=float(row["T0_pavlik"]),
            F=float(row["F_pavlik"]),
        )
        r_w, n_w = _rt_corr_weighted(
            df_s,
            d=float(row["d"]),
            w1=float(row["w1"]),
            tau=float(row["tau_weighted"]),
            s=float(row["s_weighted"]),
            T0=float(row["T0_weighted"]),
            F=float(row["F_weighted"]),
        )

        rows.append(
            {
                "subj": subj,
                "tau_pavlik": float(row["tau_pavlik"]),
                "tau_weighted": float(row["tau_weighted"]),
                "s_pavlik": float(row["s_pavlik"]),
                "s_weighted": float(row["s_weighted"]),
                "rt_r_pavlik": r_p,
                "rt_n_pavlik": n_p,
                "rt_r_weighted": r_w,
                "rt_n_weighted": n_w,
                # still useful model-specific params to look at
                "phi": float(row["phi"]),
                "d": float(row["d"]),
                "w1": float(row["w1"]),
            }
        )

    out = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "params_rt_summary.csv")
    out.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(out.head())


if __name__ == "__main__":
    main()
