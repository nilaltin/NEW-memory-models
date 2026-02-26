import os
import numpy as np
import pandas as pd


def main():
    # scripts/ -> slim stampen/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    slim_dir = os.path.abspath(os.path.join(script_dir, ".."))

    data_csv = os.path.join(slim_dir, "data", "slimstampen_all_trials.csv")
    out_dir = os.path.join(slim_dir, "outputs")

    pav_csv = os.path.join(out_dir, "pavlik_per_subject_fits.csv")
    wgt_csv = os.path.join(out_dir, "weighted_per_subject_fits.csv")

    df = pd.read_csv(data_csv)
    pav = pd.read_csv(pav_csv)
    wgt = pd.read_csv(wgt_csv)

    #columns based on CSV  
    req_pav = {"subj", "converged", "NLL", "phi", "T0", "F"}
    req_wgt = {"subj", "converged", "NLL", "d", "T0", "F"}
    if not req_pav.issubset(pav.columns):
        raise ValueError(f"Pavlik fits missing: {sorted(req_pav - set(pav.columns))}")
    if not req_wgt.issubset(wgt.columns):
        raise ValueError(f"Weighted fits missing: {sorted(req_wgt - set(wgt.columns))}")

    # n_test per subject
    n_test_map = df[df["type"] == "test"].groupby("subj").size().to_dict()

    # Merge per subject
    m = pav.merge(wgt, on="subj", suffixes=("_actr", "_fear"), how="inner")

    # Model parameter counts (models)
    k_actr = 5  # Pavlik: phi, tau, s, T0, F
    k_fear = 6  # Weighted: d, w1, tau, s, T0, F

    #output columns
    out = pd.DataFrame()
    out["subj"] = m["subj"].astype(int)
    out["n_test"] = out["subj"].map(n_test_map).astype(int)

    out["nll_actr"] = m["NLL_actr"].astype(float)   # Pavlik
    out["nll_fear"] = m["NLL_fear"].astype(float)   # Weighted

    out["n_params_actr"] = k_actr
    out["n_params_fear"] = k_fear

    out["aic_actr"] = 2 * k_actr + 2 * out["nll_actr"]
    out["aic_fear"] = 2 * k_fear + 2 * out["nll_fear"]

    out["bic_actr"] = k_actr * np.log(out["n_test"]) + 2 * out["nll_actr"]
    out["bic_fear"] = k_fear * np.log(out["n_test"]) + 2 * out["nll_fear"]

    out["delta_aic"] = out["aic_fear"] - out["aic_actr"]
    out["delta_bic"] = out["bic_fear"] - out["bic_actr"]

    # Extra columns:
    # "c" in first table -> use your Pavlik phi
    out["c"] = m["phi"].astype(float)
    # "d" in first table -> use your Weighted d
    out["d"] = m["d"].astype(float)

    out["t0_actr"] = m["T0_actr"].astype(float)
    out["F_actr"] = m["F_actr"].astype(float)
    out["t0_fear"] = m["T0_fear"].astype(float)
    out["F_fear"] = m["F_fear"].astype(float)

    out["converged_actr"] = m["converged_actr"].astype(bool)
    out["converged_fear"] = m["converged_fear"].astype(bool)

    # Ensure exact order
    out = out[
        [
            "subj", "n_test",
            "nll_actr", "nll_fear",
            "n_params_actr", "n_params_fear",
            "aic_actr", "aic_fear",
            "bic_actr", "bic_fear",
            "delta_aic", "delta_bic",
            "c", "d",
            "t0_actr", "F_actr",
            "t0_fear", "F_fear",
            "converged_actr", "converged_fear",
        ]
    ].sort_values("subj")

    out_csv = os.path.join(out_dir, "model_comparison_summary.csv")
    out.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(out.head())


if __name__ == "__main__":
    main()
