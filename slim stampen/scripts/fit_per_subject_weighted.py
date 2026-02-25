# Fit weighted-trace ACT-R model per subject (individual d's)

import os
import pandas as pd

from modelmemorypractice1 import fit_weighted_model


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    slim_dir = os.path.abspath(os.path.join(script_dir, ".."))

    csv_path = os.path.join(slim_dir, "data", "slimstampen_all_trials.csv")
    out_dir = os.path.join(slim_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "weighted_per_subject_fits.csv")

    df = pd.read_csv(csv_path)
    subjects = sorted(df["subj"].unique())

    results = []

    for i, subj in enumerate(subjects, start=1):
        df_s = df[df["subj"] == subj].copy()

        res = fit_weighted_model(df_s, rt_only_correct=True, rt_in_ms=True)
        d, w1, tau, s, T0, F = res.x

        row = {
            "subj": int(subj),
            "n_trials": int(len(df_s)),
            "converged": bool(res.success),
            "NLL": float(res.fun),
            "d": float(d),
            "w1": float(w1),
            "tau": float(tau),
            "s": float(s),
            "T0": float(T0),
            "F": float(F),
            "status": int(res.status),
            "message": str(res.message),
            "nit": int(res.nit),
            "nfev": int(res.nfev),
            
        }
        results.append(row)

        print(f"[{i}/{len(subjects)}] subj={subj} conv={row['converged']} NLL={row['NLL']:.2f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)
    print(results_df.head())

    print("\nParameter ranges:")
    for col in ["d", "w1", "tau", "s", "T0", "F"]:
        print(f"{col}: min={results_df[col].min():.4f}, max={results_df[col].max():.4f}")

if __name__ == "__main__":
    main()          
