# Fit Pavlik ACT-R model per subject (c = 0.5 fixed)

import os
import pandas as pd

from model_pavlik import fit_pavlik_model


def main():
    # scripts/ -> slim stampen/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    slim_dir = os.path.abspath(os.path.join(script_dir, ".."))
    csv_path = os.path.join(slim_dir, "data", "slimstampen_all_trials.csv")

    out_dir = os.path.join(slim_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # --- load data ---
    df = pd.read_csv(csv_path)
    subjects = sorted(df["subj"].unique())

    print("CSV:", csv_path)
    print("N subjects:", len(subjects))
    print("Subjects (first 10):", subjects[:10], "...")

    results = []

    for i, subj in enumerate(subjects, start=1):
        df_s = df[df["subj"] == subj].copy()

        # Fit (params are [phi, tau, s, T0, F])
        res = fit_pavlik_model(df_s, rt_only_correct=True, rt_in_ms=True)
        phi, tau, s, T0, F = res.x

        row = {
            "subj": int(subj),
            "n_trials": int(len(df_s)),
            "converged": bool(res.success),
            "NLL": float(res.fun),
            "phi": float(phi),
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

        print(
            f"[{i}/{len(subjects)}] subj={subj} "
            f"conv={row['converged']} NLL={row['NLL']:.3f} "
            f"phi={row['phi']:.4f} s={row['s']:.3f}"
        )

    out_csv = os.path.join(out_dir, "pavlik_per_subject_fits.csv")
    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)
    print(out_df.head())

    print("\nParameter ranges:")
    for col in ["phi", "tau", "s", "T0", "F"]:
        print(f"{col}: min={out_df[col].min():.4f}, max={out_df[col].max():.4f}")


if __name__ == "__main__":
    main()
