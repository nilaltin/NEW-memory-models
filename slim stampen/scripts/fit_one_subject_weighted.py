# Fit weighted-trace model to ONE subject

import os
import pandas as pd

from modelmemorypractice1 import fit_weighted_model


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    slim_dir = os.path.abspath(os.path.join(script_dir, ".."))

    csv_path = os.path.join(slim_dir, "data", "slimstampen_all_trials.csv")
    out_dir = os.path.join(slim_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    default_subj = int(df["subj"].iloc[0])
    s_in = input(f"Enter subject ID (e.g., {default_subj}): ").strip()
    subject_id = int(s_in) if s_in else default_subj

    df_one = df[df["subj"] == subject_id].copy()

    print("\n=== WEIGHTED ONE-SUBJECT FIT ===")
    print("Subject:", subject_id)
    print("Trials:", len(df_one))

    res = fit_weighted_model(df_one, rt_only_correct=True, rt_in_ms=True)

    print("\n=== FIT RESULT ===")
    print("Converged:", res.success)
    print("Message:", res.message)
    print("NLL:", res.fun)
    print("Params [d, w1, tau, s, T0, F]:")
    print(res.x)

    fit_out_path = os.path.join(out_dir, f"weighted_fit_subject_{subject_id}.csv")
    d, w1, tau, s, T0, F = res.x

    pd.DataFrame([{
        "subject": int(subject_id),
        "trials": int(len(df_one)),
        "converged": bool(res.success),
        "nll": float(res.fun),
        "d": float(d),
        "w1": float(w1),
        "tau": float(tau),
        "s": float(s),
        "T0": float(T0),
        "F": float(F),
    }]).to_csv(fit_out_path, index=False)

    print("\nSaved:", fit_out_path)


if __name__ == "__main__":
    main()
