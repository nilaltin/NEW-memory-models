import os
import pandas as pd

from model_pavlik import fit_pavlik_model, replay_pavlik_actr


def main():
    # scripts/ -> slim stampen/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    slim_dir = os.path.abspath(os.path.join(script_dir, ".."))
    csv_path = os.path.join(slim_dir, "data", "slimstampen_all_trials.csv")

    out_dir = os.path.join(slim_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(csv_path)

    # Ask for subject
    default_subj = int(df["subj"].iloc[0])
    s_in = input(f"Enter subject ID (e.g., {default_subj}): ").strip()
    subject_id = int(s_in) if s_in else default_subj

    df_one = df[df["subj"] == subject_id].copy()

    print("\n=== PAVLIK ONE-SUBJECT FIT (c = 0.5 fixed) ===")
    print("Subject:", subject_id)
    print("Trials:", len(df_one))

    # Fit (params are [phi, tau, s, T0, F])
    res = fit_pavlik_model(df_one)

    print("\n=== FIT RESULT ===")
    print("Converged:", res.success)
    print("NLL:", res.fun)
    print("Params [phi, tau, s, T0, F]:")
    print(res.x)

    # Save fit summary
    fit_out_path = os.path.join(out_dir, f"pavlik_fit_subject_{subject_id}.csv")
    pd.DataFrame([{
        "subject": int(subject_id),
        "trials": int(len(df_one)),
        "converged": bool(res.success),
        "nll": float(res.fun),
        "phi": float(res.x[0]),
        "tau": float(res.x[1]),
        "s": float(res.x[2]),
        "T0": float(res.x[3]),
        "F": float(res.x[4]),
    }]).to_csv(fit_out_path, index=False)
    print("\nSaved fit summary to:")
    print(fit_out_path)

    # Replay predictions
    phi, tau, s, T0, F = res.x
    pred = replay_pavlik_actr(df_one, phi, tau, s, T0, F)

    print("\nPredictions head:")
    print(pred.head())

    # Save predictions
    pred_out_path = os.path.join(out_dir, f"pavlik_predictions_subject_{subject_id}.csv")
    pred.to_csv(pred_out_path, index=False)
    print("\nSaved predictions to:")
    print(pred_out_path)


if __name__ == "__main__":
    main()
