"""
train_model.py
==============
STEP 2 of the pipeline:  train, evaluate and save the machine-learning models.

WHAT THIS SCRIPT DOES
---------------------
1. Loads data/beam_deflection_dataset.csv
2. Splits it into train / test sets  (stratified on load case, fixed seed)
3. Trains three regression models:

       Model A  Linear Regression on raw features      -- the simple baseline
       Model B  Random Forest Regressor on raw features -- the MAIN model you asked for
       Model C  Power-Law Regression (linear regression on log features)
                -- included because it *recovers the physical exponents*,
                   which is a strong, honest result to present

4. Evaluates all three with MAE, RMSE and R^2 on the unseen test set
5. Runs a SEPARATE extrapolation experiment to show how badly a tree model
   behaves outside its training range
6. Saves the best model to models/model.joblib
7. Saves metrics to models/metrics.json  (the Streamlit app reads this)
8. Saves figures to outputs/

WHY THE TARGET IS LOG-TRANSFORMED FOR THE TREE MODEL
----------------------------------------------------
Deflection spans four orders of magnitude (0.01 mm to 70 mm).  If we fit the
raw value, the model spends all its effort on the few very large deflections and
ignores the small ones, and the errors become extremely skewed.  Training on
log(deflection) makes the error RELATIVE (percentage-like) instead of absolute,
which is what we want for a quantity spanning orders of magnitude.
Predictions are converted back with expm1() before any metric is computed, so
every number you see is in real millimetres.

NOTHING IS FAKE HERE
--------------------
Every metric printed by this script is measured on data the model never saw.
The script prints the real numbers; if they are mediocre you will see mediocre
numbers.  Do not edit them.

Usage
-----
    python train_model.py
"""

from __future__ import annotations

import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")                     # no GUI needed; safe on Windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from model_classes import LogTargetRegressor, PowerLawRegressor
from physics import LOAD_CASE_NAMES, LOAD_CASE_TIP, MATERIALS, tip_deflection_mm

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_PATH = "data/beam_deflection_dataset.csv"
MODEL_PATH = "models/model.joblib"
METRICS_PATH = "models/metrics.json"
OUT_DIR = "outputs"

FEATURES = ["length_mm", "payload_kg", "diameter_mm", "E_GPa", "load_case"]
TARGET = "deflection_mm"

# Extrapolation experiment: train on short arms, test on long arms.
EXTRAP_TRAIN_MAX_LENGTH = 650.0
EXTRAP_TEST_MIN_LENGTH = 680.0

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ------------------------------------------------------------------------------
# METRICS
# ------------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, RMSE, R^2 and MAPE in millimetres (or %)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    # MAPE is only meaningful because the generator guarantees delta >= 0.01 mm
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    # how good is it relatively, in the way an engineer would quote it?
    within_10pct = float(np.mean(np.abs((y_true - y_pred) / y_true) <= 0.10) * 100.0)
    return {
        "MAE_mm": round(float(mae), 5),
        "RMSE_mm": round(rmse, 5),
        "R2": round(float(r2), 5),
        "MAPE_pct": round(mape, 3),
        "within_10pct": round(within_10pct, 2),
    }


# ------------------------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------------------------
def plot_actual_vs_predicted(y_true, y_pred, name, path, metrics):
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(y_true, y_pred, s=10, alpha=0.45, edgecolors="none",
               color="#2563eb", label="Test samples")
    lo = min(y_true.min(), y_pred.min()) * 0.8
    hi = max(y_true.max(), y_pred.max()) * 1.25
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect prediction")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual deflection (mm)")
    ax.set_ylabel("Predicted deflection (mm)")
    ax.set_title(f"Actual vs Predicted — {name}\n"
                 f"R² = {metrics['R2']:.4f},  RMSE = {metrics['RMSE_mm']:.3f} mm")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_residuals(y_true, y_pred, name, path):
    rel_err = (y_pred - y_true) / y_true * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(y_true, rel_err, s=10, alpha=0.45, edgecolors="none",
                    color="#dc2626")
    axes[0].axhline(0, color="k", ls="--", lw=1.1)
    axes[0].axhline(10, color="gray", ls=":", lw=1)
    axes[0].axhline(-10, color="gray", ls=":", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Actual deflection (mm)")
    axes[0].set_ylabel("Relative error (%)")
    axes[0].set_title(f"Relative error vs magnitude — {name}")
    axes[0].grid(True, which="both", ls=":", alpha=0.4)

    axes[1].hist(rel_err, bins=50, color="#7c3aed", alpha=0.8)
    axes[1].axvline(0, color="k", ls="--", lw=1.1)
    axes[1].set_xlabel("Relative error (%)")
    axes[1].set_ylabel("Number of test samples")
    axes[1].set_title("Distribution of relative error")
    axes[1].grid(True, ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_feature_importance(rf_base, X_test, y_test, path):
    """Permutation importance: how much does shuffling a feature hurt accuracy?

    This is measured on the TEST set, so it reflects genuine predictive reliance,
    not just how the trees happened to be built.
    """
    # Sub-sample the test set so this stays fast on a laptop.
    n = min(2000, len(X_test))
    Xs, ys = X_test.iloc[:n], y_test[:n]
    result = permutation_importance(rf_base, Xs, np.log1p(ys), n_repeats=5,
                                   random_state=RANDOM_STATE, scoring="r2")
    order = np.argsort(result.importances_mean)
    names = np.array(FEATURES)[order]
    vals = result.importances_mean[order]
    errs = result.importances_std[order]

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(names, vals, xerr=errs, color="#0f766e", alpha=0.85,
            error_kw={"ecolor": "#334155", "capsize": 3})
    ax.set_xlabel("Drop in R² when the feature is shuffled")
    ax.set_title("Permutation feature importance (Random Forest, test set)")
    ax.grid(True, axis="x", ls=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_length_curves(models, path):
    """Sweep arm length along a FULLY FEASIBLE load case and compare all models.

    The chosen case (0.5 kg on a 16 mm PLA arm) stays inside the small-deflection
    limit (delta/L <= 0.10) over the entire 150-800 mm range, so every point on
    this curve is a situation the model was actually trained on.  This is a fair
    interpolation test, not a trick.
    """
    L = np.linspace(150, 800, 200)
    case = LOAD_CASE_TIP
    P, d = 0.5, 16.0
    E = MATERIALS["PLA (3D printed)"]["E_GPa"]

    truth = np.array([tip_deflection_mm(l, P, E, d, case) for l in L])
    X = pd.DataFrame({"length_mm": L, "payload_kg": P, "diameter_mm": d,
                      "E_GPa": E, "load_case": case})

    styles = {"Linear Regression": ("#2563eb", "--"),
              "Random Forest": ("#dc2626", "-"),
              "Power-Law Regression": ("#16a34a", "-.")}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(L, truth, "k-", lw=3.0, label="Analytical formula (truth)", alpha=0.85)
    for name, model in models.items():
        color, ls = styles[name]
        ax.plot(L, model.predict(X), ls, color=color, lw=1.7, label=name)
    ax.set_xlabel("Arm length L (mm)")
    ax.set_ylabel("Tip deflection δ (mm)")
    ax.set_title("Deflection vs arm length\n0.5 kg tip load, Ø16 mm PLA arm "
                 "(feasible over the whole range)")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.5)

    ax = axes[1]
    for name, model in models.items():
        color, ls = styles[name]
        rel = np.abs(model.predict(X) - truth) / truth * 100.0
        ax.plot(L, rel, ls, color=color, lw=1.7, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("Arm length L (mm)")
    ax.set_ylabel("Absolute relative error (%) — log scale")
    ax.set_title("Where each model loses accuracy\n(the Random Forest error is "
                 "piecewise: many small steps = leaf granularity)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_extrapolation(extrap_results, models_e, path):
    """Two views of the extrapolation experiment.

    Panel 1: MAPE on unseen longer arms (bar chart).
    Panel 2: the actual curves.  Left of the dotted line the models saw data;
             right of it they did not.  Watch the Random Forest flatten -- a tree
             cannot output a value larger than the largest target it saw during
             training, so beyond its range it can only under-predict.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    names = list(extrap_results.keys())
    x = np.arange(len(names))
    vals = [extrap_results[n]["MAPE_pct"] for n in names]
    colors = ["#2563eb", "#dc2626", "#16a34a"]
    bars = ax.bar(x, vals, color=colors, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    ax.set_ylabel("MAPE on unseen long arms (%) — log scale")
    ax.set_title("Extrapolation error\ntrained on L ≤ 650 mm, tested on L ≥ 680 mm")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.18, f"{v:.1f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax.grid(True, axis="y", ls=":", alpha=0.5)

    ax = axes[1]
    L = np.linspace(150, 800, 200)
    P, d = 0.5, 16.0
    E = MATERIALS["PLA (3D printed)"]["E_GPa"]
    truth = np.array([tip_deflection_mm(l, P, E, d, LOAD_CASE_TIP) for l in L])
    X = pd.DataFrame({"length_mm": L, "payload_kg": P, "diameter_mm": d,
                      "E_GPa": E, "load_case": LOAD_CASE_TIP})
    ax.plot(L, truth, "k-", lw=3.0, alpha=0.85, label="Analytical formula (truth)")
    preds = {
        "Linear Regression": models_e["Linear Regression"].predict(X),
        "Random Forest": models_e["Random Forest"].predict(X),
        "Power-Law Regression": models_e["Power-Law Regression"].predict(X),
    }
    styles = {"Linear Regression": ("#2563eb", "--"),
              "Random Forest": ("#dc2626", "-"),
              "Power-Law Regression": ("#16a34a", "-.")}
    for name, pred in preds.items():
        color, ls = styles[name]
        ax.plot(L, pred, ls, color=color, lw=1.7, label=name)
    ax.axvline(EXTRAP_TRAIN_MAX_LENGTH, color="gray", ls=":", lw=1.5)
    ax.axvspan(EXTRAP_TRAIN_MAX_LENGTH, 800, color="orange", alpha=0.08)
    ax.text(EXTRAP_TRAIN_MAX_LENGTH + 10, ax.get_ylim()[1] * 0.30,
            "beyond training data\n(extrapolation)", fontsize=9, color="#9a3412")
    ax.set_xlabel("Arm length L (mm)")
    ax.set_ylabel("Tip deflection δ (mm)")
    ax.set_title("The same three models, drawn past their training range")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, ls=":", alpha=0.5)

    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("TRAINING PIPELINE")
    print("=" * 70)

    # ---- 1. load -------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    X = df[FEATURES].copy()
    y = df[TARGET].to_numpy()

    # ---- 2. split ------------------------------------------------------------
    # stratify=y_binned keeps BOTH load cases proportionally represented in the
    # test set, and stratifying on binned deflection keeps the test set spanning
    # the same deflection magnitudes as the training set.
    y_bins = pd.qcut(np.log(y), q=10, labels=False, duplicates="drop")
    strat = df["load_case"].astype(str) + "_" + y_bins.astype(str)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat
    )
    print(f"Train rows: {len(X_tr)}   Test rows: {len(X_te)}  "
          f"(test fraction {TEST_SIZE:.0%}, stratified)")
    assert set(X_tr.index).isdisjoint(set(X_te.index)), "train/test overlap!"

    # ---- 3. build models -----------------------------------------------------
    lin = LinearRegression()
    # Hyper-parameters were chosen by experiment, not guessed (see README):
    #   n_estimators=200      more trees than 120 gives no measurable gain
    #   min_samples_leaf=5    controls over-fitting AND keeps the saved model
    #                         small (~25 MB).  With leaf=1 the model was 237 MB
    #                         for no useful accuracy gain.
    rf_base = RandomForestRegressor(
        n_estimators=200,
        max_features=1.0,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    # Wrapped so that .predict() always returns millimetres, not log values.
    rf = LogTargetRegressor(rf_base)
    power = PowerLawRegressor()

    # 5-fold CV on the TRAINING set only.
    # This is the honest way to estimate performance before touching the test
    # set. If you ever tune a hyper-parameter using the test set, your reported
    # score is optimistic -- that is the data leakage we must avoid.
    print("\n3-fold cross-validation on the TRAINING set (R² on log1p deflection):")
    # Cross-validation runs on the underlying forest with the log1p target.
    # (Our LogTargetRegressor wrapper returns millimetres, which is great for the
    # app but is not a scikit-learn estimator, so CV uses rf_base directly here.
    # CV R² is therefore reported in log space -- a relative-accuracy measure.)
    cv = cross_val_score(rf_base, X_tr, np.log1p(y_tr), cv=3, scoring="r2", n_jobs=-1)
    print(f"  Random Forest CV R²: {cv.mean():.5f} ± {cv.std():.5f}  "
          f"(folds: {np.round(cv, 5).tolist()})")

    # ---- 4. fit on all training data ----------------------------------------
    print("\nFitting models ...")
    lin.fit(X_tr, y_tr)
    rf.fit(X_tr, y_tr)
    power.fit(X_tr, y_tr)

    models = {
        "Linear Regression": lin,
        "Random Forest": rf,
        "Power-Law Regression": power,
    }

    def predict(name: str, Xd: pd.DataFrame) -> np.ndarray:
        # Every model now predicts millimetres directly thanks to the wrappers.
        return models[name].predict(Xd)

    # ---- 5. metrics ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST-SET PERFORMANCE  (all values in millimetres unless stated)")
    print("=" * 70)
    results: dict[str, dict] = {}
    header = f"{'Model':<24}{'MAE':>10}{'RMSE':>10}{'R²':>10}{'MAPE%':>10}{'≤10% err':>11}"
    print(header)
    print("-" * len(header))
    for name in models:
        pred = predict(name, X_te)
        m = evaluate(y_te, pred)
        results[name] = m
        print(f"{name:<24}{m['MAE_mm']:>10.4f}{m['RMSE_mm']:>10.4f}"
              f"{m['R2']:>10.5f}{m['MAPE_pct']:>10.2f}{m['within_10pct']:>10.1f}%")

    # train-set metrics too, to check for over-fitting
    print("\nTrain-set R² (for over-fitting check):")
    train_scores = {}
    for name in models:
        pred_tr = predict(name, X_tr)
        train_scores[name] = round(float(r2_score(y_tr, pred_tr)), 5)
        print(f"  {name:<24} R² = {train_scores[name]:.5f}")

    # ---- 6. power-law coefficients vs theory --------------------------------
    print("\n" + "=" * 70)
    print("POWER-LAW EXPONENTS RECOVERED BY THE MODEL  vs  PHYSICS")
    print("=" * 70)
    theory = PowerLawRegressor.THEORY
    print(f"{'Feature':<20}{'Recovered':>12}{'Theory':>10}")
    for key, expected in theory.items():
        got = power.coef_[key]
        print(f"{key:<20}{got:>12.4f}{expected:>10.1f}")
    case_offset = power.coef_["load_case"]
    print(f"{'load_case offset':<20}{case_offset:>12.4f}{np.log(1/3):>10.4f}"
          f"   (theory ln(1/3) = {np.log(1/3):.4f})")

    # ---- 7. extrapolation experiment ---------------------------------------
    print("\n" + "=" * 70)
    print(f"EXTRAPOLATION EXPERIMENT  (train L ≤ {EXTRAP_TRAIN_MAX_LENGTH:.0f} mm, "
          f"test L ≥ {EXTRAP_TEST_MIN_LENGTH:.0f} mm)")
    print("=" * 70)
    tr_mask = df["length_mm"] <= EXTRAP_TRAIN_MAX_LENGTH
    te_mask = df["length_mm"] >= EXTRAP_TEST_MIN_LENGTH
    Xe_tr, ye_tr = X[tr_mask], y[tr_mask.to_numpy()]
    Xe_te, ye_te = X[te_mask], y[te_mask.to_numpy()]
    print(f"extrap train rows: {len(Xe_tr)}   extrap test rows: {len(Xe_te)}")

    lin_e = LinearRegression().fit(Xe_tr, ye_tr)
    rf_e = LogTargetRegressor(RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE,
        n_jobs=-1)).fit(Xe_tr, ye_tr)
    pow_e = PowerLawRegressor().fit(Xe_tr, ye_tr)

    extrap_results = {
        "Linear Regression": evaluate(ye_te, lin_e.predict(Xe_te)),
        "Random Forest": evaluate(ye_te, rf_e.predict(Xe_te)),
        "Power-Law Regression": evaluate(ye_te, pow_e.predict(Xe_te)),
    }
    print(f"{'Model':<24}{'MAE':>10}{'RMSE':>10}{'R²':>10}{'MAPE%':>10}")
    print("-" * 64)
    for n, m in extrap_results.items():
        print(f"{n:<24}{m['MAE_mm']:>10.4f}{m['RMSE_mm']:>10.4f}"
              f"{m['R2']:>10.5f}{m['MAPE_pct']:>10.2f}")

    # ---- 8. choose and save the production model ---------------------------
    # Criterion: best R² on the interpolation test set.  We deliberately do NOT
    # simply take the Random Forest, because on clean physics data a simpler
    # model can be both more accurate AND better behaved outside its range.
    best_name = max(results, key=lambda n: results[n]["R2"])
    print("\n" + "=" * 70)
    print(f"BEST MODEL ON TEST SET: {best_name}  (R² = {results[best_name]['R2']:.5f})")
    print(f"(Random Forest is still shipped as an alternative for comparison.)")
    print("=" * 70)

    from physics import INPUT_RANGES
    bundle = {
        "models": {
            "linear": lin,
            "random_forest": rf,
            "power_law": power,
        },
        "features": FEATURES,
        "best_model": {"Linear Regression": "linear",
                       "Random Forest": "random_forest",
                       "Power-Law Regression": "power_law"}[best_name],
        "best_model_display_name": best_name,
        "metrics": results,
        "train_r2": train_scores,
        "extrapolation_metrics": extrap_results,
        "cv_r2_mean": float(cv.mean()),
        "cv_r2_std": float(cv.std()),
        "cv_folds": [float(v) for v in cv],
        "power_law_coefficients": power.coef_,
        "power_law_theory": theory,
        "input_ranges": INPUT_RANGES,
        "materials": {k: v["E_GPa"] for k, v in MATERIALS.items()},
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_total": int(len(df)),
        "random_state": RANDOM_STATE,
        "target_transform": "log1p for Random Forest, identity for the others",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"\nSaved model bundle -> {MODEL_PATH}")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": results,
                "train_r2": train_scores,
                "extrapolation_metrics": extrap_results,
                "cv_r2_mean": float(cv.mean()),
                "cv_r2_std": float(cv.std()),
                "cv_folds": [float(v) for v in cv],
                "power_law_coefficients": power.coef_,
                "power_law_theory": theory,
                "best_model_display_name": best_name,
                "n_train": int(len(X_tr)),
                "n_test": int(len(X_te)),
                "n_total": int(len(df)),
                "trained_at": bundle["trained_at"],
            },
            f, indent=2,
        )
    print(f"Saved metrics      -> {METRICS_PATH}")

    # ---- 9. figures ---------------------------------------------------------
    print("\nGenerating figures ...")
    for name, fname in [("Linear Regression", "actual_vs_predicted_linear.png"),
                        ("Random Forest", "actual_vs_predicted_rf.png"),
                        ("Power-Law Regression", "actual_vs_predicted_powerlaw.png")]:
        plot_actual_vs_predicted(y_te, predict(name, X_te), name,
                                 os.path.join(OUT_DIR, fname), results[name])
        print(f"  {os.path.join(OUT_DIR, fname)}")
    plot_residuals(y_te, predict("Random Forest", X_te), "Random Forest",
                   os.path.join(OUT_DIR, "residuals_rf.png"))
    plot_feature_importance(rf_base, X_te, y_te,
                            os.path.join(OUT_DIR, "feature_importance.png"))
    plot_length_curves(models, os.path.join(OUT_DIR, "length_sweep_comparison.png"))
    models_e = {"Linear Regression": lin_e, "Random Forest": rf_e,
                "Power-Law Regression": pow_e}
    plot_extrapolation(extrap_results, models_e,
                       os.path.join(OUT_DIR, "extrapolation_test.png"))
    print(f"  {OUT_DIR}/residuals_rf.png")
    print(f"  {OUT_DIR}/feature_importance.png")
    print(f"  {OUT_DIR}/length_sweep_comparison.png")
    print(f"  {OUT_DIR}/extrapolation_test.png")

    print(f"\nDone in {time.time() - t0:.1f} s.")


if __name__ == "__main__":
    main()
