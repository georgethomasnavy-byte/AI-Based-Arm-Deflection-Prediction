"""
app.py  --  AI-Based Arm Deflection Prediction
==============================================
The Streamlit user interface.  Run with:

    streamlit run app.py

Everything the app shows comes from three places, and it is worth knowing which:
  1. physics.py        -> the exact analytical formula (labelled "reference")
  2. models/model.joblib -> the trained ML models (the actual predictions)
  3. models/metrics.json -> test-set scores measured during training

The app never re-trains anything; it only loads and applies the saved model.
"""

from __future__ import annotations

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from physics import (
    INPUT_RANGES,
    LOAD_CASE_DISTRIBUTED,
    LOAD_CASE_NAMES,
    LOAD_CASE_TIP,
    MATERIALS,
    assess_plausibility,
    moment_of_inertia_circular,
    tip_deflection_mm,
)

# ------------------------------------------------------------------------------
# PAGE SET-UP
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Arm Deflection Prediction",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "models/model.joblib"
METRICS_PATH = "models/metrics.json"

# Friendly names <-> keys inside the saved bundle
MODEL_KEYS = {
    "Random Forest": "random_forest",
    "Power-Law Regression": "power_law",
    "Linear Regression": "linear",
}


# ------------------------------------------------------------------------------
# LOADING (cached so the 31 MB model is read only once per session)
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model ...")
def load_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metrics():
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


if not os.path.exists(MODEL_PATH) or not os.path.exists(METRICS_PATH):
    st.error(
        "**Trained model not found.**\n\n"
        f"Expected `{MODEL_PATH}` and `{METRICS_PATH}`.\n\n"
        "Run the pipeline first (in this order):\n\n"
        "```\npython generate_dataset.py\npython train_model.py\n```"
    )
    st.stop()

BUNDLE = load_bundle()
METRICS = load_metrics()


# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------
def predict_with(name: str, row: dict) -> float:
    """Predict deflection (mm) for one input row using the chosen model."""
    X = pd.DataFrame([row])
    key = MODEL_KEYS[name]
    value = float(BUNDLE["models"][key].predict(X)[0])
    return max(value, 0.0)          # a negative deflection is not physical


def sweep(model_name: str, base: dict, var: str, values: np.ndarray) -> np.ndarray:
    """Predict a curve while varying one input and holding the rest fixed."""
    rows = []
    for v in values:
        r = dict(base)
        r[var] = float(v)
        rows.append(r)
    X = pd.DataFrame(rows)
    pred = BUNDLE["models"][MODEL_KEYS[model_name]].predict(X)
    return np.maximum(pred, 0.0)


def style_axes(ax):
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ------------------------------------------------------------------------------
# SIDEBAR -- INPUTS
# ------------------------------------------------------------------------------
st.sidebar.title("🦾 Input Parameters")
st.sidebar.caption("Change the values, then press **Predict Deflection**.")


def _set_E_from_material():
    """When the user picks a material, copy its Young's modulus into the E field."""
    st.session_state["E_GPa"] = MATERIALS[st.session_state["material"]]["E_GPa"]


material = st.sidebar.selectbox(
    "Material",
    options=list(MATERIALS.keys()),
    index=list(MATERIALS.keys()).index("Aluminium 6061-T6"),
    key="material",
    on_change=_set_E_from_material,
    help="Choosing a material fills in its typical Young's modulus. "
         "You can still override the value below.",
)

if "E_GPa" not in st.session_state:
    st.session_state["E_GPa"] = MATERIALS[material]["E_GPa"]

with st.sidebar.form("input_form", border=True):
    load_case_label = st.radio(
        "Load case",
        options=[LOAD_CASE_NAMES[LOAD_CASE_TIP], LOAD_CASE_NAMES[LOAD_CASE_DISTRIBUTED]],
        index=0,
        help="Tip point load: the mass hangs at the free end.\n\n"
             "Uniformly distributed: the mass is spread evenly along the arm "
             "(for the same total mass this bends the arm less).",
    )

    length_mm = st.number_input(
        "Arm length L (mm)",
        min_value=float(INPUT_RANGES["length_mm"][0]),
        max_value=float(INPUT_RANGES["length_mm"][1]),
        value=400.0, step=10.0, format="%.1f",
    )

    payload_kg = st.number_input(
        "Payload mass (kg)",
        min_value=float(INPUT_RANGES["payload_kg"][0]),
        max_value=float(INPUT_RANGES["payload_kg"][1]),
        value=1.0, step=0.05, format="%.2f",
        help="Tip point load: the hanging mass. Distributed: the TOTAL mass "
             "spread along the arm.",
    )

    diameter_mm = st.number_input(
        "Cross-section diameter d (mm)",
        min_value=float(INPUT_RANGES["diameter_mm"][0]),
        max_value=float(INPUT_RANGES["diameter_mm"][1]),
        value=12.0, step=0.5, format="%.1f",
        help="Assumed SOLID CIRCULAR section.",
    )

    E_GPa = st.number_input(
        "Young's modulus E (GPa)",
        min_value=float(INPUT_RANGES["E_GPa"][0]),
        max_value=float(INPUT_RANGES["E_GPa"][1]),
        value=float(st.session_state["E_GPa"]),
        step=0.1, format="%.2f",
        help="Material stiffness. Filled in automatically from the material above.",
    )

    model_name = st.selectbox(
        "Prediction model",
        options=list(MODEL_KEYS.keys()),
        index=list(MODEL_KEYS.keys()).index(BUNDLE["best_model_display_name"]),
        help="The best-performing model is selected by default.",
    )

    # Note: the predict button gets its own full-width row because a two-column
    # split was clipping the label ("Predict D...") in the narrow sidebar.
    submitted = st.form_submit_button("Predict Deflection", type="primary",
                                      width='stretch')
    reset = st.form_submit_button("Reset all inputs", width='stretch')

if reset:
    st.session_state.clear()
    st.rerun()

load_case = (LOAD_CASE_TIP if load_case_label == LOAD_CASE_NAMES[LOAD_CASE_TIP]
             else LOAD_CASE_DISTRIBUTED)

if submitted:
    st.session_state["result"] = {
        "length_mm": float(length_mm),
        "payload_kg": float(payload_kg),
        "diameter_mm": float(diameter_mm),
        "E_GPa": float(E_GPa),
        "load_case": int(load_case),
        "material": material,
        "model_name": model_name,
    }


# ------------------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------------------
st.title("AI-Based Arm Deflection Prediction")
st.markdown(
    "Predict the **tip deflection of a flexible robotic arm**, modelled as a "
    "cantilever beam, using machine-learning models trained on a synthetic "
    "dataset generated from Euler–Bernoulli beam theory."
)

result = st.session_state.get("result")

if result is None:
    st.info(
        "👈 **Set the parameters in the sidebar and press `Predict Deflection` to begin.**\n\n"
        "Try this example: aluminium arm, 400 mm long, Ø 12 mm, 1.0 kg at the tip."
    )
    st.stop()


# ------------------------------------------------------------------------------
# RE-VALIDATE THE STORED INPUTS ON EVERY RUN
# ------------------------------------------------------------------------------
problems = []
for label, val, key, unit in [
    ("Arm length", result["length_mm"], "length_mm", "mm"),
    ("Payload mass", result["payload_kg"], "payload_kg", "kg"),
    ("Diameter", result["diameter_mm"], "diameter_mm", "mm"),
    ("Young's modulus", result["E_GPa"], "E_GPa", "GPa"),
]:
    lo, hi = INPUT_RANGES[key]
    if not np.isfinite(val):
        problems.append(f"{label} is not a valid number.")
    elif val < lo or val > hi:
        problems.append(f"{label} = {val:g} {unit} is outside the supported range "
                        f"({lo:g}–{hi:g} {unit}).")

if problems:
    st.error("**Input validation failed**\n\n" + "\n".join(f"- {p}" for p in problems))
    st.stop()


# ------------------------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------------------------
row = {
    "length_mm": result["length_mm"],
    "payload_kg": result["payload_kg"],
    "diameter_mm": result["diameter_mm"],
    "E_GPa": result["E_GPa"],
    "load_case": result["load_case"],
}

reference_mm = tip_deflection_mm(
    result["length_mm"], result["payload_kg"], result["E_GPa"],
    result["diameter_mm"], result["load_case"],
)

pred_mm = predict_with(result["model_name"], row)
rel_err_vs_ref = (pred_mm - reference_mm) / reference_mm * 100.0

assessment = assess_plausibility(
    result["length_mm"], result["payload_kg"], result["diameter_mm"],
    result["load_case"], result["material"], pred_mm,
)

st.divider()
c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
with c1:
    st.metric("Predicted tip deflection", f"{pred_mm:.2f} mm",
              help=f"From the {result['model_name']} model.")
with c2:
    st.metric("Analytical reference", f"{reference_mm:.2f} mm",
              help="Closed-form beam equation, NOT machine learning. Shown for comparison.")
with c3:
    st.metric("ML vs formula", f"{rel_err_vs_ref:+.1f} %",
              help="How far the ML prediction sits from the analytical value.")
with c4:
    st.metric("Deflection / length", f"{assessment['deflection_ratio']*100:.1f} %",
              help="Above 10% the small-deflection theory stops being valid.")

if pred_mm < 0.005:
    st.caption("A predicted deflection below 0.005 mm is effectively zero at "
               "practical measurement resolution.")

if assessment["warnings"]:
    for w in assessment["warnings"]:
        st.warning(w)
else:
    st.success(
        f"Result is physically consistent: the arm droops "
        f"{pred_mm:.2f} mm on a {result['length_mm']:.0f} mm span "
        f"({assessment['deflection_ratio']*100:.1f}% of its length), and the "
        f"estimated root bending stress ({assessment['stress_MPa']:.0f} MPa) is "
        f"within the safe working limit for {result['material']}."
    )

st.caption(
    "**How to read this:** the *predicted* value comes from the trained ML model. "
    "The *analytical reference* comes from the textbook beam equation. On synthetic "
    "data the formula is the ground truth; the ML model is an approximation of it."
)


# ------------------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Charts", "🔬 Parameter Effects", "📊 Model Performance", "ℹ️ About & Limitations"]
)

# ---------------------------------------------------------------- TAB 1: CHARTS
with tab1:
    case_label = LOAD_CASE_NAMES[result["load_case"]]
    st.markdown(f"All curves below hold the other inputs fixed. "
                f"Load case: **{case_label}**.")

    # --- 1. deflection vs length -------------------------------------------
    left, right = st.columns(2)

    with left:
        L_grid = np.linspace(INPUT_RANGES["length_mm"][0],
                             INPUT_RANGES["length_mm"][1], 120)
        y_ml = sweep(result["model_name"], row, "length_mm", L_grid)
        y_ref = np.array([tip_deflection_mm(l, result["payload_kg"], result["E_GPa"],
                                            result["diameter_mm"], result["load_case"])
                          for l in L_grid])
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(L_grid, y_ref, "k-", lw=2.6, alpha=0.8, label="Analytical formula")
        ax.plot(L_grid, y_ml, "--", lw=2.0, color="#2563eb",
                label=f"{result['model_name']} (ML)")
        ax.plot([result["length_mm"]], [pred_mm], "o", ms=11, mfc="#f59e0b",
                mec="k", mew=1.2, zorder=5, label="Your input")
        ax.set_xlabel("Arm length L (mm)")
        ax.set_ylabel("Tip deflection δ (mm)")
        ax.set_title("Deflection vs arm length  (note the L³ shape)")
        ax.legend(fontsize=8)
        style_axes(ax)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    # --- 2. deflection vs payload ------------------------------------------
    with right:
        P_grid = np.linspace(INPUT_RANGES["payload_kg"][0],
                             INPUT_RANGES["payload_kg"][1], 120)
        y_ml = sweep(result["model_name"], row, "payload_kg", P_grid)
        y_ref = np.array([tip_deflection_mm(result["length_mm"], p, result["E_GPa"],
                                            result["diameter_mm"], result["load_case"])
                          for p in P_grid])
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(P_grid, y_ref, "k-", lw=2.6, alpha=0.8, label="Analytical formula")
        ax.plot(P_grid, y_ml, "--", lw=2.0, color="#2563eb",
                label=f"{result['model_name']} (ML)")
        ax.plot([result["payload_kg"]], [pred_mm], "o", ms=11, mfc="#f59e0b",
                mec="k", mew=1.2, zorder=5, label="Your input")
        ax.set_xlabel("Payload mass (kg)")
        ax.set_ylabel("Tip deflection δ (mm)")
        ax.set_title("Deflection vs payload  (straight line: δ ∝ load)")
        ax.legend(fontsize=8)
        style_axes(ax)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    # --- 3. diameter sweep + bar comparison --------------------------------
    left, right = st.columns(2)

    with left:
        d_grid = np.linspace(INPUT_RANGES["diameter_mm"][0],
                             INPUT_RANGES["diameter_mm"][1], 120)
        y_ml = sweep(result["model_name"], row, "diameter_mm", d_grid)
        y_ref = np.array([tip_deflection_mm(result["length_mm"], result["payload_kg"],
                                            result["E_GPa"], d, result["load_case"])
                          for d in d_grid])
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(d_grid, y_ref, "k-", lw=2.6, alpha=0.8, label="Analytical formula")
        ax.plot(d_grid, y_ml, "--", lw=2.0, color="#2563eb",
                label=f"{result['model_name']} (ML)")
        ax.plot([result["diameter_mm"]], [pred_mm], "o", ms=11, mfc="#f59e0b",
                mec="k", mew=1.2, zorder=5, label="Your input")
        ax.set_xlabel("Diameter d (mm)")
        ax.set_ylabel("Tip deflection δ (mm)")
        ax.set_title("Deflection vs thickness  (δ ∝ 1/d⁴ — very sensitive)")
        ax.legend(fontsize=8)
        style_axes(ax)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with right:
        names = list(MODEL_KEYS.keys())
        preds = [predict_with(n, row) for n in names]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        colors = ["#dc2626", "#16a34a", "#2563eb"]
        bars = ax.bar([n.replace(" ", "\n") for n in names], preds,
                      color=colors, alpha=0.85)
        ax.axhline(reference_mm, color="k", ls="--", lw=1.6,
                   label=f"Analytical formula ({reference_mm:.2f} mm)")
        for b, v in zip(bars, preds):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel("Predicted deflection (mm)")
        ax.set_title("All three models on your inputs")
        ax.legend(fontsize=8)
        style_axes(ax)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with st.expander("Show the numbers behind these charts"):
        table = pd.DataFrame(
            {n: [round(predict_with(n, row), 4)] for n in MODEL_KEYS},
            index=["Predicted deflection (mm)"],
        ).T
        table["Difference vs formula (mm)"] = (table["Predicted deflection (mm)"]
                                               - reference_mm).round(4)
        table["Error (%)"] = ((table["Predicted deflection (mm)"] - reference_mm)
                              / reference_mm * 100).round(2)
        st.dataframe(table, width='stretch')
        I = moment_of_inertia_circular(result["diameter_mm"])
        st.caption(
            f"Derived quantities: I = πd⁴/64 = {I:.4e} m⁴  |  "
            f"root bending stress ≈ {assessment['stress_MPa']:.1f} MPa  |  "
            f"material: {result['material']} "
            f"(E = {result['E_GPa']:.2f} GPa)"
        )

# --------------------------------------------------- TAB 2: PARAMETER EFFECTS
with tab2:
    st.markdown(
        "### How sensitive is the deflection to each input?\n"
        "Each input is increased by **10%** while the others are held constant. "
        "The table shows how much the predicted deflection changes."
    )

    rows = []
    base_pred = pred_mm
    for label, key, unit, theory_exp in [
        ("Arm length", "length_mm", "mm", 3.0),
        ("Payload", "payload_kg", "kg", 1.0),
        ("Diameter", "diameter_mm", "mm", -4.0),
        ("Young's modulus", "E_GPa", "GPa", -1.0),
    ]:
        bumped = dict(row)
        bumped[key] = row[key] * 1.10
        if bumped[key] > INPUT_RANGES[key][1]:
            bumped[key] = INPUT_RANGES[key][1]
        new_pred = predict_with(result["model_name"], bumped)
        pct = (new_pred - base_pred) / base_pred * 100.0
        rows.append({
            "Parameter": f"{label} ({unit})",
            "Current value": f"{row[key]:.3g}",
            "+10% gives": f"{bumped[key]:.3g}",
            "Change in deflection": f"{pct:+.1f} %",
            "Expected from theory": f"{theory_exp*10:+.0f} %",
        })

    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.markdown(
        """
**Reading this table**

* **Arm length is the most dangerous parameter.** Theory says a 10% longer arm
  gives about **+33%** deflection, because δ grows with **L³**. This is why long,
  slender arms are hard to keep stiff.
* **Diameter is even more dramatic per unit change**, because δ ∝ 1/d⁴: a 10%
  thicker arm cuts deflection by roughly 32%. Doubling the diameter reduces
  deflection by about 16 times.
* **Payload is linear**: double the load, double the droop.
* **Young's modulus is linear but often a weak lever**, because most metals sit
  in a narrow stiffness band. Going from aluminium to steel (≈3× stiffer) only
  buys you a factor of 3, whereas changing the diameter is far more powerful.
* Where the observed change differs from theory, that difference is the model's
  approximation error — most visible for the Random Forest.
        """
    )

    st.info(
        "**Design insight:** if your arm droops too much, increasing the diameter "
        "is almost always more effective than changing material."
    )

# ------------------------------------------------- TAB 3: MODEL PERFORMANCE
with tab3:
    st.markdown("### How well do the models actually perform?")
    st.caption(
        f"All figures below are measured on **{METRICS['n_test']} rows the models "
        f"never saw during training** (a held-out {METRICS['n_test']/METRICS['n_total']*100:.0f}% "
        f"of the dataset, selected with a fixed random seed)."
    )

    perf = pd.DataFrame(METRICS["metrics"]).T.reset_index()
    perf.columns = ["Model", "MAE (mm)", "RMSE (mm)", "R²", "MAPE (%)",
                    "Within 10% (%)"]
    perf = perf[["Model", "MAE (mm)", "RMSE (mm)", "R²", "MAPE (%)", "Within 10% (%)"]]
    st.dataframe(
        perf.style.format({
            "MAE (mm)": "{:.4f}", "RMSE (mm)": "{:.3f}", "R²": "{:.5f}",
            "MAPE (%)": "{:.2f}", "Within 10% (%)": "{:.1f}",
        }),
        width='stretch', hide_index=True,
    )

    st.markdown(
        f"""
**What the metrics mean**

| Metric | Plain-language meaning |
|---|---|
| **MAE** | Average error in millimetres — "typically off by this much". |
| **RMSE** | Similar, but large errors are punished more heavily. RMSE ≫ MAE means a few big misses. |
| **R²** | Fraction of the variation in deflection explained by the model. 1.0 = perfect, 0 = no better than always guessing the mean. |
| **MAPE** | Average error as a percentage. Useful here because deflection spans 0.01 – 80 mm. |
| **Within 10%** | How often the prediction lands within 10% of the true value. |

Cross-validation on the training set gave the Random Forest
**R² = {METRICS['cv_r2_mean']:.4f} ± {METRICS['cv_r2_std']:.4f}**
(3 folds), consistent with its test score — so it is not over-fitted.
        """
    )

    st.divider()
    st.markdown("### The most interesting result: recovered physics")
    st.markdown(
        "The power-law model was fitted as a **straight line in log-space**. "
        "It was told nothing about beam theory. Yet the exponents it learned "
        "match the theoretical ones almost exactly — including the cube on length "
        "and the fourth power on diameter:"
    )
    labels = {
        "log_length_mm": ("Length L", 3.0),
        "log_payload_kg": ("Payload P", 1.0),
        "log_diameter_mm": ("Diameter d", -4.0),
        "log_E_GPa": ("Young's modulus E", -1.0),
    }
    coef_rows = []
    for key, (label, theory) in labels.items():
        got = METRICS["power_law_coefficients"][key]
        coef_rows.append({
            "Quantity": label,
            "Exponent learned by model": round(got, 4),
            "Exponent from beam theory": theory,
            "Difference": round(got - theory, 4),
        })
    st.dataframe(pd.DataFrame(coef_rows), width='stretch', hide_index=True)

    st.divider()
    st.markdown("### Interpolation vs extrapolation")
    st.markdown(
        f"The models were trained on arms up to **650 mm** and then tested on arms "
        f"of **680–800 mm**, which they never saw. This measures **extrapolation** — "
        f"predicting outside the training range."
    )
    ext = pd.DataFrame(METRICS["extrapolation_metrics"]).T.reset_index()
    ext.columns = ["Model", "MAE (mm)", "RMSE (mm)", "R²", "MAPE (%)", "Within 10% (%)"]
    st.dataframe(
        ext[["Model", "MAE (mm)", "RMSE (mm)", "R²", "MAPE (%)"]]
        .style.format({"MAE (mm)": "{:.4f}", "RMSE (mm)": "{:.3f}",
                       "R²": "{:.5f}", "MAPE (%)": "{:.2f}"}),
        width='stretch', hide_index=True,
    )
    st.markdown(
        """
**Why the Random Forest degrades so badly outside its range:** a decision tree
predicts by averaging the training targets that land in a leaf. It therefore
**cannot output a value larger than the largest target it ever saw**. Past the
end of its data, its predictions flatten out — the curve you saw in
`outputs/extrapolation_test.png`. The physics-based power-law model has no such
limitation because it learned a *relationship*, not a lookup table.
        """
    )

    st.divider()
    st.markdown("### Training-set figures")
    st.caption(
        "Generated by `train_model.py` into the `outputs/` folder. Click to enlarge."
    )
    img_cols = st.columns(3)
    images = [
        ("outputs/actual_vs_predicted_rf.png", "Random Forest: actual vs predicted"),
        ("outputs/actual_vs_predicted_powerlaw.png", "Power-law: actual vs predicted"),
        ("outputs/feature_importance.png", "Feature importance"),
        ("outputs/residuals_rf.png", "Random Forest residuals"),
        ("outputs/length_sweep_comparison.png", "Length sweep: all models"),
        ("outputs/extrapolation_test.png", "Extrapolation test"),
    ]
    for i, (path, caption) in enumerate(images):
        with img_cols[i % 3]:
            if os.path.exists(path):
                st.image(path, caption=caption, width='stretch')
            else:
                st.caption(f"({path} not found — run train_model.py)")

# ------------------------------------------------------ TAB 4: ABOUT
with tab4:
    st.markdown(
        f"""
### What this application does

It predicts the **tip deflection** of a flexible robotic arm modelled as a
cantilever beam — a beam clamped rigidly at one end and free at the other.
Two load cases are supported:

| Load case | Formula | Note |
|---|---|---|
| Tip point load | δ = P·g·L³ / (3·E·I) | Mass hangs at the free end |
| Uniformly distributed | δ = M·g·L³ / (8·E·I) | Mass spread along the arm |

with **I = π·d⁴ / 64** for a solid circular section.

The **3/8 factor**: for the same total mass, a distributed load deflects the arm
only 37.5% as much as a tip load, because its resultant force acts at the
mid-point rather than at the tip.

### Where the data comes from — stated plainly

> The dataset is **synthetic**. It was generated by evaluating the beam equations
> above 40,000 times over the supported input ranges, then adding ±1.2% relative
> noise and a 0.01 mm noise floor to imitate real instrument readings.
>
> **It contains no experimental measurements and no sensor data.**
> The machine-learning models therefore learn an *approximation of a known
> engineering formula*. They do not discover new physics.

This is a legitimate way to demonstrate an end-to-end ML pipeline, and it is
exactly how a surrogate model would be bootstrapped before real test-rig data
became available. The pipeline in `generate_dataset.py` and `train_model.py` is
the same one you would use once real measurements exist — only the data source
would change.

### Data quality controls applied

* Samples where **δ > L/10** are rejected, because small-deflection beam theory
  (which the whole dataset rests on) is invalid beyond that point.
* Samples where the **root bending stress exceeds 60% of the material strength**
  are rejected, because such a beam would yield before reaching that deflection.
* Deflections **below 0.01 mm** are discarded as unmeasurable.
* A **fixed random seed (42)** makes the dataset and every reported score
  exactly reproducible on any machine.

### Assumptions and limitations — please state these in your report

1. **Synthetic data, not experiments.** No physical arm, no sensors, no test rig.
2. **Small-deflection theory only.** Valid while δ ≲ L/10. The app warns you when
   you leave that region rather than silently returning a number.
3. **Linear-elastic, homogeneous, isotropic material** obeying Hooke's law.
4. **Static loading only.** No vibration, no resonance, no fatigue, no dynamics.
5. **Tip point load or uniform load only.** Not a general distributed or moving load.
6. **Solid circular cross-section only.** Not hollow tubes, rectangles, or I-beams
   (though the theory generalises easily by changing I).
7. **Ideal rigid clamp.** Real joints have compliance that adds extra deflection.
8. **No self-weight, no shear deformation, no large-angle geometry.**
9. **Extrapolation is unsafe.** Inputs are limited to the trained ranges
   (L: 150–800 mm, load: 0.05–5 kg, d: 6–40 mm, E: 2.8–200 GPa). Outside these,
   the app refuses rather than guessing. Tree models in particular cannot
   extrapolate at all.
10. **Material values are textbook typical values**, not certified for design.
    Real 3D-printed PLA, for example, varies strongly with infill and orientation.

### Honest note on which model "wins"

The power-law regression model is the most accurate (R² ≈ 0.9998) because the
underlying physics is genuinely a power law, so a power law fits it almost
perfectly. The Random Forest (R² ≈ 0.944) is the more general-purpose model and
is the one requested for this project; it is more instructive precisely because
it *cannot* do what the power law does, and studying that failure teaches more
than a perfect score would. Both are shipped, and this app lets you switch
between them.

### Files in this project

| File | Purpose |
|---|---|
| `physics.py` | Beam equations, material data, input ranges, validation |
| `model_classes.py` | Model wrapper classes that must be importable to save/load |
| `generate_dataset.py` | Creates `data/beam_deflection_dataset.csv` |
| `train_model.py` | Trains, evaluates, plots, saves `models/model.joblib` |
| `app.py` | This Streamlit interface |
| `models/metrics.json` | Test-set scores shown on the Model Performance tab |
        """
    )

    st.divider()
    st.caption(
        f"Model trained {METRICS['trained_at']} on {METRICS['n_total']:,} synthetic "
        f"samples ({METRICS['n_train']:,} train / {METRICS['n_test']:,} test). "
        f"Random seed {BUNDLE['random_state']}."
    )
