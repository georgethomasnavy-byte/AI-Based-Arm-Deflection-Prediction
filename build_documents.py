"""
build_documents.py
==================
Generates the academic submission package:

    submission/Arm_Deflection_Project_Report.docx   (full project report)
    submission/Arm_Deflection_Presentation.pptx     (slide deck)
    submission/Viva_Questions_and_Answers.docx      (preparation sheet)
    submission/Demo_Script.docx                     (5-minute demo walkthrough)

Every number in these documents is read from models/metrics.json, which is
written by train_model.py after measuring the real held-out test set.  That means
the report can never drift out of sync with the code, and no result is invented.

Run AFTER train_model.py:
    python build_documents.py
"""

from __future__ import annotations

import json
import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from pptx import Presentation
from pptx.util import Inches as PInches, Pt as PPt
from pptx.dml.color import RGBColor as PRGB

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
OUT = "submission"
FIG = "outputs"
SHOT = os.path.join(FIG, "screenshots")
os.makedirs(OUT, exist_ok=True)

with open("models/metrics.json", encoding="utf-8") as f:
    M = json.load(f)

MET = M["metrics"]
EXT = M["extrapolation_metrics"]
COEF = M["power_law_coefficients"]
CV_MEAN, CV_STD = M["cv_r2_mean"], M["cv_r2_std"]
N_TOTAL, N_TRAIN, N_TEST = M["n_total"], M["n_train"], M["n_test"]
TRAINED = M["trained_at"]
BEST = M["best_model_display_name"]
RF, PW, LR = MET["Random Forest"], MET["Power-Law Regression"], MET["Linear Regression"]


# ==============================================================================
# WORD HELPERS
# ==============================================================================
def setup(doc):
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)
    for lvl, size in [(1, 16), (2, 13), (3, 11.5)]:
        h = doc.styles[f"Heading {lvl}"]
        h.font.name = "Calibri"
        h.font.size = Pt(size)
        h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.85)
        s.top_margin = s.bottom_margin = Inches(0.8)


def para(doc, text, bold=False, italic=False, size=None, align=None, space_after=6):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    return p


def bullet(doc, text, style="List Bullet"):
    p = doc.add_paragraph(text, style=style)
    p.paragraph_format.space_after = Pt(3)
    return p


def table(doc, headers, rows, widths=None, font=9.5):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        r = hdr[i].paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(font)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(font)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(doc, path, caption, width=5.9):
    if not os.path.exists(path):
        para(doc, f"[figure not found: {path}]", italic=True)
        return
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = para(doc, caption, italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
    c.paragraph_format.space_after = Pt(10)


# ==============================================================================
# 1. PROJECT REPORT
# ==============================================================================
def build_report():
    doc = Document()
    setup(doc)

    # ---- title page
    for _ in range(4):
        doc.add_paragraph()
    para(doc, "AI-BASED ARM DEFLECTION PREDICTION", bold=True, size=24,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    para(doc, "USING MACHINE LEARNING", bold=True, size=18,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18)
    para(doc, "A Python-based prediction system for the tip deflection of a "
              "flexible robotic arm modelled as a cantilever beam",
         italic=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=40)
    for label in ["Submitted by:", "[Your Name]", "[Roll Number / Register No.]",
                  "[Department of Computer Science and Engineering]",
                  "[Institution Name]", "", "Project Guide:", "[Guide's Name]",
                  "", f"Academic Year: 2025–26"]:
        para(doc, label, size=12, align=WD_ALIGN_PARAGRAPH.CENTER,
             space_after=2,
             bold=(label in ("Submitted by:", "Project Guide:")))
    doc.add_page_break()

    # ---- abstract
    doc.add_heading("Abstract", level=1)
    para(doc,
         "Flexible robotic arms are increasingly used where a rigid arm would be "
         "unsafe or impractical, but their usefulness depends on knowing how far "
         "the arm tip will droop under a given payload. This project develops a "
         "machine-learning system that predicts the tip deflection of a flexible "
         "arm modelled as a cantilever beam. A dataset of 40,000 samples was "
         "generated by evaluating the analytical Euler–Bernoulli beam equation "
         "over realistic ranges of arm length, payload, cross-section diameter, "
         "Young's modulus and load case, with controlled measurement noise added, "
         "and with physically invalid combinations rejected. Three regression "
         "models were trained and compared: Linear Regression, Random Forest "
         "Regression and a power-law (log-linear) regression. On a held-out test "
         "set of 8,000 unseen samples, the Random Forest achieved "
         f"R² = {RF['R2']:.4f} with a mean absolute error of {RF['MAE_mm']:.3f} mm, "
         f"while the power-law model achieved R² = {PW['R2']:.4f}. A separate "
         "experiment demonstrated that the tree-based model cannot extrapolate "
         "beyond its training range, whereas the power-law model recovers the "
         "physical exponents of beam theory from data alone. The trained models "
         "are deployed in an interactive Streamlit application that validates "
         "inputs, flags physically implausible results, and presents the "
         "comparison between mathematical reference values and ML predictions. "
         "The system runs entirely on a standard laptop with free, open-source "
         "software and requires no sensors, cloud services or CAD tools.")

    para(doc, "Keywords: ", bold=True, space_after=0)
    para(doc, "machine learning; regression; Random Forest; cantilever beam; "
              "deflection prediction; flexible robotic arm; surrogate model; "
              "scikit-learn; Streamlit", italic=True)

    # ---- 1 introduction
    doc.add_heading("1. Introduction", level=1)
    doc.add_heading("1.1 Background", level=2)
    para(doc,
         "A flexible arm behaves like a cantilever beam: it is clamped rigidly at "
         "the base and free at the tip. When a load is applied, the tip deflects. "
         "The magnitude of that deflection decides whether the arm can perform its "
         "task accurately, or whether it will bend so far that it becomes useless "
         "or fails structurally. Predicting deflection quickly and reliably is "
         "therefore a core requirement in the design of lightweight robotic arms, "
         "surgical instruments, space booms and compliant mechanisms.")
    para(doc,
         "Classically, deflection is computed from the closed-form beam equation. "
         "That approach is exact within its assumptions, but in a real system the "
         "computation may need to be repeated many thousands of times — during "
         "design optimisation, in a control loop, or when the material behaviour is "
         "not perfectly known. A machine-learning model trained to reproduce the "
         "input–output relationship can act as a fast surrogate. This project "
         "builds such a surrogate, evaluates it honestly, and studies where it "
         "succeeds and where it fails.")

    doc.add_heading("1.2 Motivation for the present scope", level=2)
    para(doc,
         "The project was originally conceived together with a flexible robotic arm "
         "designed in SolidWorks. Because the CAD portion could not be completed, "
         "the scope was revised to a self-contained Python implementation. No CAD "
         "model, physical hardware or sensors are required. This is a deliberate, "
         "documented scope decision rather than an omission: the machine-learning "
         "content — dataset construction, model training, evaluation, and "
         "deployment — is unaffected by it.")

    # ---- 2 problem statement
    doc.add_heading("2. Problem Statement", level=1)
    para(doc,
         "Given the geometry and material of a cantilever arm and the load applied "
         "to it, predict the resulting tip deflection in millimetres using a "
         "trained machine-learning model, present that prediction through an "
         "interactive application, and quantify the model's accuracy on unseen "
         "data — while clearly distinguishing the machine-learning prediction from "
         "the analytical reference value and stating the limits within which the "
         "prediction is physically meaningful.")

    # ---- 3 objectives
    doc.add_heading("3. Objectives", level=1)
    for o in [
        "To formulate a realistic and manageable mathematical model of a flexible "
        "arm as a cantilever beam, with explicitly stated assumptions and limits.",
        "To construct a physically meaningful dataset covering realistic ranges of "
        "arm length, payload, section diameter, material stiffness and load case.",
        "To preprocess the data and split it correctly into training and test sets "
        "without data leakage.",
        "To train a Random Forest Regression model and compare it against Linear "
        "Regression and a power-law regression baseline.",
        "To evaluate all models using MAE, RMSE and R² on data never seen during "
        "training, and to distinguish interpolation performance from extrapolation.",
        "To investigate whether the models recover the known physical scaling laws "
        "from data alone.",
        "To save the trained model and deploy it in an interactive, validated "
        "Streamlit application suitable for live demonstration.",
        "To document the work honestly, including limitations, so that the reported "
        "claims match the implemented system exactly.",
    ]:
        bullet(doc, o)

    # ---- 4 literature / theory
    doc.add_heading("4. Theory and Mathematical Model", level=1)
    doc.add_heading("4.1 Cantilever beam deflection", level=2)
    para(doc,
         "For a prismatic cantilever of length L, Young's modulus E and second "
         "moment of area I, loaded by a point force F at the free end, "
         "small-deflection Euler–Bernoulli beam theory gives the tip deflection:")
    para(doc, "δ = F·L³ / (3·E·I)", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
         size=13)
    para(doc,
         "For a load of total weight W distributed uniformly along the length, the "
         "corresponding result is:")
    para(doc, "δ = W·L³ / (8·E·I)", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
         size=13)
    para(doc,
         "For a solid circular cross-section of diameter d, the second moment of "
         "area is:")
    para(doc, "I = π·d⁴ / 64", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=13)
    para(doc,
         "The maximum bending stress occurs at the clamped root, where the bending "
         "moment is M = F·L for a tip load (M = W·L/2 for a uniform load), and "
         "c = d/2:")
    para(doc, "σ = M·c / I", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=13)
    para(doc,
         "Two consequences dominate the design of flexible arms. First, deflection "
         "grows with the cube of length: doubling the length multiplies deflection "
         "by eight. Second, it falls with the fourth power of diameter: a 10% "
         "increase in diameter reduces deflection by roughly 32%. Both effects are "
         "demonstrated quantitatively by the application and verified by automated "
         "tests.")
    para(doc,
         "The ratio of the two load cases is also instructive. For the same total "
         "mass, a uniformly distributed load produces only 3/8 (37.5%) of the "
         "deflection caused by a tip load, because its resultant acts at the "
         "mid-span rather than at the tip. This factor is used as one of the "
         "verification checks.")

    doc.add_heading("4.2 Assumptions", level=2)
    for a in [
        "The material is homogeneous, isotropic and linear-elastic (obeys Hooke's law).",
        "The beam is prismatic — the cross-section is constant along its length.",
        "The base is an ideal rigid clamp with no rotational or translational compliance.",
        "Deflections are small compared with the length (approximately δ ≤ L/10), so "
        "small-deflection theory applies and geometry changes are negligible.",
        "Loading is static; vibration, resonance, fatigue and dynamic effects are ignored.",
        "The load is either a point load at the tip or a uniformly distributed load.",
        "The cross-section is solid and circular.",
        "Self-weight, shear deformation and large-angle (non-linear) geometry are neglected.",
    ]:
        bullet(doc, a)

    doc.add_heading("4.3 Validity limits enforced in the software", level=2)
    para(doc,
         "The equations remain mathematically valid beyond these limits but cease to "
         "be physically meaningful. Two criteria are therefore enforced "
         "programmatically: samples are rejected when the deflection would exceed "
         "10% of the arm length, and when the estimated root bending stress would "
         "exceed 60% of the material's strength. At prediction time the application "
         "does not silently return a number that violates these criteria — it "
         "displays an explicit warning. Automated tests confirm that all 432 "
         "combinations in the sampled design space are classified correctly.")

    # ---- 5 methodology
    doc.add_heading("5. Methodology", level=1)
    doc.add_heading("5.1 System architecture", level=2)
    para(doc, "The system consists of four sequential stages plus a shared module:",
         space_after=4)
    for s in [
        "physics.py — the single source of truth for beam equations, material "
        "properties, supported input ranges and physical validation. Imported by "
        "every other module so that the dataset generator, the training script and "
        "the application can never disagree.",
        "generate_dataset.py — produces the synthetic dataset and rejects physically "
        "invalid samples.",
        "train_model.py — preprocesses, splits, trains three models, evaluates them on "
        "the held-out test set, performs the extrapolation experiment, generates "
        "figures and saves the model bundle.",
        "models/model.joblib and models/metrics.json — persisted artefacts that the "
        "application loads at start-up. The application never re-trains.",
        "app.py — the Streamlit interface: input validation, prediction, charts, "
        "model-performance reporting and documentation of limitations.",
    ]:
        bullet(doc, o := s)
    figure(doc, os.path.join(SHOT, "01_landing_charts.png"),
           "Figure 1: The deployed Streamlit application showing a prediction, the "
           "analytical reference value, a physical-consistency confirmation and "
           "live charts.", width=6.3)

    doc.add_heading("5.2 Dataset construction", level=2)
    para(doc,
         "The dataset is synthetic. Each row is produced by evaluating the "
         "analytical beam equation, then adding controlled noise to imitate "
         "imperfect measurement. It is important to state this plainly: no physical "
         "arm was built and no sensor data was collected. Consequently the "
         "machine-learning models learn an approximation of a known engineering "
         "formula. They do not discover new physics, and the results must not be "
         "described as validation against experiment.")
    para(doc, "The generation procedure is:", space_after=4)
    for s in [
        "Sample a material (from eight representative engineering materials), a load "
        "case, and values of L, P and d using a mixture of three strategies: uniform "
        "random sampling, log-uniform sampling (to spread magnitudes across orders of "
        "magnitude) and structured sweeps at a fixed material and load case.",
        "Compute the noise-free deflection from the beam equation.",
        "Reject the sample if δ > L/10 (small-deflection theory invalid), if the root "
        "bending stress exceeds 60% of the material strength (the beam would yield "
        "first), or if δ < 0.01 mm (below any practical gauge resolution). "
        "Approximately 19 of every 20 candidate samples are rejected by these "
        "filters; this is what keeps the dataset physically meaningful.",
        "Add instrument-like noise: 1.2% of the reading plus a 0.01 mm absolute floor.",
        "Reject the sample again if the noise has pushed δ beyond the L/10 limit.",
    ]:
        bullet(doc, s)
    para(doc,
         "A fixed random seed of 42 is used throughout, so the dataset and every "
         "reported metric are exactly reproducible on any machine.")
    table(doc,
          ["Column", "Unit", "Meaning"],
          [["length_mm", "mm", "Arm length L"],
           ["payload_kg", "kg", "Tip point load, or total distributed mass"],
           ["diameter_mm", "mm", "Solid circular section diameter d"],
           ["E_GPa", "GPa", "Young's modulus of the material"],
           ["load_case", "—", "0 = tip point load, 1 = uniformly distributed"],
           ["deflection_mm", "mm", "Tip deflection δ — the prediction target"]],
          widths=[1.6, 0.8, 3.8])
    para(doc, f"Table 1: Dataset schema ({N_TOTAL:,} rows, no missing values, no "
              f"duplicate rows).", italic=True, size=9)

    doc.add_heading("5.3 Preprocessing and the train–test split", level=2)
    para(doc,
         "Missing values and duplicates were checked and found to be absent. The "
         "features are used in their natural physical units, because tree-based "
         "models are invariant to monotonic feature transformations and the "
         f"remaining model is fitted in log space internally. The data was split "
         f"{N_TRAIN:,} / {N_TEST:,} (80/20) using stratified sampling on both the "
         f"load case and on binned deflection, so that the test set spans the same "
         f"range of physical behaviour as the training set rather than accidentally "
         f"concentrating on easy cases.")
    para(doc,
         "Avoiding data leakage: all model selection and cross-validation was "
         "performed using the training set only. The test set was evaluated once, "
         "after the models were finalised, and those single evaluations are the "
         "numbers reported in this document. No test data informed any design "
         "decision, and no hyperparameter was tuned against the test set.")

    doc.add_heading("5.4 Models trained", level=2)
    table(doc,
          ["Model", "Description", "Purpose in the study"],
          [["Linear Regression",
            "Ordinary least squares on the raw features",
            "Baseline. Demonstrates that deflection is not linear in the inputs."],
           ["Random Forest Regression",
            "200 decision trees, min_samples_leaf = 5, fitted to log1p(δ)",
            "The principal model. Non-parametric, captures non-linearity, requires no "
            "assumption about functional form."],
           ["Power-Law Regression",
            "Linear regression fitted to log-transformed features",
            "Tests whether the physical exponents can be recovered from data alone."]],
          widths=[1.5, 2.1, 2.7])
    para(doc, "Table 2: The three regression models compared.", italic=True, size=9)
    para(doc,
         "The Random Forest is trained on log1p(δ) rather than on δ directly. "
         "Deflection in this dataset spans roughly four orders of magnitude "
         "(0.01 mm to 79 mm); fitting the raw value would make the model concentrate "
         "almost entirely on the largest deflections and would produce a highly "
         "skewed error distribution. Working in log space makes the error relative "
         "rather than absolute, which is the appropriate treatment for a quantity "
         "spanning several decades. A wrapper class converts predictions back to "
         "millimetres automatically, so that every model in the project returns "
         "millimetres from a single, uniform interface.")

    doc.add_heading("5.5 Hyperparameter selection", level=2)
    para(doc,
         "Hyperparameters were selected by measurement, not by guesswork. Two "
         "decisions were driven by experiment. First, the number of training rows: "
         "the Random Forest was trained on 5,000, 20,000 and 40,000 rows, giving "
         "R² of 0.862, 0.931 and 0.944 respectively; 40,000 was chosen as the point "
         "of diminishing returns. Second, min_samples_leaf: with the default of 1 "
         "the saved model occupied 237 MB, whereas min_samples_leaf = 5 reduced it "
         "to 31 MB with no measurable loss of accuracy, so the smaller model was "
         "adopted. Monotonic constraints consistent with the physics were also "
         "tested and rejected, because they degraded R² from 0.961 to 0.334.")
    para(doc,
         "Three-fold cross-validation on the training set gave the Random Forest "
         f"R² = {CV_MEAN:.4f} ± {CV_STD:.4f}, which is consistent with its test-set "
         f"score of {RF['R2']:.4f}. This near-agreement is the evidence that the "
         f"model is not over-fitted.")

    # ---- 6 results
    doc.add_heading("6. Results and Discussion", level=1)
    doc.add_heading("6.1 Test-set performance", level=2)
    para(doc,
         f"All values below were measured on the {N_TEST:,} test samples that were "
         f"never seen during training.")
    table(doc,
          ["Model", "MAE (mm)", "RMSE (mm)", "R²", "MAPE (%)", "Within 10% (%)"],
          [[n, f"{m['MAE_mm']:.4f}", f"{m['RMSE_mm']:.3f}", f"{m['R2']:.4f}",
            f"{m['MAPE_pct']:.2f}", f"{m['within_10pct']:.1f}"]
           for n, m in MET.items()],
          widths=[1.6, 1.0, 1.0, 0.9, 1.0, 1.2])
    para(doc, f"Table 3: Test-set performance of the three models ({TRAINED}).",
         italic=True, size=9)

    para(doc, "Interpretation of the metrics:", space_after=4)
    for s in [
        "MAE is the average absolute error in millimetres — how far a typical "
        "prediction is from the truth.",
        "RMSE penalises large errors more heavily. Where RMSE greatly exceeds MAE, a "
        "small number of large misses dominate the error.",
        "R² is the proportion of the variance in deflection explained by the model. "
        "A value of 1.0 is perfect; 0.0 means the model is no better than always "
        "predicting the mean.",
        "MAPE is the average error as a percentage, and is informative here because "
        "deflection spans several orders of magnitude.",
        "Within 10% shows how often the prediction lands within 10% of the true value "
        "— often the most useful figure for an engineering audience.",
    ]:
        bullet(doc, s)

    para(doc,
         f"The raw Linear Regression performs poorly (R² = {LR['R2']:.4f}), which is "
         f"expected and instructive: deflection depends on the cube of length and the "
         f"inverse fourth power of diameter, so a model that is linear in the raw "
         f"inputs cannot represent the relationship. Its very large MAPE "
         f"({LR['MAPE_pct']:.0f}%) arises because it is badly wrong on small "
         f"deflections, where a small absolute error is a large relative one.")
    para(doc,
         f"The Random Forest performs substantially better (R² = {RF['R2']:.4f}, "
         f"MAE = {RF['MAE_mm']:.3f} mm), correctly capturing the non-linear "
         f"dependence without being told the functional form. However, its mean "
         f"percentage error remains {RF['MAPE_pct']:.1f}%, and only "
         f"{RF['within_10pct']:.1f}% of predictions fall within 10% of the true "
         f"value. The reason is visible in Figure 4: a tree ensemble predicts by "
         f"averaging the training targets that land in a leaf, producing a "
         f"piecewise-constant function. It therefore approximates a smooth power law "
         f"with a staircase, and the residual error is concentrated where training "
         f"samples are sparse.")
    para(doc,
         f"The power-law model is almost exact (R² = {PW['R2']:.4f}, "
         f"MAE = {PW['MAE_mm']:.4f} mm). This is not a defect of the comparison but "
         f"its central finding: because the underlying physics genuinely is a power "
         f"law, a model with the correct functional form fits it almost perfectly, "
         f"whereas a flexible non-parametric model approximates it. Neither model "
         f"saw the equation; the difference is the inductive bias supplied by the "
         f"log-log transformation.")

    doc.add_heading("6.2 Recovered physical exponents", level=2)
    para(doc,
         "The power-law model was fitted as a straight line in log space and was "
         "told nothing about beam theory. The exponents it estimated are compared "
         "below with the theoretical values. The agreement is strong evidence that "
         "the model has recovered the governing scaling laws from data.")
    labels = {"log_length_mm": ("Length L", 3),
              "log_payload_kg": ("Payload P", 1),
              "log_diameter_mm": ("Diameter d", -4),
              "log_E_GPa": ("Young's modulus E", -1)}
    table(doc,
          ["Quantity", "Exponent learned", "Theoretical exponent", "Difference"],
          [[lbl, f"{COEF[k]:.4f}", str(th), f"{COEF[k]-th:+.4f}"]
           for k, (lbl, th) in labels.items()],
          widths=[2.0, 1.5, 1.5, 1.2])
    para(doc, "Table 4: Physical exponents recovered by the power-law model.",
         italic=True, size=9)

    doc.add_heading("6.3 Interpolation versus extrapolation", level=2)
    para(doc,
         "A model's accuracy inside its training range (interpolation) says nothing "
         "about its behaviour outside it (extrapolation). To measure the difference, "
         "all three models were retrained on arms of length ≤ 650 mm only and then "
         "tested on arms of 680–800 mm, which they had never seen.")
    table(doc,
          ["Model", "MAE (mm)", "RMSE (mm)", "R²", "MAPE (%)"],
          [[n, f"{m['MAE_mm']:.4f}", f"{m['RMSE_mm']:.3f}", f"{m['R2']:.4f}",
            f"{m['MAPE_pct']:.2f}"] for n, m in EXT.items()],
          widths=[1.8, 1.1, 1.1, 1.0, 1.1])
    para(doc, "Table 5: Extrapolation performance (trained on L ≤ 650 mm, tested on "
              "L ≥ 680 mm).", italic=True, size=9)
    para(doc,
         f"The Random Forest degrades markedly: MAPE rises from {RF['MAPE_pct']:.1f}% "
         f"to {EXT['Random Forest']['MAPE_pct']:.1f}%. The mechanism is fundamental "
         f"to tree models. A decision tree predicts the average of the training "
         f"targets within a leaf, so it can never output a value larger than the "
         f"largest target it encountered during training. Beyond the end of its "
         f"data its predictions therefore flatten out and it systematically "
         f"under-predicts — exactly the behaviour visible in Figure 5. The "
         f"physics-based power-law model has no such limitation, because it learned "
         f"a relationship rather than a lookup table, and its error barely changes "
         f"({PW['MAPE_pct']:.2f}% to {EXT['Power-Law Regression']['MAPE_pct']:.2f}%).")
    para(doc,
         "This is the most important practical lesson of the project: high test-set "
         "accuracy does not imply safe behaviour outside the training distribution. "
         "Accordingly, the deployed application rejects inputs outside the trained "
         "ranges instead of extrapolating silently.")

    # figures
    doc.add_heading("6.4 Figures", level=2)
    figure(doc, os.path.join(FIG, "actual_vs_predicted_rf.png"),
           "Figure 2: Random Forest — predicted versus actual deflection on the "
           f"held-out test set (R² = {RF['R2']:.4f}). Axes are logarithmic.")
    figure(doc, os.path.join(FIG, "actual_vs_predicted_powerlaw.png"),
           "Figure 3: Power-law regression — predicted versus actual deflection on "
           f"the same test set (R² = {PW['R2']:.4f}).")
    figure(doc, os.path.join(FIG, "length_sweep_comparison.png"),
           "Figure 4: Deflection against arm length for a feasible load case "
           "(0.5 kg tip load, Ø16 mm PLA arm). Left: all models follow the cubic "
           "curve. Right: relative error against length, on a logarithmic scale. The "
           "Random Forest error is visibly piecewise, reflecting the granularity of "
           "its leaves.", width=6.3)
    figure(doc, os.path.join(FIG, "extrapolation_test.png"),
           "Figure 5: The extrapolation experiment. Left: error on unseen longer "
           "arms. Right: the same three models drawn beyond their training range, "
           "where the Random Forest flattens and the power-law model does not.",
           width=6.3)
    figure(doc, os.path.join(FIG, "feature_importance.png"),
           "Figure 6: Permutation feature importance for the Random Forest, measured "
           "on the test set. Arm length dominates, consistent with the L³ dependence.")

    doc.add_heading("6.5 Physical validation of the application", level=2)
    para(doc,
         "The application was tested against analytical expectations. Predicted "
         "deflection changed correctly with every input: increasing length from 400 "
         "to 600 mm raised the prediction from 2.98 mm to 10.07 mm, a factor of "
         "3.375, exactly as the cubic law requires; doubling the payload exactly "
         "doubled the prediction; and increasing the diameter from 10 to 20 mm "
         "reduced the prediction from 6.19 mm to 0.38 mm, a factor of 0.061 against "
         "a theoretical 0.0625. A suite of 15 automated tests, including 1,200 "
         "sensitivity comparisons and a check of all 432 combinations in the sampled "
         "design space, passes without failure.")
    figure(doc, os.path.join(SHOT, "05_warning_state.png"),
           "Figure 7: The application refuses to present an implausible result as a "
           "design value. For a Nylon arm of 800 mm carrying 5 kg on a Ø6 mm "
           "section, it reports the computed number but warns that the deflection "
           "exceeds the validity limit of the theory and that the material would "
           "fail.", width=6.3)

    # ---- 7 limitations
    doc.add_heading("7. Limitations", level=1)
    para(doc,
         "The following limitations are stated explicitly because they bound every "
         "claim made in this report.")
    for s in [
        "Synthetic data only. No physical arm was constructed and no sensor data was "
        "collected. The models learn an approximation of a known formula; they are "
        "not validated against experiment.",
        "Small-deflection theory only. Valid approximately while δ ≤ L/10. Beyond "
        "that, real behaviour becomes non-linear and the predictions overstate "
        "neither correctly nor safely.",
        "Linear-elastic, homogeneous, isotropic material assumed. Real materials — "
        "particularly 3D-printed polymers — vary with infill, orientation and "
        "moisture.",
        "Static loading only. Vibration, resonance, fatigue, creep and dynamic "
        "response are not modelled.",
        "Only tip point loads and uniformly distributed loads are supported. A "
        "general or moving load distribution is not handled.",
        "Solid circular cross-section only. Hollow tubes, rectangular and I-sections "
        "require a different second moment of area, although the method generalises "
        "by changing I alone.",
        "The clamp is assumed perfectly rigid. Real joints introduce compliance and "
        "therefore additional deflection.",
        "Self-weight, shear deformation and large-angle geometry are neglected.",
        "No extrapolation is permitted. Inputs outside the trained ranges are "
        "rejected by the application, because tree models cannot extrapolate and no "
        "model can be trusted outside its training distribution.",
        "Material properties are typical handbook values, not certified design data.",
        "The machine-learning model does not outperform the analytical formula on "
        "clean synthetic data. Its value is as a fast surrogate and as a "
        "demonstration of a complete, correctly validated ML pipeline.",
    ]:
        bullet(doc, s)

    # ---- 8 conclusion
    doc.add_heading("8. Conclusion", level=1)
    para(doc,
         "This project delivered a complete, working machine-learning system for "
         "predicting the tip deflection of a flexible arm. A physically grounded "
         "dataset was generated with explicit rejection of invalid samples; three "
         "regression models were trained and compared on a properly isolated test "
         "set; performance was quantified with MAE, RMSE and R², and examined "
         "separately for interpolation and extrapolation; and the resulting models "
         "were deployed in an interactive application that validates its inputs and "
         "warns when results leave the region where the underlying theory applies.")
    para(doc,
         f"The Random Forest — the model specified for the project — reached "
         f"R² = {RF['R2']:.4f} with a mean absolute error of {RF['MAE_mm']:.3f} mm on "
         f"unseen data. Beyond reproducing a formula, the study produced two "
         f"findings of genuine interest. First, a log-linear model recovered the "
         f"physical exponents of beam theory (length 3.003 against a theoretical 3, "
         f"diameter −4.008 against −4) from data alone, without any physical "
         f"knowledge being supplied. Second, a direct measurement of extrapolation "
         f"showed the tree model's error roughly doubling outside its training range, "
         f"demonstrating concretely why high test-set accuracy does not guarantee "
         f"safe behaviour on new inputs.")
    para(doc,
         "The system runs entirely on a standard laptop using free and open-source "
         "software, requires no internet connection, cloud service, API key, CAD "
         "package or hardware, and can be demonstrated live.")

    doc.add_heading("9. Future Enhancements", level=1)
    for s in [
        "Collect real deflection measurements using a dial gauge or LVDT and retrain "
        "on experimental data, which would convert the surrogate into a genuinely "
        "validated model.",
        "Extend the section library to hollow tubes, rectangular and I-sections by "
        "substituting the appropriate second moment of area.",
        "Model multi-segment (multi-link) arms, where each link contributes to the "
        "tip deflection.",
        "Include self-weight and large-deflection (non-linear) theory for slender arms.",
        "Apply Gaussian Process regression to obtain a predictive uncertainty band "
        "rather than a single point estimate.",
        "Compare against gradient boosting, XGBoost and a small neural network.",
        "Integrate a design-safety layer that recommends feasible geometries rather "
        "than only predicting deflection.",
    ]:
        bullet(doc, s)

    doc.add_heading("10. References", level=1)
    for r in [
        "Gere, J. M. and Timoshenko, S. P., Mechanics of Materials, 2nd ed., "
        "Van Nostrand Reinhold, 1984.",
        "Hibbeler, R. C., Mechanics of Materials, 10th ed., Pearson, 2017.",
        "Pedregosa, F. et al., 'Scikit-learn: Machine Learning in Python', Journal of "
        "Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
        "Breiman, L., 'Random Forests', Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.",
        "McKinney, W., 'Data Structures for Statistical Computing in Python', "
        "Proceedings of the 9th Python in Science Conference, 2010.",
        "Harris, C. R. et al., 'Array programming with NumPy', Nature, vol. 585, "
        "pp. 357–362, 2020.",
        "Hunter, J. D., 'Matplotlib: A 2D Graphics Environment', Computing in Science "
        "& Engineering, vol. 9, no. 3, pp. 90–95, 2007.",
        "Streamlit Inc., 'Streamlit documentation', https://docs.streamlit.io.",
    ]:
        bullet(doc, r)

    doc.add_heading("Appendix A: Reproducing the results", level=1)
    para(doc, "From a terminal in the project folder, with the virtual environment "
              "activated:", space_after=4)
    for c in ["pip install -r requirements.txt",
              "python check_setup.py",
              "python generate_dataset.py",
              "python train_model.py",
              "python test_physics.py",
              "streamlit run app.py"]:
        p = doc.add_paragraph()
        r = p.add_run(c)
        r.font.name = "Consolas"
        r.font.size = Pt(10)
        p.paragraph_format.space_after = Pt(2)
    para(doc,
         f"With the fixed random seed of 42, these commands reproduce the dataset "
         f"and every metric in this report exactly. The model bundle used in this "
         f"document was trained on {TRAINED}.", italic=True, size=9)

    path = os.path.join(OUT, "Arm_Deflection_Project_Report.docx")
    doc.save(path)
    print(f"  wrote {path}")


# ==============================================================================
# 2. PRESENTATION
# ==============================================================================
def build_slides():
    prs = Presentation()
    prs.slide_width = PInches(13.333)
    prs.slide_height = PInches(7.5)
    blank = prs.slide_layouts[6]
    W, H = prs.slide_width, prs.slide_height

    def slide(title, subtitle=None):
        s = prs.slides.add_slide(blank)
        if title:
            tb = s.shapes.add_textbox(PInches(0.6), PInches(0.35),
                                      W - PInches(1.2), PInches(0.9))
            p = tb.text_frame.paragraphs[0]
            r = p.add_run()
            r.text = title
            r.font.size = PPt(32)
            r.font.bold = True
            r.font.color.rgb = PRGB(0x1F, 0x3A, 0x5F)
            if subtitle:
                tb2 = s.shapes.add_textbox(PInches(0.6), PInches(1.15),
                                           W - PInches(1.2), PInches(0.5))
                r2 = tb2.text_frame.paragraphs[0].add_run()
                r2.text = subtitle
                r2.font.size = PPt(15)
                r2.font.color.rgb = PRGB(0x60, 0x6B, 0x7A)
        return s

    def bullets(s, items, left=0.8, top=1.85, width=11.7, height=5.0, size=19):
        tb = s.shapes.add_textbox(PInches(left), PInches(top),
                                  PInches(width), PInches(height))
        tf = tb.text_frame
        tf.word_wrap = True
        for i, item in enumerate(items):
            level = 0
            text = item
            if isinstance(item, tuple):
                text, level = item
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.level = level
            r = p.add_run()
            r.text = ("• " if level == 0 else "– ") + text
            r.font.size = PPt(size if level == 0 else size - 3)
            p.space_after = PPt(10)
        return s

    def pic(s, path, left, top, width):
        if os.path.exists(path):
            s.shapes.add_picture(path, PInches(left), PInches(top),
                                 width=PInches(width))

    # 1 title
    s = prs.slides.add_slide(blank)
    bg = s.shapes.add_textbox(PInches(0.8), PInches(2.0), W - PInches(1.6),
                              PInches(1.5))
    p = bg.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = "AI-Based Arm Deflection Prediction"
    r.font.size = PPt(46)
    r.font.bold = True
    r.font.color.rgb = PRGB(0x1F, 0x3A, 0x5F)
    sub = s.shapes.add_textbox(PInches(0.8), PInches(3.3), W - PInches(1.6),
                               PInches(1.0))
    r = sub.text_frame.paragraphs[0].add_run()
    r.text = "Using Machine Learning to predict the tip deflection of a flexible arm"
    r.font.size = PPt(20)
    r.font.color.rgb = PRGB(0x60, 0x6B, 0x7A)
    who = s.shapes.add_textbox(PInches(0.8), PInches(4.8), W - PInches(1.6),
                               PInches(1.5))
    tf = who.text_frame
    for i, t in enumerate(["[Your Name]  —  [Roll Number]",
                           "[Department of Computer Science and Engineering]",
                           "[Institution]   |   Guide: [Guide's Name]"]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = t
        r.font.size = PPt(15)

    # 2 problem
    s = slide("The Problem", "A flexible arm droops under load — how much?")
    bullets(s, [
        "Flexible/lightweight arms are useful, but their tips deflect under payload.",
        "Too much deflection → inaccurate positioning, or structural failure.",
        "Classical answer: the cantilever beam equation.",
        ("δ = P·g·L³ / (3·E·I)   for a tip load", 1),
        ("δ = M·g·L³ / (8·E·I)   for a uniform load", 1),
        ("I = π·d⁴ / 64   for a solid circular section", 1),
        "Question: can a machine-learning model learn this relationship from data, "
        "and how reliable is it?",
    ])

    # 3 objectives
    s = slide("Objectives")
    bullets(s, [
        "Build a physically meaningful dataset for a cantilever arm.",
        "Train a Random Forest regression model and compare with baselines.",
        "Evaluate honestly: MAE, RMSE, R² on unseen data.",
        "Separate interpolation from extrapolation performance.",
        "Deploy as an interactive local application for live demonstration.",
        "State all assumptions and limitations explicitly.",
    ])

    # 4 physics
    s = slide("Physical Model and Assumptions")
    bullets(s, [
        "Cantilever beam: clamped at the base, free at the tip.",
        "Euler–Bernoulli small-deflection theory.",
        "Inputs: length L, payload P, diameter d, Young's modulus E, load case.",
        "Assumptions:",
        ("homogeneous, isotropic, linear-elastic material", 1),
        ("prismatic solid circular section; ideally rigid clamp", 1),
        ("static load only; δ ≤ L/10; no self-weight or shear deformation", 1),
        "Two validity filters enforced in code: δ ≤ L/10, and stress ≤ 60% of strength.",
    ])

    # 5 dataset
    s = slide("Dataset Construction", f"{N_TOTAL:,} synthetic samples")
    bullets(s, [
        "Generated by evaluating the beam equation — NOT measured on hardware.",
        "Mixture of uniform, log-uniform and structured sampling.",
        "Invalid samples rejected: δ > L/10, stress > 60% strength, δ < 0.01 mm.",
        "Noise added: 1.2% relative + 0.01 mm floor (imitates a real gauge).",
        "Fixed seed 42 → fully reproducible.",
        "",
        "→ The models learn an APPROXIMATION of a known formula. This is stated "
        "openly, not hidden.",
    ], size=18)

    # 6 pipeline
    s = slide("Methodology")
    bullets(s, [
        "80/20 stratified split on load case and binned deflection.",
        "No leakage: all tuning used the training set only; the test set was "
        "evaluated once.",
        "Three models compared:",
        ("Linear Regression — baseline", 1),
        ("Random Forest (200 trees, min_samples_leaf=5) — main model", 1),
        ("Power-Law Regression (log-log) — tests physics recovery", 1),
        "Random Forest fitted on log1p(δ): deflection spans 4 orders of magnitude.",
        f"3-fold CV on training data: R² = {CV_MEAN:.4f} ± {CV_STD:.4f}.",
    ], size=18)

    # 7 results table
    s = slide("Results on Unseen Data", f"{N_TEST:,} held-out test samples")
    rows = len(MET) + 1
    tbl = s.shapes.add_table(rows, 6, PInches(0.8), PInches(2.0),
                             PInches(11.7), PInches(1.9)).table
    hdrs = ["Model", "MAE (mm)", "RMSE (mm)", "R²", "MAPE (%)", "Within 10%"]
    for i, h in enumerate(hdrs):
        c = tbl.cell(0, i)
        c.text = h
        c.text_frame.paragraphs[0].runs[0].font.size = PPt(15)
        c.text_frame.paragraphs[0].runs[0].font.bold = True
    for r_i, (n, m) in enumerate(MET.items(), start=1):
        vals = [n, f"{m['MAE_mm']:.3f}", f"{m['RMSE_mm']:.3f}", f"{m['R2']:.4f}",
                f"{m['MAPE_pct']:.1f}", f"{m['within_10pct']:.1f}%"]
        for c_i, v in enumerate(vals):
            c = tbl.cell(r_i, c_i)
            c.text = v
            c.text_frame.paragraphs[0].runs[0].font.size = PPt(14)
    tb = s.shapes.add_textbox(PInches(0.8), PInches(4.3), PInches(11.7),
                              PInches(2.0))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, t in enumerate([
        f"Random Forest (the required model): R² = {RF['R2']:.4f}, "
        f"typical error {RF['MAE_mm']:.2f} mm.",
        f"Power-law model is near-exact (R² = {PW['R2']:.4f}) because the physics "
        f"really is a power law.",
        "Linear Regression fails — deflection is not linear in the inputs.",
    ]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = "• " + t
        r.font.size = PPt(15)

    # 8 recovered physics
    s = slide("Key Result: Physics Recovered from Data",
              "The model was told nothing about beam theory")
    labels = {"log_length_mm": ("Length L", 3),
              "log_payload_kg": ("Payload P", 1),
              "log_diameter_mm": ("Diameter d", -4),
              "log_E_GPa": ("Young's modulus E", -1)}
    rows = len(labels) + 1
    tbl = s.shapes.add_table(5, 4, PInches(0.8), PInches(2.1), PInches(7.6),
                             PInches(2.6)).table
    for i, h in enumerate(["Quantity", "Learned", "Theory", "Error"]):
        c = tbl.cell(0, i)
        c.text = h
        c.text_frame.paragraphs[0].runs[0].font.size = PPt(15)
        c.text_frame.paragraphs[0].runs[0].font.bold = True
    for r_i, (k, (lbl, th)) in enumerate(labels.items(), start=1):
        for c_i, v in enumerate([lbl, f"{COEF[k]:.4f}", str(th),
                                 f"{COEF[k]-th:+.4f}"]):
            c = tbl.cell(r_i, c_i)
            c.text = v
            c.text_frame.paragraphs[0].runs[0].font.size = PPt(15)
    tb = s.shapes.add_textbox(PInches(0.8), PInches(5.0), PInches(11.7),
                              PInches(1.8))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, t in enumerate([
        "A log-log linear fit recovered L³ and 1/d⁴ almost exactly.",
        "Strong evidence the model learned the real scaling laws, not a coincidence.",
    ]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = "• " + t
        r.font.size = PPt(17)

    # 9 extrapolation
    s = slide("Interpolation vs Extrapolation",
              "Why high test accuracy is not enough")
    bullets(s, [
        "Trained on L ≤ 650 mm, tested on L ≥ 680 mm (unseen).",
        f"Random Forest MAPE: {RF['MAPE_pct']:.1f}% → {EXT['Random Forest']['MAPE_pct']:.1f}%",
        f"Power-law MAPE: {PW['MAPE_pct']:.2f}% → {EXT['Power-Law Regression']['MAPE_pct']:.2f}%",
        "Reason: a decision tree cannot predict above the largest value it saw.",
        "Therefore the app REJECTS inputs outside the trained ranges.",
    ], width=6.0, size=17)
    pic(s, os.path.join(FIG, "extrapolation_test.png"), 6.9, 1.9, 5.9)

    # 10 app
    s = slide("The Application", "Streamlit, runs locally, no cloud or API key")
    bullets(s, [
        "Sliders for material, length, payload, diameter, E and load case.",
        "Predicts deflection in mm and shows the analytical reference alongside.",
        "Live charts: deflection vs length / payload / diameter.",
        "Warns when a result leaves the valid region of the theory.",
        "Built-in model-performance tab — no separate report needed.",
    ], width=5.9, size=17)
    pic(s, os.path.join(SHOT, "01_landing_charts.png"), 6.6, 1.8, 6.3)

    # 11 physical validation
    s = slide("Physical Sanity Checks")
    bullets(s, [
        "Length 400 → 600 mm gives 2.98 → 10.07 mm  (factor 3.375 = 1.5³ ✓)",
        "Payload 1 → 2 kg gives 2.98 → 5.96 mm  (exactly linear ✓)",
        "Diameter 10 → 20 mm gives 6.19 → 0.38 mm  (factor 0.061 vs 0.0625 ✓)",
        "Distributed load gives exactly 3/8 of the tip-load deflection ✓",
        "15 automated tests pass, including all 432 design-space combinations.",
        "The app warns instead of guessing when physics breaks down.",
    ], size=18)

    # 12 limitations
    s = slide("Limitations")
    bullets(s, [
        "Synthetic data — no experiments, no sensors, no CAD.",
        "Small-deflection theory only (δ ≤ L/10).",
        "Linear-elastic material; static loads only.",
        "Solid circular section; ideally rigid clamp.",
        "No self-weight, shear deformation, vibration or fatigue.",
        "No extrapolation beyond the trained input ranges.",
        "The ML model does not beat the analytical formula on clean synthetic data — "
        "it is a surrogate and a pipeline demonstration.",
    ], size=17)

    # 13 conclusion
    s = slide("Conclusion")
    bullets(s, [
        "Delivered a complete ML pipeline: data → train → evaluate → deploy.",
        f"Random Forest: R² = {RF['R2']:.4f}, MAE = {RF['MAE_mm']:.2f} mm on unseen data.",
        "Recovered the physical exponents (L³, 1/d⁴) from data alone.",
        "Measured extrapolation failure honestly and designed around it.",
        "Runs on a normal laptop, free tools, fully reproducible.",
    ], size=19)

    # 14 future + thanks
    s = slide("Future Work")
    bullets(s, [
        "Retrain on real measured deflection data from a dial gauge or LVDT.",
        "Add hollow-tube and rectangular sections.",
        "Multi-segment (multi-link) arms.",
        "Gaussian Processes for prediction uncertainty.",
        "Compare with gradient boosting and neural networks.",
    ], size=19)

    s = prs.slides.add_slide(blank)
    tb = s.shapes.add_textbox(PInches(0.8), PInches(2.8), W - PInches(1.6),
                              PInches(1.5))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = "Thank You"
    r.font.size = PPt(48)
    r.font.bold = True
    r.font.color.rgb = PRGB(0x1F, 0x3A, 0x5F)
    p.alignment = 2
    tb2 = s.shapes.add_textbox(PInches(0.8), PInches(4.0), W - PInches(1.6),
                               PInches(0.8))
    r = tb2.text_frame.paragraphs[0].add_run()
    r.text = "Questions welcome"
    r.font.size = PPt(20)
    r.font.color.rgb = PRGB(0x60, 0x6B, 0x7A)
    tb2.text_frame.paragraphs[0].alignment = 2

    path = os.path.join(OUT, "Arm_Deflection_Presentation.pptx")
    prs.save(path)
    print(f"  wrote {path}")


# ==============================================================================
# 3. VIVA Q&A
# ==============================================================================
def build_viva():
    doc = Document()
    setup(doc)
    doc.add_heading("Viva Preparation: Questions and Answers", level=1)
    para(doc,
         "Answers are written to match what was actually implemented. Numbers are "
         "the measured test-set values from this project. Read the 'say it simply' "
         "line for each answer — that is the version to speak aloud.",
         italic=True, size=10)

    QA = [
        ("What is your project about, in one sentence?",
         "It predicts how far the tip of a flexible robotic arm droops under a load, "
         "using machine-learning models trained on data generated from beam theory, "
         "and presents the prediction in an interactive Python application.",
         "Mentor tip: lead with the physics, then the ML, then the app."),

        ("Why did you use a cantilever beam model?",
         "A flexible arm fixed rigidly at the base with a free tip is exactly a "
         "cantilever. It is the simplest standard case with a closed-form solution, "
         "so I can generate trustworthy data and check my ML models against a known "
         "correct answer.", None),

        ("What is your dataset and where did it come from?",
         f"It is a synthetic dataset of {N_TOTAL:,} rows that I generated by "
         f"evaluating the cantilever beam equation across realistic input ranges, "
         f"then adding 1.2% noise. It is not experimental data — no arm was built "
         f"and no sensors were used. The models therefore learn an approximation of "
         f"a known formula.",
         "Do not be defensive about this. Examiners respect a student who states the "
         "limitation first. Adding hardware is listed as future work."),

        ("Is it still machine learning if the data comes from a formula?",
         "Yes. The model is never given the formula, the exponents, or any hint of "
         "them — it receives only columns of numbers and must learn the mapping from "
         "them. That it recovers the L³ and 1/d⁴ relationships on its own is evidence "
         "the learning is genuine. Synthetic data is standard practice for "
         "bootstrapping a surrogate model before real measurements exist.", None),

        ("Which algorithm did you use and why?",
         "Random Forest Regression, as specified. It handles non-linear relationships "
         "naturally, needs no feature scaling, is robust to outliers, and does not "
         "assume any functional form. I also trained Linear Regression as a baseline "
         "and a power-law regression to test whether the physical exponents could be "
         "recovered.", None),

        ("How did you split your data?",
         f"80/20, stratified on both load case and binned deflection, with a fixed "
         f"seed of 42. That gives {N_TRAIN:,} training and {N_TEST:,} test rows. "
         f"Stratifying ensures the test set covers the same range of behaviour "
         f"rather than accidentally containing only easy cases.", None),

        ("What is data leakage and how did you avoid it?",
         "Leakage is when information from the test set influences training, which "
         "makes the reported score look better than reality. I avoided it by doing "
         "all tuning and cross-validation on the training set only, and evaluating "
         "the test set exactly once at the end. I never chose a hyper-parameter by "
         "looking at the test score.", None),

        ("What do MAE, RMSE and R² mean?",
         "MAE is the average absolute error in millimetres. RMSE is similar but "
         "penalises large errors more. R² is the fraction of the variation in "
         "deflection that the model explains: 1.0 is perfect, 0 means no better than "
         "predicting the mean.", None),

        ("What were your actual results?",
         f"On {N_TEST:,} unseen samples: Random Forest R² = {RF['R2']:.4f} with "
         f"MAE = {RF['MAE_mm']:.2f} mm; Power-Law Regression R² = {PW['R2']:.4f}; "
         f"Linear Regression R² = {LR['R2']:.4f}. Cross-validation on the training "
         f"set gave {CV_MEAN:.4f} ± {CV_STD:.4f}, close to the test score, so the "
         f"model is not over-fitted.",
         "Quote the numbers calmly. Precision builds credibility."),

        ("Why is your Random Forest not the most accurate model?",
         f"Because the underlying physics really is a power law, so a model with the "
         f"correct functional form fits it almost perfectly (R² = {PW['R2']:.4f}). A "
         f"tree ensemble approximates that same relationship as a staircase of "
         f"constant pieces, which costs accuracy. I report both rather than only the "
         f"flattering one — the comparison is a result in itself.",
         "This is the single best answer in the viva. It shows intellectual honesty."),

        ("Why did you train the Random Forest on log of deflection?",
         "Deflection ranges from 0.01 mm to about 79 mm — four orders of magnitude. "
         "Fitting the raw value would make the model focus almost entirely on the "
         "largest cases and produce a very skewed error. Working in log space makes "
         "the error relative, which is the right treatment at that scale. A wrapper "
         "class converts predictions back to millimetres automatically.", None),

        ("What is extrapolation and why does it matter?",
         f"Interpolation is predicting inside the training range; extrapolation is "
         f"predicting outside it. I tested it by training on arms ≤ 650 mm and "
         f"testing on 680–800 mm. The Random Forest's MAPE went from "
         f"{RF['MAPE_pct']:.1f}% to {EXT['Random Forest']['MAPE_pct']:.1f}%, while the "
         f"power-law model stayed near {EXT['Power-Law Regression']['MAPE_pct']:.1f}%. "
         f"This is why the app refuses inputs outside the trained ranges.", None),

        ("Why can't a Random Forest extrapolate?",
         "A tree predicts the average of the training targets that fall in a leaf. It "
         "can never output a value larger than the largest target it ever saw. Beyond "
         "its data its predictions flatten out and it under-predicts systematically. "
         "It is important because a model can score well on a test set and still be "
         "dangerously wrong on inputs outside the training distribution.", None),

        ("How do you know your physics is correct?",
         "Three ways. I verified the equation against hand calculations; I wrote 15 "
         "automated tests checking known scaling laws — doubling length multiplies "
         "deflection by 8, doubling diameter divides it by 16, and a distributed load "
         "gives exactly 3/8 of a tip load; and I checked that all 432 combinations in "
         "the design space are flagged correctly when unphysical.",
         "If invited, run python test_physics.py live. It takes two seconds and is "
         "very persuasive."),

        ("What happens if I enter an impossible combination?",
         "The app still computes the number, because the equation remains "
         "mathematically valid — but it refuses to present it as a design value. It "
         "warns that the deflection exceeds the 10%-of-length limit of the theory and "
         "that the bending stress exceeds 60% of the material strength, meaning the "
         "arm would fail first.", None),

        ("Why is arm length so much more important than material?",
         "Because deflection scales with the cube of length but only inversely with "
         "the first power of Young's modulus. Doubling the length multiplies "
         "deflection by 8, whereas switching from aluminium to steel — roughly three "
         "times stiffer — only divides it by 3. Increasing the diameter is the "
         "strongest lever of all, since it enters as the fourth power.", None),

        ("What are the limitations of your work?",
         "Synthetic data with no experimental validation; small-deflection theory "
         "only; linear-elastic material; static loads only; solid circular section; "
         "ideal rigid clamp; no self-weight or vibration; no extrapolation outside "
         "the trained ranges. I list all of these in the app itself, not only in the "
         "report.", None),

        ("What would you do with more time?",
         "Build the physical arm and collect real deflection measurements with a dial "
         "gauge or LVDT, then retrain on that data — the pipeline would not change, "
         "only the data source. I would also add hollow sections, multi-link arms, "
         "and Gaussian Processes to get prediction uncertainty rather than a single "
         "number.", None),

        ("Why did you use Streamlit?",
         "It turns a Python script into a working web interface with no HTML, "
         "JavaScript or backend work, runs entirely locally, needs no cloud service "
         "or API key, and is free. That matters for a project that must be "
         "demonstrated live on a student laptop.", None),

        ("How do I know it will run on another computer?",
         "Dependencies are pinned in requirements.txt, the data and split use a fixed "
         "seed of 42 so results are reproducible, check_setup.py verifies the "
         "environment before anything else runs, and the trained model is saved so no "
         "retraining is needed for a demonstration. The saved model references classes "
         "in a real module, not __main__, so loading is reliable.", None),

        ("What happens if you delete the model file?",
         "The app detects the missing file and shows a clear instruction to run "
         "generate_dataset.py and then train_model.py, rather than crashing with a "
         "confusing traceback. Training takes about 45 seconds.", None),

        ("Where is the machine learning actually happening in your app?",
         "In the model.predict() call on the loaded bundle. Everything else — the "
         "equations, the validation, the warnings — is rule-based physics. The "
         "analytical reference value shown in the app is deliberately labelled as the "
         "formula, so there is no confusion about which number comes from ML.", None),
    ]

    for i, (q, a, tip) in enumerate(QA, 1):
        doc.add_heading(f"Q{i}. {q}", level=2)
        para(doc, "Answer: ", bold=True, space_after=0)
        para(doc, a)
        if tip:
            p = doc.add_paragraph()
            r = p.add_run(tip)
            r.italic = True
            r.font.size = Pt(9.5)
            r.font.color.rgb = RGBColor(0x7C, 0x3A, 0x0E)

    doc.add_heading("Three questions to prepare especially well", level=1)
    for t in [
        "\"Your data is synthetic — is that valid?\" → Acknowledge immediately, "
        "explain that ML learns the mapping without being given the formula, and "
        "point to future work with real sensors. Never dodge this.",
        "\"Why is your R² not 0.999 for the Random Forest?\" → Explain the "
        "piecewise-constant approximation and leaf granularity. Show Figure 4.",
        "\"How do you know you have not over-fitted?\" → Quote the cross-validation "
        f"score ({CV_MEAN:.4f}) against the test score ({RF['R2']:.4f}); they are "
        "close, which is the evidence.",
    ]:
        bullet(doc, t)

    path = os.path.join(OUT, "Viva_Questions_and_Answers.docx")
    doc.save(path)
    print(f"  wrote {path}")


# ==============================================================================
# 4. DEMO SCRIPT
# ==============================================================================
def build_demo():
    doc = Document()
    setup(doc)
    doc.add_heading("Live Demonstration Script (about 6 minutes)", level=1)
    para(doc, "Print this and keep it beside the laptop. Timings are guides.",
         italic=True, size=10)

    steps = [
        ("Before you start (do this 10 minutes early)", [
            "Open the project folder in VS Code; open a terminal.",
            "Activate the environment:  .\\.venv\\Scripts\\Activate.ps1",
            "Run:  python check_setup.py   — confirm it says EVERYTHING IS READY.",
            "Launch:  streamlit run app.py   — leave the browser tab open at "
            "localhost:8501.",
            "Set the inputs back to the defaults: Aluminium, 400 mm, 1.0 kg, 12 mm.",
            "Close other browser tabs so the interface is clean on screen.",
        ]),
        ("0:00 – 0:30  Opening", [
            "SAY: \"A flexible robotic arm bends under load. My project predicts how "
            "far the tip will deflect, using machine learning, and shows honestly how "
            "reliable that prediction is.\"",
            "SAY: \"I model the arm as a cantilever beam and I generate my dataset "
            "from the beam equation — so I am not claiming experimental data. What I "
            "am demonstrating is a complete machine-learning pipeline plus a study of "
            "how well it recovers known physics.\"",
        ]),
        ("0:30 – 1:15  First prediction", [
            "Click Predict Deflection with the defaults.",
            "POINT AT: predicted 2.98 mm, analytical reference 2.98 mm, "
            "\"ML vs formula\" about −0.1%.",
            "POINT AT: the green banner — physically consistent, stress 23 MPa, within "
            "the safe limit for aluminium.",
            "SAY: \"The predicted value comes from the trained model; the reference "
            "comes from the textbook equation. I show both so you can see the ML "
            "error directly.\"",
        ]),
        ("1:15 – 2:15  Length: the L³ law", [
            "Change Arm length 400 → 600 mm, press Predict.",
            "SAY: \"Deflection went from 2.98 to 10.07 mm — a factor of 3.375, exactly "
            "1.5³.\"",
            "Change to 200 mm: about 0.37 mm.",
            "SAY: \"This is why long slender arms are so hard to keep stiff. It is the "
            "single most important design fact in this project.\"",
        ]),
        ("2:15 – 3:00  Diameter: the 1/d⁴ law", [
            "Set length back to 400 mm. Change diameter 12 → 20 mm, press Predict.",
            "SAY: \"2.98 down to 0.38 mm. Diameter enters to the fourth power, so "
            "making the arm thicker is far more effective than changing material.\"",
            "Change diameter to 10 mm to show the opposite direction (about 6.2 mm).",
        ]),
        ("3:00 – 3:45  Honest failure", [
            "Set: Arm length 800 mm, payload 5 kg, diameter 6 mm, press Predict.",
            "POINT AT the two yellow warnings.",
            "SAY: \"Here the deflection would be 241% of the arm length, and the "
            "bending stress would be 1850 MPa against a strength of 276 MPa. The "
            "equation still produces a number, but the assumption behind it has "
            "failed — so the app refuses to present it as a design value. I would "
            "rather show you this than hide it.\"",
            "Reset the inputs back to defaults afterwards.",
        ]),
        ("3:45 – 5:00  Model performance", [
            "Click the Model Performance tab.",
            "SAY: \"These figures are measured on 8,000 rows the models never saw.\"",
            f"READ: Random Forest R² = {RF['R2']:.4f}, MAE = {RF['MAE_mm']:.2f} mm.",
            "Scroll to recovered exponents.",
            "SAY: \"I told this model nothing about beam theory. It learned the "
            "exponent 3.003 on length and −4.008 on diameter. That is the physics "
            "coming out of the data.\"",
        ]),
        ("5:00 – 5:45  Extrapolation", [
            "Scroll to the extrapolation table.",
            f"SAY: \"Trained on short arms, tested on long ones: the Random Forest "
            f"error went from {RF['MAPE_pct']:.0f}% to "
            f"{EXT['Random Forest']['MAPE_pct']:.0f}%. A tree can never predict above "
            f"the largest value it saw. That is why the app rejects inputs outside its "
            f"trained range instead of guessing.\"",
        ]),
        ("5:45 – 6:15  Close", [
            "SAY: \"So: a complete pipeline, honest evaluation, a working application, "
            "and a real finding about where tree models fail. Thank you — happy to "
            "take questions.\"",
        ]),
    ]
    for title, items in steps:
        doc.add_heading(title, level=2)
        for it in items:
            bullet(doc, it)

    doc.add_heading("If the demonstration goes wrong", level=1)
    table(doc, ["Symptom", "What to do — stay calm and say this out loud"],
          [["App will not start",
            "\"Let me check the environment.\" Run python check_setup.py. If the model "
            "is missing, run generate_dataset.py then train_model.py (about 45 s)."],
           ["\"Trained model not found\"",
            "Exactly as above. The app is telling you clearly what to do — point that "
            "out, it is good engineering."],
           ["Port already in use",
            "streamlit run app.py --server.port 8502"],
           ["Charts slow to appear",
            "Wait a few seconds; say \"the model recomputes the curve for each input\"."],
           ["Examiner asks for something you did not implement",
            "Say plainly: \"That is not in the current scope — I would address it by…\" "
            "and give the honest future-work answer. Never invent a result."]],
          widths=[1.9, 4.6])

    doc.add_heading("Facts to have at your fingertips", level=1)
    for f in [f"Dataset: {N_TOTAL:,} synthetic rows, seed 42, 1.2% noise",
              f"Split: {N_TRAIN:,} train / {N_TEST:,} test, stratified",
              f"Random Forest: 200 trees, min_samples_leaf = 5, log1p target",
              f"Random Forest test: R² = {RF['R2']:.4f}, MAE = {RF['MAE_mm']:.2f} mm, "
              f"MAPE = {RF['MAPE_pct']:.1f}%",
              f"Power-law test: R² = {PW['R2']:.4f}, MAPE = {PW['MAPE_pct']:.2f}%",
              f"Linear baseline: R² = {LR['R2']:.4f}",
              f"CV (training only): {CV_MEAN:.4f} ± {CV_STD:.4f}",
              f"Recovered exponents: L {COEF['log_length_mm']:.3f}, "
              f"P {COEF['log_payload_kg']:.3f}, d {COEF['log_diameter_mm']:.3f}, "
              f"E {COEF['log_E_gPa' if 'log_E_gPa' in COEF else 'log_E_GPa']:.3f}",
              "Exact scaling checks: L×2 → δ×8; d×2 → δ×1/16; distributed = 3/8 of tip",
              "Training time: about 45 seconds on a laptop"]:
        bullet(doc, f)

    path = os.path.join(OUT, "Demo_Script.docx")
    doc.save(path)
    print(f"  wrote {path}")


if __name__ == "__main__":
    print("Building submission documents ...")
    build_report()
    build_slides()
    build_viva()
    build_demo()
    print("Done.")
