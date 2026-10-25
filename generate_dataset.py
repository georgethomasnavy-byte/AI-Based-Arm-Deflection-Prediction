"""
generate_dataset.py
===================
STEP 1 of the pipeline:  create the dataset that the ML model will learn from.

IMPORTANT AND HONEST STATEMENT ABOUT THE DATA
---------------------------------------------
This dataset is SYNTHETIC.  It is not measured on a real arm with real sensors.
Each row is produced by evaluating the closed-form cantilever-beam equation from
physics.py, then adding a small amount of controlled measurement noise so the
data behaves like imperfect real-world readings.

Consequence you must state in your report and viva:
    the ML model learns an APPROXIMATION of a known engineering formula.
It is not discovering new physics, and it must not be described as trained on
experimental data.

WHY USE SYNTHETIC DATA AT ALL?
------------------------------
1. Real deflection data for flexible arms is not freely available and we have no
   test rig.
2. Because we know the true answer, we can measure the model's error exactly and
   study HOW it fails (interpolation vs extrapolation).  With real data we could
   never separate model error from noise.
3. The pipeline built here is exactly the pipeline you would use on real sensor
   data later: only the data source would change.

WHAT ONE ROW MEANS
------------------
    length_mm        arm length                            (mm)
    payload_kg       tip point load OR total distributed mass (kg)
    diameter_mm      solid circular cross-section diameter   (mm)
    E_GPa            Young's modulus of the material         (GPa)
    load_case        0 = tip point load, 1 = uniformly distributed
    deflection_mm    TIP DEFLECTION -- the target the model predicts (mm)

HOW SAMPLING WORKS
------------------
Rather than totally random numbers, we use a mixture of three strategies so that
the dataset is realistic and not clustered in one corner of the input space:

  35%  near-feasible random  : wide random values, accepted only if physically valid
  40%  log-uniform random    : spreads values across orders of magnitude
  25%  structured grid       : one material / load case at a time, sweeping L and P

Any candidate sample that breaks the small-deflection limit (delta > L/10) or
that would exceed 60% of the material's strength is REJECTED.  This keeps every
row physically meaningful and, just as importantly, stops the model from being
trained on nonsense.

Usage
-----
    python generate_dataset.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from physics import (
    G,
    INPUT_RANGES,
    LOAD_CASE_DISTRIBUTED,
    LOAD_CASE_NAMES,
    LOAD_CASE_TIP,
    MATERIALS,
    moment_of_inertia_circular,
    tip_deflection_mm,
)

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
RANDOM_SEED = 42          # fixed seed => the dataset is reproducible every run
N_TARGET = 40000          # number of accepted rows we aim for
                          # (40k rows: chosen by experiment -- see README section
                          #  'How many rows do we need?'.  Training takes ~1 min.)

# Measurement-noise model.  Real instrumentation has error, so we simulate it.
#   relative noise : proportional to the deflection (e.g. 1.2% of the reading)
#   absolute noise : a fixed floor (e.g. 0.01 mm) that dominates tiny deflections
NOISE_RELATIVE = 0.012
NOISE_ABSOLUTE_MM = 0.010

# Useful measurement window.  A deflection smaller than this is below the
# resolution of any practical dial gauge / LVDT on a student budget, so we do
# not generate those rows: they would be pure noise and would make relative
# errors explode during evaluation.
MIN_DEFLECTION_MM = 0.01

DATA_PATH = "data/beam_deflection_dataset.csv"


def _accept(L: float, P: float, d: float, E: float, case: int, strength: float) -> bool:
    """Physical feasibility filter (see physics.MAX_DEFLECTION_RATIO / MAX_STRESS_SAFETY).

    A sample is kept only if ALL of the following hold:
      * deflection is inside the measurable window (>= MIN_DEFLECTION_MM)
      * deflection stays below L/10  (small-deflection theory valid)
      * root bending stress stays below 60% of the material strength
    """
    delta = tip_deflection_mm(L, P, E, d, case)
    if delta < MIN_DEFLECTION_MM:
        return False
    if delta / L > 0.10:                      # small-deflection theory broken
        return False
    # root bending stress check
    I = moment_of_inertia_circular(d)
    c = (d / 1000.0) / 2.0
    W = P * G
    M_b = W * (L / 1000.0) * (1.0 if case == LOAD_CASE_TIP else 0.5)
    stress_MPa = (M_b * c / I) / 1e6
    if stress_MPa > 0.60 * strength:          # would yield in reality
        return False
    return True


def generate(n_target: int = N_TARGET, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build the synthetic dataset and return it as a pandas DataFrame."""
    rng = np.random.default_rng(seed)

    materials = list(MATERIALS.keys())
    rows: list[dict] = []

    lo_L, hi_L = INPUT_RANGES["length_mm"]
    lo_P, hi_P = INPUT_RANGES["payload_kg"]
    lo_d, hi_d = INPUT_RANGES["diameter_mm"]

    attempts = 0
    max_attempts = n_target * 400          # safety valve so we can never loop forever

    while len(rows) < n_target and attempts < max_attempts:
        attempts += 1
        material = materials[rng.integers(len(materials))]
        E = MATERIALS[material]["E_GPa"]
        strength = MATERIALS[material]["strength_MPa"]
        case = int(rng.integers(2))
        u = rng.random()

        if u < 0.35:
            # --- strategy 1: near-feasible random values -----------------------
            L = rng.uniform(lo_L, hi_L)
            d = rng.uniform(lo_d, hi_d)
            P = rng.uniform(lo_P, hi_P)

        elif u < 0.75:
            # --- strategy 2: log-uniform, spreads out magnitudes --------------
            L = float(np.exp(rng.uniform(np.log(lo_L), np.log(hi_L))))
            d = float(np.exp(rng.uniform(np.log(lo_d), np.log(hi_d))))
            P = float(np.exp(rng.uniform(np.log(lo_P), np.log(hi_P))))

        else:
            # --- strategy 3: structured sweep for one material / load case ----
            L = rng.uniform(lo_L, hi_L)
            d = rng.uniform(lo_d, hi_d)
            # load chosen relative to what this beam can safely carry, so the
            # sweep produces a smooth spread of feasible deflections
            I = moment_of_inertia_circular(d)
            capacity_N = 0.35 * strength * 1e6 * I / ((L / 1000.0) * (d / 2000.0))
            if case == LOAD_CASE_DISTRIBUTED:
                capacity_N *= 2.0
            P_max = min(hi_P, max(lo_P * 1.5, capacity_N / G))
            P = rng.uniform(lo_P, P_max) if P_max > lo_P else rng.uniform(lo_P, hi_P)

        if not _accept(L, P, d, E, case, strength):
            continue

        true_delta = tip_deflection_mm(L, P, E, d, case)

        # add instrument-like noise, then never allow a non-physical value
        noise = rng.normal(0.0, 1.0) * np.sqrt(
            (NOISE_RELATIVE * true_delta) ** 2 + NOISE_ABSOLUTE_MM ** 2
        )
        measured = max(true_delta + noise, NOISE_ABSOLUTE_MM)

        # The filter above was applied to the noise-free value.  Noise could still
        # push a sample just past the L/10 limit, so re-check after corruption.
        if measured / L > 0.10:
            continue

        rows.append(
            {
                "length_mm": round(L, 2),
                "payload_kg": round(P, 4),
                "diameter_mm": round(d, 2),
                "E_GPa": E,
                "load_case": case,
                "deflection_mm": round(measured, 5),
            }
        )

    df = pd.DataFrame(rows)
    return df


def summarise(df: pd.DataFrame) -> None:
    """Print a readable report about the generated dataset."""
    print("=" * 68)
    print("DATASET SUMMARY")
    print("=" * 68)
    print(f"rows                : {len(df)}")
    print(f"columns             : {list(df.columns)}")
    print(f"missing values      : {int(df.isna().sum().sum())}")
    print(f"duplicate rows      : {int(df.duplicated().sum())}")
    print()
    print("Feature ranges actually covered:")
    print(df[["length_mm", "payload_kg", "diameter_mm", "E_GPa", "deflection_mm"]]
          .describe().T.to_string(float_format=lambda v: f"{v:10.4f}"))
    print()
    counts = df["load_case"].value_counts().sort_index()
    for case, n in counts.items():
        print(f"  {LOAD_CASE_NAMES[case]:<28}: {n:5d} rows ({n/len(df)*100:4.1f}%)")
    print()
    print("Sanity checks:")
    print(f"  any deflection <= 0        : {bool((df['deflection_mm'] <= 0).any())}")
    print(f"  any length <= 0            : {bool((df['length_mm'] <= 0).any())}")
    print(f"  max deflection / length    : {(df['deflection_mm']/df['length_mm']).max():.4f}"
          f"   (limit 0.10 for small-deflection theory)")
    print(f"  min deflection (mm)        : {df['deflection_mm'].min():.5f}")


def main() -> None:
    print("Generating synthetic cantilever-beam dataset ...")
    df = generate()
    summarise(df)
    df.to_csv(DATA_PATH, index=False)
    print()
    print(f"Saved {len(df)} rows to {DATA_PATH}")


if __name__ == "__main__":
    main()
