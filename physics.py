"""
physics.py
==========
Single source of truth for the cantilever-beam physics used in this project.

Every other script (generate_dataset.py, train_model.py, app.py) imports from
this file.  That guarantees the dataset generator, the trained model and the
user interface all use exactly the same equations, units and limits.

--------------------------------------------------------------------------------
PHYSICAL MODEL
--------------------------------------------------------------------------------
We model the flexible arm as an idealised CANTILEVER BEAM:
  - one end is rigidly clamped (the shoulder / base mount),
  - the other end is completely free (the gripper / tool end),
  - the beam is straight, horizontal, prismatic (same cross-section all along),
  - the material is homogeneous, isotropic and linear-elastic (obeys Hooke's law),
  - deflection is small compared with the length ("small-deflection" /
    Euler-Bernoulli beam theory),
  - shear deformation, self-weight vibration, buckling and large-angle
    geometry are all neglected.

We support TWO load cases:

  (1) TIP POINT LOAD, magnitude P (kg hung at the free end)
          delta = P * g * L^3 / (3 * E * I)

  (2) UNIFORMLY DISTRIBUTED LOAD, total mass M (kg) spread evenly along the arm
          delta = M * g * L^3 / (8 * E * I)

  Note that both grow with L cubed.  For the SAME total mass, a distributed
  load causes 3/8 of the deflection of an end load, because its resultant
  acts closer to the support.  That factor 1/3 vs 1/8 is a genuine physical
  result and a good viva question.

  SECOND MOMENT OF AREA for a solid circular cross-section of diameter d:
          I = pi * d^4 / 64

--------------------------------------------------------------------------------
UNITS
--------------------------------------------------------------------------------
Public API uses human-friendly engineering units:
    length        L : millimetres   (mm)
    load          P : kilograms     (kg)
    diameter      d : millimetres   (mm)
    Young modulus E : gigapascals   (GPa)
    deflection    delta : millimetres (mm)

Internally everything is converted to SI (m, kg, N, Pa, m^4) before the
formulas are evaluated, then converted back to mm.  Mixing units is one of the
most common beginner errors in engineering code, so we centralise it here.
"""

from __future__ import annotations

import math

# Standard gravitational acceleration (m/s^2).  Value used throughout.
G = 9.80665

# ------------------------------------------------------------------------------
# MATERIAL DATABASE
# ------------------------------------------------------------------------------
# E_GPa      : Young's modulus (stiffness of the material), gigapascals
# strength_MPa: indicative yield strength (or tensile strength for the
#              composite, which does not yield).  Used ONLY for a plausibility
#              warning, not for the deflection calculation.
# density    : kg/m^3, provided for reference / possible future work.
#
# These are typical handbook values for the named grades.  Real parts vary with
# heat treatment, moisture, print orientation, fibre lay-up, etc.  They are
# representative values, not measurements made by us.
# ------------------------------------------------------------------------------
MATERIALS: dict[str, dict[str, float]] = {
    "Aluminium 6061-T6":      {"E_GPa": 68.9, "strength_MPa": 276.0, "density": 2700.0},
    "Mild Steel (AISI 1018)": {"E_GPa": 200.0, "strength_MPa": 370.0, "density": 7870.0},
    "Stainless Steel 304":    {"E_GPa": 193.0, "strength_MPa": 215.0, "density": 8000.0},
    "Brass (C260)":           {"E_GPa": 100.0, "strength_MPa": 200.0, "density": 8530.0},
    "Carbon Fibre (UD epoxy)": {"E_GPa": 135.0, "strength_MPa": 600.0, "density": 1600.0},
    "PLA (3D printed)":       {"E_GPa": 3.5,  "strength_MPa": 50.0,  "density": 1240.0},
    "Acrylic (PMMA)":         {"E_GPa": 3.1,  "strength_MPa": 65.0,  "density": 1180.0},
    "Nylon 6/6":              {"E_GPa": 2.8,  "strength_MPa": 80.0,  "density": 1140.0},
}

# ------------------------------------------------------------------------------
# SUPPORTED INPUT RANGES
# ------------------------------------------------------------------------------
# The ML model is trained ONLY on data inside these ranges.  Predictions outside
# them are extrapolation and are not trustworthy (especially for tree models).
# ------------------------------------------------------------------------------
INPUT_RANGES: dict[str, tuple[float, float]] = {
    "length_mm":      (150.0, 800.0),
    "payload_kg":     (0.05, 5.0),
    "diameter_mm":    (6.0, 40.0),
    "E_GPa":          (2.8, 200.0),
}

# Load-case identifiers used consistently as the model feature
LOAD_CASE_TIP = 0
LOAD_CASE_DISTRIBUTED = 1
LOAD_CASE_NAMES = {
    LOAD_CASE_TIP: "Tip point load",
    LOAD_CASE_DISTRIBUTED: "Uniformly distributed load",
}

# Feasibility criteria used to reject physically meaningless samples.
MAX_DEFLECTION_RATIO = 0.10   # delta must stay below L/10 for small-deflection theory
MAX_STRESS_SAFETY = 0.60      # bending stress must stay below 0.6 * strength


# ------------------------------------------------------------------------------
# GEOMETRY
# ------------------------------------------------------------------------------
def moment_of_inertia_circular(diameter_mm: float) -> float:
    """Second moment of area (m^4) of a SOLID CIRCULAR section.

    I = pi * d^4 / 64      with d in metres.
    """
    d_m = diameter_mm / 1000.0
    return math.pi * d_m ** 4 / 64.0


# ------------------------------------------------------------------------------
# DEFLECTION MODELS (closed-form / analytical reference values)
# ------------------------------------------------------------------------------
def tip_deflection_mm(
    length_mm: float,
    load_kg: float,
    E_GPa: float,
    diameter_mm: float,
    load_case: int = LOAD_CASE_TIP,
) -> float:
    """Analytical tip deflection of a cantilever, returned in millimetres.

    Parameters
    ----------
    length_mm   : arm length, mm
    load_kg     : tip point load (kg) OR total distributed mass (kg),
                  depending on `load_case`
    E_GPa       : Young's modulus, GPa
    diameter_mm : solid circular diameter, mm
    load_case   : LOAD_CASE_TIP (0) or LOAD_CASE_DISTRIBUTED (1)

    Returns
    -------
    float : tip deflection in mm (positive = drooping downwards)
    """
    L = length_mm / 1000.0          # m
    E = E_GPa * 1e9                 # Pa
    I = moment_of_inertia_circular(diameter_mm)   # m^4
    W = load_kg * G                 # N

    if load_case == LOAD_CASE_TIP:
        delta_m = W * L ** 3 / (3.0 * E * I)
    elif load_case == LOAD_CASE_DISTRIBUTED:
        delta_m = W * L ** 3 / (8.0 * E * I)
    else:
        raise ValueError(f"Unknown load_case: {load_case!r}")

    return delta_m * 1000.0         # mm


def max_bending_stress_MPa(
    length_mm: float,
    load_kg: float,
    diameter_mm: float,
    load_case: int = LOAD_CASE_TIP,
) -> float:
    """Maximum bending stress at the clamped root, in MPa.

    Tip point load        : M_b = W * L
    Uniformly distributed : M_b = W * L / 2
    Bending stress        : sigma = M_b * c / I   with c = d/2
    """
    L = length_mm / 1000.0
    I = moment_of_inertia_circular(diameter_mm)
    c = (diameter_mm / 1000.0) / 2.0
    W = load_kg * G

    if load_case == LOAD_CASE_TIP:
        M_b = W * L
    elif load_case == LOAD_CASE_DISTRIBUTED:
        M_b = W * L / 2.0
    else:
        raise ValueError(f"Unknown load_case: {load_case!r}")

    return (M_b * c / I) / 1e6       # Pa -> MPa


# ------------------------------------------------------------------------------
# VALIDATION HELPERS
# ------------------------------------------------------------------------------
def check_inputs_in_range(
    length_mm: float,
    load_kg: float,
    E_GPa: float,
    diameter_mm: float,
) -> list[str]:
    """Return a list of human-readable problems with the inputs.

    An empty list means all inputs lie inside the supported (trained) ranges.
    """
    problems: list[str] = []
    values = {
        "Arm length": (length_mm, "length_mm", "mm"),
        "Load": (load_kg, "payload_kg", "kg"),
        "Diameter": (diameter_mm, "diameter_mm", "mm"),
        "Young's modulus": (E_GPa, "E_GPa", "GPa"),
    }
    for label, (value, key, unit) in values.items():
        lo, hi = INPUT_RANGES[key]
        if value < lo or value > hi:
            problems.append(
                f"{label} = {value:g} {unit} is outside the trained range "
                f"({lo:g} to {hi:g} {unit}). The prediction is an extrapolation "
                f"and may be unreliable."
            )
    return problems


def assess_plausibility(
    length_mm: float,
    load_kg: float,
    diameter_mm: float,
    load_case: int,
    material_name: str | None = None,
    deflection_mm: float | None = None,
) -> dict:
    """Check whether a prediction is physically meaningful.

    Returns a dict with:
        deflection_ratio : delta / L
        stress_MPa       : maximum bending stress
        strength_MPa     : material strength (or None)
        warnings         : list[str]
        small_deflection_ok : bool
        stress_ok        : bool
    """
    if deflection_mm is None:
        raise ValueError("deflection_mm is required")

    stress = max_bending_stress_MPa(length_mm, load_kg, diameter_mm, load_case)
    ratio = deflection_mm / length_mm

    strength = None
    if material_name and material_name in MATERIALS:
        strength = MATERIALS[material_name]["strength_MPa"]

    warnings: list[str] = []
    small_deflection_ok = ratio <= MAX_DEFLECTION_RATIO
    if not small_deflection_ok:
        warnings.append(
            f"Deflection is {ratio*100:.0f}% of the arm length "
            f"({deflection_mm:.1f} mm on {length_mm:.0f} mm). Above about 10% the "
            f"small-deflection beam theory used to generate the dataset is no "
            f"longer valid, and a real arm would bend, yield or fail. "
            f"Treat this result as a warning, not a design value."
        )

    stress_ok = True
    if strength is not None:
        stress_ok = stress <= MAX_STRESS_SAFETY * strength
        if not stress_ok:
            warnings.append(
                f"Estimated bending stress ({stress:.0f} MPa) exceeds "
                f"{MAX_STRESS_SAFETY*100:.0f}% of the material strength for "
                f"{material_name} ({strength:.0f} MPa). The arm would yield or "
                f"break before reaching the predicted deflection."
            )

    return {
        "deflection_ratio": ratio,
        "stress_MPa": stress,
        "strength_MPa": strength,
        "warnings": warnings,
        "small_deflection_ok": small_deflection_ok,
        "stress_ok": stress_ok,
    }


def is_physically_valid(
    length_mm: float,
    load_kg: float,
    diameter_mm: float,
    load_case: int,
    material: str,
) -> bool:
    """True if the sample satisfies the small-deflection and stress criteria.

    Used by the dataset generator to discard unrealistic combinations.
    """
    delta = tip_deflection_mm(length_mm, load_kg, MATERIALS[material]["E_GPa"],
                             diameter_mm, load_case)
    if delta / length_mm > MAX_DEFLECTION_RATIO:
        return False
    stress = max_bending_stress_MPa(length_mm, load_kg, diameter_mm, load_case)
    if stress > MAX_STRESS_SAFETY * MATERIALS[material]["strength_MPa"]:
        return False
    return True


if __name__ == "__main__":
    # A tiny self-test you can run to confirm the physics module behaves.
    print("physics.py self-test")
    print("-" * 55)
    # Worked example from standard beam tables:
    # L = 500 mm, P = 2 kg, Aluminium (E = 68.9 GPa), d = 10 mm
    d = 10.0
    I = moment_of_inertia_circular(d)
    print(f"I for d={d} mm          : {I:.4e} m^4")
    delta = tip_deflection_mm(500.0, 2.0, 68.9, d)
    print(f"Tip load 2 kg, 500 mm   : {delta:.3f} mm")
    delta_u = tip_deflection_mm(500.0, 2.0, 68.9, d, LOAD_CASE_DISTRIBUTED)
    print(f"Distributed 2 kg total  : {delta_u:.3f} mm")
    print(f"ratio distributed/tip   : {delta_u/delta:.4f}  (theory: 0.375)")
    print(f"root bending stress     : {max_bending_stress_MPa(500.0, 2.0, d):.2f} MPa")
