"""
test_physics.py
===============
Automated tests for the physics module.  Run with:

    python test_physics.py

These tests check our code against known analytical results and against the
scaling laws of beam theory.  They are also excellent viva material: if a teacher
asks "how do you know your equations are right?", you can run this file and show
that the code reproduces textbook values and the expected L³ and 1/d⁴ behaviour.

No pytest needed — plain Python, so it runs anywhere.
"""

from __future__ import annotations

import math

import numpy as np

from physics import (
    G,
    LOAD_CASE_DISTRIBUTED,
    LOAD_CASE_TIP,
    MATERIALS,
    max_bending_stress_MPa,
    moment_of_inertia_circular,
    tip_deflection_mm,
)

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}   {detail}")


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


print("=" * 70)
print("PHYSICS TEST SUITE")
print("=" * 70)

# ---------------------------------------------------------------- 1. geometry
print("\n1. Second moment of area")
d = 10.0                                        # mm
expected_I = math.pi * (0.01 ** 4) / 64.0       # m^4, worked by hand
check("I for d=10 mm matches pi*d^4/64",
      close(moment_of_inertia_circular(d), expected_I),
      f"got {moment_of_inertia_circular(d):.6e}, expected {expected_I:.6e}")
check("I scales with d^4",
      close(moment_of_inertia_circular(20.0) / moment_of_inertia_circular(10.0),
            16.0, 1e-9))

# ------------------------------------------------------------ 2. known values
print("\n2. Deflection against a hand-calculation")
# L = 500 mm, P = 2 kg, E = 68.9 GPa (aluminium), d = 10 mm
L, P, E, dia = 500.0, 2.0, 68.9, 10.0
I = moment_of_inertia_circular(dia)
hand = (P * G) * (L / 1000) ** 3 / (3 * E * 1e9 * I) * 1000
code = tip_deflection_mm(L, P, E, dia, LOAD_CASE_TIP)
check("tip load deflection matches hand calculation", close(code, hand, 1e-12),
      f"code {code:.6f} vs hand {hand:.6f}")
check("value is plausible for a 0.5 m aluminium rod (~24 mm)", 20 < code < 30,
      f"got {code:.2f} mm")

# --------------------------------------------------------------- 3. scaling
print("\n3. Scaling laws (the physics the report depends on)")
base = tip_deflection_mm(300.0, 1.0, 68.9, 12.0)
check("doubling length multiplies deflection by 8 (L^3)",
      close(tip_deflection_mm(600.0, 1.0, 68.9, 12.0) / base, 8.0, 1e-9))
check("doubling load doubles deflection (linear in P)",
      close(tip_deflection_mm(300.0, 2.0, 68.9, 12.0) / base, 2.0, 1e-9))
check("doubling diameter divides deflection by 16 (1/d^4)",
      close(tip_deflection_mm(300.0, 1.0, 68.9, 24.0) / base, 1 / 16, 1e-9))
check("doubling Young's modulus halves deflection (1/E)",
      close(tip_deflection_mm(300.0, 1.0, 137.8, 12.0) / base, 0.5, 1e-9))

# --------------------------------------------------------- 4. load-case ratio
print("\n4. Distributed load vs tip load")
tip = tip_deflection_mm(400.0, 2.0, 100.0, 15.0, LOAD_CASE_TIP)
dist = tip_deflection_mm(400.0, 2.0, 100.0, 15.0, LOAD_CASE_DISTRIBUTED)
check("distributed deflection is 3/8 of tip deflection for the same total mass",
      close(dist / tip, 3 / 8, 1e-9), f"ratio = {dist/tip:.6f}")

# ------------------------------------------------------------- 5. monotonicity
print("\n5. Monotonicity over a random sweep (physically obvious, easy to break)")
rng = np.random.default_rng(0)
violations = 0
for _ in range(300):
    L1 = rng.uniform(150, 800)
    P1 = rng.uniform(0.05, 5)
    E1 = rng.uniform(2.8, 200)
    d1 = rng.uniform(6, 40)
    case = int(rng.integers(2))
    ref = tip_deflection_mm(L1, P1, E1, d1, case)
    if tip_deflection_mm(L1 * 1.01, P1, E1, d1, case) <= ref:      # longer -> more
        violations += 1
    if tip_deflection_mm(L1, P1 * 1.01, E1, d1, case) <= ref:      # heavier -> more
        violations += 1
    if tip_deflection_mm(L1, P1, E1 * 1.01, d1, case) >= ref:      # stiffer -> less
        violations += 1
    if tip_deflection_mm(L1, P1, E1, d1 * 1.01, case) >= ref:      # thicker -> less
        violations += 1
check("all 1200 sensitivity checks behave correctly", violations == 0,
      f"{violations} violations")

# ------------------------------------------------------------------ 6. stress
print("\n6. Bending stress")
# sigma = M*c/I with M = P*g*L, c = d/2
L, P, dia = 500.0, 2.0, 10.0
I = moment_of_inertia_circular(dia)
hand_stress = (P * G * (L / 1000)) * (dia / 2000) / I / 1e6
check("root bending stress matches hand calculation",
      close(max_bending_stress_MPa(L, P, dia), hand_stress, 1e-12),
      f"got {max_bending_stress_MPa(L,P,dia):.4f}, hand {hand_stress:.4f}")
check("stress is in a realistic range for steel/aluminium (<500 MPa)",
      max_bending_stress_MPa(L, P, dia) < 500)

# ------------------------------------------------- 7. the safety net actually fires
print("\n7. Physical plausibility safety net")
# The raw formula is valid mathematics but only MEANINGFUL while deflection is
# small compared with length.  For extreme combinations (e.g. a long thin Nylon
# arm with a heavy tip load) it returns values like 47000 mm on an 800 mm arm.
# The formula is not wrong; the ASSUMPTION behind it has failed.
#
# So the invariant to test is not "the formula never returns big numbers" -- it
# is "whenever the result is unphysical, the validation layer SAYS SO".
from physics import assess_plausibility, MAX_DEFLECTION_RATIO

checked = 0
mishandled = []
for mat, props in MATERIALS.items():
    for case in (LOAD_CASE_TIP, LOAD_CASE_DISTRIBUTED):
        for L in (150, 400, 800):
            for d in (6, 20, 40):
                for P in (0.05, 1.0, 5.0):
                    checked += 1
                    v = tip_deflection_mm(L, P, props["E_GPa"], d, case)
                    a = assess_plausibility(L, P, d, case, mat, v)
                    broke_small_deflection = (v / L) > MAX_DEFLECTION_RATIO
                    broke_strength = a["stress_MPa"] > 0.60 * props["strength_MPa"]
                    should_warn = broke_small_deflection or broke_strength
                    if should_warn and not a["warnings"]:
                        mishandled.append((mat, L, d, P, case, round(v, 1)))
                    if not should_warn and a["warnings"]:
                        mishandled.append(("spurious", mat, L, d, P, case))

check(f"all {checked} design-space combinations are classified correctly "
      f"(warned exactly when unphysical)",
      not mishandled, f"{len(mishandled)} mishandled, e.g. {mishandled[:3]}")

# A realistic engineering case must come back CLEAN (no warnings).
a = assess_plausibility(400.0, 1.0, 12.0, LOAD_CASE_TIP, "Aluminium 6061-T6",
                        tip_deflection_mm(400.0, 1.0, 68.9, 12.0))
check("a sensible case (400 mm Al, 1 kg, d=12) raises no warning",
      a["warnings"] == [], f"got {a['warnings']}")

# An absurd case MUST warn.
v_bad = tip_deflection_mm(800.0, 5.0, MATERIALS["Nylon 6/6"]["E_GPa"], 6.0)
a_bad = assess_plausibility(800.0, 5.0, 6.0, LOAD_CASE_TIP, "Nylon 6/6", v_bad)
check("an absurd case (800 mm Nylon, 5 kg, d=6) is flagged",
      len(a_bad["warnings"]) >= 1, f"deflection was {v_bad:.0f} mm")

# -------------------------------------------------------------------- summary
print("\n" + "=" * 70)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 70)
if failed:
    raise SystemExit(1)
print("All physics tests passed.")
