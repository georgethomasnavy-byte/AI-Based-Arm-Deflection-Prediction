"""
model_classes.py
================
Classes that are stored INSIDE the saved model file must live in a real,
importable module -- not inside train_model.py.

Why this matters (a real bug we hit and fixed)
----------------------------------------------
joblib/pickle does not save a class definition; it saves the *path* to the
class, e.g. "model_classes.PowerLawRegressor".  If the class had been defined
in train_model.py and we ran that file directly, Python records it as
"__main__.PowerLawRegressor".  That path is meaningless to app.py, so loading
the model would fail with:

    AttributeError: Can't get attribute 'PowerLawRegressor' on <module '__main__'>

Keeping the class here means BOTH train_model.py and app.py import it from the
same place, and loading always works.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Logarithms of the four continuous physical features.
LOG_COLUMNS = ["length_mm", "payload_kg", "diameter_mm", "E_GPa"]


class PowerLawRegressor:
    """Linear regression fitted in log-log space, i.e. a learned power law.

    Physics background
    ------------------
        delta = C * P * L^3 / (E * I),  and  I = pi*d^4/64
    so
        ln(delta) = ln(C') + 1*ln(P) + 3*ln(L) - 1*ln(E) - 4*ln(d)

    Fitting a straight line to ln(delta) therefore *estimates the exponents*.
    We do not hard-code 3, 1, -1 or -4 anywhere: the model is free to learn any
    coefficients, and the fact that it lands on the physically correct ones is
    a RESULT we report, not an assumption we imposed.

    `load_case` enters as an additive offset and captures the change from a
    1/3 factor (tip load) to a 1/8 factor (distributed load), i.e. ln(1/3) vs
    ln(1/8) relative to a common baseline.
    """

    #: Physically expected exponents, used only for comparison/printing.
    THEORY = {
        "log_length_mm": 3.0,
        "log_payload_kg": 1.0,
        "log_diameter_mm": -4.0,
        "log_E_GPa": -1.0,
    }

    def __init__(self):
        self.model = LinearRegression()
        self.coef_: dict[str, float] = {}

    @staticmethod
    def transform(X: pd.DataFrame) -> pd.DataFrame:
        """Turn the raw features into the log features the linear model needs."""
        Z = pd.DataFrame(index=X.index)
        for col in LOG_COLUMNS:
            Z["log_" + col] = np.log(X[col].to_numpy(dtype=float))
        # load_case is already 0/1, so it stays linear (it is a shift, not a scale)
        Z["load_case"] = X["load_case"].to_numpy(dtype=float)
        return Z

    def fit(self, X: pd.DataFrame, y) -> "PowerLawRegressor":
        Z = self.transform(X)
        y = np.asarray(y, dtype=float)
        self.model.fit(Z, np.log(y))
        self.coef_ = dict(zip(Z.columns.tolist(), self.model.coef_.tolist()))
        self.coef_["intercept"] = float(self.model.intercept_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.transform(X)
        return np.exp(self.model.predict(Z))


class LogTargetRegressor:
    """Wraps a regressor so that it trains on log1p(y) but predicts in real mm.

    Why wrap instead of calling np.expm1 everywhere?
    ------------------------------------------------
    During development we hit a real bug: one plotting function used the raw
    Random Forest, whose output is in log space, and drew a deflection of
    "3 mm" where the true value was 19 mm.  Wrapping makes it IMPOSSIBLE to
    forget the inverse transform -- every model in this project now has a
    .predict() method that returns millimetres.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, np.log1p(np.asarray(y, dtype=float)))
        return self

    def predict(self, X) -> np.ndarray:
        return np.expm1(self.estimator.predict(X))
