import importlib
import time
from typing import Any, Protocol


class Predictor(Protocol):
    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]: ...


class SimulatedPredictor:
    """Stand-in model with a fixed per-call cost and a small per-item cost, like a GPU forward pass."""

    def __init__(self, fixed_ms: float = 20.0, per_item_ms: float = 1.0):
        self.fixed = fixed_ms / 1000
        self.per_item = per_item_ms / 1000

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        time.sleep(self.fixed + self.per_item * len(texts))
        out = []
        for t in texts:
            if "__fail__" in t:
                raise ValueError("simulated model failure")
            out.append({"label": "long" if len(t) > 40 else "short", "length": len(t)})
        return out


class SklearnPredictor:
    """Loads a joblib-saved scikit-learn pipeline that supports predict_proba."""

    def __init__(self, path: str):
        import joblib

        self.pipe = joblib.load(path)

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        proba = self.pipe.predict_proba(texts)
        classes = list(self.pipe.classes_)
        out = []
        for row in proba:
            i = int(row.argmax())
            out.append({"label": str(classes[i]), "score": float(row[i])})
        return out


def load_predictor(spec: str) -> Predictor:
    # "simulated", "sklearn:<path>" or "<module>:<factory>"
    if spec == "simulated":
        return SimulatedPredictor()
    kind, _, arg = spec.partition(":")
    if kind == "sklearn":
        return SklearnPredictor(arg)
    module = importlib.import_module(kind)
    return getattr(module, arg)()
