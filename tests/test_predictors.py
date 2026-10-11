import pytest

from inferq.predictors import SimulatedPredictor, load_predictor


def test_simulated_labels():
    out = SimulatedPredictor(0, 0).predict_batch(["hi", "x" * 50])
    assert [o["label"] for o in out] == ["short", "long"]


def test_simulated_failure():
    with pytest.raises(ValueError):
        SimulatedPredictor(0, 0).predict_batch(["__fail__"])


def test_sklearn_predictor(tmp_path):
    joblib = pytest.importorskip("joblib")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(TfidfVectorizer(), LogisticRegression())
    pipe.fit(["great film", "awful film", "great fun", "awful bore"], ["pos", "neg", "pos", "neg"])
    path = tmp_path / "m.joblib"
    joblib.dump(pipe, path)
    pred = load_predictor(f"sklearn:{path}")
    out = pred.predict_batch(["great", "awful"])
    assert [o["label"] for o in out] == ["pos", "neg"]
    assert 0.5 < out[0]["score"] <= 1.0


def test_load_by_dotted_path():
    pred = load_predictor("inferq.predictors:SimulatedPredictor")
    assert hasattr(pred, "predict_batch")
